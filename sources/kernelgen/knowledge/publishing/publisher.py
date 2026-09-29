"""Single-writer, deterministic publication of one complete candidate batch."""

from __future__ import annotations

import fcntl
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from kernelgen.knowledge.catalog import (
    FilesystemCatalog,
    assert_catalog_paths_clean,
    atomic_text,
    catalog_changed_paths,
)
from kernelgen.knowledge.config import KnowledgeReviewerMode
from kernelgen.knowledge.index import SQLiteKnowledgeIndex
from kernelgen.knowledge.models import (
    CandidateConcept,
    CandidateReviewRecord,
    Concept,
    ObservationRecord,
    RuntimeCandidate,
)
from kernelgen.knowledge.contracts.runtime import PublishResult
from kernelgen.knowledge.layout import CatalogLayout, safe_name
from kernelgen.knowledge.publishing.materialize import (
    CandidateMaterializer,
    _same_observation,
)
from kernelgen.knowledge.publishing.merge import merge_concepts
from kernelgen.knowledge.publishing.reviewer import (
    ReviewCallable,
    defer_candidate_batch,
    review_candidate_batch,
)
from kernelgen.knowledge.validation import iter_jsonl
from kernelgen.knowledge.vocabulary import (
    Vocabulary,
    concept_vocabulary_warnings,
)


def _resolve_reviewer_mode(
    reviewer_mode: KnowledgeReviewerMode | str | None,
    reviewer: ReviewCallable | None,
) -> KnowledgeReviewerMode:
    if reviewer_mode is None:
        return (
            KnowledgeReviewerMode.ENFORCE
            if reviewer is not None
            else KnowledgeReviewerMode.OFF
        )
    return KnowledgeReviewerMode(reviewer_mode)


class BatchPublisher:
    """Validate in staging, then publish Observations before Concepts."""

    def __init__(
        self,
        catalog_root: Path,
        fact_reader,
    ):
        self.catalog_root = Path(catalog_root)
        self.layout = CatalogLayout(self.catalog_root)
        self.fact_reader = fact_reader
        self.vocabulary = Vocabulary.from_catalog(self.catalog_root)
        self.materializer = CandidateMaterializer(
            self.catalog_root,
            self.fact_reader,
            self.vocabulary,
        )

    def publish(
        self,
        workspaces: Sequence[Path],
        *,
        run_id: str,
        batch_id: str,
        reviewer: ReviewCallable | None = None,
        reviewer_mode: KnowledgeReviewerMode | str | None = None,
    ) -> PublishResult:
        reviewer_mode = _resolve_reviewer_mode(reviewer_mode, reviewer)
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        self.layout.publish_lock.parent.mkdir(parents=True, exist_ok=True)
        with self.layout.publish_lock.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            replay = self._published_batch(batch_id)
            if replay is not None:
                return replay
            return self._publish_locked(
                workspaces,
                run_id=run_id,
                batch_id=batch_id,
                reviewer=reviewer,
                reviewer_mode=reviewer_mode,
            )

    def publish_static(
        self,
        candidates: Sequence[CandidateConcept],
        *,
        batch_id: str,
    ) -> PublishResult:
        """Publish reviewed source-only Candidates through the same transaction."""

        self.catalog_root.mkdir(parents=True, exist_ok=True)
        self.layout.publish_lock.parent.mkdir(parents=True, exist_ok=True)
        with self.layout.publish_lock.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            replay = self._published_batch(batch_id)
            if replay is not None:
                return replay
            catalog = FilesystemCatalog(self.catalog_root)
            old_snapshot = catalog.snapshot()
            collected = self.materializer.validate_static(candidates)
            if not collected:
                return PublishResult(status="noop")
            return self._publish_fresh_locked(
                collected,
                old_snapshot=old_snapshot,
                run_id="static",
                batch_id=batch_id,
            )

    def _publish_locked(
        self,
        workspaces: Sequence[Path],
        *,
        run_id: str,
        batch_id: str,
        reviewer: ReviewCallable | None,
        reviewer_mode: KnowledgeReviewerMode,
    ) -> PublishResult:
        catalog = FilesystemCatalog(self.catalog_root)
        old_snapshot = catalog.snapshot()
        collected = self.materializer.collect(workspaces, run_id)
        usage_observations = self.materializer.collect_usage_observations(
            workspaces,
            run_id=run_id,
        )
        if not collected and not usage_observations:
            return PublishResult(
                status="noop",
                reviewer_mode=reviewer_mode.value,
            )

        return self._publish_fresh_locked(
            collected,
            old_snapshot=old_snapshot,
            run_id=run_id,
            batch_id=batch_id,
            usage_observations=usage_observations,
            reviewer=reviewer,
            reviewer_mode=reviewer_mode,
            require_review=True,
        )

    def _publish_fresh_locked(
        self,
        fresh: Sequence[
            tuple[Path | None, CandidateConcept | RuntimeCandidate]
        ],
        *,
        old_snapshot: str,
        run_id: str,
        batch_id: str,
        usage_observations: dict[str, ObservationRecord] | None = None,
        reviewer: ReviewCallable | None = None,
        reviewer_mode: KnowledgeReviewerMode = KnowledgeReviewerMode.OFF,
        require_review: bool = False,
    ) -> PublishResult:
        preexisting_dirty_paths = catalog_changed_paths(self.catalog_root)
        catalog = FilesystemCatalog(self.catalog_root)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".kb-publish-{safe_name(batch_id)}-",
                dir=str(self.catalog_root.parent),
            )
        )
        try:
            shutil.copytree(
                self.catalog_root,
                staging,
                dirs_exist_ok=True,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".derived",
                    ".publish.lock",
                ),
            )
            staged_catalog = FilesystemCatalog(staging)
            published_observations = self._existing_observations()
            (
                observations,
                valid,
                rejected,
                audit_warnings,
            ) = self.materializer.materialize_observations(
                fresh,
                run_id=run_id,
                published_observations=published_observations,
            )
            review_records: list[CandidateReviewRecord] = []
            review_warnings: list[str] = []
            approved_candidate_ids: list[str] = []
            shadow_approved_candidate_ids: list[str] = []
            deferred_candidate_ids: list[str] = []
            review_rejected_candidate_ids: list[str] = []
            if require_review and valid:
                if reviewer_mode == KnowledgeReviewerMode.OFF:
                    review_result = defer_candidate_batch(
                        staged_catalog,
                        valid,
                        observations,
                        run_id=run_id,
                        batch_id=batch_id,
                    )
                else:
                    review_result = review_candidate_batch(
                        staged_catalog,
                        valid,
                        observations,
                        run_id=run_id,
                        batch_id=batch_id,
                        reviewer=reviewer,
                        reviewer_mode=reviewer_mode.value,
                    )
                review_records = review_result.records
                review_warnings = review_result.warnings
                decision_approved_candidate_ids = sorted(
                    candidate_id
                    for record in review_records
                    if record.decision == "approve"
                    for candidate_id in record.candidate_ids
                )
                decision_deferred_candidate_ids = sorted(
                    candidate_id
                    for record in review_records
                    if record.decision == "defer"
                    for candidate_id in record.candidate_ids
                )
                review_rejected_candidate_ids = sorted(
                    candidate_id
                    for record in review_records
                    if record.decision == "reject"
                    for candidate_id in record.candidate_ids
                )
                if reviewer_mode == KnowledgeReviewerMode.ENFORCE:
                    valid = review_result.candidates
                    approved_candidate_ids = decision_approved_candidate_ids
                    deferred_candidate_ids = decision_deferred_candidate_ids
                elif reviewer_mode == KnowledgeReviewerMode.SHADOW:
                    valid = []
                    shadow_approved_candidate_ids = (
                        decision_approved_candidate_ids
                    )
                    deferred_candidate_ids = sorted(
                        {
                            *decision_approved_candidate_ids,
                            *decision_deferred_candidate_ids,
                        }
                    )
                else:
                    valid = []
                    deferred_candidate_ids = decision_deferred_candidate_ids
            eligible_candidate_ids = {
                candidate.candidate_id for candidate, _ in valid
            }
            for key, item in (usage_observations or {}).items():
                prior = (
                    observations.get(key)
                    or published_observations.get(key)
                )
                if prior is not None and not _same_observation(prior, item):
                    raise ValueError(f"observation id collision: {key}")
                observations[key] = prior or item
            (
                changed,
                created,
                updated,
                contested,
                merge_rejected,
            ) = merge_concepts(
                staged_catalog,
                valid,
                observations,
            )
            rejected = sorted({*rejected, *merge_rejected})
            audit_warnings.extend(
                warning
                for warning in concept_vocabulary_warnings(
                    staged_catalog.iter_concepts(),
                    self.vocabulary,
                )
                if warning.startswith("catalog has ")
            )
            audit_warnings = sorted(set(audit_warnings))
            referenced_observations = {
                item.observation_ref
                for concept in staged_catalog.iter_concepts()
                for item in concept.evidence
            }
            observations = {
                key: value
                for key, value in observations.items()
                if key in referenced_observations or value.applications
            }
            transaction_paths = [
                catalog.concept_path(concept)
                for concept in changed
            ]
            if observations:
                transaction_paths.append(
                    self.layout.observations / f"{safe_name(run_id)}.jsonl"
                )
            if review_records:
                transaction_paths.append(self._review_audit_path(batch_id))
            transaction_paths.append(self._audit_path(batch_id))
            assert_catalog_paths_clean(
                self.catalog_root,
                transaction_paths,
                changed_paths=preexisting_dirty_paths,
            )
            self._write_observations(
                staging,
                run_id,
                observations.values(),
            )

            # Observations land first. A crash may leave an unreferenced fact,
            # never a Concept pointing at a missing fact.
            self._write_observations(
                self.catalog_root,
                run_id,
                observations.values(),
            )
            backups = self._backup_concepts(changed)
            try:
                for concept in changed:
                    catalog.write_concept(concept)
                new_snapshot = catalog.snapshot()
                SQLiteKnowledgeIndex(self.layout.index).rebuild(
                    catalog.iter_concepts(),
                    new_snapshot,
                    observations=catalog.iter_observations(),
                    retrievals=catalog.iter_retrieval_events(),
                    usage_snapshot=catalog.usage_snapshot(),
                )
                observations_added = sorted(
                    set(observations) - set(published_observations)
                )
                review_audit_ref = ""
                if review_records:
                    review_path = self._write_review_audit(
                        batch_id,
                        review_records,
                    )
                    review_audit_ref = str(
                        review_path.relative_to(self.catalog_root)
                    )
                result = PublishResult(
                    status="published" if changed or observations_added else "noop",
                    reviewer_mode=reviewer_mode.value,
                    created=created,
                    updated=updated,
                    observations_added=observations_added,
                    contested=contested,
                    processed_candidates=sorted(
                        candidate_id
                        for candidate_id in eligible_candidate_ids
                        if not any(
                            reason.startswith(f"{candidate_id}:")
                            for reason in rejected
                        )
                    ),
                    approved_candidates=approved_candidate_ids,
                    shadow_approved_candidates=shadow_approved_candidate_ids,
                    deferred_candidates=deferred_candidate_ids,
                    review_rejected_candidates=(
                        review_rejected_candidate_ids
                    ),
                    rejected=rejected,
                    audit_warnings=audit_warnings,
                    review_warnings=review_warnings,
                    audit_ref=str(
                        self._audit_path(batch_id).relative_to(self.catalog_root)
                    ),
                    review_audit_ref=review_audit_ref,
                )
                self._write_audit(
                    batch_id,
                    result,
                    snapshot_before=old_snapshot,
                    snapshot_after=new_snapshot,
                    observations=observations.values(),
                    review_records=review_records,
                )
                return result
            except BaseException:
                self._restore_concepts(backups)
                restored_snapshot = catalog.snapshot()
                SQLiteKnowledgeIndex(self.layout.index).rebuild(
                    catalog.iter_concepts(),
                    restored_snapshot,
                    observations=catalog.iter_observations(),
                    retrievals=catalog.iter_retrieval_events(),
                    usage_snapshot=catalog.usage_snapshot(),
                )
                raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _write_observations(
        self,
        root: Path,
        run_id: str,
        records: Iterable[ObservationRecord],
    ) -> None:
        path = (
            CatalogLayout(Path(root)).observations
            / f"{safe_name(run_id)}.jsonl"
        )
        existing = {
            str(raw["id"]): raw
            for _, raw in iter_jsonl(path)
        } if path.is_file() else {}
        for record in records:
            raw = record.model_dump(mode="json")
            prior = existing.get(record.id)
            if prior is not None and prior != raw:
                raise ValueError(
                    f"published observation changed: {record.id}"
                )
            existing[record.id] = raw
        content = "".join(
            json.dumps(existing[key], sort_keys=True, ensure_ascii=False) + "\n"
            for key in sorted(existing)
        )
        if content:
            atomic_text(path, content)

    def _existing_observations(self) -> dict[str, ObservationRecord]:
        existing = {}
        observation_root = self.catalog_root / "observations"
        if not observation_root.exists():
            return existing
        for path in sorted(observation_root.glob("**/*.jsonl")):
            for _, raw in iter_jsonl(path):
                record = ObservationRecord.model_validate(raw)
                existing[record.id] = record
        return existing

    def _backup_concepts(
        self,
        concepts: Sequence[Concept],
    ) -> dict[Path, str | None]:
        catalog = FilesystemCatalog(self.catalog_root)
        backups = {}
        for concept in concepts:
            path = catalog.concept_path(concept)
            backups[path] = (
                path.read_text(encoding="utf-8") if path.is_file() else None
            )
        return backups

    def _restore_concepts(self, backups: dict[Path, str | None]) -> None:
        for path, content in backups.items():
            if content is None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            else:
                atomic_text(path, content)

    def _audit_path(self, batch_id: str) -> Path:
        return self.layout.publish_audit / f"{safe_name(batch_id)}.json"

    def _review_audit_path(self, batch_id: str) -> Path:
        return self.layout.publish_reviews / f"{safe_name(batch_id)}.jsonl"

    def _write_review_audit(
        self,
        batch_id: str,
        records: Sequence[CandidateReviewRecord],
    ) -> Path:
        path = self._review_audit_path(batch_id)
        content = "".join(
            json.dumps(
                record.model_dump(mode="json"),
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
            for record in sorted(records, key=lambda item: item.review_id)
        )
        atomic_text(path, content)
        return path

    def _published_batch(self, batch_id: str) -> PublishResult | None:
        path = self._audit_path(batch_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("batch_id") != batch_id:
            raise ValueError(f"publish audit batch mismatch: {path}")
        prior = PublishResult.model_validate(payload.get("result") or {})
        return PublishResult(
            status="noop",
            reviewer_mode=prior.reviewer_mode,
            approved_candidates=prior.approved_candidates,
            shadow_approved_candidates=prior.shadow_approved_candidates,
            deferred_candidates=prior.deferred_candidates,
            review_rejected_candidates=prior.review_rejected_candidates,
            audit_warnings=prior.audit_warnings,
            review_warnings=prior.review_warnings,
            audit_ref=str(path.relative_to(self.catalog_root)),
            review_audit_ref=prior.review_audit_ref,
        )

    def _write_audit(
        self,
        batch_id: str,
        result: PublishResult,
        *,
        snapshot_before: str,
        snapshot_after: str,
        observations: Iterable[ObservationRecord],
        review_records: Sequence[CandidateReviewRecord],
    ) -> None:
        payload = {
            "schema_version": "1.0",
            "batch_id": batch_id,
            "reviewer_mode": result.reviewer_mode,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_before": snapshot_before,
            "snapshot_after": snapshot_after,
            "result": result.model_dump(mode="json"),
            "application_links": _application_links(observations),
            "review_decisions": [
                item.model_dump(mode="json")
                for item in sorted(
                    review_records,
                    key=lambda record: record.review_id,
                )
            ],
        }
        atomic_text(
            self._audit_path(batch_id),
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
        )


def _application_links(
    observations: Iterable[ObservationRecord],
) -> list[dict]:
    links = []
    for observation in observations:
        for application in observation.applications:
            reference = application.concept_ref
            if application.source_ref is not None:
                source = application.source_ref
                reference = (
                    f"{source.resource}@{source.revision}::{source.locator}"
                )
            link = {
                "observation_ref": observation.id,
                "knowledge_ref": reference,
                "disposition": application.disposition,
                "lineage_status": application.lineage_status,
            }
            if application.agent_assessment is not None:
                link["agent_assessment"] = (
                    application.agent_assessment.model_dump(mode="json")
                )
            links.append(link)
    return sorted(
        links,
        key=lambda item: (
            item["observation_ref"],
            item["knowledge_ref"],
            item["disposition"],
        ),
    )

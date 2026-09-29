"""Deterministic Concept identity, evidence, and lifecycle merging."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Sequence

from kernelgen.knowledge.catalog import FilesystemCatalog, concept_ref
from kernelgen.knowledge.layout import safe_name
from kernelgen.knowledge.models import (
    CandidateConcept,
    Concept,
    ConceptEvidence,
    ManagedMetadata,
    ObservationRecord,
    Scope,
)
from kernelgen.knowledge.validation import canonical_concept_hash


def canonical_candidate_scope(scope: Scope) -> Scope:
    """Return the exact Scope representation used for merge identity."""

    return _without_permissive_exact(scope)


def candidate_group_key(candidate: CandidateConcept) -> tuple:
    """Return the deterministic pre-merge identity used by Reviewer and merge."""

    if candidate.publish_action == "update":
        return ("target", candidate.target_concept_id)
    scope_key = json.dumps(
        canonical_candidate_scope(candidate.scope).model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "identity",
        candidate.proposed_kind,
        candidate.claim_key,
        scope_key,
    )


def merge_concepts(
    catalog: FilesystemCatalog,
    candidates: Sequence[
        tuple[CandidateConcept, list[ConceptEvidence]]
    ],
    observations: dict[str, ObservationRecord],
) -> tuple[
    list[Concept],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    grouped = defaultdict(list)
    for candidate, refs in candidates:
        candidate = candidate.model_copy(
            update={"scope": canonical_candidate_scope(candidate.scope)}
        )
        group_key = candidate_group_key(candidate)
        grouped[group_key].append((candidate, refs))

    changed = []
    created = []
    updated = []
    contested = []
    rejected = []
    now = datetime.now(timezone.utc)
    for group_key, group in sorted(
        grouped.items(),
        key=lambda item: str(item[0]),
    ):
        group.sort(key=lambda item: item[0].candidate_id)
        candidate = group[0][0]
        kind = candidate.proposed_kind
        claim_key = candidate.claim_key
        runtime_candidates = [
            item
            for item, _ in group
            if not item.created_by.startswith("source-ingest:")
        ]
        static_candidates = [
            item
            for item, _ in group
            if item.created_by.startswith("source-ingest:")
        ]
        invalid_catalog_refs = False
        for item, _ in group:
            try:
                _validate_catalog_refs(item, catalog)
            except ValueError as exc:
                rejected.append(f"{item.candidate_id}: {exc}")
                invalid_catalog_refs = True
        if invalid_catalog_refs:
            continue
        target_id = (
            candidate.target_concept_id
            if candidate.publish_action == "update"
            else None
        )
        if target_id is not None:
            try:
                existing = catalog.get_concept(target_id)
            except KeyError:
                rejected.append(
                    f"{candidate.candidate_id}: target concept not found: "
                    f"{target_id}"
                )
                continue
            if existing.kind != kind:
                rejected.append(
                    f"{candidate.candidate_id}: target kind mismatch: "
                    f"{target_id}"
                )
                continue
            if existing.status == "deprecated":
                rejected.append(
                    f"{candidate.candidate_id}: target concept is deprecated: "
                    f"{target_id}"
                )
                continue
        else:
            existing = next(
                (
                    item
                    for item in catalog.iter_concepts()
                    if item.kind == kind
                    and item.claim_key == claim_key
                    and (
                        _without_permissive_exact(item.scope) == candidate.scope
                        or (
                            bool(static_candidates)
                            and
                            candidate.proposed_id is not None
                            and item.id == candidate.proposed_id
                        )
                    )
                ),
                None,
            )
        try:
            evidence = _merge_concept_evidence(
                [
                    link
                    for _, candidate_evidence in group
                    for link in candidate_evidence
                ]
            )
        except ValueError as exc:
            rejected.extend(
                f"{item.candidate_id}: {exc}" for item, _ in group
            )
            continue
        sources = _unique_models(
            [
                source
                for item, _ in group
                for source in item.source_refs
            ]
        )
        static_source_resources = {
            source.resource
            for item in static_candidates
            for source in item.source_refs
        }
        state = _evidence_state(evidence, sources)
        if existing is None:
            concept_id = _available_concept_id(
                candidate.proposed_id,
                kind=kind,
                claim_key=claim_key,
                scope=candidate.scope.model_dump(mode="json"),
                catalog=catalog,
            )
            concept = Concept(
                id=concept_id,
                kind=kind,
                title=candidate.title,
                summary=candidate.summary,
                claim_key=claim_key,
                domains=sorted(
                    {
                        domain
                        for item, _ in group
                        for domain in item.domains
                    }
                ),
                verified=[],
                sources=sources,
                scope=candidate.scope,
                retrieval=candidate.retrieval,
                evidence_state=state,
                evidence=evidence,
                relations=candidate.relations,
                managed=ManagedMetadata(
                    content_hash="sha256:" + "0" * 64,
                    created_by="kernelgen/publisher-v1",
                    created_at=now,
                    updated_at=now,
                ),
                body=candidate.body,
            )
            created.append(concept_ref(concept))
        else:
            try:
                combined_evidence = _merge_concept_evidence(
                    [*existing.evidence, *evidence]
                )
            except ValueError as exc:
                rejected.extend(
                    f"{item.candidate_id}: {exc}" for item, _ in group
                )
                continue
            # A reviewed static import is authoritative for the revision and
            # locator of each SourcePackage it cites. Preserve citations from
            # other packages, but do not accumulate stale revisions forever.
            retained_sources = [
                source
                for source in existing.sources
                if source.resource not in static_source_resources
            ]
            combined_sources = _unique_models([*retained_sources, *sources])
            combined_state = _evidence_state(
                combined_evidence,
                combined_sources,
            )
            static_update = bool(static_candidates) or candidate.publish_action == "update"
            static_changed = static_update and (
                existing.title != candidate.title
                or existing.summary != candidate.summary
                or existing.domains != sorted(set(candidate.domains))
                or existing.retrieval != candidate.retrieval
                or existing.relations != candidate.relations
                or existing.scope != candidate.scope
                or existing.body != candidate.body
            )
            if (
                combined_evidence == existing.evidence
                and combined_sources == existing.sources
                and combined_state == existing.evidence_state
                and not static_changed
            ):
                continue
            updates = {
                "sources": combined_sources,
                "evidence_state": combined_state,
                "evidence": combined_evidence,
                "managed": existing.managed.model_copy(
                    update={
                        "content_hash": "sha256:" + "0" * 64,
                        "updated_at": now,
                    }
                ),
            }
            if static_update:
                updates.update(
                    {
                        "title": candidate.title,
                        "summary": candidate.summary,
                        "claim_key": existing.claim_key,
                        "domains": sorted(set(candidate.domains)),
                        "retrieval": candidate.retrieval,
                        "relations": candidate.relations,
                        "scope": candidate.scope,
                        "body": candidate.body,
                    }
                )
            concept = existing.model_copy(update=updates)
            updated.append(concept_ref(concept))
        concept = concept.model_copy(
            update={
                "managed": concept.managed.model_copy(
                    update={"content_hash": canonical_concept_hash(concept)}
                )
            }
        )
        if concept.evidence_state == "contested":
            contested.append(concept_ref(concept))
        catalog.write_concept(concept)
        changed.append(concept)
        _deprecate_superseded(
            catalog,
            concept,
            now=now,
            changed=changed,
            updated=updated,
        )
    return changed, created, updated, contested, sorted(rejected)


def _evidence_state(evidence, sources) -> str:
    supports = any(
        item.stance in {"supports", "illustrates"} for item in evidence
    )
    refutes = any(item.stance == "refutes" for item in evidence)
    if supports and refutes:
        return "contested"
    if refutes:
        return "falsified"
    if evidence:
        workspaces = {
            item.observation_ref.rsplit(":round-", 1)[0]
            for item in evidence
        }
        return "corroborated" if len(workspaces) > 1 else "observed"
    if sources:
        return "source_supported"
    raise ValueError("Concept has neither source nor measured evidence")


def _available_concept_id(
    preferred: str | None,
    *,
    kind: str,
    claim_key: str,
    scope: dict,
    catalog: FilesystemCatalog,
) -> str:
    slug = safe_name(claim_key.replace(".", "-")).lower()
    proposed = (
        preferred
        if preferred and preferred.startswith(f"kg:{kind}:")
        else f"kg:{kind}:{slug}"
    )
    ids = {item.id for item in catalog.iter_concepts()}
    if proposed not in ids:
        return proposed
    suffix = 2
    while f"{proposed}-{suffix}" in ids:
        suffix += 1
    return f"{proposed}-{suffix}"


def _unique_models(items):
    by_value = {
        json.dumps(
            item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ): item
        for item in items
    }
    return [by_value[key] for key in sorted(by_value)]


def _merge_concept_evidence(
    items: Sequence[ConceptEvidence],
) -> list[ConceptEvidence]:
    """Merge duplicate citations without losing their claim-relative meaning."""

    by_observation: dict[str, list[ConceptEvidence]] = defaultdict(list)
    for item in items:
        by_observation[item.observation_ref].append(item)
    merged = []
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    for observation_ref in sorted(by_observation):
        group = by_observation[observation_ref]
        stances = {item.stance for item in group}
        confidence = min(
            (item.confidence for item in group),
            key=confidence_rank.__getitem__,
        )
        merged.append(
            ConceptEvidence(
                observation_ref=observation_ref,
                stance=group[0].stance if len(stances) == 1 else "illustrates",
                rationale=min(item.rationale for item in group),
                confidence=confidence,
            )
        )
    return merged



def _validate_catalog_refs(
    candidate: CandidateConcept,
    catalog: FilesystemCatalog,
) -> None:
    concept_ids = {item.id for item in catalog.iter_concepts()}
    missing_relations = sorted(
        {item.target for item in candidate.relations} - concept_ids
    )
    if missing_relations:
        raise ValueError(
            "relation targets do not exist: " + ", ".join(missing_relations)
        )
    package_ids = {item.id for item in catalog.iter_source_packages()}
    missing_packages = sorted(
        {
            item.resource
            for item in candidate.source_refs
            if item.resource.startswith("source:")
            and item.resource not in package_ids
        }
    )
    if missing_packages:
        raise ValueError(
            "source packages do not exist: " + ", ".join(missing_packages)
        )


def _deprecate_superseded(
    catalog: FilesystemCatalog,
    concept: Concept,
    *,
    now: datetime,
    changed: list[Concept],
    updated: list[str],
) -> None:
    changed_ids = {item.id for item in changed}
    updated_ids = set(updated)
    for relation in concept.relations:
        if relation.type != "supersedes":
            continue
        if relation.target == concept.id:
            raise ValueError(
                f"Concept cannot supersede itself: {concept.id}"
            )
        previous = catalog.get_concept(relation.target)
        if previous.status == "deprecated":
            continue
        replacement = previous.model_copy(
            update={
                "status": "deprecated",
                "managed": previous.managed.model_copy(
                    update={
                        "content_hash": "sha256:" + "0" * 64,
                        "updated_at": now,
                    }
                ),
            }
        )
        replacement = replacement.model_copy(
            update={
                "managed": replacement.managed.model_copy(
                    update={
                        "content_hash": canonical_concept_hash(replacement)
                    }
                )
            }
        )
        catalog.write_concept(replacement)
        if replacement.id not in changed_ids:
            changed.append(replacement)
            changed_ids.add(replacement.id)
        if replacement.id not in updated_ids:
            updated.append(concept_ref(replacement))
            updated_ids.add(replacement.id)



def _without_permissive_exact(scope: Scope) -> Scope:
    if scope.numerics.exact is not False:
        return scope
    return scope.model_copy(
        update={
            "numerics": scope.numerics.model_copy(
                update={"exact": None}
            )
        }
    )

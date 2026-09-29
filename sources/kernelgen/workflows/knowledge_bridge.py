"""Thin KernelGen-to-knowledge integration boundary."""

from __future__ import annotations

import fcntl
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from kernelgen.data.experiment_plan import ExperimentPlan
from kernelgen.knowledge.bootstrap import build_workspace_services
from kernelgen.knowledge.catalog import (
    atomic_text,
    catalog_changed_paths,
    commit_catalog,
)
from kernelgen.knowledge.config import KnowledgeConfig
from kernelgen.knowledge.context import (
    KnowledgeWorkspaceMaterializer,
    build_operator_signature,
    build_target_context,
)
from kernelgen.knowledge.records import FileRetrievalAudit
from kernelgen.knowledge.round_index import SQLiteRoundIndex
from kernelgen.knowledge.run_archive import KernelGenRunArchive
from kernelgen.knowledge.run_facts import KernelGenRunFactReader
from kernelgen.knowledge.publishing.publisher import BatchPublisher
from kernelgen.knowledge.publishing.reviewer import ReviewCallable
from kernelgen.knowledge.models import (
    CandidateDraft,
    RuntimeCandidate,
)
from kernelgen.knowledge.contracts.runtime import (
    PublishResult,
    WorkspaceKnowledgeState,
)
from kernelgen.knowledge.layout import (
    CatalogLayout,
    DerivedLayout,
    KnowledgeLayout,
)
from kernelgen.knowledge.solutions import SolutionManifest, SolutionRegistry, SolutionSeed


class KernelGenKnowledgeBridge:
    def __init__(
        self,
        *,
        config: KnowledgeConfig,
        definition: Any,
        target_hardware: str,
        implementation_language: str,
        eval_server_url: str,
        run_id: str,
        benchmark_id: str = "",
        start_mode: str = "fresh",
    ):
        self.config = config
        self.definition = definition
        self.target_hardware = target_hardware
        self.implementation_language = implementation_language
        self.run_id = run_id
        self.benchmark_id = benchmark_id or "default"
        self.start_mode = start_mode
        self.parent_solution_ref = ""
        service_status = _service_status(eval_server_url)
        self.operator_signature = build_operator_signature(
            definition,
            catalog_root=self.config.catalog_root,
        )
        self.target_context = build_target_context(
            target_hardware=target_hardware,
            implementation_language=implementation_language,
            service_status=service_status,
        )

    @property
    def enabled(self) -> bool:
        return True

    def materializer(self) -> KnowledgeWorkspaceMaterializer:
        return KnowledgeWorkspaceMaterializer(
            catalog_root=self.config.catalog_root,
            mode=self.config.mode,
            run_id=self.run_id,
            operator_signature=self.operator_signature,
            target_context=self.target_context,
            start_mode=self.start_mode,
            parent_solution_ref=self.parent_solution_ref,
            derived_root=self.config.resolved_derived_root,
            index_rebuild_enabled=self.config.index_rebuild_enabled,
        )

    def resolve_fork_seed(self) -> SolutionSeed:
        """Return the exact-scope best kernel used to seed a forked run."""
        if not self.enabled or not self.config.reads_v1:
            raise ValueError("fork requires a readable V1 knowledge catalog")
        seed = SolutionRegistry(self.config.catalog_root).resolve(
            self.operator_signature,
            self.target_context,
            benchmark_id=self.benchmark_id,
        )
        if seed is None:
            raise ValueError(
                "no exact best solution for definition, target, and benchmark"
            )
        self.start_mode = "fork"
        self.parent_solution_ref = seed.manifest.solution_ref
        return seed

    def promote_solution(self, workspace: Path) -> SolutionManifest | None:
        """Promote one workspace best kernel into its exact registry slot."""
        if not self.config.writes_v1:
            return None
        workspace = Path(workspace)
        state_path = KnowledgeLayout(workspace).state
        if not state_path.is_file():
            return None
        state = WorkspaceKnowledgeState.model_validate_json(
            state_path.read_text(encoding="utf-8")
        )
        from kernelgen.data.ledger import Ledger

        ledger = Ledger(workspace)
        best = ledger.best
        code = best["code"]
        geo_mean = float(best.get("geo_mean") or 0.0)
        round_num = int(best.get("round") or 0)
        if not code or geo_mean <= 0 or round_num <= 0:
            return None
        layout = CatalogLayout(self.config.catalog_root)
        layout.publish_transaction_lock.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with layout.publish_transaction_lock.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            catalog_paths_before = catalog_changed_paths(
                self.config.catalog_root
            )
            archive = KernelGenRunArchive(
                self.config.resolved_run_archive_root
            ).archive_workspace(
                workspace,
                state,
                self.operator_signature,
                self.target_context,
                run_id=state.run_id,
            )
            manifest = SolutionRegistry(self.config.catalog_root).promote(
                signature=self.operator_signature,
                target=self.target_context,
                benchmark_id=self.benchmark_id,
                run_id=state.run_id,
                workspace_id=state.workspace_id,
                round_num=round_num,
                archive_ref=archive.ref,
                code=code,
                geo_mean=geo_mean,
                start_mode=state.start_mode,
                parent_solution_ref=state.parent_solution_ref,
            )
            if manifest is not None:
                transaction_paths = (
                    catalog_changed_paths(self.config.catalog_root)
                    - catalog_paths_before
                )
                commit_catalog(
                    self.config.catalog_root,
                    f"{state.run_id}: promoted {manifest.solution_ref}",
                    paths=transaction_paths,
                )
            return manifest

    def publish_epoch(
        self,
        workspaces: Iterable[Path],
        *,
        epoch_num: int,
        reviewer: ReviewCallable | None = None,
    ):
        workspaces = [Path(item) for item in workspaces]
        initialized = [
            item
            for item in workspaces
            if KnowledgeLayout(item).state.is_file()
        ]
        if not initialized:
            return None
        if not self.config.writes_v1:
            result = PublishResult(
                status="noop",
                reviewer_mode=self.config.reviewer_mode.value,
            )
            rendered = result.model_dump_json(indent=2) + "\n"
            for workspace in initialized:
                atomic_text(
                    KnowledgeLayout(workspace).publish_result,
                    rendered,
                )
            return result
        layout = CatalogLayout(self.config.catalog_root)
        layout.publish_transaction_lock.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with layout.publish_transaction_lock.open(
            "a+",
            encoding="utf-8",
        ) as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if reviewer is None:
                return self._publish_epoch_locked(
                    initialized,
                    epoch_num=epoch_num,
                )
            return self._publish_epoch_locked(
                initialized,
                epoch_num=epoch_num,
                reviewer=reviewer,
            )

    def _publish_epoch_locked(
        self,
        workspaces: list[Path],
        *,
        epoch_num: int,
        reviewer: ReviewCallable | None = None,
    ):
        """Publish one epoch while holding the catalog transaction lock."""
        catalog_paths_before = catalog_changed_paths(
            self.config.catalog_root
        )
        run_archive = KernelGenRunArchive(
            self.config.resolved_run_archive_root
        )
        for workspace in workspaces:
            state = json.loads(
                KnowledgeLayout(workspace).state.read_text(encoding="utf-8")
            )
            run_archive.archive_workspace(
                workspace,
                WorkspaceKnowledgeState.model_validate(state),
                self.operator_signature,
                self.target_context,
                run_id=self.run_id,
            )
        SQLiteRoundIndex(
            DerivedLayout(self.config.resolved_derived_root).round_index
        ).rebuild(self.config.resolved_run_archive_root)
        batch_id = f"{self.run_id}-epoch-{epoch_num}"
        result = BatchPublisher(
            self.config.catalog_root,
            KernelGenRunFactReader(run_archive),
        ).publish(
            workspaces,
            run_id=self.run_id,
            batch_id=batch_id,
            reviewer=reviewer,
            reviewer_mode=self.config.reviewer_mode,
        )
        rendered = result.model_dump_json(indent=2) + "\n"
        for workspace in workspaces:
            layout = KnowledgeLayout(workspace)
            if layout.state.is_file():
                atomic_text(layout.publish_result, rendered)
        if result.status == "published":
            transaction_paths = (
                catalog_changed_paths(self.config.catalog_root)
                - catalog_paths_before
            )
            commit_catalog(
                self.config.catalog_root,
                (
                    f"{batch_id}: published "
                    f"{len(result.processed_candidates)} V1 candidates"
                ),
                paths=transaction_paths,
            )
        return result


def validate_knowledge_uses(workspace: Path, plan: ExperimentPlan) -> None:
    layout = KnowledgeLayout(Path(workspace))
    if not layout.state.is_file():
        return
    # Knowledge-use declarations are provenance notes, not permission tokens.
    # Their schema is validated with the plan; missing/stale query logs must not
    # block a real evaluation.


def recent_detail_read_refs(
    workspace: Path,
    *,
    after: datetime | None,
) -> list[str]:
    """Return successful Concept and Source detail reads after a round boundary.

    Retrieval audit is advisory provenance. Missing or malformed audit state
    must not block a real evaluation.
    """

    layout = KnowledgeLayout(Path(workspace))
    try:
        if not layout.state.is_file():
            return []
        records = FileRetrievalAudit(
            layout.retrieval_log,
            legacy_query_path=layout.query_log,
        ).read_all()
        references: list[str] = []
        seen: set[str] = set()
        for record in records:
            if (
                record.status != "success"
                or record.operation not in {"get_knowledge", "get_source"}
                or (after is not None and record.created_at <= after)
            ):
                continue
            returned = (
                record.returned_refs
                if record.operation == "get_knowledge"
                else record.returned_sources
            )
            for raw_reference in returned:
                reference = str(raw_reference)
                if reference and reference not in seen:
                    seen.add(reference)
                    references.append(reference)
        return references
    except (OSError, TypeError, ValueError):
        return []


def append_candidate_drafts(
    workspace: Path,
    drafts: Iterable[CandidateDraft],
) -> list[str]:
    layout = KnowledgeLayout(Path(workspace))
    if not layout.state.is_file():
        return []
    state = json.loads(layout.state.read_text(encoding="utf-8"))
    services = build_workspace_services(workspace)
    existing = services.outbox.read_all()
    next_index = len(existing) + 1
    written = []
    for index, draft in enumerate(drafts, start=next_index):
        candidate_id = (
            f"{_safe(state['run_id'])}:{_safe(state['workspace_id'])}:"
            f"candidate-{index:04d}"
        )
        candidate = RuntimeCandidate(
            candidate_id=candidate_id,
            created_by=state["workspace_id"],
            **draft.model_dump(mode="python"),
        )
        services.outbox.append(candidate)
        written.append(candidate_id)
    return written


def _service_status(server_url: str) -> dict:
    if not server_url:
        raise ValueError("KernelGen Server URL is required for Knowledge mode")
    from kernelgen.tools.kernelgen_server_adapter import get_service_status

    return get_service_status(server_url, timeout=45)


def _safe(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in value
    )
    return normalized.strip("-") or "unknown"

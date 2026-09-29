"""Composition roots for catalog and workspace knowledge services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kernelgen.knowledge.records import (
    FileCandidateOutbox,
    FileRetrievalAudit,
)
from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.index import SQLiteKnowledgeIndex
from kernelgen.knowledge.round_index import SQLiteRoundIndex
from kernelgen.knowledge.query import (
    GetKnowledge,
    QueryKnowledge,
)
from kernelgen.knowledge.sources import SourceSearcher
from kernelgen.knowledge.config import KnowledgeConfig
from kernelgen.knowledge.models import (
    OperatorSignature,
    TargetContext,
    KnowledgeUsageScope,
)
from kernelgen.knowledge.contracts.runtime import WorkspaceKnowledgeState
from kernelgen.knowledge.layout import (
    CatalogLayout,
    DerivedLayout,
    KnowledgeLayout,
)


@dataclass(frozen=True)
class KnowledgeServices:
    query: QueryKnowledge
    get: GetKnowledge
    outbox: FileCandidateOutbox
    audit: FileRetrievalAudit
    retrieval: FileRetrievalAudit
    sources: SourceSearcher
    rounds: SQLiteRoundIndex


def build_workspace_services(workspace: Path) -> KnowledgeServices:
    workspace = Path(workspace).resolve()
    layout = KnowledgeLayout(workspace)
    state = WorkspaceKnowledgeState.model_validate_json(
        layout.state.read_text(encoding="utf-8")
    )
    catalog_root = resolve_workspace_catalog(workspace, state)
    derived = DerivedLayout(
        resolve_workspace_derived(workspace, state, catalog_root)
    )
    allow_index_rebuild = state.index_rebuild_enabled
    catalog = FilesystemCatalog(catalog_root)
    events = FileRetrievalAudit(
        layout.retrieval_log,
        legacy_query_path=layout.query_log,
    )
    index = SQLiteKnowledgeIndex(
        derived.index,
        read_only=not allow_index_rebuild,
    )
    signature = OperatorSignature.model_validate_json(
        layout.operator_signature.read_text(encoding="utf-8")
    )
    target = TargetContext.model_validate_json(
        layout.target_context.read_text(encoding="utf-8")
    )
    return KnowledgeServices(
        query=QueryKnowledge(
            catalog,
            index,
            events,
            allow_index_rebuild=allow_index_rebuild,
        ),
        get=GetKnowledge(catalog, events),
        outbox=FileCandidateOutbox(layout.candidates),
        audit=events,
        retrieval=events,
        sources=SourceSearcher(
            catalog_root,
            events,
            usage_index=index,
            usage_scope=KnowledgeUsageScope.from_context(
                signature,
                target,
            ),
            source_index_path=derived.source_index,
            allow_index_rebuild=allow_index_rebuild,
        ),
        rounds=SQLiteRoundIndex(
            derived.round_index,
            read_only=not allow_index_rebuild,
        ),
    )


def resolve_workspace_catalog(
    workspace: Path,
    state: WorkspaceKnowledgeState,
) -> Path:
    """Resolve the live catalog selected for a workspace."""

    workspace = Path(workspace).resolve()
    catalog_root = Path(state.catalog_ref)
    if not catalog_root.is_absolute():
        catalog_root = (workspace / catalog_root).resolve()
    return catalog_root


def resolve_workspace_derived(
    workspace: Path,
    state: WorkspaceKnowledgeState,
    catalog_root: Path,
) -> Path:
    """Resolve workspace-selected disposable indexes without mutating state."""

    if state.derived_ref:
        derived_root = Path(state.derived_ref)
        if not derived_root.is_absolute():
            derived_root = (Path(workspace).resolve() / derived_root).resolve()
    else:
        derived_root = CatalogLayout(catalog_root).derived
    if state.mode == "read_only_v1" and state.index_rebuild_enabled:
        catalog = Path(catalog_root).resolve()
        derived = derived_root.resolve()
        if derived == catalog or catalog in derived.parents:
            raise RuntimeError(
                "read_only_v1 rebuildable indexes must be outside "
                "catalog_root"
            )
    return derived_root


def build_catalog_services(
    config: KnowledgeConfig,
    *,
    audit_path: Path,
    candidate_path: Path,
) -> KnowledgeServices:
    catalog = FilesystemCatalog(config.catalog_root)
    derived = DerivedLayout(config.resolved_derived_root)
    allow_index_rebuild = config.index_rebuild_enabled
    events = FileRetrievalAudit(
        audit_path.with_name("retrieval-log.jsonl"),
        legacy_query_path=audit_path,
    )
    index = SQLiteKnowledgeIndex(
        derived.index,
        read_only=not allow_index_rebuild,
    )
    return KnowledgeServices(
        query=QueryKnowledge(
            catalog,
            index,
            events,
            allow_index_rebuild=allow_index_rebuild,
        ),
        get=GetKnowledge(catalog, events),
        outbox=FileCandidateOutbox(candidate_path),
        audit=events,
        retrieval=events,
        sources=SourceSearcher(
            config.catalog_root,
            events,
            usage_index=index,
            source_index_path=derived.source_index,
            allow_index_rebuild=allow_index_rebuild,
        ),
        rounds=SQLiteRoundIndex(
            derived.round_index,
            read_only=not allow_index_rebuild,
        ),
    )

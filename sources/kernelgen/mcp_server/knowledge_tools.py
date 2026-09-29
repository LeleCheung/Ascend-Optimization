"""MCP adapter functions for framework-neutral knowledge services."""

from __future__ import annotations

from pathlib import Path

from kernelgen.data.profile_analysis import ProfileFinding
from kernelgen.knowledge.bootstrap import build_workspace_services
from kernelgen.knowledge.context import build_query_context
from kernelgen.knowledge.models import (
    OperatorSignature,
    TargetContext,
    ProfileFindingContext,
)
from kernelgen.knowledge.contracts.runtime import (
    RoundSearchResult,
    WorkspaceKnowledgeState,
)
from kernelgen.knowledge.layout import KnowledgeLayout
from kernelgen.tools.profile_round import validate_profile_query_findings


def query_knowledge(
    workspace: Path,
    *,
    phase: str,
    task: str,
    question: str,
    round_num: int | None = None,
    draft_findings: list[ProfileFinding] | None = None,
) -> dict:
    if draft_findings is not None:
        finding_contexts = [
            ProfileFindingContext(
                category=item.category,
                label=item.label,
                confidence=item.confidence,
                workload_uuids=item.workload_uuids,
            )
            for item in draft_findings
        ]
        if round_num is not None:
            try:
                finding_contexts = validate_profile_query_findings(
                    workspace,
                    round_num,
                    draft_findings,
                )
            except ValueError:
                # Querying is advisory. Missing profile artifacts must not make
                # otherwise useful model findings unsearchable.
                pass
    else:
        finding_contexts = None
    context = build_query_context(
        workspace,
        phase=phase,
        task=task,
        question=question,
        round_num=round_num,
        draft_findings=finding_contexts,
    )
    bundle = build_workspace_services(workspace).query.execute(
        context,
        origin="mcp",
    )
    state = _state(workspace)
    if state.mode == "shadow":
        return {
            "status": "SHADOW",
            "query_id": bundle.query_id,
            "snapshot": bundle.snapshot,
            "result_counts": {
                "direct": len(bundle.direct),
                "analogies": len(bundle.analogies),
                "conflicts": len(bundle.conflicts),
            },
            "message": "V1 query recorded; Agent output remains on legacy KB.",
        }
    return bundle.model_dump(mode="json")


def get_knowledge(
    workspace: Path,
    *,
    query_id: str = "",
    concept_refs: list[str],
    detail_level: str = "full",
) -> dict:
    state = _state(workspace)
    if state.mode == "shadow":
        raise ValueError("get_knowledge is unavailable in shadow mode")
    documents = build_workspace_services(workspace).get.execute(
        query_id,
        concept_refs,
        detail_level,
        origin="mcp",
    )
    return {
        "query_id": query_id,
        "documents": [item.model_dump(mode="json") for item in documents],
    }


def query_sources(
    workspace: Path,
    *,
    query: str,
    package_ids: list[str] | None = None,
    path_globs: list[str] | None = None,
    include_source_only: bool = False,
    max_results: int = 12,
) -> dict:
    state = _state(workspace)
    if state.mode == "shadow":
        return {
            "status": "SHADOW",
            "message": "Source query is disabled for Agent decisions in shadow mode.",
        }
    result = build_workspace_services(workspace).sources.search(
        query,
        package_ids=package_ids,
        path_globs=path_globs,
        include_source_only=include_source_only,
        max_results=max_results,
        origin="mcp",
    )
    return result.model_dump(mode="json")


def get_source(
    workspace: Path,
    *,
    query_id: str = "",
    source_package: str,
    path: str,
    line_start: int = 1,
    line_end: int | None = None,
) -> dict:
    state = _state(workspace)
    if state.mode == "shadow":
        raise ValueError("get_source is unavailable in shadow mode")
    document = build_workspace_services(workspace).sources.read(
        query_id=query_id,
        source_package=source_package,
        path=path,
        line_start=line_start,
        line_end=line_end,
        origin="mcp",
    )
    return document.model_dump(mode="json")


def search_rounds(
    workspace: Path,
    *,
    query: str,
    scope: str = "exact",
    max_results: int = 12,
) -> dict:
    """Search archived rounds without turning every round into KB knowledge."""

    if scope not in {"exact", "definition"}:
        raise ValueError("scope must be exact or definition")
    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")
    layout = KnowledgeLayout(Path(workspace))
    signature = OperatorSignature.model_validate_json(
        layout.operator_signature.read_text(encoding="utf-8")
    )
    target = TargetContext.model_validate_json(
        layout.target_context.read_text(encoding="utf-8")
    )
    hits = build_workspace_services(workspace).rounds.search(
        query,
        definition_id=signature.definition_id,
        target_backend=target.backend if scope == "exact" else "",
        target_architecture=(
            target.architecture if scope == "exact" else ""
        ),
        target_device=target.device if scope == "exact" else "",
        limit=max_results,
    )
    return RoundSearchResult(
        query=query,
        scope=scope,
        hits=hits,
    ).model_dump(mode="json")


def _state(workspace: Path) -> WorkspaceKnowledgeState:
    return WorkspaceKnowledgeState.model_validate_json(
        KnowledgeLayout(Path(workspace)).state.read_text(encoding="utf-8")
    )

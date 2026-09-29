"""Runtime contracts shared across query, workspace, and publishing adapters."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kernelgen.knowledge.models import (
    KnowledgeUsageSummary,
    QueryPhase,
    QueryTask,
)


class RuntimeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WorkspaceKnowledgeState(RuntimeContract):
    schema_version: Literal["1.0"] = "1.0"
    catalog_ref: str = Field(min_length=1)
    mode: Literal["read_write_v1", "read_only_v1"]
    derived_ref: str = ""
    index_rebuild_enabled: bool = True
    run_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    operator_signature_ref: str = Field(min_length=1)
    target_context_ref: str = Field(min_length=1)
    start_mode: Literal["fresh", "fork", "resume"] = "fresh"
    parent_solution_ref: str = ""

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_snapshots(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            value.pop("snapshot", None)
            value.pop("round_snapshot", None)
            if "index_rebuild_enabled" not in value:
                value["index_rebuild_enabled"] = (
                    value.get("mode") != "read_only_v1"
                )
        return value


class SearchRoute(RuntimeContract):
    route: str = Field(min_length=1)
    term: str = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0)


class SearchHit(RuntimeContract):
    concept_id: str = Field(pattern=r"^kg:(reference|method|diagnostic|experience):")
    score: float = 0.0
    matched_on: List[str] = Field(default_factory=list)


class RoundSearchHit(RuntimeContract):
    run_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    round_num: int = Field(gt=0)
    archive_ref: str = Field(pattern=r"^archive://")
    definition_id: str = Field(min_length=1)
    target_backend: str = Field(min_length=1)
    target_architecture: str = ""
    target_device: str = Field(min_length=1)
    status: str = Field(min_length=1)
    geo_mean: Optional[float] = None
    performance_baseline_round_num: Optional[int] = Field(
        default=None,
        gt=0,
    )
    geo_mean_delta_pct: Optional[float] = None
    strategy: str = Field(min_length=1)
    code_changes: str = Field(min_length=1)
    key_params: Dict[str, object] = Field(default_factory=dict)
    profile_summary: str = ""
    conclusion: str = ""
    score: float = 0.0


class RoundSearchResult(RuntimeContract):
    schema_version: Literal["1.0"] = "1.0"
    query: str
    scope: Literal["exact", "definition"]
    hits: List[RoundSearchHit] = Field(default_factory=list)


class ScopeMatch(RuntimeContract):
    level: Literal["direct", "analogy", "incompatible", "unknown"]
    matched_on: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)


class QueryRecord(RuntimeContract):
    schema_version: Literal["1.0"] = "1.0"
    query_id: str = Field(pattern=r"^query:[A-Za-z0-9._:-]+$")
    snapshot: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    origin: Literal["application", "orchestrator", "mcp"] = "application"
    phase: QueryPhase
    task: QueryTask
    question: str = ""
    returned_refs: List[str] = Field(default_factory=list)
    result_levels: Dict[str, Literal["direct", "analogy", "conflict"]] = Field(
        default_factory=dict
    )
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_context_fingerprint(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            value.pop("context_fingerprint", None)
        return value


class RetrievalRecord(RuntimeContract):
    """One completed knowledge read, independent of the calling interface."""

    schema_version: Literal["1.0"] = "1.0"
    event_id: str = Field(
        pattern=r"^(query|source-query|retrieval):[A-Za-z0-9._:-]+$"
    )
    operation: Literal[
        "query_knowledge",
        "get_knowledge",
        "query_sources",
        "get_source",
    ]
    origin: Literal["application", "orchestrator", "mcp"]
    status: Literal["success", "error"]
    query_id: str = ""
    snapshot: str = Field(
        default="",
        pattern=r"^(sha256:[0-9a-f]{64})?$",
    )
    request: Dict[str, object] = Field(default_factory=dict)
    returned_refs: List[str] = Field(default_factory=list)
    returned_sources: List[str] = Field(default_factory=list)
    result_levels: Dict[
        str,
        Literal["direct", "analogy", "conflict"],
    ] = Field(default_factory=dict)
    error: str = ""
    created_at: datetime


class KnowledgeDocument(RuntimeContract):
    query_id: str = Field(default="", pattern=r"^(query:[A-Za-z0-9._:-]+)?$")
    concept_ref: str = Field(
        pattern=r"^kg:(reference|method|diagnostic|experience):"
        r"[a-z0-9][a-z0-9._-]*$"
    )
    title: str
    summary: str
    body: str
    sources: List[dict] = Field(default_factory=list)
    evidence: List[dict] = Field(default_factory=list)


class SourceSearchHit(RuntimeContract):
    source_package: str = Field(pattern=r"^source:[a-z0-9][a-z0-9._-]*$")
    revision: str = Field(min_length=1)
    path: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    classification: Literal[
        "source_only",
        "indexed",
        "direct",
        "extract",
        "federated",
    ]
    usage_policy: Literal[
        "redistributable",
        "restricted",
        "federated_only",
        "unknown",
    ]
    usage_notes: str = ""
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    score: float = Field(gt=0)
    snippet: str = Field(min_length=1)
    usage_summary: KnowledgeUsageSummary = Field(
        default_factory=KnowledgeUsageSummary
    )
    recommendation_reasons: List[str] = Field(default_factory=list)
    promoted_concept_refs: List[str] = Field(default_factory=list)


class SourceSearchResult(RuntimeContract):
    schema_version: Literal["1.0"] = "1.0"
    query_id: str = Field(pattern=r"^source-query:[A-Za-z0-9._:-]+$")
    snapshot: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    query: str = Field(min_length=1)
    expanded_terms: List[str] = Field(default_factory=list)
    hits: List[SourceSearchHit] = Field(default_factory=list)


class SourceDocument(RuntimeContract):
    schema_version: Literal["1.0"] = "1.0"
    query_id: str = Field(
        default="",
        pattern=r"^(source-query:[A-Za-z0-9._:-]+)?$",
    )
    snapshot: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_package: str = Field(pattern=r"^source:[a-z0-9][a-z0-9._-]*$")
    revision: str = Field(min_length=1)
    path: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    usage_policy: Literal[
        "redistributable",
        "restricted",
        "federated_only",
        "unknown",
    ]
    usage_notes: str = ""
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    total_lines: int = Field(ge=0)
    truncated: bool = False
    content: str


class PublishResult(RuntimeContract):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["published", "noop"]
    reviewer_mode: Literal["off", "shadow", "enforce"] = "off"
    created: List[str] = Field(default_factory=list)
    updated: List[str] = Field(default_factory=list)
    observations_added: List[str] = Field(default_factory=list)
    contested: List[str] = Field(default_factory=list)
    processed_candidates: List[str] = Field(default_factory=list)
    approved_candidates: List[str] = Field(default_factory=list)
    shadow_approved_candidates: List[str] = Field(default_factory=list)
    deferred_candidates: List[str] = Field(default_factory=list)
    review_rejected_candidates: List[str] = Field(default_factory=list)
    rejected: List[str] = Field(default_factory=list)
    audit_warnings: List[str] = Field(default_factory=list)
    review_warnings: List[str] = Field(default_factory=list)
    audit_ref: str = ""
    review_audit_ref: str = ""

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_reviewer_mode(cls, value):
        if isinstance(value, dict) and "reviewer_mode" not in value:
            value = dict(value)
            reviewed = bool(value.get("review_audit_ref")) or any(
                value.get(field)
                for field in (
                    "approved_candidates",
                    "deferred_candidates",
                    "review_rejected_candidates",
                )
            )
            value["reviewer_mode"] = "enforce" if reviewed else "off"
        return value

"""Runtime target, operator, query, and retrieval result models."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from kernelgen.knowledge.models.base import (
    SCHEMA_VERSION,
    ConceptKind,
    EvidenceState,
    QueryPhase,
    QueryTask,
    StrictModel,
)
from kernelgen.knowledge.models.concept import NumericsScope
from kernelgen.knowledge.models.observation import KnowledgeUsageSummary


class SoftwareContext(StrictModel):
    language: str = ""
    language_version: str = ""
    compiler: str = ""
    compiler_version: str = ""
    runtime: str = ""
    runtime_version: str = ""
    library: str = ""
    library_version: str = ""
    driver_version: str = ""


class TargetMetadata(StrictModel):
    complete: bool = False
    missing: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_consistency(self) -> "TargetMetadata":
        if self.complete and self.missing:
            raise ValueError("complete target metadata cannot list missing fields")
        return self


class TargetContext(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    backend: str = Field(min_length=1)
    vendor: str = ""
    architecture: str = ""
    device: str = Field(min_length=1)
    capabilities: List[str] = Field(default_factory=list)
    software: SoftwareContext = Field(default_factory=SoftwareContext)
    metadata: Optional[TargetMetadata] = None
    source: Literal["eval_service", "profile_service", "orchestrator", "fixture"]

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_fingerprint(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            value.pop("fingerprint", None)
        return value


class OperatorSignature(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    definition_id: str = Field(min_length=1)
    definition_name: str = Field(min_length=1)
    op_type: str = ""
    motifs: List[str] = Field(default_factory=list)
    dataflow: List[str] = Field(default_factory=list)
    dtypes: List[str] = Field(default_factory=list)
    layouts: List[str] = Field(default_factory=list)
    workload_features: Dict[str, Any] = Field(default_factory=dict)
    required_capabilities: List[str] = Field(default_factory=list)
    numerics: NumericsScope = Field(default_factory=NumericsScope)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_placeholders(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            value.pop("semantic_names", None)
            value.pop("structure_paths", None)
            value.pop("extensions", None)
        return value


class ProfileFindingContext(StrictModel):
    category: str = Field(min_length=1)
    label: str = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]
    workload_uuids: List[str] = Field(default_factory=list)


class QueryContext(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    phase: QueryPhase
    task: QueryTask
    round_num: Optional[int] = Field(default=None, gt=0)
    question: str = Field(min_length=1)
    operator_signature: OperatorSignature
    target_context: TargetContext
    findings: List[ProfileFindingContext] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    max_results: int = Field(default=12, ge=1, le=100)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_workload_summary(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            value.pop("workload_summary", None)
        return value

class BundleCoverage(StrictModel):
    matched_routes: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)


class KnowledgeResult(StrictModel):
    concept_ref: str = Field(
        pattern=r"^kg:(reference|method|diagnostic|experience):[a-z0-9][a-z0-9._-]*$"
    )
    kind: ConceptKind
    summary: str = Field(min_length=1)
    matched_on: List[str] = Field(default_factory=list)
    scope_gaps: List[str] = Field(default_factory=list)
    evidence_state: EvidenceState
    usage_summary: KnowledgeUsageSummary = Field(
        default_factory=KnowledgeUsageSummary
    )
    recommendation_reasons: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_payload_fields(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            value.pop("usage", None)
            value.pop("evidence", None)
        return value


class KnowledgeBundle(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    query_id: str = Field(pattern=r"^query:[A-Za-z0-9._:-]+$")
    snapshot: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    direct: List[KnowledgeResult] = Field(default_factory=list)
    analogies: List[KnowledgeResult] = Field(default_factory=list)
    conflicts: List[KnowledgeResult] = Field(default_factory=list)
    coverage: BundleCoverage = Field(default_factory=BundleCoverage)

    @model_validator(mode="after")
    def validate_result_buckets(self) -> "KnowledgeBundle":
        if any(item.scope_gaps for item in self.direct):
            raise ValueError("direct results cannot contain scope_gaps")
        return self

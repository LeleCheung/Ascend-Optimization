"""Concept, evidence, and applicability scope models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from kernelgen.knowledge.models.base import (
    SCHEMA_VERSION,
    ConceptKind,
    ConceptStatus,
    EvidenceConfidence,
    EvidenceStance,
    EvidenceState,
    QueryPhase,
    QueryTask,
    StrictModel,
    TargetLevel,
    WorkloadOperator,
    _TYPE_TO_KIND,
    _VERSION_CONSTRAINT,
)


class VerificationInfo(StrictModel):
    by: str = Field(min_length=1)
    at: datetime


class SourceReference(StrictModel):
    resource: str = Field(min_length=1)
    revision: str = ""
    locator: str = ""
    title: str = ""

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, value):
        """Read old Concept files while publishing only exact source identity."""
        if isinstance(value, dict):
            value = dict(value)
            value.setdefault("locator", value.pop("id", ""))
            value.pop("author", None)
            value.pop("usage_count", None)
            value.pop("last_modified", None)
        return value

    @property
    def id(self) -> str:
        """Compatibility accessor for callers reading pre-revision references."""
        return self.locator


class SoftwareScope(StrictModel):
    language: str = ""
    language_version: str = ""
    compiler: str = ""
    compiler_version: str = ""
    runtime: str = ""
    runtime_version: str = ""
    library: str = ""
    library_version: str = ""
    driver_version: str = ""

    @field_validator(
        "language_version",
        "compiler_version",
        "runtime_version",
        "library_version",
        "driver_version",
    )
    @classmethod
    def validate_version_constraint(cls, value: str) -> str:
        if value and not _VERSION_CONSTRAINT.fullmatch(value):
            raise ValueError("version constraint contains unsupported characters")
        return value


class TargetScope(StrictModel):
    level: TargetLevel
    backend: str = ""
    architecture: str = ""
    devices: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    software: SoftwareScope = Field(default_factory=SoftwareScope)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_fingerprint(cls, value):
        """Read existing V1 files without publishing opaque identity again."""
        if isinstance(value, dict) and "target_fingerprint" in value:
            value = dict(value)
            value.pop("target_fingerprint", None)
        return value


class OperatorScope(StrictModel):
    definition_ids: List[str] = Field(default_factory=list)
    op_types: List[str] = Field(default_factory=list)
    motifs: List[str] = Field(default_factory=list)
    dataflow: List[str] = Field(default_factory=list)
    dtypes: List[str] = Field(default_factory=list)
    layouts: List[str] = Field(default_factory=list)


class WorkloadPredicate(StrictModel):
    field: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")
    op: WorkloadOperator
    value: Any

    @model_validator(mode="after")
    def validate_value_shape(self) -> "WorkloadPredicate":
        if self.op in {"in", "not_in"}:
            if not isinstance(self.value, list) or not self.value:
                raise ValueError(f"{self.op} requires a non-empty list value")
            if any(isinstance(item, (list, dict)) for item in self.value):
                raise ValueError(f"{self.op} list items must be scalar values")
        elif isinstance(self.value, (list, dict)):
            raise ValueError(f"{self.op} requires a scalar value")
        return self


class WorkloadScope(StrictModel):
    all: List[WorkloadPredicate] = Field(default_factory=list)
    any: List[WorkloadPredicate] = Field(default_factory=list)
    excludes: List[WorkloadPredicate] = Field(default_factory=list)


class NumericsScope(StrictModel):
    exact: Optional[bool] = None
    allow_tf32: Optional[bool] = None
    accumulation_dtypes: List[str] = Field(default_factory=list)
    max_abs_error: Optional[float] = Field(default=None, ge=0)
    max_rel_error: Optional[float] = Field(default=None, ge=0)


class Scope(StrictModel):
    target: TargetScope
    operator: OperatorScope = Field(default_factory=OperatorScope)
    workloads: WorkloadScope = Field(default_factory=WorkloadScope)
    numerics: NumericsScope = Field(default_factory=NumericsScope)


class RetrievalMetadata(StrictModel):
    phases: List[QueryPhase] = Field(default_factory=list)
    tasks: List[QueryTask] = Field(default_factory=list)
    symptoms: List[str] = Field(default_factory=list)
    techniques: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class ConceptRelation(StrictModel):
    type: Literal[
        "requires",
        "addresses",
        "contradicts",
        "supersedes",
        "refines",
        "broader_than",
        "narrower_than",
        "related",
    ]
    target: str = Field(pattern=r"^kg:(reference|method|diagnostic|experience):")


class ManagedMetadata(StrictModel):
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    created_by: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime


class ConceptEvidence(StrictModel):
    """Why one immutable Observation is evidence for this Concept."""

    observation_ref: str = Field(
        pattern=(
            r"^kg:observation:[A-Za-z0-9._-]+:"
            r"[A-Za-z0-9._-]+:round-[1-9][0-9]*$"
        )
    )
    stance: EvidenceStance
    rationale: str = Field(min_length=1, max_length=1000)
    confidence: EvidenceConfidence = "medium"


class Concept(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    id: str = Field(pattern=r"^kg:(reference|method|diagnostic|experience):[a-z0-9][a-z0-9._-]*$")
    kind: ConceptKind
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    claim_key: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    domains: List[str] = Field(min_length=1)
    status: ConceptStatus = "stable"
    verified: List[VerificationInfo] = Field(default_factory=list)
    stale_after: Optional[date] = None
    sources: List[SourceReference] = Field(default_factory=list)
    scope: Scope
    retrieval: RetrievalMetadata = Field(default_factory=RetrievalMetadata)
    evidence_state: EvidenceState
    evidence: List[ConceptEvidence] = Field(default_factory=list)
    relations: List[ConceptRelation] = Field(default_factory=list)
    managed: ManagedMetadata
    body: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_metadata(cls, value):
        """Read legacy type/generated fields without publishing them again."""
        if not isinstance(value, dict):
            return value
        value = dict(value)
        value.pop("revision", None)
        legacy_type = value.pop("type", None)
        kind = value.get("kind")
        if legacy_type is not None:
            legacy_kind = _TYPE_TO_KIND.get(legacy_type)
            if legacy_kind is None:
                raise ValueError(
                    f"unsupported legacy Concept type: {legacy_type}"
                )
            if kind is not None and kind != legacy_kind:
                raise ValueError("type and kind do not match")
            value.setdefault("kind", legacy_kind)

        legacy_generated = value.pop("generated", None)
        managed = dict(value.get("managed") or {})
        if isinstance(legacy_generated, dict):
            managed.setdefault("created_by", legacy_generated.get("by"))
            managed.setdefault("created_at", legacy_generated.get("at"))
        value["managed"] = managed
        return value

    @model_validator(mode="after")
    def validate_kind_and_support(self) -> "Concept":
        expected_prefix = f"kg:{self.kind}:"
        if not self.id.startswith(expected_prefix):
            raise ValueError("id prefix does not match kind")
        if self.kind == "experience" and not self.evidence:
            raise ValueError("experience requires at least one Observation")
        if self.kind == "reference" and not self.evidence and not self.sources:
            raise ValueError("reference requires sources or measured observations")
        if self.evidence_state != "source_supported" and not self.evidence:
            raise ValueError(
                f"{self.evidence_state} concept requires measured observations"
            )
        if self.evidence_state == "source_supported" and not self.sources:
            raise ValueError("source_supported concept requires sources")
        return self

"""Agent-authored knowledge candidates before publication."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import AliasChoices, Field, model_validator

from kernelgen.knowledge.models.base import (
    SCHEMA_VERSION,
    ConceptKind,
    EvidenceConfidence,
    EvidenceStance,
    PublishAction,
    StrictModel,
)
from kernelgen.knowledge.models.concept import (
    ConceptRelation,
    RetrievalMetadata,
    Scope,
    SourceReference,
)


class ObservationIntent(StrictModel):
    round_num: int = Field(gt=0)
    claim_stance: EvidenceStance = Field(
        validation_alias=AliasChoices("claim_stance", "stance"),
        description=(
            "Direction of this observation relative to the candidate Concept claim: "
            "supports agrees with the claim, refutes contradicts it, and illustrates "
            "is relevant without directional proof"
        ),
    )
    rationale: str = Field(min_length=1, max_length=1000)
    confidence: EvidenceConfidence = "medium"

    @property
    def stance(self) -> EvidenceStance:
        """Compatibility accessor for Publisher and legacy callers."""

        return self.claim_stance


class CandidateConcept(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9._:-]+$")
    publish_action: PublishAction = "create"
    target_concept_id: Optional[str] = Field(
        default=None,
        pattern=r"^kg:(reference|method|diagnostic|experience):[a-z0-9][a-z0-9._-]*$",
    )
    change_reason: str = ""
    proposed_kind: ConceptKind
    proposed_id: Optional[str] = Field(
        default=None,
        pattern=r"^kg:(reference|method|diagnostic|experience):[a-z0-9][a-z0-9._-]*$",
    )
    claim_key: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    domains: List[str] = Field(min_length=1)
    scope: Scope
    retrieval: RetrievalMetadata = Field(default_factory=RetrievalMetadata)
    body: str = Field(min_length=1)
    relations: List[ConceptRelation] = Field(default_factory=list)
    observation_intents: List[ObservationIntent] = Field(default_factory=list)
    source_refs: List[SourceReference] = Field(default_factory=list)
    created_by: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_snapshot(cls, value):
        if isinstance(value, dict) and "base_snapshot" in value:
            value = dict(value)
            value.pop("base_snapshot", None)
        return value

    @model_validator(mode="after")
    def validate_candidate_support(self) -> "CandidateConcept":
        if self.publish_action == "update" and self.target_concept_id is None:
            raise ValueError("update candidate requires target_concept_id")
        if self.publish_action == "create" and self.target_concept_id is not None:
            raise ValueError("create candidate cannot set target_concept_id")
        if not self.observation_intents and not self.source_refs:
            raise ValueError(
                "candidate requires observation_intents or source_refs"
            )
        return self


class RuntimeScopeHints(StrictModel):
    """Agent-authored reusable tags not derivable from the Definition."""

    motifs: List[str] = Field(default_factory=list)


class CandidateDraft(StrictModel):
    """Agent-authored proposal before workspace identity is attached."""

    publish_action: PublishAction = "create"
    target_concept_id: Optional[str] = Field(
        default=None,
        pattern=r"^kg:(reference|method|diagnostic|experience):[a-z0-9][a-z0-9._-]*$",
    )
    change_reason: str = ""
    proposed_kind: ConceptKind
    proposed_id: Optional[str] = Field(
        default=None,
        pattern=r"^kg:(reference|method|diagnostic|experience):[a-z0-9][a-z0-9._-]*$",
    )
    claim_key: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    domains: List[str] = Field(min_length=1)
    scope_hints: RuntimeScopeHints = Field(default_factory=RuntimeScopeHints)
    retrieval: RetrievalMetadata = Field(default_factory=RetrievalMetadata)
    body: str = Field(min_length=1)
    relations: List[ConceptRelation] = Field(default_factory=list)
    observation_intents: List[ObservationIntent] = Field(default_factory=list)
    source_refs: List[SourceReference] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_scope(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        legacy_scope = value.pop("scope", None)
        if "scope_hints" not in value and isinstance(legacy_scope, dict):
            operator = legacy_scope.get("operator")
            if isinstance(operator, dict) and operator.get("motifs"):
                value["scope_hints"] = {
                    "motifs": list(operator["motifs"])
                }
        return value

    @model_validator(mode="after")
    def validate_draft_support(self) -> "CandidateDraft":
        if self.publish_action == "update" and self.target_concept_id is None:
            raise ValueError("update candidate requires target_concept_id")
        if self.publish_action == "create" and self.target_concept_id is not None:
            raise ValueError("create candidate cannot set target_concept_id")
        if not self.observation_intents and not self.source_refs:
            raise ValueError(
                "candidate draft requires observation_intents or source_refs"
            )
        return self


class RuntimeCandidate(CandidateDraft):
    """Workspace proposal awaiting Python-owned scope binding."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9._:-]+$")
    created_by: str = Field(min_length=1)

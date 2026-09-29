"""Strict contracts for semantic review of runtime knowledge candidates."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import Field, model_validator

from kernelgen.knowledge.models.base import (
    SCHEMA_VERSION,
    ConceptKind,
    ConceptStatus,
    EvidenceState,
    StrictModel,
)
from kernelgen.knowledge.models.candidate import CandidateConcept
from kernelgen.knowledge.models.concept import Scope
from kernelgen.knowledge.models.observation import ObservationRecord


class ReviewConceptContext(StrictModel):
    """Bounded current-Catalog context supplied to the Reviewer."""

    id: str = Field(
        pattern=(
            r"^kg:(reference|method|diagnostic|experience):"
            r"[a-z0-9][a-z0-9._-]*$"
        )
    )
    kind: ConceptKind
    claim_key: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    scope: Scope
    status: ConceptStatus
    evidence_state: EvidenceState
    evidence_refs: List[str] = Field(default_factory=list)
    source_refs: List[str] = Field(default_factory=list)
    body_excerpt: str = ""


class CandidateReviewUnit(StrictModel):
    """One exact-deduplicated candidate group and immutable evidence records."""

    review_id: str = Field(pattern=r"^review:[A-Za-z0-9._:-]+$")
    candidate_ids: List[str] = Field(min_length=1)
    candidates: List[CandidateConcept] = Field(min_length=1)
    observations: List[ObservationRecord] = Field(default_factory=list)
    nearby_concepts: List[ReviewConceptContext] = Field(default_factory=list)
    automatic_checks: List[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> "CandidateReviewUnit":
        actual = sorted(item.candidate_id for item in self.candidates)
        if self.candidate_ids != actual:
            raise ValueError("candidate_ids must exactly match sorted candidates")
        return self


class KnowledgeReviewInput(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    batch_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    units: List[CandidateReviewUnit] = Field(min_length=1)


class CandidateReviewDecision(StrictModel):
    """The only semantic decision an independent Reviewer may return."""

    review_id: str = Field(pattern=r"^review:[A-Za-z0-9._:-]+$")
    decision: Literal["approve", "reject", "defer"]
    merge_action: Literal[
        "create_new",
        "update_existing",
        "attach_evidence",
        "none",
    ]
    target_concept_id: Optional[str] = Field(
        default=None,
        pattern=(
            r"^kg:(reference|method|diagnostic|experience):"
            r"[a-z0-9][a-z0-9._-]*$"
        ),
    )
    evidence_refs: List[str] = Field(default_factory=list)
    conflict_refs: List[str] = Field(default_factory=list)
    scope_assessment: Literal[
        "compatible",
        "too_broad",
        "too_narrow",
        "conflicting",
        "unverifiable",
    ]
    reusable: bool
    confidence: Literal["low", "medium", "high"]
    reason: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "CandidateReviewDecision":
        if self.decision == "approve":
            if self.merge_action == "none":
                raise ValueError("approve requires a concrete merge_action")
            if not self.evidence_refs:
                raise ValueError("approve requires at least one evidence_ref")
            if self.scope_assessment != "compatible":
                raise ValueError("approve requires compatible scope")
            if not self.reusable:
                raise ValueError("approve requires a reusable claim")
        else:
            if self.merge_action != "none":
                raise ValueError("reject/defer must use merge_action=none")
            if self.target_concept_id is not None:
                raise ValueError("reject/defer cannot set target_concept_id")
        needs_target = self.merge_action in {
            "update_existing",
            "attach_evidence",
        }
        if needs_target != (self.target_concept_id is not None):
            raise ValueError(
                "update_existing/attach_evidence require target_concept_id; "
                "other actions forbid it"
            )
        return self


class KnowledgeReviewOutput(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    decisions: List[CandidateReviewDecision] = Field(min_length=1)


class CandidateReviewRecord(StrictModel):
    """Program-normalized, replayable decision persisted with a publish batch."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    batch_id: str = Field(min_length=1)
    review_id: str = Field(pattern=r"^review:[A-Za-z0-9._:-]+$")
    candidate_ids: List[str] = Field(min_length=1)
    decision: Literal["approve", "reject", "defer"]
    merge_action: Literal[
        "create_new",
        "update_existing",
        "attach_evidence",
        "none",
    ]
    target_concept_id: Optional[str] = None
    evidence_refs: List[str] = Field(default_factory=list)
    conflict_refs: List[str] = Field(default_factory=list)
    scope_assessment: Literal[
        "compatible",
        "too_broad",
        "too_narrow",
        "conflicting",
        "unverifiable",
    ]
    reusable: bool
    confidence: Literal["low", "medium", "high"]
    reason: str = Field(min_length=1, max_length=4000)
    reviewer_mode: Literal["off", "shadow", "enforce"] = "enforce"
    reviewer_status: Literal["completed", "fallback", "disabled"]
    review_batch_num: int = Field(default=1, ge=1)
    review_batch_count: int = Field(default=1, ge=1)
    detail_read_refs: List[str] = Field(default_factory=list)
    detail_read_event_ids: List[str] = Field(default_factory=list)
    observation_refs: List[str] = Field(default_factory=list)
    reviewed_at: datetime


__all__ = [
    "CandidateReviewDecision",
    "CandidateReviewRecord",
    "CandidateReviewUnit",
    "KnowledgeReviewInput",
    "KnowledgeReviewOutput",
    "ReviewConceptContext",
]

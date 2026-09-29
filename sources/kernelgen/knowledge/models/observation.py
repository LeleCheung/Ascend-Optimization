"""Measured observations and knowledge-application feedback."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from kernelgen.knowledge.models.base import (
    KnowledgeApplicationEffect,
    KnowledgeAssessmentStatus,
    KnowledgeUsageMode,
    KnowledgeUseDisposition,
    KnowledgeUseRole,
    StrictModel,
)


class ObservationLocator(StrictModel):
    path: str = Field(min_length=1)
    round_num: int = Field(gt=0)


class ObservationOutcome(StrictModel):
    status: str = Field(min_length=1)
    geo_mean_speedup: Optional[float] = Field(default=None, gt=0)
    workload_results_ref: str = Field(min_length=1)


class ObservationParent(StrictModel):
    round_num: int = Field(gt=0)
    status: str = Field(min_length=1)


class KnowledgeUsageScope(StrictModel):
    """Exact measured scope used for feedback ranking."""

    definition_id: str = Field(min_length=1)
    target_backend: str = Field(min_length=1)
    target_architecture: str = ""
    target_device: str = Field(min_length=1)
    software: Dict[str, str] = Field(default_factory=dict)
    workload_features: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_context(
        cls,
        operator_signature,
        target_context,
    ) -> "KnowledgeUsageScope":
        return cls(
            definition_id=operator_signature.definition_id,
            target_backend=target_context.backend,
            target_architecture=target_context.architecture,
            target_device=target_context.device,
            software={
                key: value
                for key, value in target_context.software.model_dump(
                    mode="python"
                ).items()
                if value
            },
            workload_features=operator_signature.workload_features,
        )


class SourceApplicationReference(StrictModel):
    resource: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    locator: str = Field(min_length=1)


class KnowledgeApplicationAssessment(StrictModel):
    assessment: KnowledgeAssessmentStatus
    rationale: str = Field(min_length=1, max_length=2000)


class KnowledgeApplication(StrictModel):
    """One plan declaration frozen with its measured Observation."""

    concept_ref: str = Field(
        default="",
        pattern=(
            r"^(kg:(reference|method|diagnostic|experience):"
            r"[a-z0-9][a-z0-9._-]*)?$"
        ),
    )
    source_ref: Optional[SourceApplicationReference] = None
    query_event_id: str = Field(
        default="",
        pattern=r"^((query|source-query):[A-Za-z0-9._:-]+)?$",
    )
    role: KnowledgeUseRole
    disposition: KnowledgeUseDisposition
    application_note: str = Field(default="", max_length=2000)
    affected_parts: List[str] = Field(default_factory=list)
    lineage_status: Literal[
        "complete",
        "query_missing",
        "detail_not_read",
        "legacy_unknown",
    ] = "legacy_unknown"
    agent_assessment: Optional[KnowledgeApplicationAssessment] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_concept_ref(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            ref = str(value.get("concept_ref") or "")
            concept_id, separator, revision = ref.rpartition("@")
            if separator and revision.isdigit():
                value["concept_ref"] = concept_id
        return value

    @model_validator(mode="after")
    def validate_reference(self) -> "KnowledgeApplication":
        if bool(self.concept_ref) == (self.source_ref is not None):
            raise ValueError(
                "knowledge application requires exactly one reference"
            )
        if self.disposition == "adapted" and not self.application_note:
            raise ValueError(
                "adapted knowledge application requires application_note"
            )
        if (
            self.source_ref is not None
            and self.disposition in {"adopted", "adapted"}
            and not self.application_note
        ):
            raise ValueError(
                "applied Source application requires application_note"
            )
        if any(not item for item in self.affected_parts):
            raise ValueError("affected_parts cannot contain empty values")
        if len(set(self.affected_parts)) != len(self.affected_parts):
            raise ValueError("affected_parts cannot contain duplicates")
        return self


class ObservationComparison(StrictModel):
    performance_baseline_round_num: Optional[int] = Field(
        default=None,
        gt=0,
    )
    geo_mean_delta_pct: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_baseline_name(cls, value):
        if (
            isinstance(value, dict)
            and "baseline_round_num" in value
            and "performance_baseline_round_num" not in value
        ):
            value = dict(value)
            value["performance_baseline_round_num"] = value.pop(
                "baseline_round_num"
            )
        return value


def classify_knowledge_effect(
    *,
    usage_mode: KnowledgeUsageMode,
    parent_status: Optional[str],
    current_status: str,
    performance_baseline_round_num: Optional[int],
    geo_mean_delta_pct: Optional[float],
    materiality_pct: float = 0.0,
) -> KnowledgeApplicationEffect:
    """Classify one measured round without assigning per-item causality."""

    if usage_mode == "none" or parent_status is None:
        return "unclassified"
    parent_passed = parent_status == "PASSED"
    current_passed = current_status == "PASSED"
    if not parent_passed and current_passed:
        return "correctness_recovered"
    if parent_passed and not current_passed:
        return "correctness_regressed"
    if (
        not current_passed
        or performance_baseline_round_num is None
        or geo_mean_delta_pct is None
    ):
        return "unclassified"
    if abs(geo_mean_delta_pct) <= max(0.0, materiality_pct):
        return "no_material_change"
    return (
        "performance_improved"
        if geo_mean_delta_pct > 0
        else "performance_regressed"
    )


class ObservationRecord(StrictModel):
    """A frozen, claim-independent fact materialized from one measured round."""

    schema_version: Literal["2.0"] = "2.0"
    id: str = Field(
        pattern=(
            r"^kg:observation:[A-Za-z0-9._-]+:"
            r"[A-Za-z0-9._-]+:round-[1-9][0-9]*$"
        )
    )
    run_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    definition_id: str = Field(min_length=1)
    round_num: int = Field(gt=0)
    target_context_ref: str = Field(min_length=1)
    operator_signature_ref: str = Field(min_length=1)
    ledger_locator: ObservationLocator
    solution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome: ObservationOutcome
    experiment_plan: Dict[str, Any]
    code_changes: str
    agent_conclusion: Dict[str, Any]
    evaluated_at: Optional[datetime] = None
    experiment_parent: Optional[ObservationParent] = None
    evaluation_scope: Optional[KnowledgeUsageScope] = None
    comparison: ObservationComparison = Field(
        default_factory=ObservationComparison
    )
    applications: List[KnowledgeApplication] = Field(default_factory=list)
    usage_mode: KnowledgeUsageMode = "none"
    artifact_refs: List[str] = Field(default_factory=list)
    recorded_at: datetime

    @model_validator(mode="after")
    def validate_identity(self) -> "ObservationRecord":
        if self.ledger_locator.round_num != self.round_num:
            raise ValueError("ledger_locator.round_num must match round_num")
        suffix = f":{self.workspace_id}:round-{self.round_num}"
        if not self.id.endswith(suffix):
            raise ValueError(
                "observation id must encode workspace_id and round_num"
            )
        applied = sum(
            item.disposition in {"adopted", "adapted"}
            for item in self.applications
        )
        expected_mode = (
            "none" if applied == 0 else "single" if applied == 1 else "combined"
        )
        if self.usage_mode != expected_mode:
            raise ValueError(
                f"usage_mode must be {expected_mode!r} for applications"
            )
        return self

    def knowledge_effect(
        self,
        *,
        materiality_pct: float = 0.0,
    ) -> KnowledgeApplicationEffect:
        """Classify measured effect without claiming per-item causality."""

        return classify_knowledge_effect(
            usage_mode=self.usage_mode,
            parent_status=(
                self.experiment_parent.status
                if self.experiment_parent is not None
                else None
            ),
            current_status=self.outcome.status,
            performance_baseline_round_num=(
                self.comparison.performance_baseline_round_num
            ),
            geo_mean_delta_pct=self.comparison.geo_mean_delta_pct,
            materiality_pct=materiality_pct,
        )


class KnowledgeUsageSummary(StrictModel):
    """Rebuildable same-scope usage facts for one Concept or Source ref."""

    retrieved_count: int = Field(default=0, ge=0)
    considered_count: int = Field(default=0, ge=0)
    applied_count: int = Field(default=0, ge=0)
    evaluated_count: int = Field(default=0, ge=0)
    single_success_count: int = Field(default=0, ge=0)
    combined_success_count: int = Field(default=0, ge=0)
    correctness_recovery_count: int = Field(default=0, ge=0)
    regression_count: int = Field(default=0, ge=0)
    agent_confirmed_count: int = Field(default=0, ge=0)
    agent_partially_confirmed_count: int = Field(default=0, ge=0)
    agent_not_confirmed_count: int = Field(default=0, ge=0)
    agent_inconclusive_count: int = Field(default=0, ge=0)
    last_used_at: Optional[datetime] = None
    observation_refs: List[str] = Field(default_factory=list)

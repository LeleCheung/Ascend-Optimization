"""Coder-authored experiment plan frozen before an authoritative evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


PlanKind = Literal["baseline", "performance", "correctness_fix", "diagnostic"]
EffectDirection = Literal[
    "increase",
    "decrease",
    "maintain",
    "establish_baseline",
    "fix",
    "diagnose",
]
PlanOrigin = Literal[
    "baseline",
    "analyzer",
    "epoch_direction",
    "profile_next_experiment",
    "knowledge_base",
    "coder",
]
KnowledgeUseRole = Literal[
    "constraint",
    "hypothesis",
    "implementation",
    "diagnostic",
]
KnowledgeUseDisposition = Literal["adopted", "adapted", "rejected"]


class ExpectedEffect(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    metric: str = Field(min_length=1)
    direction: EffectDirection
    mechanism: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_estimate(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            value.pop("estimated_change_pct_min", None)
            value.pop("estimated_change_pct_max", None)
        return value


class PlanSource(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    origin: PlanOrigin
    parent_round_num: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_source(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            if "origin" not in value and "kind" in value:
                value["origin"] = value.pop("kind")
            else:
                value.pop("kind", None)
            if "parent_round_num" not in value and "round_num" in value:
                value["parent_round_num"] = value.pop("round_num")
            else:
                value.pop("round_num", None)
            value.pop("analysis_path", None)
            value.pop("detail", None)
        return value


class SourceUseReference(BaseModel):
    """Exact Source fragment declared by one measured experiment."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    resource: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    locator: str = Field(min_length=1)


class KnowledgeUse(BaseModel):
    """How one Concept or Source fragment was applied to a submitted solution."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    concept_ref: str = Field(
        default="",
        pattern=(
            r"^(kg:(reference|method|diagnostic|experience):"
            r"[a-z0-9][a-z0-9._-]*)?$"
        ),
    )
    source_ref: Optional[SourceUseReference] = None
    query_event_id: str = Field(
        pattern=r"^((query|source-query):[A-Za-z0-9._:-]+)?$",
    )
    role: KnowledgeUseRole
    disposition: KnowledgeUseDisposition = Field(
        description=(
            "New submissions use adopted or adapted; rejected is accepted only "
            "when reading historical ledgers."
        ),
        json_schema_extra={"enum": ["adopted", "adapted"]},
    )
    application_note: str = Field(max_length=2000)
    affected_parts: List[str]

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
    def validate_reference_and_application(self) -> "KnowledgeUse":
        if bool(self.concept_ref) == (self.source_ref is not None):
            raise ValueError(
                "knowledge use requires exactly one of concept_ref or source_ref"
            )
        if self.disposition == "adapted" and not self.application_note:
            raise ValueError("adapted knowledge use requires application_note")
        if (
            self.source_ref is not None
            and self.disposition in {"adopted", "adapted"}
            and not self.application_note
        ):
            raise ValueError("applied Source use requires application_note")
        if any(not item for item in self.affected_parts):
            raise ValueError("affected_parts cannot contain empty values")
        if len(set(self.affected_parts)) != len(self.affected_parts):
            raise ValueError("affected_parts cannot contain duplicates")
        return self

    def reference_key(self) -> tuple[str, ...]:
        if self.concept_ref:
            return ("concept", self.concept_ref)
        source = self.source_ref
        return ("source", source.resource, source.revision, source.locator)


class ExperimentPlan(BaseModel):
    """One focused, falsifiable experiment submitted with ``eval_round``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: PlanKind
    strategy: str = Field(min_length=1)
    code_changes: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    expected_effect: ExpectedEffect
    source: PlanSource
    key_params: Dict[str, Any] = Field(default_factory=dict)
    knowledge_uses: List[KnowledgeUse] = Field(
        description=(
            "Knowledge concretely embodied in this submitted solution. "
            "Use an explicit empty list when none was applied."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_placeholders(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            value.pop("target_workloads", None)
            value.pop("risks", None)
        return value

    @model_validator(mode="after")
    def validate_baseline_semantics(self) -> "ExperimentPlan":
        if self.kind == "baseline":
            if self.source.origin != "baseline":
                raise ValueError("baseline plan requires source.origin='baseline'")
            if self.expected_effect.direction != "establish_baseline":
                raise ValueError("baseline plan requires expected_effect.direction='establish_baseline'")
        elif self.expected_effect.direction == "establish_baseline":
            raise ValueError("establish_baseline is only valid for a baseline plan")
        return self

    def validate_for_submission(self) -> "ExperimentPlan":
        """Require every new knowledge-use claim to describe applied solution code."""

        references = set()
        for use in self.knowledge_uses:
            references.add(use.reference_key())
            if use.disposition not in {"adopted", "adapted"}:
                raise ValueError(
                    "new knowledge uses must be adopted or adapted; "
                    "rejected is historical-only"
                )
            if not use.query_event_id:
                raise ValueError("applied knowledge use requires query_event_id")
            if not use.application_note:
                raise ValueError("applied knowledge use requires application_note")
            if not use.affected_parts:
                raise ValueError("applied knowledge use requires affected_parts")
        if len(references) != len(self.knowledge_uses):
            raise ValueError(
                "knowledge_uses cannot contain duplicate references"
            )
        return self

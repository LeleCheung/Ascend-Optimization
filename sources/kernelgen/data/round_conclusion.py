"""Coder-authored conclusion recorded after evaluation and required profiling."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kernelgen.data.experiment_plan import SourceUseReference


KnowledgeAssessmentStatus = Literal[
    "confirmed",
    "partially_confirmed",
    "not_confirmed",
    "inconclusive",
]


class KnowledgeAssessment(BaseModel):
    """Coder's post-evaluation assessment of one applied knowledge item."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    concept_ref: str = Field(
        default="",
        pattern=(
            r"^(kg:(reference|method|diagnostic|experience):"
            r"[a-z0-9][a-z0-9._-]*)?$"
        ),
    )
    source_ref: Optional[SourceUseReference] = None
    assessment: KnowledgeAssessmentStatus
    rationale: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_reference(self) -> "KnowledgeAssessment":
        if bool(self.concept_ref) == (self.source_ref is not None):
            raise ValueError(
                "knowledge assessment requires exactly one of "
                "concept_ref or source_ref"
            )
        return self

    def reference_key(self) -> tuple[str, ...]:
        if self.concept_ref:
            return ("concept", self.concept_ref)
        source = self.source_ref
        return ("source", source.resource, source.revision, source.locator)


class RoundConclusion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    round_num: int = Field(gt=0)
    expectation_status: Literal[
        "baseline",
        "met",
        "partially_met",
        "not_met",
        "not_evaluable",
    ]
    root_cause: str = Field(min_length=1)
    perf_gap_analysis: str = ""
    next_suggestion: str = Field(min_length=1)
    optimization_level: Literal[
        "L1_architecture",
        "L2_memory",
        "L3_parameter",
        "host_side_special",
    ]
    debug_lesson: str = ""
    strategy_evolution: str = ""
    knowledge_assessments: List[KnowledgeAssessment] = Field(
        default_factory=list,
        description=(
            "One post-evaluation Agent assessment for each applied item in "
            "the frozen ExperimentPlan. Empty when no knowledge was applied."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_placeholders(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            for key in (
                "architecture_tag",
                "suggestion_followed",
                "key_numbers",
                "composable",
                "composition_group",
                "diagnostic_results",
            ):
                value.pop(key, None)
        return value

    @model_validator(mode="after")
    def validate_unique_assessments(self) -> "RoundConclusion":
        references = [
            item.reference_key() for item in self.knowledge_assessments
        ]
        if len(set(references)) != len(references):
            raise ValueError(
                "knowledge_assessments cannot contain duplicate references"
            )
        return self

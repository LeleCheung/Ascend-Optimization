"""Small, explicitly simulated contracts for the lifecycle orchestration pilot."""

from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kernelgen.cli.models import RunMode
from kernelgen.framework.workflow import WorkflowContext, WorkflowResult, WorkflowSummary


STAGES = (
    "pytest_generate", "pytest_review", "optimize", "code_review",
    "local_ci", "pr_submit", "pr_review_fix", "pr_followup",
)


class OptimizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: RunMode = RunMode.SIMPLE_OPT


class OperatorDevelopmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage_order: ClassVar[tuple[str, ...]] = STAGES

    operator: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
    stages: tuple[str, ...] = STAGES
    optimize: OptimizeConfig | None = None
    dummy: bool = False
    resume: bool = False

    @field_validator("stages")
    @classmethod
    def selected_stages(cls, stages):
        if not stages or len(set(stages)) != len(stages):
            raise ValueError("stages must be nonempty and unique")
        unknown = set(stages) - set(cls.stage_order)
        if unknown:
            raise ValueError(f"unknown stages: {sorted(unknown)}")
        return tuple(stage for stage in cls.stage_order if stage in stages)

    @model_validator(mode="after")
    def optimization_config(self):
        if "optimize" not in self.stages:
            if self.optimize is not None:
                raise ValueError("optimize config requires the optimize stage")
        elif self.optimize is None:
            self.optimize = OptimizeConfig()
        return self

    def stage_plan(self) -> dict:
        """Canonical selection/config snapshot, excluding invocation-only flags."""
        return self.model_dump(mode="json", include={"stages", "optimize"})


@dataclass(frozen=True)
class OperatorWorkflowContext(WorkflowContext):
    """Business input adapter, not part of the generic stage contract."""

    optimize: OptimizeConfig | None = None


OperatorDevelopmentOutput = WorkflowResult[WorkflowSummary]


class DummyWorkflowCall:
    """Only synthetic receipts; never imports an Agent or invokes an external tool."""

    def __init__(self, *, fail_at=None, wait_at=None, cancel_at=None):
        for stage in (fail_at, wait_at, cancel_at):
            if stage is not None and stage not in STAGES:
                raise ValueError(f"unknown stage: {stage}")
        self.fail_at, self.wait_at, self.cancel_at = fail_at, wait_at, cancel_at

    def __call__(self, context: OperatorWorkflowContext) -> WorkflowResult:
        if context.stage == self.cancel_at:
            context.control.request_cancel("dummy cancellation during a stage")
        return WorkflowResult(
            simulated=True,
            state=("FAILED" if context.stage == self.fail_at else
                   "WAITING" if context.stage == self.wait_at else "SUCCEEDED"),
            output={
                "adapter": "TestWriterAgent" if context.stage == "pytest_generate" else context.stage,
                "called": False,
                "inputs": [str(path) for path in context.inputs],
                **({"mode": context.optimize.mode.value} if context.optimize is not None else {}),
            },
        )

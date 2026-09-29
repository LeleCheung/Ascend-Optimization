"""Validated input and output contracts for :mod:`kernelgen.workflows.optimization.kernelgen`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from kernelgen.data.catalog import DEFAULT_CATALOG_NAME
from kernelgen.data.constants import DEFAULT_CODER_COUNT, DEFAULT_EPOCH_COUNT, DEFAULT_MAX_ROUNDS
from kernelgen.data.implementation import ImplementationLanguage
from kernelgen.framework.models import DefinitionModel, EvaluationContractModel


class KernelGenOptions(BaseModel):
    """Algorithm options independent of the prepared operator Definition."""
    model_config = ConfigDict(extra="forbid")

    target_hardware: str
    implementation_language: ImplementationLanguage = (
        ImplementationLanguage.TRITON
    )
    n_parallel: int = Field(default=DEFAULT_CODER_COUNT, ge=1)
    n_epoch: int = Field(default=DEFAULT_EPOCH_COUNT, ge=1)
    cross_epoch_knowledge: bool = Field(
        default=True,
        description=("Reuse new epoch knowledge and synthesis directions; false preserves "
                     "best-code seeding and optionally reads a fixed read_only_v1 Catalog"),
    )
    start_epoch: int = 1
    start_mode: Literal["fresh", "fork", "resume"] | None = None
    reference_code_source: str = Field(
        default="",
        validation_alias=AliasChoices("reference_code_source", "reference_triton_source"),
        description="Read-only design evidence, possibly incorrect; never a seed, oracle or timing baseline",
    )
    reference_code_prompt: str = Field(
        default="",
        validation_alias=AliasChoices("reference_code_prompt", "reference_triton_prompt"),
        description="Provenance, known failures and transfer limitations of the reference source",
    )
    initial_seed_code: str = Field(
        default="",
        description=(
            "Externally validated Native baseline used to seed epoch 1; "
            "empty means a cold start or a Knowledge fork seed"
        ),
    )
    initial_seed_is_validated_baseline: bool = Field(
        default=False,
        description=(
            "Require the first epoch's first agent to evaluate and profile the "
            "initial seed unchanged before optimizing it"
        ),
    )
    knowledge_run_id: str = ""
    eval_server_url: str = ""
    evaluation_contract: EvaluationContractModel = Field(
        default_factory=EvaluationContractModel
    )
    catalog_name: str = DEFAULT_CATALOG_NAME
    evaluation_snapshot: dict[str, Any] | None = None
    warmup_ms: int = Field(default=1000, ge=0)
    benchmark_ms: int = Field(default=100, gt=0)
    num_trials: int = Field(default=1, gt=0)
    eval_timeout_seconds: int = Field(default=1500, gt=0)
    max_coder_sessions: int = Field(default=3, ge=1)
    profile_enabled: bool = True
    timeout: int = Field(default=3600, gt=0)
    early_stop_rounds: int = 3
    min_rounds: int = 2
    max_round: int = Field(default=DEFAULT_MAX_ROUNDS, ge=1)


class KernelGenInput(KernelGenOptions):
    definition: DefinitionModel = Field(
        description="kernel definition — a plain dict auto-converts"
    )

    @model_validator(mode="after")
    def validate_initial_seed(self) -> "KernelGenInput":
        if self.reference_code_prompt and not self.reference_code_source.strip():
            raise ValueError("reference_code_prompt requires reference_code_source")
        if self.initial_seed_is_validated_baseline and not self.initial_seed_code:
            raise ValueError(
                "initial_seed_is_validated_baseline requires initial_seed_code"
            )
        return self


class AgentSummary(BaseModel):
    status: str
    geo_mean: Optional[float] = None
    workspace: str = ""


class KernelGenOutput(BaseModel):
    definition_name: str
    status: str
    best_geo_mean: Optional[float] = None
    best_code: str = ""
    num_agents: int = 0
    per_agent: List[AgentSummary] = Field(default_factory=list)


@dataclass(frozen=True)
class EpochResult:
    """In-memory selection evidence; never a second persistent best record."""

    definition_name: str
    per_agent: tuple[AgentSummary, ...] = ()
    best_workspace_path: Path | None = None
    best_round: int = 0
    best_geo_mean: float | None = None
    best_code: str = ""

    @property
    def status(self) -> str:
        return "PASSED" if self.best_geo_mean is not None else "FAILED"

    def as_output(self) -> KernelGenOutput:
        return KernelGenOutput(
            definition_name=self.definition_name, status=self.status,
            best_geo_mean=self.best_geo_mean, best_code=self.best_code,
            num_agents=len(self.per_agent), per_agent=list(self.per_agent),
        )


class FailedAgent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    error_type: str
    error: str


class EpochCompletionManifest(BaseModel):
    """Durable completion boundary for full or partially successful epochs."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["complete"] = "complete"
    attempted_agents: List[str]
    successful_agents: List[str]
    failed_agents: List[FailedAgent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_agent_partition(self) -> "EpochCompletionManifest":
        attempted = self.attempted_agents
        successful = self.successful_agents
        failed = [item.name for item in self.failed_agents]
        invalid = [
            name
            for name in [*attempted, *successful, *failed]
            if not name.startswith("agent") or not name[5:].isdigit()
        ]
        if invalid:
            raise ValueError("manifest agent names must match agent<index>")
        if not attempted or len(set(attempted)) != len(attempted):
            raise ValueError("attempted_agents must be non-empty and unique")
        if not successful or len(set(successful)) != len(successful):
            raise ValueError("successful_agents must be non-empty and unique")
        if len(set(failed)) != len(failed):
            raise ValueError("failed agent names must be unique")
        if set(successful) & set(failed):
            raise ValueError("successful and failed agents must be disjoint")
        if set(successful) | set(failed) != set(attempted):
            raise ValueError(
                "successful and failed agents must partition attempted_agents"
            )
        return self


__all__ = [
    "AgentSummary",
    "EpochCompletionManifest",
    "FailedAgent",
    "KernelGenInput",
    "KernelGenOutput",
]

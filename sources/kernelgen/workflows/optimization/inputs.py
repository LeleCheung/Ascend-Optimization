"""Mode-specific options; do not reinterpret KernelGen options as SimpleOpt."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from kernelgen.knowledge.config import KnowledgeConfig
from kernelgen.workflows.optimization.kernelgen.contracts import KernelGenOptions
from kernelgen.workflows.optimization.options import SimpleOptInput


class SimpleOptimizationOptions(SimpleOptInput):
    mode: Literal["simple_opt"] = "simple_opt"
    target_hardware: str | None = Field(default=None, min_length=1)
    seed_code_path: Path | None = None


class KernelOptimizationOptions(KernelGenOptions):
    mode: Literal["kernelgen"] = "kernelgen"
    definition_name: str
    target_hardware: str | None = Field(default=None, min_length=1)
    eval_server_url: str = "http://localhost:8000"
    reference_code_path: Path | None = None
    reference_code_prompt_path: Path | None = None
    seed_code_path: Path | None = None
    knowledge_config: KnowledgeConfig | None = None
    finalize_epoch: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_sources(self):
        if self.reference_code_prompt_path and not self.reference_code_path:
            raise ValueError("reference_code_prompt_path requires reference_code_path")
        if self.reference_code_source or self.reference_code_prompt or self.initial_seed_code or self.initial_seed_is_validated_baseline:
            raise ValueError("Operator optimization accepts code paths, not a second inline code source")
        if self.evaluation_snapshot is not None:
            raise ValueError("evaluation_snapshot is prepared by OperatorOptimize")
        if self.start_epoch < 1 or self.start_epoch > self.n_epoch:
            raise ValueError("start_epoch must be between 1 and n_epoch")
        if self.start_epoch > 1 and self.start_mode not in (None, "resume"):
            raise ValueError("start_epoch > 1 requires start_mode=resume")
        if self.start_mode == "fork" and self.knowledge_config is None:
            raise ValueError("start_mode=fork requires Knowledge configuration")
        if self.finalize_epoch is not None and self.finalize_epoch > self.n_epoch:
            raise ValueError("finalize_epoch must be between 1 and n_epoch")
        return self


OptimizationOptions = Annotated[
    SimpleOptimizationOptions | KernelOptimizationOptions, Field(discriminator="mode")
]

"""Validated inputs for the single-Coder SimpleOpt workflow."""

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from kernelgen.data.catalog import DEFAULT_CATALOG_NAME
from kernelgen.data.constants import DEFAULT_MAX_ROUNDS
from kernelgen.data.implementation import ImplementationLanguage
from kernelgen.data.timeout_policy import DEFAULT_EVAL_TIMEOUT_SECONDS


class SimpleOptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    definition_name: str = Field(
        description="Definition name, e.g. 'flaggems_rsqrt'"
    )
    catalog_name: str = Field(
        default=DEFAULT_CATALOG_NAME,
        description=(
            "Server-owned built-in Catalog name; defaults to the configured "
            "canonical Catalog"
        ),
    )
    target_hardware: str = Field(
        default="Ascend910B",
        description="Target hardware, e.g. 'Ascend910B' or 'A100'",
    )
    eval_server_url: str = Field(default="http://localhost:8000")
    implementation_language: ImplementationLanguage = ImplementationLanguage.TRITON
    profile_enabled: bool = Field(
        default=False,
        description=(
            "Request optional profiling for new-best rounds and ensure one "
            "best-effort profile of the final best before distillation"
        ),
    )
    warmup_ms: int = Field(default=1000, ge=0)
    benchmark_ms: int = Field(default=100, gt=0)
    num_trials: int = Field(default=1, gt=0)
    eval_timeout_seconds: int = Field(
        default=DEFAULT_EVAL_TIMEOUT_SECONDS,
        gt=0,
    )
    early_stop_rounds: int = Field(default=3, ge=0)
    min_rounds: int = Field(default=2, ge=1)
    max_round: int = Field(default=DEFAULT_MAX_ROUNDS, ge=1)
    max_coder_sessions: int = Field(default=3, ge=1)
    destination_passing_style: Optional[bool] = Field(
        default=None,
        description="Override automatic DPS inference from the reference run signature",
    )
    reference_code_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices(
            "reference_code_path",
            "reference_code",
            "reference_triton_path",
            "reference_triton",
        ),
        description=(
            "Optional local source in any language (Triton, CUDA, Ascend C, "
            "Torch, etc.), injected as bounded, read-only design evidence"
        ),
    )
    reference_code_prompt_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices("reference_code_prompt_path", "reference_triton_prompt_path"),
        description=(
            "Optional local path to guidance describing the reference code "
            "source's hardware provenance, useful ideas, and transfer limitations"
        ),
    )
    knowledge_catalog_path: Optional[Path] = Field(
        default=None,
        description=(
            "Optional V1 Knowledge Catalog root; a value enables the "
            "Knowledge Coder"
        ),
    )

    @model_validator(mode="after")
    def validate_reference_code_inputs(self) -> "SimpleOptInput":
        if (
            self.reference_code_prompt_path is not None
            and self.reference_code_path is None
        ):
            raise ValueError(
                "reference_code_prompt_path requires reference_code_path"
            )
        return self

    @property
    def reference_code(self) -> Optional[Path]:
        """Convenience accessor for the normalized source path."""
        return self.reference_code_path

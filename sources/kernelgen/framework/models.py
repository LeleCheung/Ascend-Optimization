"""Shared input models (ADR-3 #9).

``DefinitionModel`` is the kernel definition an agent optimizes, as a nested
pydantic model. Agents' InputModels embed it (``definition: DefinitionModel``), so
a workflow passing a plain JSON dict is auto-converted (pydantic recurses) — clean,
host-testable (no flashinfer_bench / torch), and it enforces the one field that
actually matters: ``name`` is required (no silent "" fallback).

inputs/outputs stay ``Dict[str, Any]`` on purpose — agents only render shape/dtype
into the prompt text; real shape/dtype validation is the eval server's job, and a
strict per-tensor model would reject legitimate structural variants (strides, fp8
scales, ...).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class EvaluationContractModel(BaseModel):
    """Numerical correctness contract exposed to optimization agents.

    These values describe the evaluator that will judge submitted kernels. They
    are intentionally separate from the operator definition because evaluator
    configuration can override or supplement definition metadata.
    """

    tolerance_mode: Literal["", "strict", "fixed"] = ""
    atol: Optional[float] = Field(default=None, ge=0)
    rtol: Optional[float] = Field(default=None, ge=0)
    required_matched_ratio: float = Field(default=1.0, gt=0, le=1)
    check_output_dtype: bool = True
    reject_non_finite: bool = True
    consider_reduced_precision: bool = True

    @model_validator(mode="after")
    def _require_tolerance_pair(self) -> "EvaluationContractModel":
        if (self.atol is None) != (self.rtol is None):
            raise ValueError("evaluation atol and rtol must be provided together")
        has_fixed_values = self.atol is not None
        if not self.tolerance_mode and has_fixed_values:
            self.tolerance_mode = "fixed"
        if self.tolerance_mode == "fixed" and not has_fixed_values:
            raise ValueError("fixed evaluation tolerance requires atol and rtol")
        if self.tolerance_mode == "strict" and has_fixed_values:
            raise ValueError(
                "strict evaluation tolerance cannot declare fixed atol and rtol"
            )
        return self


class DefinitionModel(BaseModel):
    name: str                                    # required — the one field that must exist
    op_type: str = ""
    description: str = ""                        # human-readable description
    tags: List[str] = Field(default_factory=list)  # e.g. ["source:flaggems", "aten:gelu"]
    constraints: List[Any] = Field(default_factory=list)  # axis constraints (syntax-checked, not executed)
    axes: Dict[str, Any] = Field(default_factory=dict)    # {name: {type, value?}}
    inputs: Dict[str, Any] = Field(default_factory=dict)  # {name: {shape, dtype}}
    outputs: Dict[str, Any] = Field(default_factory=dict) # {name: {shape, dtype}}
    # Exact public ``run`` ABI when the source catalog declares parameter
    # kinds/defaults (for example ``(self, mat1, mat2, *, beta=1, alpha=1)``).
    # Legacy/native-v5 definitions leave this empty and retain the historical
    # positional-count prompt contract.
    run_signature: str = ""
    reference: str = ""                          # reference implementation source
    custom_inputs_entrypoint: Optional[str] = None  # e.g. "gen_inputs", server calls this instead of shape-based generation
    custom_valid_entrypoint: Optional[str] = None  # e.g. "valid", server delegates numeric verdict to it
    mutation: Optional[Dict[str, Any]] = None    # e.g. {"inputs": ["x"]} for in-place ops

    @property
    def dps_param_count(self) -> int:
        return len(self.inputs) + len(self.outputs)

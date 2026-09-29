"""Public wire models for the KernelGen Server v6 protocol."""

from __future__ import annotations

import inspect
import re
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from .version import ApiVersion, KERNELGEN_API_VERSION


_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SourceFile(StrictModel):
    path: str = Field(min_length=1)
    content: str

    @model_validator(mode="after")
    def validate_path(self) -> "SourceFile":
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"source path must be relative and cannot traverse parents: {self.path}"
            )
        if self.path.endswith("/"):
            raise ValueError("source path must identify a file")
        return self


class ReferenceDevice(str, Enum):
    TARGET = "target"
    CPU = "cpu"


class ParameterKind(str, Enum):
    POSITIONAL_ONLY = "positional_only"
    POSITIONAL_OR_KEYWORD = "positional_or_keyword"
    VAR_POSITIONAL = "var_positional"
    KEYWORD_ONLY = "keyword_only"


class Parameter(StrictModel):
    name: str = Field(min_length=1)
    kind: ParameterKind = ParameterKind.POSITIONAL_OR_KEYWORD
    required: bool
    default: Optional[JsonValue] = None
    type_hint: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"invalid Python parameter name: {value!r}")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "Parameter":
        has_default = "default" in self.model_fields_set
        if self.kind == ParameterKind.VAR_POSITIONAL:
            if not self.required or has_default:
                raise ValueError(
                    "var_positional parameters require required=true and no default"
                )
            return self
        if self.required and has_default:
            raise ValueError("required parameters cannot declare a default")
        if not self.required and not has_default:
            raise ValueError("optional parameters must declare a default")
        return self


class Effects(StrictModel):
    mutates: List[str] = Field(default_factory=list)
    returns_alias_of: Dict[str, str] = Field(default_factory=dict)


class Definition(StrictModel):
    """Trusted executable contract for one public operator."""

    api_version: ApiVersion = KERNELGEN_API_VERSION
    name: str = Field(min_length=1)
    description: str = ""
    parameters: List[Parameter]
    # Void operators use an empty list; returned tensors/scalars keep stable
    # names for alias checks and prompt-facing metadata.
    outputs: List[str]
    effects: Effects = Field(default_factory=Effects)
    # Native catalogs attach these trusted evaluation assets while resolving a
    # binding. Framework catalogs keep the public Definition pure.
    reference: Optional[str] = None
    reference_device: ReferenceDevice = ReferenceDevice.TARGET
    source_policy_id: Optional[str] = Field(default=None, min_length=1)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"Definition name must be a Python identifier: {value!r}")
        return value

    @field_validator("outputs")
    @classmethod
    def validate_outputs(cls, value: List[str]) -> List[str]:
        if len(value) != len(set(value)):
            raise ValueError("Definition outputs must be unique")
        if any(not _IDENTIFIER.fullmatch(item) for item in value):
            raise ValueError("Definition outputs must be Python identifiers")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "Definition":
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Definition parameter names must be unique")

        inspect.Signature(
            [
                inspect.Parameter(
                    parameter.name,
                    getattr(inspect.Parameter, parameter.kind.value.upper()),
                    default=(
                        inspect.Parameter.empty
                        if parameter.required is not False
                        else parameter.default
                    ),
                )
                for parameter in self.parameters
            ]
        )

        parameters = set(names)
        outputs = set(self.outputs)

        def check_effects(mutates: List[str], aliases: Dict[str, str]) -> None:
            unknown_mutates = set(mutates) - parameters
            unknown_outputs = set(aliases) - outputs
            unknown_targets = set(aliases.values()) - parameters
            if unknown_mutates or unknown_outputs or unknown_targets:
                raise ValueError(
                    "effects reference unknown parameters/outputs: "
                    f"mutates={sorted(unknown_mutates)}, "
                    f"outputs={sorted(unknown_outputs)}, "
                    f"targets={sorted(unknown_targets)}"
                )
            if len(mutates) != len(set(mutates)):
                raise ValueError("effects.mutates must be unique")

        check_effects(self.effects.mutates, self.effects.returns_alias_of)
        return self


class Tolerance(StrictModel):
    rtol: Optional[float] = Field(default=None, ge=0)
    atol: Optional[float] = Field(default=None, ge=0)
    atol_scale: float = Field(default=1.0, gt=0)
    required_matched_ratio: float = Field(default=1.0, gt=0, le=1)
    equal_nan: bool = False


class SourceCondition(StrictModel):
    """Conjunction of pinned source-test flags; unknown policy is an error."""

    flags: Dict[Literal["support_fp64", "support_bf16", "support_int64"], bool] = Field(min_length=1)


class Workload(StrictModel):
    """One KernelBench/FlashInfer workload and its generator context."""

    name: str = Field(min_length=1)
    inputs: Dict[str, JsonValue]
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    tolerance: Optional[Tolerance] = None
    source_condition: Optional[SourceCondition] = None
    input_path: Optional[str] = Field(default=None, min_length=1)
    output_path: Optional[str] = Field(default=None, min_length=1)

    @field_validator("input_path", "output_path")
    @classmethod
    def validate_tensor_path(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not PurePosixPath(value).is_absolute():
            raise ValueError("file-backed tensor paths must be absolute")
        return value

    @field_validator("inputs")
    @classmethod
    def validate_input_names(cls, value: Dict[str, JsonValue]) -> Dict[str, JsonValue]:
        bad = [name for name in value if not _IDENTIFIER.fullmatch(name)]
        if bad:
            raise ValueError(f"Workload input names must be Python identifiers: {bad}")
        return value

    def context(self) -> Dict[str, Any]:
        context = {
            "name": self.name,
            "inputs": self.inputs,
            "seed": self.seed,
        }
        if self.input_path is not None:
            context["input_path"] = self.input_path
        if self.output_path is not None:
            context["output_path"] = self.output_path
        return context


class Language(str, Enum):
    PYTHON = "python"
    TRITON = "triton"


class Implementation(StrictModel):
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    language: Language
    entrypoint: str = Field(pattern=r"^[^:]+::run$")
    sources: List[SourceFile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources(self) -> "Implementation":
        paths = [source.path for source in self.sources]
        if len(paths) != len(set(paths)):
            raise ValueError("source paths must be unique")
        entry_path = self.entrypoint.split("::", 1)[0]
        if entry_path not in paths:
            raise ValueError(f"entry source is missing: {entry_path}")
        return self


class EvaluationSettings(StrictModel):
    warmup_ms: int = Field(default=1000, ge=0)
    benchmark_ms: int = Field(default=100, gt=0)
    num_trials: int = Field(default=1, gt=0)
    tolerance_mode: Literal["strict", "fixed"] = "strict"
    rtol: float = Field(default=1e-2, ge=0)
    atol: float = Field(default=1e-2, ge=0)
    timeout_seconds: int = Field(default=300, gt=0)


class EvaluateRequest(StrictModel):
    """Simplified V6 workload runner for KernelBench and FlashInfer-bench."""

    api_version: ApiVersion = KERNELGEN_API_VERSION
    definition: Definition
    implementation: Implementation
    correctness_workloads: List[Workload] = Field(default_factory=list)
    timing_workloads: List[Workload] = Field(default_factory=list)
    settings: EvaluationSettings = Field(default_factory=EvaluationSettings)

    def wire_payload(self) -> Dict[str, Any]:
        """Serialize without inventing absent V6 fields as explicit nulls."""
        payload = self.model_dump(
            mode="json",
            exclude_unset=True,
            by_alias=True,
        )
        payload["api_version"] = self.api_version
        payload["definition"]["api_version"] = self.definition.api_version
        return payload

    @model_validator(mode="after")
    def validate_request(self) -> "EvaluateRequest":
        if self.definition.api_version != self.api_version:
            raise ValueError("definition.api_version must match request api_version")
        if self.implementation.definition != self.definition.name:
            raise ValueError("implementation.definition must match definition.name")
        if not self.correctness_workloads and not self.timing_workloads:
            raise ValueError("at least one correctness or timing workload is required")
        workloads = [*self.correctness_workloads, *self.timing_workloads]
        if any(w.source_condition is not None for w in workloads) and not self.definition.source_policy_id:
            raise ValueError("conditional workloads require Definition.source_policy_id")
        if self.definition.source_policy_id and self.api_version != "v6.2":
            raise ValueError("source policy requires api_version=v6.2")
        if self.api_version != "v6.2" and any(
            workload.input_path is not None or workload.output_path is not None
            for workload in workloads
        ):
            raise ValueError("file-backed workloads require api_version=v6.2")
        if any(workload.output_path is not None for workload in self.timing_workloads):
            raise ValueError(
                "timing workloads cannot use output_path; KGS must time the reference"
            )
        if any(
            workload.output_path is not None
            for workload in self.correctness_workloads
        ):
            if not self.definition.outputs:
                raise ValueError("output_path cannot represent a void operator")
            represented_mutations = set(
                self.definition.effects.returns_alias_of.values()
            )
            missing_mutations = (
                set(self.definition.effects.mutates) - represented_mutations
            )
            if missing_mutations:
                raise ValueError(
                    "output_path cannot represent mutated parameters that are not "
                    f"returned by alias: {sorted(missing_mutations)}"
                )
        if self.definition.reference_device == ReferenceDevice.CPU and self.timing_workloads:
            raise ValueError("CPU references are correctness-only and cannot be timing baselines")
        names = [
            workload.name
            for workload in self.correctness_workloads + self.timing_workloads
        ]
        if len(names) != len(set(names)):
            raise ValueError("workload names must be globally unique across phases")

        return self


class EvaluatorBinding(StrictModel):
    """Trusted catalog entry selected by the workflow, not by the Coder."""

    catalog_name: Optional[str] = Field(default=None, min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    bundle_id: Optional[str] = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    definition: str = Field(min_length=1)

    @model_validator(mode="after")
    def select_source(self) -> "EvaluatorBinding":
        if (self.catalog_name is None) == (self.bundle_id is None):
            raise ValueError("binding requires exactly one of catalog_name or bundle_id")
        return self


class InspectRequest(StrictModel):
    api_version: ApiVersion = KERNELGEN_API_VERSION
    binding: EvaluatorBinding

    def wire_payload(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_unset=True, by_alias=True)


class OperatorContractRequest(InspectRequest):
    include_test_sources: bool = False


class TestReviewSources(StrictModel):
    """Read-only source evidence, not execution or review approval."""

    framework_revision: str = ""
    files: List[SourceFile] = Field(min_length=1)


class OperatorContract(StrictModel):
    """Server-owned source contract; no target execution or readiness claim."""

    binding: EvaluatorBinding
    kind: Literal["native", "flaggems"]
    catalog_api_version: ApiVersion
    definition: Definition
    correctness_workloads: List[Workload]
    timing_workloads: List[Workload]
    test_sources: Optional[TestReviewSources] = None

    @model_validator(mode="after")
    def matching_definition(self) -> "OperatorContract":
        if self.definition.name != self.binding.definition:
            raise ValueError("operator contract definition differs from binding")
        return self


class BoundEvaluateRequest(StrictModel):
    """Evaluate one candidate through the adapter selected by its catalog."""

    api_version: ApiVersion = KERNELGEN_API_VERSION
    binding: EvaluatorBinding
    implementation: Implementation
    settings: EvaluationSettings = Field(default_factory=EvaluationSettings)

    @model_validator(mode="after")
    def validate_request(self) -> "BoundEvaluateRequest":
        if self.implementation.definition != self.binding.definition:
            raise ValueError("implementation.definition must match binding.definition")
        return self

    def wire_payload(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_unset=True, by_alias=True)


class ReferenceSettings(StrictModel):
    timeout_seconds: int = Field(default=1500, gt=0)


class ReferenceRequest(InspectRequest):
    """Execute the original Gems core benchmark baseline, without a candidate."""

    benchmark_fingerprint: str = Field(min_length=1)
    settings: ReferenceSettings = Field(default_factory=ReferenceSettings)


class ReferenceResult(StrictModel):
    status: Literal["PASSED", "FAILED", "UNSUPPORTED", "ALL_SKIP", "NO_CASES",
                    "RUNTIME_ERROR", "TIMEOUT", "SUSPECTED_DEVICE_ERROR"]
    scope: Literal["benchmark_core"] = "benchmark_core"
    benchmark_fingerprint: str | None = None
    report: Dict[str, Any] = Field(default_factory=dict)
    log: str = ""


class AdapterCase(StrictModel):
    case_id: str = Field(min_length=1)
    phase: Literal["timing"] = "timing"
    ordinal: int = Field(ge=0)
    profile_eligible: bool = True
    dtype: Optional[str] = None
    shape: Optional[JsonValue] = None
    params: Dict[str, JsonValue] = Field(default_factory=dict)


class CaseList(StrictModel):
    schema_version: Literal["kernelgen.case-list/v1"] = "kernelgen.case-list/v1"
    adapter_kind: Literal["native", "flaggems"]
    operator: str = Field(min_length=1)
    benchmark_fingerprint: str = Field(min_length=1)
    cases: List[AdapterCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_case_ids(self) -> "CaseList":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id values must be unique")
        return self


class CandidateContract(StrictModel):
    entrypoint: Literal["run"] = "run"
    signature: str = Field(min_length=1)


class AdapterCapabilities(StrictModel):
    preflight: bool = True
    profile: bool = False


class AdapterManifest(StrictModel):
    kind: Literal["native", "flaggems"]
    adapter_version: str = Field(default="1", min_length=1)
    benchmark_fingerprint: str = Field(min_length=1)
    candidate_contract: CandidateContract
    capabilities: AdapterCapabilities
    case_list: CaseList


class PreflightResult(StrictModel):
    api_version: ApiVersion = KERNELGEN_API_VERSION
    status: Literal[
        "PASSED", "FAILED", "RUNTIME_ERROR", "TIMEOUT", "SUSPECTED_DEVICE_ERROR"
    ]
    stage: str = Field(min_length=1)
    log: str = ""
    is_hack: bool = False
    hack_reason: str = ""
    benchmark_fingerprint: Optional[str] = None
    num_cases: int = Field(default=0, ge=0)


class WorkloadStatus(str, Enum):
    PASSED = "PASSED"
    SKIP = "SKIP"
    INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
    RUNTIME_ERROR = "RUNTIME_ERROR"


class CorrectnessResult(StrictModel):
    status: WorkloadStatus
    max_absolute_error: Optional[float] = 0.0
    max_relative_error: Optional[float] = 0.0
    matched_ratio: float = 1.0
    message: str = ""
    metrics: Dict[str, JsonValue] = Field(default_factory=dict)


class TimingResult(StrictModel):
    status: WorkloadStatus
    latency_ms: Optional[float] = None
    reference_latency_ms: Optional[float] = None
    speedup: Optional[float] = None
    message: str = ""


class EvaluationStatus(str, Enum):
    PASSED = "PASSED"
    ALL_SKIP = "ALL_SKIP"
    PARTIAL_PASS = "PARTIAL_PASS"
    INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TIMEOUT = "TIMEOUT"
    SUSPECTED_DEVICE_ERROR = "SUSPECTED_DEVICE_ERROR"


class WorkloadResult(StrictModel):
    uuid: str = Field(min_length=1)
    axes: Dict[str, JsonValue] = Field(default_factory=dict)
    phase: Literal["correctness", "timing"]
    status: WorkloadStatus
    speedup: Optional[float] = None
    latency_ms: Optional[float] = None
    reference_latency_ms: Optional[float] = None
    abs_err: Optional[float] = None
    rel_err: Optional[float] = None
    matched_ratio: Optional[float] = None
    metrics: Dict[str, JsonValue] = Field(default_factory=dict)
    log: str = ""


class EvaluateResponse(StrictModel):
    """Authoritative flat result consumed directly by optimization clients."""

    api_version: ApiVersion = KERNELGEN_API_VERSION
    reference_source: Literal["primary", "torch_fallback"] = "primary"
    status: EvaluationStatus
    device: str
    server_backend: str
    is_hack: bool = False
    hack_reason: str = ""
    geo_mean: Optional[float] = None
    min_speedup: Optional[float] = None
    worst_workload_uuid: Optional[str] = None
    latency_ms: Optional[float] = None
    abs_err: Optional[float] = None
    rel_err: Optional[float] = None
    num_workloads: int = Field(ge=0)
    num_passed: int = Field(ge=0)
    timing_skipped: bool = False
    log: str = ""
    per_workload: List[WorkloadResult] = Field(default_factory=list)

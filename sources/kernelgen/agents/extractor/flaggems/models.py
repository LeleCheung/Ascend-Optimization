"""Legacy translated FlagGems Definition/Workload models.

New V6 FlagGems integrations use :mod:`adapter`; these models remain available
only to read and audit earlier extraction artifacts.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from kernelgen.agents.extractor.flaggems.abi_validation import (
    FRAMEWORK_TYPES,
    finite_json,
    fixed_float_casts,
    parse_reference,
    validate_gen_inputs_randomness,
    validate_hook_context_access,
    validate_hook_signature,
    validate_run_abi,
)
from kernelgen.agents.extractor.flaggems.workload_validation import (
    parse_and_bind_call,
    tokens,
    validate_effects,
    validate_input_type,
)


V6_GROUPS = {
    "attention",
    "backward",
    "conv",
    "convolution",
    "gemm",
    "indexing",
    "interpolate",
    "linalg",
    "norm",
    "pointwise",
    "pooling",
    "reduction",
    "rnn",
    "scatter",
    "shape",
}

# Compatibility alias for callers that imported the old catalog grouping name.
V5_GROUPS = V6_GROUPS

AbiType = Literal[
    "Tensor",
    "Optional[Tensor]",
    "List[Tensor]",
    "Optional[List[Tensor]]",
    "List[Optional[Tensor]]",
    "int",
    "float",
    "bool",
    "str",
    "Scalar",
    "Optional[int]",
    "Optional[float]",
    "Optional[bool]",
    "List[int]",
    "List[float]",
    "List[bool]",
    "Tuple[int,int]",
    "Tuple[bool,bool,bool]",
    "Tuple[int,...]",
    "Tuple[float,...]",
    "Tuple[bool,...]",
    "Tuple[Tensor,...]",
    "Tuple[Optional[Tensor],...]",
    "dtype",
    "device",
    "layout",
    "memory_format",
    "Generator",
    "Optional[dtype]",
    "Optional[device]",
    "Optional[layout]",
    "Optional[memory_format]",
    "Optional[Generator]",
]

ParameterKind = Literal[
    "positional_only",
    "positional_or_keyword",
    "var_positional",
    "keyword_only",
    "var_keyword",
]

_KIND_ORDER = {
    "positional_only": 0,
    "positional_or_keyword": 1,
    "var_positional": 2,
    "keyword_only": 3,
    "var_keyword": 4,
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class FlagGemsExtractorInput(StrictModel):
    operator: str
    flaggems_repo: str = "third_party/FlagGems"
    default_params: Optional[Dict[str, Any]] = None
    case_list_path: str | None = None


class Parameter(StrictModel):
    name: str = Field(min_length=1)
    type: AbiType | None = None
    kind: ParameterKind
    required: bool | None = None
    default: JsonValue | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_kind_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        kind = data.get("kind")
        has_type = "type" in data
        has_required = "required" in data
        has_default = "default" in data
        if kind in {"var_positional", "var_keyword"}:
            if has_required or has_default:
                raise ValueError(f"{kind} parameters must omit required/default")
            if kind == "var_keyword" and has_type:
                raise ValueError("var_keyword type is omitted in v6")
        else:
            if not has_type:
                raise ValueError("ordinary parameters require an ABI type")
            if not has_required:
                raise ValueError("ordinary parameters require required=true/false")
            if data.get("required") is True and has_default:
                raise ValueError("required parameters must omit default")
            if data.get("required") is False and not has_default:
                raise ValueError("optional parameters must include default")
        return data

    @model_validator(mode="after")
    def validate_parameter_type(self) -> Parameter:
        if not self.name.isidentifier():
            raise ValueError(f"invalid Python parameter name {self.name!r}")
        if self.kind == "var_positional" and self.type is not None:
            if not self.type.startswith("Tuple["):
                raise ValueError("typed var_positional parameters require Tuple[...] type")
        if self.required is False:
            finite_json(self.default, f"parameter {self.name}.default")
            if self.default is None and self.type is not None:
                if not self.type.startswith("Optional["):
                    raise ValueError(
                        f"parameter {self.name}: default null requires Optional[...] type"
                    )
        return self


class EffectPredicate(StrictModel):
    is_value: JsonValue | None = Field(default=None, alias="is")
    is_not_value: JsonValue | None = Field(default=None, alias="is_not")

    @model_validator(mode="before")
    @classmethod
    def validate_operator(cls, data: Any) -> Any:
        if not isinstance(data, dict) or len(data) != 1:
            raise ValueError("effect predicate requires exactly one of is/is_not")
        if set(data) not in ({"is"}, {"is_not"}):
            raise ValueError("effect predicate supports only is/is_not")
        finite_json(next(iter(data.values())), "effect predicate")
        return data


class EffectCase(StrictModel):
    when: Dict[str, EffectPredicate]
    mutates: List[str] = Field(default_factory=list)
    returns_alias_of: Dict[str, str] = Field(default_factory=dict)


class Effects(StrictModel):
    mutates: List[str] = Field(default_factory=list)
    returns_alias_of: Dict[str, str] = Field(default_factory=dict)
    cases: List[EffectCase] = Field(default_factory=list)


class WorkloadInput(StrictModel):
    type: Literal["random", "custom", "scalar", "literal"]
    shape: JsonValue | None = None
    dtype: str | None = None
    value: JsonValue | None = None
    generator_params: Dict[str, JsonValue] | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_kind_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        kind = data.get("type")
        fields = set(data)
        if kind == "random":
            if data.get("shape") is None or not data.get("dtype"):
                raise ValueError("random inputs require shape and dtype")
            if fields & {"value", "generator_params"}:
                raise ValueError("random inputs cannot contain value/generator_params")
        elif kind == "custom":
            if "generator_params" not in data:
                raise ValueError("custom inputs require generator_params")
            if "value" in data:
                raise ValueError("custom inputs cannot contain value")
        elif kind in {"scalar", "literal"}:
            if "value" not in data:
                raise ValueError(f"{kind} inputs require value")
            if fields & {"shape", "dtype", "generator_params"}:
                raise ValueError(
                    f"{kind} inputs cannot contain shape/dtype/generator_params"
                )
            if kind == "scalar" and not (
                isinstance(data.get("value"), (int, float, bool))
                and data.get("value") is not None
            ):
                raise ValueError("scalar value must be int, float, or bool")
        return data

    @model_validator(mode="after")
    def validate_json_values(self) -> WorkloadInput:
        if self.type in {"scalar", "literal"}:
            finite_json(self.value)
        if self.generator_params is not None:
            finite_json(self.generator_params, "generator_params")
        if self.type == "random":
            if not isinstance(self.shape, list) or not all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension >= 0
                for dimension in self.shape
            ):
                raise ValueError("random shape must be a list of non-negative integers")
        if self.dtype is not None and (
            not self.dtype or self.dtype.startswith("torch.")
        ):
            raise ValueError("dtype must be a non-empty normalized token")
        return self


class Tolerance(StrictModel):
    rtol: float | None = Field(default=None, ge=0)
    atol: float | None = Field(default=None, ge=0)
    atol_scale: float = Field(default=1.0, gt=0)
    required_matched_ratio: float = Field(default=1.0, gt=0, le=1)
    equal_nan: bool = False


class ExpectedException(StrictModel):
    raises: str = Field(min_length=1)
    message_regex: str | None = None


class Workload(StrictModel):
    name: str = Field(min_length=1)
    inputs: Dict[str, WorkloadInput]
    call: str = Field(min_length=1)
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    tolerance: Tolerance | None = None
    expect: ExpectedException | None = None

    @model_validator(mode="after")
    def validate_input_names(self) -> Workload:
        invalid = sorted(name for name in self.inputs if not name.isidentifier())
        if invalid:
            raise ValueError(f"invalid Workload input names {invalid}")
        return self


class Definition(StrictModel):
    api_version: Literal["v6.0"] = "v6.0"
    name: str = Field(min_length=1)
    description: str = ""
    parameters: List[Parameter]
    outputs: List[str] = Field(min_length=1)
    effects: Effects
    reference: str = Field(min_length=1)
    correctness_reference: str | None = None
    reference_device: Literal["target", "cpu"] = "target"
    custom_valid_entrypoint: Literal["valid"] | None = None

    @model_validator(mode="after")
    def validate_definition_shape(self) -> Definition:
        if not self.name.isidentifier() or self.name in {"run", "gen_inputs", "valid"}:
            raise ValueError(f"invalid public callable name {self.name!r}")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicate parameter names")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError(f"{self.name}: duplicate output names")
        if any(not output.isidentifier() for output in self.outputs):
            raise ValueError(f"{self.name}: outputs must be Python identifiers")

        previous = -1
        seen_var_positional = False
        seen_var_keyword = False
        positional_default_seen = False
        for parameter in self.parameters:
            order = _KIND_ORDER[parameter.kind]
            if order < previous:
                raise ValueError(f"{self.name}: parameters are not in Python ABI order")
            previous = order
            if parameter.kind == "var_positional":
                if seen_var_positional:
                    raise ValueError(f"{self.name}: multiple var_positional parameters")
                seen_var_positional = True
            elif parameter.kind == "var_keyword":
                if seen_var_keyword:
                    raise ValueError(f"{self.name}: multiple var_keyword parameters")
                seen_var_keyword = True
            elif parameter.kind in {"positional_only", "positional_or_keyword"}:
                if parameter.required is False:
                    positional_default_seen = True
                elif positional_default_seen:
                    raise ValueError(
                        f"{self.name}: required positional parameter follows a default"
                    )
        validate_effects(self)
        return self


class ExtractorResult(StrictModel):
    record_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    group: str
    definition: Definition
    correctness_workloads: List[Workload] = Field(default_factory=list)
    timing_workloads: List[Workload] = Field(default_factory=list)

    @property
    def workloads(self) -> List[Workload]:
        return self.correctness_workloads + self.timing_workloads

    @property
    def catalog_id(self) -> str:
        return self.record_id or self.definition.name


class FlagGemsExtractorOutput(StrictModel):
    results: List[ExtractorResult]

    @model_validator(mode="after")
    def validate_v6_protocol(self) -> FlagGemsExtractorOutput:
        catalog_ids: set[str] = set()
        workload_names: set[str] = set()
        for result in self.results:
            definition = result.definition
            if result.group not in V6_GROUPS:
                raise ValueError(
                    f"{definition.name}: unsupported v6 group {result.group!r}; "
                    f"expected one of {sorted(V6_GROUPS)}"
                )
            if definition.name.startswith("flaggems_"):
                raise ValueError(
                    f"{definition.name}: v6 name must be the real public callable, "
                    "without the legacy flaggems_ prefix"
                )
            if result.catalog_id in catalog_ids:
                raise ValueError(f"duplicate catalog record_id {result.catalog_id!r}")
            catalog_ids.add(result.catalog_id)

            module, functions = parse_reference(
                definition.name,
                definition.reference,
                "reference",
            )
            run = functions.get("run")
            if run is None:
                raise ValueError(f"{definition.name}: reference must define run()")
            validate_run_abi(
                definition.name,
                definition.parameters,
                run,
                "reference",
            )
            if functions.get(definition.name) is not None:
                raise ValueError(
                    f"{definition.name}: reference must export only internal run(); "
                    "the Server binds the public symbol"
                )
            for node in module.body:
                targets: list[ast.expr] = []
                value: ast.expr | None = None
                if isinstance(node, ast.Assign):
                    targets = node.targets
                    value = node.value
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                    value = node.value
                if any(
                    isinstance(target, ast.Name) and target.id == definition.name
                    for target in targets
                ) and not (isinstance(value, ast.Name) and value.id == "run"):
                    raise ValueError(
                        f"{definition.name}: existing public symbol must be the same "
                        "object as run"
                    )

            if definition.correctness_reference is not None:
                _, correctness_functions = parse_reference(
                    definition.name,
                    definition.correctness_reference,
                    "correctness_reference",
                )
                correctness_run = correctness_functions.get("run")
                if correctness_run is None:
                    raise ValueError(
                        f"{definition.name}: correctness_reference must define run()"
                    )
                validate_run_abi(
                    definition.name,
                    definition.parameters,
                    correctness_run,
                    "correctness_reference",
                )

            gen_inputs = functions.get("gen_inputs")
            if gen_inputs is not None:
                validate_hook_signature(
                    definition.name,
                    gen_inputs,
                    ("ctx", "device"),
                )
                validate_hook_context_access(definition.name, gen_inputs)
                validate_gen_inputs_randomness(definition.name, gen_inputs)
            if definition.custom_valid_entrypoint is not None:
                valid = functions.get("valid")
                if valid is None:
                    raise ValueError(f"{definition.name}: reference must define valid()")
                validate_hook_signature(
                    definition.name,
                    valid,
                    ("ref_outputs", "sol_outputs", "inputs", "ctx"),
                )
                validate_hook_context_access(definition.name, valid)
            elif "valid" in functions:
                raise ValueError(
                    f"{definition.name}: valid() requires custom_valid_entrypoint='valid'"
                )

            fixed_casts = fixed_float_casts(run)
            if result.timing_workloads and fixed_casts:
                raise ValueError(
                    f"{definition.name}: benchmark baseline run() contains fixed "
                    f"floating-point conversions {sorted(fixed_casts)}; keep run() "
                    "at benchmark dtype and use correctness_reference for pytest-only "
                    "conversions"
                )
            if definition.reference_device == "cpu" and result.timing_workloads:
                raise ValueError(
                    f"{definition.name}: CPU references are correctness-only"
                )
            if not result.workloads:
                raise ValueError(f"{definition.name}: no workloads")

            requires_gen_inputs = False
            parameter_map = {
                parameter.name: parameter for parameter in definition.parameters
            }
            for workload in result.workloads:
                if workload.name in workload_names:
                    raise ValueError(f"duplicate workload name {workload.name!r}")
                workload_names.add(workload.name)
                bound = parse_and_bind_call(definition, workload)
                if workload.expect is not None and workload in result.timing_workloads:
                    raise ValueError(
                        f"{definition.name}/{workload.name}: expected exceptions are "
                        "correctness-only"
                    )
                if any(spec.type == "custom" for spec in workload.inputs.values()):
                    requires_gen_inputs = True
                for parameter_name, value in bound.arguments.items():
                    parameter = parameter_map[parameter_name]
                    if parameter.type is None:
                        continue
                    for token in tokens(value):
                        spec = workload.inputs[token.name]
                        validate_input_type(
                            f"{definition.name}/{workload.name}/{token.name}",
                            parameter.type,
                            spec,
                        )
                        if (
                            spec.type == "custom"
                            or parameter.type.startswith("Tuple[")
                            or (
                                parameter.type in FRAMEWORK_TYPES
                                and not (spec.type == "literal" and spec.value is None)
                            )
                        ):
                            requires_gen_inputs = True
            if requires_gen_inputs and gen_inputs is None:
                raise ValueError(
                    f"{definition.name}: custom/framework/Tuple inputs require fixed "
                    "gen_inputs(ctx, device)"
                )
        return self

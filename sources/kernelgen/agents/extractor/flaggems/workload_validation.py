"""Static validation helpers for V6 calls, inputs, and effects."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kernelgen.agents.extractor.flaggems.abi_validation import FRAMEWORK_TYPES

if TYPE_CHECKING:
    from kernelgen.agents.extractor.flaggems.models import (
        Definition,
        Parameter,
        Workload,
        WorkloadInput,
    )


_INSPECT_KINDS = {
    "positional_only": inspect.Parameter.POSITIONAL_ONLY,
    "positional_or_keyword": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "var_positional": inspect.Parameter.VAR_POSITIONAL,
    "keyword_only": inspect.Parameter.KEYWORD_ONLY,
    "var_keyword": inspect.Parameter.VAR_KEYWORD,
}


@dataclass(frozen=True)
class InputToken:
    name: str


def inspect_signature(parameters: list[Parameter]) -> inspect.Signature:
    result = []
    for parameter in parameters:
        default: Any = inspect.Parameter.empty
        if parameter.required is False:
            default = parameter.default
        result.append(
            inspect.Parameter(
                parameter.name,
                _INSPECT_KINDS[parameter.kind],
                default=default,
            )
        )
    return inspect.Signature(result)


def parse_and_bind_call(
    definition: Definition,
    workload: Workload,
) -> inspect.BoundArguments:
    prefix = f"{definition.name}/{workload.name}"
    try:
        expression = ast.parse(workload.call, mode="eval").body
    except SyntaxError as error:
        raise ValueError(f"{prefix}: invalid call expression: {error}") from error
    if not isinstance(expression, ast.Call):
        raise ValueError(f"{prefix}: call must be one Python call expression")
    if not isinstance(expression.func, ast.Name) or expression.func.id != definition.name:
        raise ValueError(
            f"{prefix}: call must invoke public symbol {definition.name!r}"
        )

    positional: list[InputToken] = []
    referenced: list[str] = []
    starred_names: set[str] = set()
    for argument in expression.args:
        if isinstance(argument, ast.Name):
            name = argument.id
            positional.append(InputToken(name))
        elif isinstance(argument, ast.Starred) and isinstance(argument.value, ast.Name):
            name = argument.value.id
            starred_names.add(name)
            # One token is enough to verify that a *args slot exists. Runtime
            # binding validates the concrete tuple cardinality.
            positional.append(InputToken(name))
        else:
            raise ValueError(
                f"{prefix}: positional arguments must be input names or *input"
            )
        referenced.append(name)

    keywords: dict[str, InputToken] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError(f"{prefix}: dynamic **kwargs expansion is not supported")
        if not isinstance(keyword.value, ast.Name):
            raise ValueError(f"{prefix}: keyword values must be Workload input names")
        if keyword.arg in keywords:
            raise ValueError(f"{prefix}: duplicate keyword {keyword.arg!r}")
        name = keyword.value.id
        keywords[keyword.arg] = InputToken(name)
        referenced.append(name)

    input_names = set(workload.inputs)
    unknown = set(referenced) - input_names
    unused = input_names - set(referenced)
    if unknown or unused:
        raise ValueError(
            f"{prefix}: call/input mismatch: unknown={sorted(unknown)}, "
            f"unused={sorted(unused)}"
        )

    var_positional = next(
        (
            parameter
            for parameter in definition.parameters
            if parameter.kind == "var_positional"
        ),
        None,
    )
    if starred_names and var_positional is None:
        raise ValueError(f"{prefix}: call uses *input but ABI has no *args parameter")
    if (
        var_positional is not None
        and var_positional.name in input_names
        and var_positional.name not in starred_names
    ):
        raise ValueError(
            f"{prefix}: var-positional input {var_positional.name!r} must be expanded"
        )

    try:
        return inspect_signature(definition.parameters).bind(
            *positional,
            **keywords,
        )
    except TypeError as error:
        raise ValueError(
            f"{prefix}: call does not bind to Definition.parameters: {error}"
        ) from error


def tokens(value: Any) -> list[InputToken]:
    if isinstance(value, InputToken):
        return [value]
    if isinstance(value, (tuple, list)):
        return [token for item in value for token in tokens(item)]
    if isinstance(value, dict):
        return [token for item in value.values() for token in tokens(item)]
    return []


def validate_input_type(
    prefix: str,
    parameter_type: str,
    spec: WorkloadInput,
) -> None:
    optional = parameter_type.startswith("Optional[")
    if optional and spec.type == "literal" and spec.value is None:
        return
    if parameter_type in {"Tensor", "Optional[Tensor]"}:
        allowed = {"random", "custom"}
    elif parameter_type in {
        "List[Tensor]",
        "Optional[List[Tensor]]",
        "List[Optional[Tensor]]",
        "Tuple[Tensor,...]",
        "Tuple[Optional[Tensor],...]",
    }:
        allowed = {"custom"}
    elif parameter_type in {"Generator", "Optional[Generator]"}:
        allowed = {"custom"}
    elif parameter_type in FRAMEWORK_TYPES:
        allowed = {"literal", "custom"}
    else:
        allowed = {"scalar", "literal", "custom"}
    if spec.type not in allowed:
        raise ValueError(
            f"{prefix}: input type {spec.type!r} cannot materialize ABI "
            f"type {parameter_type}"
        )
    if spec.type == "custom":
        return

    value = spec.value
    scalar_type = parameter_type
    if scalar_type.startswith("Optional["):
        scalar_type = scalar_type[len("Optional[") : -1]
    valid = True
    if scalar_type == "int":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif scalar_type == "float":
        valid = isinstance(value, float)
    elif scalar_type == "bool":
        valid = isinstance(value, bool)
    elif scalar_type == "str":
        valid = isinstance(value, str)
    elif scalar_type == "Scalar":
        valid = isinstance(value, (int, float, bool))
    elif scalar_type in {"List[int]", "Tuple[int,int]", "Tuple[int,...]"}:
        valid = isinstance(value, list) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
        if scalar_type == "Tuple[int,int]":
            valid = valid and len(value) == 2
    elif scalar_type in {"List[float]", "Tuple[float,...]"}:
        valid = isinstance(value, list) and all(
            isinstance(item, float) for item in value
        )
    elif scalar_type in {
        "List[bool]",
        "Tuple[bool,bool,bool]",
        "Tuple[bool,...]",
    }:
        valid = isinstance(value, list) and all(
            isinstance(item, bool) for item in value
        )
        if scalar_type == "Tuple[bool,bool,bool]":
            valid = valid and len(value) == 3
    elif scalar_type in {"dtype", "device", "layout", "memory_format"}:
        valid = isinstance(value, str) and bool(value)
    if not valid:
        raise ValueError(
            f"{prefix}: JSON value {value!r} does not match ABI type "
            f"{parameter_type}"
        )


def validate_effects(definition: Definition) -> None:
    parameters = {parameter.name for parameter in definition.parameters}
    outputs = set(definition.outputs)

    def validate_one(
        label: str,
        mutates: list[str],
        aliases: dict[str, str],
    ) -> None:
        if len(mutates) != len(set(mutates)):
            raise ValueError(f"{definition.name}: duplicate effects mutation in {label}")
        unknown_mutations = set(mutates) - parameters
        if unknown_mutations:
            raise ValueError(
                f"{definition.name}: {label} mutates unknown parameters "
                f"{sorted(unknown_mutations)}"
            )
        unknown_outputs = set(aliases) - outputs
        unknown_targets = set(aliases.values()) - parameters
        if unknown_outputs or unknown_targets:
            raise ValueError(
                f"{definition.name}: {label} alias mapping has unknown outputs="
                f"{sorted(unknown_outputs)} or parameters={sorted(unknown_targets)}"
            )

    validate_one(
        "effects",
        definition.effects.mutates,
        definition.effects.returns_alias_of,
    )
    for index, case in enumerate(definition.effects.cases):
        unknown_conditions = set(case.when) - parameters
        if unknown_conditions:
            raise ValueError(
                f"{definition.name}: effects.cases[{index}] references unknown "
                f"parameters {sorted(unknown_conditions)}"
            )
        validate_one(
            f"effects.cases[{index}]",
            case.mutates,
            case.returns_alias_of,
        )

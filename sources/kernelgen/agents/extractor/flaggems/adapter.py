"""Extract the common, evaluator-neutral V6 Definition from FlagGems source."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .source_inventory import _operator_entry, collect_source_inventory


ParameterKind = Literal[
    "positional_only",
    "positional_or_keyword",
    "var_positional",
    "keyword_only",
]


class DefinitionParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: ParameterKind = "positional_or_keyword"
    required: bool
    default: Any = None
    type_hint: str | None = None

    @model_validator(mode="after")
    def validate_default(self) -> "DefinitionParameter":
        has_default = "default" in self.model_fields_set
        if self.kind == "var_positional":
            if not self.required or has_default:
                raise ValueError(
                    "var_positional parameters require required=true and no default"
                )
        elif self.required == has_default:
            raise ValueError(
                "required parameters omit default; optional parameters include it"
            )
        return self


class DefinitionEffects(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mutates: list[str] = Field(default_factory=list)
    returns_alias_of: dict[str, str] = Field(default_factory=dict)


class FlagGemsDefinitionSpec(BaseModel):
    """The same public Definition schema consumed by every V6 evaluator."""

    model_config = ConfigDict(extra="forbid")

    api_version: Literal["v6.0"] = "v6.0"
    name: str
    description: str = ""
    parameters: list[DefinitionParameter]
    outputs: list[str] = Field(default_factory=lambda: ["out"])
    effects: DefinitionEffects = Field(default_factory=DefinitionEffects)


# These implementations intentionally expose ``*args/**kwargs`` wrappers or do
# not have a directly exported Python function.  Their real public contracts
# are fixed by the matching ATen schema and native pytest calls in this pinned
# FlagGems revision.  Keeping the exceptions explicit is safer than inferring an
# ABI from one benchmark input tuple.
_CANONICAL_SIGNATURES = {
    "digamma_": "def digamma_(self: Tensor): pass",
    "dunder_ior_scalar": (
        "def dunder_ior_scalar(self: Tensor, other: Scalar): pass"
    ),
    "dunder_ior_tensor": (
        "def dunder_ior_tensor(self: Tensor, other: Tensor): pass"
    ),
    "einsum": (
        "def einsum(equation: str, tensors: List[Tensor], *, "
        "path: Optional[List[int]] = None): pass"
    ),
    "functional_sym_constrain_range_for_size": (
        "def functional_sym_constrain_range_for_size("
        "size: Scalar, min: Optional[int], max: Optional[int], "
        "dep_token: Tensor): pass"
    ),
    "functional_sym_constrain_range": (
        "def functional_sym_constrain_range("
        "size: Scalar, min: Optional[int], max: Optional[int], "
        "dep_token: Tensor): pass"
    ),
    "i0_": "def i0_(self: Tensor): pass",
    "ilshift": "def ilshift(self: Tensor, other: Tensor): pass",
    "ilshift__": "def ilshift__(self: Tensor, other: Tensor): pass",
    "linalg_ldl_factor": (
        "def linalg_ldl_factor(self: Tensor, *, hermitian: bool = False): pass"
    ),
    "linalg_ldl_factor_ex": (
        "def linalg_ldl_factor_ex(self: Tensor, *, hermitian: bool = False, "
        "check_errors: bool = False): pass"
    ),
    "log_softmax_backward_data": (
        "def log_softmax_backward_data(grad_output: Tensor, output: Tensor, "
        "dim: int, input_dtype: dtype): pass"
    ),
    "logit_": "def logit_(self: Tensor, eps: Optional[float] = None): pass",
    "mvlgamma_": "def mvlgamma_(self: Tensor, p: int): pass",
    "selu_": "def selu_(self: Tensor): pass",
    "zero": "def zero(self: Tensor): pass",
}

_OUTPUTS = {
    "amp_foreach_non_finite_check_and_unscale_": [],
    "cudnn_batch_norm_backward": ["grad_input", "grad_weight", "grad_bias"],
    "flash_attention_forward": ["output", "logsumexp"],
    "fused_adam_": [],
    "kthvalue": ["values", "indices"],
    "linear_backward": ["grad_input", "grad_weight", "grad_bias"],
    "linalg_ldl_factor": ["LD", "pivots"],
    "linalg_ldl_factor_ex": ["LD", "pivots", "info"],
    "linalg_slogdet": ["sign", "logabsdet"],
    # The public result is Tensor[] and its length equals input.ndim.  Treat the
    # list as one PyTree output so 2-D and 3-D workloads share one Definition.
    "nonzero_numpy": ["out"],
    "rnn_relu": ["output", "hidden"],
    "rrelu_with_noise_functional": ["output", "noise"],
    # The public forward ABI is a nine-tuple, even though the benchmark's
    # framework baseline only returns the primary attention tensor.  Keep the
    # source pytest's auxiliary-output assertions expressible in Native valid().
    "scaled_dot_product_cudnn_attention": [
        "output", "logsumexp", "cum_seq_q", "cum_seq_k", "max_q", "max_k",
        "philox_seed", "philox_offset", "debug_attn_mask",
    ],
    "scaled_dot_product_cudnn_attention_backward": [
        "grad_query",
        "grad_key",
        "grad_value",
    ],
    "thnn_fused_lstm_cell": ["hy", "cy", "workspace"],
    "thnn_fused_lstm_cell_backward_impl": [
        "grad_input_gates",
        "grad_hidden_gates",
        "grad_cx",
        "grad_input_bias",
        "grad_hidden_bias",
    ],
}

_MULTI_MUTATES = {
    "amp_foreach_non_finite_check_and_unscale_": ["tensors", "found_inf"],
    "dunder_ior_scalar": ["self"],
    "dunder_ior_tensor": ["self"],
    "fused_adam_": [
        "self",
        "grads",
        "exp_avgs",
        "exp_avg_sqs",
        "max_exp_avg_sqs",
    ],
    # The public catalog id omits the dunder spelling used by the underlying
    # ATen overload, so the ordinary trailing-underscore rule cannot detect it.
    "ilshift": ["self"],
}

_VOID_OPERATORS = {
    "amp_foreach_non_finite_check_and_unscale_",
    "fused_adam_",
}


def _signature_source(inventory: Any, operator: str) -> str:
    override = _CANONICAL_SIGNATURES.get(operator)
    if override is not None:
        return override
    signatures = list(inventory.public_signatures)
    if len(signatures) != 1:
        raise ValueError(
            f"{operator}: expected one deterministic public signature, got "
            f"{len(signatures)}"
        )
    function = ast.parse(signatures[0].source).body[0]
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError(f"{operator}: public signature is not a function")
    if function.args.kwarg is not None:
        raise ValueError(
            f"{operator}: unresolved **kwargs public ABI requires a canonical signature"
        )
    return signatures[0].source


def _type_hint(argument: ast.arg) -> str | None:
    return ast.unparse(argument.annotation) if argument.annotation is not None else None


def _parameter(
    argument: ast.arg,
    *,
    kind: ParameterKind,
    default: ast.expr | None,
) -> DefinitionParameter:
    fields: dict[str, Any] = {
        "name": argument.arg,
        "kind": kind,
        "required": default is None,
    }
    hint = _type_hint(argument)
    if hint:
        fields["type_hint"] = hint
    if default is not None:
        fields["default"] = _default_value(default)
    return DefinitionParameter(**fields)


def _default_value(default: ast.expr) -> Any:
    """Normalize trusted framework constants into JSON-compatible tokens."""

    if (
        isinstance(default, ast.Attribute)
        and isinstance(default.value, ast.Name)
        and default.value.id == "torch"
    ):
        return default.attr
    return ast.literal_eval(default)


def _parameters(signature_source: str) -> list[DefinitionParameter]:
    function = ast.parse(signature_source).body[0]
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("public signature is not a function")
    arguments = function.args
    if arguments.kwarg is not None:
        raise ValueError("V6 Definition does not accept unresolved **kwargs")

    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(arguments.defaults)
    ) + list(arguments.defaults)
    result = [
        _parameter(
            argument,
            kind=(
                "positional_only"
                if index < len(arguments.posonlyargs)
                else "positional_or_keyword"
            ),
            default=default,
        )
        for index, (argument, default) in enumerate(zip(positional, defaults))
    ]
    if arguments.vararg is not None:
        fields: dict[str, Any] = {
            "name": arguments.vararg.arg,
            "kind": "var_positional",
            "required": True,
        }
        hint = _type_hint(arguments.vararg)
        if hint:
            fields["type_hint"] = hint
        result.append(DefinitionParameter(**fields))
    result.extend(
        _parameter(argument, kind="keyword_only", default=default)
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
    )
    return result


def _description(flaggems_repo: str, operator: str) -> str:
    entry = _operator_entry(flaggems_repo, operator)
    if not entry:
        return ""
    description = entry.get("description")
    return str(description).strip() if description else ""


def _effects(
    operator: str,
    parameters: list[DefinitionParameter],
    outputs: list[str],
) -> DefinitionEffects:
    names = {parameter.name for parameter in parameters}
    mutates = [name for name in _MULTI_MUTATES.get(operator, []) if name in names]
    if not mutates and operator.endswith("_") and parameters:
        mutates = [parameters[0].name]
    aliases = (
        {outputs[0]: mutates[0]}
        if len(outputs) == 1 and mutates and operator not in _VOID_OPERATORS
        else {}
    )
    return DefinitionEffects(mutates=mutates, returns_alias_of=aliases)


def extract_flaggems_definition(
    flaggems_repo: str | Path,
    operator: str,
) -> FlagGemsDefinitionSpec:
    """Extract one pure V6 Definition; pytest assets remain Server-owned."""

    repo = Path(flaggems_repo).resolve()
    inventory = collect_source_inventory(str(repo), operator)
    parameters = _parameters(_signature_source(inventory, operator))
    outputs = list(_OUTPUTS.get(operator, ["out"]))
    return FlagGemsDefinitionSpec(
        name=operator,
        description=_description(str(repo), operator),
        parameters=parameters,
        outputs=outputs,
        effects=_effects(operator, parameters, outputs),
    )


# Temporary import compatibility for the POC name.  It returns the common
# Definition now; no adapter registry record is generated.
extract_flaggems_adapter_spec = extract_flaggems_definition
FlagGemsAdapterSpec = FlagGemsDefinitionSpec
AdapterParameter = DefinitionParameter


__all__ = [
    "AdapterParameter",
    "DefinitionEffects",
    "DefinitionParameter",
    "FlagGemsAdapterSpec",
    "FlagGemsDefinitionSpec",
    "extract_flaggems_adapter_spec",
    "extract_flaggems_definition",
]

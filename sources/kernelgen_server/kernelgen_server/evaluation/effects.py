"""V6 mutation and return-alias correctness gates."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from ..protocol.schema import Definition, EvaluationSettings, Workload
from .compare import Comparison, compare, compare_structure


@dataclass(frozen=True)
class ResolvedEffects:
    mutates: tuple[str, ...]
    returns_alias_of: dict[str, str]


def resolve_effects(
    definition: Definition,
    bound: inspect.BoundArguments,
) -> ResolvedEffects:
    del bound
    return ResolvedEffects(
        tuple(definition.effects.mutates),
        dict(definition.effects.returns_alias_of),
    )


def named_outputs(definition: Definition, output: Any) -> dict[str, Any]:
    if not definition.outputs:
        if output is None or (
            isinstance(output, (tuple, list)) and len(output) == 0
        ):
            return {}
        raise TypeError(
            "operator declares no outputs but returned "
            f"{type(output).__name__}"
        )
    if len(definition.outputs) == 1:
        return {definition.outputs[0]: output}
    if not isinstance(output, (tuple, list)) or len(output) != len(definition.outputs):
        raise TypeError(
            f"operator declares {len(definition.outputs)} outputs but returned "
            f"{type(output).__name__}"
        )
    return dict(zip(definition.outputs, output, strict=True))


def tensors_alias(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    try:
        return bool(torch._C._is_alias_of(left, right))
    except (AttributeError, RuntimeError):
        try:
            return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        except RuntimeError:
            return False


def check_aliases(
    aliases: dict[str, str],
    outputs: dict[str, Any],
    bound: inspect.BoundArguments,
    label: str,
) -> str:
    for output_name, parameter_name in aliases.items():
        if not tensors_alias(outputs[output_name], bound.arguments[parameter_name]):
            return (
                f"{label} output {output_name!r} does not alias "
                f"parameter {parameter_name!r}"
            )
    return ""


def _merge(comparisons: list[tuple[str, Comparison]]) -> Comparison:
    for label, comparison in comparisons:
        if not comparison.passed:
            message = f"{label}: {comparison.message or 'values differ'}"
            return Comparison(
                False,
                comparison.max_absolute_error,
                comparison.max_relative_error,
                comparison.matched_ratio,
                message,
            )
    absolute = [
        value.max_absolute_error
        for _, value in comparisons
        if value.max_absolute_error is not None
    ]
    relative = [
        value.max_relative_error
        for _, value in comparisons
        if value.max_relative_error is not None
    ]
    return Comparison(
        True,
        max(absolute, default=0.0),
        max(relative, default=0.0),
        min((value.matched_ratio for _, value in comparisons), default=1.0),
        "",
    )


def compare_outputs_and_mutations(
    definition: Definition,
    effects: ResolvedEffects,
    reference_output: Any,
    candidate_output: Any,
    reference_before: dict[str, Any],
    candidate_before: dict[str, Any],
    reference_bound: inspect.BoundArguments,
    candidate_bound: inspect.BoundArguments,
    settings: EvaluationSettings,
    workload: Workload,
    *,
    compare_declared_values: bool = True,
    compare_return_contract: bool = True,
) -> Comparison:
    comparisons = []
    if compare_return_contract:
        comparisons.append(
            (
                "return value",
                (
                    compare(reference_output, candidate_output, settings, workload)
                    if compare_declared_values
                    else compare_structure(reference_output, candidate_output)
                ),
            )
        )
    declared = set(effects.mutates)
    for name in effects.mutates:
        comparisons.append(
            (
                f"mutated input {name!r}",
                (
                    compare(
                        reference_bound.arguments[name],
                        candidate_bound.arguments[name],
                        settings,
                        workload,
                    )
                    if compare_declared_values
                    else compare_structure(
                        reference_bound.arguments[name],
                        candidate_bound.arguments[name],
                    )
                ),
            )
        )
    for name in reference_before:
        if name in declared:
            continue
        comparisons.extend(
            [
                (
                    f"reference unexpectedly mutated input {name!r}",
                    compare(
                        reference_before[name],
                        reference_bound.arguments[name],
                        settings,
                        workload,
                    ),
                ),
                (
                    f"candidate unexpectedly mutated input {name!r}",
                    compare(
                        candidate_before[name],
                        candidate_bound.arguments[name],
                        settings,
                        workload,
                    ),
                ),
            ]
        )
    return _merge(comparisons)


__all__ = [
    "ResolvedEffects",
    "check_aliases",
    "compare_outputs_and_mutations",
    "named_outputs",
    "resolve_effects",
    "tensors_alias",
]

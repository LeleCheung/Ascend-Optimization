"""Default PyTree correctness comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from .pytree import leaves, structure, to_cpu
from ..protocol.schema import EvaluationSettings, Workload


_STRICT_RTOL = {
    "torch.bool": 0.0,
    "torch.uint8": 0.0,
    "torch.int8": 0.0,
    "torch.int16": 0.0,
    "torch.int32": 0.0,
    "torch.int64": 0.0,
    "torch.float16": 1e-3,
    "torch.bfloat16": 0.016,
    "torch.float32": 1.3e-6,
    "torch.float64": 1e-7,
    "torch.complex64": 1.3e-6,
    "torch.complex128": 1e-7,
}

_STRICT_ATOL = {
    "torch.bool": 0.0,
    "torch.uint8": 0.0,
    "torch.int8": 0.0,
    "torch.int16": 0.0,
    "torch.int32": 0.0,
    "torch.int64": 0.0,
    "torch.float16": 1e-4,
    "torch.bfloat16": 1e-4,
    "torch.float32": 1e-4,
    "torch.float64": 1e-4,
    "torch.complex64": 1e-4,
    "torch.complex128": 1e-4,
}


@dataclass
class Comparison:
    passed: bool
    max_absolute_error: Optional[float] = 0.0
    max_relative_error: Optional[float] = 0.0
    matched_ratio: float = 1.0
    message: str = ""


def compare_structure(reference: Any, candidate: Any) -> Comparison:
    """Compare PyTree shape and leaf contracts without comparing leaf values."""

    reference = to_cpu(reference)
    candidate = to_cpu(candidate)
    if structure(reference) != structure(candidate):
        return Comparison(False, message="PyTree structures differ")

    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None

    for ref_leaf, cand_leaf in zip(leaves(reference), leaves(candidate), strict=True):
        ref = ref_leaf.value
        cand = cand_leaf.value
        both_tensors = (
            torch is not None
            and isinstance(ref, torch.Tensor)
            and isinstance(cand, torch.Tensor)
        )
        if not both_tensors and type(ref) is not type(cand):
            return Comparison(False, message=f"leaf type differs at {ref_leaf.path}")
        if both_tensors:
            if ref.shape != cand.shape:
                return Comparison(False, message=f"tensor shape differs at {ref_leaf.path}")
            if ref.dtype != cand.dtype:
                return Comparison(False, message=f"tensor dtype differs at {ref_leaf.path}")

    return Comparison(True)


def compare(
    reference: Any,
    candidate: Any,
    settings: EvaluationSettings,
    workload: Workload,
) -> Comparison:
    reference = to_cpu(reference)
    candidate = to_cpu(candidate)
    if structure(reference) != structure(candidate):
        return Comparison(False, message="PyTree structures differ")

    tolerance = workload.tolerance
    required_ratio = tolerance.required_matched_ratio if tolerance else 1.0
    atol_scale = tolerance.atol_scale if tolerance else 1.0
    equal_nan = tolerance.equal_nan if tolerance else False

    max_abs = 0.0
    max_rel = 0.0
    matched = 0
    total = 0
    error_is_undefined = False

    for ref_leaf, cand_leaf in zip(leaves(reference), leaves(candidate), strict=True):
        ref = ref_leaf.value
        cand = cand_leaf.value
        if type(ref) is not type(cand):
            try:
                import torch

                both_tensors = isinstance(ref, torch.Tensor) and isinstance(cand, torch.Tensor)
            except ImportError:  # pragma: no cover
                both_tensors = False
            if not both_tensors:
                return Comparison(False, message=f"leaf type differs at {ref_leaf.path}")

        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None

        if torch is not None and isinstance(ref, torch.Tensor):
            if ref.shape != cand.shape:
                return Comparison(False, message=f"tensor shape differs at {ref_leaf.path}")
            if ref.dtype != cand.dtype:
                return Comparison(False, message=f"tensor dtype differs at {ref_leaf.path}")
            if ref.is_floating_point() or ref.is_complex():
                dtype_name = str(ref.dtype)
                default_rtol = (
                    _STRICT_RTOL.get(dtype_name, settings.rtol)
                    if settings.tolerance_mode == "strict"
                    else settings.rtol
                )
                default_atol = (
                    _STRICT_ATOL.get(dtype_name, settings.atol)
                    if settings.tolerance_mode == "strict"
                    else settings.atol
                )
                rtol = (
                    tolerance.rtol
                    if tolerance and tolerance.rtol is not None
                    else default_rtol
                )
                atol = (
                    tolerance.atol
                    if tolerance and tolerance.atol is not None
                    else default_atol
                ) * atol_scale
                ref_value = ref.to(torch.complex64 if ref.is_complex() else torch.float32)
                cand_value = cand.to(torch.complex64 if cand.is_complex() else torch.float32)
                both_finite = torch.isfinite(ref_value) & torch.isfinite(cand_value)
                same_infinity = (
                    torch.isinf(ref_value)
                    & torch.isinf(cand_value)
                    & torch.eq(ref_value, cand_value)
                )
                same_nan = (
                    torch.isnan(ref_value) & torch.isnan(cand_value)
                    if equal_nan
                    else torch.zeros_like(both_finite)
                )
                if not (both_finite | same_infinity | same_nan).all().item():
                    return Comparison(
                        False,
                        max_absolute_error=None,
                        max_relative_error=None,
                        matched_ratio=0.0,
                        message=f"non-finite tensor mismatch at {ref_leaf.path}",
                    )
                raw_abs_error = (cand_value - ref_value).abs()
                abs_error = torch.where(
                    both_finite,
                    raw_abs_error,
                    torch.zeros_like(raw_abs_error),
                )
                raw_denominator = ref_value.abs()
                denominator = torch.where(
                    both_finite,
                    raw_denominator,
                    torch.zeros_like(raw_denominator),
                )
                relative = abs_error / (denominator + 1e-8)
                ok = same_infinity | same_nan | (
                    both_finite & (abs_error <= atol + rtol * denominator)
                )
                count = ref.numel()
                total += count
                matched += int(ok.sum().item())
                if count:
                    max_abs = max(max_abs, float(abs_error.max().item()))
                    max_rel = max(max_rel, float(relative.max().item()))
            else:
                count = ref.numel()
                total += count
                equal = torch.eq(ref, cand)
                matched += int(equal.sum().item())
                if not bool(equal.all().item()):
                    # Integer and boolean mismatches do not have a meaningful
                    # floating-point error. JSON cannot encode infinity, so
                    # leave the error fields undefined instead.
                    error_is_undefined = True
            continue

        total += 1
        if isinstance(ref, float) and isinstance(cand, float):
            rtol = tolerance.rtol if tolerance and tolerance.rtol is not None else settings.rtol
            atol = (
                tolerance.atol if tolerance and tolerance.atol is not None else settings.atol
            ) * atol_scale
            both_finite = math.isfinite(ref) and math.isfinite(cand)
            same_infinity = math.isinf(ref) and cand == ref
            same_nan = equal_nan and math.isnan(ref) and math.isnan(cand)
            if not (both_finite or same_infinity or same_nan):
                return Comparison(
                    False,
                    max_absolute_error=None,
                    max_relative_error=None,
                    matched_ratio=0.0,
                    message=f"non-finite scalar mismatch at {ref_leaf.path}",
                )
            error = abs(cand - ref) if both_finite else 0.0
            relative = error / (abs(ref) + 1e-8) if both_finite else 0.0
            ok = same_infinity or same_nan or error <= atol + rtol * abs(ref)
            matched += int(ok)
            max_abs = max(max_abs, error)
            max_rel = max(max_rel, relative)
        else:
            matched += int(ref == cand)

    ratio = 1.0 if total == 0 else matched / total
    return Comparison(
        ratio >= required_ratio,
        max_absolute_error=None if error_is_undefined else max_abs,
        max_relative_error=None if error_is_undefined else max_rel,
        matched_ratio=ratio,
        message="" if ratio >= required_ratio else "values differ",
    )

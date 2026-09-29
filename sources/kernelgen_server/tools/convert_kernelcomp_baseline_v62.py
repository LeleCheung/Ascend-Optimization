#!/usr/bin/env python3
"""Convert the kernel-comp baseline problem set to a native v6.2 catalog.

The source is the kernel-competition baseline repository
(``kernel-comp-baseline``): for each of its ~212 problems under
``problems/<group>/<problem>/`` it ships ``spec.md`` (signature, dtype
constraints, tolerance), ``reference_torch.py`` (the pure-torch correctness
ground truth), and ``cases.py`` (shared CORRECTNESS_CASES/BENCH_CASES grids).

This converter maps each problem to a v6.2 operator package:

- ``definition.json`` — public ABI from the ``reference`` signature, type
  hints from the ``spec.md`` signature line, output names from the spec
  return signature.
- ``oracle.py`` — ``REFERENCE_DEVICE = 'target'`` plus ``run()`` with the
  reference body (cross-imports inlined).  When the source cases need
  non-declarative inputs (zero buffers, randperm index tensors, literal
  index arrays, last-dim slice views) a ``gen_inputs`` hook is emitted that
  rebuilds them from a small build-DSL embedded in the workload.  When the
  source defines a custom ``_check``, it is embedded verbatim (with its
  helper functions and the harness ``assert_close``) behind a thin ``valid``
  wrapper.
- ``correctness_full.jsonl`` / ``timing_full.jsonl`` — the source case grids
  expressed with declarative v6.2 recipes (``random``/``scalar``) or the
  custom build-DSL, with the per-dtype tolerance contract from
  ``harness.correctness`` written explicitly per workload.  Case parameters
  are extracted by executing ``cases.py`` against a torch stub that records
  tensor-construction calls, so no CUDA device is needed at conversion time.

Operators are named ``sglang_<problem>`` and grouped by their source group.
The complete source grids are retained; active files are capped by the shared
sampler (``tools/sample_v62_catalog_workloads.py``).
"""

from __future__ import annotations

import argparse
import ast
import functools
import importlib.util
import inspect
import json
import re
import shutil
import sys
import tempfile
import textwrap
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# The repo's ``tools/`` directory is a namespace package, but a stray
# ``tools`` package inside an installed nvfuser egg shadows it on sys.path.
# Load the shared sampler by file path so this tool works in any environment.
_SAMPLER_PATH = Path(__file__).resolve().parent / "sample_v62_catalog_workloads.py"
_SAMPLER_SPEC = importlib.util.spec_from_file_location(
    "sample_v62_catalog_workloads", _SAMPLER_PATH
)
_SAMPLER = importlib.util.module_from_spec(_SAMPLER_SPEC)
assert _SAMPLER_SPEC.loader is not None
sys.modules.setdefault("sample_v62_catalog_workloads", _SAMPLER)
_SAMPLER_SPEC.loader.exec_module(_SAMPLER)
ALGORITHM = _SAMPLER.ALGORITHM
sample_catalog = _SAMPLER.sample_catalog

CATALOG_NAME = "kernelcomp-baseline"
TARGET_API_VERSION = "v6.2"
WORKLOAD_LIMIT = 200

SOURCE_REVISION = "4e3d7199b143072a0ef21fc095392247a24fca84"

# Problems excluded from conversion with the reason (mirrors the source
# README's known-broken list).
SKIPPED_PROBLEMS: dict[tuple[str, str], str] = {
    (
        "quantization",
        "mxfp8_block_scaled_matmul",
    ): "upstream SGLang removed mxfp8_block_scaled_matmul_triton; the source "
    "correctness test fails at collection (see kernel-comp-baseline README).",
    (
        "gemm",
        "grouped_gemm",
    ): "case construction calls sglang's compute_grouped_gemm_metadata; the "
    "oracle would need sglang at runtime (not installed in the target env).",
    (
        "utility",
        "gpu_tensor_hash",
    ): "reference delegates to the sglang gpu_tensor_hash kernel for "
    "bit-exactness; sglang is not importable in the target env.",
    (
        "utility",
        "murmur_hash32",
    ): "reference delegates to the sglang murmur_hash32 kernel; sglang is not "
    "importable in the target env.",
}

# Contract fixes for problems whose v6.2 expression differs from the raw
# source (found by the ref-as-solution self-check):
#   force_cal_sum: pin the cal_sum case kwarg so every workload returns the
#       declared 3-tuple (the source grid mixes cal_sum True/False, which a
#       fixed-output Definition cannot express).
#   outputs: override the Definition output names/arity.
#   wrap_single_return: wrap the oracle run so a single-value return becomes
#       (value, None) — normalizes a conditionally-2-output operator to the
#       declared 2 outputs.
PROBLEM_FIXES: dict[tuple[str, str], dict[str, Any]] = {
    ("quantization", "per_token_quant_int8_sum"): {"force_cal_sum": True},
    ("diffusion", "cat_pad_channels_last_3d"): {
        "outputs": ["conv_input", "cache"],
        "wrap_single_return": True,
    },
    # These dequantize ops store fp8 cache bytes that legitimately decode to
    # fp8 NaN patterns; the mutation gate compares the unmodified input
    # against itself, so NaN equality must be allowed.
    ("dsa", "dequantize_k_cache_fast"): {"equal_nan": True},
    ("dsa", "dequantize_k_cache_paged"): {"equal_nan": True},
    ("dsa", "gather_dequant_requant_fp8_paged"): {"equal_nan": True},
}


def _apply_fix_rows(rows: list[dict[str, Any]], fix: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply per-workload contract fixes (cal_sum pinning, NaN tolerance)."""
    if fix.get("force_cal_sum"):
        for row in rows:
            cal = row["inputs"].get("cal_sum")
            if isinstance(cal, dict) and cal.get("type") == "scalar":
                cal["value"] = True
    if fix.get("equal_nan"):
        for row in rows:
            tolerance = row.get("tolerance")
            if tolerance is None:
                # Embedded-constructor workloads may lack an inferred
                # tolerance; give them the harness default so the NaN
                # allowance actually applies.
                tolerance = {
                    "rtol": 1e-2,
                    "atol": 1e-2,
                    "atol_scale": 1.0,
                    "required_matched_ratio": 1.0,
                }
                row["tolerance"] = tolerance
            tolerance["equal_nan"] = True
    return rows


# ---------------------------------------------------------------------------
# Torch stub: extract case parameters without a CUDA device.
# ---------------------------------------------------------------------------


class _GeneratorStub:
    def __init__(self) -> None:
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> "_GeneratorStub":
        self.seed = int(seed)
        return self


class DType(str):
    """A torch dtype carried through case extraction.  Subclasses str so
    equality with plain dtype-name strings still works; case parameters that
    are dtypes (e.g. ``out_dtype=torch.float16``) are materialized back to
    real torch.dtype objects at runtime via the gen_inputs build-DSL."""

    __slots__ = ()


class DerivedScalar(int):
    """A Python scalar derived from a case tensor at runtime (e.g.
    ``int(seq_lens.sum())``).  Subclasses int so it can flow through case
    logic, but is materialized through the build-DSL so the dependency on the
    tensor's runtime value is preserved."""

    def __new__(cls, operation: str, operand: "TensorStub") -> "DerivedScalar":
        instance = super().__new__(cls, 0)
        instance.operation = operation
        instance.operand = operand
        return instance


class TensorStub:
    """Records how one case tensor was constructed."""

    def __init__(
        self,
        kind: str,
        *,
        shape: Sequence[int] | None = None,
        dtype: str | None = None,
        seed: int | None = None,
        **params: Any,
    ) -> None:
        self.spec: dict[str, Any] = {"kind": kind}
        if shape is not None:
            self.spec["shape"] = [int(dim) for dim in shape]
        if dtype is not None:
            self.spec["dtype"] = dtype
        if seed is not None:
            self.spec["seed"] = seed
        self.spec.update(params)

    @staticmethod
    def _dtype_name(dtype: Any) -> str:
        """Normalize a dtype to the bare name used by v6.2 recipes
        (``float16``, not ``torch.float16``)."""
        name = str(dtype)
        return name[len("torch.") :] if name.startswith("torch.") else name

    # --- value-carrying transforms ---
    def __getitem__(self, item: Any) -> "TensorStub":
        if isinstance(item, TensorStub):
            return TensorStub("mask_index", source=self, mask=item)
        return TensorStub("slice", source=self, index=_serialize_index(item))

    def __setitem__(self, key: Any, value: Any) -> None:
        self.spec.setdefault("writes", []).append(
            {"index": _serialize_index(key), "value": value}
        )

    def __iter__(self):
        if self.spec.get("kind") != "tensor_literal":
            raise TypeError("only literal case tensors are iterable")
        return iter(self.spec["values"])

    def cumsum(self, dim: int = 0) -> "TensorStub":
        values = _constant_values(self)
        if values is None:
            raise TypeError(
                "cumsum requires a constant-valued case tensor (literal/full)"
            )
        total = 0
        result: list[int] = []
        for value in values:
            total += int(value)
            result.append(total)
        return TensorStub(
            "tensor_literal", values=result, dtype=_dtype_of(self) or "int64"
        )

    def item(self) -> Any:
        if self.spec.get("kind") != "tensor_literal" or len(self.spec.get("values", [])) != 1:
            raise TypeError("only scalar literal case tensors support item()")
        return self.spec["values"][0]

    def tolist(self) -> Any:
        if self.spec.get("kind") != "tensor_literal":
            raise TypeError("only literal case tensors support tolist()")
        return list(self.spec["values"])

    def split(self, split_size: Any, dim: int = 0) -> tuple["TensorStub", ...]:
        ndim = len(_shape_of(self))
        dim = dim if dim >= 0 else dim + ndim
        sizes = split_size if isinstance(split_size, (list, tuple)) else None
        if sizes is None:
            size = int(split_size)
            sizes = _chunk_sizes(_shape_of(self)[dim], size)
        result: list[TensorStub] = []
        start = 0
        for size in sizes:
            index: list[Any] = [{"start": None, "stop": None, "step": None}] * ndim
            index[dim] = {"start": start, "stop": start + int(size), "step": None}
            result.append(TensorStub("slice", source=self, index=index))
            start += int(size)
        return tuple(result)

    def expand(self, *shape: Any) -> "TensorStub":
        return TensorStub("expand", source=self, shape=[int(d) for d in shape])

    def expand_as(self, other: Any) -> "TensorStub":
        return TensorStub("expand", source=self, shape=_shape_of(other))

    def permute(self, *dims: Any) -> "TensorStub":
        return TensorStub("permute", source=self, dims=[int(d) for d in dims])

    def flip(self, dims: Any) -> "TensorStub":
        return TensorStub("flip", source=self, dims=[int(d) for d in dims])

    def roll(self, shifts: Any, dims: Any = None) -> "TensorStub":
        return TensorStub("roll", source=self, shifts=shifts, dims=dims)

    def unsqueeze(self, dim: int) -> "TensorStub":
        return TensorStub("unsqueeze", source=self, dim=int(dim))

    def squeeze(self, dim: Any = None) -> "TensorStub":
        return TensorStub(
            "squeeze", source=self, dim=None if dim is None else int(dim)
        )

    def narrow(self, dim: int, start: int, length: int) -> "TensorStub":
        return TensorStub("narrow", source=self, dim=int(dim), start=int(start), length=int(length))

    def repeat(self, *sizes: Any) -> "TensorStub":
        return TensorStub("repeat", source=self, sizes=[int(s) for s in sizes])

    def tile(self, *sizes: Any) -> "TensorStub":
        return TensorStub("repeat", source=self, sizes=[int(s) for s in sizes])

    def masked_fill(self, mask: "TensorStub", value: Any) -> "TensorStub":
        return TensorStub("masked_fill", source=self, mask=mask, value=value)

    def logical_not(self) -> "TensorStub":
        return TensorStub("logical_not", source=self)

    def _comparison(self, kind: str, other: Any) -> "TensorStub":
        if isinstance(other, TensorStub):
            return TensorStub(kind, source=self, rhs=other)
        return TensorStub(kind, source=self, value=other)

    def __gt__(self, other: Any) -> "TensorStub":
        return self._comparison("compare_gt", other)

    def __ge__(self, other: Any) -> "TensorStub":
        return self._comparison("compare_ge", other)

    def __lt__(self, other: Any) -> "TensorStub":
        return self._comparison("compare_lt", other)

    def __le__(self, other: Any) -> "TensorStub":
        return self._comparison("compare_le", other)

    def __eq__(self, other: Any) -> "TensorStub":  # type: ignore[override]
        return self._comparison("compare_eq", other)

    def __ne__(self, other: Any) -> "TensorStub":  # type: ignore[override]
        return self._comparison("compare_ne", other)

    def __and__(self, other: Any) -> "TensorStub":
        if isinstance(other, TensorStub):
            return TensorStub("logical_and", source=self, rhs=other)
        return TensorStub("logical_and", source=self, value=other)

    def __or__(self, other: Any) -> "TensorStub":
        if isinstance(other, TensorStub):
            return TensorStub("logical_or", source=self, rhs=other)
        return TensorStub("logical_or", source=self, value=other)

    def __invert__(self) -> "TensorStub":
        return TensorStub("logical_not", source=self)

    def to(self, dtype: Any, *args: Any, **kwargs: Any) -> "TensorStub":
        return TensorStub("to", source=self, dtype=self._dtype_name(dtype))

    def float(self) -> "TensorStub":
        return self.to("float32")

    def long(self) -> "TensorStub":
        return self.to("int64")

    def int(self) -> "TensorStub":
        return self.to("int32")

    def half(self) -> "TensorStub":
        return self.to("float16")

    def bfloat16(self) -> "TensorStub":
        return self.to("bfloat16")

    def double(self) -> "TensorStub":
        return self.to("float64")

    def bool(self) -> "TensorStub":
        return self.to("bool")

    def reshape(self, *shape: Any) -> "TensorStub":
        return TensorStub("reshape", source=self, shape=[int(d) for d in shape])

    def view(self, *shape: Any) -> "TensorStub":
        return TensorStub("reshape", source=self, shape=[int(d) for d in shape])

    def t(self) -> "TensorStub":
        return TensorStub("transpose", source=self)

    def transpose(self, *dims: Any) -> "TensorStub":
        return TensorStub("transpose", source=self, dims=[int(d) for d in dims])

    def clone(self) -> "TensorStub":
        return self  # values are identical; only storage identity differs

    def detach(self) -> "TensorStub":
        return self

    def contiguous(self) -> "TensorStub":
        return self

    def __mul__(self, other: Any) -> "TensorStub":
        return TensorStub("scale", source=self, value=other)

    def __rmul__(self, other: Any) -> "TensorStub":
        return TensorStub("scale", source=self, value=other)

    def __add__(self, other: Any) -> "TensorStub":
        return TensorStub("shift", source=self, value=other)

    def __radd__(self, other: Any) -> "TensorStub":
        return TensorStub("shift", source=self, value=other)

    def __sub__(self, other: Any) -> "TensorStub":
        return TensorStub("rshift", source=self, value=other)

    def __rsub__(self, other: Any) -> "TensorStub":
        return TensorStub("shift", source=self, value=other)

    def __truediv__(self, other: Any) -> "TensorStub":
        return TensorStub("divide", source=self, value=other)

    def __rtruediv__(self, other: Any) -> "TensorStub":
        return TensorStub("rdivide", source=self, value=other)

    def abs(self) -> "TensorStub":
        return TensorStub("abs", source=self)

    def clamp(self, min: Any = None, max: Any = None) -> "TensorStub":
        return TensorStub(
            "clamp", source=self, min=min, max=max, dtype=self.spec.get("dtype", "float32")
        )

    def max(self, dim: Any = None, keepdim: bool = False) -> "TensorStub":
        return TensorStub(
            "max_reduce", source=self, dim=dim, keepdim=keepdim,
            dtype=self.spec.get("dtype", "float32"),
        )

    def min(self, dim: Any = None, keepdim: bool = False) -> "TensorStub":
        return TensorStub(
            "min_reduce", source=self, dim=dim, keepdim=keepdim,
            dtype=self.spec.get("dtype", "float32"),
        )

    def sum(self, dim: Any = None, keepdim: bool = False) -> "TensorStub":
        return TensorStub(
            "sum_reduce", source=self, dim=dim, keepdim=keepdim,
            dtype=self.spec.get("dtype", "float32"),
        )

    def mean(self, dim: Any = None, keepdim: bool = False) -> "TensorStub":
        return TensorStub(
            "mean_reduce", source=self, dim=dim, keepdim=keepdim,
            dtype=self.spec.get("dtype", "float32"),
        )

    def prod(self, dim: Any = None, keepdim: bool = False) -> "TensorStub":
        return TensorStub(
            "prod_reduce", source=self, dim=dim, keepdim=keepdim,
            dtype=self.spec.get("dtype", "float32"),
        )

    def __int__(self) -> DerivedScalar:
        return DerivedScalar("int", self)

    def __float__(self) -> DerivedScalar:
        return DerivedScalar("float", self)

    # --- in-place fills become the equivalent random op ---
    def normal_(self, *args: Any) -> "TensorStub":
        return TensorStub("randn", shape=self.spec["shape"], dtype=self.spec["dtype"])

    def uniform_(self, *args: Any) -> "TensorStub":
        return TensorStub("rand", shape=self.spec["shape"], dtype=self.spec["dtype"])

    def random_(self, low: int = 0, high: int | None = None) -> "TensorStub":
        return TensorStub(
            "randint",
            shape=self.spec["shape"],
            dtype=self.spec["dtype"],
            low=int(low),
            high=int(high) if high is not None else 2**63,
        )

    def zero_(self) -> "TensorStub":
        return TensorStub("zeros", shape=self.spec["shape"], dtype=self.spec["dtype"])

    def fill_(self, value: Any) -> "TensorStub":
        return TensorStub("full", shape=self.spec["shape"], dtype=self.spec["dtype"], fill_value=value)

    def __repr__(self) -> str:
        return f"TensorStub({self.spec.get('kind')}, {self.spec.get('shape')})"


def _serialize_index(item: Any) -> list[Any]:
    """Serialize a ``__getitem__`` index into JSON-able per-dim specs."""
    dims = item if isinstance(item, tuple) else (item,)
    result: list[Any] = []
    for dim in dims:
        if dim is Ellipsis:
            result.append({"ellipsis": True})
        elif isinstance(dim, slice):
            result.append(
                {
                    "start": None if dim.start is None else int(dim.start),
                    "stop": None if dim.stop is None else int(dim.stop),
                    "step": None if dim.step is None else int(dim.step),
                }
            )
        elif isinstance(dim, int):
            result.append(int(dim))
        else:  # pragma: no cover - dynamic indices need gen_inputs authoring
            raise ValueError(f"unsupported case index: {dim!r}")
    return result


class _Iinfo:
    def __init__(self, bits: int) -> None:
        self.min = -(2 ** (bits - 1))
        self.max = 2 ** (bits - 1) - 1


class _Finfo:
    def __init__(self, max: float, min: float = 0.0, eps: float = 0.0) -> None:
        self.max = max
        self.min = min
        self.eps = eps
        self.tiny = min


_FINFO_MAX = {
    "float8_e4m3fn": 448.0,
    "float8_e5m2": 57344.0,
    "float8_e4m3fnuz": 240.0,
    "float8_e5m2fnuz": 57344.0,
    "float16": 65504.0,
    "float32": 3.4028235e38,
    "float64": 1.7976931348623157e308,
    "bfloat16": 3.38953139e38,
}


class _TorchStub(types.ModuleType):
    """A torch module that records case tensor construction instead of
    allocating real tensors."""

    def __init__(self) -> None:
        super().__init__("torch")
        for name in (
            "float32",
            "float16",
            "bfloat16",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bool",
            "complex64",
            "complex128",
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e4m3fnuz",
            "float8_e5m2fnuz",
        ):
            setattr(self, name, DType(name))

    def iinfo(self, dtype: Any) -> _Iinfo:
        bits = {
            "int8": 8,
            "int16": 16,
            "int32": 32,
            "int64": 64,
            "uint8": 8,
        }.get(str(dtype))
        if bits is None:
            raise TypeError(f"unsupported iinfo dtype: {dtype}")
        return _Iinfo(bits)

    def finfo(self, dtype: Any) -> _Finfo:
        name = str(dtype).removeprefix("torch.")
        return _Finfo(max=_FINFO_MAX.get(name, 3.4e38))

    def Generator(self, device: Any = None) -> _GeneratorStub:  # noqa: N802
        return _GeneratorStub()

    @staticmethod
    def _take_generator(generator: Any) -> int | None:
        if generator is None:
            return None
        if not isinstance(generator, _GeneratorStub):
            raise TypeError("unsupported generator in case construction")
        return generator.seed

    @staticmethod
    def _shape(value: Any) -> list[int]:
        if isinstance(value, int):
            return [value]
        return [int(d) for d in value]

    def randn(self, *shape: Any, **kwargs: Any) -> TensorStub:
        seed = self._take_generator(kwargs.get("generator"))
        return TensorStub(
            "randn",
            shape=self._shape(shape[0] if len(shape) == 1 else shape),
            dtype=self._dtype_name(kwargs.get("dtype", "float32")),
            seed=seed,
        )

    def rand(self, *shape: Any, **kwargs: Any) -> TensorStub:
        seed = self._take_generator(kwargs.get("generator"))
        return TensorStub(
            "rand",
            shape=self._shape(shape[0] if len(shape) == 1 else shape),
            dtype=self._dtype_name(kwargs.get("dtype", "float32")),
            seed=seed,
        )

    def randint(self, low: Any, high: Any, shape: Any, **kwargs: Any) -> TensorStub:
        seed = self._take_generator(kwargs.get("generator"))
        return TensorStub(
            "randint",
            shape=self._shape(shape),
            dtype=self._dtype_name(kwargs.get("dtype", "int64")),
            low=int(low),
            high=int(high),
            seed=seed,
        )

    def randperm(self, n: int, **kwargs: Any) -> TensorStub:
        seed = self._take_generator(kwargs.get("generator"))
        return TensorStub(
            "randperm",
            shape=[int(n)],
            dtype=self._dtype_name(kwargs.get("dtype", "int64")),
            seed=seed,
        )

    def tensor(self, data: Any, **kwargs: Any) -> TensorStub:
        return TensorStub(
            "tensor_literal",
            values=_jsonable(data),
            dtype=self._dtype_name(kwargs.get("dtype", "float32")),
        )

    def zeros(self, *shape: Any, **kwargs: Any) -> TensorStub:
        return TensorStub(
            "zeros",
            shape=self._shape(shape[0] if len(shape) == 1 else shape),
            dtype=self._dtype_name(kwargs.get("dtype", "float32")),
        )

    def ones(self, *shape: Any, **kwargs: Any) -> TensorStub:
        return TensorStub(
            "ones",
            shape=self._shape(shape[0] if len(shape) == 1 else shape),
            dtype=self._dtype_name(kwargs.get("dtype", "float32")),
        )

    def zeros_like(self, input: TensorStub, **kwargs: Any) -> TensorStub:
        return self.zeros(
            _shape_of(input),
            dtype=kwargs.get("dtype", input.spec.get("dtype", "float32")),
        )

    def ones_like(self, input: TensorStub, **kwargs: Any) -> TensorStub:
        return self.ones(
            _shape_of(input),
            dtype=kwargs.get("dtype", input.spec.get("dtype", "float32")),
        )

    def polar(self, abs: TensorStub, angle: TensorStub, **kwargs: Any) -> TensorStub:
        return TensorStub(
            "polar",
            shape=abs.spec["shape"],
            dtype="complex64",
            source=abs,
            angle=_build_dsl(angle),
        )

    def empty(self, *shape: Any, **kwargs: Any) -> TensorStub:
        return TensorStub(
            "empty",
            shape=self._shape(shape[0] if len(shape) == 1 else shape),
            dtype=self._dtype_name(kwargs.get("dtype", "float32")),
        )

    def full(self, shape: Any, fill_value: Any, **kwargs: Any) -> TensorStub:
        return TensorStub(
            "full",
            shape=self._shape(shape),
            dtype=self._dtype_name(kwargs.get("dtype", "float32")),
            fill_value=fill_value,
        )

    def arange(self, *args: Any, **kwargs: Any) -> TensorStub:
        resolved = [_python_scalar(a) for a in args]
        start = resolved[0] if len(resolved) == 1 else resolved[0]
        stop = resolved[1] if len(resolved) >= 2 else None
        step = kwargs.get("step", resolved[2] if len(resolved) >= 3 else 1)
        if stop is None:
            stop, start = start, 0
        return TensorStub(
            "arange",
            start=int(start),
            stop=int(stop),
            step=int(step),
            dtype=self._dtype_name(kwargs.get("dtype", "int64")),
        )

    def cat(self, tensors: Sequence[TensorStub], **kwargs: Any) -> TensorStub:
        return TensorStub("cat", sources=list(tensors))

    def cumsum(self, input: TensorStub, dim: int = 0) -> TensorStub:
        return input.cumsum(int(dim))

    def concat(self, tensors: Sequence[TensorStub], **kwargs: Any) -> TensorStub:
        return TensorStub("cat", sources=list(tensors))

    def stack(self, tensors: Sequence[TensorStub], **kwargs: Any) -> TensorStub:
        return TensorStub("stack", sources=list(tensors))

    def where(self, condition: TensorStub, x: Any, y: Any) -> TensorStub:
        return TensorStub(
            "where",
            shape=_shape_of(x),
            dtype=x.spec.get("dtype", "float32"),
            cond=condition,
            x=x,
            y=y,
        )

    def eye(self, n: int, m: int | None = None, **kwargs: Any) -> TensorStub:
        return TensorStub(
            "eye",
            shape=[n, m if m is not None else n],
            dtype=self._dtype_name(kwargs.get("dtype", "float32")),
        )

    def linspace(self, start: Any, end: Any, steps: int, **kwargs: Any) -> TensorStub:
        return TensorStub(
            "linspace",
            shape=[int(steps)],
            dtype=self._dtype_name(kwargs.get("dtype", "float32")),
            start=start,
            end=end,
        )

    def meshgrid(self, *tensors: Any, **kwargs: Any) -> tuple[TensorStub, ...]:
        indexed = kwargs.get("indexing", "ij")
        result: list[TensorStub] = []
        for index, tensor in enumerate(tensors):
            shape = list(_shape_of(tensor))
            for other_index, other in enumerate(tensors):
                if other_index != index:
                    shape.insert(other_index if indexed == "ij" else -1, 1)
            result.append(TensorStub("expand", source=tensor, shape=shape))
        return tuple(result)

    def clamp(
        self,
        input: TensorStub,
        min: Any = None,
        max: Any = None,
    ) -> TensorStub:
        return TensorStub(
            "clamp", source=input, min=min, max=max, dtype=input.spec.get("dtype", "float32")
        )

    @staticmethod
    def _dtype_name(dtype: Any) -> str:
        name = str(dtype)
        return name[len("torch.") :] if name.startswith("torch.") else name

    def _record_fill(self, kind: str, *args: Any) -> None:
        raise TypeError(f"unsupported torch fill call in case construction: {kind}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, TensorStub):
        raise TypeError("nested TensorStub in literal data")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    raise TypeError(f"unsupported literal value: {type(value).__name__}")


def _shape_of(stub: TensorStub) -> list[int]:
    """Resolve a stub's shape through transform wrappers to the leaf."""
    spec = stub.spec
    if "shape" in spec:
        return spec["shape"]
    if "source" in spec:
        return _shape_of(spec["source"])
    if "sources" in spec:
        return _shape_of(spec["sources"][0])
    raise TypeError(f"cannot resolve shape of {stub}")


def _dtype_of(stub: TensorStub) -> str | None:
    """Resolve a stub's dtype through transform wrappers to the leaf."""
    spec = stub.spec
    if "dtype" in spec and spec.get("dtype"):
        return spec["dtype"]
    if "source" in spec:
        return _dtype_of(spec["source"])
    if "sources" in spec:
        return _dtype_of(spec["sources"][0])
    return None


def _python_scalar(value: Any) -> Any:
    """Coerce a case-build value to a plain Python scalar where possible."""
    if isinstance(value, DerivedScalar):
        return value
    if isinstance(value, (int, float, bool)):
        return value
    raise TypeError(f"unsupported non-constant arange argument: {value!r}")


def _chunk_sizes(total: int, chunk: int) -> list[int]:
    return [min(chunk, total - offset) for offset in range(0, total, chunk)] or [0]


def _constant_values(stub: TensorStub) -> list[Any] | None:
    """Values of a case tensor when they are statically known (literal,
    constant-filled, or a contiguous slice/expand/repeat of one); None
    otherwise."""
    spec = stub.spec
    kind = spec["kind"]
    if kind == "tensor_literal":
        return list(spec["values"])
    if kind in {"zeros", "ones"}:
        count = 1
        for dim in spec["shape"]:
            count *= dim
        return [0 if kind == "zeros" else 1] * count
    if kind == "full":
        count = 1
        for dim in spec["shape"]:
            count *= dim
        return [spec["fill_value"]] * count
    if kind == "slice" and _single_contiguous_slice(spec.get("index", [])):
        values = _constant_values(spec["source"])
        if values is None:
            return None
        (index,) = spec["index"]
        start = index.get("start") or 0
        stop = index.get("stop")
        return values[start:stop]
    if kind == "repeat":
        values = _constant_values(spec["source"])
        if values is None:
            return None
        return values * int(spec["sizes"][0])
    if kind == "expand":
        values = _constant_values(spec["source"])
        if values is None:
            return None
        count = 1
        for dim in spec["shape"]:
            count *= dim
        if not values:
            return values
        return (values * (count // len(values) + 1))[:count]
    return None


def _single_contiguous_slice(index: Sequence[Any]) -> bool:
    return len(index) == 1 and isinstance(index[0], dict) and "start" in index[0]


# ---------------------------------------------------------------------------
# Stub environment: torch + harness + problems.* case modules.
# ---------------------------------------------------------------------------


def _install_stubs(source_root: Path) -> tuple[dict[str, Any], "ImportFinder"]:
    """Install the torch/harness stubs and a meta-path finder that resolves
    ``problems.<group>.<op>.cases`` and ``harness.*`` imports by executing
    the real source files under the stub."""
    torch_stub = _TorchStub()
    sys.modules.setdefault("torch", torch_stub)

    # ``import torch.nn.functional as F`` in sibling reference modules must
    # resolve under the stub; the functions are only imported, never called,
    # during case extraction.
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.functional = types.ModuleType("torch.nn.functional")
    torch_stub.nn = torch_nn
    sys.modules.setdefault("torch.nn", torch_nn)
    sys.modules.setdefault("torch.nn.functional", torch_nn.functional)

    harness = types.ModuleType("harness")
    harness.__path__ = []
    sys.modules.setdefault("harness", harness)
    sys.modules.pop("harness.correctness", None)
    sys.modules.pop("harness.bench", None)

    finder = ImportFinder(source_root)
    sys.meta_path.insert(0, finder)
    return (
        {
            "torch": torch_stub,
            "harness": harness,
            "int": _stub_int,
            "float": _stub_float,
        },
        finder,
    )


_real_int = int
_real_float = float


def _stub_int(value: Any, *args: Any, **kwargs: Any) -> Any:
    """``int(...)`` inside case construction: when applied to a recorded
    tensor stub, mark it as a runtime-derived scalar instead of coercing it
    to a plain (wrong) value."""
    if isinstance(value, DerivedScalar):
        return value
    if isinstance(value, TensorStub) and not args and not kwargs:
        return DerivedScalar("int", value)
    return _real_int(value, *args, **kwargs)


def _stub_float(value: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(value, DerivedScalar):
        return value
    if isinstance(value, TensorStub) and not args and not kwargs:
        return DerivedScalar("float", value)
    return _real_float(value, *args, **kwargs)


class ImportFinder:
    """Meta-path finder: execute real ``harness.*`` and ``problems.*.cases``
    files under the torch stub so cross-imports resolve during case
    extraction, while keeping the real function objects available for
    embedding into oracles."""

    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root.resolve()

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname == "harness.correctness":
            return self._spec_from_file(self.source_root / "harness" / "correctness.py", fullname)
        if fullname == "harness.bench":
            return self._spec_from_file(self.source_root / "harness" / "bench.py", fullname)
        if fullname == "problems" or fullname.startswith("problems."):
            match_cases = re.fullmatch(
                r"problems\.([a-z0-9_]+)\.([a-z0-9_]+)\.cases", fullname
            )
            if match_cases:
                group, op = match_cases.groups()
                path = self.source_root / "problems" / group / op / "cases.py"
                if path.is_file():
                    return self._spec_from_file(path, fullname)
                return None
            match_ref = re.fullmatch(
                r"problems\.([a-z0-9_]+)\.([a-z0-9_]+)\.reference_torch", fullname
            )
            if match_ref:
                group, op = match_ref.groups()
                path = self.source_root / "problems" / group / op / "reference_torch.py"
                if path.is_file():
                    return self._spec_from_file(path, fullname)
                return None
            match_module = re.fullmatch(
                r"problems\.([a-z0-9_]+)\.([a-z0-9_]+)", fullname
            )
            if match_module:
                group, module = match_module.groups()
                path = self.source_root / "problems" / group / f"{module}.py"
                if path.is_file():
                    return self._spec_from_file(path, fullname)
            # Parent packages resolve as namespace packages so the import
            # machinery can reach the real module files above.
            return importlib.util.spec_from_loader(fullname, loader=None, is_package=True)
        if fullname == "sglang.kernels.ops.moe.inkling_moe":
            module = types.ModuleType(fullname)
            module.compute_grouped_gemm_metadata = lambda *a, **k: tuple(
                TensorStub("opaque") for _ in range(4)
            )
            sys.modules[fullname] = module
            return importlib.util.spec_from_loader(fullname, _StubLoader())
        # Any other sglang import must FAIL so source modules fall back to
        # their pure-torch branches (e.g. ``_batch_utils`` imports
        # ``problems.lora._lora_batch_info`` instead).
        return None

    def _spec_from_file(self, path: Path, fullname: str) -> Any:
        spec = importlib.util.spec_from_file_location(fullname, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        return spec


class _StubLoader:
    """Loader for pre-built stub modules already installed in sys.modules."""

    def create_module(self, spec: Any) -> types.ModuleType:
        return sys.modules[spec.name]

    def exec_module(self, module: types.ModuleType) -> None:
        pass


# ---------------------------------------------------------------------------
# Case extraction.
# ---------------------------------------------------------------------------


class CaseExtractionError(Exception):
    """Raised when a case grid cannot be traced declaratively; the operator
    then falls back to the embedded-constructor gen_inputs path."""


def _extract_case_lists(
    cases_path: Path, stubs: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute a cases.py under the stubs and return the correctness and
    timing case dicts (kwargs + recorded tensor stubs)."""

    namespace: dict[str, Any] = dict(stubs)
    namespace["__name__"] = f"_kcomp_cases_{cases_path.parent.name}"
    namespace["__file__"] = str(cases_path)
    try:
        exec(compile(cases_path.read_text(encoding="utf-8"), str(cases_path), "exec"), namespace)
    except Exception as exc:  # pragma: no cover - depends on the source
        raise CaseExtractionError(f"cannot execute {cases_path}: {exc}") from exc

    correctness = namespace.get("CORRECTNESS_CASES")
    timing = namespace.get("BENCH_CASES")
    if not isinstance(correctness, list) or not isinstance(timing, list):
        raise ValueError(f"{cases_path}: missing CORRECTNESS_CASES/BENCH_CASES lists")

    def resolve_cases(cases: list[Any]) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        for case in cases:
            # Some benches hold lazy zero-arg callables to bound memory.
            if callable(case):
                case = case()
            if not isinstance(case, dict):
                raise ValueError(f"{cases_path}: case is not a mapping")
            resolved.append(case)
        return resolved

    return resolve_cases(correctness), resolve_cases(timing)


# ---------------------------------------------------------------------------
# Workload recipes: declarative random vs custom build-DSL.
# ---------------------------------------------------------------------------


def _recipe_for(value: Any) -> tuple[dict[str, Any], int | None]:
    """Convert a recorded case value into either a declarative v6.2 recipe or
    a custom build-DSL node.  Returns (recipe, seed)."""
    if isinstance(value, TensorStub):
        return _tensor_recipe(value)
    if isinstance(value, DerivedScalar):
        # A scalar derived from a tensor's runtime value can only stay
        # consistent with the tensor when the whole case is replayed by the
        # embedded constructor; force that path.
        raise ValueError("derived scalar needs the embedded constructor")
    if isinstance(value, DType):
        return ({"type": "custom", "build": {"op": "dtype", "name": str(value)}}, None)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return ({"type": "scalar", "value": value}, None)
    if isinstance(value, (list, tuple)):
        return ({"type": "scalar", "value": list(value)}, None)
    raise ValueError(f"unsupported case value: {type(value).__name__}")


def _tensor_recipe(stub: TensorStub) -> tuple[dict[str, Any], int | None]:
    spec = stub.spec
    kind = spec["kind"]
    seed = spec.get("seed")

    if kind == "opaque":
        raise ValueError("case tensor depends on external runtime state")

    # Simple random leaves become declarative recipes.
    if kind == "randn":
        return ({"type": "random", "shape": spec["shape"], "dtype": spec["dtype"]}, seed)
    if kind == "rand":
        return (
            {"type": "random", "shape": spec["shape"], "dtype": spec["dtype"], "distribution": "uniform"},
            seed,
        )
    if kind == "randint":
        return (
            {
                "type": "random",
                "shape": spec["shape"],
                "dtype": spec["dtype"],
                "distribution": "integer",
                "low": spec["low"],
                "high": spec["high"],
            },
            seed,
        )

    # A terminal cast folds into ``source_dtype`` (generate in the source
    # dtype, apply any transforms, cast to the declared dtype) — the same
    # semantics as the source's ``(randn(...) * 2 - 1).to(bf16)``.
    if kind == "to":
        inner, inner_seed = _tensor_recipe(spec["source"])
        if inner["type"] == "random":
            folded = dict(inner)
            folded["dtype"] = spec["dtype"]
            folded["source_dtype"] = inner.get("source_dtype", inner["dtype"])
            return (folded, seed if seed is not None else inner_seed)

    # Scalar arithmetic on a declarative random base folds into transforms.
    if kind in {"scale", "shift", "rshift", "divide", "rdivide"}:
        inner, inner_seed = _tensor_recipe(spec["source"])
        if inner["type"] == "random":
            op = {
                "scale": "multiply",
                "shift": "add",
                "rshift": "rsubtract",
                "divide": "divide",
                "rdivide": "rdivide",
            }[kind]
            transformed = dict(inner)
            transforms = list(inner.get("transforms", []))
            transforms.append({"op": op, "value": spec["value"]})
            transformed["transforms"] = transforms
            return (transformed, seed if seed is not None else inner_seed)

    # Everything else needs custom construction: emit a build-DSL node.
    return ({"type": "custom", "build": _build_dsl(stub)}, seed)


def _build_dsl(stub: TensorStub) -> dict[str, Any]:
    """Serialize a TensorStub tree into the build-DSL interpreted by the
    generated ``gen_inputs``.  The child of a unary op is stored under
    ``operand`` so the op's own scalar ``value`` (if any) never collides."""
    spec = stub.spec
    kind = spec["kind"]
    node: dict[str, Any] = {"op": kind}
    for key in ("shape", "dtype", "seed", "low", "high", "fill_value", "values", "args", "index", "dims", "angle", "min", "max", "value", "rhs", "mask", "dim", "start", "length", "sizes", "shifts", "cond", "x", "y", "end", "keepdim", "stop", "step"):
        if key in spec:
            value = spec[key]
            if isinstance(value, TensorStub):
                value = _build_dsl(value)
            node[key] = value
    if "source" in spec:
        node["operand"] = _build_dsl(spec["source"])
    if "sources" in spec:
        node["operands"] = [_build_dsl(source) for source in spec["sources"]]
    if "writes" in spec:
        node = {
            "op": "writes",
            "base": node,
            "writes": [
                {
                    "index": write["index"],
                    "value": (
                        _build_dsl(write["value"])
                        if isinstance(write["value"], TensorStub)
                        else write["value"]
                    ),
                }
                for write in spec["writes"]
            ],
        }
    return node


def _case_to_workloads(
    op_name: str,
    phase: str,
    cases: Sequence[Mapping[str, Any]],
    *,
    parameter_names: Sequence[str],
) -> list[dict[str, Any]]:
    workloads: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        inputs: dict[str, Any] = {}
        seeds: list[int] = []
        for name, value in case.items():
            if name not in parameter_names:
                continue
            recipe, value_seed = _recipe_for(value)
            inputs[name] = recipe
            if value_seed is not None:
                seeds.append(value_seed)
        workload: dict[str, Any] = {
            "name": f"{op_name}-{index:05d}-{phase}",
            "inputs": inputs,
        }
        if seeds:
            workload["seed"] = min(seeds)
        tolerance = _workload_tolerance(inputs)
        if tolerance is not None:
            workload["tolerance"] = tolerance
        workloads.append(workload)
    return workloads


def _workload_tolerance(inputs: Mapping[str, Any]) -> dict[str, Any] | None:
    """Pick the harness tolerance by the effective floating dtype of the
    workloads' tensors; integer-only workloads carry no tolerance (v6.2
    compares integers exactly)."""
    dtypes: list[str] = []
    for spec in inputs.values():
        if not isinstance(spec, dict) or spec.get("type") not in {"random", "custom"}:
            continue
        dtype = spec.get("dtype")
        if spec.get("type") == "custom":
            dtype = _dsl_leaf_dtype(spec.get("build", {}))
        for transform in spec.get("transforms", []) or []:
            if isinstance(transform, dict) and transform.get("op") == "cast":
                dtype = transform.get("dtype")
        if isinstance(dtype, str) and dtype:
            dtypes.append(dtype)
    floating = [
        dtype
        for dtype in dtypes
        if dtype not in {"int8", "int16", "int32", "int64", "uint8", "bool"}
    ]
    if not floating:
        return None
    return _tolerance_for_dtype(floating[0])


# ---------------------------------------------------------------------------
# Embedded-constructor mode: case grids whose construction depends on runtime
# tensor values (cumsum of random lens, item()-derived bounds, loops) cannot
# be traced declaratively.  For those the source case constructor is embedded
# into ``gen_inputs`` and the case lists are re-expressed as their call args.
# ---------------------------------------------------------------------------


def _embedded_case_plan(
    cases_path: Path,
    source_root: Path,
    stubs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any | None, list[Any], str]:
    """Return (correctness_args, timing_args, check_fn, constructor_fns,
    extra_imports) for an op whose cases cannot be traced declaratively."""
    source = cases_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    namespace: dict[str, Any] = dict(stubs)
    namespace["__name__"] = f"_kcomp_cases_{cases_path.parent.name}"
    namespace["__file__"] = str(cases_path)
    # Execute the full module; the case-list construction is what raised
    # under the stub, but everything defined before it (constants, case
    # constructors, the check) executes first and keeps its correct source
    # coordinates for inspect.getsource.
    try:
        exec(compile(source, str(cases_path), "exec"), namespace)
    except Exception:
        pass

    correctness_args = _eval_case_args(tree, namespace, "CORRECTNESS_CASES")
    timing_args = _eval_case_args(tree, namespace, "BENCH_CASES")
    check_fn = _find_check_function(tree, namespace)
    constructor_fns = _find_constructors(tree, namespace)

    # Non-torch module imports used by the constructors (e.g. an sglang
    # metadata helper) are carried into the oracle so gen_inputs can replay
    # the construction at runtime.
    extra_imports: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        module = getattr(node, "module", None) or ""
        if node.names[0].name == "torch" or module.startswith(("harness", "problems")):
            continue
        segment = ast.get_source_segment(source, node)
        if segment is not None:
            extra_imports.append(segment)
    return (
        correctness_args,
        timing_args,
        check_fn,
        constructor_fns,
        "\n".join(extra_imports),
    )


def _eval_case_args(tree: ast.Module, namespace: dict[str, Any], name: str) -> list[dict[str, Any]]:
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        ):
            continue
        result: list[dict[str, Any]] = []
        for element in _flatten_case_expr(node.value, namespace):
            result.append(_serialize_call(element, namespace))
        return result
    raise ValueError(f"{name} not found in cases.py")


def _flatten_case_expr(node: ast.expr, namespace: dict[str, Any]) -> list[ast.expr]:
    if isinstance(node, ast.List):
        flattened: list[ast.expr] = []
        for element in node.elts:
            flattened.extend(_flatten_case_expr(element, namespace))
        return flattened
    if isinstance(node, ast.ListComp):
        values: list[ast.expr] = []
        for element in _expand_comprehension(node, namespace):
            values.extend(_flatten_case_expr(element, namespace))
        return values
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_case_expr(node.left, namespace) + _flatten_case_expr(node.right, namespace)
    return [node]


def _expand_comprehension(node: ast.ListComp, namespace: dict[str, Any]) -> list[ast.expr]:
    """Expand a list comprehension over constant iterables into concrete
    element expressions (with the target names substituted by constants)."""
    expressions: list[ast.expr] = [node.elt]
    for generator in reversed(node.generators):
        iterable = _const_eval(generator.iter, namespace)
        expanded: list[ast.expr] = []
        for value in iterable:
            for expression in expressions:
                expanded.append(
                    _substitute_name(expression, generator.target, value, namespace)
                )
        expressions = expanded
    return expressions


def _substitute_name(
    node: ast.expr, target: ast.Name | ast.Tuple, value: Any, namespace: dict[str, Any]
) -> ast.expr:
    if isinstance(target, ast.Tuple):
        # ``for a, b in [(1, 2), ...]``: values are paired.
        names = [element.id for element in target.elts if isinstance(element, ast.Name)]
        if isinstance(value, (list, tuple)) and len(value) == len(names):
            result = node
            for name, item in zip(names, value):
                result = _substitute_name(result, ast.Name(id=name, ctx=ast.Load()), item, namespace)
            return result
        return node
    if isinstance(node, ast.Name) and node.id == target.id:
        return ast.Constant(value=value)
    if isinstance(node, ast.Call):
        return ast.Call(
            func=node.func,
            args=[_substitute_name(a, target, value, namespace) for a in node.args],
            keywords=[
                ast.keyword(
                    arg=keyword.arg,
                    value=_substitute_name(keyword.value, target, value, namespace),
                )
                for keyword in node.keywords
            ],
        )
    if isinstance(node, ast.BinOp):
        return ast.BinOp(
            left=_substitute_name(node.left, target, value, namespace),
            op=node.op,
            right=_substitute_name(node.right, target, value, namespace),
        )
    if isinstance(node, ast.UnaryOp):
        return ast.UnaryOp(
            op=node.op, operand=_substitute_name(node.operand, target, value, namespace)
        )
    if isinstance(node, ast.Subscript):
        return ast.Subscript(
            value=_substitute_name(node.value, target, value, namespace),
            slice=_substitute_name(node.slice, target, value, namespace),
        )
    if isinstance(node, ast.Tuple):
        return ast.Tuple(
            elts=[_substitute_name(e, target, value, namespace) for e in node.elts],
            ctx=node.ctx,
        )
    if isinstance(node, ast.List):
        return ast.List(
            elts=[_substitute_name(e, target, value, namespace) for e in node.elts],
            ctx=node.ctx,
        )
    if isinstance(node, ast.Lambda):
        defaults = [
            _substitute_name(default, target, value, namespace)
            for default in node.args.defaults
        ]
        arguments = ast.arguments(
            posonlyargs=node.args.posonlyargs,
            args=node.args.args,
            vararg=node.args.vararg,
            kwonlyargs=node.args.kwonlyargs,
            kw_defaults=node.args.kw_defaults,
            kwarg=node.args.kwarg,
            defaults=defaults,
        )
        return ast.Lambda(args=arguments, body=node.body)
    return node


def _serialize_call(node: ast.expr, namespace: dict[str, Any]) -> dict[str, Any]:
    """Serialize a constructor call (or a lazy lambda wrapping one) into
    JSON-able positional args + kwargs."""
    if isinstance(node, ast.Lambda):
        bindings = {
            argument.arg: _const_eval(default, namespace)
            for argument, default in zip(
                node.args.args, node.args.defaults, strict=True
            )
        }
        body = node.body
        if isinstance(body, ast.Call):
            return _serialize_call_with(body, namespace, bindings)
    if isinstance(node, ast.Call):
        return _serialize_call_with(node, namespace, {})
    raise ValueError(f"unsupported case expression: {ast.dump(node)[:80]}")


def _serialize_call_with(
    node: ast.Call, namespace: dict[str, Any], bindings: dict[str, Any]
) -> dict[str, Any]:
    args = [_const_eval(argument, namespace, bindings) for argument in node.args]
    kwargs = {
        keyword.arg: _const_eval(keyword.value, namespace, bindings)
        for keyword in node.keywords
        if keyword.arg is not None and keyword.arg != "check"
    }
    constructor = node.func.id if isinstance(node.func, ast.Name) else "_case"
    return {"constructor": constructor, "args": args, "kwargs": kwargs}


def _const_eval(
    node: ast.expr, namespace: dict[str, Any], bindings: dict[str, Any] | None = None
) -> Any:
    bindings = bindings or {}
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        value = bindings.get(node.id, namespace.get(node.id))
        if isinstance(value, _TorchStub):
            return value
        if isinstance(value, DType):
            return str(value)
        if isinstance(value, list):
            return _jsonable(value)
        if value is None and node.id in namespace:
            return None
        if value is not None and not isinstance(value, (int, float, str, bool)):
            raise ValueError(f"unsupported case constant: {node.id}={value!r}")
        return value
    if isinstance(node, ast.Attribute):
        value = _const_eval(node.value, namespace, bindings)
        if isinstance(value, _TorchStub):
            return str(getattr(value, node.attr))
        return getattr(value, node.attr)
    if isinstance(node, ast.BinOp):
        left = _const_eval(node.left, namespace, bindings)
        right = _const_eval(node.right, namespace, bindings)
        op = type(node.op)
        if op is ast.Add:
            return left + right
        if op is ast.Sub:
            return left - right
        if op is ast.Mult:
            return left * right
        if op is ast.Div:
            return left / right
        if op is ast.FloorDiv:
            return left // right
        if op is ast.LShift:
            return left << right
        if op is ast.RShift:
            return left >> right
        if op is ast.Pow:
            return left**right
        if op is ast.Mod:
            return left % right
        raise ValueError(f"unsupported case BinOp: {ast.dump(node.op)}")
    if isinstance(node, ast.UnaryOp):
        operand = _const_eval(node.operand, namespace, bindings)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.Not):
            return not operand
        raise ValueError(f"unsupported case UnaryOp: {ast.dump(node.op)}")
    if isinstance(node, ast.Subscript):
        base = _const_eval(node.value, namespace, bindings)
        index = _const_eval(node.slice, namespace, bindings)
        return base[index]
    if isinstance(node, ast.Tuple):
        return [_const_eval(element, namespace, bindings) for element in node.elts]
    if isinstance(node, ast.List):
        return [_const_eval(element, namespace, bindings) for element in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        # torch.tensor([...], dtype=..., device=...) inside case args becomes
        # a materializable literal marker.
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "tensor"
        ):
            data = _const_eval(node.args[0], namespace, bindings)
            dtype = "float32"
            for keyword in node.keywords:
                if keyword.arg == "dtype":
                    dtype = str(_const_eval(keyword.value, namespace, bindings))
            return {"__tensor__": _jsonable(data), "dtype": dtype}
        raise ValueError(f"unsupported case expression: {ast.dump(node)[:80]}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        builtin = node.func.id
        args = [_const_eval(argument, namespace, bindings) for argument in node.args]
        kwargs = {
            keyword.arg: _const_eval(keyword.value, namespace, bindings)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        if builtin == "max":
            return max(*args, **kwargs)
        if builtin == "min":
            return min(*args, **kwargs)
        if builtin == "sum":
            return sum(args)
        if builtin == "int":
            return int(args[0])
        if builtin == "float":
            return float(args[0])
        if builtin == "len":
            return len(args[0])
        if builtin == "abs":
            return abs(args[0])
        if builtin == "round":
            return round(args[0])
        if builtin == "sorted":
            return sorted(args[0])
        if builtin == "list":
            return list(args[0])
        if builtin == "tuple":
            return tuple(args[0])
        raise ValueError(f"unsupported case builtin: {builtin}")
    raise ValueError(f"unsupported case expression: {ast.dump(node)[:80]}")


def _find_constructors(tree: ast.Module, namespace: dict[str, Any]) -> list[Any]:
    names: set[str] = set()
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in {"CORRECTNESS_CASES", "BENCH_CASES"}
                for target in node.targets
            )
        ):
            continue
        for element in _flatten_case_expr(node.value, namespace):
            if isinstance(element, ast.Lambda):
                element = element.body
            if isinstance(element, ast.Call) and isinstance(element.func, ast.Name):
                names.add(element.func.id)
    functions: list[Any] = []
    for name in sorted(names):
        candidate = namespace.get(name)
        if callable(candidate):
            functions.append(candidate)
    return functions


def _find_check_function(tree: ast.Module, namespace: dict[str, Any]) -> Any | None:
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "CORRECTNESS_CASES"
                for target in node.targets
            )
        ):
            continue
        for element in _flatten_case_expr(node.value, namespace):
            call = element.body if isinstance(element, ast.Lambda) else element
            if not isinstance(call, ast.Call):
                continue
            for keyword in call.keywords:
                if keyword.arg != "check" or keyword.value is None:
                    continue
                if isinstance(keyword.value, ast.Name):
                    candidate = namespace.get(keyword.value.id)
                    if callable(candidate):
                        return candidate
                elif isinstance(keyword.value, ast.Call):
                    factory = namespace.get(keyword.value.func.id)
                    if callable(factory):
                        args = [_const_eval(a, namespace) for a in keyword.value.args]
                        kwargs = {
                            k.arg: _const_eval(k.value, namespace)
                            for k in keyword.value.keywords
                            if k.arg is not None
                        }
                        return factory(*args, **kwargs)
    return None


def _dsl_leaf_dtype(node: Mapping[str, Any]) -> str | None:
    """Effective dtype of a build-DSL node after any terminal cast."""
    op = node.get("op")
    if op == "dtype":
        return node.get("name")
    if op == "to":
        return node.get("dtype")
    if op in {"randn", "rand", "randint", "zeros", "ones", "empty", "full", "tensor_literal"}:
        return node.get("dtype")
    if op == "polar":
        return "complex64"
    if op == "writes":
        return _dsl_leaf_dtype(node.get("base", {}))
    if op == "clamp":
        return node.get("dtype") or _dsl_leaf_dtype(node.get("operand", {}))
    if op.startswith("compare_") or op in {"logical_and", "logical_or", "logical_not"}:
        return "bool"
    if op in {"sum_reduce", "mean_reduce", "prod_reduce"}:
        return _dsl_leaf_dtype(node.get("operand", {}))
    if op in {"max_reduce", "min_reduce"}:
        return node.get("dtype") or _dsl_leaf_dtype(node.get("operand", {}))
    if op == "mask_index":
        return _dsl_leaf_dtype(node.get("operand", {}))
    if op == "masked_fill":
        return _dsl_leaf_dtype(node.get("operand", {}))
    if op == "where":
        return _dsl_leaf_dtype(node.get("x", {}))
    if op in {"eye", "linspace"}:
        return node.get("dtype")
    if op == "arange":
        return node.get("dtype", "int64")
    if op in {"expand", "permute", "flip", "roll", "unsqueeze", "squeeze", "narrow", "repeat"}:
        return _dsl_leaf_dtype(node.get("operand", {}))
    if op in {"slice", "scale", "shift", "rshift", "divide", "rdivide", "abs", "max_reduce", "min_reduce"}:
        return _dsl_leaf_dtype(node.get("operand", {}))
    if op in {"cat", "stack"}:
        operands = node.get("operands", [])
        if operands:
            return _dsl_leaf_dtype(operands[0])
    return None


_HARNESS_TOLERANCES = {
    "float32": {"rtol": 1e-4, "atol": 1e-4},
    "bfloat16": {"rtol": 1.5e-2, "atol": 1.5e-2},
    "float16": {"rtol": 1e-2, "atol": 1e-2},
}
_DEFAULT_HARNESS_TOLERANCE = {"rtol": 1e-2, "atol": 1e-2}


def _tolerance_for_dtype(dtype_name: str) -> dict[str, Any]:
    contract = _HARNESS_TOLERANCES.get(dtype_name, _DEFAULT_HARNESS_TOLERANCE)
    return {
        "rtol": contract["rtol"],
        "atol": contract["atol"],
        "atol_scale": 1.0,
        "required_matched_ratio": 1.0,
    }


# ---------------------------------------------------------------------------
# gen_inputs emission from the build-DSL.
# ---------------------------------------------------------------------------

_GEN_INPUTS_TEMPLATE = '''


def gen_inputs(ctx, device):
    result = {}
    for name, spec in ctx["inputs"].items():
        if not (isinstance(spec, dict) and spec.get("type") == "custom"):
            continue
        result[name] = _build(spec["build"], device)
    return result


def _slice_index(items):
    result = []
    for item in items:
        if isinstance(item, dict) and "start" in item:
            result.append(slice(item["start"], item["stop"], item["step"]))
        elif isinstance(item, dict) and item.get("ellipsis"):
            result.append(Ellipsis)
        else:
            result.append(item)
    return tuple(result)


def _build(node, device):
    op = node["op"]
    if op == "dtype":
        return getattr(torch, node["name"])
    if op == "int":
        return int(_build(node["operand"], device))
    if op == "float":
        return float(_build(node["operand"], device))
    if op in {"randn", "rand"}:
        generator = torch.Generator(device=device).manual_seed(node.get("seed", 0))
        fn = torch.randn if op == "randn" else torch.rand
        return fn(
            tuple(node["shape"]),
            dtype=getattr(torch, node["dtype"]),
            device=device,
            generator=generator,
        )
    if op == "randint":
        generator = torch.Generator(device=device).manual_seed(node.get("seed", 0))
        return torch.randint(
            node["low"],
            node["high"],
            tuple(node["shape"]),
            dtype=getattr(torch, node["dtype"]),
            device=device,
            generator=generator,
        )
    if op == "randperm":
        generator = torch.Generator(device=device).manual_seed(node.get("seed", 0))
        return torch.randperm(
            node["shape"][0],
            dtype=getattr(torch, node["dtype"]),
            device=device,
            generator=generator,
        )
    if op in {"zeros", "ones", "empty"}:
        fn = getattr(torch, op)
        return fn(tuple(node["shape"]), dtype=getattr(torch, node["dtype"]), device=device)
    if op == "full":
        return torch.full(
            tuple(node["shape"]),
            node["fill_value"],
            dtype=getattr(torch, node["dtype"]),
            device=device,
        )
    if op == "tensor_literal":
        return torch.tensor(node["values"], dtype=getattr(torch, node["dtype"]), device=device)
    if op == "to":
        return _build(node["operand"], device).to(getattr(torch, node["dtype"]))
    if op == "reshape":
        return _build(node["operand"], device).reshape(tuple(node["shape"]))
    if op == "transpose":
        return _build(node["operand"], device).transpose(
            *node.get("dims", [0, 1])
        )
    if op == "slice":
        value = _build(node["operand"], device)
        return value[_slice_index(node["index"])]
    if op == "scale":
        return _build(node["operand"], device) * node["value"]
    if op == "shift":
        return _build(node["operand"], device) + node["value"]
    if op == "rshift":
        return node["value"] - _build(node["operand"], device)
    if op == "divide":
        return _build(node["operand"], device) / node["value"]
    if op == "rdivide":
        return node["value"] / _build(node["operand"], device)
    if op == "abs":
        return _build(node["operand"], device).abs()
    if op in {"max_reduce", "min_reduce", "sum_reduce", "mean_reduce", "prod_reduce"}:
        value = _build(node["operand"], device)
        fn = {
            "max_reduce": value.max,
            "min_reduce": value.min,
            "sum_reduce": value.sum,
            "mean_reduce": value.mean,
            "prod_reduce": value.prod,
        }[op]
        dim = node.get("dim")
        if dim is None:
            return fn()
        return fn(dim=dim, keepdim=node.get("keepdim", False))
    if op == "clamp":
        return torch.clamp(
            _build(node["operand"], device),
            min=node.get("min"),
            max=node.get("max"),
        )
    if op == "cat":
        return torch.cat([_build(operand, device) for operand in node["operands"]])
    if op == "stack":
        return torch.stack([_build(operand, device) for operand in node["operands"]])
    if op == "polar":
        return torch.polar(_build(node["operand"], device), _build(node["angle"], device))
    if op == "writes":
        base = _build(node["base"], device)
        for write in node["writes"]:
            value = (
                _build(write["value"], device)
                if isinstance(write["value"], dict)
                else write["value"]
            )
            base[_slice_index(write["index"])] = value
        return base
    if op in {"expand", "permute", "flip", "roll", "unsqueeze", "squeeze", "narrow", "repeat", "logical_not"}:
        value = _build(node["operand"], device)
        if op == "expand":
            return value.expand(tuple(node["shape"]))
        if op == "permute":
            return value.permute(tuple(node["dims"]))
        if op == "flip":
            return value.flip(tuple(node["dims"]))
        if op == "roll":
            return value.roll(node["shifts"], dims=node.get("dims"))
        if op == "unsqueeze":
            return value.unsqueeze(node["dim"])
        if op == "squeeze":
            return value.squeeze(node.get("dim"))
        if op == "narrow":
            return value.narrow(node["dim"], node["start"], node["length"])
        if op == "repeat":
            return value.repeat(tuple(node["sizes"]))
        return value.logical_not()
    if op.startswith("compare_"):
        left = _build(node["operand"], device)
        right = _build(node["rhs"], device) if "rhs" in node else node["value"]
        return {
            "compare_gt": left.__gt__,
            "compare_ge": left.__ge__,
            "compare_lt": left.__lt__,
            "compare_le": left.__le__,
            "compare_eq": left.eq,
            "compare_ne": left.ne,
        }[op](right)
    if op in {"logical_and", "logical_or"}:
        left = _build(node["operand"], device)
        right = _build(node["rhs"], device) if "rhs" in node else node["value"]
        return (left & right) if op == "logical_and" else (left | right)
    if op == "mask_index":
        return _build(node["operand"], device)[_build(node["mask"], device)]
    if op == "masked_fill":
        return _build(node["operand"], device).masked_fill(
            _build(node["mask"], device), node["value"]
        )
    if op == "where":
        return torch.where(
            _build(node["cond"], device),
            _build(node["x"], device) if isinstance(node["x"], dict) else node["x"],
            _build(node["y"], device) if isinstance(node["y"], dict) else node["y"],
        )
    if op in {"eye", "linspace"}:
        dtype = getattr(torch, node["dtype"])
        if op == "eye":
            return torch.eye(node["shape"][0], node["shape"][1], dtype=dtype, device=device)
        return torch.linspace(
            node["start"], node["end"], node["shape"][0], dtype=dtype, device=device
        )
    if op == "arange":
        return torch.arange(
            node["start"],
            node["stop"],
            step=node.get("step", 1),
            dtype=getattr(torch, node.get("dtype", "int64")),
            device=device,
        )
    raise ValueError(f"unsupported build op: {op!r}")
'''


# ---------------------------------------------------------------------------
# Definition and oracle generation.
# ---------------------------------------------------------------------------


def _reference_function(reference_source: str, op_name: str) -> ast.FunctionDef:
    tree = ast.parse(reference_source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    if "reference" not in functions:
        raise ValueError(f"{op_name}: reference_torch.py has no reference()")
    return functions["reference"]


def _signature_hints(spec_md: str) -> dict[str, str]:
    """Parse the spec.md signature line into parameter type hints."""
    match = re.search(r"\*\*Signature\*\*:.*?`([^`]*)`", spec_md, re.DOTALL)
    if not match:
        return {}
    signature = match.group(1).strip()
    function = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", signature)
    if function:
        signature = signature[function.end() :]
    arrow = signature.find("->")
    if arrow != -1:
        signature = signature[:arrow]
    signature = signature.strip()
    if signature.endswith(")"):
        signature = signature[:-1]
    signature = signature.strip()
    hints: dict[str, str] = {}
    for segment in _split_signature_arguments(signature):
        segment = segment.strip()
        if not segment:
            continue
        parameter = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)", segment)
        if parameter:
            hints[parameter.group(1)] = parameter.group(2).strip()
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", segment):
            hints[segment] = ""
    return hints


def _split_signature_arguments(signature: str) -> list[str]:
    segments: list[str] = []
    depth = 0
    current: list[str] = []
    for char in signature:
        if char in "[<(":
            depth += 1
        elif char in "])>":
            depth -= 1
        if char == "," and depth == 0:
            segments.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        segments.append("".join(current))
    return segments


def _output_names(spec_md: str, reference: ast.FunctionDef) -> list[str]:
    """Output names from the spec return signature when possible, else
    ``output_0...`` by the reference's returned tuple arity."""
    return_arity = 1
    for node in ast.walk(reference):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            return_arity = len(node.value.elts)
            break
    match = re.search(r"->\s*(.+)$", spec_md, re.MULTILINE)
    names: list[str] = []
    if match:
        returns = match.group(1).strip()
        if returns.startswith("("):
            inside = returns[1:]
            depth = 0
            for i, char in enumerate(inside):
                if char in "[<(":
                    depth += 1
                elif char in "])>":
                    depth -= 1
                elif char == ")" and depth == 0:
                    inside = inside[:i]
                    break
            for segment in _split_signature_arguments(inside):
                name = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", segment)
                if name:
                    names.append(name.group(1))
    if names:
        # The spec return signature may name more outputs than the reference
        # actually returns (source spec typos); the reference's tuple arity
        # is authoritative, so truncate (and pad) to it.
        names = names[:return_arity]
        while len(names) < return_arity:
            names.append(f"output_{len(names)}")
        return names
    return [f"output_{index}" for index in range(return_arity)]


def _literal_default(node: ast.expr, *, op_name: str, parameter: str) -> Any:
    # Torch dtype/device defaults normalize to their bare string name per the
    # v6.2 Definition contract (``torch.int8`` -> ``"int8"``).
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
    ):
        return node.attr
    try:
        value = ast.literal_eval(node)
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{op_name}: default for {parameter!r} is not JSON-literal"
        ) from exc
    return value


def _parameters(
    op_name: str, run: ast.FunctionDef, hints: dict[str, str]
) -> list[dict[str, Any]]:
    arguments = run.args
    if arguments.kwarg is not None:
        raise ValueError(f"{op_name}: **kwargs ABI is unsupported")

    positional = arguments.posonlyargs + arguments.args
    positional_defaults = [None] * (len(positional) - len(arguments.defaults)) + list(
        arguments.defaults
    )
    parameters: list[dict[str, Any]] = []
    for index, argument in enumerate(positional):
        default = positional_defaults[index]
        parameter: dict[str, Any] = {
            "name": argument.arg,
            "kind": (
                "positional_only"
                if index < len(arguments.posonlyargs)
                else "positional_or_keyword"
            ),
            "required": default is None,
        }
        if default is not None:
            parameter["default"] = _literal_default(
                default, op_name=op_name, parameter=argument.arg
            )
        hint = hints.get(argument.arg)
        if hint:
            parameter["type_hint"] = hint
        parameters.append(parameter)

    if arguments.vararg is not None:
        parameter = {
            "name": arguments.vararg.arg,
            "kind": "var_positional",
            "required": True,
        }
        hint = hints.get(arguments.vararg.arg)
        if hint:
            parameter["type_hint"] = hint
        parameters.append(parameter)

    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        parameter = {
            "name": argument.arg,
            "kind": "keyword_only",
            "required": default is None,
        }
        if default is not None:
            parameter["default"] = _literal_default(
                default, op_name=op_name, parameter=argument.arg
            )
        hint = hints.get(argument.arg)
        if hint:
            parameter["type_hint"] = hint
        parameters.append(parameter)
    return parameters


def _rename_reference(source: str, node: ast.FunctionDef) -> str:
    lines = source.splitlines()
    index = node.lineno - 1
    pattern = re.compile(r"^(\s*def\s+)reference(\s*\()")
    lines[index], count = pattern.subn(r"\g<1>run\g<2>", lines[index], count=1)
    if count != 1:
        raise ValueError(f"cannot rename reference() at line {node.lineno}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-import resolution for reference_torch.py.
# ---------------------------------------------------------------------------


def _inline_reference_source(source: str, source_root: Path, seen: set[Path]) -> str:
    """Inline ``from problems.<group>.<op>.reference_torch import ...``
    imports into the reference source so the oracle is self-contained."""

    tree = ast.parse(source)
    new_lines: list[str] = []
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("problems.")
        ):
            module_path = node.module.split(".")
            if not (
                len(module_path) == 4
                and module_path[0] == "problems"
                and module_path[3] == "reference_torch"
            ):
                raise ValueError(f"unsupported reference import: {node.module}")
            _, group, op, _ = module_path
            target = source_root / "problems" / group / op / "reference_torch.py"
            if target.resolve() in seen:
                raise ValueError(f"cyclic reference import: {target}")
            seen.add(target.resolve())
            imported_source = target.read_text(encoding="utf-8")
            imported_source = _inline_reference_source(imported_source, source_root, seen)
            for alias in node.names:
                if alias.name != "reference":
                    continue
                as_name = alias.asname or "reference"
                new_lines.append(
                    _closure_chunks(imported_source, ["reference"], {("reference", as_name)})
                )
            continue
        new_lines.append(ast.get_source_segment(source, node) or "")
    return "\n".join(new_lines)


def _body_name_refs(function: ast.FunctionDef) -> set[str]:
    """Module-level names a function body references (excludes its own
    arguments and locals)."""
    names: set[str] = set()
    arguments = {
        arg.arg
        for arg in function.args.posonlyargs + function.args.args + function.args.kwonlyargs
    }
    if function.args.vararg is not None:
        arguments.add(function.args.vararg.arg)
    if function.args.kwarg is not None:
        arguments.add(function.args.kwarg.arg)
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in arguments:
                names.add(node.id)
    return names


def _closure_chunks(
    source: str,
    entry_names: Sequence[str],
    renames: set[tuple[str, str]],
) -> str:
    """Extract entry functions plus the module-level functions/constants they
    transitively reference, as source text.  ``renames`` maps source name to
    emitted name."""
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assigns: dict[str, ast.Assign] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assigns[node.targets[0].id] = node
    renames_map = dict(renames)
    emitted: set[str] = set()
    chunks: list[str] = []

    def emit(name: str) -> None:
        if name in emitted:
            return
        if name in functions:
            emitted.add(name)
            for referenced in sorted(_body_name_refs(functions[name])):
                if referenced in functions or referenced in assigns:
                    emit(referenced)
            segment = ast.get_source_segment(source, functions[name])
            assert segment is not None
            new_name = renames_map.get(name)
            if new_name is not None:
                pattern = re.compile(rf"^(\s*def\s+){re.escape(name)}(\s*\()")
                segment, count = pattern.subn(rf"\g<1>{new_name}\g<2>", segment, count=1)
                assert count == 1, f"cannot rename {name}()"
            chunks.append(segment)
        elif name in assigns:
            emitted.add(name)
            segment = ast.get_source_segment(source, assigns[name])
            assert segment is not None
            chunks.append(segment)

    for name in entry_names:
        emit(name)
    return "\n".join(chunks)


def _extract_function_source(source: str, name: str, new_name: str | None = None) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            if new_name is not None:
                pattern = re.compile(rf"^(\s*def\s+){re.escape(name)}(\s*\()")
                segment, count = pattern.subn(rf"\g<1>{new_name}\g<2>", segment, count=1)
                assert count == 1, f"cannot rename {name}()"
            return segment
    raise ValueError(f"no function {name} in reference source")


# ---------------------------------------------------------------------------
# Custom-check embedding for valid.
# ---------------------------------------------------------------------------

_VALID_TEMPLATE = '''


def valid(ref_outputs, sol_outputs, inputs, ctx):
    try:
        if {multi}:
            _source_check(tuple(ref_outputs), tuple(sol_outputs))
        else:
            _source_check(ref_outputs[0], sol_outputs[0])
    except AssertionError as exc:
        return {{"passed": False, "message": str(exc), "metrics": {{}}}}
    return {{"passed": True, "message": "", "metrics": {{}}}}


'''


def _embedded_gen_inputs_source(
    parameter_names: Sequence[str], constructor_names: Sequence[str]
) -> str:
    """gen_inputs that rebuilds a case by calling the embedded constructor
    with the per-workload ``_case_args`` context."""
    names = "{" + ", ".join(repr(name) for name in parameter_names) + "}"
    dispatcher: list[str] = []
    for original in constructor_names:
        target = _constructor_alias(original)
        dispatcher.append(
            f'    if constructor == {original!r}:\n'
            f"        built = {target}(*args, **kwargs)"
        )
    dispatch = "\n".join(dispatcher) or "    built = None"
    return f'''


_DEVICE = None


def _materialize_arg(value, device):
    if isinstance(value, dict) and "__tensor__" in value:
        dtype = value.get("dtype", "float32")
        return torch.tensor(
            value["__tensor__"],
            dtype=getattr(torch, dtype),
            device=device,
        )
    return value


def gen_inputs(ctx, device):
    global _DEVICE
    _DEVICE = device
    case_args = ctx["inputs"].get("_case_args", {{}})
    args = [_materialize_arg(value, device) for value in case_args.get("args", [])]
    kwargs = {{
        key: _materialize_arg(value, device)
        for key, value in case_args.get("kwargs", {{}}).items()
    }}
    dtype = kwargs.get("dtype")
    if isinstance(dtype, str) and dtype:
        kwargs["dtype"] = getattr(torch, dtype)
    constructor = case_args.get("constructor", "_case")
{dispatch}
    parameters = {names}
    return {{
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }}
'''


def _constructor_alias(original: str) -> str:
    if original == "_case":
        return "_build_case"
    return f"_build_case_{original.lstrip('_')}"


def _embedded_workloads(
    op_name: str,
    phase: str,
    case_args: Sequence[Mapping[str, Any]],
    constructor: Any,
) -> list[dict[str, Any]]:
    """Build workload rows for the embedded-constructor path: the case is
    replayed by gen_inputs from its constructor call args."""
    signature = inspect.signature(constructor)
    positional = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    workloads: list[dict[str, Any]] = []
    for index, case in enumerate(case_args):
        bound = dict(zip(positional, case.get("args", []), strict=False))
        bound.update(case.get("kwargs", {}))
        tolerance = _embedded_tolerance(bound)
        workload: dict[str, Any] = {
            "name": f"{op_name}-{index:05d}-{phase}",
            "inputs": {"_case_args": case},
            "seed": int(bound.get("seed", 0)),
        }
        if tolerance is not None:
            workload["tolerance"] = tolerance
        workloads.append(workload)
    return workloads


def _embedded_tolerance(bound: Mapping[str, Any]) -> dict[str, Any] | None:
    for name in ("dtype", "out_dtype", "output_dtype", "in_dtype"):
        dtype = bound.get(name)
        if isinstance(dtype, str) and dtype:
            floating = dtype not in {"int8", "int16", "int32", "int64", "uint8", "bool"}
            if floating:
                return _tolerance_for_dtype(dtype)
    return None


def _embed_functions(
    entries: list[tuple[Any, str]],
    source_root: Path,
    seen: set[Path],
    *,
    include_harness: bool,
    substitute_device: bool = False,
) -> str:
    """Emit self-contained source for ``(function, rename)`` entries plus
    their transitive module-level helpers (and the harness ``assert_close``
    helpers when requested).  Deduplicates across calls via ``seen``."""
    helper_chunks: list[str] = []
    emitted: set[tuple[Path, str]] = set()
    pending: list[tuple[Any, str | None]] = []

    def enqueue(function: Any, rename_to: str | None = None) -> None:
        identity = (Path(function.__code__.co_filename).resolve(), function.__name__)
        if identity in emitted:
            return
        emitted.add(identity)
        pending.append((function, rename_to))

    def drain() -> None:
        while pending:
            function, rename_to = pending.pop(0)
            # Module-level constants the function references (e.g. ``_EPS``,
            # ``_FP8_DTYPE``, ``_BLOCK_E`` in the source cases.py).
            constant_lines: list[str] = []
            for name in _function_name_refs(function):
                value = function.__globals__.get(name)
                if value is None or callable(value):
                    continue
                if isinstance(value, DType):
                    constant_lines.append(f"{name} = torch.{value}")
                    continue
                try:
                    json.dumps(value)
                except TypeError:
                    continue
                constant_lines.append(f"{name} = {json.dumps(value)}")
            if constant_lines:
                helper_chunks.append("\n".join(constant_lines))
            # Same-file helpers the function references.
            for name in _function_name_refs(function):
                candidate = function.__globals__.get(name)
                if (
                    callable(candidate)
                    and hasattr(candidate, "__code__")
                    and Path(candidate.__code__.co_filename).resolve()
                    == Path(function.__code__.co_filename).resolve()
                ):
                    enqueue(candidate)
                elif isinstance(candidate, type):
                    _emit_class_module(candidate)
            # Closure-captured values (factory params) become module bindings.
            freevars = function.__code__.co_freevars
            if freevars:
                closure = function.__closure__ or ()
                binds: list[str] = []
                for name, cell in zip(freevars, closure, strict=True):
                    value = cell.cell_contents
                    binds.append(f"{name} = {_closure_binding(value)}")
                if binds:
                    helper_chunks.append("\n".join(binds))
            body = textwrap.dedent(inspect.getsource(function))
            if substitute_device:
                body = re.sub(
                    r'device\s*=\s*torch\.device\("cuda"\)', "device=_DEVICE", body
                )
                body = re.sub(r'device\s*=\s*"cuda"', "device=_DEVICE", body)
                # ``torch.empty`` case buffers are partially written by the
                # source constructors, leaving uninitialized memory that is
                # non-deterministic and can contain NaNs; v6.2 requires
                # deterministic construction, so zero-initialize instead.
                body = re.sub(r"\btorch\.empty\(", "torch.zeros(", body)
            if rename_to is not None and function.__name__ != rename_to:
                pattern = re.compile(
                    rf"^(\s*def\s+){re.escape(function.__name__)}(\s*\()"
                )
                body, count = pattern.subn(rf"\g<1>{rename_to}\g<2>", body, count=1)
                assert count == 1
            helper_chunks.append(body)

    for function, rename in entries:
        enqueue(function, rename)
    drain()

    if include_harness:
        harness_path = source_root / "harness" / "correctness.py"
        if harness_path.resolve() not in seen:
            seen.add(harness_path.resolve())
            harness_source = harness_path.read_text(encoding="utf-8")
            for name in ("assert_close", "tolerance_for"):
                emitted.add((harness_path.resolve(), name))
                helper_chunks.append(
                    _extract_function_source(harness_source, name, name)
                )
            harness_tree = ast.parse(harness_source)
            for node in harness_tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in {"_TOLERANCES", "_DEFAULT_TOLERANCE"}
                ):
                    helper_chunks.append(ast.get_source_segment(harness_source, node) or "")

    # Cross-module helpers (e.g. ``from problems.X.cases import _decode`` or
    # ``from problems.X.reference_torch import reference as helper``) are
    # emitted under the name the entry references them by.
    for function, _ in entries:
        for name in _function_name_refs(function):
            candidate = function.__globals__.get(name)
            if not callable(candidate):
                continue
            if not hasattr(candidate, "__code__"):
                continue
            path = Path(candidate.__code__.co_filename).resolve()
            if path == Path(function.__code__.co_filename).resolve():
                continue
            if str(path).startswith(str(source_root.resolve())):
                if isinstance(candidate, type):
                    _emit_class_module(candidate)
                else:
                    enqueue(candidate, rename_to=name)
    drain()
    return "\n".join(helper_chunks)


def _emit_class_module(cls: type) -> None:
    """Emit the whole (small, self-contained) module that defines a helper
    class such as the LoRA ``LoRABatchInfo`` dataclass, so decorators and
    defaults survive."""
    module = inspect.getmodule(cls)
    if module is None or not module.__file__:
        return
    key = (Path(module.__file__).resolve(), "<module>")
    if key in _EMITTED_CLASS_MODULES:
        return
    _EMITTED_CLASS_MODULES.add(key)
    lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
    # ``from __future__`` imports are only legal at file top; strip them.
    lines = [
        line for line in lines if not line.startswith("from __future__ import")
    ]
    _CLASS_MODULE_CHUNKS.append("\n".join(lines))


_EMITTED_CLASS_MODULES: set[tuple[Path, str]] = set()
_CLASS_MODULE_CHUNKS: list[str] = []


def _function_name_refs(function: Any) -> list[str]:
    """Names referenced by a function's body that may be helpers."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return sorted(names)


def _closure_binding(value: Any) -> str:
    """Python expression reproducing a closure-captured value (scalar or
    constant-valued case tensor) inside the oracle."""
    if isinstance(value, TensorStub):
        values = _constant_values(value)
        if values is None:
            raise ValueError(f"unsupported closure tensor: {value!r}")
        dtype = _dtype_of(value) or "float32"
        return (
            f"torch.tensor({json.dumps(values)}, "
            f"dtype=getattr(torch, {json.dumps(dtype)}))"
        )
    try:
        json.dumps(value)
    except TypeError as exc:
        raise ValueError(f"unsupported closure value: {value!r}") from exc
    return json.dumps(value)


def _embed_checker(
    check_fn: Any,
    source_root: Path,
    seen: set[Path],
) -> tuple[str, str]:
    """Return (embedded helper source, valid source) that reproduces the
    source custom ``_check`` at v6.2 runtime.  ``check_fn`` is the real
    function object obtained from stub execution."""
    helpers = _embed_functions([(check_fn, "_source_check")], source_root, seen, include_harness=True)
    valid = _VALID_TEMPLATE.format(multi="len(ref_outputs) > 1")
    return helpers, valid


# ---------------------------------------------------------------------------
# Oracle assembly.
# ---------------------------------------------------------------------------


def _assemble_oracle(
    op_name: str,
    reference_source: str,
    reference_node: ast.FunctionDef,
    source_root: Path,
    *,
    needs_gen_inputs: bool,
    check_fn: Any | None,
    constructor: Any | None = None,
    parameter_names: Sequence[str] | None = None,
    extra_imports: str | None = None,
    wrap_single_return: bool = False,
) -> str:
    global _EMITTED_CLASS_MODULES, _CLASS_MODULE_CHUNKS
    _EMITTED_CLASS_MODULES = set()
    _CLASS_MODULE_CHUNKS = []
    body = _rename_reference(reference_source, reference_node)
    body = _inline_reference_source(body, source_root, set())
    if wrap_single_return:
        body = _wrap_single_return(body, parameter_names or [])

    parts: list[str] = ["REFERENCE_DEVICE = 'target'\n"]
    # ``import torch`` is harmless to repeat; only skip it when a standalone
    # binding already exists (``import torch.nn.functional as F`` does NOT
    # bind the name ``torch``).
    binds_torch = any(
        line == "import torch"
        or (line.startswith("import torch.") and " as " not in line)
        or line.startswith("from torch")
        for line in body.splitlines()
    )
    if not binds_torch:
        parts.append("import torch")
    parts.append(body.rstrip())
    if needs_gen_inputs:
        parts.append(_GEN_INPUTS_TEMPLATE)
    if constructor is not None:
        assert parameter_names is not None
        entries = [
            (function, _constructor_alias(function.__name__))
            for function in constructor
        ]
        helpers = _embed_functions(
            entries,
            source_root,
            set(),
            include_harness=False,
            substitute_device=True,
        )
        if extra_imports:
            parts.append(extra_imports)
        parts.append(helpers)
        parts.append(
            _embedded_gen_inputs_source(
                parameter_names, [function.__name__ for function in constructor]
            )
        )
    if check_fn is not None:
        helpers, valid = _embed_checker(check_fn, source_root, set())
        parts.append(helpers)
        parts.append(valid)
    if _CLASS_MODULE_CHUNKS:
        parts.append("\n".join(_CLASS_MODULE_CHUNKS))
    oracle = _ensure_oracle_imports("\n".join(parts) + "\n")
    ast.parse(oracle, filename=f"{op_name}/oracle.py")
    return oracle


_STDLIB_IMPORTS = (
    "math",
    "functools",
    "itertools",
    "collections",
    "operator",
    "typing",
    "dataclasses",
    "copy",
    "re",
    "enum",
    "sys",
    "os",
    "random",
)


def _wrap_single_return(body: str, parameter_names: Sequence[str]) -> str:
    """Normalize a reference whose return is conditionally 1 or 2 values to a
    fixed 2-tuple: rename the original ``run`` to ``_run_impl`` and add a new
    ``run`` that returns ``(value, None)`` for the single-value path."""
    tree = ast.parse(body)
    run = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"),
        None,
    )
    if run is None:
        raise ValueError("cannot wrap run: not found")
    lines = body.splitlines()
    index = run.lineno - 1
    pattern = re.compile(r"^(\s*def\s+)run(\s*\()")
    lines[index], count = pattern.subn(r"\g<1>_run_impl\g<2>", lines[index], count=1)
    assert count == 1
    signature = ", ".join(parameter_names)
    wrapper = (
        "\n\n\ndef run("
        + signature
        + "):\n"
        + "    result = _run_impl("
        + signature
        + ")\n"
        + "    if isinstance(result, tuple):\n"
        + "        return result\n"
        + "    return result, None\n"
    )
    return "\n".join(lines) + wrapper


def _ensure_oracle_imports(oracle: str) -> str:
    """Add stdlib / torch.nn.functional imports that embedded reference or
    constructor bodies may reference but whose original import lines were
    dropped."""
    missing: list[str] = []
    for module in _STDLIB_IMPORTS:
        if re.search(rf"\b{module}\.", oracle) and not re.search(
            rf"^import {module}(\.|\s|$)", oracle, re.MULTILINE
        ):
            missing.append(f"import {module}")
    if re.search(r"\bF\.", oracle) and not re.search(
        r"^import torch\.nn\.functional", oracle, re.MULTILINE
    ):
        missing.append("import torch.nn.functional as F")
    if not missing:
        return oracle
    marker = "REFERENCE_DEVICE = 'target'\n"
    if marker in oracle:
        return oracle.replace(marker, marker + "\n".join(missing) + "\n", 1)
    return "\n".join(missing) + "\n" + oracle


def _oracle_needs_gen_inputs(workloads: Sequence[Mapping[str, Any]]) -> bool:
    for workload in workloads:
        for spec in workload["inputs"].values():
            if isinstance(spec, dict) and spec.get("type") == "custom":
                return True
    return False


def _custom_check_fn(cases: Sequence[Mapping[str, Any]], source_root: Path) -> Any | None:
    """Return the source custom ``_check`` for the correctness cases, or None
    when it is just the harness ``assert_close`` (the workload tolerance
    already reproduces it)."""
    harness_path = (source_root / "harness" / "correctness.py").resolve()
    for case in cases:
        check = case.get("check")
        if check is None:
            continue
        if isinstance(check, functools.partial):
            func = check.func
            if (
                callable(func)
                and Path(func.__code__.co_filename).resolve() == harness_path
            ):
                continue
            raise ValueError(
                f"unsupported partial check: {check!r} (not harness assert_close)"
            )
        if callable(check):
            path = Path(check.__code__.co_filename).resolve()
            if path == harness_path:
                continue
        return check
    return None


# ---------------------------------------------------------------------------
# Converter driver.
# ---------------------------------------------------------------------------


def _problem_paths(source_root: Path, groups: Sequence[str]) -> list[Path]:
    problems_root = source_root / "problems"
    paths: list[Path] = []
    for group in groups:
        group_root = problems_root / group
        if not group_root.is_dir():
            raise ValueError(f"unknown problem group: {group}")
        for problem_dir in sorted(group_root.iterdir()):
            if not problem_dir.is_dir() or problem_dir.name.startswith("."):
                continue
            if not (problem_dir / "spec.md").is_file():
                continue
            if (group, problem_dir.name) in SKIPPED_PROBLEMS:
                continue
            paths.append(problem_dir)
    return paths


def convert(
    source_root: Path,
    output_root: Path,
    *,
    groups: Sequence[str],
    force: bool = False,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and not force:
        raise FileExistsError(f"output exists (pass --force): {output_root}")

    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=parent))
    stubs, finder = _install_stubs(source_root)
    operators: list[dict[str, Any]] = []
    problem_paths = _problem_paths(source_root, groups)
    # Cross-group duplicate problem names get a group prefix so Definition
    # names stay globally unique.
    name_counts: dict[str, int] = {}
    for problem_dir in problem_paths:
        name_counts[problem_dir.name] = name_counts.get(problem_dir.name, 0) + 1
    try:
        for problem_dir in problem_paths:
            group = problem_dir.parent.name
            op_name = f"sglang_{problem_dir.name}"
            if name_counts[problem_dir.name] > 1:
                op_name = f"sglang_{group}_{problem_dir.name}"
            operator_root = temporary / "ops" / group / op_name
            operator_root.mkdir(parents=True, exist_ok=True)

            spec_md = (problem_dir / "spec.md").read_text(encoding="utf-8")
            reference_source = (problem_dir / "reference_torch.py").read_text(
                encoding="utf-8"
            )
            reference = _reference_function(reference_source, op_name)
            hints = _signature_hints(spec_md)
            parameters = _parameters(op_name, reference, hints)
            parameter_names = [parameter["name"] for parameter in parameters]
            outputs = _output_names(spec_md, reference)
            problem_key = (group, problem_dir.name)
            fix = PROBLEM_FIXES.get(problem_key, {})
            if "outputs" in fix:
                outputs = list(fix["outputs"])

            definition = {
                "api_version": TARGET_API_VERSION,
                "name": op_name,
                "description": _describe(spec_md, op_name),
                "parameters": parameters,
                "outputs": outputs,
                "effects": {"mutates": [], "returns_alias_of": {}},
            }
            _write_json(operator_root / "definition.json", definition)

            correctness_cases = timing_cases = None
            embedded_plan = None
            try:
                correctness_cases, timing_cases = _extract_case_lists(
                    problem_dir / "cases.py", stubs
                )
                for phase_cases in (correctness_cases, timing_cases):
                    _case_to_workloads(
                        op_name,
                        "correctness",
                        phase_cases,
                        parameter_names=parameter_names,
                    )
            except (CaseExtractionError, ValueError):
                embedded_plan = _embedded_case_plan(
                    problem_dir / "cases.py", source_root, stubs
                )

            counts = {}
            constructor: Any = None
            extra_imports: str | None = None
            try:
                if embedded_plan is not None:
                    (
                        correctness_args,
                        timing_args,
                        check_fn,
                        constructors,
                        extra_imports,
                    ) = embedded_plan
                    if not correctness_args or not timing_args or not constructors:
                        raise ValueError(
                            f"{op_name}: embedded case plan produced no cases"
                        )
                    constructor = constructors
                    for phase, case_args in (
                        ("correctness", correctness_args),
                        ("timing", timing_args),
                    ):
                        rows = _embedded_workloads(
                            op_name, phase, case_args, constructor[0]
                        )
                        _apply_fix_rows(rows, fix)
                        _write_jsonl(operator_root / f"{phase}.jsonl", rows)
                        counts[f"num_{phase}_workloads"] = len(rows)
                else:
                    assert correctness_cases is not None and timing_cases is not None
                    for phase, cases in (
                        ("correctness", correctness_cases),
                        ("timing", timing_cases),
                    ):
                        rows = _case_to_workloads(
                            op_name, phase, cases, parameter_names=parameter_names
                        )
                        if not rows:
                            raise ValueError(f"{op_name}: empty {phase} workload list")
                        _apply_fix_rows(rows, fix)
                        _write_jsonl(operator_root / f"{phase}.jsonl", rows)
                        counts[f"num_{phase}_workloads"] = len(rows)
                    check_fn = _custom_check_fn(correctness_cases, source_root)
            except Exception as exc:
                raise ValueError(f"{op_name}: {type(exc).__name__}: {exc}") from exc

            needs_gen_inputs = _oracle_needs_gen_inputs(
                [row for phase in ("correctness", "timing")
                 for row in _read_jsonl(operator_root / f"{phase}.jsonl")]
            )
            if check_fn is not None:
                harness_path = (source_root / "harness" / "correctness.py").resolve()
                if Path(check_fn.__code__.co_filename).resolve() == harness_path:
                    check_fn = None
            oracle = _assemble_oracle(
                op_name,
                reference_source,
                reference,
                source_root,
                needs_gen_inputs=needs_gen_inputs,
                check_fn=check_fn,
                constructor=constructor,
                parameter_names=parameter_names,
                extra_imports=extra_imports,
                wrap_single_return=bool(fix.get("wrap_single_return")),
            )
            (operator_root / "oracle.py").write_text(oracle, encoding="utf-8")

            operators.append({"name": op_name, "group": group, **counts})

        counts = {
            "operators": len(operators),
            "correctness_workloads": sum(
                int(item["num_correctness_workloads"]) for item in operators
            ),
            "timing_workloads": sum(
                int(item["num_timing_workloads"]) for item in operators
            ),
        }
        manifest = {
            "api_version": TARGET_API_VERSION,
            "name": CATALOG_NAME,
            "evaluator": "native",
            "layout": "per-operator",
            "source_format": "kernel-comp-baseline-problems",
            "source_revision": SOURCE_REVISION,
            "generated_by": "tools/convert_kernelcomp_baseline_v62.py",
            "counts": counts,
            "operators": operators,
        }
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "README.md").write_text(
            "# Kernel-comp baseline v6.2 native catalog\n\n"
            "Deterministically converted from the kernel-competition baseline "
            "repository at commit `%s`. The frozen SGLang kernels behind it are "
            "pinned by the source repository's `baseline_freeze.json`. Each "
            "operator's `oracle.py` is the problem's pure-torch "
            "`reference_torch.py` (cross-imports inlined); the source "
            "`correctness.jsonl`/`timing.jsonl` express the shared `cases.py` "
            "grids as declarative v6.2 recipes or the custom build-DSL, with "
            "the harness per-dtype tolerance written explicitly. Complete "
            "grids are retained as `*_full.jsonl`; active files are capped at "
            f"{WORKLOAD_LIMIT} per phase with `{ALGORITHM}`.\n" % SOURCE_REVISION,
            encoding="utf-8",
        )
        manifest = sample_catalog(temporary, limit=WORKLOAD_LIMIT)

        backup: Path | None = None
        if output_root.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output_root.name}-backup-", dir=parent)
            )
            backup.rmdir()
            output_root.replace(backup)
        try:
            temporary.replace(output_root)
        except BaseException:
            if backup is not None and not output_root.exists():
                backup.replace(output_root)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _describe(spec_md: str, op_name: str) -> str:
    lines = [line.strip() for line in spec_md.splitlines() if line.strip()]
    title = lines[0].lstrip("#").strip() if lines else op_name
    match = re.search(r"\*\*Signature\*\*:.*?`([^`]*)`", spec_md, re.DOTALL)
    signature = f"; signature {match.group(1)}" if match else ""
    return f"Kernel-comp baseline {title}{signature}."


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert kernel-comp baseline problems to a native v6.2 catalog."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/kernelcomp-baseline"))
    parser.add_argument(
        "--groups",
        nargs="+",
        required=True,
        help="problem groups to convert (e.g. elementwise)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = convert(
        args.source, args.output, groups=args.groups, force=args.force
    )
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()

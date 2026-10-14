# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.triton_lang_utils import triton_constexpr as constexpr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fast path: same-shape, same-dtype, contiguous tensors are treated as flat
# 1-D vectors.  One program handles VIEW tiles per kernel launch, which
# amortises the per-program control overhead that dominates the current
# implementation on medium/large shapes, while still keeping the program
# count bounded (<= 1024) so that no launch ever has a million programs.
# ---------------------------------------------------------------------------

_MAX_PROGRAMS = 1024

# Per-program loop counts per (dtype-bytes, tile) selection.
_LOOPS_FP16_MED = 4
_LOOPS_FP16_BIG = 64
_LOOPS_FP32_MED = 4
_LOOPS_FP32_BIG = 16


@triton.jit
def _logaddexp2_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    TILE: tl.constexpr,
    VIEW: tl.constexpr,
    OUT_F32: tl.constexpr,
    IS_F32: tl.constexpr,
):
    pid = tl.program_id(0)
    span = TILE * VIEW
    base = pid * span
    offs = tl.arange(0, TILE)
    for j in tl.static_range(VIEW):
        idx = base + j * TILE + offs
        m = idx < n_elements
        x = tl.load(x_ptr + idx, mask=m, other=0.0)
        y = tl.load(y_ptr + idx, mask=m, other=0.0)
        if IS_F32:
            x_f32 = x
            y_f32 = y
        else:
            x_f32 = x.to(tl.float32)
            y_f32 = y.to(tl.float32)
        mx = tl.maximum(x_f32, y_f32)
        delta = x_f32 - y_f32
        d = tl.abs(delta)
        # delta is NaN exactly when x and y are equal infinities (inf - inf).
        nan_delta = delta != delta
        # Saturating the exponent keeps exp2(-d) well behaved (0) instead of
        # relying on the tail of the exp2 approximation; max is only applied
        # when delta is finite, so NaN never reaches exp2.
        d_safe = tl.where(nan_delta, 0.0, tl.minimum(d, 200.0))
        corr = tl.log2(1.0 + tl.exp2(-d_safe))
        res = mx + corr
        # Recover max(x, y) when delta was NaN, propagate real NaNs.
        res = tl.where(nan_delta, mx, res)
        is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
        res = tl.where(is_nan, float("nan"), res)
        if OUT_F32:
            tl.store(out_ptr + idx, res, mask=m)
        else:
            tl.store(out_ptr + idx, res.to(out_ptr.dtype.element_ty), mask=m)


# ---------------------------------------------------------------------------
# Generic fallback: keeps broadcast / non-contiguous / type-promotion semantics
# via pointwise_dynamic (stable fallback, unchanged formula).
# ---------------------------------------------------------------------------


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp2_func(x, y):
    # log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|)), m = max(x, y)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    # `delta` is NaN when x and y are equal infinities (inf - inf); the result
    # is then m, e.g. logaddexp2(inf, inf) = inf, logaddexp2(-inf, -inf) = -inf.
    res = tl.where(delta != delta, m, res)
    # Genuine NaN inputs must still propagate NaN.
    is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
    return tl.where(is_nan, float("nan"), res)


_PROMOTE = {torch.float16: torch.float32, torch.bfloat16: torch.float32}


def _flat_fast(x, y, out=None):
    """Return a flat-kernel result, or None if the fast path does not apply."""
    if x.shape != y.shape:
        return None
    if x.dtype != y.dtype:
        res_dtype = _PROMOTE.get(x.dtype)
        if res_dtype is None or _PROMOTE.get(y.dtype) != res_dtype:
            return None
    else:
        res_dtype = x.dtype
    if res_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if not x.is_contiguous() or not y.is_contiguous():
        return None
    n = x.numel()
    if n == 0:
        if out is None:
            out = torch.empty(x.shape, dtype=res_dtype, device=x.device)
        return out
    if x.device.type != "npu":
        return None

    if out is None:
        out = torch.empty(x.shape, dtype=res_dtype, device=x.device)
        out_f32 = res_dtype == torch.float32
    else:
        if out.shape != x.shape or out.dtype != res_dtype:
            return None
        if not out.is_contiguous():
            return None
        out_f32 = out.dtype == torch.float32

    # Empty tiles (VIEW=1, TILE=0) are rejected: the log2/exp2 path in the
    # generic kernel would otherwise be emitted for a degenerate block.
    if res_dtype == torch.float32:
        if n >= 262144:
            TILE, VIEW = 4096, _LOOPS_FP32_BIG
        else:
            TILE, VIEW = 4096, _LOOPS_FP32_MED
    else:
        if n >= 262144:
            TILE, VIEW = 4096, _LOOPS_FP16_BIG
        else:
            TILE, VIEW = 8192, _LOOPS_FP16_MED

    span = TILE * VIEW
    if n < span:
        # Small inputs: let the generic pointwise kernel keep the many small
        # programs it already launches well (avoids the 2x launch cliff).
        return None

    grid = min((n + span - 1) // span, _MAX_PROGRAMS)
    if grid < 1 or grid > 65535:
        return None

    _logaddexp2_flat_kernel[(grid,)](
        x,
        y,
        out,
        n,
        TILE=TILE,
        VIEW=VIEW,
        OUT_F32=out_f32,
        IS_F32=(res_dtype == torch.float32),
    )
    return out


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    res = _flat_fast(self, other)
    if res is not None:
        return res
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    res = _flat_fast(self, other, out)
    if res is not None:
        return res
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

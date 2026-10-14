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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flat contiguous fast path.  The generic pointwise_dynamic path is kept as a
# fallback for broadcasting / non-contiguous / dtype-promotion cases.
# ---------------------------------------------------------------------------

_MAX_PROGRAMS = 8192
_REMAINDER_BLOCK = 1024


@triton.jit
def _logaddexp2_core(x, y):
    # log2(2**x + 2**y) = m + ln(1 + exp2(-|x - y|)) * log2(e), m = max(x, y)
    # The max is computed in the input dtype, only the difference and the
    # correction term are promoted to fp32 (fewer live UB vectors).
    m = tl.maximum(x, y)
    delta = x.to(tl.float32) - y.to(tl.float32)
    t = tl.abs(delta)
    # Equal infinities give inf - inf = NaN for `delta`; clamping the argument
    # keeps the correction term at exactly 0 so the result falls back to m
    # (logaddexp2(inf, inf) = inf, logaddexp2(-inf, -inf) = -inf).
    t = tl.where(t <= 200.0, t, 200.0)
    corr = tl.log(1.0 + tl.exp2(-t)) * 1.4426950408889634
    return m + corr.to(m.dtype)


@triton.jit
def _logaddexp2_flat_contiguous(
    x_ptr,
    y_ptr,
    out_ptr,
    out_numel,
    BLOCK: tl.constexpr,
    LOOPS: tl.constexpr,
):
    pid = tl.program_id(0)
    first = pid * (BLOCK * LOOPS)
    for i in range(LOOPS):
        offs = first + i * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        tl.store(out_ptr + offs, _logaddexp2_core(x, y))


@triton.jit
def _logaddexp2_flat_strided(
    x_ptr,
    y_ptr,
    out_ptr,
    out_numel,
    BLOCK: tl.constexpr,
):
    # Grid-stride over the regular part, one masked tail iteration at the end.
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    step = num_prog * BLOCK
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    for _ in range(0, tl.cdiv(out_numel, step)):
        offs = tl.where(offs < out_numel, offs, 0)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        tl.store(out_ptr + offs, _logaddexp2_core(x, y))
        offs += step
    tail = tl.cdiv(out_numel, step) * step + pid * _REM_BLOCK
    tail_offs = tail + tl.arange(0, _REM_BLOCK)
    mask = tail_offs < out_numel
    if tl.min(mask.to(tl.int32), axis=0) == 0:
        x = tl.load(x_ptr + tail_offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + tail_offs, mask=mask, other=0.0)
        tl.store(out_ptr + tail_offs, _logaddexp2_core(x, y), mask=mask)


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


def _try_flat_contiguous(x, y):
    """Return the flat kernel result when a fast path applies, else None.

    Conditions: both inputs are non-empty, same shape, contiguous, same dtype
    and stored in fp16/bf16/fp32, and the output only needs that dtype (no
    type promotion).
    """
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        return None
    if x.shape != y.shape or x.numel() == 0:
        return None
    if not x.is_contiguous() or not y.is_contiguous():
        return None
    if x.dtype != y.dtype:
        return None
    # Lower-precision integer/bool operands would promote to a wider dtype.
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None

    n = x.numel()
    out = torch.empty_like(x)

    tries = []
    if n >= 2 * _MAX_PROGRAMS * 8192 * 4:
        # Very large: fewer, longer-running programs (rank-2 tile 4096 x 64).
        tries.append((4096, 64, 2))
        tries.append((4096, 4, 2))
    elif n >= 2 * _MAX_PROGRAMS * 8192:
        # Medium fp16/bf16: rank-2 tile 8192 x 4.
        tries.append((8192, 4, 1))
        tries.append((4096, 4, 2))
    else:
        tries.append((2048, 1, 1))
        tries.append((1024, 4, 1))

    for block, loops, stages in tries:
        span = block * loops
        if n % span == 0 and (n // span) <= _MAX_PROGRAMS:
            grid = (n // span,)
            try:
                _logaddexp2_flat_contiguous[(grid,)](
                    x, y, out, n, BLOCK=block, LOOPS=loops, num_stages=stages
                )
            except Exception:
                logger.debug("flat contiguous kernel failed", exc_info=True)
                continue
            return out

    grid = (_MAX_PROGRAMS,)
    try:
        _logaddexp2_flat_strided[(grid,)](
            x, y, out, n, BLOCK=4096, num_stages=2
        )
    except Exception:
        logger.debug("flat strided kernel failed", exc_info=True)
        return None
    return out


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    fast = _try_flat_contiguous(self, other)
    if fast is not None:
        return fast
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    if (
        isinstance(out, torch.Tensor)
        and isinstance(self, torch.Tensor)
        and isinstance(other, torch.Tensor)
        and out.dtype == self.dtype == other.dtype
        and out.shape == self.shape == other.shape
        and out.is_contiguous()
        and self.is_contiguous()
        and other.is_contiguous()
        and out.numel() > 0
    ):
        n = out.numel()
        grid = (min(triton.cdiv(n, 4096), _MAX_PROGRAMS),)
        try:
            if n % 4096 == 0 and grid[0] * 4096 == n:
                _logaddexp2_flat_contiguous[(grid,)](
                    self, other, out, n, BLOCK=4096, LOOPS=1, num_stages=2
                )
            else:
                _logaddexp2_flat_strided[(grid,)](
                    self, other, out, n, BLOCK=4096, num_stages=2
                )
            return out
        except Exception:
            logger.debug("flat contiguous out kernel failed", exc_info=True)
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

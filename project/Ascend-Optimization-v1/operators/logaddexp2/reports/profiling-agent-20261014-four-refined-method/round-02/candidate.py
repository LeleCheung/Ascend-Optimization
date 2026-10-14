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


@triton.jit
def _logaddexp2_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
    LOOPS: tl.constexpr,
):
    # Contiguous same-shape fast path: flat 1D addressing, no rank-2 index math.
    # Grid is bounded and every program walks LOOPS tiles with a grid-stride
    # step, so the whole output is covered. The tile loop is a runtime range
    # (not tl.static_range), so 64 iterations do not unroll 64 copies of the
    # exp2/log2/load/store sequence.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    span = BLOCK * LOOPS
    base = pid * span
    for i in range(0, LOOPS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        if pid * span + span <= n:
            # Fully covered tile: skip the mask entirely.
            x = tl.load(x_ptr + offs)
            y = tl.load(y_ptr + offs)
            m = tl.maximum(x, y)  # keep max in the input dtype
            t = x.to(tl.float32) - y.to(tl.float32)
            # |x-y| is NaN only for equal infinities (inf - inf); clamp the
            # exponent so 2**(-t) -> 0 and the result falls back to m. This is
            # exact for every finite magnitude (|x-y| > 200 underflows 2**-t).
            a = tl.where(t <= 200.0, t, 200.0)
            corr = tl.log(1.0 + tl.exp2(-tl.abs(a))) * 1.4426950408889634
            res = m + corr
            # Genuine NaN inputs must still propagate NaN (max would drop it).
            isnan = (t != t) | (m != m)
            res = tl.where(isnan, float("nan"), res)
            tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty))
        else:
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=1.0)
            y = tl.load(y_ptr + offs, mask=mask, other=1.0)
            m = tl.maximum(x, y)
            t = x.to(tl.float32) - y.to(tl.float32)
            a = tl.where(t <= 200.0, t, 200.0)
            corr = tl.log(1.0 + tl.exp2(-tl.abs(a))) * 1.4426950408889634
            res = m + corr
            isnan = (t != t) | (m != m)
            res = tl.where(isnan, float("nan"), res)
            tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


# (elements, BLOCK, LOOPS, num_stages); all grids stay far below 65535.
_LOOP_CONFIGS = (
    (1 << 20, 4096, 4, 2),
    (1 << 28, 4096, 64, 2),
    (1 << 62, 8192, 16, 2),
)


def _flat_launch(x, y, out):
    n = out.numel()
    block, loops, stages = _LOOP_CONFIGS[-1]
    for threshold, b, l, s in _LOOP_CONFIGS:
        if n <= threshold:
            block, loops, stages = b, l, s
            break
    span = block * loops
    grid = (max(1, (n + span - 1) // span),)
    _logaddexp2_flat_kernel[grid](
        x, y, out, n, BLOCK=block, LOOPS=loops, num_stages=stages
    )


def _fast_eligible(x, y):
    return (
        x.shape == y.shape
        and x.is_contiguous()
        and y.is_contiguous()
        and x.dtype == y.dtype
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if _fast_eligible(self, other):
        out = torch.empty_like(self)
        _flat_launch(self, other, out)
        return out
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    if _fast_eligible(self, other) and out.shape == self.shape and out.is_contiguous():
        _flat_launch(self, other, out)
        return out
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

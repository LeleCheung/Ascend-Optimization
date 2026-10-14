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
    EVEN: tl.constexpr,
    OUT_F32: tl.constexpr,
):
    # One program handles LOOPS consecutive tiles of BLOCK elements.  The grid
    # is bounded, so every program walks its tiles with a grid stride until the
    # whole flat range is covered; there is no trailing remainder either way
    # because `n` is always an exact multiple of BLOCK * LOOPS here.
    span = BLOCK * LOOPS
    stride = tl.num_programs(0) * span
    start = tl.program_id(0) * span
    arange = tl.arange(0, BLOCK)
    for base in range(start, n, stride):
        for i in range(LOOPS):
            offs = base + i * BLOCK + arange
            if EVEN:
                xv = tl.load(x_ptr + offs)
                yv = tl.load(y_ptr + offs)
            else:
                mask = offs < n
                xv = tl.load(x_ptr + offs, mask=mask, other=0.0)
                yv = tl.load(y_ptr + offs, mask=mask, other=0.0)
            # `max` stays in the input dtype; only the difference and the
            # correction term are promoted to FP32.
            m = tl.maximum(xv, yv)
            diff = xv.to(tl.float32) - yv.to(tl.float32)
            t = tl.abs(diff)
            # inf - inf produces a NaN difference for equal infinities: the
            # result is then m, so clamp the NaN to 0 and cap the magnitude at
            # 200 (2**-200 underflows in FP32, the correction collapses to 0).
            t = tl.where(t != t, 0.0, t)
            t = tl.minimum(t, 200.0)
            # log2(1 + 2**-t) = ln(1 + exp2(-t)) * log2(e); the `+1` stays
            # inside the log so there is no cancellation for small t.
            corr = tl.log(1.0 + tl.exp2(-t)) * 1.4426950408889634
            r = m.to(tl.float32) + corr
            # Genuine NaN inputs must still propagate NaN.
            is_nan = (xv != xv) | (yv != yv)
            r = tl.where(is_nan, float("nan"), r)
            if OUT_F32:
                rv = r
            else:
                rv = r.to(out_ptr.dtype.element_ty)
            if EVEN:
                tl.store(out_ptr + offs, rv)
            else:
                tl.store(out_ptr + offs, rv, mask=mask)


def _can_fast_path(x, y, out):
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        return False
    if x.device != y.device or x.dtype != y.dtype:
        return False
    if out.device != x.device or out.dtype != x.dtype:
        return False
    if not x.is_contiguous() or not y.is_contiguous() or not out.is_contiguous():
        return False
    if x.shape != y.shape or x.shape != out.shape:
        return False
    if x.numel() == 0:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


def _flat_config(numel, itemsize):
    # tile sizes follow the previously validated split for this operator:
    # a 8192-element tile for half precision, 4096 for FP32, and 4096 for the
    # very large tensors where UB pressure matters more than reuse.
    if itemsize == 4:
        block, target = 4096, 1
    else:
        block, target = 8192, 2
    loops = 1
    total = block
    # Keep at least 4x the number of physical AIV cores so that the grid can
    # still saturate the device for small and medium tensors.
    max_programs = max(4096, numel // (block * 4))
    while total * 2 <= numel and total < max_programs * block:
        total *= 2
        loops *= 2
    return block, loops


def _launch_flat(x, y, out, block, loops):
    n = out.numel()
    span = block * loops
    grid = (min(triton.cdiv(n, span), 65535),)
    even = (n % span) == 0
    out_f32 = out.dtype == torch.float32
    _logaddexp2_flat_kernel[grid](
        x, y, out, n, BLOCK=block, LOOPS=loops, EVEN=even, OUT_F32=out_f32
    )


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if isinstance(self, torch.Tensor) and isinstance(other, torch.Tensor):
        try:
            promoted = torch.promote_types(self.dtype, other.dtype)
        except TypeError:
            promoted = None
        if promoted is not None and promoted in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            # Reproduce the generic path's type promotion without going
            # through it, so the fast path sees exactly the same operands.
            xc = self if self.dtype == promoted else self.to(promoted)
            yc = other if other.dtype == promoted else other.to(promoted)
            if xc.dtype == yc.dtype and xc.numel() == yc.numel():
                out = torch.empty_like(xc)
                if _can_fast_path(xc, yc, out):
                    block, loops = _flat_config(out.numel(), out.element_size())
                    _launch_flat(xc, yc, out, block, loops)
                    return out
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    if isinstance(self, torch.Tensor) and isinstance(other, torch.Tensor):
        try:
            promoted = torch.promote_types(self.dtype, other.dtype)
        except TypeError:
            promoted = None
        if promoted is not None and out.dtype == promoted and out.element_size() in (
            2,
            4,
        ):
            xc = self if self.dtype == promoted else self.to(promoted)
            yc = other if other.dtype == promoted else other.to(promoted)
            if xc.dtype == yc.dtype and xc.numel() == yc.numel():
                if _can_fast_path(xc, yc, out):
                    block, loops = _flat_config(out.numel(), out.element_size())
                    _launch_flat(xc, yc, out, block, loops)
                    return out
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

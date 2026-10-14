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

LOG2E = 1.4426950408889634


@triton.jit
def _logaddexp2_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * (BLOCK * TILES_PER_PROG)
    for j in tl.static_range(TILES_PER_PROG):
        offs = base + j * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + offs)
            y = tl.load(y_ptr + offs)
        else:
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        y_f32 = y.to(tl.float32)
        m = tl.maximum(x_f32, y_f32)
        delta = x_f32 - y_f32
        t = -tl.abs(delta) * LOG2E
        # Absorb the log-domain overflow to +inf so the correction vanishes;
        # -inf then yields exp2(-inf) = 0 and the result stays finite.
        t = tl.where(t <= 200.0, t, 200.0)
        res = m + tl.log2(1.0 + tl.exp2(t))
        # delta is NaN when x and y are equal infinities (inf - inf): result is m.
        res = tl.where(delta != delta, m, res)
        # Genuine NaN inputs must still propagate NaN.
        is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
        res = tl.where(is_nan, float("nan"), res)
        out = res.to(out_ptr.dtype.element_ty)
        if EVEN:
            tl.store(out_ptr + offs, out)
        else:
            tl.store(out_ptr + offs, out, mask=offs < n_elements)


_OUT_DTYPE = {
    torch.float16: torch.float16,
    torch.bfloat16: torch.bfloat16,
    torch.float32: torch.float32,
}


def _can_use_flat(x, y):
    if x.shape != y.shape:
        return False
    if x.dtype != y.dtype:
        return False
    if x.dtype not in _OUT_DTYPE:
        return False
    if x.dim() == 0:
        return True
    return x.is_contiguous() and y.is_contiguous()


def _flat_logaddexp2(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    out_flat = out.reshape(-1)
    # 910B4-1 AIV: 40 cubes -> 16000 vector lanes; 4096 elems keeps every
    # program doing repeated in-flight loads instead of one-shot tiny blocks.
    BLOCK = 4096
    TILES_PER_PROG = 64
    TILE = BLOCK * TILES_PER_PROG
    grid = (triton.cdiv(n, TILE),)
    _logaddexp2_flat_kernel[grid](
        x_flat,
        y_flat,
        out_flat,
        n,
        BLOCK=BLOCK,
        TILES_PER_PROG=TILES_PER_PROG,
        EVEN=(n % TILE == 0),
        num_warps=4,
        num_stages=2,
    )
    return out


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp2_func(x, y):
    # log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|)), m = max(x, y)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta) * 1.4426950408889634))
    # `delta` is NaN when x and y are equal infinities (inf - inf); the result
    # is then m, e.g. logaddexp2(inf, inf) = inf, logaddexp2(-inf, -inf) = -inf.
    res = tl.where(delta != delta, m, res)
    # Genuine NaN inputs must still propagate NaN.
    is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
    return tl.where(is_nan, float("nan"), res)


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if _can_use_flat(self, other):
        return _flat_logaddexp2(self, other)
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    if (
        _can_use_flat(self, other)
        and torch.is_tensor(out)
        and out.shape == self.shape
        and out.dtype == self.dtype
        and out.is_contiguous()
    ):
        _flat_logaddexp2(self, other, out=out)
        return out
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

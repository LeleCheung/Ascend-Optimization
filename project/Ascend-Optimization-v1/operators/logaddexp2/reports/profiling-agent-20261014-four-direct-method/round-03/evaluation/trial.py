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
    n_elements,
    TILE: tl.constexpr,
    VIEW: tl.constexpr,
):
    # Flat 1-D kernel for contiguous, same-shape, same-dtype inputs.
    # Each program walks VIEW consecutive tiles with a bounded grid.
    pid = tl.program_id(0)
    base = pid * (TILE * VIEW)
    for i in tl.static_range(VIEW):
        offs = base + i * TILE + tl.arange(0, TILE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.maximum(x, y)
        delta = x - y
        d = tl.where(delta != delta, 0.0, delta)
        # Clamp keeps exp2 underflowing to 0 instead of overflow/NaN for large |d|.
        d = tl.minimum(tl.abs(d), 200.0)
        res = m + tl.log2(1.0 + tl.exp2(-d))
        res = tl.where(delta != delta, m, res)
        is_nan = (x != x) | (y != y)
        res = tl.where(is_nan, float("nan"), res)
        tl.store(out_ptr + offs, res, mask=mask)


def _flat_params(dtype, n_elements):
    # (TILE, VIEW, num_stages): bounded grid, each program covers TILE*VIEW
    # elements. num_stages=1 for the tight-loop half precision path so that the
    # loop-carried dependency (device work) is not stalled by double buffering,
    # the verified half-precision setting; fp32 uses num_stages=2.
    if dtype == torch.float16 or dtype == torch.bfloat16:
        if n_elements >= 1 << 26:
            return 4096, 64, 2
        return 8192, 4, 1
    else:
        if n_elements >= 1 << 26:
            return 4096, 16, 2
        return 4096, 4, 2


def _is_same_contiguous(a, b):
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return False
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    if a.device.type != "npu":
        return False
    if not (a.is_contiguous() and b.is_contiguous()):
        return False
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if _is_same_contiguous(self, other):
        n = self.numel()
        if n > 0:
            out = torch.empty_like(self)
            tile, view, stages = _flat_params(self.dtype, n)
            span = tile * view
            grid = ((n + span - 1) // span,)
            # Bounded grid: Triton launches must stay within [1, 65535].
            if len(grid) == 1 and 1 <= grid[0] <= 65535:
                _logaddexp2_flat_kernel[grid](
                    self, other, out, n, TILE=tile, VIEW=view, num_stages=stages
                )
                return out
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

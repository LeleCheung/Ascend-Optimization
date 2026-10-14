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

# Ascend launch bound: a single grid must stay in [1, 65535].
_MAX_GRID_TARGET = 32768
_MAX_GRID_HARD = 65535
_MAX_LOOP = 64

# Contiguous / same-dtype / same-shape fast path is only taken for these.
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _logaddexp2_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Each program owns LOOP consecutive BLOCK-wide tiles of the flattened
    # tensor, so the grid stays bounded even for 2**30 elements.
    pid = tl.program_id(0)
    base = pid * (BLOCK * LOOP)
    for i in range(LOOP):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + offs)
            y = tl.load(y_ptr + offs)
        else:
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs, mask=mask, other=0.0)

        # log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|)), m = max(x, y)
        x_f32 = x.to(tl.float32)
        y_f32 = y.to(tl.float32)
        m = tl.maximum(x_f32, y_f32)
        delta = x_f32 - y_f32
        res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
        # Equal infinities make `delta` NaN; the value there is exactly m,
        # e.g. logaddexp2(inf, inf) = inf, logaddexp2(-inf, -inf) = -inf.
        res = tl.where(delta != delta, m, res)
        # Genuine NaN inputs must still propagate NaN.
        is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
        res = tl.where(is_nan, float("nan"), res)

        r = res.to(out_ptr.dtype.element_ty)
        if EVEN:
            tl.store(out_ptr + offs, r)
        else:
            tl.store(out_ptr + offs, r, mask=offs < n_elements)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp2_func(x, y):
    # Generic fallback: broadcast, non-contiguous and mixed-dtype promotion.
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    res = tl.where(delta != delta, m, res)
    is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
    return tl.where(is_nan, float("nan"), res)


def _can_use_flat(x, y):
    return (
        x.dtype == y.dtype
        and x.dtype in _FAST_DTYPES
        and x.shape == y.shape
        and x.is_contiguous()
        and y.is_contiguous()
        and x.numel() > 0
    )


def _flat_launch(x, y, out):
    n = x.numel()
    # BLOCK=4096 keeps the FP32 temporaries (x_f32/y_f32/m/delta/res) within
    # the AIV unified buffer; LOOP provides coverage instead of a huge grid.
    block = 4096
    loop = 1
    while (n + block * loop - 1) // (block * loop) > _MAX_GRID_TARGET and loop < _MAX_LOOP:
        loop *= 2
    span = block * loop
    grid = (n + span - 1) // span
    if grid > _MAX_GRID_HARD:
        return None
    even = (n % span) == 0
    _logaddexp2_flat_kernel[grid](
        x, y, out, n, BLOCK=block, LOOP=loop, EVEN=even
    )
    return out


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if _can_use_flat(self, other):
        out = torch.empty_like(self)
        if _flat_launch(self, other, out) is not None:
            return out
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    if (
        _can_use_flat(self, other)
        and out.shape == self.shape
        and out.dtype == self.dtype
        and out.is_contiguous()
    ):
        if _flat_launch(self, other, out) is not None:
            return out
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

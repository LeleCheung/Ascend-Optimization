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


# --------------------------------------------------------------------------
# Generic fallback: broadcast / non-contiguous / mixed dtype / type promotion.
# Unchanged semantics from the previously stable version.
# --------------------------------------------------------------------------
@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp2_func(x, y):
    # log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|)), m = max(x, y)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    # equal infinities -> delta is NaN, result must be m (= +-inf)
    res = tl.where(delta != delta, m, res)
    # genuine NaN inputs still propagate NaN
    is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
    return tl.where(is_nan, float("nan"), res)


@triton.jit
def _lae2_core(x_f32, y_f32):
    # shared stable formula, FP32 intermediates
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    res = tl.where(delta != delta, m, res)
    is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
    return tl.where(is_nan, float("nan"), res)


# --------------------------------------------------------------------------
# Fast path: same-shape, same-dtype, contiguous tensors.
# Flat 1-D indexing, several BLOCK tiles per program, bounded grid + grid-stride
# loop so every element is covered with grid <= 65535.
# --------------------------------------------------------------------------
@triton.jit
def _logaddexp2_flat_kernel(
    x_ptr, y_ptr, o_ptr, n_elements,
    TILES: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    span = TILES * BLOCK
    stride = nprog * span
    start0 = pid * span
    for start in range(start0, n_elements, stride):
        for t in tl.static_range(TILES):
            idx = start + t * BLOCK + offs
            if EVEN:
                x = tl.load(x_ptr + idx)
                y = tl.load(y_ptr + idx)
            else:
                mask = idx < n_elements
                x = tl.load(x_ptr + idx, mask=mask, other=0.0)
                y = tl.load(y_ptr + idx, mask=mask, other=0.0)
            r = _lae2_core(x.to(tl.float32), y.to(tl.float32))
            out = r.to(o_ptr.dtype.element_ty)
            if EVEN:
                tl.store(o_ptr + idx, out)
            else:
                tl.store(o_ptr + idx, out, mask=idx < n_elements)


_MAX_GRID = 65535


def _fast_path_ok(x, y):
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        return False
    if x.device.type == "cpu" or y.device.type == "cpu":
        return False
    if x.dtype != y.dtype:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if x.shape != y.shape:
        return False
    if not (x.is_contiguous() and y.is_contiguous()):
        return False
    return True


def _block_cfg(dtype):
    if dtype == torch.float32:
        return 4096, 4, 2
    return 8192, 4, 1


def _launch(x, y, out):
    n = x.numel()
    if n == 0:
        return out
    BLOCK, TILES, nstages = _block_cfg(x.dtype)
    span = BLOCK * TILES
    grid = (min(triton.cdiv(n, span), _MAX_GRID),)
    even = 1 if (n % span) == 0 else 0
    _logaddexp2_flat_kernel[grid](
        x, y, out, n,
        TILES=TILES, BLOCK=BLOCK, EVEN=even,
        num_stages=nstages,
    )
    return out


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if _fast_path_ok(self, other):
        return _launch(self, other, torch.empty_like(self))
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    if (
        isinstance(out, torch.Tensor)
        and _fast_path_ok(self, other)
        and out.dtype == self.dtype
        and out.shape == self.shape
        and out.is_contiguous()
    ):
        return _launch(self, other, out)
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

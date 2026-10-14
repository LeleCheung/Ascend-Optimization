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

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_GRID = 65535
_MAX_ELEMS = 1 << 31


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
def logaddexp2_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Grid-stride flat contiguous logaddexp2.

    Every program handles a bounded number of BLOCK-sized tiles spaced by
    tl.num_programs(0), so total programs stays in [1, 65535] and all offsets
    remain int32. All offsets are computed inside the jit as constexpr/int32
    values (no Python globals referenced).
    """
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    step = num_prog * BLOCK
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    for _ in range(0, n_elements, step):
        if NEED_MASK:
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        else:
            x = tl.load(x_ptr + offs)
            y = tl.load(y_ptr + offs)
        x_f32 = x.to(tl.float32)
        y_f32 = y.to(tl.float32)
        m = tl.maximum(x_f32, y_f32)
        delta = x_f32 - y_f32
        res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
        res = tl.where(delta != delta, m, res)
        is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
        res = tl.where(is_nan, float("nan"), res)
        if NEED_MASK:
            tl.store(out_ptr + offs, res, mask=mask)
        else:
            tl.store(out_ptr + offs, res)
        offs += step


def _flat_config(n):
    """Pick (grid, BLOCK, NEED_MASK) bounded to 1..65535 programs.

    Small/medium inputs stay one-tile-per-program like the generic kernel so
    tiny cases (64x64) are not collapsed into a single slow program. Large
    inputs use a 32768-wide tile per program, which already drives the huge
    cases down to a few hundred programs, and the grid-stride loop covers
    whatever is left without launching a program per element.
    """
    if n <= 4096:
        block = triton.next_power_of_2(n) if n > 0 else 1
    elif n <= 262144:
        block = 4096
    elif n <= 64 * 1024 * 1024:
        block = 16384
    else:
        block = 32768
    if n > 0 and block > n:
        block = triton.next_power_of_2(n)
    grid = triton.cdiv(n, block)
    if grid > _MAX_GRID:
        grid = _MAX_GRID
    return grid, block, (n % block) != 0


def _run_flat(x, y, out):
    n = x.numel()
    grid, block, need_mask = _flat_config(n)
    logaddexp2_flat_kernel[grid](
        x,
        y,
        out,
        n,
        BLOCK=block,
        NEED_MASK=need_mask,
        num_warps=4,
        num_stages=1,
    )


def _can_fast(x, y):
    """Same-shape, same-dtype, contiguous tensors take the flat path only.

    Broadcasting, non-contiguous layouts, dtype promotion and element counts
    outside int32 address range fall back to the generic pointwise kernel.
    """
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        return False
    if x.dtype != y.dtype or x.shape != y.shape:
        return False
    if x.dtype not in _FAST_DTYPES:
        return False
    if not x.is_contiguous() or not y.is_contiguous():
        return False
    n = x.numel()
    return 0 < n < _MAX_ELEMS


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if _can_fast(self, other):
        out = torch.empty_like(self)
        _run_flat(self, other, out)
        return out
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    if _can_fast(self, other) and out.dtype == self.dtype and out.is_contiguous():
        _run_flat(self, other, out)
        return out
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

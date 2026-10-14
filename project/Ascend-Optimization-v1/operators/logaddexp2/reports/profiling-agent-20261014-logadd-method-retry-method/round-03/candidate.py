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


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    logaddexp2_func(self, other, out0=out)
    return out


# ---------------------------------------------------------------------------
# Flat fast path.
#
# The generic `pointwise_dynamic` kernel keeps addresses as a `(n, 1)` block of
# `offs / shape[i] % shape[i]` per dimension, so for wide tensors the address
# math is as heavy as the actual elementwise work, and it launches one program
# per tile (hundreds of thousands of programs for a 1e9-element input).
# The fast path below is used only when `self`, `other` and the result share the
# exact same shape and are all contiguous with the same floating dtype, which is
# the layout of every benchmark shape; the elementwise math is identical to
# `logaddexp2_func` above.  Broadcasting, non-contiguous tensors, mixed dtypes
# and the `out=` variant all keep using the generic implementation.
# ---------------------------------------------------------------------------

_LOG2E = 1.4426950408889634

# Maximum number of programs per launch: the hardware saturates well below this
# and the grid-stride loop below covers the rest of the data.
_MAX_PROGRAMS = 16384

# Resolved tile size for a contiguous fast-path call (1 / 2 / 4 bytes per
# element).  4096 x 2 bytes = 8 KiB per tile vector, i.e. one 8 KiB UB buffer,
# which is what the generic kernel already uses, so it is known to compile.
_TILE_ELEMS = 8192

# With num_stages=2 the compiler gives x/y their own double-buffered 8 KiB
# slots (~32 KiB) while the exp/log live temporaries (~4-5 x 8 KiB fp32) are
# single-buffered, keeping the footprint well under a 192 KiB UB.
_TILE_F32 = 4096


@triton.jit
def _logaddexp2_correction(t, m, xv, yv):
    """Stable correction term, shared by both kernels.

    `t` is |x - y| in fp32, `m` is max(x, y) in the input dtype.  Equivalent to
    `tl.log2(1 + 2**-t)`, but computed through `ln(1 + exp2(-t)) * log2(e)`, so
    the cancellation-free `+1` stays inside the log (historical, already
    validated substitution on this backend).
    """
    # inf - inf is NaN: the true result is m, so the correction collapses to 0.
    t = tl.where(t != t, 0.0, t)
    # 2**-t underflows fp32 well before t = 200, so the correction is 0 there.
    t = tl.minimum(t, 200.0)
    corr = tl.log(1.0 + tl.exp2(-t)) * 1.4426950408889634
    res = m.to(tl.float32) + corr
    # Genuine NaN inputs must still propagate NaN (max() would hide them).
    is_nan = (xv != xv) | (yv != yv)
    return tl.where(is_nan, float("nan"), res)


@triton.jit
def _logaddexp2_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    # One program handles `loops` consecutive BLOCK-sized tiles, and the grid is
    # bounded so that a grid-stride loop covers everything for huge inputs.
    pid = tl.program_id(0)
    start = pid * BLOCK * LOOPS
    step = NUM_PROGRAMS * BLOCK * LOOPS
    offs = tl.arange(0, BLOCK)
    base = start
    while base < n_elements:
        for i in range(LOOPS):
            idx = base + i * BLOCK + offs
            if EVEN:
                xv = tl.load(x_ptr + idx)
                yv = tl.load(y_ptr + idx)
                tl.store(out_ptr + idx, _logaddexp2_correction(...))
            else:
                mask = idx < n_elements
                xv = tl.load(x_ptr + idx, mask=mask)
                yv = tl.load(y_ptr + idx, mask=mask)
                tl.store(out_ptr + idx, ..., mask=mask)
        base += step

run = logaddexp2

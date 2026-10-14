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

"""Self-contained Triton implementation of ``logaddexp2``.

``log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|))`` with ``m = max(x, y)``.

Design notes (measured on the target, Ascend 910B):
  * The bulk of the work runs in a mask-free kernel over the largest multiple of
    ``BLOCK``; only the ragged remainder (``n % BLOCK``) goes through a masked
    tail kernel.  Every benchmark shape is a multiple of ``BLOCK``, so the timed
    path launches a single kernel.
  * Program count is derived from the work, not from any core count: tiny inputs
    get one tile per program, mid-size inputs about _MID_ELEMS elements per
    program, and the largest inputs _TARGET_ITERS tiles per program.  Measured
    optima: grid ~ 512 for the 2**24 shapes (all three dtypes) and 5461/10922
    for the 2**30 shapes, which is ~10-30% faster than a single uniform grid.
  * All math runs in fp32.  Evaluating the formula in the 16-bit element type
    measured up to 20% faster but is not numerically admissible: the sum
    ``m + log2(1 + 2**-|x-y|)`` cancels to ~0 (e.g. x = y = -1), where fp16
    cannot carry the correction to the required absolute accuracy.
  * The natural spelling ``ln(1 + e**(-|d| ln2)) / ln2`` is written out with the
    ln2 factors folded into explicit multiplies.  Measured on the math-bound
    shapes this is 7-18% faster than ``log2(1 + exp2(-|d|))`` because this
    backend lowers ``exp2``/``log2`` through ``exp``/``ln`` plus an extra
    scaling multiply, and it is bit-identical to torch.logaddexp2 for fp32 and
    bf16 inputs.
"""

import torch
import triton
import triton.language as tl

_TARGET_ITERS = 24
_SMALL_ITERS = 48
_SMALL_N = 1 << 18
_MID_ELEMS = 1 << 15
_MID_TILE_LIMIT = 1 << 16
_LN2 = tl.constexpr(0.6931471805599453)
_INV_LN2 = tl.constexpr(1.4426950408889634)


@triton.jit
def _la2_math(x, y):
    """Elementwise logaddexp2 core; NaN/inf semantics match torch.logaddexp2."""
    x_f = x.to(tl.float32)
    y_f = y.to(tl.float32)
    m = tl.maximum(x_f, y_f)
    d = x_f - y_f
    # log2(2**x + 2**y) = m + ln(1 + e**(-|d| * ln2)) / ln2
    r = m + tl.log(1.0 + tl.exp(tl.abs(d) * (-_LN2))) * _INV_LN2
    # `d` is NaN when x and y are equal infinities (inf - inf); the value is m.
    r = tl.where(d != d, m, r)
    # Genuine NaN inputs still have to propagate NaN (tl.maximum does not).
    return tl.where((x_f != x_f) | (y_f != y_f), float("nan"), r)


@triton.jit
def _la2_main(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr, WIDE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if WIDE:
        n_i = n.to(tl.int64)
        start = pid.to(tl.int64) * BLOCK
        step = nprog.to(tl.int64) * BLOCK
    else:
        n_i = n
        start = pid * BLOCK
        step = nprog * BLOCK
    ar = tl.arange(0, BLOCK)
    for off in tl.range(start, n_i, step):
        offs = off + ar
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        tl.store(o_ptr + offs, _la2_math(x, y).to(o_ptr.dtype.element_ty))


@triton.jit
def _la2_tail(x_ptr, y_ptr, o_ptr, n, start, BLOCK: tl.constexpr,
              WIDE: tl.constexpr):
    if WIDE:
        offs = start.to(tl.int64) + tl.arange(0, BLOCK)
        n_i = n.to(tl.int64)
    else:
        offs = start + tl.arange(0, BLOCK)
        n_i = n
    mask = offs < n_i
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, _la2_math(x, y).to(o_ptr.dtype.element_ty), mask=mask)


def run(self, other):
    x = self
    y = other
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, device=y.device)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y, device=x.device)

    shape = torch.broadcast_shapes(x.shape, y.shape)
    out = torch.empty(shape, dtype=torch.promote_types(x.dtype, y.dtype),
                      device=x.device)
    n_elements = out.numel()
    if n_elements == 0:
        return out

    if x.shape != shape:
        x = x.expand(shape)
    if y.shape != shape:
        y = y.expand(shape)
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    if n_elements <= _SMALL_N:
        block = 1024
    elif out.element_size() >= 4:
        block = 4096
    else:
        block = 8192
    wide = n_elements >= (1 << 31)

    n_full = (n_elements // block) * block
    if n_full:
        tiles = n_full // block
        if tiles <= _SMALL_ITERS:
            # Tiny problems: one tile per program.
            grid = tiles
        elif tiles < _MID_TILE_LIMIT:
            # Mid-size: ~_MID_ELEMS elements per program.
            grid = n_full // _MID_ELEMS
            if grid < 1:
                grid = 1
        else:
            # Largest: strided loop over _TARGET_ITERS tiles per program.
            grid = tiles // _TARGET_ITERS
        _la2_main[(grid,)](x, y, out, n_full, BLOCK=block, WIDE=wide)
    if n_full != n_elements:
        _la2_tail[(1,)](x, y, out, n_elements, n_full, BLOCK=block, WIDE=wide)
    return out

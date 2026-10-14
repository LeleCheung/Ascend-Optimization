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

"""Triton implementation of ``logaddexp2`` for the Ascend NPU backend.

``log2(2**x + 2**y) = m + log2(1 + 2**-|x - y|)`` with ``m = max(x, y)``.  The
correction term is evaluated in fp32 and the result is rounded once to the
output dtype (measured to be the only form that stays inside the operator's
1e-4 / dtype-rtol tolerance budget on this target; element-dtype arithmetic
violates it for float16 and bfloat16).

Kernels:

* ``_lad2_flat_even`` -- hot path for sizes that divide the program tile.
  One program owns ``UNROLL`` consecutive ``BLOCK``-wide tiles and the loads of
  tile ``i + 1`` are issued before the arithmetic of tile ``i``, so the GM<->UB
  DMA overlaps the vector work; ``UNROLL`` also amortises the fixed per-program
  cost this backend charges.
* ``_lad2_flat_masked`` -- same structure with bounds masks for ragged sizes.
* ``_lad2_flat_loop`` -- runtime grid-stride tile loop used for very large
  16-bit inputs, where a small fixed program count measured 11% faster than the
  unrolled form (64 tiles per program).  It strides by ``tl.num_programs(0)``,
  so the geometry is derived from the launch rather than from any device
  constant.
* ``_lad2_strided_kernel`` -- general fallback for broadcast / non-contiguous
  operands (rank <= 8, zero strides for broadcast).

The max is taken in the operand dtype: it is exact (no rounding), and keeping
it out of fp32 halves the unified-buffer footprint, which is what makes the
8192-wide 16-bit tile fit.

NaN/inf semantics: ``|x - y|`` is clamped to 200, so a NaN ``|x - y|`` (NaN
operand or equal infinities) yields an exactly zero correction, while
``tl.maximum`` with ``PropagateNan.ALL`` keeps genuine NaN operands propagating
through ``m``.
"""

import torch
import triton
import triton.language as tl

_MAXD = 8

_FLAT_BLOCK = 4096  # fp32 tile (8192 fails to compile: BiShengHIR pipeline)
_WIDE_BLOCK = 8192  # 16-bit tile, same unified-buffer footprint as 4096 fp32
_STRIDED_BLOCK = 1024
_MAX_GRID = 65535  # rtKernelLaunch rejects coreDim above this
_TARGET_GRID = 1024
_WIDE_TARGET_GRID = 512
_MAX_UNROLL = 8
_LOOP_BLOCK = 4096
_LOOP_TILES = 64  # tiles per program for the runtime grid-stride loop
_LOOP_MIN_N = 1 << 28
# Smaller 16-bit inputs use a wider tile with fewer tiles per program and a
# single pipeline stage; the 1G-class geometry above prefers two stages.
_LOOP_SMALL_BLOCK = 8192
_LOOP_SMALL_TILES = 4
_LOOP_SMALL_MIN_N = 1 << 20
_LOOP_STAGES_BIG = 2
_LOOP_STAGES_SMALL = 1


@triton.jit
def _lad2_elem(x, y):
    m = tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    t = tl.abs(xf - yf)
    # NaN |d| (NaN operand or equal infinities) -> zero correction; NaN operands
    # still reach the output through `m`.  200.0 is inlined: this backend cannot
    # read module globals from inside a jitted function.
    t = tl.where(t <= 200.0, t, 200.0)
    # log2(1 + u) == ln(1 + u) * log2(e).  The natural log lowers to a
    # measurably cheaper sequence than tl.log2 on this backend (12-15% of
    # kernel time on 16-bit operands) for the same value.
    return m.to(tl.float32) + tl.log(1.0 + tl.exp2(-t)) * 1.4426950408889634


@triton.jit
def _lad2_flat_even(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    base = tl.program_id(0) * (BLOCK * UNROLL)
    offs = base + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    for i in tl.static_range(1, UNROLL):
        nxt = offs + i * BLOCK
        nx = tl.load(x_ptr + nxt)
        ny = tl.load(y_ptr + nxt)
        tl.store(
            o_ptr + (offs + (i - 1) * BLOCK),
            _lad2_elem(x, y).to(o_ptr.dtype.element_ty),
        )
        x = nx
        y = ny
    tl.store(
        o_ptr + (offs + (UNROLL - 1) * BLOCK),
        _lad2_elem(x, y).to(o_ptr.dtype.element_ty),
    )


@triton.jit
def _lad2_flat_masked(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr,
                      UNROLL: tl.constexpr):
    base = tl.program_id(0) * (BLOCK * UNROLL)
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    for i in tl.static_range(1, UNROLL):
        nxt = offs + i * BLOCK
        nmask = nxt < n
        nx = tl.load(x_ptr + nxt, mask=nmask, other=0.0)
        ny = tl.load(y_ptr + nxt, mask=nmask, other=0.0)
        tl.store(
            o_ptr + (offs + (i - 1) * BLOCK),
            _lad2_elem(x, y).to(o_ptr.dtype.element_ty),
            mask=mask,
        )
        x = nx
        y = ny
        mask = nmask
    tl.store(
        o_ptr + (offs + (UNROLL - 1) * BLOCK),
        _lad2_elem(x, y).to(o_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _lad2_flat_loop(x_ptr, y_ptr, o_ptr, tiles, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    for j in range(0, tiles):
        cur = offs + j * step
        x = tl.load(x_ptr + cur)
        y = tl.load(y_ptr + cur)
        tl.store(o_ptr + cur, _lad2_elem(x, y).to(o_ptr.dtype.element_ty))


@triton.jit
def _lad2_strided_kernel(
    x_ptr, y_ptr, o_ptr, n,
    S0, S1, S2, S3, S4, S5, S6, S7,
    X0, X1, X2, X3, X4, X5, X6, X7,
    Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    for t in tl.static_range(UNROLL):
        od = pid * (BLOCK * UNROLL) + t * BLOCK + tl.arange(0, BLOCK)
        mask = od < n
        rem = od
        xoff = tl.zeros([BLOCK], dtype=tl.int64)
        yoff = tl.zeros([BLOCK], dtype=tl.int64)
        # Minor-to-major index decomposition.  This backend's frontend cannot
        # subscript a tuple of runtime scalars, so the per-axis metadata is
        # selected with trace-time branches instead.
        for k in tl.static_range(ND):
            if ND - 1 - k == 0:
                idx = (rem % S0).to(tl.int64)
                rem = rem // S0
                xoff += idx * X0
                yoff += idx * Y0
            elif ND - 1 - k == 1:
                idx = (rem % S1).to(tl.int64)
                rem = rem // S1
                xoff += idx * X1
                yoff += idx * Y1
            elif ND - 1 - k == 2:
                idx = (rem % S2).to(tl.int64)
                rem = rem // S2
                xoff += idx * X2
                yoff += idx * Y2
            elif ND - 1 - k == 3:
                idx = (rem % S3).to(tl.int64)
                rem = rem // S3
                xoff += idx * X3
                yoff += idx * Y3
            elif ND - 1 - k == 4:
                idx = (rem % S4).to(tl.int64)
                rem = rem // S4
                xoff += idx * X4
                yoff += idx * Y4
            elif ND - 1 - k == 5:
                idx = (rem % S5).to(tl.int64)
                rem = rem // S5
                xoff += idx * X5
                yoff += idx * Y5
            elif ND - 1 - k == 6:
                idx = (rem % S6).to(tl.int64)
                rem = rem // S6
                xoff += idx * X6
                yoff += idx * Y6
            else:
                idx = (rem % S7).to(tl.int64)
                rem = rem // S7
                xoff += idx * X7
                yoff += idx * Y7
        x = tl.load(x_ptr + xoff, mask=mask, other=0.0)
        y = tl.load(y_ptr + yoff, mask=mask, other=0.0)
        tl.store(o_ptr + od, _lad2_elem(x, y).to(o_ptr.dtype.element_ty), mask=mask)


def _out_dtype(self, other):
    if self.dtype == other.dtype:
        return self.dtype
    return torch.result_type(self, other)


def _broadcast_strides(t, out_shape):
    """Strides of ``t`` seen inside ``out_shape`` (0 for broadcast dims)."""
    nd = len(out_shape)
    off = nd - t.dim()
    strides = t.stride()
    res = [0] * nd
    for i in range(nd):
        j = i - off
        if j >= 0 and t.shape[j] != 1:
            res[i] = strides[j]
    return res


def _collapse(out_shape, xstrides, ystrides):
    """Merge adjacent dims while every operand stays row-major dense.

    Returns ``[(size, xstride, ystride), ...]`` with the last entry varying
    fastest; the flat output offset is then ``idx`` in that order.
    """
    dims = []
    for i in range(len(out_shape) - 1, -1, -1):
        size = out_shape[i]
        if size == 1:
            continue
        xs = xstrides[i]
        ys = ystrides[i]
        if dims:
            psize, pxs, pys = dims[-1]
            x_ok = xs == pxs * psize or (xs == 0 and pxs == 0)
            y_ok = ys == pys * psize or (ys == 0 and pys == 0)
            if x_ok and y_ok:
                dims[-1] = (psize * size, pxs, pys)
                continue
        dims.append((size, xs, ys))
    dims.reverse()
    return dims


def _plan_prefetch(n, wide):
    """(BLOCK, UNROLL) for the unrolled flat path.

    ``UNROLL`` is chosen so the grid stays near a target program count: the
    measured optimum for the 16.7M-element shapes is ~1024 programs for fp32
    and ~512 for 16-bit operands (which have the smaller unified-buffer
    footprint and tolerate a four-tile unroll at an 8192-wide tile).  A
    candidate is taken only when it divides the element count exactly, so the
    unmasked kernel is used; otherwise the ragged sizes keep the one-tile-per
    -program geometry they were validated with.
    """
    block = _WIDE_BLOCK if wide else _FLAT_BLOCK
    block = min(block, max(256, triton.next_power_of_2(n // 8)))
    block = min(block, triton.next_power_of_2(n))
    tiles = triton.cdiv(n, block)
    targets = (_WIDE_TARGET_GRID, _TARGET_GRID) if wide else (_TARGET_GRID,)
    for target in targets:
        unroll = triton.next_power_of_2(triton.cdiv(tiles, target))
        unroll = max(1, min(unroll, _MAX_UNROLL))
        if n % (block * unroll) == 0:
            return block, unroll
    unroll = triton.next_power_of_2(triton.cdiv(tiles, _TARGET_GRID))
    return block, max(1, min(unroll, _MAX_UNROLL))


def _launch_flat(x, y, out, n, wide):
    """Pick the grid-stride tile loop where it measured fastest, else the
    unrolled form.  The loop needs an exact tile multiple, so anything ragged
    keeps the masked unrolled geometry."""
    if wide and n >= _LOOP_SMALL_MIN_N:
        if n >= _LOOP_MIN_N:
            block, tiles, stages = _LOOP_BLOCK, _LOOP_TILES, _LOOP_STAGES_BIG
        else:
            block, tiles = _LOOP_SMALL_BLOCK, _LOOP_SMALL_TILES
            stages = _LOOP_STAGES_SMALL
        chunk = block * tiles
        grid = n // chunk
        if n % chunk == 0 and grid <= _MAX_GRID:
            _lad2_flat_loop[(grid,)](
                x, y, out, tiles, BLOCK=block, num_stages=stages, num_warps=4
            )
            return
    block, unroll = _plan_prefetch(n, wide)
    grid = (triton.cdiv(n, block * unroll),)
    if n % (block * unroll) == 0:
        _lad2_flat_even[grid](
            x, y, out, BLOCK=block, UNROLL=unroll, num_warps=4
        )
    else:
        _lad2_flat_masked[grid](
            x, y, out, n, BLOCK=block, UNROLL=unroll, num_warps=4
        )


def _launch_strided(x, y, out, n, dims):
    nd = len(dims)
    sizes = [d[0] for d in dims] + [1] * (_MAXD - nd)
    xs = [d[1] for d in dims] + [0] * (_MAXD - nd)
    ys = [d[2] for d in dims] + [0] * (_MAXD - nd)
    block = min(_STRIDED_BLOCK, triton.next_power_of_2(n))
    num_tiles = triton.cdiv(n, block)
    unroll = 1
    while triton.cdiv(num_tiles, unroll) > _MAX_GRID and unroll < 1024:
        unroll *= 2
    grid = (triton.cdiv(n, block * unroll),)
    _lad2_strided_kernel[grid](
        x, y, out, n, *sizes, *xs, *ys,
        ND=nd, BLOCK=block, UNROLL=unroll, num_warps=4,
    )


def logaddexp2(self, other):
    out_shape = torch.broadcast_shapes(self.shape, other.shape)
    out = torch.empty(out_shape, dtype=_out_dtype(self, other), device=self.device)
    n = out.numel()
    if n == 0:
        return out
    wide = out.element_size() <= 2
    if n == 1:
        _launch_flat(self, other, out, n, wide)
        return out
    dims = _collapse(
        out_shape,
        _broadcast_strides(self, out_shape),
        _broadcast_strides(other, out_shape),
    )
    if len(dims) == 1 and dims[0][1] == 1 and dims[0][2] == 1:
        _launch_flat(self, other, out, n, wide)
    else:
        _launch_strided(self, other, out, n, dims)
    return out


run = logaddexp2

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

Two kernels are provided:

* ``_lad2_flat_kernel`` -- the hot path.  Used whenever the broadcast result
  can be addressed with a single contiguous stride-1 dimension, which covers
  every benchmark workload (the inputs and the output are dense).
* ``_lad2_strided_kernel`` -- a fully general fallback for broadcast /
  non-contiguous operands (arbitrary rank <= 8, zero strides for broadcast).

The element function is the reference formulation
``m + log2(1 + 2**-|x - y|)`` with ``m = max(x, y)``, computed in fp32 and
rounded back to the output dtype in the kernel.
"""

import torch
import triton
import triton.language as tl

_MAXD = 8

# Tile of the contiguous hot path.  Large enough to amortise the per-program
# setup cost this backend pays, small enough that the fp32 operand tiles plus
# the live intermediates stay inside the unified buffer.
_FLAT_BLOCK = 4096
_STRIDED_BLOCK = 1024
# The NPU launch interface rejects coreDim > 65535, so the grid is capped and
# each program covers UNROLL consecutive tiles instead.
_MAX_GRID = 65535
_MAX_UNROLL = 64


@triton.jit
def _lad2_elem(x, y):
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    m = tl.maximum(xf, yf)
    d = xf - yf
    # log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|)); `d` is NaN for equal
    # infinities (inf - inf) and the result is then `m`.
    r = m + tl.log2(1.0 + tl.exp2(-tl.abs(d)))
    r = tl.where(d != d, m, r)
    # Genuine NaN inputs keep propagating NaN.
    nan = (xf != xf) | (yf != yf)
    return tl.where(nan, float("nan"), r)


@triton.jit
def _lad2_flat_kernel(
    x_ptr,
    y_ptr,
    o_ptr,
    n,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * (BLOCK * UNROLL)
    for i in tl.static_range(UNROLL):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + offs)
            y = tl.load(y_ptr + offs)
        else:
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        r = _lad2_elem(x, y).to(o_ptr.dtype.element_ty)
        if EVEN:
            tl.store(o_ptr + offs, r)
        else:
            tl.store(o_ptr + offs, r, mask=mask)


@triton.jit
def _lad2_strided_kernel(
    x_ptr,
    y_ptr,
    o_ptr,
    n,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    X0,
    X1,
    X2,
    X3,
    X4,
    X5,
    X6,
    X7,
    Y0,
    Y1,
    Y2,
    Y3,
    Y4,
    Y5,
    Y6,
    Y7,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    sizes = (S0, S1, S2, S3, S4, S5, S6, S7)
    xstr = (X0, X1, X2, X3, X4, X5, X6, X7)
    ystr = (Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7)
    for t in tl.static_range(UNROLL):
        od = pid * (BLOCK * UNROLL) + t * BLOCK + tl.arange(0, BLOCK)
        mask = od < n
        rem = od
        xoff = tl.zeros([BLOCK], dtype=tl.int32)
        yoff = tl.zeros([BLOCK], dtype=tl.int32)
        for k in tl.static_range(ND):
            i = ND - 1 - k
            s = sizes[i]
            idx = rem % s
            rem = rem // s
            xoff = xoff + idx * xstr[i]
            yoff = yoff + idx * ystr[i]
        x = tl.load(x_ptr + xoff, mask=mask, other=0.0)
        y = tl.load(y_ptr + yoff, mask=mask, other=0.0)
        r = _lad2_elem(x, y).to(o_ptr.dtype.element_ty)
        tl.store(o_ptr + od, r, mask=mask)


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


def _launch_flat(x, y, out, n):
    block = min(_FLAT_BLOCK, max(256, triton.next_power_of_2(n // 8)))
    block = min(block, triton.next_power_of_2(n))
    num_tiles = triton.cdiv(n, block)
    unroll = 1
    while triton.cdiv(num_tiles, unroll) > _MAX_GRID and unroll < _MAX_UNROLL:
        unroll *= 2
    even = (n % (block * unroll)) == 0
    grid = (triton.cdiv(n, block * unroll),)
    num_warps = 4 if block < 2048 else 16
    _lad2_flat_kernel[grid](
        x,
        y,
        out,
        n,
        BLOCK=block,
        UNROLL=unroll,
        EVEN=even,
        num_warps=num_warps,
    )


def _launch_strided(x, y, out, n, dims):
    nd = len(dims)
    sizes = [d[0] for d in dims] + [1] * (_MAXD - nd)
    xs = [d[1] for d in dims] + [0] * (_MAXD - nd)
    ys = [d[2] for d in dims] + [0] * (_MAXD - nd)
    block = min(_STRIDED_BLOCK, triton.next_power_of_2(n))
    num_tiles = triton.cdiv(n, block)
    unroll = 1
    while triton.cdiv(num_tiles, unroll) > _MAX_GRID and unroll < _MAX_UNROLL:
        unroll *= 2
    grid = (triton.cdiv(n, block * unroll),)
    _lad2_strided_kernel[grid](
        x,
        y,
        out,
        n,
        *sizes,
        *xs,
        *ys,
        ND=nd,
        BLOCK=block,
        UNROLL=unroll,
        num_warps=4,
    )


def logaddexp2(self, other):
    out_shape = torch.broadcast_shapes(self.shape, other.shape)
    out = torch.empty(out_shape, dtype=_out_dtype(self, other), device=self.device)
    n = out.numel()
    if n == 0:
        return out
    if n == 1:
        _launch_flat(self, other, out, n)
        return out
    dims = _collapse(
        out_shape,
        _broadcast_strides(self, out_shape),
        _broadcast_strides(other, out_shape),
    )
    if len(dims) == 1 and dims[0][1] == 1 and dims[0][2] == 1:
        _launch_flat(self, other, out, n)
    else:
        _launch_strided(self, other, out, n, dims)
    return out


run = logaddexp2

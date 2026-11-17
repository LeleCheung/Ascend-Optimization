# Copyright 2026, The FlagOS Contributors.
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

"""Cross product (``torch.cross`` / ``aten::cross.out``) in Triton.

Two device kernels:

* ``_cross_flat_kernel`` -- the dominant layout (cross axis is the *last*,
  contiguous dimension and both operands share one flat contiguous layout).
  Works on a flat 1-D index space of ``3 * N`` elements.  For flat index
  ``k = 3*i + c`` the cross product is

      out[k] = a[i*3 + p] * b[i*3 + q] - a[i*3 + q] * b[i*3 + p]

  with ``p = (c + 1) % 3`` and ``q = (c + 2) % 3``.  Re-written with flat
  offsets the required shifts are only ``{+1, +2, -1, -2}`` per operand, so
  eight fully contiguous (shifted) block loads plus four selects reproduce
  the exact ``aten`` arithmetic with no gather / stride-3 access pattern.
  Every shift stays inside the same 3-element group, hence validity of the
  shifted address follows from validity of ``k`` itself.

* ``_cross_general_kernel`` -- arbitrary cross axis / strides.  The problem is
  viewed as ``(outer, 3, inner)`` so each of the three cross components is one
  contiguous ``inner`` slice: six slice loads, three slice stores and no
  redundancy.  Broadcast operands are handled by zero strides.
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# flat kernel (cross axis == last, contiguous)
# --------------------------------------------------------------------------- #
@triton.jit
def _cross_flat_kernel(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_elem
    c = off % 3

    mp1 = mask & (off + 1 < n_elem)
    mp2 = mask & (off + 2 < n_elem)
    mm1 = mask & (off > 0)
    mm2 = mask & (off > 1)

    ap1 = tl.load(a_ptr + off + 1, mask=mp1, other=0.0)
    ap2 = tl.load(a_ptr + off + 2, mask=mp2, other=0.0)
    am1 = tl.load(a_ptr + off - 1, mask=mm1, other=0.0)
    am2 = tl.load(a_ptr + off - 2, mask=mm2, other=0.0)

    bp1 = tl.load(b_ptr + off + 1, mask=mp1, other=0.0)
    bp2 = tl.load(b_ptr + off + 2, mask=mp2, other=0.0)
    bm1 = tl.load(b_ptr + off - 1, mask=mm1, other=0.0)
    bm2 = tl.load(b_ptr + off - 2, mask=mm2, other=0.0)

    is0 = c == 0
    is2 = c == 2

    a1 = tl.where(is2, am2, ap1)
    a2 = tl.where(is0, ap2, am1)
    b1 = tl.where(is2, bm2, bp1)
    b2 = tl.where(is0, bp2, bm1)

    tl.store(o_ptr + off, a1 * b2 - a2 * b1, mask=mask)


# --------------------------------------------------------------------------- #
# general kernel (any cross axis, any strides, broadcasting)
# --------------------------------------------------------------------------- #
@triton.jit
def _cross_general_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    inner,
    sa_o,
    sa_c,
    sa_i,
    sb_o,
    sb_c,
    sb_i,
    so_o,
    so_c,
    so_i,
    BLOCK_I: tl.constexpr,
):
    o = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    m = offs < inner

    pa = a_ptr + o * sa_o + offs * sa_i
    pb = b_ptr + o * sb_o + offs * sb_i

    a0 = tl.load(pa, mask=m, other=0.0)
    a1 = tl.load(pa + sa_c, mask=m, other=0.0)
    a2 = tl.load(pa + 2 * sa_c, mask=m, other=0.0)
    b0 = tl.load(pb, mask=m, other=0.0)
    b1 = tl.load(pb + sb_c, mask=m, other=0.0)
    b2 = tl.load(pb + 2 * sb_c, mask=m, other=0.0)

    po = o_ptr + o * so_o + offs * so_i
    tl.store(po, a1 * b2 - a2 * b1, mask=m)
    tl.store(po + so_c, a2 * b0 - a0 * b2, mask=m)
    tl.store(po + 2 * so_c, a0 * b1 - a1 * b0, mask=m)


def _pick_block(n_elem, esize):
    # Eight live input tiles (+ one output tile) have to coexist in UB; the
    # compiler double-buffers them, so keep BLOCK small enough for 4-byte
    # elements to fit.
    if n_elem <= 8192:
        return 128
    if n_elem <= 131072:
        return 1024
    return 2048 if esize == 4 else 2048


def _default_dim(input, other):
    broadcast_shape = torch.broadcast_shapes(input.shape, other.shape)
    for axis, size in enumerate(broadcast_shape):
        if size == 3:
            return axis
    raise RuntimeError(
        "no dimension of size 3 in input; cross requires a dimension of length 3"
    )


def _outer_stride(t, dim):
    if dim == 0:
        return 0
    return t.stride(dim - 1)


def _broadcast_strides(t, out_shape):
    """Strides of ``t`` expressed against ``out_shape`` (0 where broadcast)."""
    nd = len(out_shape)
    lead = nd - t.dim()
    res = [0] * nd
    for i in range(t.dim()):
        d = lead + i
        if t.shape[i] == 1 and out_shape[d] != 1:
            res[d] = 0
        else:
            res[d] = t.stride(i)
    return res


def run(input, other, dim=None, *, out):
    if dim is None:
        dim = _default_dim(input, other)
    nd = input.dim()
    if dim < 0:
        dim += nd

    if (
        dim == nd - 1
        and input.shape == out.shape
        and other.shape == out.shape
        and out.dtype == input.dtype
        and input.is_contiguous()
        and other.is_contiguous()
        and out.is_contiguous()
    ):
        n_elem = out.numel()
        if n_elem:
            blk = _pick_block(n_elem, out.element_size())
            _cross_flat_kernel[(triton.cdiv(n_elem, blk),)](
                input, other, out, n_elem, BLOCK=blk
            )
        return out

    # ---- general path: validate / resize like the aten out-variant ----
    out_shape = torch.broadcast_shapes(input.shape, other.shape)
    if input.shape[dim] != 3 or other.shape[dim] != 3:
        raise RuntimeError("cross: dimension must be of size 3")
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"cross: expected out dtype {input.dtype}, but got {out.dtype}"
        )
    if out.device != input.device:
        raise RuntimeError("cross: out must be on the same device as input")
    if tuple(out.shape) != tuple(out_shape):
        out.resize_(out_shape)

    nd_out = len(out_shape)
    sa = _broadcast_strides(input, out_shape)
    sb = _broadcast_strides(other, out_shape)
    so = _broadcast_strides(out, out_shape)

    outer = 1
    for s in out_shape[:dim]:
        outer *= s
    inner = 1
    for s in out_shape[dim + 1 :]:
        inner *= s
    if inner * outer == 0:
        return out

    blk = 64
    _cross_general_kernel[(outer, triton.cdiv(inner, blk))](
        input,
        other,
        out,
        inner,
        _outer_stride(input, dim),
        sa[dim],
        sa[dim + 1] if dim + 1 < nd_out else 0,
        _outer_stride(other, dim),
        sb[dim],
        sb[dim + 1] if dim + 1 < nd_out else 0,
        _outer_stride(out, dim),
        so[dim],
        so[dim + 1] if dim + 1 < nd_out else 0,
        BLOCK_I=blk,
    )
    return out

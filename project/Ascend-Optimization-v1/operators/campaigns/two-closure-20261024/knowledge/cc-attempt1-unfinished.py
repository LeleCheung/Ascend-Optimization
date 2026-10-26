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
"""Triton implementation of ``aten.upsample_bilinear2d`` (NCHW bilinear).

Architecture (measured on the target, not assumed)
--------------------------------------------------
Diagnostic probes on this Ascend backend established:

* a flat contiguous ``tl.load``/``tl.store`` with a *scalar* base reaches
  ~600-1000 GB/s, and a base offset of one element costs nothing
  (57.7 us vs 58.0 us for the same volume), so shifted *loads* are free;
* ``tl.gather`` costs ~24 ns per gathered element and ``off % IW`` (vector
  integer division) is 10x worse still, so both are banned from hot paths;
* a naive two-kernel separation is unusable: one Triton launch already costs
  ~62 us of host time, and the harness reports max(host, device) per call.

So every case is served by exactly **one** launch that keeps all addressing
affine and all arithmetic elementwise.

``_up2x_kernel`` (fast path, ``OH == 2*IH`` and ``OW == 2*IW`` with
``align_corners=False`` and no explicit scales)
    For this ratio PyTorch's coordinate mapping collapses to fixed weights:
    even output index ``2q`` takes ``0.25*src[q-1] + 0.75*src[q]`` and odd
    ``2q+1`` takes ``0.75*src[q] + 0.25*src[q+1]`` (clamped at the borders).
    One program owns ``ROWS`` whole input rows of one image and issues six
    contiguous loads: three row positions (q-1, q, q+1) times two column
    offsets (-1, 0).  The ``-1`` column offset makes lane ``k`` hold
    ``v[k-1]`` and ``v[k]`` at once, which is exactly the pair needed for the
    two adjacent output columns ``2k-1`` and ``2k``; joining them yields a
    fully contiguous store of ``ROWS`` output rows.  Lane 0 covers column -1
    and is masked; output columns ``0`` and ``OW-1`` of every row are written
    by a tiny ``(ROWS, 2)`` side store, so no per-element column clamp is ever
    needed.  The only border fix-ups (``blk == 0`` / ``blk == NBLK-1``) sit in
    scalar-uniform branches, so interior programs pay nothing for them.

``_bilinear_flat_kernel`` (general path)
    A flat 1-D tile over ``N*C*OH*OW`` with the four neighbours read through
    the real input strides; it covers every shape the fast path declines
    (non-2x ratios, ``align_corners=True``, 1x1 outputs, non-unit minor
    stride, explicit scales).

Both kernels reproduce PyTorch's coordinate mapping exactly:
``src = idx*(in-1)/(out-1)`` for ``align_corners`` (0 when ``out == 1``),
otherwise ``src = max((idx + 0.5)*scale - 0.5, 0)`` with ``scale = 1/scales_*``
when given and positive, else ``in/out``; the right/bottom neighbour is
``min(i0 + 1, size - 1)``, matching PyTorch's ``i1p`` edge rule.
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# fast path: exact 2x in both dimensions, affine addressing only
# --------------------------------------------------------------------------
@triton.jit
def _up2x_kernel(
    o,
    i,
    NIMG: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    ROWS: tl.constexpr,
    NBLK: tl.constexpr,
):
    pid = tl.program_id(0)
    img = pid // NBLK
    blk = pid - img * NBLK
    off = tl.arange(0, ROWS * IW)
    b0 = img * (IH * IW) + blk * (ROWS * IW)

    # three row positions: A = q-1, B = q, C = q+1 (contiguous row ranges)
    if pid == 0:
        a0 = tl.load(i + b0 - IW + off, mask=off >= IW, other=0.0).to(tl.float32)
        bm = tl.load(i + b0 - 1 + off, mask=off >= 1, other=0.0).to(tl.float32)
        am = tl.load(i + b0 - IW - 1 + off, mask=off >= IW + 1, other=0.0).to(tl.float32)
    else:
        a0 = tl.load(i + b0 - IW + off).to(tl.float32)
        bm = tl.load(i + b0 - 1 + off).to(tl.float32)
        am = tl.load(i + b0 - IW - 1 + off).to(tl.float32)
    b0v = tl.load(i + b0 + off).to(tl.float32)

    # vertical 2x for the even output rows (q-1, q) and its -1-column partner
    ve0 = 0.25 * a0 + 0.75 * b0v
    vem = 0.25 * am + 0.75 * bm
    if blk == 0:
        # first row of the image: q-1 clamps to q
        ve0 = tl.where(off < IW, b0v, ve0)
        vem = tl.where(off < IW, bm, vem)

    rows = blk * ROWS + tl.arange(0, ROWS)
    obase = img * (2 * IH * 2 * IW)
    k = tl.arange(0, IW)
    j = tl.arange(0, 2)
    ooff = (obase + (rows * 2)[:, None, None] * (2 * IW)
            + k[None, :, None] * 2 - 1 + j[None, None, :])
    msk = tl.broadcast_to((k > 0)[None, :, None], (ROWS, IW, 2))
    he = tl.reshape(tl.join(tl.reshape(vem, (ROWS, IW)),
                            tl.reshape(ve0, (ROWS, IW))), (ROWS, IW, 2))
    tl.store(o + ooff, he.to(o.dtype.element_ty), mask=msk)

    # vertical 2x for the odd output rows (q, q+1)
    if pid == NIMG * NBLK - 1:
        c0 = tl.load(i + b0 + IW + off, mask=off < ROWS * IW - IW, other=0.0).to(tl.float32)
        cm = tl.load(i + b0 + IW - 1 + off, mask=off < ROWS * IW - IW + 1,
                     other=0.0).to(tl.float32)
    else:
        c0 = tl.load(i + b0 + IW + off).to(tl.float32)
        cm = tl.load(i + b0 + IW - 1 + off).to(tl.float32)
    vd0 = 0.75 * b0v + 0.25 * c0
    vdm = 0.75 * bm + 0.25 * cm
    if blk == NBLK - 1:
        # last row of the image: q+1 clamps to q
        vd0 = tl.where(off >= ROWS * IW - IW, b0v, vd0)
        vdm = tl.where(off >= ROWS * IW - IW, bm, vdm)

    hd = tl.reshape(tl.join(tl.reshape(vdm, (ROWS, IW)),
                            tl.reshape(vd0, (ROWS, IW))), (ROWS, IW, 2))
    tl.store(o + ooff + 2 * IW, hd.to(o.dtype.element_ty), mask=msk)

    # side store: output columns 0 and OW-1 of every row (clamped weights make
    # both equal the vertical interpolant at columns 0 and IW-1)
    ridx = tl.arange(0, ROWS)
    colsel = tl.arange(0, 2) * (IW - 1)
    roff = (ridx * IW)[:, None] + colsel[None, :]
    fb = tl.load(i + b0 + roff).to(tl.float32)
    if blk == 0:
        fa = tl.load(i + b0 - IW + roff, mask=(ridx > 0)[:, None],
                     other=0.0).to(tl.float32)
        fa = tl.where((ridx == 0)[:, None], fb, fa)
    else:
        fa = tl.load(i + b0 - IW + roff).to(tl.float32)
    if blk == NBLK - 1:
        fc = tl.load(i + b0 + IW + roff, mask=(ridx < ROWS - 1)[:, None],
                     other=0.0).to(tl.float32)
        fc = tl.where((ridx == ROWS - 1)[:, None], fb, fc)
    else:
        fc = tl.load(i + b0 + IW + roff).to(tl.float32)
    ec = 0.25 * fa + 0.75 * fb
    dc = 0.75 * fb + 0.25 * fc
    colout = tl.arange(0, 2) * (2 * IW - 1)
    eoff = obase + (rows * 2)[:, None] * (2 * IW) + colout[None, :]
    tl.store(o + eoff, ec.to(o.dtype.element_ty))
    tl.store(o + eoff + 2 * IW, dc.to(o.dtype.element_ty))


# --------------------------------------------------------------------------
# general path: flat 1-D tile, neighbours read through the real strides
# --------------------------------------------------------------------------
@triton.jit
def _bilinear_flat_kernel(
    ptr_o,
    ptr_i,
    total,
    s_n,
    s_c,
    s_h,
    s_w,
    rh,
    rw,
    C: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OW
    row = offs // OW
    oh = row % OH
    nc = row // OH

    if ALIGN_CORNERS:
        if OH > 1:
            real_h = oh.to(tl.float32) * ((IH - 1.0) / (OH - 1.0))
        else:
            real_h = tl.zeros([BLOCK], dtype=tl.float32)
        if OW > 1:
            real_w = ow.to(tl.float32) * ((IW - 1.0) / (OW - 1.0))
        else:
            real_w = tl.zeros([BLOCK], dtype=tl.float32)
    else:
        real_h = tl.maximum((oh.to(tl.float32) + 0.5) * rh - 0.5, 0.0)
        real_w = tl.maximum((ow.to(tl.float32) + 0.5) * rw - 0.5, 0.0)

    h0 = real_h.to(tl.int32)
    w0 = real_w.to(tl.int32)
    hl = real_h - h0.to(tl.float32)
    wl = real_w - w0.to(tl.float32)

    h0 = tl.minimum(h0, IH - 1)
    w0 = tl.minimum(w0, IW - 1)
    h1 = tl.minimum(h0 + 1, IH - 1)
    w1 = tl.minimum(w0 + 1, IW - 1)

    n = nc // C
    c = nc - n * C
    base = n * s_n + c * s_c
    r0 = base + h0 * s_h
    r1 = base + h1 * s_h
    c0 = w0 * s_w
    c1 = w1 * s_w

    v00 = tl.load(ptr_i + (r0 + c0), mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(ptr_i + (r0 + c1), mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(ptr_i + (r1 + c0), mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(ptr_i + (r1 + c1), mask=mask, other=0.0).to(tl.float32)

    w0l = 1.0 - wl
    h0l = 1.0 - hl
    top = w0l * v00 + wl * v01
    bot = w0l * v10 + wl * v11
    val = h0l * top + hl * bot

    tl.store(ptr_o + offs, val.to(ptr_o.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------
def _reciprocal_scale(in_size, out_size, align_corners, scale):
    """PyTorch ``area_pixel_compute_scale`` (accscalar_t == float32 path)."""
    if align_corners:
        if out_size > 1:
            return (in_size - 1) / (out_size - 1)
        return 0.0
    if scale is not None and scale > 0:
        return 1.0 / float(scale)
    return in_size / out_size


def _pick_flat_block(total):
    if total >= (1 << 20):
        return 2048
    if total >= (1 << 16):
        return 1024
    if total >= (1 << 12):
        return 512
    return 256


def _pick_rows(ih, iw):
    """Rows per program for the 2x path: must divide ``ih`` and keep the tile
    (``ROWS*IW`` elements) inside the UB budget of the backend."""
    for rows in (16, 8, 4):
        if ih % rows == 0 and rows * iw <= 2048:
            return rows
    return 0


_LAUNCH_CACHE = {}


def _build(n, c, ih, iw, oh, ow, st, ac, sh, sw):
    """Resolve one metadata entry: kernel + grid + positional launch arguments.

    Only scalars and the compiled kernel handle are stored; no tensor, storage
    or data pointer is cached, and the cache key carries shape, strides, dtype
    and the coordinate options.
    """
    nimgs = n * c
    fast = (
        not ac
        and sh is None
        and sw is None
        and oh == 2 * ih
        and ow == 2 * iw
        and ih >= 4
        and iw >= 4
        and (iw & (iw - 1)) == 0
        and st[3] == 1
        and st[2] == iw
        and st[1] == ih * iw
        and st[0] == c * ih * iw
    )
    if fast:
        rows = _pick_rows(ih, iw)
        if rows:
            nblk = ih // rows
            return (_up2x_kernel, (nimgs * nblk,), (nimgs, ih, iw, rows, nblk))
    total = nimgs * oh * ow
    block = _pick_flat_block(total)
    rh = _reciprocal_scale(ih, oh, ac, sh)
    rw = _reciprocal_scale(iw, ow, ac, sw)
    return (_bilinear_flat_kernel, (triton.cdiv(total, block),),
            (total, st[0], st[1], st[2], st[3], rh, rw, c, oh, ow, ih, iw,
             bool(ac), block))


def run(input, output_size, align_corners=False, scales_h=None, scales_w=None):
    n, c, ih, iw = input.shape
    oh = int(output_size[0])
    ow = int(output_size[1])

    key = (n, c, ih, iw, oh, ow, input.stride(), align_corners, scales_h,
           scales_w, input.dtype)
    ent = _LAUNCH_CACHE.get(key)
    if ent is None:
        ent = _build(n, c, ih, iw, oh, ow, input.stride(),
                     bool(align_corners), scales_h, scales_w)
        _LAUNCH_CACHE[key] = ent

    out = torch.empty((n, c, oh, ow), device=input.device, dtype=input.dtype)
    if n * c * oh * ow == 0:
        return out
    ent[0][ent[1]](out, input, *ent[2])
    return out

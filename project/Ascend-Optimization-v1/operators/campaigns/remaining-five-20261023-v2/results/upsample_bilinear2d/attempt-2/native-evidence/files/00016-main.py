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

Round 3 architecture
--------------------
Target probes on the real device (measured, not assumed):

* A *flat 1-D contiguous* GM block transfer is ~6x more efficient per element
  than the same volume expressed as a 2-D tile whose row bases come from a
  computed vector, even when that vector is affine.
* ``tl.gather`` on a flat vector costs ~10 us per program (it scalarises);
  on the minor axis of a 2-D tile it is far cheaper but still measurable.
* A flat load whose base is off by one *element* costs only ~3% more, while a
  2-D tile with a one-element minor-axis offset costs ~2.2x.
* Per-program fixed cost dominates below ~4-8 K elements per program.

Hence the 2x fast path is written entirely with flat 1-D blocks and obtains the
+-1 column neighbours from extra (cheap) shifted flat loads instead of gathers,
and the two row parities are written with two affine 2-D stores whose rows are
``2*W2`` apart.

``_bilinear_2x_kernel`` (fast path)
    Requires ``align_corners=False``, no explicit ``scales_*``, a contiguous
    input, ``OH == 2*IH``, ``OW == 2*IW`` and ``IW`` a power of two.  For that
    geometry ``scale == 0.5`` exactly, so PyTorch's coordinate mapping is
    ``real = 0.5*idx - 0.25`` and the weights are the constants
    ``(0.25, 0.75)`` / ``(0.75, 0.25)``; the ``>= 0`` clamp reproduces the
    ``lambda == 0`` border exactly because ``0.25*v + 0.75*v == v`` to within
    one ulp.  One program owns ``R`` input rows of one ``(n, c)`` image, so the
    three source row-blocks, their +-1 column shifts and the two output row
    parities are all plain contiguous ranges.

``_bilinear_flat_kernel`` (general path)
    A flat 1-D tile over ``N*C*OH*OW`` with the four neighbours read through
    the real input strides; it covers every shape the fast path declines
    (non-2x ratios, ``align_corners=True``, 1x1 outputs, non-contiguous inputs,
    explicit scales).
"""

import torch
import triton
import triton.language as tl

_FLAT_ELEMS = 2048
_FAST_ELEMS = 2048


# --------------------------------------------------------------------------
# fast path: exact 2x in both dimensions, flat contiguous blocks only
# --------------------------------------------------------------------------
@triton.jit
def _bilinear_2x_kernel(
    ptr_o,
    ptr_i,
    total_in,
    qper,
    R: tl.constexpr,
    W: tl.constexpr,
    W2: tl.constexpr,
    IMG_IN: tl.constexpr,
    IMG_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    nc = pid // qper
    qb = pid - nc * qper
    q0 = qb * R

    N: tl.constexpr = R * W
    m = tl.arange(0, N)
    inb = nc * IMG_IN
    fa = inb + (q0 - 1) * W + m
    fb = fa + 2 * W

    # flat block of the two source rows and their +-1 column neighbours
    P = tl.load(ptr_i + fa, mask=fa >= 0, other=0.0).to(tl.float32)
    Q = tl.load(ptr_i + (fa + W)).to(tl.float32)
    S = tl.load(ptr_i + fb, mask=fb < total_in, other=0.0).to(tl.float32)
    Pl = tl.load(ptr_i + (fa - 1), mask=fa >= 1, other=0.0).to(tl.float32)
    Ql = tl.load(ptr_i + (fa + W - 1)).to(tl.float32)
    Sl = tl.load(ptr_i + (fb - 1), mask=fb <= total_in, other=0.0).to(tl.float32)
    Pr = tl.load(ptr_i + (fa + 1)).to(tl.float32)
    Qr = tl.load(ptr_i + (fa + W + 1)).to(tl.float32)
    Sr = tl.load(ptr_i + (fb + 1), mask=fb < total_in - 1, other=0.0).to(tl.float32)

    # Border rows: row q0-1 of the first block and row q0+R of the last block
    # clamp onto row q0 and row q0+R-1, which is exactly PyTorch's i1p rule.
    top = (m < W) & (q0 == 0)
    bot = (m >= N - W) & (qb == qper - 1)
    P = tl.where(top, Q, P)
    Pl = tl.where(top, Ql, Pl)
    Pr = tl.where(top, Qr, Pr)
    S = tl.where(bot, Q, S)
    Sl = tl.where(bot, Ql, Sl)
    Sr = tl.where(bot, Qr, Sr)

    ve = 0.25 * P + 0.75 * Q
    vel = 0.25 * Pl + 0.75 * Ql
    ver = 0.25 * Pr + 0.75 * Qr
    vo = 0.75 * Q + 0.25 * S
    vol = 0.75 * Ql + 0.25 * Sl
    vor = 0.75 * Qr + 0.25 * Sr

    col = m % W
    c0 = col == 0
    c1 = col == W - 1
    ee = tl.where(c0, ve, 0.25 * vel + 0.75 * ve)
    eo = tl.where(c1, ve, 0.75 * ve + 0.25 * ver)
    oe = tl.where(c0, vo, 0.25 * vol + 0.75 * vo)
    oo = tl.where(c1, vo, 0.75 * vo + 0.25 * vor)

    te = tl.reshape(tl.join(ee, eo), (R, W2))
    to = tl.reshape(tl.join(oe, oo), (R, W2))

    r = tl.arange(0, R)
    c = tl.arange(0, W2)
    off = nc * IMG_OUT + (2 * q0) * W2 + (r * (2 * W2))[:, None] + c[None, :]
    tl.store(ptr_o + off, te.to(ptr_o.dtype.element_ty))
    tl.store(ptr_o + off + W2, to.to(ptr_o.dtype.element_ty))


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
        return _FLAT_ELEMS
    if total >= (1 << 16):
        return 1024
    if total >= (1 << 12):
        return 512
    return 256


def _pick_rows(iw, ih):
    """Input rows per program for the 2x path, kept within the UB budget and
    small enough that ``R`` divides ``ih``."""
    r = max(1, _FAST_ELEMS // iw)
    r = 1 << min(4, r.bit_length() - 1)
    while r > 1 and ih % r != 0:
        r >>= 1
    return max(1, r)


def _resolve(shape, strides, out_size, ac, scales_h, scales_w, contiguous):
    """Resolve everything that depends only on tensor *metadata*.

    Only scalars are produced; no tensor, storage or data pointer is cached,
    and the caller's cache key contains shape, strides, dtype, output size and
    the coordinate options, so a different view or layout derives its own
    entry instead of reusing a stale one.
    """
    N, C, IH, IW = shape
    OH, OW = out_size
    s_n, s_c, s_h, s_w = strides

    rh = _reciprocal_scale(IH, OH, ac, scales_h)
    rw = _reciprocal_scale(IW, OW, ac, scales_w)

    if (
        contiguous
        and not ac
        and scales_h is None
        and scales_w is None
        and OH == 2 * IH
        and OW == 2 * IW
        and IW & (IW - 1) == 0
        and IW >= 16
    ):
        r = _pick_rows(IW, IH)
        qper = IH // r
        grid = (N * C * qper,)
        return (True, grid,
                (N * C * IH * IW, qper, r, IW, 2 * IW, IH * IW,
                 4 * IH * IW))

    total = N * C * OH * OW
    block = _pick_flat_block(total)
    return (False, (triton.cdiv(total, block),),
            (total, s_n, s_c, s_h, s_w, rh, rw, C, OH, OW, IH, IW, ac, block))


_LAUNCH_CACHE = {}


def run(input, output_size, align_corners=False, scales_h=None, scales_w=None):
    N, C, IH, IW = input.shape
    OH = int(output_size[0])
    OW = int(output_size[1])

    out = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    if N * C * OH * OW == 0:
        return out

    key = (N, C, IH, IW, OH, OW, bool(align_corners), scales_h, scales_w,
           input.stride(), input.dtype)
    ent = _LAUNCH_CACHE.get(key)
    if ent is None:
        ent = _resolve((N, C, IH, IW), input.stride(), (OH, OW),
                       bool(align_corners), scales_h, scales_w,
                       input.is_contiguous())
        _LAUNCH_CACHE[key] = ent

    if ent[0]:
        total_in, qper, r, iw, w2, img_in, img_out = ent[2]
        _bilinear_2x_kernel[ent[1]](
            out,
            input,
            total_in,
            qper,
            R=r,
            W=iw,
            W2=w2,
            IMG_IN=img_in,
            IMG_OUT=img_out,
        )
    else:
        total, s_n, s_c, s_h, s_w, rh, rw, kc, koh, kow, kih, kiw, ac, block = ent[2]
        _bilinear_flat_kernel[ent[1]](
            out,
            input,
            total,
            s_n,
            s_c,
            s_h,
            s_w,
            rh,
            rw,
            C=kc,
            OH=koh,
            OW=kow,
            IH=kih,
            IW=kiw,
            ALIGN_CORNERS=ac,
            BLOCK=block,
        )
    return out

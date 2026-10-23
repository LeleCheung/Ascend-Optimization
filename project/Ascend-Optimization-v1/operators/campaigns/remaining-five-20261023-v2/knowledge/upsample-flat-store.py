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

Round 2 architecture
--------------------
Round 1 measured 37.8 ns per output element, exactly linear in the element
count and identical for fp16/bf16/fp32.  A target probe showed a contiguous
block copy of the same volume needs ~0.03 ms net, so the cost is per-lane
addressing: ``tl.load(ptr + <vector of unrelated indices>)`` lowers to
per-lane non-affine loads, roughly 300x slower than block transfers.

Two kernels are therefore used.

``_bilinear_int_kernel`` (fast path)
    Taken when ``align_corners`` is False, no explicit ``scales_*`` is given,
    the last input stride is 1 and ``OW == 2*IW``.  One program owns ``BR``
    whole output rows.  Every memory access keeps a *contiguous minor
    dimension*: the two source rows are found from a data-dependent row base
    and read along the row (row-gather with a contiguous minor dimension,
    which the backend handles well), and the horizontal resample is expressed
    as an affine expansion -- output column ``2*k + j`` is selected between
    the two interpolants of input column ``k`` on a constexpr axis ``j``, so
    no index vector ever addresses memory.  The store is contiguous over the
    whole ``(BR, IW, 2)`` tile, i.e. over ``BR`` complete output rows.

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
from flag_gems.utils import libentry

# tuning knobs (overridable by diagnostics)
_TARGET_PROGRAMS = 256
_UB_ELEMS = 2048
_FLAT_ELEMS = 2048


# --------------------------------------------------------------------------
# fast path: exact 2x horizontal expansion, contiguous minor dimension
# --------------------------------------------------------------------------
@libentry()
@triton.jit
def _bilinear_int_kernel(
    ptr_o,
    ptr_i,
    n_rows,
    s_n,
    s_c,
    s_h,
    rh,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BR: tl.constexpr,
    IWP: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    oh = rows % OH
    nc = rows // OH
    n = nc // C
    c = nc - n * C

    real_h = tl.maximum((oh.to(tl.float32) + 0.5) * rh - 0.5, 0.0)
    h0f = real_h.to(tl.int32)
    hl = real_h - h0f.to(tl.float32)
    h0 = tl.minimum(h0f, IH - 1)
    h1 = tl.minimum(h0 + 1, IH - 1)

    base = n * s_n + c * s_c
    rr0 = (base + h0 * s_h)[:, None]
    rr1 = (base + h1 * s_h)[:, None]

    k = tl.arange(0, IWP)
    ka = k[None, :]
    lm = tl.broadcast_to((ka < IW) & (rows[:, None] < n_rows), (BR, IWP))
    il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (BR, IWP))
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (BR, IWP))

    hl2 = hl[:, None]
    h0l2 = 1.0 - hl2

    # Two *aligned* row loads only.  A load whose base is off by one element
    # costs ~25x more than an aligned one on this backend (measured: 6 loads
    # with +-1 taps 21.3 ms vs 2 aligned loads 0.59 ms on the 8.4M-output
    # geometry), so the +-1 column neighbours are produced in registers with
    # tl.gather, which measured nearly free (0.59 -> 0.69 ms).
    v = h0l2 * tl.load(ptr_i + (rr0 + ka), mask=lm, other=0.0).to(tl.float32) \
        + hl2 * tl.load(ptr_i + (rr1 + ka), mask=lm, other=0.0).to(tl.float32)

    vm = tl.gather(v, il, 1)
    vp = tl.gather(v, ir, 1)

    even = 0.25 * vm + 0.75 * v
    odd = 0.75 * v + 0.25 * vp

    # (BR, IWP, 2) tile: its row-major order is exactly BR output rows of
    # OW = 2*IW contiguous elements.
    val = tl.join(even, odd)
    if IWP == IW:
        # Rebuild a single affine store vector instead of the rank-3 index
        # expression, preserving the exact row-major join order.
        flat_off = pid * BR * OW + tl.arange(0, BR * OW)
        flat_val = tl.reshape(val, (BR * OW,))
        tl.store(ptr_o + flat_off, flat_val.to(ptr_o.dtype.element_ty), mask=flat_off < n_rows * OW)
    else:
        j = tl.arange(0, 2)
        ooff = (rows * OW)[:, None, None] + (ka * 2)[:, :, None] + j[None, None, :]
        om = tl.broadcast_to(
            (ka < IW)[:, :, None] & (rows[:, None, None] < n_rows), (BR, IWP, 2)
        )
        tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty), mask=om)


# --------------------------------------------------------------------------
# general path: flat 1-D tile, neighbours read through the real strides
# --------------------------------------------------------------------------
@libentry()
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


def _pick_rows_per_program(iwp, n_rows):
    """Rows per program: bounded by the UB tile budget and by a program count
    that keeps per-program launch overhead small."""
    cap = max(1, _UB_ELEMS // iwp)
    want = max(1, triton.cdiv(n_rows, _TARGET_PROGRAMS))
    br = min(cap, want)
    br = 1 << (br.bit_length() - 1)
    # prefer a divisor of n_rows so the row mask is entirely unnecessary
    while br > 1 and n_rows % br != 0:
        br >>= 1
    return max(1, br)


def _resolve(shape, strides, out_size, ac, scales_h, scales_w):
    """Resolve everything that depends only on tensor *metadata*.

    Only scalars are produced; no tensor, storage or data pointer is cached,
    and the caller's cache key contains shape, strides, dtype, output size and
    the coordinate options, so a different view or layout derives its own
    entry instead of reusing a stale one.
    """
    N, C, IH, IW = shape
    OH, OW = out_size
    s_n, s_c, s_h, s_w = strides
    n_rows = N * C * OH

    rh = _reciprocal_scale(IH, OH, ac, scales_h)
    rw = _reciprocal_scale(IW, OW, ac, scales_w)

    if (
        not ac
        and scales_h is None
        and scales_w is None
        and s_w == 1
        and IW > 1
        and OW == 2 * IW
    ):
        iwp = triton.next_power_of_2(IW)
        br = _pick_rows_per_program(iwp, n_rows)
        return (True, (triton.cdiv(n_rows, br),),
                (n_rows, s_n, s_c, s_h, rh, C, IH, IW, OH, OW, br, iwp))
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
                       bool(align_corners), scales_h, scales_w)
        _LAUNCH_CACHE[key] = ent

    if ent[0]:
        n_rows, s_n, s_c, s_h, rh, kc, kih, kiw, koh, kow, br, iwp = ent[2]
        _bilinear_int_kernel[ent[1]](
            out,
            input,
            n_rows,
            s_n,
            s_c,
            s_h,
            rh,
            C=kc,
            IH=kih,
            IW=kiw,
            OH=koh,
            OW=kow,
            BR=br,
            IWP=iwp,
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

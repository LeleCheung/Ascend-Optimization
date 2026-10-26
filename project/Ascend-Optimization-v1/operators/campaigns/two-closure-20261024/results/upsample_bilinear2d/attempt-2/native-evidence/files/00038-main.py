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
Round 2 measured the cost structure of the previous single-output-row kernel:
on the 4x32x128x128 -> 256x256 geometry it issued two 256-byte row loads per
output row (65536 transfers, ~45 GB/s effective, 316 us of 576 us device
time), while the whole 16.8 MB store cost only ~19 us.  The addressing, not
the volume, was the bottleneck.

``_blk2x_kernel`` (exact 2x fast path)
    One program owns ``QB`` consecutive input rows of one channel image and
    reads two *multi-KB contiguous blocks* (rows ``lb .. lb+QB-1`` and
    ``lb+1 .. lb+QB``), so the whole vertical neighbourhood of 2*QB output
    rows arrives in two DMA transfers instead of 4*QB row fragments.

    For the fixed 2x coordinate map (``align_corners=False``, no explicit
    scales, ``OH == 2*IH``, ``OW == 2*IW``) the weights are exactly
    ``1/4, 3/4``, so the two source rows ``(r-1, r)`` give both output rows
    that depend on them:
        even row 2r     = 0.25*V[r-1] + 0.75*V[r]
        odd  row 2(r-1)+1 = 0.75*V[r-1] + 0.25*V[r]
    i.e. one pair of mixed tiles serves two output rows, and the odd tile is
    simply stored one row lower.  No per-row index vector is ever built; the
    only data-dependent scalar is the row base ``lb``.

    The horizontal expansion keeps the round-2 trick: the +-1 column
    neighbours are produced with ``tl.gather`` inside the tile (measured
    ~90 us) rather than by misaligned row loads (measured 21 ms in round 1).
    Round 3 probes measured the two alternatives on this target and both were
    far worse: CANN ``insert_slice``/``extract_slice`` static-offset slices
    either overflow UB (flat form) or run 7x slower than the gather (2-D
    form), and +-1 element shifted block loads cost 6x more than the gather
    because their arithmetic no longer stays on the 2-D tile.

    ``lb = max(r0 - 1, 0)`` makes the first block of every image use rows
    ``(0, 1)``; that reproduces PyTorch's clamped top edge exactly, because
    ``0.25*x + 0.75*x == x``.  Consequence of the store offset: the block
    grid covers output rows ``1 .. OH-2`` only, so ``_edge2x_kernel`` writes
    the two clamped end rows (``0`` and ``OH-1``) of every image, where the
    reference value is just the horizontally resampled edge input row.

    Every write of the block grid is in bounds and adjacent blocks only ever
    write the same value to a shared row, so no store mask is needed.

``_bilinear_int_kernel`` / ``_bilinear_flat_kernel``
    Unchanged round-2 kernels covering the whole remaining domain: 2x width
    with non-contiguous strides, non-2x ratios, ``align_corners=True``,
    explicit scales, 1x1 outputs and non power-of-two widths.

All three paths reproduce PyTorch's coordinate mapping:
``src = idx*(in-1)/(out-1)`` for ``align_corners`` (0 when ``out == 1``),
otherwise ``src = max((idx + 0.5)*scale - 0.5, 0)`` with ``scale = 1/scales_*``
when given and positive, else ``in/out``; the right/bottom neighbour is
``min(i0 + 1, size - 1)``, matching PyTorch's ``i1p`` edge rule.
"""

import torch
import torch_npu  # noqa: F401  (raw current-stream query, as in round 2)
import triton
import triton.language as tl

# tuning knobs (overridable by diagnostics)
_TARGET_PROGRAMS = 256
_UB_ELEMS = 2048
_FLAT_ELEMS = 2048
_BLK_ELEMS = 2048      # elements per contiguous input block (QB * IW)
_BLK_MIN_PROGRAMS = 128


# --------------------------------------------------------------------------
# exact 2x fast path: contiguous input blocks, two output rows per mix
# --------------------------------------------------------------------------
@triton.jit
def _blk2x_kernel(
    ptr_o,
    ptr_i,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    NB: tl.constexpr,
    QB: tl.constexpr,
    H16: tl.constexpr,
):
    pid = tl.program_id(0)
    img = pid // NB
    b = pid - img * NB
    r0 = b * QB
    # first block of an image starts at row 0 so both loads stay in bounds;
    # 0.25*x + 0.75*x == x reproduces the clamped top edge exactly.
    lb = tl.maximum(r0 - 1, 0)

    L: tl.constexpr = QB * IW
    offs = tl.arange(0, L)
    ibase = img * (IH * IW) + lb * IW
    f = tl.load(ptr_i + ibase + offs)
    g = tl.load(ptr_i + ibase + IW + offs)

    if H16:
        vf = tl.reshape(0.25 * f + 0.75 * g, (QB, IW))
        vo = tl.reshape(0.75 * f + 0.25 * g, (QB, IW))
    else:
        ff = f.to(tl.float32)
        gg = g.to(tl.float32)
        vf = tl.reshape(0.25 * ff + 0.75 * gg, (QB, IW))
        vo = tl.reshape(0.75 * ff + 0.25 * gg, (QB, IW))

    k = tl.arange(0, IW)
    il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (QB, IW))
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (QB, IW))

    even_f = 0.25 * tl.gather(vf, il, 1) + 0.75 * vf
    odd_f = 0.75 * vf + 0.25 * tl.gather(vf, ir, 1)
    even_o = 0.25 * tl.gather(vo, il, 1) + 0.75 * vo
    odd_o = 0.75 * vo + 0.25 * tl.gather(vo, ir, 1)

    pe = tl.reshape(tl.join(even_f, odd_f), (QB, OW))
    po = tl.reshape(tl.join(even_o, odd_o), (QB, OW))

    oo = (2 * (lb + tl.arange(0, QB)) * OW)[:, None] + tl.arange(0, OW)[None, :]
    obase = img * (OH * OW)
    tl.store(ptr_o + obase + oo + 2 * OW, pe.to(ptr_o.dtype.element_ty))
    tl.store(ptr_o + obase + oo + OW, po.to(ptr_o.dtype.element_ty))


@triton.jit
def _edge2x_kernel(
    ptr_o,
    ptr_i,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    H16: tl.constexpr,
):
    """Output rows 0 and OH-1 of every image: the value is the horizontally
    resampled clamped edge input row (vertical weights 1/4 + 3/4 on the same
    row).  ``e`` is 0 for the top edge and 1 for the bottom edge."""
    pid = tl.program_id(0)
    img = pid // 2
    e = pid - img * 2
    k = tl.arange(0, IW)
    src = img * (IH * IW) + e * (IH - 1) * IW
    v = tl.load(ptr_i + src + k)
    if H16:
        even = 0.25 * tl.gather(v, tl.maximum(k - 1, 0), 0) + 0.75 * v
        odd = 0.75 * v + 0.25 * tl.gather(v, tl.minimum(k + 1, IW - 1), 0)
    else:
        fv = v.to(tl.float32)
        even = 0.25 * tl.gather(fv, tl.maximum(k - 1, 0), 0) + 0.75 * fv
        odd = 0.75 * fv + 0.25 * tl.gather(fv, tl.minimum(k + 1, IW - 1), 0)
    val = tl.reshape(tl.join(even, odd), (OW,))
    dst = img * (OH * OW) + e * (OH - 1) * OW
    tl.store(ptr_o + dst + tl.arange(0, OW), val.to(ptr_o.dtype.element_ty))


# --------------------------------------------------------------------------
# round-2 fast path: exact 2x horizontal expansion, per-output-row loads
# --------------------------------------------------------------------------
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

    v = h0l2 * tl.load(ptr_i + (rr0 + ka), mask=lm, other=0.0).to(tl.float32) \
        + hl2 * tl.load(ptr_i + (rr1 + ka), mask=lm, other=0.0).to(tl.float32)

    vm = tl.gather(v, il, 1)
    vp = tl.gather(v, ir, 1)

    even = 0.25 * vm + 0.75 * v
    odd = 0.75 * v + 0.25 * vp

    val = tl.join(even, odd)
    j = tl.arange(0, 2)
    ooff = (rows * OW)[:, None, None] + (ka * 2)[:, :, None] + j[None, None, :]
    om = tl.broadcast_to(
        (ka < IW)[:, :, None] & (rows[:, None, None] < n_rows), (BR, IWP, 2)
    )
    tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty), mask=om)


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


def _pick_rows_per_program(iwp, n_rows):
    cap = max(1, _UB_ELEMS // iwp)
    want = max(1, triton.cdiv(n_rows, _TARGET_PROGRAMS))
    br = min(cap, want)
    br = 1 << (br.bit_length() - 1)
    while br > 1 and n_rows % br != 0:
        br >>= 1
    return max(1, br)


def _pick_block_rows(ih, iw, images):
    """Input rows per program for the exact-2x block kernel.

    Bounded by the per-tile element budget, by ``ih // 2`` (so the image is
    split across at least two programs) and by a soft program-count floor.
    Derived from the workload only; no physical core count is assumed.
    """
    qb = min(max(1, ih // 2), max(1, _BLK_ELEMS // iw))
    qb = 1 << (qb.bit_length() - 1)
    while qb > 1 and (ih % qb != 0 or images * (ih // qb) < _BLK_MIN_PROGRAMS):
        qb >>= 1
    return max(1, qb)


def _plan(shape, strides, out_size, ac, scales_h, scales_w, dtype):
    """Resolve everything that depends only on tensor *metadata*.

    Only scalars are produced; no tensor, storage or data pointer is cached.
    The caller's cache key contains shape, strides, dtype, output size and the
    coordinate options, so a different view or layout derives its own entry
    instead of reusing a stale one.
    """
    N, C, IH, IW = shape
    OH, OW = out_size
    s_n, s_c, s_h, s_w = strides
    images = N * C
    n_rows = images * OH

    rh = _reciprocal_scale(IH, OH, ac, scales_h)
    rw = _reciprocal_scale(IW, OW, ac, scales_w)

    contiguous = (s_w == 1 and s_h == IW and s_c == IH * IW
                  and s_n == C * IH * IW)
    if (
        not ac
        and scales_h is None
        and scales_w is None
        and contiguous
        and OW == 2 * IW
        and OH == 2 * IH
        and IW >= 8
        and (IW & (IW - 1)) == 0
        and IH >= 4
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        qb = _pick_block_rows(IH, IW, images)
        nb = IH // qb
        h16 = dtype is torch.float16
        # (kind, block grid, edge grid, constexpr tuple, h16)
        return (0, (images * nb,), (images * 2,), (IH, IW, OH, OW, nb, qb), h16)

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
        return (1, (triton.cdiv(n_rows, br),), None,
                (n_rows, s_n, s_c, s_h, rh, C, IH, IW, OH, OW, br, iwp), False)

    total = images * OH * OW
    block = _pick_flat_block(total)
    return (2, (triton.cdiv(total, block),), None,
            (total, s_n, s_c, s_h, s_w, rh, rw, C, OH, OW, IH, IW, ac, block),
            False)


_PLAN_CACHE = {}
_LAUNCH_CACHE = {}
_RAW_STREAM_GETTERS = {}


def _current_raw_stream(device_index):
    getter = _RAW_STREAM_GETTERS.get(device_index)
    if getter is None:
        raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
        expected = int(torch.npu.current_stream(device_index).npu_stream)
        if raw is not None:
            try:
                if int(raw(device_index)) == expected:
                    getter = lambda: raw(device_index)
            except TypeError:
                if int(raw()) == expected:
                    getter = raw
        if getter is None:
            from triton.runtime import driver
            getter = lambda: driver.active.get_current_stream(device_index)
        _RAW_STREAM_GETTERS[device_index] = getter
    return getter()


def _ordinary(input, output_size, align_corners, scales_h, scales_w):
    """Uncached launch path; also the one that compiles a specialization."""
    N, C, IH, IW = input.shape
    OH = int(output_size[0])
    OW = int(output_size[1])
    out = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    if N * C * OH * OW == 0:
        return out

    key = (N, C, IH, IW, OH, OW, bool(align_corners), scales_h, scales_w,
           input.stride(), input.dtype)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _plan((N, C, IH, IW), input.stride(), (OH, OW),
                     bool(align_corners), scales_h, scales_w, input.dtype)
        _PLAN_CACHE[key] = plan

    kind = plan[0]
    if kind == 0:
        ih, iw, oh, ow, nb, qb = plan[3]
        _blk2x_kernel[plan[1]](out, input, IH=ih, IW=iw, OH=oh, OW=ow,
                               NB=nb, QB=qb, H16=plan[4])
        _edge2x_kernel[plan[2]](out, input, IH=ih, IW=iw, OH=oh, OW=ow,
                                H16=plan[4])
    elif kind == 1:
        n_rows, s_n, s_c, s_h, rh, kc, kih, kiw, koh, kow, br, iwp = plan[3]
        _bilinear_int_kernel[plan[1]](
            out, input, n_rows, s_n, s_c, s_h, rh,
            C=kc, IH=kih, IW=kiw, OH=koh, OW=kow, BR=br, IWP=iwp)
    else:
        (total, s_n, s_c, s_h, s_w, rh, rw, kc, koh, kow, kih, kiw, ac,
         block) = plan[3]
        _bilinear_flat_kernel[plan[1]](
            out, input, total, s_n, s_c, s_h, s_w, rh, rw,
            C=kc, OH=koh, OW=kow, IH=kih, IW=kiw,
            ALIGN_CORNERS=ac, BLOCK=block)
    return out


def _cacheable(compiled):
    """True when the compiled kernel may be re-launched through its own
    generated launcher: no compile-only run, no profiling hook registration
    and no debugger attachment."""
    launcher = compiled.run
    if getattr(launcher, "compile_only", False):
        return False
    if getattr(launcher, "enable_msprof_register_tensor", False):
        return False
    metadata = getattr(launcher, "metadata", None)
    if metadata is not None and (getattr(metadata, "debug_enabled", False)
                                 or getattr(metadata, "debug", False)):
        return False
    return True


def run(input, output_size, align_corners=False, scales_h=None, scales_w=None):
    N, C, IH, IW = input.shape
    OH = int(output_size[0])
    OW = int(output_size[1])
    out = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    if N * C * OH * OW == 0:
        return out

    strides = input.stride()
    dtype = input.dtype
    key = (N, C, IH, IW, OH, OW, bool(align_corners), scales_h, scales_w,
           strides, dtype)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        return _ordinary(input, output_size, align_corners, scales_h, scales_w)
    if plan[0] != 0:
        return _ordinary(input, output_size, align_corners, scales_h, scales_w)

    ih, iw, oh, ow, nb, qb = plan[3]
    h16 = plan[4]
    dev = input.device.index
    in_ptr = input.data_ptr()
    out_ptr = out.data_ptr()
    lkey = (key, dev, in_ptr & 15, out_ptr & 15)
    entry = _LAUNCH_CACHE.get(lkey)
    if entry is None:
        blk = _blk2x_kernel[plan[1]](out, input, IH=ih, IW=iw, OH=oh, OW=ow,
                                     NB=nb, QB=qb, H16=h16)
        edge = _edge2x_kernel[plan[2]](out, input, IH=ih, IW=iw, OH=oh, OW=ow,
                                       H16=h16)
        if not (_cacheable(blk) and _cacheable(edge)):
            return out
        entry = (blk.run.launch, blk.function, blk.packed_metadata, plan[1][0],
                 edge.run.launch, edge.function, edge.packed_metadata,
                 plan[2][0], dev)
        _LAUNCH_CACHE[lkey] = entry
        return out

    # All pointers are from this call; no input/output tensors or historical
    # streams are cached.  Every scalar specialization is part of lkey.
    stream = _current_raw_stream(entry[8])
    entry[0](entry[3], 1, 1, stream, entry[1], entry[2], None, None, None,
             out_ptr, in_ptr, None, None, None, None, None, None, None)
    entry[4](entry[7], 1, 1, stream, entry[5], entry[6], None, None, None,
             out_ptr, in_ptr, None, None, None, None, None)
    return out

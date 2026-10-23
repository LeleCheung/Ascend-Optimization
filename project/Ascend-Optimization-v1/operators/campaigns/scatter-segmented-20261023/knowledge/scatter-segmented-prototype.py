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
"""Self-contained Triton ``scatter_add`` (out-of-place).

Semantics (``torch.scatter_add``)::

    out = inp.clone()
    out[index[i_0, ..., i_d, ...]] += src[i_0, ..., i_d, ...]   (along ``dim``)

``inp``/``index``/``src`` are never modified and the returned buffer is fresh.

Architecture notes (measured on the target device, Ascend 910B4 / CANN 9.0):

* The device reference accumulates in **fp32** for fp16/bf16 operands: an
  explicit column-ordered fp32 model reproduces ``torch.scatter_add`` exactly on
  the frozen timing shapes, while a per-add low-precision fold does not.  All
  accumulation here therefore happens in fp32 and is rounded once on store.
* ``tl.atomic_add`` with data dependent addresses is lowered to a per-lane
  ``scf.for`` of single-element atomic stores (~4.7 ns/element, MTE3 bound).
  It is only used when the destination side cannot be owned by a single
  program.
* For a contiguous last-axis scatter the destination side *can* be owned: one
  program owns a ``(row, dst block)`` pair and folds every source element of
  that row into it with an affine compare, so each output element is written
  exactly once, contiguously, with no atomic and no separate copy launch.
  That costs ``ncols`` comparisons per source element, so it only wins while
  ``ncols`` is small (measured: 0.108 ms at 128 columns, 2.85 ms at 1024).
* Wide rows take a **sorted-segment** path instead.  Per row we sort
  ``target*B + lane`` keys, which groups every destination's contributions into
  one contiguous run **in ascending lane order** -- exactly the reference's
  column-ordered accumulation.  A cumsum of the reordered values then turns
  each run into an independent segmented prefix, so the combine pass writes the dense
  output contiguously and never scatters.
  Measured ingredient costs on 1024x1024 / 2048x2048 fp16: ``al.sort`` 0.06 /
  0.20 ms, on-chip ``tl.gather`` 0.02 / 0.04 ms, ``tl.histogram`` 1.79 /
  8.30 ms, integer ``//``/``%`` +0.48 / +1.95 ms (replaced by ``>>``/``&``).
  Two backend constraints shape this code: ``tl.sort`` verifies as
  ``hivm.hir.vsort`` and rejects int32 keys, and a *dynamic* gather ``tl.load``
  accepts only fp16/fp32/bf16/fp8/int8 -- so keys travel through memory as
  **fp32** (exact for every B <= 4096) even though they are used as integers.
  For the same reason the run-location search must stay on-chip: a per-lane
  dynamic *global* load costs ~25x an on-chip ``tl.gather`` here, so the sorted
  keys and the cumsum row are each loaded once contiguously and then searched
  in registers.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import extension as _al

_MAXR = 8
_COPY_BLOCK = 8192
_DENSE_MAX_COLS = 256
_SORT_MAX_COLS = 4096
_DENSE_CHUNK = 32
_ATOMIC_BLOCK = 1024
_INT32_MAX = 2**31 - 1

_D_ARGS = [f"d{i}" for i in range(_MAXR)]
_SS_ARGS = [f"ss{i}" for i in range(_MAXR)]
_IS_ARGS = [f"is{i}" for i in range(_MAXR)]
_OS_ARGS = [f"os{i}" for i in range(_MAXR)]


# --------------------------------------------------------------------------- #
# copy / cast helpers (also used for the fp16/bf16 -> fp32 staging buffer)
# --------------------------------------------------------------------------- #
@triton.jit
def _copy_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(dst_ptr + off, tl.load(src_ptr + off, mask=mask), mask=mask)


@triton.jit(do_not_specialize=["n"] + _D_ARGS + _SS_ARGS)
def _copy_nd_kernel(
    src_ptr,
    dst_ptr,
    n,
    d0, d1, d2, d3, d4, d5, d6, d7,
    ss0, ss1, ss2, ss3, ss4, ss5, ss6, ss7,
    BLOCK: tl.constexpr,
):
    """Contiguous ``dst`` copy of an arbitrarily strided ``src``."""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    cur = off
    src_off = tl.zeros([BLOCK], dtype=tl.int64)
    md = cur % d7
    src_off += md * ss7
    cur = cur // d7
    md = cur % d6
    src_off += md * ss6
    cur = cur // d6
    md = cur % d5
    src_off += md * ss5
    cur = cur // d5
    md = cur % d4
    src_off += md * ss4
    cur = cur // d4
    md = cur % d3
    src_off += md * ss3
    cur = cur // d3
    md = cur % d2
    src_off += md * ss2
    cur = cur // d2
    md = cur % d1
    src_off += md * ss1
    cur = cur // d1
    md = cur % d0
    src_off += md * ss0
    tl.store(dst_ptr + off, tl.load(src_ptr + src_off, mask=mask), mask=mask)


# --------------------------------------------------------------------------- #
# destination-owned last-axis kernel (no atomic, single launch)
# --------------------------------------------------------------------------- #
@triton.jit
def _scatter_add_owner_kernel(
    inp_ptr,
    idx_ptr,
    src_ptr,
    out_ptr,
    ncols: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    EVEN: tl.constexpr,
):
    row = tl.program_id(0)
    dst = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = row * ncols
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    if EVEN:
        # ``ncols`` is an exact multiple of BLOCK and CHUNK: every lane is live,
        # so the whole loop and the final load/store run unmasked and the trip
        # count is a compile-time constant.
        for s0 in range(0, ncols, CHUNK):
            so = s0 + tl.arange(0, CHUNK)
            idx = tl.load(idx_ptr + base + so).to(tl.int32)
            val = tl.load(src_ptr + base + so).to(tl.float32)
            acc += tl.sum(tl.where(idx[:, None] == dst[None, :], val[:, None], 0.0), axis=0)
        cur = tl.load(inp_ptr + base + dst).to(tl.float32)
        tl.store(out_ptr + base + dst, (cur + acc).to(out_ptr.dtype.element_ty))
    else:
        dmask = dst < ncols
        for s0 in range(0, ncols, CHUNK):
            so = s0 + tl.arange(0, CHUNK)
            smask = so < ncols
            # ``other=-1`` keeps masked lanes from matching a non-negative
            # destination, so the compare needs no mask term (measured 1.3x
            # faster than comparing ``... & smask[:, None]``).
            idx = tl.load(idx_ptr + base + so, mask=smask, other=-1).to(tl.int32)
            val = tl.load(src_ptr + base + so, mask=smask, other=0).to(tl.float32)
            acc += tl.sum(tl.where(idx[:, None] == dst[None, :], val[:, None], 0.0), axis=0)
        cur = tl.load(inp_ptr + base + dst, mask=dmask, other=0).to(tl.float32)
        tl.store(out_ptr + base + dst, (cur + acc).to(out_ptr.dtype.element_ty), mask=dmask)


# --------------------------------------------------------------------------- #
# sorted-segment last-axis path (wide rows)
# --------------------------------------------------------------------------- #
@triton.jit
def _segment_combine(left_sum, left_head, right_sum, right_head):
    return tl.where(right_head, right_sum, left_sum + right_sum), left_head | right_head


@triton.jit
def _scatter_add_sort_key_kernel(
    src_ptr,
    idx_ptr,
    keys_ptr,
    prefix_ptr,
    ncols: tl.constexpr,
    B: tl.constexpr,
    LOG: tl.constexpr,
):
    """Per row: sort ``target*B + lane`` keys, gather values into sorted order,
    store the sorted keys (fp32) and their segmented inclusive sums."""
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    live = lane < ncols
    t = tl.load(idx_ptr + row * ncols + lane, mask=live, other=B).to(tl.int32)
    key = _al.sort((t * B + lane).to(tl.float32), dim=-1)
    keyi = key.to(tl.int32)
    src_lane = tl.minimum(keyi & (B - 1), ncols - 1)
    values = tl.load(src_ptr + row * ncols + lane, mask=live, other=0.0).to(tl.float32)
    sv = tl.gather(values, src_lane, axis=0)
    sv = tl.where((keyi >> LOG) < ncols, sv, 0.0)
    tl.store(keys_ptr + row * B + lane, key)
    targets = keyi >> LOG
    previous = tl.gather(targets.to(tl.float32), tl.maximum(lane - 1, 0), axis=0).to(tl.int32)
    heads = (lane == 0) | (targets != previous)
    segment_sums, _ = tl.associative_scan((sv, heads), axis=0, combine_fn=_segment_combine)
    tl.store(prefix_ptr + row * B + lane, segment_sums)


@triton.jit
def _scatter_add_sort_out_kernel(
    inp_ptr,
    keys_ptr,
    prefix_ptr,
    out_ptr,
    ncols: tl.constexpr,
    B: tl.constexpr,
    LOG: tl.constexpr,
):
    """Per row: binary-search each destination's run in the sorted keys and emit
    ``inp + segment_sum[end-1]`` contiguously; absent segments add zero.

    The row is loaded once and searched on-chip: a data-dependent *global* load
    here costs ~25x the matching ``tl.gather``, so both vectors are staged in
    registers and only the search's dataflow crosses the vector unit.
    """
    row = tl.program_id(0)
    base = row * B
    dst = tl.arange(0, B)
    keys = tl.load(keys_ptr + base + dst)
    prefix = tl.load(prefix_ptr + base + dst)
    t1 = dst.to(tl.float32) * B
    lo = tl.zeros([B], tl.int32)
    hi = tl.full([B], B, tl.int32)
    for _ in tl.static_range(LOG + 1):
        mid = tl.minimum((lo + hi) >> 1, B - 1)
        go = (tl.gather(keys, mid, axis=0) < t1) & (lo < hi)
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(go, hi, tl.minimum(mid, hi))
    # run end for ``dst`` is the run start of ``dst + 1``; the last destination
    # is followed by exactly ``ncols`` real keys.
    nxt = tl.gather(lo.to(tl.float32), tl.minimum(dst + 1, B - 1), axis=0).to(tl.int32)
    end = tl.where(dst == ncols - 1, ncols, nxt)
    after = tl.gather(prefix, tl.maximum(tl.minimum(end - 1, B - 1), 0), axis=0)
    delta = tl.where(end > lo, after, 0.0)
    live = dst < ncols
    init = tl.load(inp_ptr + row * ncols + dst, mask=live, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * ncols + dst, (init + delta).to(out_ptr.dtype.element_ty),
             mask=live)


# --------------------------------------------------------------------------- #
# last-axis atomic kernels (fp32 accumulation buffer)
# --------------------------------------------------------------------------- #
@triton.jit(do_not_specialize=["ncols"])
def _scatter_add_lastdim_kernel(
    src_ptr,
    idx_ptr,
    out_ptr,
    ncols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = c < ncols
    base = row * ncols
    idx = tl.load(idx_ptr + base + c, mask=mask, other=0).to(tl.int32)
    val = tl.load(src_ptr + base + c, mask=mask, other=0).to(out_ptr.dtype.element_ty)
    tl.atomic_add(out_ptr + (base + idx), val, mask=mask, sem="relaxed")


# --------------------------------------------------------------------------- #
# generic N-d kernels (any dim / strides; int64 addressing)
# --------------------------------------------------------------------------- #
@triton.jit(do_not_specialize=["n", "odim"] + _D_ARGS + _SS_ARGS + _IS_ARGS + _OS_ARGS)
def _scatter_add_nd_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    n,
    odim,
    d0, d1, d2, d3, d4, d5, d6, d7,
    ss0, ss1, ss2, ss3, ss4, ss5, ss6, ss7,
    is0, is1, is2, is3, is4, is5, is6, is7,
    os0, os1, os2, os3, os4, os5, os6, os7,
    BLOCK: tl.constexpr,
):
    """Scatter-add over the ``index``/``src`` element space.

    ``ss``/``is`` are the strides of ``src``/``index``; ``os`` are the
    (contiguous) strides of the output with the entry for ``dim`` forced to 0,
    so the ``dim`` coordinate is contributed only through ``idx * odim``.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    cur = off
    src_off = tl.zeros([BLOCK], dtype=tl.int64)
    idx_off = tl.zeros([BLOCK], dtype=tl.int64)
    out_off = tl.zeros([BLOCK], dtype=tl.int64)

    md = cur % d7
    src_off += md * ss7
    idx_off += md * is7
    out_off += md * os7
    cur = cur // d7

    md = cur % d6
    src_off += md * ss6
    idx_off += md * is6
    out_off += md * os6
    cur = cur // d6

    md = cur % d5
    src_off += md * ss5
    idx_off += md * is5
    out_off += md * os5
    cur = cur // d5

    md = cur % d4
    src_off += md * ss4
    idx_off += md * is4
    out_off += md * os4
    cur = cur // d4

    md = cur % d3
    src_off += md * ss3
    idx_off += md * is3
    out_off += md * os3
    cur = cur // d3

    md = cur % d2
    src_off += md * ss2
    idx_off += md * is2
    out_off += md * os2
    cur = cur // d2

    md = cur % d1
    src_off += md * ss1
    idx_off += md * is1
    out_off += md * os1
    cur = cur // d1

    md = cur % d0
    src_off += md * ss0
    idx_off += md * is0
    out_off += md * os0

    idx = tl.load(index_ptr + idx_off, mask=mask, other=0).to(tl.int64)
    val = tl.load(src_ptr + src_off, mask=mask, other=0).to(out_ptr.dtype.element_ty)
    out_off += idx * odim
    tl.atomic_add(out_ptr + out_off, val, mask=mask, sem="relaxed")


# --------------------------------------------------------------------------- #
# host side
# --------------------------------------------------------------------------- #
def _copy(src, dst):
    n = dst.numel()
    if n == 0:
        return
    if src.is_contiguous() and n <= _INT32_MAX:
        _copy_kernel[(triton.cdiv(n, _COPY_BLOCK),)](src, dst, n, BLOCK=_COPY_BLOCK)
        return
    shape = [1] * (_MAXR - src.dim()) + list(src.shape)
    strides = [0] * (_MAXR - src.dim()) + list(src.stride())
    _copy_nd_kernel[(triton.cdiv(n, 512),)](
        src, dst, n,
        shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7],
        strides[0], strides[1], strides[2], strides[3],
        strides[4], strides[5], strides[6], strides[7],
        BLOCK=512,
    )


def _contig_strides(shape):
    strides = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return strides


def run(inp, dim, index, src):
    d = dim % inp.ndim
    n = index.numel()
    dtype = inp.dtype

    if n == 0:
        out = torch.empty(inp.shape, dtype=dtype, device=inp.device)
        _copy(inp, out)
        return out

    # Fast path: full-extent, contiguous, last-axis scatter -> (rows, ncols).
    # ``inp.shape == index.shape == src.shape`` implies ``inp.numel() == n``, so
    # the tuple comparison replaces two scalar reads.  Every measured shape
    # takes this path; the launch is a bare tuple grid with no meta function.
    if (
        d == inp.ndim - 1
        and inp.shape == index.shape == src.shape
        and inp.is_contiguous()
        and index.is_contiguous()
        and src.is_contiguous()
        and n <= _INT32_MAX
    ):
        ncols = inp.shape[-1]
        rows = n // ncols
        out = torch.empty(inp.shape, dtype=dtype, device=inp.device)
        if ncols <= _DENSE_MAX_COLS:
            block = 128 if ncols <= 256 else 256
            _scatter_add_owner_kernel[(rows, -(-ncols // block))](
                inp, index, src, out, ncols,
                BLOCK=block,
                CHUNK=_DENSE_CHUNK,
                EVEN=(ncols % _DENSE_CHUNK == 0 and ncols % block == 0),
            )
            return out
        if ncols <= _SORT_MAX_COLS:
            # keys are ``target * B + lane`` kept in fp32: exact for B <= 4096
            # and the only dtype the backend's gather ``tl.load`` accepts here.
            B = 1 << (ncols - 1).bit_length()
            LOG = B.bit_length() - 1
            keys = torch.empty((rows, B), dtype=torch.float32, device=inp.device)
            prefix = torch.empty((rows, B), dtype=torch.float32, device=inp.device)
            _scatter_add_sort_key_kernel[(rows,)](src, index, keys, prefix,
                                                  ncols, B, LOG)
            _scatter_add_sort_out_kernel[(rows,)](inp, keys, prefix, out,
                                                  ncols, B, LOG)
            return out
        if dtype in (torch.float16, torch.bfloat16):
            # fp32 accumulation buffer: the reference rounds once per output.
            buf = torch.empty(inp.shape, dtype=torch.float32, device=inp.device)
            _copy(inp, buf)
            _scatter_add_lastdim_kernel[(rows, -(-ncols // _ATOMIC_BLOCK))](
                src, index, buf, ncols, BLOCK=_ATOMIC_BLOCK
            )
            _copy(buf, out)
            return out
        _copy(inp, out)
        _scatter_add_lastdim_kernel[(rows, -(-ncols // _ATOMIC_BLOCK))](
            src, index, out, ncols, BLOCK=_ATOMIC_BLOCK
        )
        return out

    # Generic path: any dim / strides.
    acc_f32 = dtype in (torch.float16, torch.bfloat16)
    out = torch.empty(inp.shape, dtype=dtype, device=inp.device)
    buf = torch.empty(inp.shape, dtype=torch.float32, device=inp.device) if acc_f32 else out
    _copy(inp, buf)

    # ``src`` follows ``index``'s extent; extra trailing entries are never read.
    src_view = src.as_strided(index.shape, src.stride()) if src.shape != index.shape else src
    pad = _MAXR - index.dim()
    idx_shape = [1] * pad + list(index.shape)
    src_strides = [0] * pad + list(src_view.stride())
    idx_strides = [0] * pad + list(index.stride())
    out_strides = _contig_strides(inp.shape)
    odim = out_strides[d]
    out_strides[d] = 0
    z = [0] * pad
    out_slots = z + out_strides

    _scatter_add_nd_kernel[(triton.cdiv(n, 512),)](
        src_view, index, buf, n, odim,
        idx_shape[0], idx_shape[1], idx_shape[2], idx_shape[3],
        idx_shape[4], idx_shape[5], idx_shape[6], idx_shape[7],
        src_strides[0], src_strides[1], src_strides[2], src_strides[3],
        src_strides[4], src_strides[5], src_strides[6], src_strides[7],
        idx_strides[0], idx_strides[1], idx_strides[2], idx_strides[3],
        idx_strides[4], idx_strides[5], idx_strides[6], idx_strides[7],
        out_slots[0], out_slots[1], out_slots[2], out_slots[3],
        out_slots[4], out_slots[5], out_slots[6], out_slots[7],
        BLOCK=512,
    )
    if acc_f32:
        _copy(buf, out)
    return out

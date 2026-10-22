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

Semantics (PyTorch ``torch.scatter_add``)::

    out = inp.clone()
    out[index[i_0, ..., i_d, ...]] += src[i_0, ..., i_d, ...]   along ``dim``

with ``index.shape == src.shape``, ``index.size(k) <= inp.size(k)`` for
``k != dim`` and ``index.size(dim) <= src.size(dim)``.  The result is a fresh
buffer: ``inp``, ``index`` and ``src`` are never modified.

Baseline architecture: a device copy of ``inp`` into a freshly allocated
contiguous output followed by one generic N-d atomic-add kernel over the index
element space.  All computation that produces the returned values happens in
Triton kernels; the host only allocates, builds zero-copy views and computes
launch scalars.
"""

import torch
import triton
import triton.language as tl

_MAXR = 8

_D_ARGS = [f"d{i}" for i in range(_MAXR)]
_SS_ARGS = [f"ss{i}" for i in range(_MAXR)]
_IS_ARGS = [f"is{i}" for i in range(_MAXR)]
_OS_ARGS = [f"os{i}" for i in range(_MAXR)]

_COPY_BLOCK = 8192
_SCATTER_BLOCK = 512


def _slots(values, pad):
    """Right-align ``values`` into ``_MAXR`` slots, padding in front.

    Front padding is transparent for the sequential ``div/mod`` decomposition
    used in the kernels: a shape slot of ``1`` always yields a zero remainder
    and leaves the running quotient unchanged.
    """
    return [pad] * (_MAXR - len(values)) + list(values)


@triton.jit(do_not_specialize=["n"])
def _copy_flat_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(dst_ptr + off, tl.load(src_ptr + off, mask=mask), mask=mask)


@triton.jit(do_not_specialize=["ncols"])
def _scatter_add_lastdim_kernel(
    src_ptr,
    idx_ptr,
    out_ptr,
    ncols,
    BLOCK: tl.constexpr,
):
    """Scatter-add along the last axis of a contiguous (rows, ncols) view.

    One program owns one ``(row, column block)`` pair, so the element offset is
    ``row * ncols + col`` with no div/mod, and only four runtime arguments are
    shipped to the launcher.
    """
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < ncols
    base = row * ncols
    off = base + col
    idx = tl.load(idx_ptr + off, mask=mask, other=0).to(tl.int32)
    val = tl.load(src_ptr + off, mask=mask, other=0)
    tl.atomic_add(out_ptr + (base + idx), val, mask=mask, sem="relaxed")


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
    src_off = tl.zeros([BLOCK], dtype=tl.int32)
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
    """Generic N-d scatter-add over the ``index``/``src`` element space.

    ``ss``/``is`` are the strides of ``src``/``index``; ``os`` are the
    (contiguous) strides of the output with the entry for ``dim`` forced to 0,
    so the ``dim`` coordinate is contributed only through ``idx * odim``.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    cur = off
    src_off = tl.zeros([BLOCK], dtype=tl.int32)
    idx_off = tl.zeros([BLOCK], dtype=tl.int32)
    out_off = tl.zeros([BLOCK], dtype=tl.int32)

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

    idx = tl.load(index_ptr + idx_off, mask=mask, other=0).to(tl.int32)
    val = tl.load(src_ptr + src_off, mask=mask, other=0)
    out_off += idx * odim
    tl.atomic_add(out_ptr + out_off, val, mask=mask, sem="relaxed")


def _contig_strides(shape):
    """Row-major strides of ``shape`` (what ``torch.empty(shape)`` uses)."""
    strides = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return strides


def _launch_copy(src, dst):
    n = dst.numel()
    if n == 0:
        return
    if src.is_contiguous():
        grid = (triton.cdiv(n, _COPY_BLOCK),)
        _copy_flat_kernel[grid](src, dst, n, BLOCK=_COPY_BLOCK)
        return
    shape = _slots(src.shape, 1)
    strides = _slots(src.stride(), 0)
    grid = (triton.cdiv(n, _SCATTER_BLOCK),)
    _copy_nd_kernel[grid](
        src,
        dst,
        n,
        shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7],
        strides[0], strides[1], strides[2], strides[3], strides[4], strides[5], strides[6], strides[7],
        BLOCK=_SCATTER_BLOCK,
    )


def run(inp, dim, index, src):
    ndim = inp.ndim
    dim = dim % ndim

    if inp.numel() >= 2**31:
        raise RuntimeError("scatter_add: int32 offset range exceeded")

    out = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)

    n = index.numel()
    if n == 0:
        _launch_copy(inp, out)
        return out

    # Fast path: last-axis scatter of a fully contiguous, full-extent triple.
    # All measured shapes take this path; it removes every div/mod from the
    # element addressing and shrinks the launch to four runtime arguments.
    if (
        dim + 1 == ndim
        and index.shape == inp.shape
        and src.shape == index.shape
        and inp.is_contiguous()
        and index.is_contiguous()
        and src.is_contiguous()
    ):
        ncols = inp.shape[-1]
        if ncols != 0:
            _launch_copy(inp, out)
            rows = out.numel() // ncols
            block = 128
            while block < ncols and block < 2048:
                block *= 2
            _scatter_add_lastdim_kernel[(rows, triton.cdiv(ncols, block))](
                src, index, out, ncols, BLOCK=block
            )
            return out

    _launch_copy(inp, out)

    # ``src`` follows ``index``'s extent; extra trailing entries of ``src`` are
    # never read, exactly like the reference.
    src_view = src.as_strided(index.shape, src.stride())

    idx_shape = _slots(index.shape, 1)
    src_strides = _slots(src_view.stride(), 0)
    idx_strides = _slots(index.stride(), 0)
    out_strides = _contig_strides(inp.shape)
    odim = out_strides[dim]
    out_strides[dim] = 0
    out_slots = _slots(out_strides, 0)

    grid = (triton.cdiv(n, _SCATTER_BLOCK),)
    _scatter_add_nd_kernel[grid](
        src_view,
        index,
        out,
        n,
        odim,
        idx_shape[0], idx_shape[1], idx_shape[2], idx_shape[3],
        idx_shape[4], idx_shape[5], idx_shape[6], idx_shape[7],
        src_strides[0], src_strides[1], src_strides[2], src_strides[3],
        src_strides[4], src_strides[5], src_strides[6], src_strides[7],
        idx_strides[0], idx_strides[1], idx_strides[2], idx_strides[3],
        idx_strides[4], idx_strides[5], idx_strides[6], idx_strides[7],
        out_slots[0], out_slots[1], out_slots[2], out_slots[3],
        out_slots[4], out_slots[5], out_slots[6], out_slots[7],
        BLOCK=_SCATTER_BLOCK,
    )
    return out

"""t_copy: copy of the transpose of a 2-D tensor into a new tensor.

Reference semantics (verified on the target): ``out = input.t().clone()`` with
``out.shape == (input.shape[1], input.shape[0])`` and ``out`` contiguous.
0-D and 1-D inputs are plain copies; inputs with more than 2 dimensions are
rejected by ``t()``.

Implementation notes (established by target-side debug jobs):
  * The transpose needs one unit-stride run on the input side (length BM) and
    one on the output side (length BN); the tile is transposed in UB by
    ``tl.trans``.
  * Kernel time is dominated by per-program and per-DMA-segment cost, so each
    program owns several tiles (amortising program setup) and tiles are kept at
    <= 32 KiB of UB, the largest size that compiles at every dtype here.
  * The stock JITFunction launch path costs ~50 us of host time per call, which
    makes the small cases host bound; the compiled kernel's launcher is invoked
    directly instead, with a total fallback to the stock launch path and then to
    a fully masked generic kernel.
"""

import torch
import triton
import triton.language as tl

try:  # host-side launcher handle for the low-overhead launch path
    from triton.runtime import driver as _triton_driver
except Exception:  # pragma: no cover - optional optimisation only
    _triton_driver = None


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
@triton.jit
def _t_copy_tiled(in_ptr, out_ptr, ROWS, COLS,
                  BM: tl.constexpr, BN: tl.constexpr, T: tl.constexpr):
    """out (ROWS, COLS) = transpose of a unit-stride in (COLS, ROWS).

    One program owns ``BM`` output rows across ``T`` output column tiles of
    ``BN``.  Requires ROWS % BM == 0 and COLS % (BN * T) == 0.
    """
    pid = tl.program_id(0)
    nbn = COLS // (BN * T)
    pi = pid // nbn
    pj = pid % nbn
    ri = pi * BM + tl.arange(0, BM)
    cbase = pj * (BN * T)
    for t in range(T):
        cj = cbase + t * BN + tl.arange(0, BN)
        tile = tl.load(in_ptr + cj[:, None] * ROWS + ri[None, :])
        tl.store(out_ptr + ri[:, None] * COLS + cj[None, :], tl.trans(tile))


@triton.jit
def _t_copy_masked(in_ptr, out_ptr, ROWS, COLS,
                   in_s0, in_s1, out_s0, out_s1,
                   BM: tl.constexpr, BN: tl.constexpr):
    """Masked transposing copy for arbitrary strides and ragged shapes."""
    pi = tl.program_id(0)
    pj = tl.program_id(1)
    ri = pi * BM + tl.arange(0, BM)
    cj = pj * BN + tl.arange(0, BN)
    m_load = (cj[:, None] < COLS) & (ri[None, :] < ROWS)
    tile = tl.load(in_ptr + cj[:, None] * in_s0 + ri[None, :] * in_s1,
                   mask=m_load, other=0.0)
    m_store = (ri[:, None] < ROWS) & (cj[None, :] < COLS)
    tl.store(out_ptr + ri[:, None] * out_s0 + cj[None, :] * out_s1,
             tl.trans(tile), mask=m_store)


@triton.jit
def _copy_1d(in_ptr, out_ptr, n, in_s, out_s, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs * out_s,
             tl.load(in_ptr + offs * in_s, mask=mask), mask=mask)


# --------------------------------------------------------------------------- #
# host side
# --------------------------------------------------------------------------- #
# Tiles sized so that BM * BN * itemsize <= 32 KiB and BM * BN * T covers a
# useful amount of work; measured on the target for the 4096^2 workloads.
_BIG_TILES = {
    1: [(128, 128, 4), (256, 128, 1), (128, 128, 1)],
    2: [(128, 128, 4), (256, 128, 1), (128, 128, 1)],
    4: [(64, 128, 4), (128, 128, 1), (64, 128, 1)],
    8: [(64, 64, 4), (64, 64, 1)],
}
_SMALL_TILES = [
    (16, 64, 1), (32, 64, 1), (64, 64, 1), (16, 32, 1), (32, 32, 1),
    (8, 32, 1), (16, 16, 1), (8, 16, 1), (4, 16, 1), (4, 8, 1), (2, 8, 1),
    (2, 4, 1), (1, 4, 1), (1, 2, 1), (1, 1, 1),
]
_BIG_THRESHOLD = 1 << 20
_MASKED_TILE = (128, 128)

_LAUNCHER_CACHE = {}
_FAST_LAUNCH = True


def _pick_tile(rows, cols, itemsize):
    """First tiling that divides (rows, cols) exactly, or None."""
    cands = []
    if rows * cols >= _BIG_THRESHOLD:
        cands.extend(_BIG_TILES.get(itemsize, _BIG_TILES[4]))
        cands.extend(_SMALL_TILES)
    else:
        cands.extend(_SMALL_TILES)
        cands.extend(_BIG_TILES.get(itemsize, _BIG_TILES[4]))
    for bm, bn, t in cands:
        if rows % bm == 0 and cols % (bn * t) == 0:
            return bm, bn, t
    return None


def _launch_direct(x, out, rows, cols, bm, bn, t):
    """Launch the compiled tiled kernel without the per-call JIT binder.

    Returns True when the launch was issued, False if the caller should use the
    stock launch path.
    """
    global _FAST_LAUNCH
    if not _FAST_LAUNCH or _triton_driver is None:
        return False
    key = (rows, cols, x.dtype, bm, bn, t)
    entry = _LAUNCHER_CACHE.get(key)
    if entry is None:
        grid0 = (rows // bm) * (cols // (bn * t))
        try:
            ck = _t_copy_tiled.warmup(x, out, rows, cols, BM=bm, BN=bn, T=t,
                                      grid=(grid0,))
            init = getattr(ck, "_init_handles", None)
            if init is not None:
                init()
            active = _triton_driver.active
            entry = (ck.run, ck.function, ck.packed_metadata, grid0,
                     active.get_current_device(), active.get_current_stream)
        except Exception:
            _FAST_LAUNCH = False
            return False
        _LAUNCHER_CACHE[key] = entry
    launch, fnptr, packed, grid0, dev, get_stream = entry
    launch(grid0, 1, 1, get_stream(dev), fnptr, packed, None, None, None,
           x, out, rows, cols, bm, bn, t)
    return True


def run(input, memory_format=None):
    x = input
    ndim = x.dim()

    if ndim == 0:
        out = torch.empty((), dtype=x.dtype, device=x.device)
        _copy_1d[(1,)](x, out, 1, 0, 0, BLOCK=1)
        return out

    if ndim == 1:
        n = x.shape[0]
        out = torch.empty((n,), dtype=x.dtype, device=x.device)
        if n:
            block = 4096
            _copy_1d[(triton.cdiv(n, block),)](x, out, n, x.stride(0), 1,
                                               BLOCK=block)
        return out

    if ndim != 2:
        raise RuntimeError(
            "t_copy expects a tensor with <= 2 dimensions, but self is %dD"
            % ndim
        )

    rows, cols = x.shape[1], x.shape[0]      # out is (cols, rows)
    out = torch.empty((rows, cols), dtype=x.dtype, device=x.device)
    if rows == 0 or cols == 0:
        return out

    if x.is_contiguous():
        cfg = _pick_tile(rows, cols, x.element_size())
        if cfg is not None:
            bm, bn, t = cfg
            try:
                if _launch_direct(x, out, rows, cols, bm, bn, t):
                    return out
                grid = ((rows // bm) * (cols // (bn * t)),)
                _t_copy_tiled[grid](x, out, rows, cols, BM=bm, BN=bn, T=t)
                return out
            except Exception:
                pass  # fall through to the fully masked kernel

    bm, bn = _MASKED_TILE
    grid = (triton.cdiv(rows, bm), triton.cdiv(cols, bn))
    _t_copy_masked[grid](x, out, rows, cols,
                         x.stride(0), x.stride(1),      # in[col, row]
                         out.stride(0), out.stride(1),  # out[row, col]
                         BM=bm, BN=bn)
    return out

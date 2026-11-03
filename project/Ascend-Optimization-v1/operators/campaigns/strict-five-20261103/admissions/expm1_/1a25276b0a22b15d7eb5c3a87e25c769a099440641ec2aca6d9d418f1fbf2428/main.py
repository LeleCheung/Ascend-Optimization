"""expm1_ : in-place elementwise ``expm1(A)`` on Ascend, self-contained Triton.

Semantics reproduced from the reference:
  * ``out = expm1(A)`` is written back into ``A`` storage and ``A`` is returned.
  * Input dtype (float16 / bfloat16 / float32) is preserved on the output.
  * Strided (non-contiguous) views are read *and* written through their real
    element strides so the mutation lands in the right storage slots; the view
    object itself is returned unchanged.

Contiguous path (metadata-only decisions in the wrapper):
  * The tile body is mask-free.  Only the final program, which owns the ragged
    remainder, is predicated; every other program is guaranteed by the compiler
    -visible invariant ``pid < n // BLOCK`` to stay inside the buffer, so its
    load and store vectors carry no per-element predicate.  This is the general
    boundary condition, not a size special case: any ``n`` works.
  * Tile size is workload derived: ``BLOCK_MAX`` once there is at least one full
    tile per vector core, otherwise ``BLOCK_MIN`` so small tensors still spread
    over the available cores.  ``BLOCK_MAX`` is capped by the UB budget -
    8192 elements needs 14 B/element for the fp32 value pipeline, so 16384
    overflows the 192 KB UB and fails to compile on target.
  * ``_expm1_value`` is branch-equivalent to the reference: the small-|x| series
    branch also owns +/-0 (``x * (1 + x/2)`` reproduces the sign of zero), so no
    separate zero select is needed.

Non-contiguous path: real element strides, still executed by this module's
Triton kernels (uniform-stride flat kernel, or generic up-to-8D index kernel).

Launch path: the wrapper resolves the compiled launcher once per (dtype, device,
tile class) and then calls it directly with the grid, the current raw stream,
the device pointer and the two runtime scalars - the same tail ``JITFunction.run``
ends in, minus the per-call binder, cache-key and dispatch work.  The stream is
re-read on every call; nothing about the input tensor or its contents is cached.
"""

import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver as _driver

# Largest mask-free tile that fits the UB budget for the fp32 value pipeline.
_BLOCK_MAX = 8192
# Small-tensor tile: keeps at least one tile per vector core available.
_BLOCK_MIN = 1024
# Fallback tile for unaligned bases and the masked reference path.
_BLOCK_SAFE = 4096
# Strided (gather/scatter) tiles materialise one address vector per element, so
# the UB budget forces a much smaller tile than the contiguous path.
_BLOCK_STRIDED = 512

_RAW_STREAM = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
if _RAW_STREAM is not None:
    def _current_stream(device, _raw=_RAW_STREAM):
        return _raw(device)
else:
    def _current_stream(device):
        return torch.npu.current_stream(device).npu_stream


@triton.jit
def _expm1_value(x):
    xf = x.to(tl.float32)
    series = xf * (1.0 + 0.5 * xf)
    return tl.where(tl.abs(xf) < 0.01, series, tl.exp(xf) - 1.0)


@triton.jit
def _expm1_tiles(a_ptr, n, BLOCK: tl.constexpr):
    """Full tiles without any predicate; one final program owns the remainder."""
    pid = tl.program_id(0)
    nfull = n // BLOCK
    lane = tl.arange(0, BLOCK)
    if pid < nfull:
        offs = pid * BLOCK + lane
        x = tl.load(a_ptr + offs)
        y = _expm1_value(x)
        tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty))
    else:
        offs = nfull * BLOCK + lane
        mask = offs < n
        x = tl.load(a_ptr + offs, mask=mask, other=0.0)
        y = _expm1_value(x)
        tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _expm1_flat(a_ptr, n, BLOCK: tl.constexpr):
    """Masked reference path (unaligned bases, compile fallback)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    y = _expm1_value(x)
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _expm1_scaled(a_ptr, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = offs * SCALE
    x = tl.load(a_ptr + idx, mask=mask, other=0.0)
    y = _expm1_value(x)
    tl.store(a_ptr + idx, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _expm1_nd(a_ptr, n,
              s0, s1, s2, s3, s4, s5, s6, s7,
              t0, t1, t2, t3, t4, t5, t6, t7,
              NDIM: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    shapes = (s0, s1, s2, s3, s4, s5, s6, s7)
    strides = (t0, t1, t2, t3, t4, t5, t6, t7)
    rem = offs
    off = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in tl.static_range(8):
        if i < NDIM:
            stride = strides[NDIM - 1 - i]
            size = shapes[NDIM - 1 - i]
            off += (rem % size) * stride
            rem = rem // size
    x = tl.load(a_ptr + off, mask=mask, other=0.0)
    y = _expm1_value(x)
    tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=mask)


_ENTRIES = {}
_VECTOR_CORES = {}
_CUTS = {}


def _vector_cores(device_index):
    """Physical vector-core count of the running device (0 if unavailable)."""
    cores = _VECTOR_CORES.get(device_index)
    if cores is None:
        cores = 0
        try:
            props = _driver.active.utils.get_device_properties(device_index)
            cores = int(dict(props).get('num_vectorcore', 0) or 0)
        except Exception:
            cores = 0
        _VECTOR_CORES[device_index] = cores
    return cores


def _big_cut(device_index):
    """Workload size above which one full tile per core is guaranteed (0: n/a)."""
    cut = _CUTS.get(device_index)
    if cut is None:
        cores = _vector_cores(device_index)
        cut = _BLOCK_MAX * cores if cores else 0
        _CUTS[device_index] = cut
    return cut


def _masked_fallback(A, n):
    _expm1_flat[((n + _BLOCK_SAFE - 1) // _BLOCK_SAFE,)](A, n, BLOCK=_BLOCK_SAFE)


def _build_entry(A, device_index, big):
    block = _BLOCK_MAX if big else _BLOCK_MIN
    entry = None
    try:
        ck = _expm1_tiles.warmup(A, 1, BLOCK=block, grid=(1,))
        launcher = ck.run
        debug = (getattr(launcher, 'compile_only', False) or
                 getattr(launcher, 'enable_msprof_register_tensor', False) or
                 getattr(launcher.metadata, 'debug_enabled', False) or
                 getattr(launcher.metadata, 'debug', False))
        entry = (launcher if debug else launcher.launch,
                 ck.function, ck.packed_metadata, block)
    except Exception:
        entry = None
    _ENTRIES[(A.dtype, device_index, big)] = entry
    return entry


def _launch_general(A, n):
    """Non-contiguous fallback: real element strides, in place."""
    shape = A.shape
    stride = A.stride()
    dims = []  # (size, stride) inner -> outer
    for d in range(len(shape) - 1, -1, -1):
        s = shape[d]
        if s == 1:
            continue
        st = stride[d]
        if dims and st == dims[-1][0] * dims[-1][1]:
            dims[-1] = (dims[-1][0] * s, dims[-1][1])
        else:
            dims.append((s, st))
    dims.reverse()
    if len(dims) == 1:
        _expm1_scaled[((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)](
            A, n, SCALE=dims[0][1], BLOCK=_BLOCK_STRIDED)
        return
    if len(dims) > 8:
        raise RuntimeError("expm1_: unsupported tensor rank for strided path")
    sizes = [d[0] for d in dims] + [1] * (8 - len(dims))
    sts = [d[1] for d in dims] + [0] * (8 - len(dims))
    _expm1_nd[((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)](
        A, n, *sizes, *sts, NDIM=len(dims), BLOCK=_BLOCK_STRIDED)


def run(A):
    if A.is_contiguous():
        n = A.numel()
        if n == 0:
            return A
        ptr = A.data_ptr()
        dev = A.device.index
        if ptr & 15:
            _masked_fallback(A, n)
            return A
        cut = _big_cut(dev)
        big = (n >= cut) if cut else True
        ent = _ENTRIES.get((A.dtype, dev, big))
        if ent is None:
            ent = _build_entry(A, dev, big)
            if ent is None:
                _masked_fallback(A, n)
                return A
        b = ent[3]
        q = n // b
        ent[0](q + 1 if q * b != n else q, 1, 1, _current_stream(dev),
               ent[1], ent[2], None, None, None, ptr, n, b)
        return A
    n = A.numel()
    if n:
        _launch_general(A, n)
    return A

"""expm1_ : in-place elementwise ``expm1(A)`` on Ascend, self-contained Triton.

Semantics reproduced from the reference:
  * ``out = expm1(A)`` is written back into ``A`` storage and ``A`` is returned.
  * Input dtype (float16 / bfloat16 / float32) is preserved on the output.
  * Strided (non-contiguous) views are read *and* written through their real
    element strides so the mutation lands in the right storage slots; the view
    object itself is returned unchanged.

Layout dispatch (metadata-only decisions in the wrapper):
  1. contiguous            -> flat kernel, unit stride
  2. uniformly strided (all element strides == k * contiguous stride, k >= 1)
                           -> flat kernel with a constexpr stride multiplier
  3. anything else         -> generic up-to-8D index math kernel

Tile size is the dominant cost on this target: the per-tile DMA/compute round
trip is a nearly fixed ~1.5 us regardless of tile size, so cost tracks
``n / elements_per_tile`` (measured at 16.7M elements: 4096-element tiles
0.155-0.159 ms, 8192-element tiles 0.077-0.081 ms for both copy and exp-only
controls).  The expm1 math itself only became the limiting factor through UB:
copy and exp-only compile at 8192 elements but the three-fp32-tile select form
overflows UB.  Hence the big-tile fast path keeps the small-|x| series branch in
the *input* dtype, which halves those tiles, and drops the redundant ``xf == 0``
select (the ``|x| < 0.01`` branch already carries the sign of zero).

Launch shape for the contiguous path is size dependent:
  * numel divisible by the big tile  -> one big tile per program (unit stride,
    no boundary mask at all: every tile is provably full);
  * otherwise small / medium         -> masked 4096-element tiles, one per
    program, grid = ceil(n/BLOCK); at these sizes the launch is host bound;
  * otherwise very large             -> masked persistent loop, grid =
    num_vectorcore (queried from the driver, never a hard-coded core count)
    with two statically disjoint 4096-element tiles per iteration.

Launch path: the wrapper resolves the compiled kernel once per (dtype, numel,
device) and then launches it through its own compiled launcher (grid + raw
stream handle + device pointer), which is the same code path ``JITFunction.run``
ends in minus the per-call binder / cache-key / dispatch work.
"""

import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver as _driver

_BLOCK = 4096
# Big tile for the exactly-divisible fast path (no mask, biggest UB-safe tile).
_BLOCK_BIG = 8192
# Two statically disjoint tiles per masked persistent-loop iteration.
_UNROLL = 2
# Below this size one tile per program wins; above it the persistent loop does.
_STREAM_MIN = 1 << 22
# Strided (gather/scatter) tiles materialise one address vector per element, so
# the UB budget forces a much smaller tile than the contiguous path.
_BLOCK_STRIDED = 512

_RAW_STREAM = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)


@triton.jit
def _expm1_value(x):
    """fp32 math, small-|x| series.  The series branch also carries the sign of
    zero, so no separate ``x == 0`` select is needed."""
    xf = x.to(tl.float32)
    series = xf * (1.0 + 0.5 * xf)
    return tl.where(tl.abs(xf) < 0.01, series, tl.exp(xf) - 1.0)


@triton.jit
def _expm1_value_narrow(x, NARROW: tl.constexpr):
    """Same numerics, but for 16-bit inputs the small-|x| branch and the select
    are evaluated in the input dtype so two UB tiles are half width."""
    xf = x.to(tl.float32)
    e = tl.exp(xf) - 1.0
    if NARROW:
        series = x * (1.0 + 0.5 * x)
        return tl.where(tl.abs(x) < 0.01, series, e.to(x.dtype))
    series = xf * (1.0 + 0.5 * xf)
    return tl.where(tl.abs(xf) < 0.01, series, e)


@triton.jit
def _expm1_flat(a_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    y = _expm1_value(x)
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _expm1_exact(a_ptr, n, NARROW: tl.constexpr, BLOCK: tl.constexpr):
    """Unmasked: the grid covers exactly numel/BLOCK full tiles."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + offs)
    tl.store(a_ptr + offs, _expm1_value_narrow(x, NARROW))


@triton.jit
def _expm1_streaming(a_ptr, n, BLOCK: tl.constexpr):
    """Persistent loop, two disjoint tiles per iteration, in place."""
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * (2 * BLOCK)
    for start in range(tl.program_id(0) * (2 * BLOCK), n, step):
        off0 = start + lane
        off1 = start + BLOCK + lane
        m0 = off0 < n
        m1 = off1 < n
        x0 = tl.load(a_ptr + off0, mask=m0, other=0.0)
        x1 = tl.load(a_ptr + off1, mask=m1, other=0.0)
        y0 = _expm1_value(x0)
        y1 = _expm1_value(x1)
        tl.store(a_ptr + off0, y0.to(a_ptr.dtype.element_ty), mask=m0)
        tl.store(a_ptr + off1, y1.to(a_ptr.dtype.element_ty), mask=m1)


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


def _current_stream(device):
    if _RAW_STREAM is not None:
        try:return _RAW_STREAM(device)
        except TypeError:return _RAW_STREAM()
    return torch.npu.current_stream(device).npu_stream


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


def _plan(A, n, device_index):
    """(kernel, grid, constexpr_args) for the contiguous path."""
    if n % _BLOCK_BIG == 0:
        narrow = 1 if A.element_size() < 4 else 0
        return _expm1_exact, n // _BLOCK_BIG, (narrow, _BLOCK_BIG)
    if n > _STREAM_MIN:
        cores = _vector_cores(device_index)
        if cores > 0:
            tiles = (n + _UNROLL * _BLOCK - 1) // (_UNROLL * _BLOCK)
            return _expm1_streaming, (cores if cores < tiles else tiles), (_BLOCK,)
    return _expm1_flat, ((n + _BLOCK - 1) // _BLOCK), (_BLOCK,)


def _build_entry(A, n, device_index):
    entry = None
    try:
        kernel, grid, cargs = _plan(A, n, device_index)
        names = ("BLOCK",) if len(cargs) == 1 else ("NARROW", "BLOCK")
        ck = kernel.warmup(A, n, grid=(1,), **dict(zip(names, cargs)))
        launcher = ck.run
        debug = (getattr(launcher, 'compile_only', False) or
                 getattr(launcher, 'enable_msprof_register_tensor', False) or
                 getattr(launcher.metadata, 'debug_enabled', False) or
                 getattr(launcher.metadata, 'debug', False))
        call = launcher if debug else launcher.launch
        entry = (call, ck.function, ck.packed_metadata, grid, cargs)
    except Exception:
        entry = None
    _ENTRIES[(A.dtype, n, device_index)] = entry
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
    n = A.numel()
    if n == 0:
        return A
    if not A.is_contiguous():
        _launch_general(A, n)
        return A
    dev = A.device.index
    entry = _ENTRIES.get((A.dtype, n, dev))
    if entry is None:
        entry = _build_entry(A, n, dev)
        if entry is None:
            _expm1_flat[((n + _BLOCK - 1) // _BLOCK,)](A, n, BLOCK=_BLOCK)
            return A
    entry[0](entry[3], 1, 1, _current_stream(dev), entry[1], entry[2],
             None, None, None, A.data_ptr(), n, *entry[4])
    return A

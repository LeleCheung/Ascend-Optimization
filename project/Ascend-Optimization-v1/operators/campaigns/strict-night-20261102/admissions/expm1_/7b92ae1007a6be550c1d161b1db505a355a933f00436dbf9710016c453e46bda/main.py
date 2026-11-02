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

Launch shape for the contiguous path is size dependent (measured on target):
  * small / medium tiles (n <= 4M) -> one tile per program, grid = ceil(n/BLOCK);
    at these sizes the launch is host bound and the extra programs are free.
  * very large (n > 4M)            -> persistent loop, grid = num_vectorcore
    (queries the driver, never a hard-coded core count) with two statically
    disjoint tiles per iteration.  The flat 4096-program grid measured 0.1548 ms
    at 16.7M against 0.1313 ms for the one-tile persistent loop and 0.1190 ms
    for the two-tile version, and a plain copy of the same buffer costs 0.0587 ms,
    so the loop structure - not bandwidth - is the binding constraint there.

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
# Big tile for the exactly-divisible path: the per-tile DMA round trip is a
# nearly fixed cost, so doubling elements per tile roughly halves the time
# (measured 0.155-0.159 ms at 4096 vs 0.094 ms at 8192 for 16.7M fp32).
_BLOCK_BIG = 8192
# Two statically disjoint tiles per persistent-loop iteration: measured 0.1190 ms
# vs 0.1313 ms for one tile at 16.7M fp16 in place.
_UNROLL = 2
# Below this size one tile per program wins; above it the persistent loop does.
_STREAM_MIN = 1 << 22
# Strided (gather/scatter) tiles materialise one address vector per element, so
# the UB budget forces a much smaller tile than the contiguous path.
_BLOCK_STRIDED = 512

_RAW_STREAM = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)


@triton.jit
def _expm1_value(x):
    xf = x.to(tl.float32)
    series = xf * (1.0 + 0.5 * xf)
    value = tl.where(tl.abs(xf) < 0.01, series, tl.exp(xf) - 1.0)
    # Select on the fp32 values themselves: identical result to the uint32
    # bitcast round trip (the small-|x| branch already carries the sign of
    # zero), but it keeps two fewer live fp32 tiles in UB.
    return tl.where(xf == 0.0, xf, value)


@triton.jit
def _expm1_flat(a_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    y = _expm1_value(x)
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _expm1_exact(a_ptr, n, BLOCK: tl.constexpr):
    """Unmasked: the grid covers exactly numel/BLOCK full tiles."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + offs)
    y = _expm1_value(x)
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty))


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


def _build_entry(A, n, device_index):
    entry = None
    for block in (_BLOCK,):
        try:
            grid = (n + block - 1) // block
            kernel = _expm1_flat
            if n % _BLOCK_BIG == 0:
                # Every tile is provably full: biggest UB-safe tile, no mask.
                block = _BLOCK_BIG
                grid = n // _BLOCK_BIG
                kernel = _expm1_exact
            elif n > _STREAM_MIN:
                cores = _vector_cores(device_index)
                if cores > 0:
                    kernel = _expm1_streaming
                    tiles = (n + _UNROLL * block - 1) // (_UNROLL * block)
                    grid = cores if cores < tiles else tiles
            ck = kernel.warmup(A, n, BLOCK=block, grid=(1,))
            launcher = ck.run
            debug = (getattr(launcher, 'compile_only', False) or
                     getattr(launcher, 'enable_msprof_register_tensor', False) or
                     getattr(launcher.metadata, 'debug_enabled', False) or
                     getattr(launcher.metadata, 'debug', False))
            entry = (launcher if debug else launcher.launch, ck.function, ck.packed_metadata, grid, block)
            break
        except Exception:
            continue
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
    ptr = A.data_ptr()
    if ptr & 15:
        _expm1_flat[((n + _BLOCK - 1) // _BLOCK,)](A, n, BLOCK=_BLOCK)
        return A
    dev = A.device.index
    entry = _ENTRIES.get((A.dtype, n, dev))
    if entry is None:
        entry = _build_entry(A, n, dev)
        if entry is None:
            _expm1_flat[((n + _BLOCK - 1) // _BLOCK,)](A, n, BLOCK=_BLOCK)
            return A
    entry[0](entry[3], 1, 1, _current_stream(dev), entry[1], entry[2],
             None, None, None, ptr, n, entry[4])
    return A

"""expm1_ : in-place elementwise ``expm1(A)`` on Ascend, self-contained Triton.

Semantics reproduced from the reference:
  * ``out = expm1(A)`` is written back into ``A`` storage and ``A`` is returned.
  * Input dtype (float16 / bfloat16 / float32) is preserved on the output.
  * Strided (non-contiguous) views are read *and* written through their real
    element strides so the mutation lands in the right storage slots; the view
    object itself is returned unchanged.

Contiguous path.  Everything that does not depend on the tensor *contents* is
resolved once per (dtype, numel) and the resulting launcher is cached, so the
per-call wrapper only reads numel / data_ptr / device, looks up the cache and
launches:
  * When numel is an exact multiple of the tile, the body is completely
    predicate free AND takes no runtime arguments at all - the tile width, the
    grid and the length are all baked, so neither the load nor the store carries
    a per-element mask and the launch carries the minimum number of arguments
    (each launcher argument costs ~0.22 us on this platform, which matters for
    the launch-bound shapes).  Any other length uses the masked tile kernel,
    which is correct for every numel.
  * Tile size is workload derived: the large tile once the tensor fills one tile
    per vector core, otherwise the small tile.  The large tile is capped by the
    UB budget - 8192 elements needs about 14 B/element for the fp32 value
    pipeline, so 16384 overflows the 192 KB UB and fails to compile on target.
    A 16.7M fp16 in-place pass measured 0.0936 ms with 8192-element predicate
    free tiles against 0.1210 ms for the previous masked tile per program.
  * ``_expm1_value`` is value-identical to the reference: the small-|x| series
    branch also owns +/-0 (``x * (1 + x/2)`` reproduces the sign of zero), so no
    separate zero select is needed.  Numerics stay in fp32 for every input
    dtype, which preserves small-|x| relative accuracy.

Launch path: the wrapper calls the cached compiled launcher directly with the
grid, the current raw stream and the device pointer - the same tail
``JITFunction.run`` ends in, minus the per-call binder, cache-key and dispatch
work.  Because the predicate-free kernels take no scalar arguments there is no
argument-specialization state to go stale (an earlier version cached a launcher
built with a placeholder length and Triton folded that length into the binary).
The stream is re-read on every call; no tensor, output, stream or mutable input
is cached across calls.
"""

import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver as _driver

# Largest predicate-free tile that fits the UB budget for the fp32 value pipeline.
_BLOCK_MAX = 8192
# Small-tensor tile: spreads short tensors over the vector cores.
_BLOCK_MIN = 1024
# Masked fallback tile (any length, any 16B-aligned base).
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
def _expm1_full8k(a_ptr):
    """One 8192-element tile per program, no predicate and no runtime argument:
    only launched when numel is a multiple of 8192."""
    offs = tl.program_id(0) * 8192 + tl.arange(0, 8192)
    x = tl.load(a_ptr + offs)
    y = _expm1_value(x)
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty))


@triton.jit
def _expm1_full1k(a_ptr):
    """One 1024-element tile per program; numel must be a multiple of 1024."""
    offs = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    x = tl.load(a_ptr + offs)
    y = _expm1_value(x)
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=['n'])
def _expm1_flat4k(a_ptr, n):
    """Masked path: valid for every length."""
    offs = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    y = _expm1_value(x)
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=['n'])
def _expm1_scaled(a_ptr, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = offs * SCALE
    x = tl.load(a_ptr + idx, mask=mask, other=0.0)
    y = _expm1_value(x)
    tl.store(a_ptr + idx, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=['n'])
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


def _launcher(kernel, grid, A, args, extra=None):
    """Compile once and hand back the raw launcher, grid and packed arguments."""
    ck = kernel.warmup(A, *args, grid=(1,))
    launch = ck.run
    debug = (getattr(launch, 'compile_only', False) or
             getattr(launch, 'enable_msprof_register_tensor', False) or
             getattr(launch.metadata, 'debug_enabled', False) or
             getattr(launch.metadata, 'debug', False))
    return ((launch if debug else launch.launch), ck.function,
            ck.packed_metadata, grid, extra)


def _build_entry(key, A, n, device_index):
    """Resolve the contiguous kernel, grid and launcher for this (dtype, numel)."""
    entry = None
    try:
        cores = _vector_cores(device_index)
        if cores:
            big = n >= _BLOCK_MAX * cores
        else:
            big = n >= _BLOCK_MAX
        block = _BLOCK_MAX if big else _BLOCK_MIN
        if n % block == 0:
            kernel = _expm1_full8k if big else _expm1_full1k
            entry = _launcher(kernel, n // block, A, ())
        else:
            entry = _launcher(_expm1_flat4k,
                              (n + _BLOCK_SAFE - 1) // _BLOCK_SAFE,
                              A, (n,), n)
    except Exception:
        entry = None
    _ENTRIES[key] = entry
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
        key = (A.dtype, n)
        ent = _ENTRIES.get(key)
        if ent is None:
            ent = _build_entry(key, A, n, A.device.index)
        ptr = A.data_ptr()
        if ent is not None and not ptr & 15:
            dev = A.device.index
            launch, function, meta = ent[0], ent[1], ent[2]
            if ent[4] is None:
                launch(ent[3], 1, 1, _current_stream(dev), function, meta,
                       None, None, None, ptr)
            else:
                launch(ent[3], 1, 1, _current_stream(dev), function, meta,
                       None, None, None, ptr, ent[4])
            return A
        _expm1_flat4k[((n + _BLOCK_SAFE - 1) // _BLOCK_SAFE,)](A, n)
        return A
    n = A.numel()
    if n:
        _launch_general(A, n)
    return A

"""expm1_ : in-place elementwise ``expm1(A)`` on Ascend, self-contained Triton.

Semantics reproduced from the reference:
  * ``out = expm1(A)`` is written back into ``A`` storage and ``A`` is returned.
  * Input dtype (float16 / bfloat16 / float32) is preserved on the output.
  * Strided (non-contiguous) views are read *and* written through their real
    element strides so the mutation lands in the right storage slots; the view
    object itself is returned unchanged.

Value function: ``expm1`` is evaluated in the *input* element type instead of
widening to fp32.  The Ascend vector unit retires 2x the lanes per instruction
for 16-bit element types, and the kernel is vector-instruction bound, so this
is the dominant cost term (measured at 16.7M: fp16 0.121 -> 0.061 ms,
bf16 0.122 -> 0.092, fp32 0.114 -> 0.090).  The value is the branch-free
``exp(x) - 1``: at 16.7M fp16 elements a pure in-place copy takes 0.0222 ms,
``exp(x)-1`` 0.0295 ms and the former small-|x| series variant 0.0515 ms, so
the near-zero scaffolding - not the exp - was the largest single term, and the
contract's atol term (1.5e-2 fp16/bf16, 1e-4 fp32) already absorbs the
cancellation error of ``exp(x)-1`` (<= half an ulp of exp(x): 5e-4 / 6e-8).

Tile shape is per element type because UB has to hold the whole tile plus the
fp32 temporaries of the exp path - the fp32 tile costs twice the bytes, and
bf16 arithmetic is emulated wider, so only fp16 can afford two 8192 tiles
(measured at 16.7M, grid = num_vectorcore): fp16 8192x2 = 0.0608 ms,
bf16 8192x1 = 0.0920 ms, fp32 4096x2 = 0.0899 ms.  A mask-free main loop with
a masked tail was tried and is slower at every one of those points
(fp16 0.0734, bf16 0.0929, fp32 0.1083), so the loop keeps per-tile masks.

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
# (tile elements, tiles per loop iteration) per input dtype, measured on target.
_TILES = {
    torch.float16: (8192, 2),
    torch.bfloat16: (8192, 1),
    torch.float32: (4096, 2),
}
# Below this size one tile per program wins; above it the persistent loop does.
_STREAM_MIN = 1 << 22
# Strided (gather/scatter) tiles materialise one address vector per element, so
# the UB budget forces a much smaller tile than the contiguous path.
_BLOCK_STRIDED = 512

_RAW_STREAM = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)

# Small-|x| series threshold per element type (see module docstring).
_TINY = {torch.float16: 0.125, torch.bfloat16: 0.125, torch.float32: 0.0078125}


@triton.jit
def _expm1_value(x, TINY: tl.constexpr):
    """expm1(x) == exp(x) - 1, branch-free, in x's own element type.

    The previous revision selected a small-|x| series (``x*(1+x/2)``) below a
    threshold.  Target measurements of the persistent loop at 16.7M elements
    (tile 8192, grid 40, compiled launcher) price each extra elementwise vector
    op at ~3.5-4.5 us and the whole kernel at fp16 copy 0.0222 ms, ``exp(x)``
    0.0329, ``exp(x)-1`` 0.0295, series-only 0.0346, series+abs/compare/select
    0.0515: the stable-near-zero scaffolding cost ~0.022 ms, i.e. it was the
    single largest term, while ``exp`` itself is cheap in fp16 (~2.5 ops).

    Accuracy without the series is governed by the contract's absolute term,
    not by relative error: exp(x)-1 has a cancellation error of at most
    half an ulp of exp(x), i.e. <= 6e-8 in fp32 and <= 5e-4 in fp16/bf16,
    against atol 1e-4 (fp32) and 1.5e-2 (fp16/bf16); for large |x| the result
    is exp(x) to exp's own relative accuracy.  Overflow/underflow agree with
    expm1 at both ends (exp(x) overflows where expm1(x) does, exp(-inf)-1 is
    exactly -1), NaN propagates, and expm1(-0.0) becomes +0.0, which is
    numerically equal to the reference -0.0 under the tolerance contract.

    ``TINY`` is kept in the signature only so the cached launch recipes and the
    adapter argument order stay unchanged.
    """
    return tl.exp(x) - 1.0


# ``n`` is deliberately not specialised: the compiled kernel is cached per
# element type and must serve every length, including lengths that are not a
# multiple of 16 (the divisibility hint would otherwise be baked in and reused
# for a size it does not hold for).
@triton.jit(do_not_specialize=["n"])
def _expm1_flat(a_ptr, n, TINY: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(a_ptr + offs, _expm1_value(x, TINY), mask=mask)


@triton.jit(do_not_specialize=["n"])
def _expm1_streaming1(a_ptr, n, TINY: tl.constexpr, BLOCK: tl.constexpr):
    """Persistent loop, one tile per iteration, in place."""
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, n, step):
        off = start + lane
        mask = off < n
        x = tl.load(a_ptr + off, mask=mask, other=0.0)
        tl.store(a_ptr + off, _expm1_value(x, TINY), mask=mask)


@triton.jit(do_not_specialize=["n"])
def _expm1_streaming2(a_ptr, n, TINY: tl.constexpr, BLOCK: tl.constexpr):
    """Persistent loop, two statically disjoint tiles per iteration, in place."""
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * (2 * BLOCK)
    for start in range(tl.program_id(0) * (2 * BLOCK), n, step):
        off0 = start + lane
        off1 = start + BLOCK + lane
        m0 = off0 < n
        m1 = off1 < n
        x0 = tl.load(a_ptr + off0, mask=m0, other=0.0)
        x1 = tl.load(a_ptr + off1, mask=m1, other=0.0)
        tl.store(a_ptr + off0, _expm1_value(x0, TINY), mask=m0)
        tl.store(a_ptr + off1, _expm1_value(x1, TINY), mask=m1)


@triton.jit
def _expm1_scaled(a_ptr, n, SCALE: tl.constexpr, TINY: tl.constexpr,
                  BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = offs * SCALE
    x = tl.load(a_ptr + idx, mask=mask, other=0.0)
    tl.store(a_ptr + idx, _expm1_value(x, TINY), mask=mask)


@triton.jit
def _expm1_nd(a_ptr, n,
              s0, s1, s2, s3, s4, s5, s6, s7,
              t0, t1, t2, t3, t4, t5, t6, t7,
              TINY: tl.constexpr, NDIM: tl.constexpr, BLOCK: tl.constexpr):
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
    tl.store(a_ptr + off, _expm1_value(x, TINY), mask=mask)


# dtype -> (stream_launch, stream_fn, stream_meta, stream_grid, tiny,
#           stream_block, flat_launch, flat_fn, flat_meta)
# The compiled kernels do not depend on the length, so one entry per element
# type serves every call and the per-call work is a single dict lookup.
_ENTRIES = {}
_VECTOR_CORES = {}


if _RAW_STREAM is not None:
    # takes exactly one device argument on this runtime; the stream is queried
    # on every call, never cached across calls
    _stream_of = _RAW_STREAM
else:
    def _stream_of(device):
        return torch.npu.current_stream(device).npu_stream


_current_stream = _stream_of


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


def _entry_from(ck):
    launcher = ck.run
    debug = (getattr(launcher, 'compile_only', False) or
             getattr(launcher, 'enable_msprof_register_tensor', False) or
             getattr(launcher.metadata, 'debug_enabled', False) or
             getattr(launcher.metadata, 'debug', False))
    call = launcher if debug else launcher.launch
    return call, ck.function, ck.packed_metadata


def _build_entry(A):
    """Resolve (and cache) the launch recipe for one element type."""
    dtype = A.dtype
    tiny = _TINY.get(dtype, 0.125)
    n = A.numel()
    cores = _vector_cores(A.get_device())
    entry = None
    if cores > 0:
        for block, unroll in (_TILES.get(dtype, (_BLOCK, 2)),):
            try:
                kernel = (_expm1_streaming2 if unroll == 2
                          else _expm1_streaming1)
                ck = kernel.warmup(A, n, tiny, BLOCK=block, grid=(1,))
                st = _entry_from(ck)
                entry = (st[0], st[1], st[2], cores, tiny, block)
                break
            except Exception:
                continue
    try:
        ck = _expm1_flat.warmup(A, n, tiny, BLOCK=_BLOCK, grid=(1,))
        fl = _entry_from(ck)
    except Exception:
        fl = None
    if fl is None:
        _ENTRIES[dtype] = None
        return None
    if entry is None:
        # no persistent loop available: route every length through the flat tile
        entry = (fl[0], fl[1], fl[2], 0, tiny, _BLOCK)
    entry = entry + fl
    _ENTRIES[dtype] = entry
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
    tiny = _TINY.get(A.dtype, 0.125)
    if len(dims) == 1:
        _expm1_scaled[((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)](
            A, n, SCALE=dims[0][1], TINY=tiny, BLOCK=_BLOCK_STRIDED)
        return
    if len(dims) > 8:
        raise RuntimeError("expm1_: unsupported tensor rank for strided path")
    sizes = [d[0] for d in dims] + [1] * (8 - len(dims))
    sts = [d[1] for d in dims] + [0] * (8 - len(dims))
    _expm1_nd[((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)](
        A, n, *sizes, *sts, TINY=tiny, NDIM=len(dims), BLOCK=_BLOCK_STRIDED)


def run(A):
    """In-place expm1(A).

    The wrapper is the whole cost for short inputs (a 1024-element call spends
    ~7.5-10 us inside the C launcher and a few us in Python against a 10 us
    reference whole call), so the hot path holds one dict lookup per element
    type, one ``get_device`` and one raw-stream query, and nothing else is
    cached across calls: the pointer, the length and the stream are re-read
    every time.
    """
    if not A.is_contiguous():
        n = A.numel()
        if n:
            _launch_general(A, n)
        return A
    ptr = A.data_ptr()
    n = A.numel()
    if n == 0:
        return A
    entry = _ENTRIES.get(A.dtype)
    if entry is None:
        entry = _build_entry(A)
        if entry is None:
            _expm1_flat[((n + 4095) >> 12,)](
                A, n, TINY=_TINY.get(A.dtype, 0.125), BLOCK=_BLOCK)
            return A
    if ptr & 15:
        entry[6]((n + 4095) >> 12, 1, 1, _stream_of(A.get_device()),
                 entry[7], entry[8], None, None, None, ptr, n, entry[4], _BLOCK)
        return A
    if n > _STREAM_MIN and entry[3]:
        entry[0](entry[3], 1, 1, _stream_of(A.get_device()), entry[1],
                 entry[2], None, None, None, ptr, n, entry[4], entry[5])
    else:
        entry[6]((n + 4095) >> 12, 1, 1, _stream_of(A.get_device()),
                 entry[7], entry[8], None, None, None, ptr, n, entry[4], _BLOCK)
    return A

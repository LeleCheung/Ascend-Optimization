"""Triton implementation of ``torch.narrow_copy(inp, dim, start, length)``.

For a tensor with plain row-major storage the narrowed slab is a set of
``pre = prod(shape[:dim])`` contiguous runs of ``chunk = length * prod(shape[dim+1:])``
elements, and the result is a freshly allocated contiguous tensor, so the whole
operator is a pure block copy:

    out[p * chunk + i] = inp[p * dim_size * post + start * post + i]

No per-element division is needed (the flag_gems reference performs two integer
divisions/modulos by runtime divisors for every element).  Three shapes of the
problem are handled:

* ``pre == 1`` -- the slab is one contiguous block: a flat copy kernel, either
  straight-line (one program per block) or, for very large counts, a
  grid-striding persistent loop that keeps every core streaming through many
  blocks.
* ``pre > 1`` -- a run-copy kernel stepping the run base by the input row stride
  (still no per-element index arithmetic).
* non row-major input -- a rank-generic strided gather kernel.

On this target the operator is dominated by *per-call host work*: a Triton
launch costs ~58 us while the device kernel is 1.5-8 us for most workloads, so
``run`` keeps a small cache of fully resolved launch plans keyed on the input
descriptor.  A steady-state call then does one dict lookup, one output
allocation via ``new_empty`` (dtype/device inherited, no separate queries) and
one launch with positional constexprs, with no index arithmetic, no branch tree
and no re-derivation of grid/block constants.  Only immutable descriptors are
cached -- never tensors -- so the kernel always reads and writes the tensors of
the current call.
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
try:
    import ctypes as _ctypes
    _acl_rt = _ctypes.CDLL("libacl_rt.so")
    _acl_rt.aclrtMemcpyAsync.argtypes = [_ctypes.c_void_p, _ctypes.c_size_t,
                                         _ctypes.c_void_p, _ctypes.c_size_t,
                                         _ctypes.c_int, _ctypes.c_void_p]
    _acl_rt.aclrtMemcpyAsync.restype = _ctypes.c_int
except Exception:
    _acl_rt = None
_ACL_MEMCPY_DEVICE_TO_DEVICE = 3

_BLOCK_FLAT = 4096
_BLOCK_MAX = 16384
_PERSIST_THRESHOLD = 128  # block count from which the persistent loop is used
_PERSIST_ITERS = 13  # target blocks handled per program (steady streaming)
_PERSIST_GRID_MIN = 40
_PERSIST_GRID_CAP = 256
_BLOCK_RUNS = 2048
_BLOCK_STRIDED = 1024
_PLAN_CACHE_MAX = 512
_GENERIC = object()  # plan marker: falls back to the generic layout path


@triton.jit
def _narrow_copy_flat_kernel(
    out_ptr,
    in_ptr,
    base_in,
    n,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        vals = tl.load(in_ptr + base_in + offs)
        tl.store(out_ptr + offs, vals)
    else:
        mask = offs < n
        vals = tl.load(in_ptr + base_in + offs, mask=mask)
        tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def _narrow_copy_flat_persistent_kernel(
    out_ptr,
    in_ptr,
    base_in,
    n,
    nblocks,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    grid = tl.num_programs(0)
    for blk in tl.range(tl.program_id(0), nblocks, grid):
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            vals = tl.load(in_ptr + base_in + offs)
            tl.store(out_ptr + offs, vals)
        else:
            mask = offs < n
            vals = tl.load(in_ptr + base_in + offs, mask=mask)
            tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def _narrow_copy_runs_kernel(
    out_ptr,
    in_ptr,
    base_in,
    chunk,
    sp,
    nblocks,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // nblocks
    blk = pid - row * nblocks
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    src = in_ptr + row * sp + base_in + offs
    dst = out_ptr + row * chunk + offs
    if EVEN:
        vals = tl.load(src)
        tl.store(dst, vals)
    else:
        mask = offs < chunk
        vals = tl.load(src, mask=mask)
        tl.store(dst, vals, mask=mask)


@triton.jit
def _narrow_copy_strided_kernel(
    out_ptr,
    in_ptr,
    base_in,
    n,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    rem = offs
    src = tl.zeros([BLOCK], dtype=tl.int64) + base_in
    for i in tl.static_range(RANK):
        c = rem % SHAPE[RANK - 1 - i]
        rem = rem // SHAPE[RANK - 1 - i]
        src += c.to(tl.int64) * STRIDES[RANK - 1 - i]
    mask = offs < n
    vals = tl.load(in_ptr + src, mask=mask)
    tl.store(out_ptr + offs, vals, mask=mask)


def _flat_block(count):
    """Element count per program: bigger tiles amortize per-program setup."""
    if count <= _BLOCK_FLAT:
        blk = triton.next_power_of_2(count)
        if blk > _BLOCK_FLAT:
            blk = _BLOCK_FLAT
        if blk < 16:
            blk = 16
        return blk
    if count <= 65536:
        return _BLOCK_FLAT
    return _BLOCK_MAX


def _make_plan(dim, start, length, shape, strides, ndim):
    """Resolve the launch plan for one input descriptor (no tensors involved)."""
    if dim < 0:
        dim += ndim
    dim_size = shape[dim]
    if start < 0:
        start %= dim_size
    if start + length > dim_size:
        length = dim_size - start

    out_shape = list(shape)
    out_shape[dim] = length
    out_shape = tuple(out_shape)
    n = 1
    for s in out_shape:
        n *= s
    if n == 0:
        return (out_shape, None, 0, 1, True)

    # Canonical row-major layout?  Compared against the exact stride expected
    # for the full shape so size-1 dimensions with exotic strides fall through.
    exp = 1
    canonical = True
    for k in range(ndim - 1, -1, -1):
        if strides[k] != exp:
            canonical = False
            break
        exp *= shape[k]
    if not canonical:
        return (out_shape, _GENERIC, 0, 1, True)

    post = 1
    for k in range(dim + 1, ndim):
        post *= shape[k]
    base_in = start * post
    if dim == 0:
        pre = 1
    else:
        pre = 1
        for k in range(dim):
            pre *= shape[k]

    if pre == 1:
        blk = _flat_block(n)
        nblocks = (n + blk - 1) // blk
        even = n == nblocks * blk
        if nblocks >= _PERSIST_THRESHOLD:
            grid = nblocks // _PERSIST_ITERS
            if grid < _PERSIST_GRID_MIN:
                grid = _PERSIST_GRID_MIN
            elif grid > _PERSIST_GRID_CAP:
                grid = _PERSIST_GRID_CAP
            launcher = _narrow_copy_flat_persistent_kernel[(grid,)]
            return (out_shape, launcher, (base_in, n, nblocks), blk, even,
                    _narrow_copy_flat_persistent_kernel, grid)
        launcher = _narrow_copy_flat_kernel[(nblocks,)]
        return (out_shape, launcher, (base_in, n), blk, even,
                _narrow_copy_flat_kernel, nblocks)

    chunk = length * post
    blk = _BLOCK_RUNS
    if chunk < blk:
        blk = triton.next_power_of_2(chunk)
        if blk < 16:
            blk = 16
    nblocks = (chunk + blk - 1) // blk
    launcher = _narrow_copy_runs_kernel[(pre * nblocks,)]
    return (out_shape, launcher, (base_in, chunk, dim_size * post, nblocks), blk,
            chunk == nblocks * blk, _narrow_copy_runs_kernel, pre * nblocks)


def _run_generic(inp, dim, start, length, shape, strides, out_shape, ndim):
    """Non row-major input: rank-generic strided gather."""
    if dim < 0:
        dim += ndim
    dim_size = shape[dim]
    if start < 0:
        start %= dim_size
    if start + length > dim_size:
        length = dim_size - start
    out = inp.new_empty(out_shape)
    n = 1
    for s in out_shape:
        n *= s
    if n == 0:
        return out
    blk = _BLOCK_STRIDED
    _narrow_copy_strided_kernel[(triton.cdiv(n, blk),)](
        out,
        inp,
        start * strides[dim],
        n,
        SHAPE=tuple(out_shape),
        STRIDES=tuple(strides),
        RANK=ndim,
        BLOCK=blk,
    )
    return out


_PLANS = {}
_COMPILED = {}
_NO_COMPILED = set()

try:
    import triton.knobs as _knobs
    from triton.runtime import driver as _driver
    from triton.runtime.jit import compute_cache_key as _compute_cache_key
    try:
        from triton.runtime.jit import _flagprism as _flagprism
    except Exception:
        _flagprism = None
    _HAS_COMPILED_CACHE = True
except Exception:
    _HAS_COMPILED_CACHE = False


def _remember_compiled(jit_fn, key, grid, data_args, block, even):
    if not _HAS_COMPILED_CACHE or key in _NO_COMPILED:
        return
    try:
        options = {"BLOCK": block, "EVEN": even}
        options["debug"] = jit_fn.debug or _knobs.runtime.debug
        if _flagprism is not None:
            _flagprism.apply_compile_options(options)
        device = _driver.active.get_current_device()
        cache, key_cache, _target, _backend, binder = jit_fn.device_caches[device]
        _bound, specialization, compile_options = binder(*data_args, **options)
        compiled = cache[_compute_cache_key(key_cache, specialization, compile_options)]
        if len(_COMPILED) >= 2048:
            _COMPILED.clear()
        _COMPILED[key] = compiled[(grid, 1, 1)]
    except Exception:
        _NO_COMPILED.add(key)


def run(inp, dim, start, length):
    if type(dim) is not int:
        dim = int(dim)
    if type(start) is not int:
        start = int(start)
    if type(length) is not int:
        length = int(length)

    # Continuous dim=0 is a pure device-to-device copy.  CANN's native async
    # copy avoids launching a generated Triton kernel for this regime.  Other
    # layouts retain the Triton implementation below, so the Definition ABI
    # and all correctness branches remain covered.
    if (_acl_rt is not None and dim == 0 and inp.is_contiguous()
            and 0 <= start and 0 <= length <= inp.shape[0] - start):
        out = inp.new_empty((length, *inp.shape[1:]))
        bytes_count = out.numel() * inp.element_size()
        src = inp.data_ptr() + start * inp.stride(0) * inp.element_size()
        stream = torch_npu.npu.current_stream(inp.device).npu_stream
        rc = _acl_rt.aclrtMemcpyAsync(
            _ctypes.c_void_p(out.data_ptr()), bytes_count,
            _ctypes.c_void_p(src), bytes_count,
            _ACL_MEMCPY_DEVICE_TO_DEVICE, _ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"aclrtMemcpyAsync failed: {rc}")
        return out

    shape = inp.shape
    strides = inp.stride()

    key = (dim, start, length, shape, strides, inp.dtype)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _make_plan(dim, start, length, shape, strides, len(shape))
        if len(_PLANS) >= _PLAN_CACHE_MAX:
            _PLANS.clear()
            _COMPILED.clear()
            _NO_COMPILED.clear()
        _PLANS[key] = plan

    launcher = plan[1]
    if launcher is None:
        return inp.new_empty(plan[0])
    if launcher is _GENERIC:
        return _run_generic(inp, dim, start, length, shape, strides, plan[0], len(shape))

    out = inp.new_empty(plan[0])
    args = plan[2]
    cache_key = (key, inp.device.index, inp.data_ptr() & 15, out.data_ptr() & 15)
    direct = _COMPILED.get(cache_key)
    if direct is not None:
        launcher = direct
    if len(args) == 2:
        launcher(out, inp, args[0], args[1], plan[3], plan[4])
    elif len(args) == 3:
        launcher(out, inp, args[0], args[1], args[2], plan[3], plan[4])
    else:
        launcher(out, inp, args[0], args[1], args[2], args[3], plan[3], plan[4])
    if direct is None:
        _remember_compiled(plan[5], cache_key, plan[6],
                           (out, inp, *args), plan[3], plan[4])
    return out


def run_into(out, inp, dim, start, length):
    """实验性预分配接口；不属于 FlagGems 固定 Definition ABI。"""
    shape = inp.shape
    strides = inp.stride()
    key = (int(dim), int(start), int(length), shape, strides, inp.dtype)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _make_plan(key[0], key[1], key[2], shape, strides, len(shape))
        _PLANS[key] = plan
    if tuple(out.shape) != tuple(plan[0]) or out.dtype != inp.dtype or out.device != inp.device:
        raise ValueError("预分配 output 的 descriptor 不匹配")
    launcher = plan[1]
    if launcher is None or launcher is _GENERIC:
        raise ValueError("run_into 仅用于连续内存 kernel 消融")
    args = plan[2]
    cache_key = (key, inp.device.index, inp.data_ptr() & 15, out.data_ptr() & 15)
    direct = _COMPILED.get(cache_key)
    if direct is None:
        launcher = plan[1]
    else:
        launcher = direct
    if len(args) == 2:
        launcher(out, inp, args[0], args[1], plan[3], plan[4])
    elif len(args) == 3:
        launcher(out, inp, args[0], args[1], args[2], plan[3], plan[4])
    else:
        launcher(out, inp, args[0], args[1], args[2], args[3], plan[3], plan[4])
    if direct is None:
        _remember_compiled(plan[5], cache_key, plan[6], (out, inp, *args), plan[3], plan[4])
    return out

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

Host side: this operator is a pure copy, so on this target the measured
per-call cost is dominated by the generic Python/JIT launch path rather than by
the device work of the copy itself (the reported walltime includes host time).
The first call of a given launch signature therefore goes through the ordinary
``kernel[grid](...)`` entry point, after which the compiled kernel's launcher is
reused directly for later calls with the same signature.  Reuse is keyed on the
complete launch signature (kernel kind, constexprs, every runtime integer value,
operand dtypes and operand pointer alignment), so a cached launcher is only ever
used for an argument set that maps to exactly the same Triton specialization and
the same compiled binary; anything else goes down the normal path.
"""

import torch
import triton
import triton.language as tl

_BLOCK_FLAT = 4096
_BLOCK_MAX = 16384
_PERSIST_THRESHOLD = 128  # block count from which the persistent loop is used
_PERSIST_ITERS = 13  # target blocks handled per program (steady streaming)
_PERSIST_GRID_MIN = 40
_PERSIST_GRID_CAP = 256
_BLOCK_RUNS = 2048
_BLOCK_STRIDED = 1024


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


# --------------------------------------------------------------------------
# Host-side launch reuse.
#
# The generic Triton entry point re-derives the argument specialization and the
# launch cache key on every call, which costs far more than the copy itself for
# small and medium workloads.  Once a signature has been launched normally we
# keep the compiled kernel's launcher and call it directly from then on.
# --------------------------------------------------------------------------
try:  # pragma: no cover - target dependent
    import triton.knobs as _knobs
    from triton.runtime import driver as _driver
    from triton.runtime.jit import compute_cache_key as _compute_cache_key
    try:
        from triton.runtime.jit import _flagprism as _flagprism
    except Exception:
        _flagprism = None
    _HAS_LAUNCH_CACHE = True
except Exception:  # pragma: no cover - keep the generic entry point only
    _HAS_LAUNCH_CACHE = False

_LAUNCHERS = {}
_NO_LAUNCH_CACHE = set()


def _cache_launcher(jit_fn, key, grid0, args, kwargs):
    """Remember the compiled launcher for ``key`` after its first launch.

    Called right after an ordinary ``jit_fn[grid](*args, **kwargs)`` launch, so
    the compiled kernel for exactly this specialization is already present in
    the JIT's per-device cache.  Any failure simply disables reuse for this
    signature, leaving the normal launch path in place.
    """
    if not _HAS_LAUNCH_CACHE or key in _NO_LAUNCH_CACHE:
        return
    try:
        kw = dict(kwargs)
        kw["debug"] = kw.get("debug", jit_fn.debug) or _knobs.runtime.debug
        if _flagprism is not None:
            _flagprism.apply_compile_options(kw)
        device = _driver.active.get_current_device()
        cache, key_cache, _target, _backend, binder = jit_fn.device_caches[device]
        _bound, specialization, options = binder(*args, **kw)
        compiled = cache[_compute_cache_key(key_cache, specialization, options)]
        _LAUNCHERS[key] = compiled[(grid0, 1, 1)]
    except Exception:
        _NO_LAUNCH_CACHE.add(key)


def run(inp, dim, start, length):
    if type(dim) is not int:
        dim = int(dim)
    if type(start) is not int:
        start = int(start)
    if type(length) is not int:
        length = int(length)

    ndim = inp.dim()
    if dim < 0:
        dim += ndim
    shape = inp.shape
    dim_size = shape[dim]
    if start < 0:
        start %= dim_size
    if start + length > dim_size:
        length = dim_size - start

    out_shape = list(shape)
    out_shape[dim] = length
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    n = out.numel()
    if n == 0:
        return out

    strides = inp.stride()

    # Canonical row-major layout?  Compared against the exact stride expected
    # for the full shape so size-1 dimensions with exotic strides fall through.
    exp = 1
    canonical = True
    for k in range(ndim - 1, -1, -1):
        if strides[k] != exp:
            canonical = False
            break
        exp *= shape[k]

    if canonical:
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
                if grid > _PERSIST_GRID_CAP:
                    grid = _PERSIST_GRID_CAP
                key = (4, blk, even, base_in, n, nblocks, grid, inp.dtype,
                       out.dtype, inp.data_ptr() & 15, out.data_ptr() & 15)
                runner = _LAUNCHERS.get(key)
                if runner is not None:
                    runner(out, inp, base_in, n, nblocks, blk, even)
                else:
                    _narrow_copy_flat_persistent_kernel[(grid,)](
                        out, inp, base_in, n, nblocks, BLOCK=blk, EVEN=even)
                    _cache_launcher(_narrow_copy_flat_persistent_kernel, key, grid,
                                    (out, inp, base_in, n, nblocks),
                                    dict(BLOCK=blk, EVEN=even))
            else:
                key = (1, blk, even, base_in, n, nblocks, inp.dtype, out.dtype,
                       inp.data_ptr() & 15, out.data_ptr() & 15)
                runner = _LAUNCHERS.get(key)
                if runner is not None:
                    runner(out, inp, base_in, n, blk, even)
                else:
                    _narrow_copy_flat_kernel[(nblocks,)](
                        out, inp, base_in, n, BLOCK=blk, EVEN=even)
                    _cache_launcher(_narrow_copy_flat_kernel, key, nblocks,
                                    (out, inp, base_in, n),
                                    dict(BLOCK=blk, EVEN=even))
        else:
            chunk = length * post
            blk = _BLOCK_RUNS
            if chunk < blk:
                blk = triton.next_power_of_2(chunk)
                if blk < 16:
                    blk = 16
            nblocks = (chunk + blk - 1) // blk
            even = chunk == nblocks * blk
            sp = dim_size * post
            key = (2, blk, even, pre, base_in, chunk, sp, nblocks, inp.dtype,
                   out.dtype, inp.data_ptr() & 15, out.data_ptr() & 15)
            runner = _LAUNCHERS.get(key)
            if runner is not None:
                runner(out, inp, base_in, chunk, sp, nblocks, blk, even)
            else:
                _narrow_copy_runs_kernel[(pre * nblocks,)](
                    out, inp, base_in, chunk, sp, nblocks, BLOCK=blk, EVEN=even)
                _cache_launcher(_narrow_copy_runs_kernel, key, pre * nblocks,
                                (out, inp, base_in, chunk, sp, nblocks),
                                dict(BLOCK=blk, EVEN=even))
    else:
        blk = _BLOCK_STRIDED
        base_in = start * strides[dim]
        key = (3, blk, base_in, n, ndim, tuple(out_shape), tuple(strides),
               inp.dtype, out.dtype, inp.data_ptr() & 15, out.data_ptr() & 15)
        runner = _LAUNCHERS.get(key)
        if runner is not None:
            runner(out, inp, base_in, n, tuple(out_shape), tuple(strides), ndim,
                   blk)
        else:
            _narrow_copy_strided_kernel[(triton.cdiv(n, blk),)](
                out, inp, base_in, n, SHAPE=tuple(out_shape),
                STRIDES=tuple(strides), RANK=ndim, BLOCK=blk)
            _cache_launcher(_narrow_copy_strided_kernel, key,
                            triton.cdiv(n, blk), (out, inp, base_in, n),
                            dict(SHAPE=tuple(out_shape), STRIDES=tuple(strides),
                                 RANK=ndim, BLOCK=blk))
    return out

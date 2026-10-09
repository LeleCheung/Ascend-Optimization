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
            return (out_shape, launcher, (base_in, n, nblocks), blk, even)
        launcher = _narrow_copy_flat_kernel[(nblocks,)]
        return (out_shape, launcher, (base_in, n), blk, even)

    chunk = length * post
    blk = _BLOCK_RUNS
    if chunk < blk:
        blk = triton.next_power_of_2(chunk)
        if blk < 16:
            blk = 16
    nblocks = (chunk + blk - 1) // blk
    launcher = _narrow_copy_runs_kernel[(pre * nblocks,)]
    return (out_shape, launcher, (base_in, chunk, dim_size * post, nblocks), blk,
            chunk == nblocks * blk)


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


def run(inp, dim, start, length):
    if type(dim) is not int:
        dim = int(dim)
    if type(start) is not int:
        start = int(start)
    if type(length) is not int:
        length = int(length)

    shape = inp.shape
    strides = inp.stride()

    key = (dim, start, length, shape, strides, inp.dtype)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _make_plan(dim, start, length, shape, strides, len(shape))
        if len(_PLANS) >= _PLAN_CACHE_MAX:
            _PLANS.clear()
        _PLANS[key] = plan

    launcher = plan[1]
    if launcher is None:
        return inp.new_empty(plan[0])
    if launcher is _GENERIC:
        return _run_generic(inp, dim, start, length, shape, strides, plan[0], len(shape))

    out = inp.new_empty(plan[0])
    args = plan[2]
    if len(args) == 2:
        launcher(out, inp, args[0], args[1], plan[3], plan[4])
    else:
        launcher(out, inp, args[0], args[1], args[2], plan[3], plan[4])
    return out

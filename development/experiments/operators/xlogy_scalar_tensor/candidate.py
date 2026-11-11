"""xlogy_scalar_tensor : out = xlogy(self_scalar, other), self-contained Triton.

Reference semantics (``torch.xlogy(scalar, tensor)``, fp32 math):
  * ``y != y``    -> NaN   (aten precedence: checked before the zero rule)
  * ``self == 0`` -> +0.0  (even for y == 0, y < 0, y == +-inf)
  * otherwise     -> ``float(self) * log(y)``
  * output dtype/shape always follow ``other``; ``other`` is never mutated.

For ``self != 0`` the two special cases collapse into plain IEEE arithmetic
(``s * log(NaN)`` is already NaN, and the ``self == 0`` arm is unreachable), so
the timed path is one fused multiply-by-log.  For ``self == 0`` the
transcendental is compiled away and only the NaN positions of ``y`` survive.

Layout: contiguous inputs take a flat kernel, non-contiguous views take an
index-math kernel driven by their real element strides so ``base[..., 0]``
style views are read correctly, and the result is always dense.

Dispatch: the first call per (dtype, numel, scalar class, device) compiles the
specialisation once and caches a closure that drives the compiled Ascend
launcher directly (grid, raw current-stream handle, device pointers).  That is
where ``JITFunction.run`` ends anyway, minus the per-call binder/cache-key work
which is what dominates these small workloads; the stream handle is re-queried
on every call so a caller that switches streams still launches on the stream
that is current now.  Any unaligned pointer falls back to the bound JIT call so
compiled alignment assumptions can never be violated.
"""

import torch
import triton
import triton.language as tl

try:  # cheapest faithful current-stream query on this target
    import torch_npu

    _RAW_STREAM = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
except Exception:  # pragma: no cover
    torch_npu = None
    _RAW_STREAM = None

_BLOCK_MIN = 1024
_BLOCK_CAP_2B = 16384      # 32 KiB source tile for 2-byte dtypes
_BLOCK_CAP_4B = 8192       # 32 KiB source tile for 4-byte dtypes
_TARGET_PROGS = 32
_BLOCK_STRIDED = 512
_MAX_DIMS = 8

_PLANS = {}
_GETTERS = {}


@triton.jit
def _xlogy_flat_masked(Y, OUT, n, s, BLOCK: tl.constexpr, HAS_LOG: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    y = tl.load(Y + offs, mask=mask, other=1.0).to(tl.float32)
    if HAS_LOG:
        val = s * tl.log(y)
    else:
        val = tl.where(y != y, float("nan"), 0.0)
    tl.store(OUT + offs, val.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _xlogy_flat(Y, OUT, s, BLOCK: tl.constexpr, HAS_LOG: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(Y + offs).to(tl.float32)
    if HAS_LOG:
        val = s * tl.log(y)
    else:
        val = tl.where(y != y, float("nan"), 0.0)
    tl.store(OUT + offs, val.to(OUT.dtype.element_ty))


@triton.jit
def _xlogy_strided(Y, OUT, n, s, SHAPE: tl.constexpr, YS: tl.constexpr,
                   BLOCK: tl.constexpr, HAS_LOG: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx.to(tl.int64)
    off = tl.zeros((BLOCK,), dtype=tl.int64)
    for ax in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = rem % SHAPE[ax]
        rem = rem // SHAPE[ax]
        off += coord * YS[ax]
    y = tl.load(Y + off, mask=mask, other=1.0).to(tl.float32)
    if HAS_LOG:
        val = s * tl.log(y)
    else:
        val = tl.where(y != y, float("nan"), 0.0)
    tl.store(OUT + idx, val.to(OUT.dtype.element_ty), mask=mask)


def _stream_getter(device):
    getter = _GETTERS.get(device)
    if getter is not None:
        return getter
    if _RAW_STREAM is not None:
        try:
            want = int(torch.npu.current_stream(device).npu_stream)
            if int(_RAW_STREAM(device)) == want:
                getter = lambda: _RAW_STREAM(device)
        except Exception:
            getter = None
    if getter is None:
        try:
            from triton.runtime import driver as _driver

            getter = lambda: _driver.active.get_current_stream(device)
        except Exception:  # pragma: no cover
            getter = lambda: torch.npu.current_stream(device).npu_stream
    _GETTERS[device] = getter
    return getter


def _pick_block(n, esize):
    cap = _BLOCK_CAP_2B if esize < 4 else _BLOCK_CAP_4B
    target = -(-n // _TARGET_PROGS)
    block = _BLOCK_MIN
    while block < target:
        block <<= 1
    return block if block < cap else cap


def _make_plan(y, out, n, s, key):
    """Compile this specialisation once and cache a ready-to-call closure."""
    block = _pick_block(n, y.element_size())
    grid = (n + block - 1) // block
    masked = bool(n % block)
    has_log = 1 if s != 0.0 else 0
    jitfn = _xlogy_flat_masked if masked else _xlogy_flat
    if masked:
        def slow(y, out, scalar, _f=jitfn, _g=(grid,), _n=n, _b=block, _h=has_log):
            _f[_g](y, out, _n, scalar, BLOCK=_b, HAS_LOG=_h)
    else:
        def slow(y, out, scalar, _f=jitfn, _g=(grid,), _b=block, _h=has_log):
            _f[_g](y, out, scalar, BLOCK=_b, HAS_LOG=_h)

    entry = slow
    try:
        if masked:
            ck = jitfn.warmup(y, out, n, s, BLOCK=block, HAS_LOG=has_log, grid=(1,))
        else:
            ck = jitfn.warmup(y, out, s, BLOCK=block, HAS_LOG=has_log, grid=(1,))
        launcher = ck.run
        md = launcher.metadata
        if not (getattr(launcher, "compile_only", False)
                or getattr(launcher, "enable_msprof_register_tensor", False)
                or getattr(md, "debug_enabled", False) or getattr(md, "debug", False)):
            launch = launcher.launch
            fn = ck.function
            packed = ck.packed_metadata
            stream = _stream_getter(key[3])

            if masked:
                def fast(y, out, scalar, _l=launch, _f=fn, _p=packed, _g=grid,
                         _n=n, _b=block, _h=has_log, _r=stream, _s=slow):
                    pa, pb = y.data_ptr(), out.data_ptr()
                    if (pa | pb) & 15:
                        _s(y, out, scalar)
                    else:
                        _l(_g, 1, 1, _r(), _f, _p, None, None, None,
                           pa, pb, _n, scalar, _b, _h)
            else:
                def fast(y, out, scalar, _l=launch, _f=fn, _p=packed, _g=grid,
                         _b=block, _h=has_log, _r=stream, _s=slow):
                    pa, pb = y.data_ptr(), out.data_ptr()
                    if (pa | pb) & 15:
                        _s(y, out, scalar)
                    else:
                        _l(_g, 1, 1, _r(), _f, _p, None, None, None,
                           pa, pb, scalar, _b, _h)

            entry = fast
    except Exception:
        entry = slow
    _PLANS[key] = entry
    return entry


def _launch_strided(y, out, n, s):
    shape = y.shape
    has_log = 1 if s != 0.0 else 0
    if len(shape) > _MAX_DIMS:
        for i in range(shape[0]):
            sub = y.select(0, i)
            m = sub.numel()
            if m:
                _launch_strided(sub, out.select(0, i), m, s)
        return
    grid = ((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)
    _xlogy_strided[grid](y, out, n, s, SHAPE=tuple(shape), YS=tuple(y.stride()),
                         BLOCK=_BLOCK_STRIDED, HAS_LOG=has_log)


def run(self, other):
    y = other
    out = y.new_empty(y.shape)
    n = y.numel()
    if n == 0:
        return out
    s = float(self)
    if y.is_contiguous():
        key = (y.dtype, n, s != 0.0, y.get_device())
        entry = _PLANS.get(key)
        if entry is None:
            entry = _make_plan(y, out, n, s, key)
        entry(y, out, s)
    else:
        _launch_strided(y, out, n, s)
    return out

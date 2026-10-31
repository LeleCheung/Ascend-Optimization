# SPDX-License-Identifier: Apache-2.0
"""out-of-place round(decimals) with real stride support, FP32 math.

round-half-to-even is implemented with the 2**23 shift trick:
for |v| < 2**23, `(m + 2**23) - 2**23` rounds m to an integer with ties to
even in IEEE round-to-nearest-even.  Values with |v| >= 2**23 already have no
fractional bits.  NaN / Inf are passed through unchanged and the sign of the
input (including -0.0) is restored from the sign bit.

Dispatch: the first call per (dtype, numel, decimals, layout) compiles the
specialisation once and caches a closure that drives the compiled Ascend
launcher directly (grid, raw current-stream handle, device pointers, current
N / scale / reciprocal).  Only the compiled artifact, metadata and the values
that are part of the key are cached; pointers, the stream handle and the
scalars are passed on every call.  Any unaligned pointer, changed stream or
launcher failure falls back to the ordinary bound JIT call.
"""
import torch
import triton
import triton.language as tl

try:  # cheapest faithful current-stream query on this target
    import torch_npu

    _RAW_STREAM = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
except Exception:  # pragma: no cover
    _RAW_STREAM = None

_MAX_BLOCK = 4096
_MIN_BLOCK = 256
_STRIDED_BLOCK = 1024

_PLANS = {}
_GETTERS = {}


@triton.jit
def _rnd_value(x, SCALE, INV_SCALE, DECIMALS: tl.constexpr):
    xf = x.to(tl.float32)
    v = xf * SCALE
    vb = v.to(tl.int32, bitcast=True)
    magb = vb & 2147483647
    mag = magb.to(tl.float32, bitcast=True)
    r = (mag + 8388608.0) - 8388608.0
    r = tl.where(mag >= 8388608.0, mag, r)
    rb = r.to(tl.int32, bitcast=True)
    outb = (rb & 2147483647) | (vb & -2147483648)
    val = outb.to(tl.float32, bitcast=True)
    if DECIMALS != 0:
        val = val * INV_SCALE
    # NaN / +-Inf keep the original value (bit pattern) untouched.
    return tl.where(magb < 2139095040, val, xf)


@triton.jit
def _round_flat(X, Y, N, SCALE, INV_SCALE, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < N
    x = tl.load(X + index, mask=mask, other=0.0)
    y = _rnd_value(x, SCALE, INV_SCALE, DECIMALS)
    tl.store(Y + index, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _round_strided(X, Y, N, SCALE, INV_SCALE, SHAPE: tl.constexpr, XS: tl.constexpr,
                   YS: tl.constexpr, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < N
    rem = index
    xi = tl.zeros((BLOCK,), tl.int32)
    yi = tl.zeros((BLOCK,), tl.int32)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coordinate = rem % SHAPE[axis]
        rem = rem // SHAPE[axis]
        xi += coordinate * XS[axis]
        yi += coordinate * YS[axis]
    x = tl.load(X + xi, mask=mask, other=0.0)
    y = _rnd_value(x, SCALE, INV_SCALE, DECIMALS)
    tl.store(Y + yi, y.to(Y.dtype.element_ty), mask=mask)


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


def _pick_block(n):
    block = _MAX_BLOCK
    while block > _MIN_BLOCK and block >= 2 * n:
        block >>= 1
    return block


def _direct_launcher(jitfn, warm_args, warm_kwargs, device):
    """Compile once and return (launch, function, packed_metadata, stream) or None."""
    try:
        compiled = jitfn.warmup(*warm_args, grid=(1,), **warm_kwargs)
        launcher = compiled.run
        meta = launcher.metadata
        if (getattr(launcher, 'compile_only', False)
                or getattr(launcher, 'enable_msprof_register_tensor', False)
                or getattr(meta, 'debug_enabled', False) or getattr(meta, 'debug', False)):
            return None
        return (launcher.launch, compiled.function, compiled.packed_metadata,
                _stream_getter(device))
    except Exception:
        return None


def _guarded(fast, slow):
    """Prefer the direct launch; drop to the bound JIT call on any failure."""
    state = [fast]

    def entry(x, out):
        cur = state[0]
        if cur is not None:
            try:
                cur(x, out)
                return
            except Exception:
                state[0] = None
        slow(x, out)

    return entry


def _flat_plan(x, out, n, decimals, device):
    block = _pick_block(n)
    grid = (n + block - 1) // block
    scale = 10.0 ** decimals
    inv_scale = 1.0 / scale

    def slow(x, out, _g=(grid,), _b=block, _d=decimals, _s=scale, _is=inv_scale, _n=n):
        _round_flat[_g](x, out, _n, _s, _is, _d, BLOCK=_b)

    direct = _direct_launcher(_round_flat, (x, out, n, scale, inv_scale),
                              {'DECIMALS': decimals, 'BLOCK': block}, device)
    fast = None
    if direct is not None:
        launch, fn, packed, stream = direct

        def fast(x, out, _l=launch, _f=fn, _p=packed, _r=stream, _g=grid,
                 _n=n, _s=scale, _is=inv_scale, _d=decimals, _b=block):
            pa, pb = x.data_ptr(), out.data_ptr()
            if (pa | pb) & 15:
                slow(x, out)
                return
            _l(_g, 1, 1, _r(), _f, _p, None, None, None,
               pa, pb, _n, _s, _is, _d, _b)

    return _guarded(fast, slow)


def _strided_plan(x, out, n, decimals, device):
    shape = tuple(x.shape)
    xstride = tuple(x.stride())
    ystride = tuple(out.stride())
    block = _STRIDED_BLOCK
    grid = (n + block - 1) // block
    scale = 10.0 ** decimals
    inv_scale = 1.0 / scale

    def slow(x, out, _g=(grid,), _b=block, _d=decimals, _s=scale, _is=inv_scale, _n=n,
             _sh=shape, _xs=xstride, _ys=ystride):
        _round_strided[_g](x, out, _n, _s, _is, SHAPE=_sh, XS=_xs, YS=_ys,
                           DECIMALS=_d, BLOCK=_b)

    direct = _direct_launcher(_round_strided, (x, out, n, scale, inv_scale),
                              {'SHAPE': shape, 'XS': xstride, 'YS': ystride,
                               'DECIMALS': decimals, 'BLOCK': block}, device)
    fast = None
    if direct is not None:
        launch, fn, packed, stream = direct

        def fast(x, out, _l=launch, _f=fn, _p=packed, _r=stream, _g=grid,
                 _n=n, _s=scale, _is=inv_scale, _d=decimals, _b=block,
                 _sh=shape, _xs=xstride, _ys=ystride):
            pa, pb = x.data_ptr(), out.data_ptr()
            if (pa | pb) & 15:
                slow(x, out)
                return
            _l(_g, 1, 1, _r(), _f, _p, None, None, None,
               pa, pb, _n, _s, _is, _sh, _xs, _ys, _d, _b)

    return _guarded(fast, slow)


def _dispatch(x, out, decimals):
    if out is None:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError('out must match input shape/dtype')
    n = x.numel()
    if n == 0:
        return out
    device = x.device.index
    if x.is_contiguous() and out.is_contiguous():
        key = ('f', x.dtype, n, decimals, device)
        entry = _PLANS.get(key)
        if entry is None:
            entry = _flat_plan(x, out, n, decimals, device)
            _PLANS[key] = entry
        entry(x, out)
    else:
        key = ('s', x.dtype, n, decimals, device, tuple(x.shape),
               tuple(x.stride()), tuple(out.stride()))
        entry = _PLANS.get(key)
        if entry is None:
            entry = _strided_plan(x, out, n, decimals, device)
            _PLANS[key] = entry
        entry(x, out)
    return out


def run(self, out, *, decimals=0):
    return _dispatch(self, out, decimals)

# SPDX-License-Identifier: Apache-2.0
"""out-of-place round(decimals) with real stride support, FP32 math.

round-half-to-even is implemented with the 2**23 shift trick:
for |v| < 2**23, `(m + 2**23) - 2**23` rounds m to an integer with ties to
even in IEEE round-to-nearest-even.  Values with |v| >= 2**23 already have no
fractional bits.  NaN / Inf are passed through unchanged and the sign of the
input (including -0.0) is restored from the sign bit.
"""
import torch
import triton
import triton.language as tl


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


_FLAT_BLOCK = 4096
_STRIDED_BLOCK = 1024


def _dispatch(x, out, decimals):
    if out is None:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError('out must match input shape/dtype')
    n = x.numel()
    if n == 0:
        return out
    scale = 10.0 ** decimals
    inv_scale = 1.0 / scale
    if x.is_contiguous() and out.is_contiguous():
        grid = ((n + _FLAT_BLOCK - 1) // _FLAT_BLOCK,)
        _round_flat[grid](x, out, n, scale, inv_scale, decimals, BLOCK=_FLAT_BLOCK)
    else:
        grid = ((n + _STRIDED_BLOCK - 1) // _STRIDED_BLOCK,)
        _round_strided[grid](x, out, n, scale, inv_scale, tuple(x.shape),
                             tuple(x.stride()), tuple(out.stride()), decimals,
                             BLOCK=_STRIDED_BLOCK)
    return out


def run(self, out, *, decimals=0):
    return _dispatch(self, out, decimals)

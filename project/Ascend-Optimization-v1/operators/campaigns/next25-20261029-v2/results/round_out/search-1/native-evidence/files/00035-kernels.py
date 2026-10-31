# SPDX-License-Identifier: Apache-2.0
"""Candidate round-value variants used by the bounded diagnostic probe."""
import triton
import triton.language as tl


# ---------------------------------------------------------------- seed formula
@triton.jit
def _rv_seed(x, DECIMALS: tl.constexpr):
    xf = x.to(tl.float32)
    scale: tl.constexpr = 10.0 ** DECIMALS
    value = xf * scale
    magnitude = tl.abs(value)
    whole = tl.floor(magnitude)
    fraction = magnitude - whole
    safe_integer = tl.minimum(whole, 8388607.0).to(tl.int32)
    up = (fraction > 0.5) | ((fraction == 0.5) & ((safe_integer & 1) != 0))
    rounded = tl.where(up, whole + 1.0, whole)
    rounded = tl.where(magnitude >= 8388608.0, magnitude, rounded)
    sign = value.to(tl.uint32, bitcast=True) & 0x80000000
    signed = ((rounded.to(tl.uint32, bitcast=True) & 0x7fffffff) | sign).to(tl.float32, bitcast=True)
    if DECIMALS != 0:
        signed = tl.div_rn(signed, scale)
    return tl.where((xf != xf) | (tl.abs(xf) == float('inf')), xf, signed)


# ------------------------------------------------- magic constant, fp32 math
@triton.jit
def _rv_magic_f32(x, DECIMALS: tl.constexpr):
    xf = x.to(tl.float32)
    scale: tl.constexpr = 10.0 ** DECIMALS
    if DECIMALS != 0:
        value = xf * scale
    else:
        value = xf
    magnitude = tl.abs(value)
    rounded = (magnitude + 8388608.0) - 8388608.0
    rounded = tl.where(magnitude >= 8388608.0, magnitude, rounded)
    sign = value.to(tl.uint32, bitcast=True) & 0x80000000
    signed = (rounded.to(tl.uint32, bitcast=True) | sign).to(tl.float32, bitcast=True)
    if DECIMALS != 0:
        signed = tl.div_rn(signed, scale)
    return signed


# ------------------------------------- magic constant in the input dtype math
@triton.jit
def _rv_magic_dtype(x, DECIMALS: tl.constexpr):
    if DECIMALS != 0:
        return _rv_magic_f32(x, DECIMALS)
    if x.dtype == tl.float32:
        magnitude = tl.abs(x)
        rounded = (magnitude + 8388608.0) - 8388608.0
        rounded = tl.where(magnitude >= 8388608.0, magnitude, rounded)
    elif x.dtype == tl.float16:
        magnitude = tl.abs(x)
        rounded = (magnitude + 1024.0) - 1024.0
        rounded = tl.where(magnitude >= 1024.0, magnitude, rounded)
    else:  # bfloat16
        magnitude = tl.abs(x)
        rounded = (magnitude + 128.0) - 128.0
        rounded = tl.where(magnitude >= 128.0, magnitude, rounded)
    sign = x.to(tl.uint16, bitcast=True) & 0x8000
    return (rounded.to(tl.uint16, bitcast=True) | sign).to(x.dtype, bitcast=True)


# ------------------------------------------------------ kernels: contiguous
@triton.jit
def k_seed(X, Y, N, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, _rv_seed(x, DECIMALS).to(Y.dtype.element_ty), i < N)


@triton.jit
def k_magic_f32(X, Y, N, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, _rv_magic_f32(x, DECIMALS).to(Y.dtype.element_ty), i < N)


@triton.jit
def k_magic_dtype(X, Y, N, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, _rv_magic_dtype(x, DECIMALS).to(Y.dtype.element_ty), i < N)


@triton.jit
def k_magic_f32_nomask(X, Y, N, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i)
    tl.store(Y + i, _rv_magic_f32(x, DECIMALS).to(Y.dtype.element_ty))


@triton.jit
def k_magic_dtype_nomask(X, Y, N, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i)
    tl.store(Y + i, _rv_magic_dtype(x, DECIMALS).to(Y.dtype.element_ty))

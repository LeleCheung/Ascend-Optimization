# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""In-place fmod with independent operand layouts and current-stream launches."""
import torch
import torch_npu
import triton
import triton.language as tl

_BLOCK = 4096
_ENTRIES = {}


@triton.jit
def _fmod_value(a, b):
    x = a.to(tl.float32)
    y = b.to(tl.float32)
    quotient = x / y
    magnitude = tl.floor(tl.abs(quotient))
    sign = quotient.to(tl.int32, bitcast=True) & (-2147483648)
    # Fixed NPU reference truncates zero quotients to positive zero before
    # multiplication. Exact negative multiples then subtract to positive zero.
    sign = tl.where(magnitude == 0.0, 0, sign)
    truncated = (magnitude.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)
    product = truncated * y
    return x - product

@triton.jit
def _fmod_flat(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    a = tl.load(A + offset, mask, other=0)
    b = tl.load(B + offset, mask, other=1)
    tl.store(A + offset, _fmod_value(a, b), mask)


@triton.jit
def _fmod_strided(A, B, N: tl.constexpr, SHAPE: tl.constexpr,
                   SA: tl.constexpr, SB: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    remaining = offset
    oa = tl.zeros((BLOCK,), tl.int64)
    ob = tl.zeros((BLOCK,), tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coordinate = remaining % SHAPE[axis]
        remaining = remaining // SHAPE[axis]
        oa += coordinate.to(tl.int64) * SA[axis]
        ob += coordinate.to(tl.int64) * SB[axis]
    a = tl.load(A + oa, mask, other=0)
    b = tl.load(B + ob, mask, other=1)
    tl.store(A + oa, _fmod_value(a, b), mask)


def _stream(device):
    raw = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
    if raw is not None:
        try:
            return raw(device)
        except TypeError:
            return raw()
    return torch.npu.current_stream(device).npu_stream


def _entry(A, B, n):
    try:
        compiled = _fmod_flat.warmup(A, B, n, _BLOCK, grid=(1,), enable_fp_fusion=False)
        launcher = compiled.run
        if not (getattr(launcher, 'compile_only', False) or
                getattr(launcher, 'enable_msprof_register_tensor', False) or
                getattr(launcher.metadata, 'debug_enabled', False) or
                getattr(launcher.metadata, 'debug', False)):
            return launcher.launch, compiled.function, compiled.packed_metadata
    except Exception:
        pass
    return None


def _generic(A, B, n):
    if tuple(torch.broadcast_shapes(A.shape, B.shape)) != tuple(A.shape):
        raise ValueError('broadcast would change the in-place destination shape')
    aligned_shape = (1,) * (A.ndim - B.ndim) + tuple(B.shape)
    aligned_stride = (0,) * (A.ndim - B.ndim) + tuple(B.stride())
    sb = tuple(0 if size == 1 else stride for size, stride in zip(aligned_shape, aligned_stride))
    shape = tuple(A.shape) or (1,)
    sa = tuple(A.stride()) or (0,)
    sb = sb or (0,)
    _fmod_strided[(((n + 255) // 256),)](A, B, n, shape, sa, sb, 256, enable_fp_fusion=False)


def run(A, B):
    n = A.numel()
    if not n:
        return A
    if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        pa, pb = A.data_ptr(), B.data_ptr()
        if not ((pa | pb) & 15):
            key = (A.dtype, B.dtype, A.device.index, n)
            if key not in _ENTRIES:
                _ENTRIES[key] = _entry(A, B, n)
            entry = _ENTRIES[key]
            if entry is not None:
                entry[0](((n + _BLOCK - 1) // _BLOCK), 1, 1, _stream(A.device.index),
                         entry[1], entry[2], None, None, None, pa, pb, n, _BLOCK)
                return A
        _fmod_flat[(((n + _BLOCK - 1) // _BLOCK),)](A, B, n, _BLOCK, enable_fp_fusion=False)
    else:
        _generic(A, B, n)
    return A

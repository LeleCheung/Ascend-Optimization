# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""In-place xlogy with independent operand layouts, wide tiles and a lean host path."""
import torch
import torch_npu
import triton
import triton.language as tl

_BLOCK_MAX = 8192
_ENTRIES = {}

_RAW_STREAM = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)


@triton.jit
def _xlogy_value(a, b):
    x = a.to(tl.float32)
    y = b.to(tl.float32)
    product = x * tl.log(y)
    return tl.where(y != y, float('nan'), tl.where(x == 0.0, 0.0, product))


@triton.jit
def _xlogy_flat(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    a = tl.load(A + offset, mask, other=0)
    b = tl.load(B + offset, mask, other=1)
    tl.store(A + offset, _xlogy_value(a, b), mask)


@triton.jit
def _xlogy_strided(A, B, N: tl.constexpr, SHAPE: tl.constexpr,
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
    tl.store(A + oa, _xlogy_value(a, b), mask)


def _stream(device):
    return torch.npu.current_stream(device).npu_stream


def _bind_stream():
    """Resolve the current-stream getter once so the hot path makes a single call."""
    raw = _RAW_STREAM
    if raw is not None:
        try:
            raw(0)
            return raw
        except TypeError:
            def _noarg(device, _raw=raw):
                return _raw()
            return _noarg
        except Exception:
            return _stream
    return _stream


_GET_STREAM = _bind_stream()


def _block_for(n):
    if n >= _BLOCK_MAX:
        return _BLOCK_MAX
    block = 256
    while block < n:
        block <<= 1
    return block


def _compile(A, B, n):
    block = _block_for(n)
    grid = (n + block - 1) // block
    entry = None
    try:
        compiled = _xlogy_flat.warmup(A, B, n, block, grid=(1,))
        launcher = compiled.run
        if not (getattr(launcher, 'compile_only', False) or
                getattr(launcher, 'enable_msprof_register_tensor', False) or
                getattr(launcher.metadata, 'debug_enabled', False) or
                getattr(launcher.metadata, 'debug', False)):
            entry = (launcher.launch, compiled.function, compiled.packed_metadata,
                     grid, block, A.device.index)
    except Exception:
        entry = None
    return entry


def _generic(A, B, n):
    if tuple(torch.broadcast_shapes(A.shape, B.shape)) != tuple(A.shape):
        raise ValueError('broadcast would change the in-place destination shape')
    aligned_shape = (1,) * (A.ndim - B.ndim) + tuple(B.shape)
    aligned_stride = (0,) * (A.ndim - B.ndim) + tuple(B.stride())
    sb = tuple(0 if size == 1 else stride for size, stride in zip(aligned_shape, aligned_stride))
    shape = tuple(A.shape) or (1,)
    sa = tuple(A.stride()) or (0,)
    sb = sb or (0,)
    _xlogy_strided[(triton.cdiv(n, 256),)](A, B, n, shape, sa, sb, 256)


def run(A, B):
    n = A.numel()
    if not n:
        return A
    if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        key = (A.dtype, B.dtype, n)
        try:
            entry = _ENTRIES[key]
        except KeyError:
            entry = _ENTRIES[key] = _compile(A, B, n)
        if entry is not None:
            pa = A.data_ptr()
            pb = B.data_ptr()
            if not ((pa | pb) & 15):
                launch, function, metadata, grid, block, device = entry
                launch(grid, 1, 1, _GET_STREAM(device), function, metadata,
                       None, None, None, pa, pb, n, block)
                return A
        block = _block_for(n)
        _xlogy_flat[((n + block - 1) // block,)](A, B, n, block)
    else:
        _generic(A, B, n)
    return A

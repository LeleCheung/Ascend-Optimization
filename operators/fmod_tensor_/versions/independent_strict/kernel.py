# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""严格Triton原地浮点余数。保持原NPU公式和禁FMA设置，
大输入用真实Vector核数组织完整块流式处理；独立尾块及广播stride路径保留。
缓存新增第二输入dtype；当前stream与输入指针逐次传入。
"""
import torch
import torch_npu
import triton
import triton.language as tl

_BLOCK = 4096
_ENTRIES = {}
_RAW_STREAM = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)


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
def _fmod_stream(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    full = (N // step) * step
    for start in tl.range(pid * BLOCK, full, step, num_stages=1):
        a = tl.load(A + start + lane)
        b = tl.load(B + start + lane)
        tl.store(A + start + lane, _fmod_value(a, b))
    start = full + pid * BLOCK
    if start < N:
        offset = start + lane
        a = tl.load(A + offset, offset < N, other=0)
        b = tl.load(B + offset, offset < N, other=1)
        tl.store(A + offset, _fmod_value(a, b), offset < N)


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


def _raw_stream(device):
    if _RAW_STREAM is not None:
        try:
            return _RAW_STREAM(device)
        except TypeError:
            return _RAW_STREAM()
    return torch.npu.current_stream(device).npu_stream


def _entry(A, B, n, grid):
    """Return (raw_launch, function, packed_metadata, grid) or False."""
    try:
        kernel = _fmod_stream if n >= 131072 else _fmod_flat
        compiled = kernel.warmup(A, B, n, _BLOCK, grid=(1,), enable_fp_fusion=False)
        launcher = compiled.run
        if not (getattr(launcher, 'compile_only', False) or
                getattr(launcher, 'enable_msprof_register_tensor', False) or
                getattr(launcher.metadata, 'debug_enabled', False) or
                getattr(launcher.metadata, 'debug', False)):
            return launcher.launch, compiled.function, compiled.packed_metadata, grid
    except Exception:
        pass
    return False


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
    if n and A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        pa = A.data_ptr()
        pb = B.data_ptr()
        if not ((pa | pb) & 15):
            device = A.get_device()
            key = (A.dtype, B.dtype, device, n)
            entry = _ENTRIES.get(key)
            if entry is None:
                grid = (n + _BLOCK - 1) // _BLOCK
                if n >= 131072:
                    props = triton.runtime.driver.active.utils.get_device_properties(device)
                    cores = max(1, int(props.get('num_vectorcore') or props.get('num_aicore') or 1))
                    grid = min(grid, cores)
                entry = _ENTRIES[key] = _entry(A, B, n, grid)
            if entry:
                entry[0](entry[3], 1, 1, _raw_stream(device),
                         entry[1], entry[2], None, None, None, pa, pb, n, _BLOCK)
                return A
        kernel = _fmod_stream if n >= 131072 else _fmod_flat
        grid = (n + _BLOCK - 1) // _BLOCK
        if n >= 131072:
            props = triton.runtime.driver.active.utils.get_device_properties(A.get_device())
            cores = max(1, int(props.get('num_vectorcore') or props.get('num_aicore') or 1))
            grid = min(grid, cores)
        kernel[(grid,)](A, B, n, _BLOCK, enable_fp_fusion=False)
    elif n:
        _generic(A, B, n)
    return A



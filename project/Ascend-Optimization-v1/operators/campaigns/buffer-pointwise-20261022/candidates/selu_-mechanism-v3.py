# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Codex缓冲机制候选；父版本及编译选项见candidate-provenance.json。
"""SELU原地计算：轻量启动、完整stride、按dtype控制分块；本次输入地址和stream每次读取。"""

def _hgkda_address_type(n):

    class DeviceAddress(int):

        def size(self):
            return (n,)
    return DeviceAddress
import torch
import torch_npu
import triton
import triton.language as tl
_BLOCK_MAX = 32768
_BLOCK_MIN = 1024
_BLOCK_TARGET_PROGRAMS = 8
_BLOCK_STRIDED = 1024

@triton.jit
def _selu_nomask(X, BLOCK: tl.constexpr):
    """numel % BLOCK == 0 (host-guarded), so the tail mask can be dropped."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    if X.dtype.element_ty == tl.float16:
        y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    else:
        xf = x.to(tl.float32)
        y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))

@triton.jit
def _selu_masked(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    if X.dtype.element_ty == tl.float16:
        y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    else:
        xf = x.to(tl.float32)
        y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)

@triton.jit
def _selu_strided(X, N: tl.constexpr, SHAPE: tl.constexpr, STRIDES: tl.constexpr, BLOCK: tl.constexpr):
    """Non-contiguous input: logical index -> storage offset, written in place."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remainder = index
    offset = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = remainder % SHAPE[axis]
        remainder = remainder // SHAPE[axis]
        offset += coord.to(tl.int64) * STRIDES[axis]
    x = tl.load(X + offset, index < N, other=1)
    if X.dtype.element_ty == tl.float16:
        y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    else:
        xf = x.to(tl.float32)
        y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
    tl.store(X + offset, y.to(X.dtype.element_ty), index < N)
_fast_get = {}
_FAST = {}
_fast_get = _FAST.get
_STREAM = {}

def _pick_block(n, dtype):
    maximum = 16384
    if n >= maximum * _BLOCK_TARGET_PROGRAMS:
        return maximum
    target = -(-n // _BLOCK_TARGET_PROGRAMS)
    block = _BLOCK_MIN
    while block < target:
        block <<= 1
    return min(block, maximum)

def _resolve_stream(device):
    """Cheapest faithful current-stream accessor for this device, resolved once."""
    getter = _STREAM.get(device)
    if getter is not None:
        return getter
    raw = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
    if raw is not None:
        try:
            want = int(torch.npu.current_stream().npu_stream)
            try:
                if int(raw(device)) == want:
                    getter = lambda: raw(device)
            except TypeError:
                if int(raw()) == want:
                    getter = raw
        except Exception:
            getter = None
    if getter is None:
        from triton.runtime import driver
        getter = lambda: driver.active.get_current_stream(device)
    _STREAM[device] = getter
    return getter

def _standard_launch(n, block, masked):
    """Ordinary Triton launch: no alignment assumption, safe for any pointer."""
    if masked:

        def launch(X, ptr, n=n, block=block):
            _selu_masked[triton.cdiv(n, block),](X, n, BLOCK=block, enable_select_analysis=False, multibuffer=False)
    else:

        def launch(X, ptr, n=n, block=block):
            _selu_nomask[n // block,](X, BLOCK=block, enable_select_analysis=False, multibuffer=False)
    return launch

def _make_launch(A, n):
    """Build the per-specialization launcher, degrading to a plain Triton launch."""
    block = _pick_block(n, A.dtype)
    masked = bool(n % block)
    if A.data_ptr() & 15:
        return _standard_launch(n, block, masked)
    try:
        if masked:
            compiled = _selu_masked.warmup(A, n, BLOCK=block, grid=(1,), enable_select_analysis=False, multibuffer=False)
            grid = triton.cdiv(n, block)
        else:
            compiled = _selu_nomask.warmup(A, BLOCK=block, grid=(1,), enable_select_analysis=False, multibuffer=False)
            grid = n // block
        launcher = compiled.run
        if getattr(launcher, 'compile_only', False) or getattr(launcher, 'enable_msprof_register_tensor', False) or getattr(launcher.metadata, 'debug_enabled', False):
            return _standard_launch(n, block, masked)
        call = launcher.launch
        function = compiled.function
        metadata = compiled.packed_metadata
        stream = _resolve_stream(A.device.index)
    except Exception:
        return _standard_launch(n, block, masked)
    if masked:

        def launch(X, ptr, grid=grid, call=call, function=function, metadata=metadata, stream=stream, n=n, *, _address=_hgkda_address_type(n)):
            call(grid, 1, 1, stream(), function, metadata, None, None, None, _address(ptr), n, None)
    else:

        def launch(X, ptr, grid=grid, call=call, function=function, metadata=metadata, stream=stream, *, _address=_hgkda_address_type(n)):
            call(grid, 1, 1, stream(), function, metadata, None, None, None, _address(ptr), None)
    return launch

def _dispatch(A, n, ptr):
    if n == 0:
        return A
    if not A.is_contiguous():
        _selu_strided[triton.cdiv(n, _BLOCK_STRIDED),](A, n, tuple(A.shape), tuple(A.stride()), BLOCK=_BLOCK_STRIDED)
        return A
    key = (n, ptr & 15)
    entry = _fast_get(key)
    if entry is not None and A.dtype is entry[0]:
        entry[1](A, ptr)
        return A
    launch = _make_launch(A, n)
    _FAST[key] = (A.dtype, launch)
    launch(A, ptr)
    return A

def run(A):
    n = A.numel()
    ptr = A.data_ptr()
    entry = _fast_get((n, ptr & 15))
    if entry is not None and A.dtype is entry[0] and A.is_contiguous():
        entry[1](A, ptr)
        return A
    return _dispatch(A, n, ptr)

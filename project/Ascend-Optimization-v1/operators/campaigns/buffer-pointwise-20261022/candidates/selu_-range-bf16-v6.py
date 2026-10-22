# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Codex整合：BF16正半轴先FP32缩放并写回BF16；只对非正输入计算半精度指数。
"""SELU原地计算：轻量启动、完整stride、按dtype控制分块；本次输入地址和stream每次读取。

宿主快路径按 numel 建表（不是 (numel, 对齐) 元组），Tensor 方法在模块级绑定，
launcher 闭包只接收本 Tensor 与本次地址；每次调用仍然读取本次地址、dtype、
连续性、以及当前 stream，对齐条件只用于选择 launcher 形态。
"""

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
    elif X.dtype.element_ty == tl.bfloat16:
        positive = (x.to(tl.float32) * 1.0507009873554805).to(tl.bfloat16)
        h = tl.minimum(x.to(tl.float32), 0.0).to(tl.float16)
        negative = (1.0507009873554805 * (1.6732632423543772 * (tl.exp(h) - 1.0))).to(tl.bfloat16)
        y = tl.where(x > 0, positive, negative)
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
    elif X.dtype.element_ty == tl.bfloat16:
        positive = (x.to(tl.float32) * 1.0507009873554805).to(tl.bfloat16)
        h = tl.minimum(x.to(tl.float32), 0.0).to(tl.float16)
        negative = (1.0507009873554805 * (1.6732632423543772 * (tl.exp(h) - 1.0))).to(tl.bfloat16)
        y = tl.where(x > 0, positive, negative)
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
_TAB = {}
_numel = torch.Tensor.numel
_data_ptr = torch.Tensor.data_ptr
_is_contiguous = torch.Tensor.is_contiguous
_TAB_GET = _TAB.get
_STREAM = {}

def _pick_block(n, dtype):
    maximum = 8192 if dtype is torch.float32 else 16384
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
            _selu_masked[triton.cdiv(n, block),](X, n, BLOCK=block)
    else:

        def launch(X, ptr, n=n, block=block):
            _selu_nomask[n // block,](X, BLOCK=block)
    return launch

def _make_launch(A, n):
    """Build (aligned launcher, fallback launcher); both take (tensor, address)."""
    block = _pick_block(n, A.dtype)
    masked = bool(n % block)
    standard = _standard_launch(n, block, masked)
    if _data_ptr(A) & 15:
        return (standard, standard)
    try:
        if masked:
            compiled = _selu_masked.warmup(A, n, BLOCK=block, grid=(1,))
            grid = triton.cdiv(n, block)
        else:
            compiled = _selu_nomask.warmup(A, BLOCK=block, grid=(1,))
            grid = n // block
        launcher = compiled.run
        if getattr(launcher, 'compile_only', False) or getattr(launcher, 'enable_msprof_register_tensor', False) or getattr(launcher.metadata, 'debug_enabled', False):
            return (standard, standard)
        call = launcher.launch
        function = compiled.function
        metadata = compiled.packed_metadata
        stream = _resolve_stream(A.device.index)
    except Exception:
        return (standard, standard)
    if masked:

        def launch(X, ptr, grid=grid, call=call, function=function, metadata=metadata, stream=stream, n=n, *, _address=_hgkda_address_type(n)):
            call(grid, 1, 1, stream(), function, metadata, None, None, None, _address(ptr), n, None)
    else:

        def launch(X, ptr, grid=grid, call=call, function=function, metadata=metadata, stream=stream, *, _address=_hgkda_address_type(n)):
            call(grid, 1, 1, stream(), function, metadata, None, None, None, _address(ptr), None)
    return (launch, standard)

def _dispatch(A):
    n = _numel(A)
    ptr = _data_ptr(A)
    if n == 0:
        return A
    if not _is_contiguous(A):
        _selu_strided[triton.cdiv(n, _BLOCK_STRIDED),](A, n, tuple(A.shape), tuple(A.stride()), BLOCK=_BLOCK_STRIDED)
        return A
    pair = _make_launch(A, n)
    _TAB[n] = (A.dtype, pair[0], pair[1])
    pair[1 if ptr & 15 else 0](A, ptr)
    return A

def run(A):
    ptr = _data_ptr(A)
    entry = _TAB_GET(_numel(A))
    if entry is not None and entry[0] is A.dtype and _is_contiguous(A):
        entry[1 + (ptr & 15 != 0)](A, ptr)
        return A
    return _dispatch(A)

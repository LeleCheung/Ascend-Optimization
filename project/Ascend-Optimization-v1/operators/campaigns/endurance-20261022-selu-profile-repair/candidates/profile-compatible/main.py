# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Host-side fast path: cache the compiled launcher primitives per input address and
# invoke the generated launcher directly with the current bare address and stream.
"""SELU in-place: minimal per-call host work, full stride support, dtype-sized UB blocks.

Per call the fast path reads the current data pointer and the current stream, checks
dtype/numel/contiguity against the cached specialization, then calls the compiled
launcher directly (no Tensor argument marshalling).  Inputs whose pointer is not
16-byte aligned, masked tails, non-contiguous layouts, debug/profiling launches and
any binding failure keep the ordinary Triton entry.
"""
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


_FAST = {}
_fast_get = _FAST.get
_STREAM = {}


def _pick_block(n, dtype):
    maximum = 16384 if dtype is torch.float16 else 8192
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


def _make_entry(A, n):
    """(dtype, n, masked, call, grid, function, metadata, stream_getter) or fallback.

    A fallback entry stores masked=None and the plain launch callable at index 3.
    """
    block = _pick_block(n, A.dtype)
    masked = bool(n % block)
    if A.data_ptr() & 15:
        return (A.dtype, n, None, _standard_launch(n, block, masked))
    try:
        if masked:
            compiled = _selu_masked.warmup(A, n, BLOCK=block, grid=(1,))
            grid = triton.cdiv(n, block)
        else:
            compiled = _selu_nomask.warmup(A, BLOCK=block, grid=(1,))
            grid = n // block
        launcher = compiled.run
        requires_tensor = bool(getattr(launcher, 'compile_only', False)
                or getattr(launcher, 'enable_msprof_register_tensor', False)
                or getattr(launcher.metadata, 'debug_enabled', False)
                or getattr(launcher.metadata, 'debug', False))
        # 采集需要本次Tensor元数据，继续执行已选中的同一kernel，不切换persistent策略。
        call = launcher if requires_tensor else launcher.launch
        function = compiled.function
        metadata = compiled.packed_metadata
        stream = _resolve_stream(A.device.index)
    except Exception:
        return (A.dtype, n, None, _standard_launch(n, block, masked))
    return (A.dtype, n, masked, call, grid, function, metadata, stream, requires_tensor)


def _dispatch(A, ptr):
    n = A.numel()
    if n == 0:
        return A
    if not A.is_contiguous():
        _selu_strided[triton.cdiv(n, _BLOCK_STRIDED),](
            A, n, tuple(A.shape), tuple(A.stride()), BLOCK=_BLOCK_STRIDED)
        return A
    entry = _make_entry(A, n)
    _FAST[ptr] = entry
    masked = entry[2]
    argument = A if len(entry) > 8 and entry[8] else ptr
    if masked is None:
        entry[3](A, ptr)
    elif masked:
        entry[3](entry[4], 1, 1, entry[7](), entry[5], entry[6],
                 None, None, None, argument, n, None)
    else:
        entry[3](entry[4], 1, 1, entry[7](), entry[5], entry[6],
                 None, None, None, argument, None)
    return A


def run(A):
    ptr = A.data_ptr()
    entry = _fast_get(ptr)
    if entry is not None and A.is_contiguous() and A.numel() == entry[1] and A.dtype is entry[0]:
        masked = entry[2]
        argument = A if len(entry) > 8 and entry[8] else ptr
        if masked is None:
            entry[3](A, ptr)
        elif masked:
            entry[3](entry[4], 1, 1, entry[7](), entry[5], entry[6],
                     None, None, None, argument, A.numel(), None)
        else:
            entry[3](entry[4], 1, 1, entry[7](), entry[5], entry[6],
                     None, None, None, argument, None)
        return A
    return _dispatch(A, ptr)

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""In-place negative: Triton computation with the historical validated launcher pattern.

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
_BLOCK_STRIDED = 256

@triton.jit
def _negative_nomask(X, BLOCK: tl.constexpr):
    """numel % BLOCK == 0 (host-guarded), so the tail mask can be dropped."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    if X.dtype.element_ty == tl.float32:
        y = (x.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
    else:
        y = (x.to(tl.uint16, bitcast=True) ^ 0x8000).to(X.dtype.element_ty, bitcast=True)
    tl.store(X + offs, y.to(X.dtype.element_ty))

@triton.jit
def _negative_masked(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    if X.dtype.element_ty == tl.float32:
        y = (x.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
    else:
        y = (x.to(tl.uint16, bitcast=True) ^ 0x8000).to(X.dtype.element_ty, bitcast=True)
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)

@triton.jit
def _negative_streaming(X, N, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, N, step):
        offset = start + lane
        x = tl.load(X + offset, offset < N, other=0)
        if X.dtype.element_ty == tl.float32:
            y = (x.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
        else:
            y = (x.to(tl.uint16, bitcast=True) ^ 0x8000).to(X.dtype.element_ty, bitcast=True)
        tl.store(X + offset, y, offset < N)

@triton.jit
def _negative_strided(X, N: tl.constexpr, SHAPE: tl.constexpr, STRIDES: tl.constexpr, BLOCK: tl.constexpr):
    """Non-contiguous input: logical index -> storage offset, written in place."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remainder = index
    offset = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = remainder % SHAPE[axis]
        remainder = remainder // SHAPE[axis]
        offset += coord.to(tl.int64) * STRIDES[axis]
    x = tl.load(X + offset, index < N, other=1)
    if X.dtype.element_ty == tl.float32:
        y = (x.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
    else:
        y = (x.to(tl.uint16, bitcast=True) ^ 0x8000).to(X.dtype.element_ty, bitcast=True)
    tl.store(X + offset, y.to(X.dtype.element_ty), index < N)
_FAST = {}
_fast_get = _FAST.get
_STREAM = {}

def _pick_block(n, dtype):
    maximum = 8192
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
            _negative_masked[triton.cdiv(n, block),](X, n, BLOCK=block)
    else:

        def launch(X, ptr, n=n, block=block):
            _negative_nomask[n // block,](X, BLOCK=block)
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
        if n >= 1048576:
            compiled = _negative_streaming.warmup(A, n, BLOCK=block, grid=(1,))
            grid = min(40, (n + block - 1) // block)
            masked = True
        elif masked:
            compiled = _negative_masked.warmup(A, n, BLOCK=block, grid=(1,))
            grid = triton.cdiv(n, block)
        else:
            compiled = _negative_nomask.warmup(A, BLOCK=block, grid=(1,))
            grid = n // block
        launcher = compiled.run
        if getattr(launcher, 'compile_only', False) or getattr(launcher, 'enable_msprof_register_tensor', False) or getattr(launcher.metadata, 'debug_enabled', False) or getattr(launcher.metadata, 'debug', False):
            return (A.dtype, n, None, _standard_launch(n, block, masked))
        call = launcher.launch
        function = compiled.function
        metadata = compiled.packed_metadata
        stream = _resolve_stream(A.device.index)
    except Exception:
        return (A.dtype, n, None, _standard_launch(n, block, masked))
    return (A.dtype, n, masked, call, grid, function, metadata, stream)

def _dispatch(A, ptr):
    n = A.numel()
    if n == 0:
        return A
    if not A.is_contiguous():
        _negative_strided[triton.cdiv(n, _BLOCK_STRIDED),](A, n, tuple(A.shape), tuple(A.stride()), BLOCK=_BLOCK_STRIDED)
        return A
    entry = _make_entry(A, n)
    _FAST[ptr] = entry
    masked = entry[2]
    if masked is None:
        entry[3](A, ptr)
    elif masked:
        entry[3](entry[4], 1, 1, entry[7](), entry[5], entry[6], None, None, None, ptr, n, None)
    else:
        entry[3](entry[4], 1, 1, entry[7](), entry[5], entry[6], None, None, None, ptr, None)
    return A

def _run_python(A):
    ptr = A.data_ptr()
    entry = _fast_get(ptr)
    if entry is not None and A.is_contiguous() and (A.numel() == entry[1]) and (A.dtype is entry[0]):
        masked = entry[2]
        if masked is None:
            entry[3](A, ptr)
        elif masked:
            entry[3](entry[4], 1, 1, entry[7](), entry[5], entry[6], None, None, None, ptr, A.numel(), None)
        else:
            entry[3](entry[4], 1, 1, entry[7](), entry[5], entry[6], None, None, None, ptr, None)
        return A
    return _dispatch(A, ptr)

# C++ only handles host metadata and argument assembly; all math remains Triton.
import importlib.util as _import_util
import os as _os
from pathlib import Path as _Path
_project = _Path(_os.environ.get('HGKDA_PROJECT', '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1'))
_loader_spec = _import_util.spec_from_file_location('_hgkda_bridge_loader', _project / 'tools/profiling/hgkda/next_phase/load_host_bridge.py')
_loader = _import_util.module_from_spec(_loader_spec)
_loader_spec.loader.exec_module(_loader)
_bridge_module = _loader.load_host_bridge()

def _bridge_fallback(A):
    output = _run_python(A)
    if A.numel() and A.is_contiguous() and not (A.data_ptr() & 15):
        entry = _FAST.get(A.data_ptr())
        if entry is not None and entry[2] is not None:
            if entry[2]:
                arguments = (entry[4], 1, 1, None, entry[5], entry[6], None, None, None, None, entry[1], None)
            else:
                arguments = (entry[4], 1, 1, None, entry[5], entry[6], None, None, None, None, None)
            _BRIDGE.register_plan(A, entry[3], arguments, entry[7])
    return output

_BRIDGE = _bridge_module.InplaceLaunchCache(_bridge_fallback)

def run(A):
    return _BRIDGE.run(A)

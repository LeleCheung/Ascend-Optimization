# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FlagGems hardswish math with bounded streaming tiles and compiled launches."""
import torch
import torch_npu
import triton
import triton.language as tl

_ENTRIES = {}
_STREAM = {}


@triton.jit
def _hardswish_stream(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, N, step):
        offset = start + lane
        x = tl.load(X + offset, offset < N, other=0).to(tl.float32)
        inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        result = x * (inner * (1.0 / 6.0))
        tl.store(Y + offset, result, offset < N)


@triton.jit
def _hardswish_pair(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * (2 * BLOCK)
    for start in range(tl.program_id(0) * (2 * BLOCK), N, step):
        first = start + lane
        second = first + BLOCK
        a = tl.load(X + first, first < N, other=0).to(tl.float32)
        b = tl.load(X + second, second < N, other=0).to(tl.float32)
        ia = tl.minimum(tl.maximum(a + 3.0, 0.0), 6.0)
        ib = tl.minimum(tl.maximum(b + 3.0, 0.0), 6.0)
        ya = a * (ia * (1.0 / 6.0))
        yb = b * (ib * (1.0 / 6.0))
        tl.store(Y + first, ya, first < N)
        tl.store(Y + second, yb, second < N)


def _stream_getter(device):
    getter = _STREAM.get(device)
    if getter is not None:
        return getter
    raw = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
    if raw is not None:
        try:
            expected = int(torch.npu.current_stream(device).npu_stream)
            try:
                if int(raw(device)) == expected:
                    getter = lambda: raw(device)
            except TypeError:
                if int(raw()) == expected:
                    getter = raw
        except Exception:
            getter = None
    if getter is None:
        from triton.runtime import driver
        getter = lambda: driver.active.get_current_stream(device)
    _STREAM[device] = getter
    return getter


def _entry(x, out, n, block, device):
    entry = None
    kernel = _hardswish_pair if n >= 2621440 else _hardswish_stream
    try:
        compiled = kernel.warmup(x, out, n, block, grid=(1,))
        launcher = compiled.run
        if not (getattr(launcher, 'compile_only', False) or
                getattr(launcher, 'enable_msprof_register_tensor', False) or
                getattr(launcher.metadata, 'debug_enabled', False) or
                getattr(launcher.metadata, 'debug', False)):
            entry = (launcher.launch, compiled.function, compiled.packed_metadata,
                     _stream_getter(device))
    except Exception:
        entry = None
    return entry


def _compute(x, out):
    n = x.numel()
    if not n:
        return out
    block = min(4096, max(32, 1 << (n - 1).bit_length()))
    grid = min(40, ((n + block - 1) // block)) if n >= 131072 else ((n + block - 1) // block)
    kernel = _hardswish_pair if n >= 2621440 else _hardswish_stream
    xp, yp = x.data_ptr(), out.data_ptr()
    if (xp | yp) & 15:
        kernel[(grid,)](x, out, n, block)
        return out
    device = x.device.index
    key = (x.dtype, out.dtype, device, n, block)
    if key not in _ENTRIES:
        _ENTRIES[key] = _entry(x, out, n, block, device)
    entry = _ENTRIES[key]
    if entry is None:
        kernel[(grid,)](x, out, n, block)
    else:
        entry[0](grid, 1, 1, entry[3](), entry[1], entry[2],
                 None, None, None, xp, yp, n, block)
    return out


def run(x):
    contiguous = x.contiguous()
    out = torch.empty(contiguous.shape, dtype=contiguous.dtype, device=contiguous.device)
    return _compute(contiguous, out)

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import torch
import torch_npu
import triton
import triton.language as tl

_ENTRIES = {}
_STREAM = {}
_BLOCK_S = 64


@triton.jit
def _qkv_tile(X, BIAS, Q, K, V,
              BATCH: tl.constexpr, SEQ: tl.constexpr, HEADS: tl.constexpr, D: tl.constexpr,
              XS0: tl.constexpr, XS1: tl.constexpr, XS2: tl.constexpr,
              BIAS_STRIDE: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(1)
    batch = bh // HEADS
    head = bh % HEADS
    rows = tl.program_id(0) * BS + tl.arange(0, BS)
    columns = tl.arange(0, BD)
    channels = head * D + columns
    mask = (rows[:, None] < SEQ) & (columns[None, :] < D)
    input_offset = batch * XS0 + rows[:, None] * XS1 + channels[None, :] * XS2
    output_offset = (bh * SEQ + rows[:, None]) * D + columns[None, :]
    hidden = HEADS * D

    q = tl.load(X + input_offset, mask, other=0).to(tl.float32)
    qb = tl.load(BIAS + channels * BIAS_STRIDE, columns < D, other=0).to(tl.float32)
    q_value = (q + qb[None, :]) * (D ** -0.5)
    tl.store(Q + output_offset, q_value, mask)

    k = tl.load(X + input_offset + hidden * XS2, mask, other=0)
    kb = tl.load(BIAS + (hidden + channels) * BIAS_STRIDE, columns < D, other=0)
    tl.store(K + output_offset, k + kb[None, :], mask)

    v = tl.load(X + input_offset + 2 * hidden * XS2, mask, other=0)
    vb = tl.load(BIAS + (2 * hidden + channels) * BIAS_STRIDE, columns < D, other=0)
    tl.store(V + output_offset, v + vb[None, :], mask)


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


def _compile(X, BIAS, Q, K, V, constants, grid, device):
    entry = None
    try:
        kernel = _qkv_tile.warmup(X, BIAS, Q, K, V, *constants, grid=(1, 1))
        launcher = kernel.run
        if not (getattr(launcher, 'compile_only', False) or
                getattr(launcher, 'enable_msprof_register_tensor', False) or
                getattr(launcher.metadata, 'debug_enabled', False) or
                getattr(launcher.metadata, 'debug', False)):
            entry = (launcher.launch, kernel.function, kernel.packed_metadata,
                     _stream_getter(device))
    except Exception:
        entry = None
    return entry


def run(qkv, qkv_bias, num_heads):
    if qkv.ndim != 3 or qkv_bias.ndim != 1:
        raise ValueError('expected 3D qkv and 1D bias')
    batch, seq, channels = qkv.shape
    if num_heads <= 0 or channels % (3 * num_heads) or qkv_bias.numel() != channels:
        raise ValueError('inconsistent head count or bias size')
    dimension = channels // (3 * num_heads)
    if dimension == 0:
        raise ValueError('head dimension must be positive')
    packed = torch.empty((3, batch, num_heads, seq, dimension), dtype=qkv.dtype, device=qkv.device)
    q, k, v = packed.unbind(0)
    if not batch or not seq:
        return q, k, v
    constants = (batch, seq, num_heads, dimension, *qkv.stride(),
                 qkv_bias.stride(0), _BLOCK_S, 1 << (dimension - 1).bit_length())
    grid = ((seq + _BLOCK_S - 1) // _BLOCK_S, batch * num_heads)
    pointers = (qkv.data_ptr(), qkv_bias.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr())
    # 每次检查所有地址；非对齐输入保持普通JIT的真实对齐特化。
    if any(pointer & 15 for pointer in pointers):
        _qkv_tile[grid](qkv, qkv_bias, q, k, v, *constants)
        return q, k, v
    device = qkv.device.index
    key = (qkv.dtype, qkv_bias.dtype, device, constants)
    if key not in _ENTRIES:
        _ENTRIES[key] = _compile(qkv, qkv_bias, q, k, v, constants, grid, device)
    entry = _ENTRIES[key]
    if entry is None:
        _qkv_tile[grid](qkv, qkv_bias, q, k, v, *constants)
    else:
        entry[0](grid[0], grid[1], 1, entry[3](), entry[1], entry[2],
                 None, None, None, *pointers, *constants)
    return q, k, v

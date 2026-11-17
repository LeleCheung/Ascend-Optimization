# SPDX-License-Identifier: Apache-2.0
"""支持输入/out非连续布局的偶数舍入；使用FP32计算，保留特殊值和零符号。"""
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _round_value(x, DECIMALS: tl.constexpr):
    xf = x.to(tl.float32)
    scale: tl.constexpr = 10.0 ** DECIMALS
    value = xf * scale
    magnitude = tl.abs(value)
    # FP32 addition in [2^23, 2^24] rounds to an integer with ties to even.
    # Above 2^23, FP32 inputs already have no fractional bits.
    shifted = magnitude + 8388608.0
    rounded = shifted - 8388608.0
    rounded = tl.where(magnitude >= 8388608.0, magnitude, rounded)
    sign = value.to(tl.uint32, bitcast=True) & 0x80000000
    signed = ((rounded.to(tl.uint32, bitcast=True) & 0x7fffffff) | sign).to(tl.float32, bitcast=True)
    if DECIMALS != 0:
        signed = tl.div_rn(signed, scale)
    # NaN和原有Inf直接保留；避免它们进入整数奇偶判断后改变语义。
    return tl.where((xf != xf) | (tl.abs(xf) == float('inf')), xf, signed)


@triton.jit
def _round_flat(X, Y, N, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < N, other=0)
    tl.store(Y + index, _round_value(x, DECIMALS).to(Y.dtype.element_ty), index < N)


@triton.jit
def _round_strided(X, Y, N, SHAPE: tl.constexpr, XS: tl.constexpr, YS: tl.constexpr,
                   DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remainder = index
    xi = tl.full((BLOCK,), 0, tl.int64)
    yi = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coordinate = remainder % SHAPE[axis]
        remainder = remainder // SHAPE[axis]
        xi += coordinate.to(tl.int64) * XS[axis]
        yi += coordinate.to(tl.int64) * YS[axis]
    x = tl.load(X + xi, index < N, other=0)
    tl.store(Y + yi, _round_value(x, DECIMALS).to(Y.dtype.element_ty), index < N)


_ENTRIES = {}


def _stream(device):
    raw = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
    if raw is not None:
        try:return raw(device)
        except TypeError:return raw()
    return torch.npu.current_stream(device).npu_stream


def _dispatch(x, out, decimals):
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('当前实现支持float16、bfloat16、float32')
    if out is None:out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError('out必须与输入具有相同形状、dtype和设备')
    n = x.numel()
    if not n:return out
    if not x.is_contiguous() or not out.is_contiguous():
        _round_strided[((n + 255) // 256,)](x, out, n, tuple(x.shape), tuple(x.stride()),
                                          tuple(out.stride()), decimals, BLOCK=256)
        return out
    device = x.device.index
    key = (device, x.dtype, n, decimals)
    entry = _ENTRIES.get(key)
    if entry is None:
        block = min(4096, max(256, 1 << (((n + 7) // 8) - 1).bit_length()))
        grid = (n + block - 1) // block
        launch = None
        try:
            compiled = _round_flat.warmup(x, out, n, decimals, BLOCK=block, grid=(1,))
            launcher = compiled.run
            if not (getattr(launcher, 'compile_only', False) or
                    getattr(launcher, 'enable_msprof_register_tensor', False) or
                    getattr(launcher.metadata, 'debug_enabled', False) or
                    getattr(launcher.metadata, 'debug', False)):
                launch = (launcher.launch, compiled.function, compiled.packed_metadata)
        except Exception:
            pass
        entry = grid, block, launch
        _ENTRIES[key] = entry
    launch = entry[2]
    if launch and not ((x.data_ptr() | out.data_ptr()) & 15):
        launch[0](entry[0], 1, 1, _stream(device), launch[1], launch[2], None, None, None,
                  x.data_ptr(), out.data_ptr(), n, decimals, entry[1])
    else:
        _round_flat[(entry[0],)](x, out, n, decimals, BLOCK=entry[1])
    return out


def run(self, out, *, decimals=0):
    return _dispatch(self, out, decimals)

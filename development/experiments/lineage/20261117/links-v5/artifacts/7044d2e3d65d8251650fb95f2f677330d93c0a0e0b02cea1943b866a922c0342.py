# SPDX-License-Identifier: Apache-2.0
"""Triton 向零截断：保留 IEEE 正负零，直接支持非连续输入。

浮点值先精确扩展到 FP32，再清除小数对应的尾数位；NaN/Inf 不变。
仅缓存已编译代码与启动元数据，每次重新分配输出、读取输入指针和当前流。
"""
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _trunc_bits(x):
    bits = x.to(tl.float32).to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255).to(tl.int32) - 127
    shift = tl.minimum(tl.maximum(23 - exponent, 0), 23)
    whole = bits & (0xffffffff << shift)
    whole = tl.where(exponent < 0, bits & 0x80000000, whole)
    return whole.to(tl.float32, bitcast=True)


@triton.jit
def _fix_contiguous(X, Y, N, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < N, other=0)
    tl.store(Y + index, _trunc_bits(x).to(Y.dtype.element_ty), index < N)


@triton.jit
def _fix_strided(X, Y, N, SHAPE: tl.constexpr, STRIDES: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remainder = index
    offset = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coordinate = remainder % SHAPE[axis]
        remainder = remainder // SHAPE[axis]
        offset += coordinate.to(tl.int64) * STRIDES[axis]
    x = tl.load(X + offset, index < N, other=0)
    tl.store(Y + index, _trunc_bits(x).to(Y.dtype.element_ty), index < N)


_ENTRIES = {}


def _stream(device):
    raw = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
    if raw is not None:
        try:
            return raw(device)
        except TypeError:
            return raw()
    return torch.npu.current_stream(device).npu_stream


def run(self):
    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('此实现支持 float16、bfloat16、float32')
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    n = self.numel()
    if not n:
        return out
    if not self.is_contiguous():
        _fix_strided[(triton.cdiv(n, 256),)](self, out, n, tuple(self.shape), tuple(self.stride()), BLOCK=256)
        return out
    block = min(4096, max(256, triton.next_power_of_2(triton.cdiv(n, 8))))
    grid = triton.cdiv(n, block)
    device = self.device.index
    key = (device, self.dtype, n, block)
    entry = _ENTRIES.get(key)
    if entry is None:
        try:
            compiled = _fix_contiguous.warmup(self, out, n, BLOCK=block, grid=(1,))
            launcher = compiled.run
            if (getattr(launcher, 'compile_only', False) or
                    getattr(launcher, 'enable_msprof_register_tensor', False) or
                    getattr(launcher.metadata, 'debug_enabled', False) or
                    getattr(launcher.metadata, 'debug', False)):
                entry = False
            else:
                entry = launcher.launch, compiled.function, compiled.packed_metadata
        except Exception:
            entry = False
        _ENTRIES[key] = entry
    if entry and not ((self.data_ptr() | out.data_ptr()) & 15):
        entry[0](grid, 1, 1, _stream(device), entry[1], entry[2], None, None, None,
                 self.data_ptr(), out.data_ptr(), n, None)
    else:
        _fix_contiguous[(grid,)](self, out, n, BLOCK=block)
    return out

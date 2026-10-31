# SPDX-License-Identifier: Apache-2.0
"""Triton 向零截断 (torch.fix) —— 保留特殊值语义，支持非连续输入。

数值做法：对 |x| 取 vfloor（正数即 floor，负数再补回原符号位），
因此每元素只需 6 条向量指令（cvt / and / floor / and / or / cvt），
而不是 floor+ceil+select 的 9 条；NaN/Inf 经 floor 原样传递。

性能做法：
* 连续输入且 BLOCK 整除 n 时走无掩码内核（无 mask 计算、整块 DMA）；
* 每个 program 处理更大 tile（上限 8192 元素，避免 UB 溢出）；
* 宿主侧只 warmup 一次，之后直接用已编译产物与启动元数据；
  每次重新分配输出、重新读取数据指针与当前流。
"""
import torch
import torch_npu
import triton
import triton.language as tl

_OK_DTYPE = frozenset((torch.float16, torch.bfloat16, torch.float32))
_empty_like = torch.empty_like
_MAX_BLOCK = 8192

_RAW = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
if _RAW is None:
    def _raw_stream(dev):
        return torch.npu.current_stream(dev).npu_stream
else:
    _raw_stream = _RAW


@triton.jit
def _trunc(v):
    """向零截断，保留 NaN/Inf 与零的符号位。"""
    u = v.to(tl.float32).to(tl.uint32, bitcast=True)
    mag = (u & 0x7FFFFFFF).to(tl.float32, bitcast=True)
    return (tl.floor(mag).to(tl.uint32, bitcast=True) | (u & 0x80000000)).to(tl.float32, bitcast=True)


@triton.jit
def _fix_nomask(X, Y, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index)
    tl.store(Y + index, _trunc(x).to(Y.dtype.element_ty))


@triton.jit
def _fix_masked(X, Y, N, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < N, other=0)
    tl.store(Y + index, _trunc(x).to(Y.dtype.element_ty), index < N)


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
    tl.store(Y + index, _trunc(x).to(Y.dtype.element_ty), index < N)


_ENTRIES = {}


def _build(self, out, n):
    """编译一次并缓存启动元数据；entry = [gridx, jit, launch, func, meta, dev, block, masked]。"""
    low = n & -n
    if low >= 512:
        block = low if low < _MAX_BLOCK else _MAX_BLOCK
        masked = False
        kern = _fix_nomask
    else:
        block = 1024 if n > 1024 else 256
        masked = True
        kern = _fix_masked
    gridx = (n + block - 1) // block
    jit = kern[(gridx,)]
    launch = func = meta = None
    try:
        if masked:
            compiled = kern.warmup(self, out, n, BLOCK=block, grid=(1,))
        else:
            compiled = kern.warmup(self, out, BLOCK=block, grid=(1,))
        runner = compiled.run
        if not (getattr(runner, 'compile_only', False) or
                getattr(runner, 'enable_msprof_register_tensor', False) or
                getattr(runner.metadata, 'debug_enabled', False) or
                getattr(runner.metadata, 'debug', False)):
            launch = runner.launch
            func = compiled.function
            meta = compiled.packed_metadata
    except Exception:
        launch = None
    entry = [gridx, jit, launch, func, meta, self.device.index, block, masked]
    _ENTRIES[(self.dtype, n)] = entry
    return entry


def run(self):
    dtype = self.dtype
    if dtype not in _OK_DTYPE:
        raise TypeError('Supported floating dtypes: float16, bfloat16, float32')
    n = self.numel()
    if not n:
        return _empty_like(self)
    if not self.is_contiguous():
        out = torch.empty(self.shape, dtype=dtype, device=self.device)
        _fix_strided[((n + 255) // 256,)](self, out, n, tuple(self.shape),
                                          tuple(self.stride()), BLOCK=256)
        return out
    out = _empty_like(self)
    entry = _ENTRIES.get((dtype, n))
    if entry is None:
        entry = _build(self, out, n)
    xp = self.data_ptr()
    yp = out.data_ptr()
    launch = entry[2]
    if launch is not None and not ((xp | yp) & 15):
        try:
            launch(entry[0], 1, 1, _raw_stream(entry[5]), entry[3], entry[4],
                   None, None, None, xp, yp, None)
            return out
        except TypeError:
            entry[2] = None
    if entry[7]:
        entry[1](self, out, n, BLOCK=entry[6])
    else:
        entry[1](self, out, BLOCK=entry[6])
    return out

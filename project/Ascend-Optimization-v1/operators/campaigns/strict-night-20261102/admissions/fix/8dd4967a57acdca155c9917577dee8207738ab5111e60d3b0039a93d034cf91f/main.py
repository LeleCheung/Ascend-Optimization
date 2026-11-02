# SPDX-License-Identifier: Apache-2.0
"""Triton 向零截断 (torch.fix) —— 保留特殊值语义，支持非连续输入。

数值做法：对 |x| 取 vfloor（正数即 floor，负数再补回原符号位），
因此每元素只需 6 条向量指令（cvt / and / floor / and / or / cvt），
而不是 floor+ceil+select 的 9 条；NaN/Inf 经 floor 原样传递。

性能做法：
* 小图：一个 program 处理一个 tile，无掩码（BLOCK 整除 n 时）；
* 大图：persistent 多 tile 内核，program 数取设备 vector_core_num
  （运行期查询，按可用 tile 数封顶），每个 program 顺序流过多个 tile，
  循环由编译器流水（num_stages=2），避免上千个 program 的调度与
  单 tile load->compute->store 无法重叠的问题；
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
_LOOP_MIN_N = 8 * _MAX_BLOCK   # >= 8 tiles 才值得走 persistent 路径
_STAGES = 2

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


@triton.jit
def _fix_loop(X, Y, TILES: tl.constexpr, BLOCK: tl.constexpr):
    """persistent：n 是 BLOCK 的整数倍；按实际启动 program 数跨步。"""
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    for tile in tl.range(pid, TILES, step):
        index = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + index)
        tl.store(Y + index, _trunc(x).to(Y.dtype.element_ty))


@triton.jit
def _fix_loop_tail(X, Y, N: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    """persistent + 一个掩码尾块（由 0 号 program 负责）。"""
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    for tile in tl.range(pid, TILES - 1, step):
        index = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + index)
        tl.store(Y + index, _trunc(x).to(Y.dtype.element_ty))
    if pid == 0:
        index = (TILES - 1) * BLOCK + tl.arange(0, BLOCK)
        keep = index < N
        x = tl.load(X + index, keep, other=0)
        tl.store(Y + index, _trunc(x).to(Y.dtype.element_ty), keep)


_ENTRIES = {}
_CORES = {}


def _vector_cores(dev):
    """设备 vector 核数；查询失败时返回 0（改由任务规模推导 program 数）。"""
    try:
        return _CORES[dev]
    except KeyError:
        pass
    cores = 0
    try:
        props = torch.npu.get_device_properties(dev)
        for attr in ('vector_core_num', 'multi_processor_count'):
            value = getattr(props, attr, None)
            if value:
                cores = int(value)
                break
    except Exception:
        cores = 0
    _CORES[dev] = cores
    return cores


def _grid_for(tiles, dev):
    cores = _vector_cores(dev)
    if cores <= 0:
        cores = max(1, tiles // 32)   # 任务规模推导，不硬编码物理核数
    return tiles if tiles < cores else cores


def _usable(compiled):
    runner = compiled.run
    return not (getattr(runner, 'compile_only', False) or
                getattr(runner, 'enable_msprof_register_tensor', False) or
                getattr(runner.metadata, 'debug_enabled', False) or
                getattr(runner.metadata, 'debug', False))


def _build(self, out, n):
    """编译一次并缓存启动元数据。

    entry = [gridx, jit, launch, func, meta, dev, block, mode]
    mode: 0 单 tile 无掩码 / 1 单 tile 掩码 / 2 persistent / 3 persistent+尾块
    """
    dev = self.device.index
    jit = launch = func = meta = None
    masked = False
    if n >= _LOOP_MIN_N:
        block = _MAX_BLOCK
        tiles = n // block
        if n % block:
            tiles += 1
            kern = _fix_loop_tail
            mode = 3
        else:
            kern = _fix_loop
            mode = 2
        gridx = _grid_for(tiles, dev)
        jit = kern[(gridx,)]
        try:
            if mode == 3:
                compiled = kern.warmup(self, out, N=n, TILES=tiles, BLOCK=block,
                                       grid=(gridx,), num_stages=_STAGES)
            else:
                compiled = kern.warmup(self, out, TILES=tiles, BLOCK=block,
                                       grid=(gridx,), num_stages=_STAGES)
            if _usable(compiled):
                launch = compiled.run.launch
                func = compiled.function
                meta = compiled.packed_metadata
        except Exception:
            launch = None
    else:
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
        mode = 1 if masked else 0
        try:
            if masked:
                compiled = kern.warmup(self, out, n, BLOCK=block, grid=(1,))
            else:
                compiled = kern.warmup(self, out, BLOCK=block, grid=(1,))
            if _usable(compiled):
                launch = compiled.run.launch
                func = compiled.function
                meta = compiled.packed_metadata
        except Exception:
            launch = None
    entry = [gridx, jit, launch, func, meta, dev, block, mode, n]
    _ENTRIES[(self.dtype, n, dev)] = entry
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
    key = (dtype, n, self.device.index)
    entry = _ENTRIES.get(key)
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
    mode = entry[7]
    if mode == 2:
        entry[1](self, out, TILES=entry[8] // entry[6], BLOCK=entry[6], num_stages=_STAGES)
    elif mode == 3:
        tiles = (entry[8] + entry[6] - 1) // entry[6]
        entry[1](self, out, N=entry[8], TILES=tiles, BLOCK=entry[6], num_stages=_STAGES)
    elif mode == 1:
        entry[1](self, out, n, BLOCK=entry[6])
    else:
        entry[1](self, out, BLOCK=entry[6])
    return out

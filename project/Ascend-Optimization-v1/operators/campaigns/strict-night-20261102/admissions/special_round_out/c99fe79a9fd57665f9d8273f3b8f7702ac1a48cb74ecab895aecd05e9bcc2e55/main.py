"""严格Triton偶数舍入：小输入直线核、大输入完整块流式循环、真实stride路径。

舍入核：
  * FP32 路径用 2^23 大常数法（half-to-even）+ 大数钳位 + 符号位 OR 恢复，
    共 7 个向量算子（16bit 输入再加一进一出两次转换）。
  * decimals==0 且 dtype 为 16bit 时走原生 16bit 大常数法：不依赖 16bit
    bitcast，也不用 tl.abs(16bit)，幅值与符号都用 where 表达，共 9 个原生算子，
    16bit 向量单元每拍吞吐约为 FP32 的两倍。
  * decimals!=0 一律走 FP32 精确路径（乘 10^k、舍入、div_rn 除回）。
host 侧：raw launcher + 单条 memo 快路径；只用当前 stream getter，不缓存输出。
"""
import torch
import torch_npu
import triton
import triton.language as tl

_PLANS = {}
_FAST = None
_RAW = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)


def _stream(device):
    if _RAW is not None:
        return _RAW(device)
    return triton.runtime.driver.active.get_current_stream(device)


@triton.jit
def _round_fp32(xf):
    """FP32：（|x|+2^23）-2^23 精确 half-to-even；钳位保住 |x|>=2^23；符号位 OR 回填。"""
    magnitude = tl.abs(xf)
    rounded = (magnitude + 8388608.0) - 8388608.0
    rounded = tl.where(magnitude >= 8388608.0, magnitude, rounded)
    return (rounded.to(tl.uint32, bitcast=True)
            | (xf.to(tl.uint32, bitcast=True) & 2147483648)).to(tl.float32, bitcast=True)


@triton.jit
def _value(x, DECIMALS: tl.constexpr):
    if DECIMALS != 0:
        scale: tl.constexpr = 10.0 ** DECIMALS
        return tl.div_rn(_round_fp32(x.to(tl.float32) * scale), scale).to(x.dtype)
    if x.dtype == tl.float32:
        return _round_fp32(x)
    # 原生 16bit 路径：常数取该 dtype 的 2^(有效位数)，同区间内 ulp 恰为 1。
    magic: tl.constexpr = 1024.0 if x.dtype == tl.float16 else 128.0
    neg = x < 0
    magnitude = tl.where(neg, -x, x)
    rounded = (magnitude + magic) - magic
    rounded = tl.where(magnitude >= magic, magnitude, rounded)
    return tl.where(neg, -rounded, rounded)


@triton.jit
def _flat(X, Y, N: tl.constexpr, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < N, other=0)
    tl.store(Y + index, _value(x, DECIMALS), index < N)


@triton.jit
def _streaming(X, Y, N: tl.constexpr, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    full = (N // step) * step
    for start in tl.range(pid * BLOCK, full, step, num_stages=2):
        x = tl.load(X + start + lane)
        tl.store(Y + start + lane, _value(x, DECIMALS))
    start = full + pid * BLOCK
    if start < N:
        index = start + lane
        x = tl.load(X + index, index < N, other=0)
        tl.store(Y + index, _value(x, DECIMALS), index < N)


@triton.jit
def _strided(X, Y, N: tl.constexpr, SHAPE: tl.constexpr, XS: tl.constexpr,
             YS: tl.constexpr, DECIMALS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rem = index
    xi = tl.full((BLOCK,), 0, tl.int64)
    yi = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE)-1, -1, -1):
        coord = rem % SHAPE[axis]
        rem = rem // SHAPE[axis]
        xi += coord.to(tl.int64) * XS[axis]
        yi += coord.to(tl.int64) * YS[axis]
    x = tl.load(X + xi, index < N, other=0)
    tl.store(Y + yi, _value(x, DECIMALS), index < N)


def _build(x, out, n, decimals, device):
    block = min(4096, max(256, triton.next_power_of_2(n)))
    grid = (n + block - 1) // block
    kernel = _flat
    if n >= 131072:
        props = triton.runtime.driver.active.utils.get_device_properties(device)
        cores = max(1, int(props.get('num_vectorcore') or props.get('num_aicore') or 1))
        grid = min(grid, cores)
        kernel = _streaming
    entry = None
    try:
        compiled = kernel.warmup(x, out, n, decimals, block, grid=(1,))
        launcher = compiled.run
        md = launcher.metadata
        if not (getattr(launcher, 'compile_only', False)
                or getattr(launcher, 'enable_msprof_register_tensor', False)
                or getattr(md, 'debug_enabled', False) or getattr(md, 'debug', False)):
            entry = (launcher.launch, compiled.function, compiled.packed_metadata)
    except Exception:
        pass
    return kernel, block, grid, entry


def _slow(x, out, decimals):
    """Build (or look up) the launch plan and run it once."""
    global _FAST
    if out is None:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if out.shape != x.shape or out.dtype != x.dtype or out.get_device() != x.get_device():
        raise ValueError('out必须与输入同形状、同dtype、同设备')
    n = x.numel()
    if not n:
        return out
    if not x.is_contiguous() or not out.is_contiguous():
        _strided[((n + 255)//256,)](x, out, n, tuple(x.shape), tuple(x.stride()),
                                    tuple(out.stride()), decimals, 256)
        return out
    device = x.get_device()
    key = (device, x.dtype, n, decimals)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _PLANS[key] = _build(x, out, n, decimals, device)
    kernel, block, grid, entry = plan
    xp, yp = x.data_ptr(), out.data_ptr()
    use_raw = entry is not None and not ((xp | yp) & 15)
    if use_raw:
        _FAST = (decimals, x.dtype, xp, yp, n, entry[0], grid, device,
                 entry[1], entry[2], block, out.dtype)
        entry[0](grid, 1, 1, _stream(device), entry[1], entry[2], None, None, None,
                 xp, yp, n, decimals, block)
    else:
        _FAST = None
        kernel[grid,](x, out, n, decimals, block)
    return out


def run(self, out, *, decimals=0):
    x = self
    f = _FAST
    if f is not None and out is not None:
        if (x.data_ptr() == f[2] and out.data_ptr() == f[3] and decimals == f[0]
                and x.dtype is f[1] and out.dtype is f[11] and x.numel() == f[4]
                and x.is_contiguous() and out.is_contiguous()):
            f[5](f[6], 1, 1, _stream(f[7]), f[8], f[9], None, None, None,
                 f[2], f[3], f[4], f[0], f[10])
            return out
    return _slow(x, out, decimals)

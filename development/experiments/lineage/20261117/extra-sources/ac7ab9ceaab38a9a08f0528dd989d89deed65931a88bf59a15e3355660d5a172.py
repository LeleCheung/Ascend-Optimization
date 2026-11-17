"""严格Triton偶数舍入：小输入直线核、大输入完整块流式循环、真实stride路径。
FP32大常数舍入并恢复符号位；保留decimals、特殊值和当前stream。
host 快路径：raw launcher + 单条 memo，减少每调用 Python 层次（小输入主导开销）。
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
def _value(x, DECIMALS: tl.constexpr):
    xf = x.to(tl.float32)
    scale: tl.constexpr = 10.0 ** DECIMALS
    value = xf * scale
    magnitude = tl.abs(value)
    rounded = (magnitude + 8388608.0) - 8388608.0
    rounded = tl.where(magnitude >= 8388608.0, magnitude, rounded)
    sign = value.to(tl.uint32, bitcast=True) & 2147483648
    signed = ((rounded.to(tl.uint32, bitcast=True) & 2147483647) | sign).to(tl.float32, bitcast=True)
    if DECIMALS != 0:
        signed = tl.div_rn(signed, scale)
    return tl.where((xf != xf) | (tl.abs(xf) == float('inf')), xf, signed)


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

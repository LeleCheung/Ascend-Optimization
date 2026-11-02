"""严格Triton偶数舍入：小输入直线核、大输入完整块流式循环、真实stride路径。
FP32大常数舍入并恢复符号位；保留decimals、特殊值和当前stream。
KG第二轮的舍入和流式核；Codex补齐输出尺寸调整与设备检查，所有设备计算仍为Triton。
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
    if out.dtype != x.dtype or out.get_device() != x.get_device():
        raise ValueError('out必须与输入同dtype、同设备')
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
    if out is None:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if out.shape != x.shape:
        out.resize_(x.shape)
    if out.dtype != x.dtype or out.get_device() != x.get_device():
        raise ValueError('out必须与输入同dtype、同设备')
    f = _FAST
    if f is not None and out is not None:
        if (x.data_ptr() == f[2] and out.data_ptr() == f[3] and decimals == f[0]
                and x.dtype is f[1] and out.dtype is f[11] and x.numel() == f[4]
                and x.get_device() == f[7]
                and x.is_contiguous() and out.is_contiguous()):
            f[5](f[6], 1, 1, _stream(f[7]), f[8], f[9], None, None, None,
                 f[2], f[3], f[4], f[0], f[10])
            return out
    return _slow(x, out, decimals)


# 宿主桥只处理元数据与编译参数；所有计算及非连续回退仍为上述Triton。
import hashlib as _host_hash
import importlib.util as _host_import
import sys as _host_sys
from pathlib import Path as _HostPath
_host_binary = _HostPath("/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-night-20261102/host-glue/out-v1/_hgkda_host_bridge_out_v1.so")
assert _host_hash.sha256(_host_binary.read_bytes()).hexdigest() == "11e4286a9b31280a70ed93fa3c606de41caddd18e54b4fdada4b6dff51f4349e"
assert torch.__version__ == "2.9.0+cpu" and torch._C._GLIBCXX_USE_CXX11_ABI == True
_bridge_module = _host_sys.modules.get("_hgkda_host_bridge_out_v1")
if _bridge_module is None:
    _host_spec = _host_import.spec_from_file_location("_hgkda_host_bridge_out_v1", _host_binary)
    _bridge_module = _host_import.module_from_spec(_host_spec)
    _host_spec.loader.exec_module(_bridge_module)
    _host_sys.modules["_hgkda_host_bridge_out_v1"] = _bridge_module
else:
    assert _HostPath(_bridge_module.__file__).resolve() == _host_binary.resolve()
_python_triton_run = run


def _bridge_fallback(x, out):
    result = _python_triton_run(x, out, decimals=0)
    n = x.numel()
    if (n and x.is_contiguous() and out.is_contiguous()
            and x.shape == out.shape and x.device == out.device
            and x.dtype == out.dtype and not ((x.data_ptr() | out.data_ptr()) & 15)):
        device = x.get_device()
        plan = _PLANS.get((device, x.dtype, n, 0))
        if plan is not None:
            kernel, block, grid, entry = plan
            if entry is not None:
                arguments = (grid, 1, 1, None, entry[1], entry[2], None, None, None,
                             None, None, n, 0, block)
                _BRIDGE.register_plan(x, out, entry[0], arguments, lambda: _stream(device))
    return result


_BRIDGE = _bridge_module.OutLaunchCache(_bridge_fallback)


def run(self, out, *, decimals=0):
    if decimals != 0 or out is None:
        return _python_triton_run(self, out, decimals=decimals)
    return _BRIDGE.run(self, out)

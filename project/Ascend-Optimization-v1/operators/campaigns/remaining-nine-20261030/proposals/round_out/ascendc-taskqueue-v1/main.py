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


_STREAM = {}


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


def _dispatch(x, out, decimals):
    dtype = x.dtype
    device_object = x.device
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('当前实现支持float16、bfloat16、float32')
    if out is None:out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if out.shape != x.shape or out.dtype != dtype or out.device != device_object:
        raise ValueError('out必须与输入具有相同形状、dtype和设备')
    n = x.numel()
    if not n:return out
    if not x.is_contiguous() or not out.is_contiguous():
        _round_strided[((n + 255) // 256,)](x, out, n, tuple(x.shape), tuple(x.stride()),
                                          tuple(out.stride()), decimals, BLOCK=256)
        return out
    device = device_object.index
    key = (device, dtype, n, decimals)
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
        entry = grid, block, launch, (_stream_getter(device) if launch else None)
        _ENTRIES[key] = entry
    launch = entry[2]
    xp, yp = x.data_ptr(), out.data_ptr()
    if launch and not ((xp | yp) & 15):
        launch[0](entry[0], 1, 1, entry[3](), launch[1], launch[2], None, None, None,
                  xp, yp, n, decimals, entry[1])
    else:
        _round_flat[(entry[0],)](x, out, n, decimals, BLOCK=entry[1])
    return out


def run(input, *, decimals=0, out=None):
    return _dispatch(input, out, decimals)


_original_run=run

import hashlib as _hashlib
import importlib.util as _importlib
from pathlib import Path as _Path
_host_path=_Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/remaining-nine-20261030/proposals/round_out/ascendc-taskqueue-v1/_hgkda_remaining_host_v1.so')
_device_path=_Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/remaining-nine-20261030/proposals/round_out/ascendc-taskqueue-v1/libhgkda_remaining.so')
if _hashlib.sha256(_host_path.read_bytes()).hexdigest()!='6d54d6f9cc7136ab4f933c11186cadc5d6f050ca1adfb9c73295441cca07970f':
    raise RuntimeError('宿主二进制身份不符')
if _hashlib.sha256(_device_path.read_bytes()).hexdigest()!='610ed2ab21e28a5ed578796d8f2aa8be8bd12b4c012e13a3e0bd94fced9ea47a':
    raise RuntimeError('计算二进制身份不符')
if torch.__version__!='2.9.0+cpu' or torch._C._GLIBCXX_USE_CXX11_ABI!=True:
    raise RuntimeError('Torch版本或ABI不符')
_spec=_importlib.spec_from_file_location('_hgkda_remaining_host_v1',_host_path)
_host=_importlib.module_from_spec(_spec);_spec.loader.exec_module(_host)
_host.configure_library(str(_device_path))

def _native_round_fallback(x, out=None):
    return _original_run(x,decimals=0,out=out)


def run(input, *, decimals=0, out=None):
    if decimals!=0:
        return _original_run(input,decimals=decimals,out=out)
    return _host.execute(input,out,3,_native_round_fallback)

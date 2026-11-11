"""HG-KDA log10: retained CC math, bounded-grid large-input streaming, dynamic addresses and streams."""

import torch
import triton
import triton.language as tl

try:  # cheapest faithful current-stream query on this target
    import torch_npu

    _RAW_STREAM = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
except Exception:  # pragma: no cover
    torch_npu = None
    _RAW_STREAM = None

_TILE_MAX = 32768          # largest tile this backend compiles for both dtypes
_TILE_MIN = 1024
_HOIST_MIN = 8192          # 4-byte data: 2-way hoist once a tile has 2 halves
_BLOCK_STRIDED = 512       # gather/scatter tiles materialise address vectors
_MAX_DIMS = 8
_LOG10E = 0.4342944819032518

_ENTRIES = {}
_STREAM_GETTERS = {}


@triton.jit
def _log10_flat(x_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = tl.log(x) * 0.4342944819032518
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty))


@triton.jit
def _log10_flat_masked(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    y = tl.log(x) * 0.4342944819032518
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _log10_pair(x_ptr, SUB: tl.constexpr):
    """Two independent loads issued before either store: more memory in flight."""
    off0 = tl.program_id(0) * (2 * SUB) + tl.arange(0, SUB)
    off1 = off0 + SUB
    a = tl.load(x_ptr + off0).to(tl.float32)
    b = tl.load(x_ptr + off1).to(tl.float32)
    ya = tl.log(a) * 0.4342944819032518
    yb = tl.log(b) * 0.4342944819032518
    tl.store(x_ptr + off0, ya.to(x_ptr.dtype.element_ty))
    tl.store(x_ptr + off1, yb.to(x_ptr.dtype.element_ty))


@triton.jit
def _log10_scaled(x_ptr, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = offs * SCALE
    x = tl.load(x_ptr + idx, mask=mask, other=1.0).to(tl.float32)
    y = tl.log(x) * 0.4342944819032518
    tl.store(x_ptr + idx, y.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _log10_nd(x_ptr, n,
              s0, s1, s2, s3, s4, s5, s6, s7,
              t0, t1, t2, t3, t4, t5, t6, t7,
              NDIM: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    shapes = (s0, s1, s2, s3, s4, s5, s6, s7)
    strides = (t0, t1, t2, t3, t4, t5, t6, t7)
    rem = offs
    off = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in tl.static_range(8):
        if i < NDIM:
            size = shapes[NDIM - 1 - i]
            stride = strides[NDIM - 1 - i]
            off += (rem % size) * stride
            rem = rem // size
    x = tl.load(x_ptr + off, mask=mask, other=1.0).to(tl.float32)
    y = tl.log(x) * 0.4342944819032518
    tl.store(x_ptr + off, y.to(x_ptr.dtype.element_ty), mask=mask)


def _pick_tile(n):
    """Elements per program: cover the input with as few tiles as UB allows."""
    if n >= _TILE_MAX:
        return _TILE_MAX
    tile = _TILE_MIN
    while tile < n:
        tile <<= 1
    return tile


def _resolve_stream(device):
    """Cheapest faithful current-stream accessor for this device, resolved once.

    The getter itself is re-evaluated on every call, so a caller that switches
    streams still launches on the stream that is current now.
    """
    getter = _STREAM_GETTERS.get(device)
    if getter is not None:
        return getter
    getter = None
    if _RAW_STREAM is not None:
        try:
            want = int(torch.npu.current_stream().npu_stream)
            if int(_RAW_STREAM(device)) == want:
                getter = lambda: _RAW_STREAM(device)
        except Exception:
            getter = None
    if getter is None:
        try:
            from triton.runtime import driver as _driver

            def getter(device=device):
                return _driver.active.get_current_stream(device)
        except Exception:  # pragma: no cover - last resort
            def getter():
                return torch.npu.current_stream().npu_stream
    _STREAM_GETTERS[device] = getter
    return getter



@triton.jit
def _log10_streaming(x_ptr, N: tl.constexpr, SUB: tl.constexpr):
    lane = tl.arange(0, SUB)
    step = tl.num_programs(0) * 2 * SUB
    for start in range(tl.program_id(0) * 2 * SUB, N, step):
        off0 = start + lane
        off1 = off0 + SUB
        if N % (2 * SUB) == 0:
            a = tl.load(x_ptr + off0).to(tl.float32)
            b = tl.load(x_ptr + off1).to(tl.float32)
        else:
            a = tl.load(x_ptr + off0, off0 < N, other=1.0).to(tl.float32)
            b = tl.load(x_ptr + off1, off1 < N, other=1.0).to(tl.float32)
        ya = tl.log(a) * 0.4342944819032518
        yb = tl.log(b) * 0.4342944819032518
        if N % (2 * SUB) == 0:
            tl.store(x_ptr + off0, ya.to(x_ptr.dtype.element_ty))
            tl.store(x_ptr + off1, yb.to(x_ptr.dtype.element_ty))
        else:
            tl.store(x_ptr + off0, ya.to(x_ptr.dtype.element_ty), off0 < N)
            tl.store(x_ptr + off1, yb.to(x_ptr.dtype.element_ty), off1 < N)


def _make_streaming_entry(A, n, key):
    sub = 8192
    grid = min(40, triton.cdiv(n, 2 * sub))
    tail = (n, sub)
    def fallback(A):
        _log10_streaming[(grid,)](A, N=n, SUB=sub)
    entry = (None, None, None, grid, tail, None, fallback)
    try:
        ck = _log10_streaming.warmup(A, N=n, SUB=sub, grid=(1,))
        launcher = ck.run
        if not (getattr(launcher, 'compile_only', False) or
                getattr(launcher, 'enable_msprof_register_tensor', False) or
                getattr(launcher.metadata, 'debug_enabled', False) or
                getattr(launcher.metadata, 'debug', False)):
            entry = (launcher.launch, ck.function, ck.packed_metadata,
                     grid, tail, _resolve_stream(key[2]), fallback)
    except Exception:
        pass
    _ENTRIES[key] = entry
    return entry

def _build_entry(A, n, key):
    if n >= 8 * 40 * _TILE_MAX:
        return _make_streaming_entry(A, n, key)
    tile = _pick_tile(n)
    esize = A.element_size()
    masked = bool(n % tile)
    if masked or esize < 4 or tile < _HOIST_MIN:
        jitfn = _log10_flat_masked if masked else _log10_flat
        grid = (triton.cdiv(n, tile),)
        # Tail of the compiled launcher's positional argument list: every kernel
        # parameter after the data pointer, constexprs included (the Ascend
        # launcher builds one PyArg format unit per parameter).
        tail = (n, tile) if masked else (tile,)
        if masked:
            def fallback(A, _f=jitfn, _g=grid, _n=n, _b=tile):
                _f[_g](A, _n, BLOCK=_b)
            warm = lambda: jitfn.warmup(A, n, BLOCK=tile, grid=(1,))
        else:
            def fallback(A, _f=jitfn, _g=grid, _b=tile):
                _f[_g](A, BLOCK=_b)
            warm = lambda: jitfn.warmup(A, BLOCK=tile, grid=(1,))
    else:
        sub = tile >> 1
        jitfn = _log10_pair
        grid = (n // tile,)
        tail = (sub,)

        def fallback(A, _f=jitfn, _g=grid, _b=sub):
            _f[_g](A, SUB=_b)
        warm = lambda: jitfn.warmup(A, SUB=sub, grid=(1,))

    entry = (None, None, None, grid[0], tail, None, fallback)
    try:
        ck = warm()
        launcher = ck.run
        if getattr(launcher, "compile_only", False) or getattr(
                launcher, "enable_msprof_register_tensor", False):
            pass
        else:
            md = launcher.metadata
            if not (getattr(md, "debug_enabled", False) or getattr(md, "debug", False)):
                entry = (launcher.launch, ck.function, ck.packed_metadata,
                         grid[0], tail, _resolve_stream(key[2]), fallback)
    except Exception:
        entry = (None, None, None, grid[0], tail, None, fallback)
    _ENTRIES[key] = entry
    return entry


def _launch_strided(A, n):
    """Non-contiguous in-place path: real element strides, view object untouched."""
    shape = A.shape
    stride = A.stride()
    dims = []  # (size, stride) inner -> outer, adjacent dims merged when compact
    for d in range(len(shape) - 1, -1, -1):
        s = shape[d]
        if s == 1:
            continue
        st = stride[d]
        if dims and st == dims[-1][0] * dims[-1][1]:
            dims[-1] = (dims[-1][0] * s, dims[-1][1])
        else:
            dims.append((s, st))
    dims.reverse()
    if len(dims) <= 1:
        scale = dims[0][1] if dims else 1
        _log10_scaled[((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)](
            A, n, SCALE=scale, BLOCK=_BLOCK_STRIDED)
        return
    if len(dims) > _MAX_DIMS:
        # Extremely rare (>8 non-mergeable axes): peel the outermost axis.
        for i in range(shape[0]):
            sub = A.select(0, i)
            m = sub.numel()
            if m:
                _launch_strided(sub, m)
        return
    sizes = [d[0] for d in dims] + [1] * (_MAX_DIMS - len(dims))
    sts = [d[1] for d in dims] + [0] * (_MAX_DIMS - len(dims))
    _log10_nd[((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)](
        A, n, *sizes, *sts, NDIM=len(dims), BLOCK=_BLOCK_STRIDED)


def _triton_run(A):
    n = A.numel()
    if n == 0:
        return A
    if not A.is_contiguous():
        _launch_strided(A, n)
        return A
    pointer = A.data_ptr()
    if pointer & 15:
        _launch_strided(A, n)
        return A
    key = (A.dtype, n, A.get_device())
    entry = _ENTRIES.get(key)
    if entry is None:
        entry = _build_entry(A, n, key)
    launch = entry[0]
    if launch is not None:
        launch(entry[3], 1, 1, entry[5](), entry[1], entry[2],
               None, None, None, pointer, *entry[4])
    else:
        entry[6](A)
    return A


# 计算来自本项目Ascend C源码；C++只负责元数据、任务队列及启动。
import hashlib as _hashlib
import importlib.util as _importlib
from pathlib import Path as _Path

_HOST_PATH=_Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/threeway-20261030/proposals/log10_/ascendc-taskqueue-v2/_hgkda_ascendc_host_v2.so')
_KERNEL_PATH=_Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/threeway-20261030/proposals/log10_/ascendc-taskqueue-v2/libhgkda_log10.so')
if _hashlib.sha256(_HOST_PATH.read_bytes()).hexdigest()!='6c4d698eea28474afa9d5f11332cc7f8a10f7e5b9cd409d02ff26b288af0e581':
    raise RuntimeError('Ascend C宿主二进制身份不符')
if _hashlib.sha256(_KERNEL_PATH.read_bytes()).hexdigest()!='2c024ed0f22cab65a1b41feddd86c4e41e675bafcf4720c2b2ec2aed1ed2417a':
    raise RuntimeError('Ascend C计算二进制身份不符')
if torch.__version__!='2.9.0+cpu' or torch._C._GLIBCXX_USE_CXX11_ABI!=True:
    raise RuntimeError('宿主封装的Torch版本或ABI不符')
_spec=_importlib.spec_from_file_location('_hgkda_ascendc_host_v2',_HOST_PATH)
_host=_importlib.module_from_spec(_spec);_spec.loader.exec_module(_host)
_host.configure_library(str(_KERNEL_PATH))


def run(A):
    return _host.run_log10(A,_triton_run)

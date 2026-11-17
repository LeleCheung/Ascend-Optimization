"""严格Triton原地log2：小输入复用直线核，大输入按实际Vector核数流式循环。
完整块不做逐元素mask，独立尾块保护；保持FP32 log2及真实stride分支。
编译缓存与当前stream处理沿用原严格Triton版本，不调用CANN或PyTorch数值操作。
"""

import torch
import triton
import triton.language as tl

_BLOCK_MIN = 1024
_BLOCK_CAP_1B = 16384      # 2-byte dtypes: 32 KiB source tile
_BLOCK_CAP_4B = 8192       # 4-byte dtypes: 32 KiB source tile
_TARGET_PROGS = 32
_BLOCK_STRIDED = 512       # gather/scatter tiles materialise address vectors

_ENTRIES = {}
_STREAM_GETTERS = {}


@triton.jit
def _log2_flat(x_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.log2(x.to(tl.float32))
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty))


@triton.jit
def _log2_flat_masked(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    y = tl.log2(x.to(tl.float32))
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _log2_stream(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    full = (n // step) * step
    for start in tl.range(pid * BLOCK, full, step, num_stages=2):
        x = tl.load(x_ptr + start + lane)
        y = tl.log2(x.to(tl.float32))
        tl.store(x_ptr + start + lane, y.to(x_ptr.dtype.element_ty))
    start = full + pid * BLOCK
    if start < n:
        offs = start + lane
        x = tl.load(x_ptr + offs, mask=offs < n, other=1.0)
        y = tl.log2(x.to(tl.float32))
        tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=offs < n)


@triton.jit
def _log2_scaled(x_ptr, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = offs * SCALE
    x = tl.load(x_ptr + idx, mask=mask, other=1.0)
    y = tl.log2(x.to(tl.float32))
    tl.store(x_ptr + idx, y.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _log2_nd(x_ptr, n,
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
    x = tl.load(x_ptr + off, mask=mask, other=1.0)
    y = tl.log2(x.to(tl.float32))
    tl.store(x_ptr + off, y.to(x_ptr.dtype.element_ty), mask=mask)


def _pick_block(n, esize):
    """Elements per program: aim for ~32 programs, capped at the staging tile."""
    cap = _BLOCK_CAP_1B if esize < 4 else _BLOCK_CAP_4B
    target = -(-n // _TARGET_PROGS)
    block = _BLOCK_MIN
    while block < target:
        block <<= 1
    if block > cap:
        block = cap
    return block


def _resolve_stream(device):
    """Cheapest faithful current-stream accessor for this device, resolved once.

    The getter itself is re-evaluated on every call, so a caller that switches
    streams still launches on the stream that is current now.
    """
    getter = _STREAM_GETTERS.get(device)
    if getter is not None:
        return getter
    getter = None
    try:
        import torch_npu  # noqa: F401

        raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
        if raw is not None:
            want = int(torch.npu.current_stream().npu_stream)
            if int(raw(device)) == want:
                getter = lambda: raw(device)
    except Exception:
        getter = None
    if getter is None:
        try:
            from triton.runtime import driver as _driver

            def getter(device=device):
                return _driver.active.get_current_stream(device)
        except Exception:
            def getter():  # pragma: no cover - last resort
                return torch.npu.current_stream().npu_stream
    _STREAM_GETTERS[device] = getter
    return getter


def _build_entry(A, n, key):
    block = _pick_block(n, A.element_size())
    streaming = n >= 131072
    if streaming:
        block = min(block, 4096)
    grid = (triton.cdiv(n, block),)
    if streaming:
        props = triton.runtime.driver.active.utils.get_device_properties(key[2])
        cores = max(1, int(props.get('num_vectorcore') or props.get('num_aicore') or 1))
        grid = (min(grid[0], cores),)
    masked = bool(n % block) or streaming
    jitfn = _log2_stream if streaming else (_log2_flat_masked if masked else _log2_flat)
    # Tail of the compiled launcher's positional argument list: every kernel
    # parameter after the data pointer, constexprs included (the Ascend launcher
    # builds one PyArg format unit per parameter, `constexpr` included).
    tail = (n, block) if masked else (block,)
    if masked:
        def fallback(A, _f=jitfn, _g=grid, _n=n, _b=block):
            _f[_g](A, _n, BLOCK=_b)
    else:
        def fallback(A, _f=jitfn, _g=grid, _b=block):
            _f[_g](A, BLOCK=_b)
    entry = (None, None, None, grid[0], tail, None, fallback)
    try:
        if masked:
            ck = jitfn.warmup(A, n, BLOCK=block, grid=(1,))
        else:
            ck = jitfn.warmup(A, BLOCK=block, grid=(1,))
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
        _log2_scaled[((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)](
            A, n, SCALE=scale, BLOCK=_BLOCK_STRIDED)
        return
    if len(dims) > 8:
        # Extremely rare (>8 non-mergeable axes): peel the outer axis on the host.
        for i in range(shape[0]):
            sub = A.select(0, i)
            if sub.numel():
                _launch_strided(sub, sub.numel())
        return
    sizes = [d[0] for d in dims] + [1] * (8 - len(dims))
    sts = [d[1] for d in dims] + [0] * (8 - len(dims))
    _log2_nd[((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)](
        A, n, *sizes, *sts, NDIM=len(dims), BLOCK=_BLOCK_STRIDED)


def run(A):
    n = A.numel()
    if n == 0:
        return A
    if not A.is_contiguous():
        _launch_strided(A, n)
        return A
    key = (A.dtype, n, A.get_device())
    entry = _ENTRIES.get(key)
    if entry is None:
        entry = _build_entry(A, n, key)
    launch = entry[0]
    if launch is not None:
        launch(entry[3], 1, 1, entry[5](), entry[1], entry[2],
               None, None, None, A.data_ptr(), *entry[4])
    else:
        entry[6](A)
    return A



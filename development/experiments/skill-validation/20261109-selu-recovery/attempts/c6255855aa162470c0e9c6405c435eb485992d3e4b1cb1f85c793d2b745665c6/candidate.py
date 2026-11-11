# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

# Kernel 常量必须是 tl.constexpr，Ascend 代码生成器不接受纯 Python 全局常量。
_SCALE = tl.constexpr(1.0507009873554805)
# 把 alpha * scale 折叠，负半轴只需一次乘法。
_ALPHA_SCALE = tl.constexpr(1.7580993408473766)

_BLOCK_F16 = 8192
_BLOCK_F32 = 4096
_BLOCK_STRIDED = 2048
_MAX_RANK = 8

# 16 字节对齐是向量化搬运的前提；不满足时退回普通 Triton 调度。
_ALIGN_BYTES = 16

# 编译缓存：键 -> (launcher, compiled, function, packed_metadata, grid, masked, n)。
# 只保存编译句柄与启动元数据，绝不保存张量、结果或旧 stream。
_LAUNCH_CACHE = {}


@triton.jit
def _selu_flat_nomask(x_ptr, BLOCK: tl.constexpr, F16MATH: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    if F16MATH:
        y = tl.where(x > 0, x * _SCALE, (tl.exp(x) - 1.0) * _ALPHA_SCALE)
    else:
        xf = x.to(tl.float32)
        yf = tl.where(xf > 0, xf * _SCALE, (tl.exp(xf) - 1.0) * _ALPHA_SCALE)
        y = yf.to(x.dtype)
    tl.store(x_ptr + offs, y.to(x.dtype))


@triton.jit
def _selu_flat_masked(x_ptr, n_elements, BLOCK: tl.constexpr, F16MATH: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if F16MATH:
        y = tl.where(x > 0, x * _SCALE, (tl.exp(x) - 1.0) * _ALPHA_SCALE)
    else:
        xf = x.to(tl.float32)
        yf = tl.where(xf > 0, xf * _SCALE, (tl.exp(xf) - 1.0) * _ALPHA_SCALE)
        y = yf.to(x.dtype)
    tl.store(x_ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _selu_strided(
    x_ptr,
    n_elements,
    base,
    BLOCK: tl.constexpr,
    s0: tl.constexpr, s1: tl.constexpr, s2: tl.constexpr, s3: tl.constexpr,
    s4: tl.constexpr, s5: tl.constexpr, s6: tl.constexpr, s7: tl.constexpr,
    t0: tl.constexpr, t1: tl.constexpr, t2: tl.constexpr, t3: tl.constexpr,
    t4: tl.constexpr, t5: tl.constexpr, t6: tl.constexpr, t7: tl.constexpr,
    RANK: tl.constexpr,
    IDX64: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    if IDX64:
        rem = idx.to(tl.int64)
        off = tl.zeros([BLOCK], dtype=tl.int64)
    else:
        rem = idx
        off = tl.zeros([BLOCK], dtype=tl.int32)
    o = rem % s0
    rem = rem // s0
    off += o * t0
    if RANK > 1:
        o = rem % s1
        rem = rem // s1
        off += o * t1
    if RANK > 2:
        o = rem % s2
        rem = rem // s2
        off += o * t2
    if RANK > 3:
        o = rem % s3
        rem = rem // s3
        off += o * t3
    if RANK > 4:
        o = rem % s4
        rem = rem // s4
        off += o * t4
    if RANK > 5:
        o = rem % s5
        rem = rem // s5
        off += o * t5
    if RANK > 6:
        o = rem % s6
        rem = rem // s6
        off += o * t6
    if RANK > 7:
        o = rem % s7
        off += o * t7
    ptr = x_ptr + (off + base)
    x = tl.load(ptr, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    yf = tl.where(xf > 0, xf * _SCALE, (tl.exp(xf) - 1.0) * _ALPHA_SCALE)
    tl.store(ptr, yf.to(x.dtype), mask=mask)


def _collapse(shape, stride):
    """去掉 size-1 维，并合并已经连续的相邻维。"""
    dims = [(int(s), int(t)) for s, t in zip(shape, stride) if int(s) != 1]
    out = []
    for s, t in dims:
        if out and out[-1][1] == t * s:
            ps, pt = out.pop()
            out.append((ps * s, t))
        else:
            out.append((s, t))
    return out


def _current_device_index():
    """读取当前设备下标；失败时返回 None，由调用方保守进入设备上下文。"""
    try:
        return int(torch_npu.npu.current_device())
    except Exception:
        return None


def _debug_forced_fallback(launcher, compiled):
    """调试/采集模式为真时必须走普通 Triton，避免绕开登记与调试路径。"""
    if getattr(launcher, "compile_only", False):
        return True
    if getattr(compiled, "compile_only", False):
        return True
    if getattr(launcher, "enable_msprof_register_tensor", False):
        return True
    if getattr(compiled, "enable_msprof_register_tensor", False):
        return True
    if getattr(getattr(compiled, "metadata", None), "debug_enabled", False):
        return True
    if getattr(getattr(launcher, "metadata", None), "debug_enabled", False):
        return True
    return False


def _do_launch(launcher, grid, stream, fn, packed, ptr, ne, masked):
    """按后端 ABI 直接发射一次，参数个数固定，不做任何运行时探测。

    后端 make_launcher 的格式串为 "iiiKKOOOO" + 逐签名参数的 format_of + debug_abi：
    前 9 个槽位是常规前缀（grid/1/1/stream/function/packed/metadata 占位），
    之后严格按内核签名顺序传显式参数——先是当前指针，masked 变体再传元素数 n，
    最后是本内核两个显式 constexpr 的占位（BLOCK、F16MATH）。constexpr 在后端
    同样各保留一个 O 槽位，所以这两个尾部 None 是参数占位，不是隐藏参数，
    不能省略，也不能在末尾继续追加。
    """
    if masked:
        # 13 个参数：前缀 9 + 指针 + n + BLOCK 占位 + F16MATH 占位。
        launcher.launch(
            grid, 1, 1, stream, fn, packed, None, None, None, ptr, ne, None, None
        )
    else:
        # 12 个参数：前缀 9 + 指针 + BLOCK 占位 + F16MATH 占位。
        launcher.launch(
            grid, 1, 1, stream, fn, packed, None, None, None, ptr, None, None
        )


def _compile_flat(x, n_elements, block, f16_math, masked, dev):
    """首次进入该键时编译；warmup 只编译，不执行原地计算。"""
    kernel = _selu_flat_masked if masked else _selu_flat_nomask
    # 该 grid 只缓存给直接 C 启动器使用：后端 "iiiKKOOOO" 前缀前三项均为 int，
    # 不能传 tuple，故这里传标量；普通 Triton 路径仍在下方用 lambda 生成 tuple。
    grid = triton.cdiv(n_elements, block)

    def _do_compile():
        if masked:
            compiled = kernel.warmup(
                x, n_elements, BLOCK=block, F16MATH=f16_math, grid=(1,)
            )
        else:
            compiled = kernel.warmup(x, BLOCK=block, F16MATH=f16_math, grid=(1,))
        launcher = compiled.run
        return (
            launcher,
            compiled,
            compiled.function,
            compiled.packed_metadata,
            grid,
            masked,
            n_elements,
        )

    # 设备上下文只在首次编译或切设备时进入。
    idx = dev.index
    if idx is None or _current_device_index() != idx:
        with torch_npu.npu.device(dev):
            return _do_compile()
    return _do_compile()


def _launch_plain_flat(x, n_elements, block, f16_math, masked):
    """经验证的普通 Triton 回退：未对齐、编译失败或调试模式。"""
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    with torch_npu.npu.device(x.device):
        if masked:
            _selu_flat_masked[grid](x, n_elements, BLOCK=block, F16MATH=f16_math)
        else:
            _selu_flat_nomask[grid](x, BLOCK=block, F16MATH=f16_math)


def _launch_cached_flat(x, n_elements, block, f16_math, masked):
    """连续对齐快路径：直接调用已编译启动器；返回 False 表示需回退。"""
    dev = x.device
    ptr = int(x.data_ptr())
    if ptr % _ALIGN_BYTES != 0:
        return False
    key = (dev.index, x.dtype, tuple(x.shape), block, masked)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        try:
            entry = _compile_flat(x, n_elements, block, f16_math, masked, dev)
        except Exception:
            return False
        launcher = entry[0]
        compiled = entry[1]
        if _debug_forced_fallback(launcher, compiled):
            _LAUNCH_CACHE.pop(key, None)
            return False
        _LAUNCH_CACHE[key] = entry
    launcher, compiled, fn, packed, grid, is_masked, ne = entry
    if _debug_forced_fallback(launcher, compiled):
        return False
    # 每次调用都取当前 stream 与当前张量指针，绝不复用旧值。
    stream = torch_npu.npu.current_stream(dev).npu_stream
    idx = dev.index
    if idx is None or _current_device_index() != idx:
        with torch_npu.npu.device(dev):
            _do_launch(launcher, grid, stream, fn, packed, ptr, ne, is_masked)
        return True
    _do_launch(launcher, grid, stream, fn, packed, ptr, ne, is_masked)
    return True


def _strided_launch(x, base, n_elements, dims):
    if len(dims) > _MAX_RANK:
        # 宿主侧拆分最外层，保证 kernel 秩不超过 8。
        s0, t0 = dims[0]
        inner = 1
        for s, _ in dims[1:]:
            inner *= s
        for i in range(s0):
            _strided_launch(x, base + i * t0, inner, dims[1:])
        return
    span = base + 1
    for s, t in dims:
        span += (s - 1) * t
    idx64 = span > 0x7FFFFFFF
    shapes = [d[0] for d in dims] + [1] * (_MAX_RANK - len(dims))
    strides = [d[1] for d in dims] + [1] * (_MAX_RANK - len(dims))
    rank = len(dims)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    with torch_npu.npu.device(x.device):
        _selu_strided[grid](
            x,
            n_elements,
            base,
            BLOCK=_BLOCK_STRIDED,
            s0=shapes[0], s1=shapes[1], s2=shapes[2], s3=shapes[3],
            s4=shapes[4], s5=shapes[5], s6=shapes[6], s7=shapes[7],
            t0=strides[0], t1=strides[1], t2=strides[2], t3=strides[3],
            t4=strides[4], t5=strides[5], t6=strides[6], t7=strides[7],
            RANK=rank,
            IDX64=idx64,
        )


def _selu_inplace(x):
    n_elements = x.numel()
    if n_elements == 0:
        return x
    f16_math = x.dtype == torch.float16
    if x.is_contiguous():
        block = _BLOCK_F16 if f16_math else _BLOCK_F32
        if n_elements < block:
            # 短张量缩小分块，尾块仍能用足够少的 lane 覆盖。
            block = 1 << max(7, (n_elements - 1).bit_length())
        masked = (n_elements % block) != 0
        if not _launch_cached_flat(x, n_elements, block, f16_math, masked):
            _launch_plain_flat(x, n_elements, block, f16_math, masked)
        return x
    dims = _collapse(x.shape, x.stride())
    if not dims:
        # 全部维为 size 1 -> 单元素；视图指针已含 storage_offset，相对基准为 0。
        dims = [(1, 1)]
    _strided_launch(x, 0, n_elements, dims)
    return x


def run(A):
    return _selu_inplace(A)

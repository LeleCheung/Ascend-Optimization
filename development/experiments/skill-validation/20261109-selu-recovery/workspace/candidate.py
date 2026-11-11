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

# 大输入分块上限：FP16 用到 16384，BF16/FP32 收在 8192。
_BLOCK_MAX_F16 = 16384
_BLOCK_MAX_OTHER = 8192
# 小/中输入按约 8 个 program 反推 2 的幂分块，尾块一律用 mask 保证正确。
_TARGET_PROGRAMS = 8
_BLOCK_MIN = 128
_BLOCK_STRIDED = 2048
_MAX_RANK = 8

# 16 字节对齐是向量化搬运的前提；不满足时退回普通 Triton 调度。
_ALIGN_BYTES = 16

# 编译缓存：键 -> (launcher, bound_launch, compiled, function, packed_metadata,
# grid, masked, n)。只保存编译句柄、绑定启动方法与不可变元数据，
# 绝不保存张量、结果或旧 stream。
_LAUNCH_CACHE = {}

# 快速命中表：键 (device_index, dtype, numel) -> (bound_launch, function,
# packed_metadata, grid, masked, n)。只登记编译已通过调试/编译/采集检查的普通
# 产物；该编译产物的 launcher 模式与 metadata 在整个生命周期内固定，故命中时
# 无需重复 getattr。表内不含张量、结果或旧 stream。
_FAST_CACHE = {}


def _resolve_raw_stream_fn():
    """模块初始化只缓存“查询当前 stream 的函数对象”，不缓存返回的句柄。

    后端 backend_register.py 中 torch_npu 的 get_current_stream 本身优先
    _npu_getCurrentRawStreamNoWait(device)，不存在才用 _npu_getCurrentRawStream；
    生成的启动器默认 TRITON_ENABLE_TASKQUEUE=true，通过任务队列异步发射，
    故 NoWait 与之一致。两者都只查询运行时 stream 元数据，不执行目标计算。
    """
    api = getattr(torch_npu, "_C", None)
    if api is None:
        return None
    fn = getattr(api, "_npu_getCurrentRawStreamNoWait", None)
    if fn is not None:
        return fn
    return getattr(api, "_npu_getCurrentRawStream", None)


# 只保存函数对象本身；每次调用都重新查询当前 stream。
_RAW_STREAM_FN = _resolve_raw_stream_fn()


def _resolve_device_fn():
    """缓存“查询当前设备”的函数对象，每次调用读取当前设备，不缓存返回值。

    优先 C 接口 _npu_getDevice（免去 Python 层包装），不存在时回退到公共
    torch_npu.npu.current_device。二者都只查询设备元数据，不执行目标计算。
    """
    api = getattr(torch_npu, "_C", None)
    fn = getattr(api, "_npu_getDevice", None) if api is not None else None
    if fn is not None:
        return fn
    npu = getattr(torch_npu, "npu", None)
    if npu is not None:
        return getattr(npu, "current_device", None)
    return None


# 只保存函数对象本身；每次调用都重新查询当前设备。
_DEVICE_FN = _resolve_device_fn()


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
    if _DEVICE_FN is None:
        return None
    try:
        return int(_DEVICE_FN())
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


def _do_launch(launch, grid, stream, fn, packed, ptr, ne, masked):
    """按后端 ABI 直接发射一次，参数个数固定，不做任何运行时探测。

    launch 是缓存的 NPULauncher.launch 绑定方法；调用参数与 launcher.launch 等价。

    后端 make_launcher 的格式串为 "iiiKKOOOO" + 逐签名参数的 format_of + debug_abi：
    前 9 个槽位是常规前缀（grid/1/1/stream/function/packed/metadata 占位），
    之后严格按内核签名顺序传显式参数——先是当前指针，masked 变体再传元素数 n，
    最后是本内核两个显式 constexpr 的占位（BLOCK、F16MATH）。constexpr 在后端
    同样各保留一个 O 槽位，所以这两个尾部 None 是参数占位，不是隐藏参数，
    不能省略，也不能在末尾继续追加。
    """
    if masked:
        # 13 个参数：前缀 9 + 指针 + n + BLOCK 占位 + F16MATH 占位。
        launch(grid, 1, 1, stream, fn, packed, None, None, None, ptr, ne, None, None)
    else:
        # 12 个参数：前缀 9 + 指针 + BLOCK 占位 + F16MATH 占位。
        launch(grid, 1, 1, stream, fn, packed, None, None, None, ptr, None, None)


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
            launcher.launch,
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
    idx = dev.index
    if idx is None:
        return False
    ptr = int(x.data_ptr())
    if ptr % _ALIGN_BYTES != 0:
        return False
    # 该核只依赖连续元素个数，无需每次构造 shape tuple。
    key = (idx, x.dtype, n_elements, block, masked)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        try:
            entry = _compile_flat(x, n_elements, block, f16_math, masked, dev)
        except Exception:
            return False
        launcher = entry[0]
        compiled = entry[2]
        if _debug_forced_fallback(launcher, compiled):
            _LAUNCH_CACHE.pop(key, None)
            return False
        _LAUNCH_CACHE[key] = entry
    launcher, launch, compiled, fn, packed, grid, is_masked, ne = entry
    if _debug_forced_fallback(launcher, compiled):
        return False
    # 每次调用都取当前张量指针，绝不复用旧值。
    # 用模块初始化缓存的函数对象重新查询当前 stream；禁止缓存返回的句柄。
    if _RAW_STREAM_FN is None:
        return False
    try:
        stream = _RAW_STREAM_FN(idx)
    except Exception:
        return False
    if _current_device_index() != idx:
        with torch_npu.npu.device(dev):
            _do_launch(launch, grid, stream, fn, packed, ptr, ne, is_masked)
    else:
        _do_launch(launch, grid, stream, fn, packed, ptr, ne, is_masked)
    # 发射成功后登记快速表。该编译产物的 debug/compile-only/msprof 标志已在
    # 上面检查并排除，模式在生命周期内固定；任何失败都不会留下半成品条目。
    if _DEVICE_FN is not None:
        _FAST_CACHE[(idx, x.dtype, ne)] = (launch, fn, packed, grid, is_masked, ne)
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


def _select_block(n_elements, f16_math):
    """按元素数选 2 的幂分块，上限取 dtype 上限。

    小/中输入以约 _TARGET_PROGRAMS 个 program 反推分块，避免超小输入用大块；
    块大于元素数时由 masked 变体处理，不为了省 mask 而越界。
    """
    cap = _BLOCK_MAX_F16 if f16_math else _BLOCK_MAX_OTHER
    want = (n_elements + _TARGET_PROGRAMS - 1) // _TARGET_PROGRAMS
    block = 1 << max(_BLOCK_MIN.bit_length() - 1, (want - 1).bit_length())
    return block if block < cap else cap


def _selu_inplace(x):
    n_elements = x.numel()
    if n_elements == 0:
        return x
    f16_math = x.dtype == torch.float16
    if x.is_contiguous():
        block = _select_block(n_elements, f16_math)
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
    """公开入口：原地 SELU。

    命中快速表时只核对连续性、当前 dtype/numel/device、指针对齐与当前设备，
    读取当前 raw stream 后按已验证 ABI 直接发射一次；不重算分块、不逐次读取
    同一编译 metadata。任何发射前检查不通过或未命中都进入原有可靠路径。
    """
    n_elements = A.numel()
    if n_elements:
        idx = A.device.index
        if idx is not None and A.is_contiguous():
            entry = _FAST_CACHE.get((idx, A.dtype, n_elements))
            if entry is not None and _DEVICE_FN is not None:
                # 当前设备每次重新查询；不一致说明该产物不属于当前上下文。
                try:
                    if _DEVICE_FN() != idx:
                        entry = None
                except Exception:
                    entry = None
                if entry is not None:
                    ptr = A.data_ptr()
                    if ptr % _ALIGN_BYTES == 0:
                        try:
                            stream = _RAW_STREAM_FN(idx)
                        except Exception:
                            stream = None
                        if stream is not None:
                            launch, fn, packed, grid, is_masked, n = entry
                            # 已验证 ABI：前缀 9 槽 + 指针 + [n] + 两个
                            # constexpr 占位。发射后异常直接上抛，绝不回退重发。
                            if is_masked:
                                launch(grid, 1, 1, stream, fn, packed,
                                       None, None, None, ptr, n, None, None)
                            else:
                                launch(grid, 1, 1, stream, fn, packed,
                                       None, None, None, ptr, None, None)
                            return A
    return _selu_inplace(A)

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""昇腾设备上的原地 log10，全部设备计算与布局处理严格由 Triton 完成。

公共入口 ``run(A)``；其余辅助函数以单下划线命名。

相对 starter.py 的改写要点：
  * 删除 ``torch.log10_`` 的目标设备回退，也删除由 ``contiguous()``/``copy_``
    组成的非连续路径；非连续输入按真实 shape/stride 原地读写。
  * 布局处理只在宿主侧做元数据合并：删除 size-1 维、合并内存相邻维（合并后
    保留最内侧 stride）、把 unit-stride 维移到最内侧，不建中间张量。
  * 内核三条路径：平坦连续、统一步长（rank-1 等距视图）、行式多维；
    行式路径的前导维由标量 program_id 解码，块内沿最内侧维搬运。
  * 平坦连续路径改为有限网格 + 格内循环：program 数不超过有限上限，
    每个 program 以 loop_stride 为跨度循环处理本段，单 program 搬运粒度
    为 BLOCK 个元素。上一版对 1.6e7 元素输入发射 8192 个 program，
    设备任务约 366us 而参考约 26us，且三种 dtype 时间几乎相同，指向
    逐 program 调度开销而非带宽；本版把这部分开销压到格内循环。
  * 宿主侧只缓存网格与分块常量，不缓存张量、地址、结果或 stream。

本版（相对上一版的唯一机制）新增「编译启动器直发」宿主发射路径：
  * 上一版正式反馈 geo_mean=0.2278，1024 元素 79.3us、65536 元素 70.0us、
    1048576 元素 69.7us、4096x4096 fp16 71.5us，参考约 15–16us；候选时间
    与元素数、dtype 均无关联，指向与设备工作量无关的宿主发射固定开销。
  * 因此本轮不再改动分块、网格与布局路径，只改宿主如何把已编译产物发出去：
    普通调度每次都要做实参绑定、特化键计算与缓存查找；改为首次普通调度后
    登记该产物的启动器句柄、function 与 packed_metadata，后续按当前实参、
    当前 device 与当前 raw stream 直接调用，缓存命中即发射。
  * 直发表的键覆盖 Triton 的编译期特化维度（dtype、指针 16 字节对齐、
    整数 ==1/%16/宽度），任一维度不同不复用；stream、device 与指针每次
    重新读取，不缓存旧 stream，也不缓存张量结果。
  * 存在采集／调试钩子时禁用直发表；设备发射异常不重发原地计算，只有
    宿主侧 ABI 不匹配才退回普通调度。
"""

import torch
import triton
import triton.language as tl

# 1 / ln(10)。内核中以字面量内联，避免每次启动多传一个标量参数。
_LOG10E = 0.4342944819032518

# 单 program 每轮处理的元素数。历史设备异常出现在更大分块上（UB 多缓冲放大），
# 这里从 2048 提升到 4096 作为单步改动：fp32 中间量约 16KB/缓冲，仍留有余量。
_BLOCK = 4096

# 有限网格的 program 数上限。参考 CANN 实现的 Block Dim 为 40（与设备向量核数
# 一致），此处取 40，使大输入以单波次长循环推进，块调度开销按循环次数摊薄。
_PROGRAMS = 40

# 散布类路径（统一步长、行式多维）保持 2048：这些路径每轮除值缓冲外还要
# 持有地址向量且每 program 元素数随视图变化，UB 压力高于平坦路径；本轮
# 不改变它们的分块，避免与平坦路径的机制改动混淆。
_SCATTER_BLOCK = 2048

# 单次发射的 grid 轴上限。
_GRID_LIMIT = 65535

# 编译启动器直发表的键 -> (launcher, function, packed_metadata)。
# 只缓存编译产物句柄与宿主发射元数据；张量、地址、stream、当前 device
# 每次调用重新读取，绝不进入该表。
_FAST_LAUNCH = {}

# knobs.runtime 的惰性句柄：用于判断当前是否处于采集／调试模式。
_KNOBS_SENTINEL = object()
_KNOBS_RUNTIME = _KNOBS_SENTINEL

# 当前活动后端驱动句柄的惰性缓存。驱动对象是进程级后端，不是 stream。
_DRIVER_SENTINEL = object()
_ACTIVE_DRIVER = _DRIVER_SENTINEL


@triton.jit
def _log10_flat_kernel(
    x_ptr,
    n_elements,
    loop_stride,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """平坦连续路径：有限网格 + 格内循环，块内直接连续寻址。

    program 数在宿主侧压到不超过有限上限，loop_stride = grid * BLOCK，
    每个 program 从 pid*BLOCK 起以 loop_stride 为跨度循环到段尾。这样
    大输入的 program 数由「元素数/BLOCK」降为固定上限，逐 program 的
    调度开销被循环摊薄，单轮搬运粒度提升到 BLOCK 个元素。

    n_elements 为段长，掩码用绝对平坦下标比较；EVEN 表示段长是 BLOCK
    的整数倍，此时每轮都是满块，不需要掩码。
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    if EVEN:
        for offset in tl.range(start, n_elements, loop_stride):
            offsets = offset + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offsets)
            y = tl.log(x.to(tl.float32)) * 0.4342944819032518
            tl.store(x_ptr + offsets, y.to(x.dtype))
    else:
        for offset in tl.range(start, n_elements, loop_stride):
            offsets = offset + tl.arange(0, BLOCK)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
            y = tl.log(x.to(tl.float32)) * 0.4342944819032518
            tl.store(x_ptr + offsets, y.to(x.dtype), mask=mask)


@triton.jit
def _log10_uniform_kernel(
    x_ptr,
    base_off,
    n_elements,
    step,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """统一步长路径：一维等距视图（如 x[::2]），按真实步长原地读写。"""
    pid = tl.program_id(axis=0)
    rel = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        ptrs = x_ptr + (base_off + rel) * step
        x = tl.load(ptrs)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(ptrs, y.to(x.dtype))
    else:
        mask = rel < n_elements
        ptrs = x_ptr + (base_off + rel) * step
        x = tl.load(ptrs, mask=mask, other=1.0)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(ptrs, y.to(x.dtype), mask=mask)


@triton.jit
def _log10_rows_kernel(
    x_ptr,
    dim1,
    dim2,
    stride0,
    stride1,
    stride2,
    inner,
    inner_stride,
    outer_off,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """行式多维路径：一个 program 处理最内侧维的一段。

    前导（最多三维）下标由标量 program_id 整除／取模解码，块内无动态除法；
    inner_stride 可为非 1（转置、间隔切片等视图合并后的外侧步长）。
    """
    pid_n = tl.program_id(axis=0)
    pid_o = tl.program_id(axis=1) + outer_off
    base = (
        (pid_o // (dim1 * dim2)) * stride0
        + ((pid_o // dim2) % dim1) * stride1
        + (pid_o % dim2) * stride2
    )
    index = pid_n * BLOCK + tl.arange(0, BLOCK)
    ptrs = x_ptr + base + index * inner_stride
    if EVEN:
        x = tl.load(ptrs)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(ptrs, y.to(x.dtype))
    else:
        mask = index < inner
        x = tl.load(ptrs, mask=mask, other=1.0)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(ptrs, y.to(x.dtype), mask=mask)


def _merge_dims(shape, strides):
    """删除 size-1 维，合并内存相邻维。

    相邻维 (S, s_out)、(N, s_in) 满足 s_out == N * s_in 时合并为
    (S*N, s_in)：合并后必须保留最内侧 stride，否则地址映射会越界。
    """
    new_shape = []
    new_strides = []
    for size, stride in zip(shape, strides):
        size = int(size)
        stride = int(stride)
        if size == 1:
            continue
        if new_shape and new_strides[-1] == stride * size:
            new_shape[-1] *= size
            new_strides[-1] = stride
        else:
            new_shape.append(size)
            new_strides.append(stride)
    if not new_shape:
        new_shape = [1]
        new_strides = [1]
    return new_shape, new_strides


def _inner_last(shape, strides):
    """把 unit-stride 维移到最内侧，使块内访存连续。"""
    inner = -1
    for index, stride in enumerate(strides):
        if stride == 1:
            inner = index
            break
    if inner < 0 or inner == len(shape) - 1:
        return shape, strides
    shape = list(shape)
    strides = list(strides)
    shape.append(shape.pop(inner))
    strides.append(strides.pop(inner))
    return shape, strides


def _block_for(n_elements, cap):
    """按实际规模选择分块：大输入用该路径的固定上限，小输入降到 2 的幂。"""
    if n_elements >= cap:
        return cap
    block = triton.next_power_of_2(n_elements)
    if block < 2:
        block = 2
    return block


def _runtime_knobs():
    """惰性取得 knobs.runtime 模块句柄；取得失败时返回 None。"""
    global _KNOBS_RUNTIME
    if _KNOBS_RUNTIME is _KNOBS_SENTINEL:
        try:
            from triton.runtime import knobs

            _KNOBS_RUNTIME = knobs.runtime
        except Exception:
            _KNOBS_RUNTIME = None
    return _KNOBS_RUNTIME


def _hooks_quiet():
    """当前是否没有采集／调试钩子。有钩子时禁用直发表，避免绕过统计回调。"""
    runtime = _runtime_knobs()
    if runtime is None:
        return False
    return (
        getattr(runtime, "launch_enter_hook", None) is None
        and getattr(runtime, "launch_exit_hook", None) is None
    )


def _current_stream():
    """每次调用读取当前 device 与当前 raw stream，不缓存旧 stream。"""
    global _ACTIVE_DRIVER
    driver = _ACTIVE_DRIVER
    if driver is _DRIVER_SENTINEL:
        from triton.runtime import driver as _driver_module

        driver = _driver_module.active
        _ACTIVE_DRIVER = driver
    device = driver.get_current_device()
    return driver.get_current_stream(device)


def _launch_key(tag, ptrs, ints, block, even):
    """直发表的键：算子标记 + 分块 + 是否整块 + 各入参的编译期特化类别。

    键必须覆盖 Triton 的入参特化维度：张量 dtype 与指针对齐（16 字节）、
    整数的 ==1、%16 以及 int32/int64 宽度。任一维度不同的产物不可复用，
    否则会用到错误的特化二进制。
    """
    parts = [tag]
    for tensor in ptrs:
        parts.append(str(tensor.dtype))
    parts.append(str(block))
    parts.append("e" if even else "o")
    for tensor in ptrs:
        parts.append("a" if (int(tensor.data_ptr()) % 16) == 0 else "u")
    for value in ints:
        value = int(value)
        parts.append(
            ("one" if value == 1 else "")
            + ("16" if value % 16 == 0 else "n")
            + ("32" if -2147483648 <= value < 2147483648 else "64")
        )
    return tuple(parts)


def _cache_dicts(jit_fn):
    """取出 JIT 函数的编译产物缓存字典，兼容多种缓存容器形状。"""
    caches = getattr(jit_fn, "device_caches", None)
    if not caches:
        return ()
    try:
        values = list(caches.values())
    except Exception:
        return ()
    found = []
    for value in values:
        candidate = value
        if isinstance(value, (tuple, list)):
            if not value:
                continue
            candidate = value[0]
        if isinstance(candidate, dict):
            found.append(candidate)
        else:
            inner = getattr(candidate, "cache", None)
            if isinstance(inner, dict):
                found.append(inner)
    return tuple(found)


def _cache_snapshot(jit_fn):
    """记录发射前的缓存键集合，用于精确识别本次发射新增的产物。"""
    snapshot = {}
    for cache in _cache_dicts(jit_fn):
        snapshot[id(cache)] = frozenset(cache.keys())
    return snapshot


def _new_kernel(jit_fn, snapshot):
    """返回本次发射新增的编译产物对象；无法唯一确定时返回 None。"""
    for cache in _cache_dicts(jit_fn):
        before = snapshot.get(id(cache))
        if before is None:
            continue
        for key, value in list(cache.items()):
            if key in before:
                continue
            if hasattr(value, "packed_metadata") and hasattr(value, "run"):
                return value
    return None


def _capture_launcher(jit_fn, key, snapshot):
    """把本次普通调度用到的编译产物登记为直发表条目。

    只登记句柄与元数据；登记失败静默放弃，下次仍走普通调度。
    """
    kernel = _new_kernel(jit_fn, snapshot)
    if kernel is None:
        return
    try:
        launcher = kernel.run
        function = getattr(kernel, "function", None)
        packed_metadata = kernel.packed_metadata
    except Exception:
        return
    if launcher is None or function is None or packed_metadata is None:
        return
    _FAST_LAUNCH[key] = (launcher, function, packed_metadata)


def _emit(jit_fn, key, grid, full_args, dynamic_args):
    """统一的单机制发射。

    命中直发表时以当前 stream 与当前实参直接调用编译启动器，跳过 JIT 的
    实参绑定、特化键计算与缓存查找；未命中或处于采集模式时走普通调度，
    并在成功后登记本次产物。设备发射异常不重发原地计算：仅宿主侧 ABI
    不匹配（TypeError/AttributeError）才退回普通调度。
    """
    grid_0 = int(grid[0])
    grid_1 = int(grid[1]) if len(grid) > 1 else 1
    grid_2 = int(grid[2]) if len(grid) > 2 else 1
    if _hooks_quiet():
        entry = _FAST_LAUNCH.get(key)
        if entry is not None:
            launcher, function, packed_metadata = entry
            try:
                launcher(
                    grid_0,
                    grid_1,
                    grid_2,
                    _current_stream(),
                    function,
                    packed_metadata,
                    None,
                    None,
                    None,
                    *dynamic_args,
                )
                return
            except (TypeError, AttributeError):
                _FAST_LAUNCH.pop(key, None)
        snapshot = _cache_snapshot(jit_fn)
        jit_fn[grid](*full_args)
        _capture_launcher(jit_fn, key, snapshot)
        return
    jit_fn[grid](*full_args)


def _launch_flat(x, n_elements):
    """平坦连续路径发射：program 数压到 _PROGRAMS 以内，其余由格内循环完成。"""
    block = _block_for(n_elements, _BLOCK)
    blocks = (n_elements + block - 1) // block
    grid = blocks if blocks < _PROGRAMS else _PROGRAMS
    loop_stride = grid * block
    even = (n_elements % block) == 0
    _emit(
        _log10_flat_kernel,
        _launch_key("flat", (x,), (n_elements, loop_stride), block, even),
        (grid,),
        (x, n_elements, loop_stride, block, even),
        (x, n_elements, loop_stride),
    )


def _launch_uniform(x, n_elements, step):
    block = _block_for(n_elements, _SCATTER_BLOCK)
    blocks = (n_elements + block - 1) // block
    if blocks <= _GRID_LIMIT:
        even = (n_elements % block) == 0
        _emit(
            _log10_uniform_kernel,
            _launch_key("uniform", (x,), (0, n_elements, step), block, even),
            (blocks,),
            (x, 0, n_elements, step, block, even),
            (x, 0, n_elements, step),
        )
        return
    span = _GRID_LIMIT * block
    start = 0
    while start < n_elements:
        count = n_elements - start
        if count > span:
            count = span
        chunk_blocks = (count + block - 1) // block
        even = (count % block) == 0
        _emit(
            _log10_uniform_kernel,
            _launch_key("uniform", (x,), (start, count, step), block, even),
            (chunk_blocks,),
            (x, start, count, step, block, even),
            (x, start, count, step),
        )
        start += count


def _outer_triple(outer_shape, outer_strides):
    """把前导下标折叠为最多三个 (dim, stride)，缺失维取 dim=1、stride=0。"""
    count = len(outer_shape)
    if count == 0:
        return 1, 1, 0, 0, 0
    if count == 1:
        return 1, 1, outer_strides[0], 0, 0
    if count == 2:
        return outer_shape[1], 1, outer_strides[0], outer_strides[1], 0
    return (
        outer_shape[1],
        outer_shape[2],
        outer_strides[0],
        outer_strides[1],
        outer_strides[2],
    )


def _launch_rows(x, shape, strides):
    inner = int(shape[-1])
    inner_stride = int(strides[-1])
    outer_shape = shape[:-1]
    dim1, dim2, stride0, stride1, stride2 = _outer_triple(outer_shape, strides[:-1])
    outer = 1
    for size in outer_shape:
        outer *= int(size)
    block = _block_for(inner, _SCATTER_BLOCK)
    blocks_inner = (inner + block - 1) // block
    even = (inner % block) == 0
    scalars = (dim1, dim2, stride0, stride1, stride2, inner, inner_stride)
    start = 0
    while start < outer:
        count = outer - start
        if count > _GRID_LIMIT:
            count = _GRID_LIMIT
        _emit(
            _log10_rows_kernel,
            _launch_key("rows", (x,), scalars + (start,), block, even),
            (blocks_inner, count),
            (x,) + scalars + (start, block, even),
            (x,) + scalars + (start,),
        )
        start += count


def _dispatch(x):
    shape, strides = _merge_dims(x.shape, x.stride())
    if len(shape) > 4:
        # 沿实际第 0 维用元数据切片展开前导维，不构造中间张量。
        # 循环上限取实际第 0 维长度：该维长度为 1 时被合并删除，
        # 此时 x[0] 不改变地址，仅降低秩。
        for index in range(x.shape[0]):
            _dispatch(x[index])
        return
    shape, strides = _inner_last(shape, strides)
    shape, strides = _merge_dims(shape, strides)
    if len(shape) == 1:
        n_elements = int(shape[0])
        step = int(strides[0])
        if step == 1:
            _launch_flat(x, n_elements)
        else:
            _launch_uniform(x, n_elements, step)
        return
    _launch_rows(x, shape, strides)


def run(A):
    if not isinstance(A, torch.Tensor):
        raise TypeError("run 需要一个 torch.Tensor 作为第一个参数。")
    if A.numel() == 0:
        return A
    _dispatch(A)
    return A

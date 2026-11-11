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

本轮修订（只修宿主直发路径的三处缺陷，内核与布局路径不变）：
  * 直发表键加入当前 device：直发表条目按 (device, 特化键) 索引，跨设备
    一律不复用编译句柄。
  * 冷启动登记：发射前快照缺失某个缓存字典时按空集合处理，发射后只在
    「新增产物唯一」时登记，杜绝把别的特化产物登记到当前键。
  * 钩子查询改为本机事实位置 ``triton.knobs``（``triton.runtime.knobs``
    在 3.5.1 不存在）；查询失败只禁用直发，不永久污染状态。

历史轮次要点（保留供追溯，均已被本版取代）：
  * 直发表最初从未登记条目（钩子非 None 判据错误），随后改为按
    ``JITFunction.run`` 返回值登记、按 HookChain 的 ``calls`` 是否为空判据，
    直发才真正生效；launch-probe 确认生效后 ``_FAST_LAUNCH`` 有 1–2 条记录。
  * 该版正式反馈 geo_mean=0.3327：1024 元素约 55–57us、65536 与 1048576 约
    45–48us、4096x4096 约 63–66us，参考 16–33us。候选时间几乎只随
    「是否有一次宿主调用」变化，元素数与 dtype 的影响不足 10us。

本轮机制（唯一改动：把「每次重建的发射参数」换成一次建好、多次直发的计划）：
  * host-profile 给出宿主逐项证据：单次调用约 137us（采样口径），其中
    ``NPULauncher.__call__`` 含 C 启动器实参解析约 34us，
    ``_current_device_stream`` 约 26us（后端注册表 execute_func 两层派发后
    才调到 ``_npu_getDevice`` / ``_npu_getCurrentRawStreamNoWait``），
    ``_launch_key`` + ``_block_for`` 约 31us，布局合并约 11us，
    ``_quiet_runtime`` 约 6us。逐项都与设备工作量无关。
  * 因此本版只改宿主组织方式：按 (dtype, shape, stride, 16 字节对齐类,
    device) 缓存一份「发射计划」，内含分块、网格、核函数实参尾部、
    ``function``、``packed_metadata`` 与原生 ``launch`` C 入口；热路径只做
    一次字典查询、一次 device 比对、一次 stream 查询和一次 C 调用。
  * 原生入口依据本机事实：``CompiledKernel.run`` 是 ``NPULauncher``，其
    ``__call__`` 只做 compile_only／tensor 注册／调试网关判定后转调
    ``self.launch``；登记时已排除这些模式，故热路径直接调用 ``launch``。
  * 正确性边界不变：stride、offset、dtype、原地语义、当前 device 与当前
    raw stream 均按次读取；device 变化即退回通用派发；钩子门每次比对
    HookChain 的 ``calls``，一旦有回调立即改走普通 JIT 调度；只有宿主侧
    实参解析失败（TypeError/AttributeError，设备任务尚未下发）才把该计划
    整体降级，不重发原地计算。
  * 内核数学、分块常量、有限网格循环、布局合并与行式解码全未改动，
    性能差异只能来自宿主发射路径。
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

# 发射计划缓存：(dtype, shape, stride, 对齐类, device) -> 计划。
# 计划内只保存与该次调用无关的常量（分块、网格、核函数实参尾部）和已经登记
# 好的编译产物句柄；张量引用、数据地址、stream 一律不进入缓存，每次调用重新
# 传入或重新查询，绝不缓存结果张量、旧地址或旧 stream。
_PLANS = {}


class _AlwaysActiveGates(object):
    """钩子状态不可判定时的保守门：始终让热路径退回普通 JIT 调度。"""

    __slots__ = ("calls",)

    def __init__(self):
        self.calls = (True,)


_ALWAYS_ACTIVE = _AlwaysActiveGates()

# 快速 device／stream 查询句柄。惰性解析一次，解析失败时退回通用派发。
_STREAM_READY = False
_FAST_DEVICE = None
_FAST_STREAM = None
_GENERIC_STREAM = None

# knobs.runtime 的惰性句柄：用于判断当前是否处于采集／调试模式。
_KNOBS_SENTINEL = object()
_KNOBS_RUNTIME = _KNOBS_SENTINEL


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
    """惰性取得 knobs.runtime 句柄。

    本机事实：knobs 暴露在 ``triton.knobs``；``triton.runtime.knobs``
    在 3.5.1 不存在。两者都取不到时返回 None，仅表示「钩子状态未知」，
    由调用方保守地走普通调度，不回写永久禁用标记。
    """
    global _KNOBS_RUNTIME
    if _KNOBS_RUNTIME is _KNOBS_SENTINEL:
        runtime = None
        try:
            from triton.runtime import knobs as _knobs_module

            runtime = _knobs_module.runtime
        except Exception:
            runtime = None
        if runtime is None:
            try:
                import triton.knobs as _knobs_module

                runtime = getattr(_knobs_module, "runtime", None)
            except Exception:
                runtime = None
        if runtime is None:
            runtime = getattr(getattr(triton, "knobs", None), "runtime", None)
        if runtime is None:
            # 保持哨兵，后续调用可重新查询；不写死「永久缺失」。
            return None
        _KNOBS_RUNTIME = runtime
    return _KNOBS_RUNTIME


def _hook_chain_active(chain):
    """判断发射钩子容器里是否真的挂有回调。

    本机事实（runtime-api.json 的 runtime_hooks 段）：``knobs.runtime`` 的
    ``launch_enter_hook`` / ``launch_exit_hook`` 是类 ``HookChain`` 的常驻
    容器，字段本身永远非 None；其真实状态由内部 ``calls`` 列表是否为空决定，
    ``HookChain.__call__`` 在空列表时什么也不做。因此「非 None」不能当作
    「存在采集／调试钩子」，否则快速表会被永久禁用。

    取不到 ``calls`` 列表时无法判定，按「有回调」保守处理，仍走普通调度。
    """
    if chain is None:
        return False
    calls = getattr(chain, "calls", None)
    if calls is None:
        return True
    try:
        return len(calls) != 0
    except Exception:
        return True


def _quiet_runtime():
    """没有采集／调试钩子时返回 knobs.runtime 句柄，否则返回 None。

    钩子状态无法查询时返回 None，宁可走普通调度，也不盲目绕过回调。
    """
    runtime = _runtime_knobs()
    if runtime is None:
        return None
    if _hook_chain_active(getattr(runtime, "launch_enter_hook", None)):
        return None
    if _hook_chain_active(getattr(runtime, "launch_exit_hook", None)):
        return None
    return runtime


def _resolve_stream_hooks():
    """惰性解析「当前 device」与「当前 raw stream」的最短查询路径。

    本机事实（runtime-api 的 get_current_stream 策略）：后端驱动的
    ``get_current_device()`` 与 ``get_current_stream(device)`` 都先经过
    ``backend_strategy_registry`` 的 register/execute_func 两三层派发，最后
    才落到 ``torch_npu._C._npu_getDevice`` 与
    ``torch_npu._C._npu_getCurrentRawStreamNoWait``；host-profile 显示这层
    派发每次调用约占十微秒量级。这里把最后两个 C 入口缓存成函数对象，热路径
    直接调用；任一入口缺失就退回通用派发，不改变语义。

    只缓存函数对象，不缓存任何 device／stream 的返回值。
    """
    global _STREAM_READY, _FAST_DEVICE, _FAST_STREAM, _GENERIC_STREAM
    if _STREAM_READY:
        return
    from triton.runtime import driver as _driver_module

    driver = _driver_module.active
    generic = driver.get_current_stream
    _GENERIC_STREAM = generic
    fast_device = None
    fast_stream = None
    try:
        from torch_npu._C import _npu_getDevice as fast_device
    except Exception:
        fast_device = None
    try:
        from torch_npu._C import _npu_getCurrentRawStreamNoWait as fast_stream
    except Exception:
        fast_stream = None
    if fast_device is None:
        fast_device = driver.get_current_device
    if fast_stream is None:
        fast_stream = generic
    _FAST_DEVICE = fast_device
    _FAST_STREAM = fast_stream
    _STREAM_READY = True


def _stream_for(device):
    """按「当前 device 是否仍是计划里的 device」选择最短的 stream 查询路径。

    当前 device 每次重新读取；device 变化时退回通用派发（其内部按传入
    device 查询），绝不复用旧 stream。
    """
    if _FAST_DEVICE() == device:
        return _FAST_STREAM(device)
    return _GENERIC_STREAM(device)


def _plan_gates():
    """返回两个钩子门，供热路径零成本判定当前是否存在采集／调试回调。

    返回的是 ``knobs.runtime`` 上的 HookChain 本体，热路径只看其 ``calls``
    列表是否为空。回调后来才挂上时，同一容器会立即变真，热路径随之让位给
    普通 JIT 调度；因此不会绕过真实回调。状态不可判定时返回恒真的保守门。
    """
    runtime = _quiet_runtime()
    if runtime is None:
        return _ALWAYS_ACTIVE, _ALWAYS_ACTIVE
    enter = getattr(runtime, "launch_enter_hook", None)
    exit_hook = getattr(runtime, "launch_exit_hook", None)
    if enter is None or exit_hook is None:
        return _ALWAYS_ACTIVE, _ALWAYS_ACTIVE
    return enter, exit_hook


def _make_op(compiled, jit_fn, grid, tail):
    """把一次发射登记为计划操作：优先直发原生入口，否则保留 JIT 调度。

    原生入口依据本机事实：``CompiledKernel.run`` 是 ``NPULauncher`` 实例，
    其 ``__call__`` 只做 compile_only／tensor 注册／调试网关判定后转调
    ``self.launch``（C 入口）。这里缓存 ``launch`` 本体，热路径省掉那层
    Python 包装与模块属性写入。

    只有已排除 compile_only、tensor 注册、调试模式与活动钩子的普通产物
    才登记直发；其余情况 ``launch`` 保持 None，该操作永远走普通 JIT 调度。
    """
    launch = None
    function = None
    packed_metadata = None
    if compiled is not None:
        try:
            if hasattr(compiled, "result"):
                compiled = compiled.result()
            launcher = compiled.run
            native = getattr(launcher, "launch", None)
            meta = getattr(launcher, "metadata", None)
            if meta is None:
                meta = getattr(compiled, "metadata", None)
            if (
                native is not None
                and not getattr(launcher, "compile_only", False)
                and not getattr(launcher, "enable_msprof_register_tensor", False)
                and not getattr(meta, "debug_enabled", False)
                and not getattr(meta, "debug", False)
            ):
                function = getattr(compiled, "function", None)
                packed_metadata = getattr(compiled, "packed_metadata", None)
                if function is not None and packed_metadata is not None:
                    launch = native
        except Exception:
            launch = None
            function = None
            packed_metadata = None
    grid = tuple(grid)
    grid_0 = int(grid[0])
    grid_1 = int(grid[1]) if len(grid) > 1 else 1
    grid_2 = int(grid[2]) if len(grid) > 2 else 1
    return (launch, function, packed_metadata, grid_0, grid_1, grid_2, tail, jit_fn, grid)


def _build_plan(x, key):
    """建立并缓存一次调用的发射计划；首次调用同时完成编译与发射。

    计划的键覆盖 dtype、shape、stride、16 字节对齐类与 device，因此 Triton
    的全部编译期特化维度（dtype、指针对齐、整数的 ==1／%16／宽度）都已被
    键固定，复用编译产物不会串用别的特化二进制；device 进入键保证句柄不跨
    设备复用。
    """
    _resolve_stream_hooks()
    specs, recurse = _plan_specs(x)
    ops = []
    for jit_fn, grid, tail in specs:
        # 普通 JIT 调度一次：完成编译并发射，返回值即本次使用的编译产物。
        compiled = jit_fn[grid](x, *tail)
        ops.append(_make_op(compiled, jit_fn, grid, tail))
    enter, exit_hook = _plan_gates()
    device = _FAST_DEVICE()
    plan = (device, enter, exit_hook, tuple(ops), recurse)
    _PLANS[key] = plan
    return plan


def _emit_ops(ops, x, device):
    """慢速发射：逐操作走普通 JIT 调度（钩子生效）或退回直发。"""
    for op in ops:
        if op[0] is None:
            op[7][op[8]](x, *op[6])
        else:
            op[0](
                op[3],
                op[4],
                op[5],
                _stream_for(device),
                op[1],
                op[2],
                None,
                None,
                None,
                x,
                *op[6],
            )


def _demote_plan(key, plan):
    """宿主侧 ABI 不匹配时把该计划整体降级为普通 JIT 调度。

    只在实参解析阶段失败（TypeError/AttributeError，设备任务尚未下发）时
    调用；此时原地计算未发生，退回普通调度补齐不会重复计算。
    """
    ops = tuple((None,) + op[1:] for op in plan[3])
    _PLANS[key] = (plan[0], plan[1], plan[2], ops, plan[4])


def _plan_key(x):
    """计划键：dtype、shape、stride、16 字节对齐类与 device。"""
    return (x.dtype, x.shape, x.stride(), x.data_ptr() & 15, x.device)


def _run_one(x):
    """单张量的热路径：计划命中即直发，否则建计划或退回普通调度。"""
    key = _plan_key(x)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _build_plan(x, key)
    ops = plan[3]
    if not ops:
        if plan[4]:
            for index in range(x.shape[0]):
                _run_one(x[index])
        return
    if plan[1].calls or plan[2].calls or len(ops) != 1 or ops[0][0] is None:
        _emit_ops(ops, x, plan[0])
        return
    op = ops[0]
    try:
        op[0](
            op[3],
            op[4],
            op[5],
            _stream_for(plan[0]),
            op[1],
            op[2],
            None,
            None,
            None,
            x,
            *op[6],
        )
    except (TypeError, AttributeError):
        _demote_plan(key, plan)
        _emit_ops(_PLANS[key][3], x, plan[0])


def _flat_specs(n_elements):
    """平坦连续路径的发射规格：program 数压到 _PROGRAMS 以内，其余靠格内循环。"""
    block = _block_for(n_elements, _BLOCK)
    blocks = (n_elements + block - 1) // block
    grid = blocks if blocks < _PROGRAMS else _PROGRAMS
    loop_stride = grid * block
    even = (n_elements % block) == 0
    return [(_log10_flat_kernel, (grid,), (n_elements, loop_stride, block, even))]


def _uniform_specs(n_elements, step):
    """统一步长路径的发射规格；超出单轴网格上限时按段切分。"""
    block = _block_for(n_elements, _SCATTER_BLOCK)
    blocks = (n_elements + block - 1) // block
    if blocks <= _GRID_LIMIT:
        even = (n_elements % block) == 0
        return [
            (_log10_uniform_kernel, (blocks,), (0, n_elements, step, block, even))
        ]
    specs = []
    span = _GRID_LIMIT * block
    start = 0
    while start < n_elements:
        count = n_elements - start
        if count > span:
            count = span
        chunk_blocks = (count + block - 1) // block
        even = (count % block) == 0
        specs.append(
            (
                _log10_uniform_kernel,
                (chunk_blocks,),
                (start, count, step, block, even),
            )
        )
        start += count
    return specs


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


def _rows_specs(shape, strides):
    """行式多维路径的发射规格；前导维超出网格上限时按段切分。"""
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
    specs = []
    start = 0
    while start < outer:
        count = outer - start
        if count > _GRID_LIMIT:
            count = _GRID_LIMIT
        specs.append(
            (
                _log10_rows_kernel,
                (blocks_inner, count),
                scalars + (start, block, even),
            )
        )
        start += count
    return specs


def _plan_specs(x):
    """按合并后的布局选择路径，返回 (发射规格列表, 是否需要递归展开)。

    布局合并只在建立计划时做一次：删除 size-1 维、合并内存相邻维、把
    unit-stride 维移到最内侧，全程只操作元数据，不构造中间张量。
    """
    if x.numel() == 0:
        return (), False
    shape, strides = _merge_dims(x.shape, x.stride())
    if len(shape) > 4:
        # 维度合并后仍超过四维：沿实际第 0 维用元数据切片展开前导维。
        return (), True
    shape, strides = _inner_last(shape, strides)
    shape, strides = _merge_dims(shape, strides)
    if len(shape) == 1:
        n_elements = int(shape[0])
        step = int(strides[0])
        if step == 1:
            return _flat_specs(n_elements), False
        return _uniform_specs(n_elements, step), False
    return _rows_specs(shape, strides), False


def run(A):
    if not isinstance(A, torch.Tensor):
        raise TypeError("run 需要一个 torch.Tensor 作为第一个参数。")
    _run_one(A)
    return A

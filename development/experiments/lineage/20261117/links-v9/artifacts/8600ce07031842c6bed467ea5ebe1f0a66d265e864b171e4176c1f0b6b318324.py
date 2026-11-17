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
  * 平坦连续路径按规模分流：不超过 _TILE_MAX 元素时由一个 program 用单块
    直接处理（无循环、无网格调度余量）；更大输入用 _PROGRAMS 个 program，
    每 program 每轮相邻预读两块（相距 BLOCK），循环轮数、步长与子块大小全部
    为编译期常量，残余块由一次带掩码的收尾块补齐。
  * 宿主侧只缓存网格与分块常量，不缓存张量、地址、结果或 stream；热路径按
    计划直发已编译产物（见 ``_run_one``），钩子、device、stream 与地址逐次核验。

本轮唯一性能机制：16 位输入（fp16/bf16）直接在自身精度上取对数。
  * 上一轮完整评测（源码 69699663，几何为 40 program × 相邻两 8192 载荷）：
    geo_mean=0.7614，其中 fp32 1.6e7 元素为 1.344，而 fp16 0.4345、bf16
    0.6272。三种 dtype 在相同元素数下的候选耗时几乎相同（37.66/40.38/37.55
    微秒），而参考 fp16 只需 16.36 微秒：候选受「每元素算术」限制（约 4.4e11
    元素/秒），并未到达访存上限；参考 fp16 已接近该规模的数据搬移下限。
  * 因此本轮只改 ``_log10_math``：不再把输入升到 FP32 再算再降回，16 位输入
    直接送入 ``tl.log``。这样省掉两次宽度转换，并让向量通道按 16 位宽度处理
    两倍元素；FP32 输入路径的数学与几何完全不变（仍是已验证的
    ``tl.log(x) * 0.4342944819032518``）。
  * 精度依据：合同对 16 位取 atol=rtol=0.015，16 位对数自身的舍入误差约
    1e-3 量级，余量充足；特殊值、负值与别名语义由同一内核逐元素保持。
  * 未知：本后端是否真的为 16 位 ``math.log`` 生成窄位宽序列（若前端或后端
    仍内部升到 FP32，则耗时与上一轮相同，不会变差）；16 位对数对次正规数、
    ±inf 与 NaN 的处理需由预检的特殊值检查确认。
"""

import torch
import triton
import triton.language as tl

# 平坦连续路径：中等输入用一个 program 单块直接处理的元素上限。取 16384 使
# fp32 载荷 64KB、fp16/bf16 转 fp32 后的中间量也是 64KB，与成对路径的单轮
# 载荷同量级，UB 仍有充足余量（设备可用 1572864 bits = 192KB）。
_TILE_MAX = 16384

# 平坦连续路径：大输入的相邻配对子块大小。相邻两块相距 BLOCK，一轮共
# 2*BLOCK 元素；8192 与历史已验证的成对载荷一致。
_PAIR_BLOCK = 8192

# 有限网格的 program 数上限。参考 CANN 实现的 Block Dim 为 40（与设备向量核数
# 一致），此处取 40，使大输入以单波次长循环推进。
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


# 位运算常量全部以字面量写进内核体，不使用模块级全局：本后端 JIT 只允许
# 内核访问 ``tl.constexpr`` 实例化的全局，普通 int/float 全局会在编译期抛
# NameError（Cannot access global variable ...）。换底常数同理写成字面量，
# 既避开该限制，也不会进入嵌套 JIT 的全局值依赖哈希。


@triton.jit
def _log10_math(x):
    """log10 = ln(x) * (1/ln(10))，在输入自身精度上计算。

    FP32 输入：与此前已完整通过 36 项正式用例及双方各 66 项补测的数学一致
    （``tl.log`` 后乘换底常数）。
    16 位输入（fp16/bf16）：不再升到 FP32 再降回，直接对 16 位张量取对数，
    去掉两次宽度转换并让向量通道一次处理两倍元素。合同对 16 位给出的
    atol=rtol=0.015 覆盖 16 位对数自身的舍入误差。
    """
    return tl.log(x) * 0.4342944819032518



@triton.jit
def _log10_tile_kernel(x_ptr, n_elements, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    """中等连续输入：单个 program 用一块直接覆盖全部元素，无循环、无尾链。

    ``BLOCK`` 与 ``EVEN`` 都是编译期常量：整块整除时既无动态掩码也无比较，
    只剩一次连续 load、一次对数、一次连续 store。
    """
    index = tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + index)
        y = _log10_math(x)
        tl.store(x_ptr + index, y.to(x.dtype))
    else:
        mask = index < n_elements
        x = tl.load(x_ptr + index, mask=mask, other=1.0)
        y = _log10_math(x)
        tl.store(x_ptr + index, y.to(x.dtype), mask=mask)


@triton.jit
def _log10_pair_kernel(
    x_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    ROUNDS: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    """大连续输入：有限网格 + 相邻成对预读，轮数／步长全部为编译期常量。

    第 p 个 program 的块序列是 ``2p, 2p+1, 2p+2*GRID, 2p+2*GRID+1, ...``，
    即每轮相邻取两块（相距一个 BLOCK），先把两次 load 都发出再统一计算与写回。
    ``ROUNDS = FULL // (2*GRID)``（``FULL = n_elements // BLOCK`` 为真正的满块数）
    由宿主传入，因此主循环体里没有任何掩码、比较或辅助缓冲，循环界限与步长
    也都是编译期常量。用满块数而不用 ``BLOCKS = ceil(n/BLOCK)`` 是关键：向上取整
    的总块数可能含一个不满的尾块，按它排无掩码轮次会越界。
    """
    pid = tl.program_id(axis=0)
    index = tl.arange(0, BLOCK)
    span = 2 * GRID * BLOCK
    base = 2 * pid * BLOCK
    for round_index in tl.range(0, ROUNDS):
        offset = base + round_index * span
        ptrs0 = x_ptr + offset + index
        ptrs1 = ptrs0 + BLOCK
        x0 = tl.load(ptrs0)
        x1 = tl.load(ptrs1)
        tl.store(ptrs0, _log10_math(x0).to(x0.dtype))
        tl.store(ptrs1, _log10_math(x1).to(x1.dtype))
    # 残余块：主循环只覆盖 2*ROUNDS*GRID 个「完整满块」，其后每 program 至多
    # 再负责两块（不足 2*GRID 块），块下标 b = 2*(pid + ROUNDS*GRID) 与 b+1，
    # 与主循环的块序列自然衔接、互不重叠。越界块与最后一个不满块的越界元素
    # 都由同一个掩码排除，不用标量分支。
    residual = 2 * (pid + ROUNDS * GRID)
    for extra in tl.static_range(2):
        block_index = residual + extra
        offsets = block_index * BLOCK + index
        mask = (block_index < BLOCKS) & (offsets < n_elements)
        x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
        tl.store(x_ptr + offsets, _log10_math(x).to(x.dtype), mask=mask)


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
        y = _log10_math(x)
        tl.store(ptrs, y.to(x.dtype))
    else:
        mask = rel < n_elements
        ptrs = x_ptr + (base_off + rel) * step
        x = tl.load(ptrs, mask=mask, other=1.0)
        y = _log10_math(x)
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
        y = _log10_math(x)
        tl.store(ptrs, y.to(x.dtype))
    else:
        mask = index < inner
        x = tl.load(ptrs, mask=mask, other=1.0)
        y = _log10_math(x)
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
    """建立并缓存一次调用的发射计划；本次调用同时完成恰好一次原地计算。

    计划的键覆盖 dtype、shape、stride、16 字节对齐类与 device，因此 Triton
    的全部编译期特化维度（dtype、指针对齐、整数的 ==1／%16／宽度）都已被
    键固定，复用编译产物不会串用别的特化二进制；device 进入键保证句柄不跨
    设备复用。

    每个操作这里都走普通 JIT 调度一次，这一次调用已经完成了该操作在设备上
    的原地计算；因此调用方在本函数返回后不得再发射同一批操作，否则冷启动会
    计算两次。编译／调试模式在这里由 ``_make_op`` 逐个排除后登记：非普通
    模式的产物不会进入直发表，热路径因此无需重读环境变量。
    """
    _resolve_stream_hooks()
    specs, recurse = _plan_specs(x)
    ops = []
    for jit_fn, grid, tail in specs:
        # 普通 JIT 调度一次：完成编译与唯一一次发射，返回本次使用的编译产物。
        compiled = jit_fn[grid](x, *tail)
        ops.append(_make_op(compiled, jit_fn, grid, tail))
    enter, exit_hook = _plan_gates()
    device = _FAST_DEVICE()
    plan = (device, enter, exit_hook, tuple(ops), recurse)
    _PLANS[key] = plan
    return plan


def _emit_ops(ops, x):
    """普通调度发射：逐操作完整走当前 JIT 调度路径。

    该路径每次重新解析当前设备、当前 stream、钩子与编译／调试模式，因此
    活动回调、模式变化和 compile-only 都能被真实尊重。绝不在这里改调原生
    入口，否则钩子分支会悄悄绕过采集。
    """
    for op in ops:
        op[7][op[8]](x, *op[6])


def _demote_plan(key, plan):
    """宿主侧 ABI 不匹配时把该计划整体降级为普通 JIT 调度。

    只在实参解析阶段失败（TypeError/AttributeError，设备任务尚未下发）时
    调用；此时原地计算未发生，退回普通调度补齐不会重复计算。
    """
    ops = tuple((None,) + op[1:] for op in plan[3])
    _PLANS[key] = (plan[0], plan[1], plan[2], ops, plan[4])


def _plan_key(x, ptr):
    """计划键：dtype、shape、stride、16 字节对齐类与 device。

    ``ptr`` 由调用方按本次调用查询一次后传入，键里只保留对齐类（``ptr & 15``），
    不保存地址本身，因此同一布局、同一对齐类的后续调用仍能命中同一计划。
    """
    return (x.dtype, x.shape, x.stride(), ptr & 15, x.device)


def _run_one(x):
    """单张量的热路径：计划命中即直发，否则建计划或退回普通调度。

    本次调用只查询一次数据地址（用于计划键的对齐类，同时作为热路径整数
    地址实参），不缓存地址本身。
    """
    ptr = x.data_ptr()
    key = _plan_key(x, ptr)
    plan = _PLANS.get(key)
    if plan is None:
        # 建计划的那次普通 JIT 调用已经把本次原地计算做完，这里直接返回，
        # 保证冷启动恰好计算一次；只有「无发射操作、需递归展开」的计划例外。
        plan = _build_plan(x, key)
        if plan[3]:
            return
    ops = plan[3]
    if not ops:
        if plan[4]:
            for index in range(x.shape[0]):
                _run_one(x[index])
        return
    # 钩子门逐次核验：任一不满足即完整走普通 JIT 调度。编译／调试模式不在
    # 这里重读：``compile_only`` 与 ``enable_msprof_register_tensor`` 是
    # ``NPULauncher.__init__`` 期间由环境变量固定的实例常量，``debug`` 来自
    # 该产物自身的 metadata，普通 JIT 调度对已编译产物同样不再重读；非普通
    # 模式产物在建计划时已被 ``_make_op`` 排除，不会进入直发表。
    if plan[1].calls or plan[2].calls or len(ops) != 1 or ops[0][0] is None:
        _emit_ops(ops, x)
        return
    op = ops[0]
    try:
        # 设备指针实参改传「本次调用刚查询到的整数地址」：本机生成的启动器
        # getPointer 对 PyLong 分支直接解析指针，而 Tensor 分支每次都要做
        # PyObject_GetAttrString("data_ptr") + 调用 + aclrtPointerGetAttributes；
        # 计划键已用同一 ptr 固定了 16 字节对齐类，产物特化因此不变。
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
            ptr,
            *op[6],
        )
    except (TypeError, AttributeError):
        # 宿主侧实参解析失败：设备任务尚未下发，原地计算未发生。
        _demote_plan(key, plan)
        _emit_ops(_PLANS[key][3], x)


def _flat_specs(n_elements):
    """平坦连续路径的发射规格：按规模在「单块直处理」与「有限网格成对循环」间分流。

    规模不超过 _TILE_MAX 时只发一个 program，``BLOCK`` 取覆盖全部元素的 2 的幂，
    整除时 ``EVEN`` 为编译期真，内核里没有掩码与循环；否则按 _PAIR_BLOCK 切成
    块，program 数压到 _PROGRAMS 以内，每 program 每轮相邻两块。

    无掩码轮数只按真正的满块数 ``full = n_elements // block`` 计算
    （``rounds = full // (2*grid)``），因此主循环覆盖的块严格落在满块范围内；
    ``blocks - 2*grid*rounds <= 2*grid`` 保证其后每个 program 的两块残余覆盖
    剩余块（含唯一的可能不满尾块），由内核掩码处理，二者互斥且合计完整。
    """
    if n_elements <= _TILE_MAX:
        block = triton.next_power_of_2(n_elements)
        if block < 2:
            block = 2
        even = (n_elements % block) == 0
        return [(_log10_tile_kernel, (1,), (n_elements, block, even))]
    block = _PAIR_BLOCK
    blocks = (n_elements + block - 1) // block
    full = n_elements // block
    # 每 program 每轮两块，故 program 数取块数的一半，且不超过 _PROGRAMS。
    grid = (blocks + 1) // 2
    if grid > _PROGRAMS:
        grid = _PROGRAMS
    rounds = full // (2 * grid)
    return [
        (_log10_pair_kernel, (grid,), (n_elements, block, grid, rounds, blocks))
    ]


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

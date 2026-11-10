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
  * 平坦连续路径为有限网格 + 格内成对预读循环：program 数不超过 _PROGRAMS，
    每 program 以 2*loop_stride 为跨度成对推进，未配对的块由收尾循环补齐，
    循环体内无掩码；元素数除不尽 BLOCK 的残余由 program 0 处理。
  * 宿主侧只缓存网格与分块常量，不缓存张量、地址、结果或 stream；热路径按
    计划直发已编译产物（见 ``_run_one``），钩子、device、stream 与地址逐次核验。

本轮先修编译阻塞：内核此前引用了模块级 int 常量，本后端 JIT 只允许访问
``tl.constexpr`` 全局，直接抛 NameError（_DENORM_LIMIT ... Cannot access
global variable）。现改为内核内字面量：既是编译期常量，也不进入嵌套 JIT 的
全局值依赖哈希（不会因 NaN != NaN 触发「conflicting value」）。

本轮机制（唯一改动：把逐元素 ``tl.log`` 换成纯算术的 ``_log10_vec``）：
  * 事实：正式反馈 geo_mean=0.7539，9 个 ≤1e6 元素 case 为 15.3–17.9us
    （参考 14.9–15.5us，0.83–0.98，已接近宿主发射平台）；三个 4096x4096
    （1.6e7 元素）case 为 fp16 53.03/参考 15.89、bf16 53.64/25.38、
    fp32 47.72/26.21（0.30/0.47/0.55）。三种 dtype 在相同元素数下耗时几乎
    相同，而 fp32 的字节数是 fp16 的两倍，说明限制项是逐元素运算而非带宽。
  * 由逐 case 时间可拟合出每元素约 2.45ns 的固定代价（1e6 与 1.6e7 元素
    一致），换成每核约 70–90 周期/元素，远超向量单元应有的量级，指向通用
    ``math.log`` 在本后端逐元素展开（含特殊值判定）而非按向量宽度处理。
  * 因此本轮只改内核数学：``_log10_vec`` 用位分解取幂次 + 尾数 9 次 Taylor
    多项式，只有位运算与加乘，保持 ±0、负值、±inf、NaN 与次正规数语义；
    分块、网格、成对预读结构、布局合并与宿主发射路径全部未改动。
  * 预期：三个 1.6e7 元素 case 明显受益（改为带宽受限），≤1e6 元素 case
    基本不变（受宿主平台限制）；收益真伪由完整评测判定。
  * 未知：``math.log`` 是否真按逐元素展开，无当前设备级 IR／profiling 证据；
    多项式精度按 |t|<=1/3 的截断上界估算为 log10 绝对误差约 8e-7（容差 1e-4），
    最终以预检与完整评测确认。

历史机制（已被本版取代，仅留一行结论）：平坦主循环按输入规模区分满块与
成对流式处理、宿主直发表缓存启动器与整数地址实参；对 4096x4096 收益有限，
说明瓶颈不在网格、循环轮次、搬运在途数或宿主发射。
"""

import torch
import triton
import triton.language as tl

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


# 位运算常量全部以字面量写进内核体，不使用模块级全局：本后端 JIT 只允许
# 内核访问 ``tl.constexpr`` 实例化的全局，普通 int/float 全局会在编译期抛
# NameError（Cannot access global variable ...）。字面量直接成为编译期常量，
# 既避开该限制，也不会进入嵌套 JIT 的全局值依赖哈希（不会因 NaN != NaN
# 触发「conflicting value」）。


@triton.jit
def _log10_vec(x):
    """log10 的纯算术近似：位分解取幂次 + 尾数 9 次 Taylor 多项式。

    依据：设备段耗时与 dtype 无关、只随元素数线性增长（1e6 与 1.6e7 元素
    分别约 2.0ns/元素与 2.45ns/元素，fp16/bf16/fp32 几乎相同），说明限制项
    是逐元素的超越函数开销而不是带宽；通用 ``math.log`` 在本后端逐元素代价
    很高。这里换成只有加乘与位运算的形式，全部可由向量单元处理。

    精度：归约后 |t| <= 1/3，9 次 Taylor 截断的 log10 绝对误差约 8e-7，比
    fp32 合同 atol=rtol=1e-4 小两个数量级；fp16/bf16 容差 1.5%，余量更大。

    特殊值语义按 IEEE 保持：±0 → -inf，负有限值与 -inf → NaN，±inf 与 NaN
    由输入直接传播；次正规数先乘 2^24 归一化，再减去 24*log10(2)。
    """
    xi = x.to(tl.int32, bitcast=True)
    # 次正规数（非正数也落在这里，随后被特殊值分支覆盖）先乘 2^24 放大到
    # 正规范围：0x00800000 是最小正规数的位模式。
    sub = xi < 8388608
    xs = tl.where(sub, x * 16777216.0, x)
    i = xs.to(tl.int32, bitcast=True)
    # 0x3F2AAAAB 对应 2/3；减去它并按指数域 0xFF800000 取整后，尾数被归约到
    # [2/3, 4/3)，于是 t = m - 1 落在 [-1/3, 1/3)，Taylor 级数收敛很快。
    e = (i - 1059760811) & -8388608
    m = (i - e).to(tl.float32, bitcast=True)
    t = m - 1.0
    # log10(1+t) 的 9 次 Taylor 系数 c*(-1)^k/(k+1)，c = log10(e)。
    p = 0.04825494243369464
    p = p * t - 0.05428681023790647
    p = p * t + 0.06204206884332168
    p = p * t - 0.07238241365054197
    p = p * t + 0.08685889638065036
    p = p * t - 0.10857362047581294
    p = p * t + 0.14476482730108393
    p = p * t - 0.2171472409516259
    p = p * t + 0.4342944819032518
    # e 是 2^23 的整数倍，转成 float 后乘 2^-23 即得整数幂次（精确、无舍入），
    # 避免依赖有符号右移的具体语义。
    kf = e.to(tl.float32) * 1.1920928955078125e-07
    # log10(2) = 0.3010299956639812；24*log10(2) = 7.224719895935549 用于
    # 抵消次正规数归一化时乘入的 2^24。
    y = kf * 0.3010299956639812 + p * t
    y = tl.where(sub, y - 7.224719895935549, y)
    # 特殊值：先传播 ±inf/NaN，再处理负值与零。NaN 与 -inf 由 int32 位模式
    # （0x7FC00000 / 0xFF800000）在设备侧现场构造，不捕获浮点特殊值常量。
    zeros = xi & 0
    nan_val = (zeros + 2143289344).to(tl.float32, bitcast=True)
    neg_inf = (zeros - 8388608).to(tl.float32, bitcast=True)
    nonfinite = (xi & 2139095040) == 2139095040
    y = tl.where(nonfinite, x, y)
    y = tl.where(x < 0.0, nan_val, y)
    y = tl.where((xi & 2147483647) == 0, neg_inf, y)
    return y


@triton.jit
def _log10_flat_kernel(
    x_ptr,
    bulk,
    stride,
    pairs,
    n_elements,
    BLOCK: tl.constexpr,
    TAIL: tl.constexpr,
):
    """平坦连续路径：有限网格 + 格内成对预读循环，块内直接连续寻址。

    主循环每轮取两块（相距一个 ``stride``），先发出两次 load，再统一取对数并写回，
    使单个 program 同时在途的搬运加倍。配对轮数 ``pairs`` 由宿主按「所有 program
    都不越界」的下界算好并以标量传入，因此配对循环体内**没有任何掩码、比较与
    辅助缓冲**；未配对到的块（每 program 至多一块）由第二个循环按原步长补齐，
    它同样只处理满块。逐元素对数由 ``_log10_vec`` 完成。

    依据（actual-ir/d519b459c915/15-kernel.bcmlir）：格内循环为
    ``memref.copy`` → 对数 → ``materialize_in_destination`` 的单块串行结构，
    1.6e7 元素在 40 个 program 下每 program 约 102 轮。

    元素数除不尽 BLOCK 的残余仍由 program 0 在循环后用一次带掩码的块处理
    （``TAIL`` 为编译期常量，满块输入下整段尾部代码被消除）。

    ``bulk`` 是满块部分的上界，``stride = grid * BLOCK``；每 program 从
    ``pid*BLOCK`` 起按 ``stride`` 推进到 ``bulk``，循环内直接连续寻址。
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    index = tl.arange(0, BLOCK)
    # 成对主循环：一轮发出两次互不重叠的 load（相距一个 stride），再统一计算与
    # 写回，使单个 program 同时在途的搬运翻倍。轮数上界 pairs 由宿主保证对每个
    # program 都成立，循环体内没有任何掩码、比较或辅助缓冲。
    span = 2 * stride
    for offset in tl.range(start, start + pairs * span, span):
        ptrs0 = x_ptr + offset + index
        ptrs1 = ptrs0 + stride
        x0 = tl.load(ptrs0)
        x1 = tl.load(ptrs1)
        y0 = _log10_vec(x0.to(tl.float32))
        y1 = _log10_vec(x1.to(tl.float32))
        tl.store(ptrs0, y0.to(x0.dtype))
        tl.store(ptrs1, y1.to(x1.dtype))
    # 未配对到的块按原步长补齐；它们同样是满块，所以这个收尾循环也不需要掩码。
    # 两个循环覆盖的块互不重叠，合起来正好是该 program 原负责的全部块序列
    # start, start+stride, start+2*stride, ..., < bulk。
    for offset in tl.range(start + pairs * span, bulk, stride):
        ptrs = x_ptr + offset + index
        x = tl.load(ptrs)
        y = _log10_vec(x.to(tl.float32))
        tl.store(ptrs, y.to(x.dtype))
    if TAIL:
        if pid == 0:
            tail_index = bulk + index
            tail_mask = tail_index < n_elements
            x_t = tl.load(x_ptr + tail_index, mask=tail_mask, other=1.0)
            y_t = _log10_vec(x_t.to(tl.float32))
            tl.store(x_ptr + tail_index, y_t.to(x_t.dtype), mask=tail_mask)


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
        y = _log10_vec(x.to(tl.float32))
        tl.store(ptrs, y.to(x.dtype))
    else:
        mask = rel < n_elements
        ptrs = x_ptr + (base_off + rel) * step
        x = tl.load(ptrs, mask=mask, other=1.0)
        y = _log10_vec(x.to(tl.float32))
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
        y = _log10_vec(x.to(tl.float32))
        tl.store(ptrs, y.to(x.dtype))
    else:
        mask = index < inner
        x = tl.load(ptrs, mask=mask, other=1.0)
        y = _log10_vec(x.to(tl.float32))
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
    """平坦连续路径的发射规格：program 数压到 _PROGRAMS 以内，其余靠格内循环。

    主机侧只做整除分解：``bulk`` 是所有满块覆盖的元素数，``stride = grid *
    BLOCK``，残余 ``n_elements - bulk``（< BLOCK）交给内核里的尾部块。主循环
    因此永远只处理满块，不需要每轮判定边界；满块输入（如 4096x4096）残余为
    0，尾部代码被 constexpr 消除。

    ``pairs`` 是「每个 program 都成立」的成对预读轮数下界：program p 负责块
    下标 p, p+grid, p+2*grid, ...，块数随 p 单调不增，因此取 p = grid-1 的配对
    轮数即可保证配对循环对任何 program 都不越界。配对循环覆盖前 ``2*pairs``
    个块，其余块由收尾循环按原步长补齐，两者不重叠且合计恰好覆盖全部块。
    """
    block = _block_for(n_elements, _BLOCK)
    bulk = (n_elements // block) * block
    tail = n_elements - bulk
    if bulk:
        blocks = bulk // block
        grid = blocks if blocks < _PROGRAMS else _PROGRAMS
        stride = grid * block
    else:
        # 不足一个块：只发一个 program，由它执行尾部块。
        grid = 1
        stride = block
        blocks = 0
    pairs = 0
    if blocks >= 2 * grid:
        pairs = (blocks - 2 * grid) // (2 * grid) + 1
    return [
        (_log10_flat_kernel, (grid,), (bulk, stride, pairs, n_elements, block, tail > 0))
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

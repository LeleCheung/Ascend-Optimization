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

import torch
import triton
import triton.language as tl

# 严格 Triton 原地 log10：设备计算、index 变换与地址映射全部由 Triton 核完成，
# 不保留任何目标设备的 PyTorch 回退、contiguous 拷贝或 copy_ 回写。
# 数学路径与历史已验证机制一致：自然对数与换底乘法保持输入 dtype，不做显式 FP32 提升。

# 单个 program 每个 tile 的输入+输出缓冲合计预算（UB 为 192KB，留出索引/临时量余量）。
_TILE_BUDGET_BYTES = 131072
_LARGE_BLOCK = 8192      # 最大输入的单块载荷（每轮取相邻两个满块）
_LARGE_NPROG = 40        # 最大输入的有限网格 program 数
_STREAM_MIN_ELEMS = 1 << 20


@triton.jit
def _log10_flat(x_ptr, n_elements, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, tl.log(x) * 0.4342944819032518)
    else:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=1.0)
        tl.store(x_ptr + offs, tl.log(x) * 0.4342944819032518, mask=mask)


@triton.jit
def _log10_stream(x_ptr, NPAIR: tl.constexpr, GRID: tl.constexpr,
                  BLOCK: tl.constexpr, BASE_TAIL: tl.constexpr,
                  TAIL: tl.constexpr):
    # 有限网格成对流式：每轮取相邻两个满块（共 2*BLOCK 元素），两次 load 与两次
    # store 覆盖互斥地址，可同时进入流水；主循环无掩码。循环长度、步长与尾部起点
    # 均为编译期常量，避免逐轮动态取模/除法和掩码构造。
    pid = tl.program_id(axis=0)
    for p in range(pid, NPAIR, GRID):
        off0 = p * (2 * BLOCK) + tl.arange(0, BLOCK)
        x0 = tl.load(x_ptr + off0)
        x1 = tl.load(x_ptr + off0 + BLOCK)
        tl.store(x_ptr + off0, tl.log(x0) * 0.4342944819032518)
        tl.store(x_ptr + off0 + BLOCK, tl.log(x1) * 0.4342944819032518)
    if TAIL > 0:
        # 尾部只由 pid 0 处理，与主循环覆盖互斥；TAIL < 2*BLOCK。
        if pid == 0:
            local = tl.arange(0, 2 * BLOCK)
            offs = BASE_TAIL + local
            mask = local < TAIL
            xt = tl.load(x_ptr + offs, mask=mask, other=1.0)
            tl.store(x_ptr + offs, tl.log(xt) * 0.4342944819032518, mask=mask)


@triton.jit
def _log10_stride1(x_ptr, n_elements, stride, BLOCK: tl.constexpr):
    # 维度合并后只剩一个非单位步长维度：直接以标量步长散列寻址，兼容 offset 视图。
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    ptrs = x_ptr + offs * stride
    x = tl.load(ptrs, mask=mask, other=1.0)
    tl.store(ptrs, tl.log(x) * 0.4342944819032518, mask=mask)


@triton.jit
def _log10_nd(x_ptr, n_elements, d0, d1, d2, d3, t0, t1, t2, t3,
              BLOCK: tl.constexpr):
    # 最多 4 个不可合并维度的通用寻址；缺失维度置 size=1、stride=0，不参与地址计算。
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    p1 = d1 * d2 * d3
    p2 = d2 * d3
    i0 = offs // p1
    r0 = offs - i0 * p1
    i1 = r0 // p2
    r1 = r0 - i1 * p2
    i2 = r1 // d3
    r2 = r1 - i2 * d3
    addr = i0 * t0 + i1 * t1 + i2 * t2 + r2 * t3
    ptrs = x_ptr + addr
    x = tl.load(ptrs, mask=mask, other=1.0)
    tl.store(ptrs, tl.log(x) * 0.4342944819032518, mask=mask)


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_PLAN_CACHE = {}

# ---- 编译启动器快速路径 ---------------------------------------------------
# 正式反馈中 12 个计时用例耗时几乎相同（约 66–72 微秒，随规模几乎不变的平台），
# 说明损失集中在宿主发射，而非设备算术。普通 JIT 调度在每次热调用中重复解析
# 参数、计算特化键、查询后端策略；这里在首次 JIT 调用（该次已执行过一次原地
# 计算）后登记当次实际使用的 CompiledKernel，热调用只做「布局／对齐判定 +
# 当前 stream + 当前整数地址 + 原生 launch」。
# 动态值（stream、地址）逐次读取，不缓存；编译产物缓存键含设备身份。
_FAST = {}
_FAST_LIMIT = 4096

try:
    from triton import knobs as _knobs
except Exception:
    _knobs = None

# 当前后端优先使用 NoWait 原始流（backend_register.get_current_stream 的同一合同）。
_RAW_STREAM = None
try:
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _RAW_STREAM
except Exception:
    try:
        from torch_npu._C import _npu_getCurrentRawStream as _RAW_STREAM
    except Exception:
        _RAW_STREAM = None

_DRV_GET_STREAM = None
if _RAW_STREAM is None:
    try:
        from triton.backends.ascend.driver import driver as _ascend_driver
    except Exception:
        _ascend_driver = None
    if _ascend_driver is not None:
        try:
            _DRV_GET_STREAM = _ascend_driver.active.get_current_stream
        except Exception:
            _DRV_GET_STREAM = None


def _idle_hooks():
    # 返回 (enter, exit) 供原生 launch 使用；存在活动回调时返回 None，调用方必须
    # 退回普通 JIT。空 HookChain 容器不等于没有钩子：只有确认 calls 为空列表才
    # 视为无回调，此时传 None 与传空链等价；其它形态一律按存在钩子处理。
    if _knobs is None:
        return None
    rt = _knobs.runtime
    enter = rt.launch_enter_hook
    leave = rt.launch_exit_hook
    if enter is not None:
        calls = getattr(enter, "calls", None)
        if not isinstance(calls, list) or calls:
            return None
        enter = None
    if leave is not None:
        calls = getattr(leave, "calls", None)
        if not isinstance(calls, list) or calls:
            return None
        leave = None
    return (enter, leave)


def _fast_key(x, n, ptr):
    # 键覆盖设备、dtype、numel 与实际指针对齐类（编译特化按 16 字节对齐判定）。
    return (x.get_device(), x.dtype, n, (ptr & 15) == 0)


def _register_fast(x, n, block, even, kernel, ptr):
    # 只登记普通启动器产物：排除 compile-only、张量注册与调试隐藏参数模式。
    if kernel is None or len(_FAST) >= _FAST_LIMIT:
        return
    try:
        launcher = getattr(kernel, "run", None)
        if launcher is None:
            launcher = kernel._run
        launch = launcher.launch
        function = kernel.function
        packed = kernel.packed_metadata
        if function is None or packed is None:
            return
        if getattr(launcher, "compile_only", False):
            return
        if getattr(launcher, "enable_msprof_register_tensor", False):
            return
        if bool(getattr(getattr(kernel, "metadata", None), "debug_enabled", False)):
            return
        grid = triton.cdiv(n, block)
    except Exception:
        return
    if _RAW_STREAM is None and _DRV_GET_STREAM is None:
        return
    _FAST[_fast_key(x, n, ptr)] = (launch, function, packed, grid, block, even)


def _fast_flat(x, n):
    # 平坦连续热调用：布局身份（连续、numel、对齐、设备、dtype）完全命中才发射。
    ptr = x.data_ptr()
    plan = _FAST.get(_fast_key(x, n, ptr))
    if plan is None:
        return False
    hooks = _idle_hooks()
    if hooks is None:
        return False
    dev = x.get_device()
    if _RAW_STREAM is not None:
        stream = _RAW_STREAM(dev)
    else:
        stream = _DRV_GET_STREAM(dev)
    launch, function, packed, grid, block, even = plan
    launch(grid, 1, 1, stream, function, packed, None, hooks[0], hooks[1],
           ptr, n, block, even)
    return True


def _tile_cap(esize):
    cap = _TILE_BUDGET_BYTES // (2 * esize)
    if cap < 256:
        cap = 256
    return triton.next_power_of_2(cap)


def _flat_plan(n, esize):
    # 按输入规模分流：中等输入单块大载荷直取，最大输入转为有限网格流式循环。
    key = (n, esize)
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    cap = _tile_cap(esize)
    if n <= cap:
        block = triton.next_power_of_2(n)
        if block < 256:
            block = 256
        if block > cap:
            block = cap
        plan = ("flat", block, triton.cdiv(n, block))
    elif n <= _STREAM_MIN_ELEMS:
        plan = ("flat", cap, triton.cdiv(n, cap))
    else:
        # 最大输入：单块载荷按「两个相邻块」的 UB 预算收缩（load+store 各 2 块）。
        block = _TILE_BUDGET_BYTES // (4 * esize)
        if block > _LARGE_BLOCK:
            block = _LARGE_BLOCK
        if block < 2048:
            block = 2048
        plan = ("stream", block, _LARGE_NPROG)
    if len(_PLAN_CACHE) < 4096:
        _PLAN_CACHE[key] = plan
    return plan


def _collapse_dims(shape, stride):
    # 只做元数据层的连续维度合并；size==1 的维度不参与寻址。
    merged = []
    for size, st in zip(shape, stride):
        if size == 1:
            continue
        if merged:
            outer = merged[-1]
            if outer[1] == st * size:
                # 合并后的连续块以最内层步长为准：shape[(2,3),(3,1)]→[[6,1]]，
                # 若保留外层步长 3 会按 0,3,6,... 越界寻址。
                outer[0] *= size
                outer[1] = st
                continue
        merged.append([size, st])
    return merged


def _launch_flat(x, n, esize):
    kind, block, grid = _flat_plan(n, esize)
    if kind == "flat":
        even = (n % block == 0)
        # 普通 JIT 调用返回本次实际使用的 CompiledKernel；冷调用已执行一次计算，
        # 返回后只登记、不重发。
        kernel = _log10_flat[(grid,)](x, n, BLOCK=block, EVEN=even)
        _register_fast(x, n, block, even, kernel, x.data_ptr())
    else:
        # 相邻双块流式：主循环覆盖前 NPAIR*2*BLOCK 个元素，尾部剩余元素单独带掩码。
        pair = 2 * block
        npair = n // pair
        tail = n - npair * pair
        _log10_stream[(grid,)](x, NPAIR=npair, GRID=grid, BLOCK=block,
                               BASE_TAIL=npair * pair, TAIL=tail)


def _launch(x):
    n = x.numel()
    if n == 0:
        return x
    # 连续输入的常见路径先做连续性判定，命中即走已登记的编译启动器，
    # 不再每次重建布局合并结果与特化字符串；未命中仍走普通 JIT。
    if x.is_contiguous() and _fast_flat(x, n):
        return x
    esize = x.element_size()
    merged = _collapse_dims(x.shape, x.stride())
    if not merged:
        # numel==1 的全部维度均为 1：按单元素平坦路径处理。
        _launch_flat(x, 1, esize)
        return x
    if len(merged) == 1:
        size, st = merged[0]
        if st == 1:
            _launch_flat(x, size, esize)
        else:
            cap = _tile_cap(esize)
            block = cap if size > cap else max(256, triton.next_power_of_2(size))
            _log10_stride1[(triton.cdiv(size, block),)](x, size, st, BLOCK=block)
        return x
    if len(merged) > 4:
        # 合并维与原始维不再一一对应（size==1 的维被跳过、相邻维被折叠），
        # 因此不能按 merged[0][0] 去切原始 x[i]。改为挑选第一个原始长度 >1 的轴
        # 做纯元数据切片，沿该轴每次调用都使 numel 严格减小，递归必然收敛。
        axis = -1
        for i in range(x.dim()):
            if x.shape[i] > 1:
                axis = i
                break
        if axis < 0:
            return x
        for i in range(x.shape[axis]):
            _launch(x.select(axis, i))
        return x
    dims = [1, 1, 1, 1]
    strides = [0, 0, 0, 0]
    for i, (size, st) in enumerate(merged):
        dims[i] = size
        strides[i] = st
    cap = _tile_cap(esize)
    block = cap if n > cap else max(256, triton.next_power_of_2(n))
    _log10_nd[(triton.cdiv(n, block),)](x, n, dims[0], dims[1], dims[2], dims[3],
                                        strides[0], strides[1], strides[2], strides[3],
                                        BLOCK=block)
    return x


def run(A):
    if not isinstance(A, torch.Tensor):
        raise TypeError("run expects a torch.Tensor as its first argument.")
    if A.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("run supports float16/bfloat16/float32/float64 tensors.")
    if A.numel() == 0:
        return A
    return _launch(A)

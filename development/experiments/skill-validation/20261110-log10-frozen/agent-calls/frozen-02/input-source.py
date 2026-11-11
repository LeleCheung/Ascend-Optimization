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
_LARGE_BLOCK = 8192      # 最大输入的分块
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
def _log10_stream(x_ptr, n_elements, NPROG: tl.constexpr, BLOCK: tl.constexpr):
    # 有限网格循环：每 program 处理满块（无掩码），尾部不足一块的元素由唯一 program 带掩码处理。
    pid = tl.program_id(axis=0)
    num_full = n_elements // BLOCK
    for blk in range(pid, num_full, NPROG):
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, tl.log(x) * 0.4342944819032518)
    tail_blk = num_full
    if tail_blk * BLOCK < n_elements:
        if pid == tail_blk % NPROG:
            offs = tail_blk * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=1.0)
            tl.store(x_ptr + offs, tl.log(x) * 0.4342944819032518, mask=mask)


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
        plan = ("stream", _LARGE_BLOCK, _LARGE_NPROG)
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
                outer[0] *= size
                continue
        merged.append([size, st])
    return merged


def _launch_flat(x, n, esize):
    kind, block, grid = _flat_plan(n, esize)
    if kind == "flat":
        _log10_flat[(grid,)](x, n, BLOCK=block, EVEN=(n % block == 0))
    else:
        _log10_stream[(grid,)](x, n, NPROG=grid, BLOCK=block)


def _launch(x):
    n = x.numel()
    if n == 0:
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
        # 超过 4 维时按最外层逐片调用（仅创建元数据视图，无数据搬运）。
        for i in range(merged[0][0]):
            _launch(x[i])
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

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
  * 宿主侧只缓存网格与分块常量，不缓存张量、地址、结果或 stream。
"""

import torch
import triton
import triton.language as tl

# 1 / ln(10)。内核中以字面量内联，避免每次启动多传一个标量参数。
_LOG10E = 0.4342944819032518

# 单 program 元素数。历史设备异常出现在更大分块上（UB 多缓冲放大），
# 这里固定为对 fp16/bf16/fp32 都安全的 2048；小输入按实际规模降到 2 的幂。
_BLOCK = 2048

# 单次发射的 grid 轴上限。
_GRID_LIMIT = 65535


@triton.jit
def _log10_flat_kernel(
    x_ptr,
    base_off,
    n_elements,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """平坦连续路径：存储顺序与逻辑顺序一致，块内直接连续寻址。

    n_elements 是本段长度，掩码按段内相对下标比较；base_off 是本段在
    平坦存储中的起始偏移（分段发射时推进）。
    """
    pid = tl.program_id(axis=0)
    rel = pid * BLOCK + tl.arange(0, BLOCK)
    offsets = base_off + rel
    if EVEN:
        x = tl.load(x_ptr + offsets)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(x_ptr + offsets, y.to(x.dtype))
    else:
        mask = rel < n_elements
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


def _block_for(n_elements):
    """按实际规模选择分块：大输入用固定 2048，小输入降到 2 的幂。"""
    if n_elements >= _BLOCK:
        return _BLOCK
    block = triton.next_power_of_2(n_elements)
    if block < 2:
        block = 2
    return block


def _launch_flat(x, n_elements, base_off):
    block = _block_for(n_elements)
    blocks = (n_elements + block - 1) // block
    if blocks <= _GRID_LIMIT:
        _log10_flat_kernel[(blocks,)](
            x, base_off, n_elements, block, (n_elements % block) == 0
        )
        return
    # 超大输入按轴上限分段发射，段内偏移由标量 base_off 推进。
    span = _GRID_LIMIT * block
    start = 0
    while start < n_elements:
        count = n_elements - start
        if count > span:
            count = span
        chunk_blocks = (count + block - 1) // block
        _log10_flat_kernel[(chunk_blocks,)](
            x, base_off + start, count, block, (count % block) == 0
        )
        start += count


def _launch_uniform(x, n_elements, step):
    block = _block_for(n_elements)
    blocks = (n_elements + block - 1) // block
    if blocks <= _GRID_LIMIT:
        _log10_uniform_kernel[(blocks,)](
            x, 0, n_elements, step, block, (n_elements % block) == 0
        )
        return
    span = _GRID_LIMIT * block
    start = 0
    while start < n_elements:
        count = n_elements - start
        if count > span:
            count = span
        chunk_blocks = (count + block - 1) // block
        _log10_uniform_kernel[(chunk_blocks,)](
            x, start, count, step, block, (count % block) == 0
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
    block = _block_for(inner)
    blocks_inner = (inner + block - 1) // block
    even = (inner % block) == 0
    start = 0
    while start < outer:
        count = outer - start
        if count > _GRID_LIMIT:
            count = _GRID_LIMIT
        _log10_rows_kernel[(blocks_inner, count)](
            x,
            dim1,
            dim2,
            stride0,
            stride1,
            stride2,
            inner,
            inner_stride,
            start,
            block,
            even,
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
            _launch_flat(x, n_elements, 0)
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

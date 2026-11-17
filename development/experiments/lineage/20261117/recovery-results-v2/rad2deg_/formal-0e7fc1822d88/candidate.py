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

"""严格Triton原地弧度转角度。保留CC第三轮分块与精简启动；
Codex将冷启动改为只编译后执行一次，隔离设备缓存并保留profiling回退。
非连续输入仍按真实stride在Triton内读写；当前stream逐次查询。
数学语义：先在fp32中乘常数再转回输入dtype（与上一版0.9625x起点一致）；
16位输入的bf16/fp16原生乘法在设备上反而更慢，已回退该原生dtype方案。
"""

import torch
import triton
import triton.language as tl

try:  # 10x cheaper than the Triton driver wrapper, same value (raw rtStream_t)
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _get_stream
except Exception:  # pragma: no cover
    _get_stream = None

if _get_stream is None:  # pragma: no cover
    def _get_stream(device):
        return triton.runtime.driver.active.get_current_stream(device)

# 180.0 / pi, matching the reference multiply constant.
_INV = 57.29577951308232

# Tile size per size class. A tile has to be large enough that one program's
# prologue/epilogue is amortised over real data movement, but a tiny input
# should not be spread over thousands of masked lanes.
_BLOCK_SMALL = 8192
_BLOCK_LARGE = 32768
_N_LARGE = 131072
_MAX_I32 = 0x7FFFFFFF

# 单一计划缓存：按设备、dtype、指针对齐、标量特化和尺寸类缓存编译产物与
# 宿主发射所需的不变元数据（launch/function/packed_metadata/block）。
# 只缓存不可变身份；不缓存张量、指针或stream，这些每次动态查询。
_plans = {}


@triton.jit(do_not_specialize=["n_elements"])
def _rad2deg_contig_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    # Multiply in fp32 then cast back (original precision semantics).
    y = (x.to(tl.float32) * 57.29577951308232).to(x.dtype)
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit(do_not_specialize=["n_elements"])
def _rad2deg_strided_kernel(
    x_ptr,
    n_elements,
    s0, s1, s2, s3, s4, s5, s6, s7,
    t0, t1, t2, t3, t4, t5, t6, t7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Generic N-d (NDIM <= 8) in-place scaling over a strided view.

    A linear index over the logical element order is decomposed into the
    multi-index (last axis fastest) and recombined with the storage strides,
    so the write lands on exactly the same element that was read.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    rem = idx
    off = tl.zeros_like(idx)

    if NDIM >= 1:
        off += (rem % s7) * t7
        rem = rem // s7
    if NDIM >= 2:
        off += (rem % s6) * t6
        rem = rem // s6
    if NDIM >= 3:
        off += (rem % s5) * t5
        rem = rem // s5
    if NDIM >= 4:
        off += (rem % s4) * t4
        rem = rem // s4
    if NDIM >= 5:
        off += (rem % s3) * t3
        rem = rem // s3
    if NDIM >= 6:
        off += (rem % s2) * t2
        rem = rem // s2
    if NDIM >= 7:
        off += (rem % s1) * t1
        rem = rem // s1
    if NDIM >= 8:
        off += (rem % s0) * t0

    x = tl.load(x_ptr + off, mask=mask)
    # Same fp32-then-cast arithmetic as the contiguous kernel.
    y = (x.to(tl.float32) * 57.29577951308232).to(x.dtype)
    tl.store(x_ptr + off, y, mask=mask)


def _install(a, n, block, key):
    # warmup只编译，不提交原地运算；失败后走一次普通Triton启动。
    compiled = _rad2deg_contig_kernel.warmup(a, n, BLOCK=block, grid=(1,))
    launcher = compiled.run
    metadata = launcher.metadata
    if (getattr(launcher, 'compile_only', False)
            or getattr(launcher, 'enable_msprof_register_tensor', False)
            or getattr(metadata, 'debug_enabled', False)
            or getattr(metadata, 'debug', False)):
        return None
    plan = (launcher.launch, compiled.function, compiled.packed_metadata, block)
    _plans[key] = plan
    return plan


def _launch_strided(a, n, block=_BLOCK_SMALL):
    shape = list(a.shape)
    strides = list(a.stride())

    # Collapse adjacent axes whose strides compose exactly, so the ranks the
    # kernel has to unroll stay small. This is a pure index-space rewrite.
    i = len(shape) - 1
    while i > 0:
        if strides[i - 1] == strides[i] * shape[i] and shape[i] != 0:
            shape[i - 1] *= shape[i]
            strides[i - 1] = strides[i]
            del shape[i]
            del strides[i]
        i -= 1

    # Rank above the kernel's unroll width: peel the leading axes with zero-copy
    # views (never exercised by the contracted workloads, kept for generality).
    while len(shape) > 8:
        for k in range(a.shape[0]):
            sub = a[k]
            _launch_strided(sub, sub.numel(), block)
        return

    rank = len(shape)
    pad_s = [1] * 8
    pad_t = [0] * 8
    for d in range(rank):
        pad_s[8 - rank + d] = shape[d]
        pad_t[8 - rank + d] = strides[d]

    grid = ((n + block - 1) // block,)
    _rad2deg_strided_kernel[grid](
        a, n,
        pad_s[0], pad_s[1], pad_s[2], pad_s[3],
        pad_s[4], pad_s[5], pad_s[6], pad_s[7],
        pad_t[0], pad_t[1], pad_t[2], pad_t[3],
        pad_t[4], pad_t[5], pad_t[6], pad_t[7],
        NDIM=rank,
        BLOCK=block,
    )


def _run_impl(A, _plans=_plans, _plan_get=_plans.get, _stream=_get_stream):
    n = A.numel()
    if not A.is_contiguous():
        if n:
            _launch_strided(A, n)
        return A
    if not n:
        return A

    # 尺寸类决定分块与grid；同一计划缓存按该位区分，避免两套字典和分支。
    large = n > _N_LARGE
    block = _BLOCK_LARGE if large else _BLOCK_SMALL

    if n > _MAX_I32:
        _rad2deg_contig_kernel[((n + block - 1) // block,)](A, n, BLOCK=block)
        return A

    device = A.get_device()
    ptr = A.data_ptr()
    grid = (n + block - 1) // block
    key = (device, A.dtype, ptr & 15, n == 1, large)
    plan = _plan_get(key)
    if plan is None:
        try:
            plan = _install(A, n, block, key)
        except Exception:
            plan = None
    if plan is None:
        _rad2deg_contig_kernel[grid](A, n, BLOCK=block)
    else:
        # 每次查询输入设备的当前stream，不缓存stream或输入指针。
        launch, fn, meta, block = plan
        launch(grid, 1, 1, _stream(device), fn, meta,
               None, None, None, ptr, n, block)
    return A


def run(A):
    # 公开ABI严格匹配Definition：仅一个参数。内部默认绑定留在私有实现中。
    return _run_impl(A)

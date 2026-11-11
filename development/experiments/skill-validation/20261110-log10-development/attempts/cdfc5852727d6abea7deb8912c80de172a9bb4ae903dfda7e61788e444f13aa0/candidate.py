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

"""In-place log10 for Ascend, implemented strictly with Triton kernels.

Public entry point: ``run(A)``.  Every device computation, layout handling and
fallback path is executed by the Triton kernels below; no ``contiguous``,
``copy_`` or PyTorch math op performs any part of the in-place update.
"""

import torch
import triton
import triton.language as tl

# 1 / ln(10)
_LOG10E = 0.4342944819032518

# Per-program element tile.  Larger tiles overflow the UB once the backend
# applies its multi-buffer tiling (observed during preflight), so the tile is
# held at 1024 and large tensors are streamed with a per-program loop instead.
_TILE = 1024


@triton.jit
def _log10_contig_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr, ITERS: tl.constexpr):
    """Contiguous path: each program streams ITERS consecutive tiles."""
    pid = tl.program_id(axis=0)
    base = pid * (BLOCK_SIZE * ITERS)
    for k in range(ITERS):
        offsets = base + k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(x_ptr + offsets, y.to(x.dtype), mask=mask)


@triton.jit
def _log10_step_kernel(x_ptr, n_elements, stride0, BLOCK_SIZE: tl.constexpr):
    """Uniform positive/negative step path (one strided element per lane)."""
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    ptrs = x_ptr + idx * stride0
    x = tl.load(ptrs, mask=mask, other=1.0)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(ptrs, y.to(x.dtype), mask=mask)


@triton.jit
def _log10_2d_kernel(
    x_ptr,
    stride_m,
    stride_n,
    rows,
    cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (off_m[:, None] < rows) & (off_n[None, :] < cols)
    ptrs = x_ptr + off_m[:, None] * stride_m + off_n[None, :] * stride_n
    x = tl.load(ptrs, mask=mask, other=1.0)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(ptrs, y.to(x.dtype), mask=mask)


@triton.jit
def _log10_nd_kernel(
    x_ptr,
    stride0,
    stride1,
    stride2,
    dim1,
    dim2,
    n_last,
    stride_last,
    blocks_n,
    BLOCK_SIZE: tl.constexpr,
):
    """Rank 3/4 strided path: block the last dim, one scalar decode per program."""
    pid = tl.program_id(axis=0)
    pid_o = pid // blocks_n
    pid_n = pid % blocks_n
    base = (
        (pid_o // (dim1 * dim2)) * stride0
        + ((pid_o // dim2) % dim1) * stride1
        + (pid_o % dim2) * stride2
    )
    idx = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_last
    ptrs = x_ptr + base + idx * stride_last
    x = tl.load(ptrs, mask=mask, other=1.0)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(ptrs, y.to(x.dtype), mask=mask)


def _iters_for(n_elements):
    """Number of 1024-element tiles one program streams (keeps UB tile small)."""
    tiles = (n_elements + _TILE - 1) // _TILE
    if tiles <= 2048:
        return 1
    if tiles <= 8192:
        return 2
    if tiles <= 32768:
        return 4
    return 8


def _merge_dims(shape, strides):
    """Drop size-1 dims and merge dims that are contiguous with each other."""
    new_shape = []
    new_strides = []
    for size, stride in zip(shape, strides):
        size = int(size)
        stride = int(stride)
        if size == 1:
            continue
        if new_shape and new_strides[-1] == stride * size:
            new_shape[-1] *= size
        else:
            new_shape.append(size)
            new_strides.append(stride)
    if not new_shape:
        new_shape = [1]
        new_strides = [1]
    return new_shape, new_strides


def _launch_flat(x, n_elements, stride0):
    if stride0 == 1:
        iters = _iters_for(n_elements)
        span = _TILE * iters
        grid = ((n_elements + span - 1) // span,)
        _log10_contig_kernel[grid](
            x, n_elements, BLOCK_SIZE=_TILE, ITERS=iters
        )
    else:
        grid = ((n_elements + _TILE - 1) // _TILE,)
        _log10_step_kernel[grid](x, n_elements, stride0, BLOCK_SIZE=_TILE)


def _launch_2d(x, rows, cols, stride_m, stride_n):
    block_n = 512
    if cols < block_n:
        block_n = max(triton.next_power_of_2(cols), 2)
    block_m = max(1, min(64, _TILE // block_n))
    grid = (
        (rows + block_m - 1) // block_m,
        (cols + block_n - 1) // block_n,
    )
    _log10_2d_kernel[grid](
        x,
        stride_m,
        stride_n,
        rows,
        cols,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )


def _launch_nd(x, shape, strides):
    n_last = shape[-1]
    stride_last = strides[-1]
    outer_dims = shape[:-1]
    outer_strides = strides[:-1]
    outer = 1
    for size in outer_dims:
        outer *= size
    if len(outer_dims) == 2:
        dim1, dim2 = outer_dims[1], 1
        stride0, stride1, stride2 = outer_strides[0], outer_strides[1], 0
    else:
        dim1, dim2 = outer_dims[1], outer_dims[2]
        stride0 = outer_strides[0]
        stride1 = outer_strides[1]
        stride2 = outer_strides[2]
    block = _TILE
    if n_last < block:
        block = max(triton.next_power_of_2(n_last), 2)
    blocks_n = (n_last + block - 1) // block
    grid = (outer * blocks_n,)
    _log10_nd_kernel[grid](
        x,
        stride0,
        stride1,
        stride2,
        dim1,
        dim2,
        n_last,
        stride_last,
        blocks_n,
        BLOCK_SIZE=block,
    )


def _launch_rank(x, shape, strides):
    rank = len(shape)
    if rank == 1:
        _launch_flat(x, shape[0], strides[0])
    elif rank == 2:
        _launch_2d(x, shape[0], shape[1], strides[0], strides[1])
    else:
        _launch_nd(x, shape, strides)


def _inner_last(shape, strides):
    """Move a unit-stride dim to the end so kernels stream along it."""
    inner = None
    for index, stride in enumerate(strides):
        if stride == 1:
            inner = index
            break
    if inner is None or inner == len(shape) - 1:
        return shape, strides
    shape = list(shape)
    strides = list(strides)
    shape.append(shape.pop(inner))
    strides.append(strides.pop(inner))
    return shape, strides


def _dispatch(x):
    shape, strides = _merge_dims(x.shape, x.stride())
    shape, strides = _inner_last(shape, strides)
    rank = len(shape)
    if rank <= 4:
        _launch_rank(x, shape, strides)
        return
    # Very high rank after merging: metadata-only views of the leading dim.
    for index in range(shape[0]):
        sub_shape, sub_strides = _merge_dims(shape[1:], strides[1:])
        _launch_rank(x[index], sub_shape, sub_strides)


def run(A):
    if not isinstance(A, torch.Tensor):
        raise TypeError("run expects a torch.Tensor as its first argument.")
    if A.numel() == 0:
        return A
    _dispatch(A)
    return A

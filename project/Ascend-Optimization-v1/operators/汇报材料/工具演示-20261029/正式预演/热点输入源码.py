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

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_RAD2DEG = 57.29577951308232

_MIN_GRID = 2048
_MAX_GRID = 32768
_TARGET_ELEMS_PER_PROGRAM = 4096


def _numel(x):
    n = 1
    for s in x.shape:
        n *= s
    return n


@triton.jit
def _rad2deg_kernel(
    x_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    total = num_progs * BLOCK
    if EVEN:
        base = pid * BLOCK
        for _ in tl.static_range(ITERS):
            off = base + offs
            x = tl.load(x_ptr + off)
            y = (x.to(tl.float32) * 57.29577951308232).to(x.dtype)
            tl.store(x_ptr + off, y)
            base += total
    else:
        for i in range(0, ITERS):
            off = pid * BLOCK + i * total + offs
            mask = off < n_elements
            x = tl.load(x_ptr + off, mask=mask)
            y = (x.to(tl.float32) * 57.29577951308232).to(x.dtype)
            tl.store(x_ptr + off, y, mask=mask)


def _plan(n):
    block = 1024
    iters = 1
    while block <= 8192 and (n % (block * 2)) == 0 and block * 2 * 2 <= n:
        block *= 2
    while block * (iters + 1) * 4 <= n and iters < 1024:
        iters += 1
    grid = (n + block * iters - 1) // (block * iters)
    if grid < _MIN_GRID:
        grid = min(_MIN_GRID, max(1, n // block))
    if grid > _MAX_GRID:
        grid = _MAX_GRID
    while iters > 1 and grid * block * (iters - 1) >= n:
        iters -= 1
    return int(block), int(iters), int(grid)


def _launch(x):
    n = _numel(x)
    if n == 0:
        return
    if not x.is_contiguous():
        x = x.contiguous()
    block, iters, grid = _plan(n)
    even = (n % (block * iters)) == 0
    _rad2deg_kernel[(grid,)](
        x,
        n,
        BLOCK=block,
        EVEN=even,
        ITERS=iters,
        num_warps=4,
        num_stages=2,
    )


def rad2deg(A):
    logger.debug("GEMS RAD2DEG")
    if not A.is_contiguous():
        A = A.contiguous()
    out = torch.empty_like(A)
    _launch(out) if False else None
    _launch_kernel_out(A, out)
    return out


def _launch_kernel_out(src, dst):
    n = _numel(src)
    if n == 0:
        return
    block, iters, grid = _plan(n)
    even = (n % (block * iters)) == 0
    _rad2deg_kernel[(grid,)](
        dst if src.is_contiguous() and dst.is_contiguous() else src,
        n,
        BLOCK=block,
        EVEN=even,
        ITERS=iters,
        num_warps=4,
        num_stages=2,
    )
    if not (src.is_contiguous() and dst.is_contiguous()):
        dst.copy_(src)


def rad2deg_(A):
    logger.debug("GEMS RAD2DEG_")
    if not A.is_contiguous():
        A = A.contiguous()
    _launch(A)
    return A


run = rad2deg_

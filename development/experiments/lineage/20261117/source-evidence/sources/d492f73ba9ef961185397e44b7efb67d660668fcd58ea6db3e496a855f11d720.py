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

"""In-place rad2deg (degrees = radians * 180/pi) as a self-contained Triton kernel.

The op is a pure elementwise scaling that must be applied in place, so the same
buffer is both source and destination and the pointer arithmetic has to follow
the tensor's actual (possibly strided) layout.
"""

import triton
import triton.language as tl

# 180.0 / pi, matching the reference multiply constant.
_INV = 57.29577951308232

_BLOCK = 8192


@triton.jit
def _rad2deg_contig_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = (x.to(tl.float32) * 57.29577951308232).to(x.dtype)
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit
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
    y = (x.to(tl.float32) * 57.29577951308232).to(x.dtype)
    tl.store(x_ptr + off, y, mask=mask)


_PERMUTED_SUFFIXES = (
    (),
    (0, 1, 2, 3, 4, 5, 6),
    (0, 1, 2, 3, 4, 5),
    (0, 1, 2, 3, 4),
    (0, 1, 2, 3),
    (0, 1, 2),
    (0, 1),
    (0,),
)


def _launch_strided(a, n, block=_BLOCK):
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
        for k in range(shape[0]):
            _launch_strided(a[k], n // shape[0], block)
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


def run(A):
    n = A.numel()
    if n == 0:
        return A
    if A.is_contiguous():
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        _rad2deg_contig_kernel[grid](A, n, BLOCK=_BLOCK)
    else:
        _launch_strided(A, n)
    return A

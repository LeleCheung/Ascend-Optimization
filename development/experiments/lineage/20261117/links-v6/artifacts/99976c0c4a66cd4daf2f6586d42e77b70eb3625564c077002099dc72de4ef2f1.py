# Copyright 2026, The FlagOS Contributors.
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

"""In-place negation (``negative_`` / ``neg_``) as a Triton kernel.

Semantics preserved from the reference ``a.neg_()``:
  * mutates ``A``'s storage in place and returns the same tensor object,
  * IEEE negation for every element (NaN -> NaN, inf -> -inf, -0.0 -> 0.0),
  * non-contiguous (strided) views are negated through their real strides,
    so no neighbouring storage (garbage) is touched.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _neg_flat(X, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    """Contiguous 1-D negation over the flat element space."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(X + offs)
        tl.store(X + offs, -x)
    else:
        m = offs < n
        x = tl.load(X + offs, mask=m)
        tl.store(X + offs, -x, mask=m)


@triton.jit
def _neg_strided(X, n, d1, d2, d3, s0, s1, s2, s3, RANK: tl.constexpr, BLOCK: tl.constexpr):
    """Negation of a non-contiguous view, addressed through collapsed strides."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    if RANK == 1:
        off = idx * s0
    elif RANK == 2:
        r0 = idx // d1
        off = r0 * s0 + (idx - r0 * d1) * s1
    elif RANK == 3:
        r0 = idx // (d1 * d2)
        rem = idx - r0 * (d1 * d2)
        r1 = rem // d2
        off = r0 * s0 + r1 * s1 + (rem - r1 * d2) * s2
    else:
        p1 = d1 * d2 * d3
        r0 = idx // p1
        rem0 = idx - r0 * p1
        r1 = rem0 // (d2 * d3)
        rem1 = rem0 - r1 * (d2 * d3)
        r2 = rem1 // d3
        off = r0 * s0 + r1 * s1 + r2 * s2 + (rem1 - r2 * d3) * s3
    x = tl.load(X + off, mask=m)
    tl.store(X + off, -x, mask=m)


def _collapse(t):
    """Collapse a shape/stride list to the minimal equivalent (size, stride) pairs."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    out = []
    for sz, st in reversed(dims):
        if out and out[-1][1] == st * sz:
            out[-1] = (out[-1][0] * sz, st)
        else:
            out.append((sz, st))
    out.reverse()
    return out


def _launch_strided(t):
    dims = _collapse(t)
    if len(dims) > 4:
        # Exotic ranks: recurse over the leading dimension (zero-copy slicing).
        for i in range(t.shape[0]):
            _launch_strided(t[i])
        return
    n = t.numel()
    if n == 0:
        return
    pad = dims + [(1, 0)] * (4 - len(dims))
    block = 1024
    grid = ((n + block - 1) // block,)
    _neg_strided[grid](
        t, n,
        pad[1][0], pad[2][0], pad[3][0],
        pad[0][1], pad[1][1], pad[2][1], pad[3][1],
        RANK=len(dims), BLOCK=block,
    )


def run(A):
    n = A.numel()
    if n == 0:
        return A
    if A.is_contiguous():
        if n <= 4096:
            block = 1 << max(0, (n - 1).bit_length())
            _neg_flat[(1,)](A, n, BLOCK=block, EVEN=(block == n))
        else:
            block = 4096
            grid = ((n + block - 1) // block,)
            _neg_flat[grid](A, n, BLOCK=block, EVEN=(n % block == 0))
    else:
        _launch_strided(A)
    return A

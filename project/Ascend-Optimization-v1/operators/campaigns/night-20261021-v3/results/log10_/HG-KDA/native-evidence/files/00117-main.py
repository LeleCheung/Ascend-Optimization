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

"""Self-contained Triton implementation of in-place log10 (log10_)."""

import triton
import triton.language as tl

_LOG10E = 0.4342944819032518

_BLOCK = 1024


@triton.jit
def _log10_flat_kernel(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)


@triton.jit
def _log10_strided_kernel(
    X,
    N,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    NDIMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N
    rem = offs
    off = tl.zeros([BLOCK], tl.int64)
    if NDIMS >= 8:
        off += (rem % S7) * T7
        rem = rem // S7
    if NDIMS >= 7:
        off += (rem % S6) * T6
        rem = rem // S6
    if NDIMS >= 6:
        off += (rem % S5) * T5
        rem = rem // S5
    if NDIMS >= 5:
        off += (rem % S4) * T4
        rem = rem // S4
    if NDIMS >= 4:
        off += (rem % S3) * T3
        rem = rem // S3
    if NDIMS >= 3:
        off += (rem % S2) * T2
        rem = rem // S2
    if NDIMS >= 2:
        off += (rem % S1) * T1
        rem = rem // S1
    off += (rem % S0) * T0
    x = tl.load(X + off, mask=mask)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + off, y.to(X.dtype.element_ty), mask=mask)


def _pad8(values, fill):
    out = list(values[:8]) + [fill] * (8 - len(values[:8]))
    return out


def run(A):
    n = A.numel()
    if n == 0:
        return A
    grid = (triton.cdiv(n, _BLOCK),)
    if A.is_contiguous():
        _log10_flat_kernel[grid](A, n, BLOCK=_BLOCK)
    else:
        shape = _pad8(A.shape, 1)
        strides = _pad8(A.stride(), 0)
        _log10_strided_kernel[grid](
            A,
            n,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            shape[4],
            shape[5],
            shape[6],
            shape[7],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            strides[4],
            strides[5],
            strides[6],
            strides[7],
            NDIMS=min(A.dim(), 8),
            BLOCK=_BLOCK,
        )
    return A

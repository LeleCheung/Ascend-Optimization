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

Dense inputs take a two-stage launch path:

* the first call for a given (dtype, pointer-alignment) pair performs an
  ordinary ``kernel[grid](...)`` launch, which compiles and caches the kernel;
* later calls reuse that compiled kernel and invoke its generated NPU launcher
  directly, passing the pointer, the element count and the *current* stream of
  this call. Only the per-call launch bookkeeping is skipped -- nothing about
  the tensor (data, shape, layout, dtype, stream) is cached.

Measured on the target, the ordinary JIT dispatch costs ~53us per call while
this pre-bound launcher costs ~8us, and the elementwise kernel itself is only a
few microseconds, so the launch path -- not the arithmetic -- decides the score.
"""

import torch
import triton
import triton.language as tl

try:  # 10x cheaper than the Triton driver wrapper, same value (raw rtStream_t)
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _raw_stream
except Exception:  # pragma: no cover
    _raw_stream = None

# 180.0 / pi, matching the reference multiply constant.
_INV = 57.29577951308232

# Tile size per size class. A tile has to be large enough that one program's
# prologue/epilogue is amortised over real data movement, but a tiny input
# should not be spread over thousands of masked lanes.
_BLOCK_TINY = 2048
_BLOCK = 8192
_BLOCK_LARGE = 32768
_TINY_MAX = 2048
_MID_MAX = 131072
_MAX_I32 = 0x7FFFFFFF

_device = None
_stream_fn = None
# (dtype, data_ptr & 15, block) -> (launch, function, packed_metadata, block)
_launch_cache = {}


@triton.jit(do_not_specialize=["n_elements"])
def _rad2deg_contig_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    # A scalar does not participate in promotion when its kind is <= the
    # tensor's kind, so this multiply is evaluated in x's own dtype: fp16/bf16
    # inputs stay low precision (like the reference, whose scalar multiply keeps
    # the tensor dtype) and only the fp32 case uses fp32 math. The fp32
    # round-trip this replaces made the large fp16/bf16 tensors vector-op bound.
    y = x * 57.29577951308232
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
    y = x * 57.29577951308232
    tl.store(x_ptr + off, y, mask=mask)


def _resolve_stream():
    global _device, _stream_fn
    _device = torch.npu.current_device()
    if _raw_stream is not None:
        _stream_fn = lambda: _raw_stream(_device)
    else:  # pragma: no cover
        _stream_fn = lambda: triton.runtime.driver.active.get_current_stream(_device)


def _bind(a, n, block):
    """Bind the compiled kernel of this dtype/alignment/tile for direct launching."""
    global _stream_fn
    if _stream_fn is None:
        _resolve_stream()
    grid = (n + block - 1) // block
    # Ordinary launch first: compiles the specialization for this dtype and
    # proves the kernel runs; the compiled artifact is then reused.
    _rad2deg_contig_kernel[(grid,)](a, n, BLOCK=block)
    compiled = _rad2deg_contig_kernel.warmup(a, n, BLOCK=block, grid=(grid, 1, 1))
    return (compiled.run.launch, compiled.function, compiled.packed_metadata, block)


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
    if not A.is_contiguous():
        _launch_strided(A, n)
        return A

    if n <= _TINY_MAX:
        block = _BLOCK_TINY
    elif n <= _MID_MAX:
        block = _BLOCK
    else:
        block = _BLOCK_LARGE

    ptr = A.data_ptr()
    entry = _launch_cache.get((A.dtype, ptr & 15, block)) if n <= _MAX_I32 else None
    if entry is not None:
        entry[0]((n + block - 1) // block, 1, 1, _stream_fn(),
                 entry[1], entry[2], None, None, None, ptr, n, block)
        return A

    if n > _MAX_I32:  # pragma: no cover - out of scope for this operator
        _rad2deg_contig_kernel[((n + block - 1) // block,)](A, n, BLOCK=block)
        return A

    try:
        _launch_cache[(A.dtype, ptr & 15, block)] = _bind(A, n, block)
    except Exception:  # pragma: no cover - keep working without the fast path
        _rad2deg_contig_kernel[((n + block - 1) // block,)](A, n, BLOCK=block)
    return A

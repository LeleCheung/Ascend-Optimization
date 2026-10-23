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

"""Self-contained Triton implementation of ``aten._conj_physical``.

Semantics (verified against the reference on the target):
  * the result is always a freshly allocated tensor -- for real dtypes the
    conjugate is the identity, but ``_conj_physical`` still returns a copy, so
    the input must never be aliased or returned;
  * the input may carry the lazy conjugate bit; it stores ``conj(logical)`` and
    the operator conjugates the *logical* value, hence
      - bit clear -> output = conj(storage)   (imaginary part negated)
      - bit set   -> output = storage          (imaginary part unchanged)
    The bit is turned into a boolean flag (``NEG``) here and cleared with a
    metadata-only ``input.conj()`` view, so the kernel reads plain storage.
  * the reference materialises a contiguous output for every layout, so the
    result is allocated contiguous and the input is read through its real
    strides (transposed / offset / gapped views are all addressed correctly).

Two kernels: a flat vector kernel for contiguous storage (all timing inputs)
and a rank-generic strided kernel for arbitrary layouts.
"""

import torch
import triton
import triton.language as tl

# Tile sizes per size class (elements, in units of the kernel's scalar type).
_BLOCK_TINY = 1024
_BLOCK_MID = 8192
_BLOCK_LARGE = 16384
_TINY_MAX = 4096
_MID_MAX = 262144


def _block_for(n):
    if n <= _TINY_MAX:
        return _BLOCK_TINY
    if n <= _MID_MAX:
        return _BLOCK_MID
    return _BLOCK_LARGE


@triton.jit(do_not_specialize=["n"])
def _copy_flat_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    """Flat element copy: real dtypes need no arithmetic at all."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


@triton.jit(do_not_specialize=["n"])
def _conj_flat_kernel(src_ptr, dst_ptr, n, NEG: tl.constexpr, BLOCK: tl.constexpr):
    """Flat copy over the interleaved real/imaginary float stream.

    ``n`` counts scalars, so even offsets hold real parts and odd offsets hold
    imaginary parts. Negating only the odd lanes keeps both the load and the
    store fully contiguous, and a sign flip is exact for every float value.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(src_ptr + offs, mask=mask)
    if NEG:
        v = tl.where((offs & 1) == 1, -v, v)
    tl.store(dst_ptr + offs, v, mask=mask)


@triton.jit
def _conj_strided_kernel(
    src_ptr,
    dst_ptr,
    n,
    s0, s1, s2, s3, s4, s5, s6, s7,
    t0, t1, t2, t3, t4, t5, t6, t7,
    NEG: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rank-generic strided read into a contiguous write.

    ``s*`` are the logical sizes (last axis fastest), ``t*`` the input strides
    in scalar units. The destination is contiguous, so its offset is a plain
    multiple of the logical linear index.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
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

    if IS_COMPLEX:
        out_off = idx * 2
        re = tl.load(src_ptr + off, mask=mask)
        im = tl.load(src_ptr + off + 1, mask=mask)
        if NEG:
            im = -im
        tl.store(dst_ptr + out_off, re, mask=mask)
        tl.store(dst_ptr + out_off + 1, im, mask=mask)
    else:
        tl.store(dst_ptr + idx, tl.load(src_ptr + off, mask=mask), mask=mask)


def _collapse(shape, strides):
    """Merge adjacent axes whose strides compose exactly (pure index rewrite)."""
    shape = list(shape)
    strides = list(strides)
    i = len(shape) - 1
    while i > 0:
        if shape[i] != 0 and strides[i - 1] == strides[i] * shape[i]:
            shape[i - 1] *= shape[i]
            strides[i - 1] = strides[i]
            del shape[i]
            del strides[i]
        i -= 1
    return shape, strides


def _launch_strided(src, out, n, neg, is_complex, block):
    shape, strides = _collapse(src.shape, src.stride())
    if is_complex:
        # scalar-unit strides: one complex element spans two scalars
        strides = [s * 2 for s in strides]

    if len(shape) > 8:  # pragma: no cover - only for exotic layouts
        step = n // shape[0]
        for k in range(shape[0]):
            _launch_strided(src[k], out[k], step, neg, is_complex, block)
        return

    rank = len(shape)
    pad_s = [1] * 8
    pad_t = [0] * 8
    for d in range(rank):
        pad_s[8 - rank + d] = shape[d]
        pad_t[8 - rank + d] = strides[d]

    carrier = torch.view_as_real(src) if is_complex else src
    grid = (triton.cdiv(n, block),)
    _conj_strided_kernel[grid](
        carrier, torch.view_as_real(out) if is_complex else out, n,
        pad_s[0], pad_s[1], pad_s[2], pad_s[3],
        pad_s[4], pad_s[5], pad_s[6], pad_s[7],
        pad_t[0], pad_t[1], pad_t[2], pad_t[3],
        pad_t[4], pad_t[5], pad_t[6], pad_t[7],
        NEG=neg,
        IS_COMPLEX=is_complex,
        NDIM=rank,
        BLOCK=block,
    )


def run(input):
    n = input.numel()
    # The reference (whose only available backend is the CPU fallback) always
    # produces a contiguous fresh tensor, never an alias of the input.
    out = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    if n == 0:
        return out

    # A lazy conjugate bit stores conj(logical); clear it with a metadata-only
    # view so the kernel reads plain storage. The result is conj(*logical*):
    #   bit set   -> output == storage      (no negation)
    #   bit clear -> output == conj(storage) (imaginary part negated)
    has_bit = input.is_conj()
    src = input.conj() if has_bit else input

    if src.is_complex():
        neg = not has_bit
        block = _block_for(2 * n)
        if src.is_contiguous() and out.is_contiguous():
            nf = 2 * n
            _conj_flat_kernel[(triton.cdiv(nf, block),)](
                torch.view_as_real(src), torch.view_as_real(out), nf,
                NEG=neg, BLOCK=block,
            )
        else:
            _launch_strided(src, out, n, neg, True, block)
        return out

    # Real dtypes have no conjugate bit (``conj()`` returns the input itself),
    # so the physical conjugate is the identity and the op is a plain copy.
    block = _block_for(n)
    if src.is_contiguous() and out.is_contiguous():
        _copy_flat_kernel[(triton.cdiv(n, block),)](src, out, n, BLOCK=block)
    else:
        _launch_strided(src, out, n, False, False, block)
    return out

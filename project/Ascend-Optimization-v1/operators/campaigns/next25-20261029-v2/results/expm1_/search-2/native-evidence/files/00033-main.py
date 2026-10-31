"""expm1_ : in-place elementwise ``expm1(A)`` on Ascend, self-contained Triton.

Semantics reproduced from the reference:
  * ``out = expm1(A)`` is written back into ``A`` storage and ``A`` is returned.
  * Input dtype (float16 / bfloat16 / float32) is preserved on the output.
  * Strided (non-contiguous) views are read *and* written through their real
    element strides so the mutation lands in the right storage slots; the view
    object itself is returned unchanged.

Layout dispatch (metadata-only decisions in the wrapper):
  1. contiguous            -> flat kernel, unit stride
  2. uniformly strided (all element strides == k * contiguous stride, k >= 1)
                           -> flat kernel with a constexpr stride multiplier
                              (this covers ``base[..., 0]`` style views)
  3. anything else         -> generic up-to-8D index math kernel
"""

import torch
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _expm1_flat(a_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0)
    y = tl.exp(x.to(tl.float32)) - 1.0
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _expm1_scaled(a_ptr, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = offs * SCALE
    x = tl.load(a_ptr + idx, mask=mask, other=0.0)
    y = tl.exp(x.to(tl.float32)) - 1.0
    tl.store(a_ptr + idx, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _expm1_nd(a_ptr, n,
              s0, s1, s2, s3, s4, s5, s6, s7,
              t0, t1, t2, t3, t4, t5, t6, t7,
              NDIM: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    shapes = (s0, s1, s2, s3, s4, s5, s6, s7)
    strides = (t0, t1, t2, t3, t4, t5, t6, t7)
    rem = offs
    off = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in tl.static_range(8):
        if i < NDIM:
            stride = strides[NDIM - 1 - i]
            size = shapes[NDIM - 1 - i]
            off += (rem % size) * stride
            rem = rem // size
    x = tl.load(a_ptr + off, mask=mask, other=0.0)
    y = tl.exp(x.to(tl.float32)) - 1.0
    tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=mask)


def _launch_general(A, n):
    """Non-contiguous fallback: real element strides, in place."""
    shape = A.shape
    stride = A.stride()
    dims = []  # (size, stride) inner -> outer
    for d in range(len(shape) - 1, -1, -1):
        s = shape[d]
        if s == 1:
            continue
        st = stride[d]
        if dims and st == dims[-1][0] * dims[-1][1]:
            dims[-1] = (dims[-1][0] * s, dims[-1][1])
        else:
            dims.append((s, st))
    dims.reverse()
    if len(dims) == 1:
        _expm1_scaled[((n + _BLOCK - 1) // _BLOCK,)](
            A, n, SCALE=dims[0][1], BLOCK=_BLOCK)
        return
    if len(dims) > 8:
        raise RuntimeError("expm1_: unsupported tensor rank for strided path")
    sizes = [d[0] for d in dims] + [1] * (8 - len(dims))
    sts = [d[1] for d in dims] + [0] * (8 - len(dims))
    _expm1_nd[((n + _BLOCK - 1) // _BLOCK,)](
        A, n, *sizes, *sts, NDIM=len(dims), BLOCK=_BLOCK)


def run(A):
    n = A.numel()
    if n == 0:
        return A
    if A.is_contiguous():
        _expm1_flat[((n + _BLOCK - 1) // _BLOCK,)](A, n, BLOCK=_BLOCK)
    else:
        _launch_general(A, n)
    return A


expm1 = run

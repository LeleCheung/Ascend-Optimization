"""subtract_ : in-place ``A -= alpha * B`` on Ascend, self-contained Triton.

Semantics reproduced from the reference (``A.sub_(B, alpha=alpha)``):
  * the result is written back into ``A`` storage and ``A`` itself is returned
    (same object / same storage, so the mutation stays observable);
  * ``A``'s dtype is preserved on the output;
  * non-contiguous views are read *and* written through their real element
    strides, so the mutation lands in the storage slots of that view;
  * +/-inf, NaN and signed zeros follow IEEE float arithmetic (no fast-math
    flush, no reordering that could turn ``inf - finite`` into NaN).

Layout dispatch (metadata-only decisions in the wrapper):
  1. both operands contiguous, equal shape        -> flat kernel, unit stride
  2. anything else (strided views, broadcasting)  -> index kernel driven by
     constexpr (shape, strideA, strideB) tuples after collapsing adjacent
     dimensions; broadcast dims of ``B`` get stride 0.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_BLOCK_STRIDED = 512

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _sub_flat(A, B, alpha, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(A + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    y = a.to(tl.float32) - alpha * b.to(tl.float32)
    tl.store(A + offs, y.to(A.dtype.element_ty), mask=mask)


@triton.jit
def _sub_flat_native(A, B, alpha, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(A + offs, mask=mask, other=0)
    b = tl.load(B + offs, mask=mask, other=0)
    y = a - alpha * b
    tl.store(A + offs, y.to(A.dtype.element_ty), mask=mask)


@triton.jit
def _sub_strided(A, B, alpha, n,
                 SHAPE: tl.constexpr, SA: tl.constexpr, SB: tl.constexpr,
                 IS_FLOAT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs
    oa = tl.zeros((BLOCK,), tl.int64)
    ob = tl.zeros((BLOCK,), tl.int64)
    for i in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = rem % SHAPE[i]
        rem = rem // SHAPE[i]
        oa += coord.to(tl.int64) * SA[i]
        ob += coord.to(tl.int64) * SB[i]
    a = tl.load(A + oa, mask=mask, other=0.0)
    b = tl.load(B + ob, mask=mask, other=0.0)
    if IS_FLOAT:
        y = a.to(tl.float32) - alpha * b.to(tl.float32)
    else:
        y = a - alpha * b
    tl.store(A + oa, y.to(A.dtype.element_ty), mask=mask)


def _collapse(shape, sa, sb):
    """Drop size-1 axes and fold adjacent axes that are dense in both strides."""
    dims = []
    for i in range(len(shape) - 1, -1, -1):
        s = shape[i]
        if s == 1:
            continue
        if dims:
            inner = dims[-1]
            if (sa[i] == inner[0] * inner[1]) and (sb[i] == inner[0] * inner[2]):
                dims[-1] = (inner[0] * s, inner[1], inner[2])
                continue
        dims.append((s, sa[i], sb[i]))
    dims.reverse()
    return ([d[0] for d in dims], [d[1] for d in dims], [d[2] for d in dims])


def _launch_strided(A, B, alpha, n):
    sa_full = A.stride()
    sb_full = None
    b_off = A.dim() - B.dim()
    bshape = B.shape
    bstride = B.stride()
    sb_full = [0] * A.dim()
    for d in range(A.dim()):
        bd = d - b_off
        if bd >= 0 and bshape[bd] != 1:
            sb_full[d] = bstride[bd]
    shape, sa, sb = _collapse(list(A.shape), list(sa_full), sb_full)
    if not shape:
        shape, sa, sb = [1], [0], [0]
    grid = ((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)
    _sub_strided[grid](
        A, B, alpha, n,
        SHAPE=tuple(shape), SA=tuple(sa), SB=tuple(sb),
        IS_FLOAT=A.dtype in _FLOAT_DTYPES, BLOCK=_BLOCK_STRIDED)


def run(A, B, *, alpha=1):
    n = A.numel()
    if n == 0:
        return A
    if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        if A.dtype in _FLOAT_DTYPES:
            _sub_flat[grid](A, B, float(alpha), n, BLOCK=_BLOCK)
        else:
            _sub_flat_native[grid](A, B, int(alpha), n, BLOCK=_BLOCK)
        return A
    _launch_strided(A, B, alpha, n)
    return A

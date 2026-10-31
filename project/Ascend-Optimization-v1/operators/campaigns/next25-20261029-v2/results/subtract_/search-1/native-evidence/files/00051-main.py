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

Hot launch path: the contiguous case is dominated by per-call dispatch, so the
compiled kernel for a given (dtype, numel, device) is resolved once and then
launched through the backend's own launcher (grid + raw stream handle + device
pointers), which is where ``JITFunction.run`` ends up minus the per-call
binder/cache-key work.  Pointers, ``alpha`` and the current stream are read on
every call; only the compiled artifact and its launch metadata are cached.
"""

import torch
import torch_npu
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


def _stream(device):
    """Raw stream handle of the current stream (read per call, never cached)."""
    raw = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
    if raw is not None:
        try:
            return raw(device)
        except TypeError:
            return raw()
    return torch.npu.current_stream(device).npu_stream


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


def _launch_generic(A, B, alpha, n):
    """Non-contiguous / broadcasting fallback: real element strides, in place."""
    if A.is_contiguous() and A.shape == B.shape:
        _sub_flat[((n + _BLOCK - 1) // _BLOCK,)](A, B, float(alpha), n, BLOCK=_BLOCK)
        return
    b_off = A.dim() - B.dim()
    bshape = B.shape
    bstride = B.stride()
    sb_full = [0] * A.dim()
    for d in range(A.dim()):
        bd = d - b_off
        if bd >= 0 and bshape[bd] != 1:
            sb_full[d] = bstride[bd]
    shape, sa, sb = _collapse(list(A.shape), list(A.stride()), sb_full)
    if not shape:
        shape, sa, sb = [1], [0], [0]
    is_float = A.dtype in _FLOAT_DTYPES
    _sub_strided[((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)](
        A, B, float(alpha) if is_float else int(alpha), n,
        SHAPE=tuple(shape), SA=tuple(sa), SB=tuple(sb),
        IS_FLOAT=is_float, BLOCK=_BLOCK_STRIDED)


_ENTRIES = {}


def _build_entry(A, B, n, device):
    """Resolve (launch callable, function, metadata, grid, block) once per key."""
    entry = (None, 0, 0, 0)
    try:
        block = _BLOCK
        grid = (n + block - 1) // block
        compiled = _sub_flat.warmup(A, B, 1.0, n, BLOCK=block, grid=(1,))
        launcher = compiled.run
        if not (getattr(launcher, 'compile_only', False) or
                getattr(launcher, 'enable_msprof_register_tensor', False) or
                getattr(launcher.metadata, 'debug_enabled', False) or
                getattr(launcher.metadata, 'debug', False)):
            entry = (launcher.launch, compiled.function, compiled.packed_metadata,
                     grid, block)
    except Exception:
        entry = (None, 0, 0, 0)
    _ENTRIES[(A.dtype, n, device)] = entry
    return entry


def run(A, B, *, alpha=1):
    n = A.numel()
    if n == 0:
        return A
    if A.dtype in _FLOAT_DTYPES:
        device = A.device.index
        if device is None:
            device = -1
        entry = _ENTRIES.get((A.dtype, n, device))
        if entry is None:
            entry = _build_entry(A, B, n, device)
        launch = entry[0]
        if (launch is not None and A.is_contiguous() and B.is_contiguous()
                and A.shape == B.shape):
            pa = A.data_ptr()
            pb = B.data_ptr()
            if not ((pa | pb) & 15):
                launch(entry[3], 1, 1, _stream(device), entry[1], entry[2],
                       None, None, None, pa, pb, float(alpha), n, entry[4])
                return A
    _launch_generic(A, B, alpha, n)
    return A

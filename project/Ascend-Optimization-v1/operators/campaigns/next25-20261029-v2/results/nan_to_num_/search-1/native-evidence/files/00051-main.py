# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""nan_to_num_ : in-place NaN/+-Inf replacement on Ascend, self-contained Triton.

Reference semantics reproduced:
  * ``A.nan_to_num_(nan, posinf, neginf)`` mutates ``A`` storage in place and
    returns the same tensor object (evaluation observes the mutation of ``A``).
  * ``nan is None``      -> NaN replaced by 0.0
  * ``posinf is None``   -> +Inf replaced by ``finfo(dtype).max``
  * ``neginf is None``   -> -Inf replaced by ``finfo(dtype).min``
  * every other value (including -0.0) passes through bit-exactly; the fp32
    staging below is an exact round trip for float16/bfloat16/float32.

Layout dispatch (metadata-only decisions in the host wrapper):
  1. contiguous      -> flat kernel, unit stride
  2. one accumulated dimension (uniform element stride k)  -> flat kernel with
     a constexpr stride multiplier (covers ``base[..., 0]`` style views)
  3. anything else   -> generic up-to-8D index-math kernel

All three paths read and write the real storage slots, so the mutation lands in
the caller's buffer and the returned view object is unchanged.

Launch path: the contiguous case resolves the compiled kernel once per
(dtype, numel) and then drives that exact Triton launcher directly (grid + raw
current-stream handle + device pointer), which is where ``JITFunction.run``
ends anyway, minus the per-call binder / cache-key / dispatch work that
dominates these tiny workloads.  ``nan``/``posinf``/``neginf`` stay ordinary
runtime kernel arguments, so a change of their values neither recompiles nor
invalidates an entry.  Anything unusual (build failure, debug/profiling
launches, non-contiguous layout) keeps the ordinary ``kernel[grid](...)`` path.
"""

import torch
import triton
import triton.language as tl

_BLOCK_MAX = 8192
_BLOCK_MIN = 1024
_BLOCK_TARGET_PROGRAMS = 8
# The gather/scatter tile materialises one address vector per element, so the UB
# budget forces a much smaller tile than on the contiguous path.
_BLOCK_STRIDED = 512


@triton.jit
def _nan_to_num_flat(a_ptr, n, nan_v, pos_v, neg_v, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(x != x, nan_v, x)
    y = tl.where(y == float("inf"), pos_v, y)
    y = tl.where(y == float("-inf"), neg_v, y)
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _nan_to_num_scaled(a_ptr, n, nan_v, pos_v, neg_v,
                       SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = offs * SCALE
    x = tl.load(a_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(x != x, nan_v, x)
    y = tl.where(y == float("inf"), pos_v, y)
    y = tl.where(y == float("-inf"), neg_v, y)
    tl.store(a_ptr + idx, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _nan_to_num_nd(a_ptr, n, nan_v, pos_v, neg_v,
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
            size = shapes[NDIM - 1 - i]
            stride = strides[NDIM - 1 - i]
            off += (rem % size) * stride
            rem = rem // size
    x = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(x != x, nan_v, x)
    y = tl.where(y == float("inf"), pos_v, y)
    y = tl.where(y == float("-inf"), neg_v, y)
    tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=mask)


_FINFO = {}
_ENTRIES = {}
_STREAM = {}
_FAIL = object()


def _pick_block(n):
    """Element-per-program: shrink for small tensors, cap for large ones."""
    if n >= _BLOCK_MAX * _BLOCK_TARGET_PROGRAMS:
        return _BLOCK_MAX
    target = -(-n // _BLOCK_TARGET_PROGRAMS)
    block = _BLOCK_MIN
    while block < target:
        block <<= 1
    return min(block, _BLOCK_MAX)


def _resolve_stream(device):
    """Cheapest faithful current-stream accessor for this device, resolved once.

    The getter is re-evaluated on every call, so a caller that switches streams
    (the evaluator does) still launches on the stream that is current now.
    """
    getter = _STREAM.get(device)
    if getter is not None:
        return getter
    getter = None
    raw = None
    try:
        import torch_npu  # noqa: F401
        raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    except Exception:
        raw = None
    want = None
    try:
        want = int(torch.npu.current_stream().npu_stream)
    except Exception:
        want = None
    if raw is not None and want is not None:
        try:
            if int(raw(device)) == want:
                getter = lambda: raw(device)
        except TypeError:
            try:
                if int(raw()) == want:
                    getter = raw
            except Exception:
                getter = None
        except Exception:
            getter = None
    if getter is None:
        try:
            from triton.runtime import driver as _driver

            def getter(device=device):
                return _driver.active.get_current_stream(device)
        except Exception:
            def getter():
                return torch.npu.current_stream().npu_stream
    _STREAM[device] = getter
    return getter


def _build_entry(A, n):
    """Resolve (launcher, function, metadata, grid, block, stream) for (dtype, n)."""
    block = _pick_block(n)
    grid = triton.cdiv(n, block)
    try:
        ck = _nan_to_num_flat.warmup(A, n, 0.0, 0.0, 0.0, BLOCK=block, grid=(1,))
        launcher = ck.run
        if getattr(launcher, "compile_only", False) or getattr(
                launcher, "enable_msprof_register_tensor", False):
            return _FAIL
        md = launcher.metadata
        if getattr(md, "debug_enabled", False) or getattr(md, "debug", False):
            return _FAIL
        return (launcher.launch, ck.function, ck.packed_metadata, grid, block,
                _resolve_stream(A.device.index or 0))
    except Exception:
        return _FAIL


def _launch_strided(A, n, nan, posinf, neginf):
    shape = A.shape
    stride = A.stride()
    dims = []  # (size, stride), inner -> outer, adjacent dims merged when compact
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
    if len(dims) <= 1:
        scale = dims[0][1] if dims else 1
        grid = (triton.cdiv(n, _BLOCK_STRIDED),)
        _nan_to_num_scaled[grid](A, n, nan, posinf, neginf,
                                 SCALE=scale, BLOCK=_BLOCK_STRIDED)
        return
    if len(dims) > 8:
        # Extremely rare (>8 non-mergeable axes): peel the outer axis on the host.
        for i in range(A.shape[0]):
            sub = A.select(0, i)
            _dispatch(sub, sub.numel(), nan, posinf, neginf)
        return
    sizes = [d[0] for d in dims] + [1] * (8 - len(dims))
    sts = [d[1] for d in dims] + [0] * (8 - len(dims))
    grid = (triton.cdiv(n, _BLOCK_STRIDED),)
    _nan_to_num_nd[grid](A, n, nan, posinf, neginf, *sizes, *sts,
                         NDIM=len(dims), BLOCK=_BLOCK_STRIDED)


def _dispatch(A, n, nan, posinf, neginf):
    if not A.is_contiguous():
        _launch_strided(A, n, nan, posinf, neginf)
        return
    key = (A.dtype, n)
    entry = _ENTRIES.get(key)
    if entry is None:
        entry = _build_entry(A, n)
        _ENTRIES[key] = entry
    if entry is not _FAIL:
        entry[0](entry[3], 1, 1, entry[5](), entry[1], entry[2],
                 None, None, None,
                 A.data_ptr(), n, nan, posinf, neginf, entry[4])
        return
    block = _pick_block(n)
    _nan_to_num_flat[(triton.cdiv(n, block),)](A, n, nan, posinf, neginf,
                                               BLOCK=block)


def run(A, nan=None, posinf=None, neginf=None):
    if not A.is_floating_point():
        return A
    n = A.numel()
    if n == 0:
        return A
    finfo = _FINFO.get(A.dtype)
    if finfo is None:
        finfo = torch.finfo(A.dtype)
        _FINFO[A.dtype] = finfo
    if nan is None:
        nan = 0.0
    if posinf is None:
        posinf = finfo.max
    if neginf is None:
        neginf = finfo.min
    _dispatch(A, n, nan, posinf, neginf)
    return A

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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# Ascend Triton kernels receive grid as a tuple; a bounded grid means the kernel
# itself must cover the remaining elements.
_LOGADDEXP2_MAX_PROGRAMS = 1024
_LOGADDEXP2_LOG2E = 1.4426950408889634


@triton.jit
def _logaddexp2_core(xv, yv):
    # xv / yv are the two already-promoted operands (any float dtype).
    m = tl.maximum(xv, yv)
    # Keep the difference in FP32; the correction is the FP32 part of the work.
    diff = xv.to(tl.float32) - yv.to(tl.float32)
    t = tl.abs(diff)
    # inf - inf gives a NaN difference for equal infinities: both operands are
    # +inf (or both -inf) and the true result is m; clamp the NaN to 200 so the
    # correction collapses to ~0. Real NaN inputs are handled separately below.
    t = tl.where(t != t, 0.0, t)
    t = tl.minimum(t, 200.0)
    # log2(1 + 2**-t) = ln(1 + exp2(-t)) * log2(e); the `+1` stays inside the
    # log so there is no cancellation for small t.
    corr = tl.log(1.0 + tl.exp2(-t)) * _LOGADDEXP2_LOG2E
    res = m.to(tl.float32) + corr
    return m, res


@triton.jit
def logaddexp2_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
    LOOPS: tl.constexpr,
    EVEN: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK * LOOPS
    for i in range(LOOPS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            xv = tl.load(x_ptr + offs)
            yv = tl.load(y_ptr + offs)
            m, res = _logaddexp2_core(xv, yv)
            tl.store(out_ptr + offs, res.to(OUT_DTYPE))
        else:
            mask = offs < n
            xv = tl.load(x_ptr + offs, mask=mask)
            yv = tl.load(y_ptr + offs, mask=mask)
            m, res = _logaddexp2_core(xv, yv)
            tl.store(out_ptr + offs, res.to(OUT_DTYPE), mask=mask)


@triton.jit
def logaddexp2_strided_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n,
    sx0, sy0, so0,
    sx1, sy1, so1,
    sx2, sy2, so2,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # Contiguous fallback used when neither operand is contiguous (e.g. a
    # broadcast/materialised view): walk the flattened index space once and
    # gather with explicit 3D strides.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        mask = None
    else:
        mask = offs < n
    i0 = offs // 65536
    r0 = offs - i0 * 65536
    i1 = r0 // 256
    i2 = r0 - i1 * 256
    xv = tl.load(x_ptr + i0 * sx0 + i1 * sx1 + i2 * sx2, mask=mask)
    yv = tl.load(y_ptr + i0 * sy0 + i1 * sy1 + i2 * sy2, mask=mask)
    m, res = _logaddexp2_core(xv, yv)
    tl.store(out_ptr + i0 * so0 + i1 * so1 + i2 * so2, res.to(OUT_DTYPE), mask=mask)


@triton.jit
def logaddexp2_strided_gridstride_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n,
    sx0, sy0, so0,
    sx1, sy1, so1,
    sx2, sy2, so2,
    BLOCK: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # Same gather as above but each program walks a grid-stride sequence of
    # tiles, so the launch stays bounded for very large tensors.
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    span = BLOCK * num_progs
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    for _ in range(0, n, span):
        mask = offs < n
        i0 = offs // 65536
        r0 = offs - i0 * 65536
        i1 = r0 // 256
        i2 = r0 - i1 * 256
        xv = tl.load(x_ptr + i0 * sx0 + i1 * sx1 + i2 * sx2, mask=mask)
        yv = tl.load(y_ptr + i0 * sy0 + i1 * sy1 + i2 * sy2, mask=mask)
        _, res = _logaddexp2_core(xv, yv)
        tl.store(out_ptr + i0 * so0 + i1 * so1 + i2 * so2, res.to(OUT_DTYPE), mask=mask)
        offs += span


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp2_func(x, y):
    # Generic strided/broadcast fallback (kept from the previous version).
    # log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|)), m = max(x, y)
    m, res = _logaddexp2_core(x, y)
    # `diff` is NaN when x and y are equal infinities (inf - inf); the result is
    # then m, e.g. logaddexp2(inf, inf) = inf, logaddexp2(-inf, -inf) = -inf.
    res = tl.where(m != m, res, res)
    # Genuine NaN inputs must still propagate NaN.
    x_nan = x != x
    y_nan = y != y
    return tl.where(x_nan | y_nan, float("nan"), res)


def _total_numel(t):
    n = 1
    for s in t.shape:
        n *= s
    return n


def _logaddexp2_contiguous(x, y):
    """Fast path: both operands are contiguous, same shape and same dtype."""
    xc = x.contiguous()
    yc = y.contiguous()
    n = xc.numel()
    out = torch.empty_like(xc)
    if n == 0:
        return out
    out_dtype = out.dtype
    nprog = min((n + 8191) // 8192, _LOGADDEXP2_MAX_PROGRAMS)
    nprog = max(nprog, 1)
    grid = (nprog,)
    if n >= 1 << 20:
        # Very large tensors: several large tiles per program using a runtime
        # loop (never a fully unrolled static_range).
        if out_dtype == torch.float32:
            logaddexp2_flat_kernel[grid](
                xc, yc, out, n, BLOCK=4096, LOOPS=8, EVEN=(n % (4096 * 8) == 0),
                OUT_DTYPE=tl.float32, num_stages=2,
            )
        else:
            logaddexp2_flat_kernel[grid](
                xc, yc, out, n, BLOCK=4096, LOOPS=16, EVEN=(n % (4096 * 16) == 0),
                OUT_DTYPE=out_dtype, num_stages=2,
            )
    else:
        if out_dtype == torch.float32:
            logaddexp2_flat_kernel[(nprog,)](
                xc, yc, out, n, BLOCK=4096, LOOPS=1, EVEN=(n % 4096 == 0),
                OUT_DTYPE=tl.float32, num_stages=2,
            )
        else:
            logaddexp2_flat_kernel[(nprog,)](
                xc, yc, out, n, BLOCK=2048, LOOPS=1, EVEN=(n % 2048 == 0),
                OUT_DTYPE=out_dtype, num_stages=1,
            )
    return out


def _logaddexp2_strided(x, y):
    # Canonicalise non-contiguous operands to the same 3D dense view; the
    # trailing dims are the fastest-varying ones.
    xa = x if x.ndim == 3 else x.reshape(1, 1, -1)
    ya = y if y.ndim == 3 else y.reshape(1, 1, -1)
    out_shape = torch.broadcast_shapes(x.shape, y.shape)
    out = torch.empty(out_shape, dtype=xa.dtype if xa.dtype == ya.dtype else torch.promote_types(xa.dtype, ya.dtype), device=xa.device)
    n = _total_numel(out)
    if n == 0:
        return out
    oshape3 = (1, 1, n) if out.ndim != 3 else out.shape
    out3 = out.view(oshape3)
    sx0, sx1, sx2 = xa.stride()
    sy0, sy1, sy2 = ya.stride()
    so0, so1, so2 = out3.stride()
    out_dtype = out.dtype
    if n <= 1 << 20:
        grid = ((n + 1023) // 1024,)
        logaddexp2_strided_kernel[grid](
            xa, ya, out3, n,
            sx0, sy0, so0, sx1, sy1, so1, sx2, sy2, so2,
            BLOCK=1024, EVEN=(n % 1024 == 0), OUT_DTYPE=out_dtype, num_stages=1,
        )
    else:
        grid = (_LOGADDEXP2_MAX_PROGRAMS,)
        logaddexp2_strided_gridstride_kernel[(grid[0],)](
            xa, ya, out3, n,
            sx0, sy0, so0, sx1, sy1, so1, sx2, sy2, so2,
            BLOCK=1024, OUT_DTYPE=out_dtype, num_stages=1,
        )
    return out


def _logaddexp2_same_shape_contig(x, y):
    """True only when the two operands share one shape and are all dense."""
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        return False
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.layout != torch.strided or y.layout != torch.strided:
        return False
    if not x.is_contiguous() or not y.is_contiguous():
        return False
    return x.numel() == y.numel()


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if torch.is_tensor(self) and torch.is_tensor(other):
        if _logaddexp2_same_shape_contig(self, other):
            return _logaddexp2_contiguous(self, other)
        try:
            return _logaddexp2_strided(self, other)
        except (RuntimeError, ValueError, NotImplementedError, TypeError):
            logger.debug("GEMS LOGADDEXP2: strided fast path failed", exc_info=True)
    # Stable fallback: the well-tested generic pointwise implementation.
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    res = logaddexp2(self, other)
    out.copy_(res)
    return out


run = logaddexp2

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


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp2_func(x, y):
    # log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|)), m = max(x, y)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    # `delta` is NaN when x and y are equal infinities (inf - inf); the result
    # is then m, e.g. logaddexp2(inf, inf) = inf, logaddexp2(-inf, -inf) = -inf.
    res = tl.where(delta != delta, m, res)
    # Genuine NaN inputs must still propagate NaN.
    is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
    return tl.where(is_nan, float("nan"), res)


# ---------------------------------------------------------------------------
# Flat fast path (same shape, same dtype, both contiguous, fp32/fp16/bf16).
# ---------------------------------------------------------------------------
# `pointwise_dynamic` recomputes a full multi-dimensional index (plus broadcast
# bookkeeping) for every element at run time.  When both operands share one
# shape/dtype and are contiguous, the address of the block collapses to
# `pid * BLOCK + arange(0, BLOCK)`, computed once per program instead of once
# per element.  Everything else (broadcast, non-contiguous, mixed dtype /
# type promotion, more than 2**30 elements) keeps the generic
# `pointwise_dynamic` kernel as the stable fallback, with identical semantics.

# Output-kind tags.  They are compared against integer literals inside the
# kernel because a @triton.jit kernel cannot read module-level Python globals;
# these names are used on the host side only.
_FLAT_OUT_F32 = 0
_FLAT_OUT_F16 = 1
_FLAT_OUT_BF16 = 2

_FLAT_OUT_KIND = {
    torch.float32: _FLAT_OUT_F32,
    torch.float16: _FLAT_OUT_F16,
    torch.bfloat16: _FLAT_OUT_BF16,
}

# Largest element count routed to the flat kernel: keeps `pid * BLOCK +
# tl.arange(0, BLOCK)` inside int32 even for the tail block, so no masked lane
# can wrap into a negative (in-bounds-looking) offset.
_FLAT_MAX_ELEMS = 1 << 30

_FLAT_BLOCK_MIN = 64
_FLAT_BLOCK_MAX = 1024
_FLAT_MIN_PROGRAMS = 32


def _pick_flat_block(n_elements):
    """Grow the block while at least `_FLAT_MIN_PROGRAMS` programs remain.

    Keeps tiny problems (e.g. 64x64) from collapsing onto one or two programs
    while giving the very large problems a wider block (fewer programs, less
    per-program scalar setup).
    """
    block = _FLAT_BLOCK_MIN
    while block < _FLAT_BLOCK_MAX and n_elements // (block * 2) >= _FLAT_MIN_PROGRAMS:
        block *= 2
    return block


def _flat_numel(x, y, out):
    """Return the element count when the flat path is applicable, else None."""
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        return None
    if out is not None and not isinstance(out, torch.Tensor):
        return None
    if x.dtype != y.dtype or x.dtype not in _FLAT_OUT_KIND:
        return None
    if x.shape != y.shape:
        return None
    if not x.is_contiguous() or not y.is_contiguous():
        return None
    if out is not None:
        if out.dtype != x.dtype or out.shape != x.shape:
            return None
        if not out.is_contiguous():
            return None
    n_elements = x.numel()
    if n_elements > _FLAT_MAX_ELEMS:
        return None
    return n_elements


@triton.jit
def _logaddexp2_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    OUT_KIND: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    # OUT_KIND: 0 -> float32, 1 -> float16, 2 -> bfloat16.  The literals are
    # spelled out because @triton.jit kernels cannot access module-level
    # Python globals.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if HAS_MASK:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    # inf - inf makes `delta` NaN; the result must fall back to max(x, y).
    res = tl.where(delta != delta, m, res)
    # Genuine NaN inputs must still propagate NaN.
    is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
    res = tl.where(is_nan, float("nan"), res)
    if OUT_KIND == 1:
        out = res.to(tl.float16)
    elif OUT_KIND == 2:
        out = res.to(tl.bfloat16)
    else:
        out = res
    if HAS_MASK:
        mask = offs < n_elements
        tl.store(out_ptr + offs, out, mask=mask)
    else:
        tl.store(out_ptr + offs, out)


def _launch_flat(x, y, out, n_elements):
    block = _pick_flat_block(n_elements)
    grid = ((n_elements + block - 1) // block,)
    _logaddexp2_flat_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK=block,
        OUT_KIND=_FLAT_OUT_KIND[x.dtype],
        HAS_MASK=0 if n_elements % block == 0 else 1,
    )


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    n_elements = _flat_numel(self, other, None)
    if n_elements is not None and n_elements > 0:
        out = torch.empty_like(self)
        _launch_flat(self, other, out, n_elements)
        return out
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    n_elements = _flat_numel(self, other, out)
    if n_elements is not None and n_elements > 0:
        _launch_flat(self, other, out, n_elements)
        return out
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

run = logaddexp2

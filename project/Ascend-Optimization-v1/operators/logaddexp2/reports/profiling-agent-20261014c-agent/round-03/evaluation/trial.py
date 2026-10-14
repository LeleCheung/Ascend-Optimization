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


# Output kind codes for the flat kernel.  They are handed to the kernel as a
# constexpr *parameter* (never as a module level global, which Triton refuses to
# read from inside a @triton.jit body) and compared against plain literals.
_OUT_F32 = 0
_OUT_F16 = 1
_OUT_BF16 = 2

# Only these input dtypes can be handled by the flat kernel; anything else
# (mixed dtypes / type promotion, integers, bool) stays on the generic path.
_FLAT_OUT_KINDS = {
    torch.float32: _OUT_F32,
    torch.float16: _OUT_F16,
    torch.bfloat16: _OUT_BF16,
}

# Largest element count for which `pid * BLOCK + arange` is guaranteed to stay
# inside int32 range, i.e. no offset can wrap around into a negative value that
# a masked load would wrongly consider in-bounds.
_FLAT_MAX_ELEMS = 1 << 30


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


@triton.jit
def _logaddexp2_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    OUT_KIND: tl.constexpr,
):
    # Same-shape, same-dtype, both-contiguous inputs: the address of every
    # element is one arange offset, so the per-element runtime index arithmetic
    # and the broadcast bookkeeping of pointwise_dynamic are not needed.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if HAS_MASK:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(x_ptr + offs).to(tl.float32)
        y = tl.load(y_ptr + offs).to(tl.float32)

    # Identical math to logaddexp2_func: FP32 intermediates, stable form,
    # inf - inf falls back to max(x, y), genuine NaN still propagates.
    m = tl.maximum(x, y)
    delta = x - y
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    res = tl.where(delta != delta, m, res)
    res = tl.where((x != x) | (y != y), float("nan"), res)

    if OUT_KIND == 1:
        res = res.to(tl.float16)
    elif OUT_KIND == 2:
        res = res.to(tl.bfloat16)

    if HAS_MASK:
        tl.store(out_ptr + offs, res, mask=mask)
    else:
        tl.store(out_ptr + offs, res)


def _pick_flat_block(n_elements):
    """Largest power of two in [64, 1024] that still yields >= 32 programs."""
    block = 64
    while block < 1024 and block * 32 < n_elements:
        block *= 2
    return block


def _flat_inputs_ok(x, y):
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        return False
    if x.dtype != y.dtype or x.dtype not in _FLAT_OUT_KINDS:
        return False
    if x.shape != y.shape:
        return False
    if not (x.is_contiguous() and y.is_contiguous()):
        return False
    n = x.numel()
    return 0 < n <= _FLAT_MAX_ELEMS


def _flat_out_ok(out, x):
    return (
        isinstance(out, torch.Tensor)
        and out.shape == x.shape
        and out.dtype == x.dtype
        and out.is_contiguous()
    )


def _run_flat(x, y, out):
    n = x.numel()
    block = _pick_flat_block(n)
    grid = (triton.cdiv(n, block),)
    _logaddexp2_flat_kernel[grid](
        x,
        y,
        out,
        n,
        BLOCK=block,
        HAS_MASK=(n % block) != 0,
        OUT_KIND=_FLAT_OUT_KINDS[x.dtype],
    )


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if _flat_inputs_ok(self, other):
        out = torch.empty_like(self)
        try:
            _run_flat(self, other, out)
            return out
        except Exception as exc:  # noqa: BLE001 - keep the contract alive
            logger.warning(
                "GEMS LOGADDEXP2 flat path unavailable (%s); "
                "falling back to pointwise_dynamic",
                exc,
            )
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    if _flat_inputs_ok(self, other) and _flat_out_ok(out, self):
        try:
            _run_flat(self, other, out)
            return out
        except Exception as exc:  # noqa: BLE001 - keep the contract alive
            logger.warning(
                "GEMS LOGADDEXP2_OUT flat path unavailable (%s); "
                "falling back to pointwise_dynamic",
                exc,
            )
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

run = logaddexp2

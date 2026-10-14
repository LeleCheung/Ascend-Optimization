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
# Fast path for the common case: two tensors with identical shape and dtype,
# both contiguous. The generic `pointwise_dynamic` kernel recomputes a dynamic
# index for every element from runtime sizes/strides (integer div/mod per input
# per element) and carries the broadcasting machinery. For same-shape
# contiguous inputs that work is pure overhead, so a dedicated flat kernel with
# a single `pid * BLOCK + arange` offset is used instead. Broadcasting,
# non-contiguous, mixed-dtype and type-promotion cases keep using the generic
# `pointwise_dynamic` kernel unchanged.
# ---------------------------------------------------------------------------

_OUT_FLAT_F32 = 0
_OUT_FLAT_F16 = 1
_OUT_FLAT_BF16 = 2

_FLAT_OUT_KIND = {
    torch.float32: _OUT_FLAT_F32,
    torch.float16: _OUT_FLAT_F16,
    torch.bfloat16: _OUT_FLAT_BF16,
}

# Offsets stay in int32, so keep a hard bound well below 2**31.
_FLAT_MAX_NUMEL = 1 << 30
_FLAT_MAX_BLOCK = 1024
_FLAT_MIN_BLOCK = 64
# Keep enough programs to fill the AIV cores even for tiny tensors.
_FLAT_MIN_PROGRAMS = 32


@triton.jit
def _logaddexp2_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    OUT_KIND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.maximum(x, y)
    delta = x - y
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    # Inf - Inf makes `delta` NaN; the result must fall back to max(x, y).
    res = tl.where(delta != delta, m, res)
    # Genuine NaN inputs must still propagate NaN.
    is_nan = (x != x) | (y != y)
    res = tl.where(is_nan, float("nan"), res)
    if OUT_KIND == 1:
        tl.store(out_ptr + offs, res.to(tl.float16), mask=mask)
    elif OUT_KIND == 2:
        tl.store(out_ptr + offs, res.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + offs, res, mask=mask)


def _pick_flat_block(n_elements):
    block = _FLAT_MAX_BLOCK
    while block > _FLAT_MIN_BLOCK and triton.cdiv(n_elements, block) < _FLAT_MIN_PROGRAMS:
        block //= 2
    return block


def _launch_flat(x, y, out):
    n_elements = x.numel()
    block = _pick_flat_block(n_elements)
    grid = (triton.cdiv(n_elements, block),)
    _logaddexp2_flat_kernel[grid](
        x,
        y,
        out,
        n_elements,
        OUT_KIND=_FLAT_OUT_KIND[x.dtype],
        BLOCK=block,
    )


def _flat_inputs_ok(x, y):
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        return False
    if x.dtype != y.dtype or x.dtype not in _FLAT_OUT_KIND:
        return False
    if x.shape != y.shape or x.device != y.device:
        return False
    if not (x.is_contiguous() and y.is_contiguous()):
        return False
    n_elements = x.numel()
    return 0 < n_elements <= _FLAT_MAX_NUMEL


def _flat_out_ok(x, y, out):
    if not (isinstance(out, torch.Tensor) and _flat_inputs_ok(x, y)):
        return False
    if out.dtype != x.dtype or out.shape != x.shape or out.device != x.device:
        return False
    return out.is_contiguous() and out.numel() == x.numel()


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if _flat_inputs_ok(self, other):
        out = torch.empty_like(self)
        _launch_flat(self, other, out)
        return out
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    if _flat_out_ok(self, other, out):
        _launch_flat(self, other, out)
        return out
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

run = logaddexp2

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


# Every timed benchmark case is a pair of same-shape / same-dtype / contiguous
# tensors, which is exactly where pointwise_dynamic only charges overhead: it
# recomputes a runtime index (div/mod + stride multiply-add + broadcast
# bookkeeping) for every element.  On device the cost is independent of the
# dtype and of the number of bytes moved (2**30 elements: fp16 70.43ms,
# bf16 70.41ms, fp32 69.25ms) while no pipeline is saturated
# (aiv_vec_ratio=0.546, aiv_scalar_ratio=0.220), i.e. the scalar / addressing
# path dominates throughput, not the loads and not exp2/log2.
#
# The kernel below keeps the exact same math and stability handling but
# replaces the per-element dynamic index with a single `pid * BLOCK + arange`,
# and drops the compare+predicated access entirely when the extent divides
# evenly.  It deliberately reuses the pointwise_dynamic generated kernel name
# (logaddexp2_func_kernel_rank_1) so name based profiler attribution of this
# operator is unchanged.

_FLAT_DTYPE_KIND = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

# Keeps every element offset (and pid * BLOCK) inside int32, and keeps a
# masked tail from wrapping around into in-range offsets.
_FLAT_MAX_ELEMENTS = 1 << 30
_FLAT_MAX_BLOCK = 1024


def _pick_flat_block(n_elements):
    """Power-of-two block that keeps at least ~32 programs for small shapes.

    Tiny shapes must not collapse into one or two programs (that turns a
    launch-bound ~2us case into a regression), while large shapes want the
    biggest block that still fits the UB budget of the FP32 temporaries.
    """
    if n_elements <= 4096:
        return 128
    if n_elements <= 32768:
        return 256
    if n_elements <= 262144:
        return 512
    return _FLAT_MAX_BLOCK


@triton.jit
def logaddexp2_func_kernel_rank_1(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    OUT_KIND: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if HAS_MASK:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(x_ptr + offs).to(tl.float32)
        y = tl.load(y_ptr + offs).to(tl.float32)

    # log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|)), m = max(x, y)
    m = tl.maximum(x, y)
    delta = x - y
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    # inf - inf makes `delta` NaN; fall back to max(x, y) there.
    res = tl.where(delta != delta, m, res)
    # Genuine NaN inputs must still propagate NaN.
    res = tl.where((x != x) | (y != y), float("nan"), res)

    # OUT_KIND: 0 = fp32 (store as is), 1 = fp16, 2 = bf16.
    if OUT_KIND == 0:
        if HAS_MASK:
            tl.store(out_ptr + offs, res, mask=mask)
        else:
            tl.store(out_ptr + offs, res)
    elif OUT_KIND == 1:
        res16 = res.to(tl.float16)
        if HAS_MASK:
            tl.store(out_ptr + offs, res16, mask=mask)
        else:
            tl.store(out_ptr + offs, res16)
    else:
        resbf = res.to(tl.bfloat16)
        if HAS_MASK:
            tl.store(out_ptr + offs, resbf, mask=mask)
        else:
            tl.store(out_ptr + offs, resbf)


def _flat_applicable(a, b, out=None):
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return False
    if a.dtype not in _FLAT_DTYPE_KIND or b.dtype != a.dtype:
        return False
    if a.shape != b.shape:
        return False
    if not (a.is_contiguous() and b.is_contiguous()):
        return False
    n_elements = a.numel()
    if n_elements <= 0 or n_elements > _FLAT_MAX_ELEMENTS:
        return False
    if out is not None:
        if not isinstance(out, torch.Tensor):
            return False
        if out.dtype != a.dtype or out.shape != a.shape:
            return False
        if not out.is_contiguous():
            return False
    return True


def _launch_flat(a, b, out):
    n_elements = a.numel()
    block = _pick_flat_block(n_elements)
    grid = (triton.cdiv(n_elements, block),)
    logaddexp2_func_kernel_rank_1[grid](
        a,
        b,
        out,
        n_elements,
        BLOCK=block,
        HAS_MASK=(n_elements % block) != 0,
        OUT_KIND=_FLAT_DTYPE_KIND[a.dtype],
    )


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if _flat_applicable(self, other):
        # torch.empty (not empty_like) keeps the allocation on the same base
        # format as the generic path.
        out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        try:
            _launch_flat(self, other, out)
            return out
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("GEMS LOGADDEXP2 flat path failed (%s); fallback", exc)
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    if _flat_applicable(self, other, out):
        try:
            _launch_flat(self, other, out)
            return out
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("GEMS LOGADDEXP2_OUT flat path failed (%s); fallback", exc)
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

run = logaddexp2

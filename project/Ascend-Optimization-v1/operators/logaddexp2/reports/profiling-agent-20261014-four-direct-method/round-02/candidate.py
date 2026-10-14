import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.triton_lang_utils import triton_lang_utils as _tlu  # noqa: F401

logger = logging.getLogger(__name__)


@triton.jit
def _logaddexp2_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    TILE: tl.constexpr,
    VIEW: tl.constexpr,
):
    base = tl.program_id(0) * (TILE * VIEW)
    for i in tl.static_range(VIEW):
        offs = base + i * TILE + tl.arange(0, TILE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        y_f32 = y.to(tl.float32)
        m = tl.maximum(x_f32, y_f32)
        delta = x_f32 - y_f32
        res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
        # equal infinities make delta NaN; result must fall back to m
        res = tl.where(delta != delta, m, res)
        is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
        res = tl.where(is_nan, float("nan"), res)
        tl.store(out_ptr + offs, res, mask=mask)


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


def _launch_flat_1d(a, b, out):
    """Return True when the flat contiguous kernel handled the tensors."""
    if not (a.is_npu and b.is_npu and out.is_npu):
        return False
    if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
        return False
    if a.shape != b.shape or a.shape != out.shape:
        return False
    if a.dim() != 1:
        return False
    if a.dtype != b.dtype:
        return False
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    n = out.numel()
    if n <= 0:
        return True

    elem_bytes = out.element_size()
    # Tile/loop choice follows dtype: fp32 needs the smaller tile to keep the
    # working set small, half precision can afford a wider one.
    if elem_bytes >= 4:
        tile, view = 8192, 4
    else:
        tile, view = 16384, 4
    span = tile * view
    grid = (triton.cdiv(n, span),)
    if grid[0] <= 0 or grid[0] > 65535:
        return False
    _logaddexp2_flat_kernel[grid](a, b, out, n, TILE=tile, VIEW=view, num_warps=1)
    return True


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if (
        isinstance(self, torch.Tensor)
        and isinstance(other, torch.Tensor)
        and not torch.is_grad_enabled()
        and self.is_contiguous()
        and other.is_contiguous()
        and self.shape == other.shape
        and self.dtype == other.dtype
        and self.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        out = torch.empty_like(self)
        if _launch_flat_1d(self, other, out):
            return out
        return logaddexp2_func(self, other)
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    if (
        isinstance(self, torch.Tensor)
        and isinstance(other, torch.Tensor)
        and isinstance(out, torch.Tensor)
        and not torch.is_grad_enabled()
        and self.is_contiguous()
        and other.is_contiguous()
        and self.shape == other.shape
        and self.dtype == other.dtype
        and self.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        if _launch_flat_1d(self, other, out):
            return out
        logaddexp2_func(self, other, out0=out)
        return out
    logaddexp2_func(self, other, out0=out)
    return out


run = logaddexp2

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""基于独立Agent扁平候选修复编译常量与超限网格，保留通用回退。"""
import torch
import triton
import triton.language as tl
from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp2_general(x, y):
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    m = tl.maximum(xf, yf)
    d = xf - yf
    r = m + tl.log2(1.0 + tl.exp2(-tl.abs(d)))
    r = tl.where(d != d, m, r)
    return tl.where((xf != xf) | (yf != yf), float("nan"), r)


@triton.jit
def logaddexp2_bounded_kernel(X, Y, O, n, BLOCK: tl.constexpr):
    start = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    for base in tl.range(start, n, step):
        offsets = base + tl.arange(0, BLOCK)
        mask = offsets < n
        xf = tl.load(X + offsets, mask=mask, other=0).to(tl.float32)
        yf = tl.load(Y + offsets, mask=mask, other=0).to(tl.float32)
        m = tl.maximum(xf, yf)
        d = xf - yf
        r = m + tl.log(1.0 + tl.exp(-tl.abs(d) * 0.6931471805599453)) * 1.4426950408889634
        r = tl.where(d != d, m, r)
        r = tl.where((xf != xf) | (yf != yf), float("nan"), r)
        tl.store(O + offsets, r.to(O.dtype.element_ty), mask=mask)


def run(self, other):
    if (isinstance(self, torch.Tensor) and isinstance(other, torch.Tensor)
            and self.dtype == other.dtype and self.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and self.shape == other.shape and self.device == other.device
            and self.is_contiguous() and other.is_contiguous()
            and self.numel() < (1 << 31)):
        out = torch.empty_like(self)
        n = self.numel()
        if n:
            block = 256 if n <= 65536 else 4096
            grid = min(triton.cdiv(n, block), 4096)
            logaddexp2_bounded_kernel[(grid,)](self, other, out, n, BLOCK=block)
        return out
    return logaddexp2_general(self, other)

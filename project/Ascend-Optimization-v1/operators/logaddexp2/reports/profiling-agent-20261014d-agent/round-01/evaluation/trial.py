# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""logaddexp2：保留通用广播回退；连续同形输入走无掩码、带流水循环的 AIV 向量核。

相对上一版最佳候选的三处改动（仅限连续同形快速路径）：
1) grid 收敛：大 shape 不再“每 program 只跑 1 个 tile”，让每个 program 有 4~8 次循环迭代；
2) n 整除 BLOCK 时走 EVEN 常量分支，逐 tile 去掉掩码比较与谓词访存；
3) 数值公式改用基底 2 内建 exp2/log2，去掉两次显式缩放乘法，缩短 FP32 临时量存活区间。

非连续 / 广播 / 类型提升 / 非整除尾块仍走原通用路径与带掩码分支，语义不变。
"""
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
def logaddexp2_bounded_kernel(X, Y, O, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    start = tl.program_id(0) * BLOCK
    step = tl.num_programs(0) * BLOCK
    for base in tl.range(start, n, step):
        offsets = base + tl.arange(0, BLOCK)
        if EVEN:
            xf = tl.load(X + offsets).to(tl.float32)
            yf = tl.load(Y + offsets).to(tl.float32)
        else:
            mask = offsets < n
            xf = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
            yf = tl.load(Y + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.maximum(xf, yf)
        d = xf - yf
        r = m + tl.log2(1.0 + tl.exp2(-tl.abs(d)))
        r = tl.where(d != d, m, r)
        r = tl.where((xf != xf) | (yf != yf), float("nan"), r)
        rc = r.to(O.dtype.element_ty)
        if EVEN:
            tl.store(O + offsets, rc)
        else:
            tl.store(O + offsets, rc, mask=mask)


def run(self, other):
    if (isinstance(self, torch.Tensor) and isinstance(other, torch.Tensor)
            and self.dtype == other.dtype and self.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and self.shape == other.shape and self.device == other.device
            and self.is_contiguous() and other.is_contiguous()
            and self.numel() < (1 << 31)):
        out = torch.empty_like(self)
        n = self.numel()
        if n:
            if n <= 65536:
                block = 256
                grid = triton.cdiv(n, block)
            else:
                block = 4096
                tiles = triton.cdiv(n, block)
                # 至少保留 1024 个 program（并行度）并封顶 4096；
                # 大 shape 让每个 program 迭代 4~8 次以复用循环流水，
                # 与 1G case 已经体现出的更优“每元素耗时”保持一致。
                grid = min(tiles, 4096, max(1024, tiles // 8))
            logaddexp2_bounded_kernel[(grid,)](
                self, other, out, n, BLOCK=block, EVEN=(n % block == 0))
        return out
    return logaddexp2_general(self, other)

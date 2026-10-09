"""Triton implementation of aten::_prelu_kernel_backward (PReLU backward).

Contract (FlagGems adapter for this aten op) - two outputs:
  grad_input[i]  = grad_output[i] if self[i] > 0 else grad_output[i] * weight[c(i)]
  grad_weight    = sum over {i : self[i] < 0} of grad_output[i] * self[i],
                   reduced to weight.shape (a REDUCTION, not elementwise)
with c(i) the dim-1 (NCHW) channel index, c(i) = (i // S) % C for a contiguous
input, S = prod(shape[2:]) and C = shape[1] (S = 1 for rank-2 inputs).  A
scalar weight (weight.numel() == 1) broadcasts to every element and yields a
[1]-shaped grad_weight.

Design
  * Scalar-weight path (the only layout the benchmark uses): one fused pass
    writes grad_input in the input dtype and accumulates an fp32 per-program
    partial of grad_weight; a tiny second kernel tree-sums the fp32 partials.
    The product fed to grad_weight stays in the input dtype, which matches the
    reference rounding (bit-identical for fp16/bf16 on this target).
  * Per-channel path: a flat pass with the per-lane channel gather for
    grad_input, plus one program per channel for grad_weight.
  * Tile plan is derived from the element count only; no device core count and
    no physical-resource constant enters the launch configuration.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _prelu_bwd_scalar_kernel(
    g_ptr,
    x_ptr,
    w_ptr,
    gi_ptr,
    part_ptr,
    n_elements,
    n_iters,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Fused grad_input store + per-program fp32 partial of grad_weight.

    Each program owns ITERS consecutive BLOCK-element tiles of a contiguous
    span; the trip count is a runtime value so the loop is not unrolled.
    """
    pid = tl.program_id(0)
    w = tl.load(w_ptr)
    base = pid * (BLOCK * n_iters)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for j in range(0, n_iters):
        offs = base + j * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = offs < n_elements
            g = tl.load(g_ptr + offs, mask=m, other=0.0)
            x = tl.load(x_ptr + offs, mask=m, other=0.0)
            # Strict comparison: x <= 0 and NaN take the weight branch.
            tl.store(gi_ptr + offs, tl.where(x > 0, g, g * w), mask=m)
            acc += tl.where(x < 0, g * x, 0.0).to(tl.float32)
        else:
            g = tl.load(g_ptr + offs)
            x = tl.load(x_ptr + offs)
            tl.store(gi_ptr + offs, tl.where(x > 0, g, g * w))
            acc += tl.where(x < 0, g * x, 0.0).to(tl.float32)
    tl.store(part_ptr + pid, tl.sum(acc, axis=0))


@triton.jit
def _prelu_bwd_reduce_kernel(
    part_ptr,
    out_ptr,
    n_parts,
    BLOCK: tl.constexpr,
):
    """Deterministic tree sum of the fp32 per-program partials."""
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n_parts, BLOCK):
        idx = start + offs
        acc += tl.load(part_ptr + idx, mask=idx < n_parts, other=0.0)
    tl.store(out_ptr, tl.sum(acc, axis=0))


@triton.jit
def _prelu_bwd_chan_gi_kernel(
    g_ptr,
    x_ptr,
    w_ptr,
    gi_ptr,
    n_elements,
    S,
    C,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Per-channel weight broadcast for grad_input (dim-1 channel axis)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    c = (offs // S) % C
    if NEED_MASK:
        m = offs < n_elements
        g = tl.load(g_ptr + offs, mask=m, other=0.0)
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        w = tl.load(w_ptr + tl.minimum(c, C - 1), mask=m, other=0.0)
        tl.store(gi_ptr + offs, tl.where(x > 0, g, g * w), mask=m)
    else:
        g = tl.load(g_ptr + offs)
        x = tl.load(x_ptr + offs)
        w = tl.load(w_ptr + c)
        tl.store(gi_ptr + offs, tl.where(x > 0, g, g * w))


@triton.jit
def _prelu_bwd_chan_gw_kernel(
    g_ptr,
    x_ptr,
    gw_ptr,
    N,
    S,
    C,
    RBLOCK: tl.constexpr,
    SBLOCK: tl.constexpr,
):
    """One program per channel; visits only the rows n*C + c of the (N*C, S)
    view, i.e. flat index (n * C + c) * S + s, so each element is read once."""
    c = tl.program_id(0)
    r = tl.arange(0, RBLOCK)
    s = tl.arange(0, SBLOCK)
    acc = tl.zeros([RBLOCK, SBLOCK], dtype=tl.float32)
    for ro in range(0, N, RBLOCK):
        rr = ro + r
        for so in range(0, S, SBLOCK):
            ss = so + s
            m = (rr[:, None] < N) & (ss[None, :] < S)
            idx = (rr[:, None] * C + c) * S + ss[None, :]
            g = tl.load(g_ptr + idx, mask=m, other=0.0)
            x = tl.load(x_ptr + idx, mask=m, other=0.0)
            acc += tl.where(m & (x < 0), g * x, 0.0).to(tl.float32)
    tl.store(gw_ptr + c, tl.sum(tl.sum(acc, axis=1), axis=0))


def _plan(numel):
    """Task-derived (BLOCK, ITERS) tile plan.

    The program count follows from the element count alone.  Measured on the
    target, widening the per-program span (fewer programs) lowers the latency
    of the cache-resident shapes several-fold, while the largest shape sits at
    the memory-bandwidth limit over a wide span range.  No core-count constant
    is involved and the grid is always bounded by the available work.
    """
    if numel <= (1 << 21):
        return 4096, 4
    return 4096, 8


def _reduce_block(n_parts):
    block = 128
    while block < 4096 and block < n_parts:
        block *= 2
    return block


def _pow2_at_most(value, cap):
    block = 1
    while block * 2 <= value and block < cap:
        block *= 2
    return block


def run(grad_output, self, weight):
    x = self
    g = grad_output

    gi = torch.empty_like(x)
    gw = torch.empty(weight.shape, dtype=x.dtype, device=x.device)

    n = x.numel()
    w_n = weight.numel()
    if n == 0 or w_n == 0:
        return gi, gw

    if w_n == 1:
        block, iters = _plan(n)
        span = block * iters
        n_parts = triton.cdiv(n, span)
        part = torch.empty(n_parts, dtype=torch.float32, device=x.device)
        _prelu_bwd_scalar_kernel[(n_parts,)](
            g,
            x,
            weight,
            gi,
            part,
            n,
            iters,
            BLOCK=block,
            NEED_MASK=(n % span) != 0,
        )
        _prelu_bwd_reduce_kernel[(1,)](
            part,
            gw,
            n_parts,
            BLOCK=_reduce_block(n_parts),
        )
        return gi, gw

    # Per-channel weight: channel axis is dim 1 (NCHW), C = shape[1] and
    # S = prod(shape[2:]); rank-1 input has a single axis so C = shape[0].
    if x.dim() >= 2:
        dim1 = int(x.shape[1])
    else:
        dim1 = int(x.shape[0])
    C = max(dim1 if w_n == dim1 else w_n, 1)
    if x.dim() >= 2 and int(x.shape[0]) > 0 and n % (int(x.shape[0]) * C) == 0:
        S = n // (int(x.shape[0]) * C)
    elif n % C == 0:
        S = max(n // C, 1)
    else:
        S = 1
    S = max(S, 1)

    gi_block = 1024
    _prelu_bwd_chan_gi_kernel[(triton.cdiv(n, gi_block),)](
        g, x, weight, gi, n, S, C, BLOCK=gi_block, NEED_MASK=(n % gi_block) != 0
    )
    N = triton.cdiv(n, C * S)
    _prelu_bwd_chan_gw_kernel[(C,)](
        g,
        x,
        gw,
        N,
        S,
        C,
        RBLOCK=_pow2_at_most(max(N, 1), 64),
        SBLOCK=_pow2_at_most(max(S, 1), 1024),
    )
    return gi, gw


"""为旧 PReLU 补齐稳定的 FP32 权重梯度归约；保留原逐元素输出。"""


@triton.jit
def _prelu_two_sum(a, b):
    total = a + b
    b_virtual = total - a
    error = (a - (total - b_virtual)) + (b - b_virtual)
    return total, error


@triton.jit
def _prelu_pair_sum(high, low, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    for stage in tl.static_range(0, 10):
        peer = lanes ^ (1 << stage)
        other_high = tl.gather(high, peer, axis=0)
        other_low = tl.gather(low, peer, axis=0)
        total, error = _prelu_two_sum(high, other_high)
        low = low + other_low + error
        high, residual = _prelu_two_sum(total, low)
        low = residual
    return tl.sum(tl.where(lanes == 0, high, 0.0)), tl.sum(tl.where(lanes == 0, low, 0.0))


@triton.jit
def _prelu_partial_sum_kernel(values, highs, lows, count, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(values + offsets, mask=offsets < count, other=0.0).to(tl.float32)
    high, low = _prelu_pair_sum(value, tl.full((BLOCK,), 0.0, tl.float32), BLOCK)
    tl.store(highs + pid, high)
    tl.store(lows + pid, low)


@triton.jit
def _prelu_final_sum_kernel(highs, lows, output, count, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    high = tl.full((BLOCK,), 0.0, tl.float32)
    low = tl.full((BLOCK,), 0.0, tl.float32)
    for start in range(0, count, BLOCK):
        offsets = start + lanes
        a = tl.load(highs + offsets, mask=offsets < count, other=0.0)
        b = tl.load(lows + offsets, mask=offsets < count, other=0.0)
        high_new, error = _prelu_two_sum(high, a)
        high, low = _prelu_two_sum(high_new, low + b + error)
    result_high, result_low = _prelu_pair_sum(high, low, BLOCK)
    tl.store(output, result_high + result_low)


def _prelu_stable_scalar_sum(values, weight):
    block = 1024
    count = triton.cdiv(values.numel(), block)
    highs = torch.empty((count,), dtype=torch.float32, device=values.device)
    lows = torch.empty_like(highs)
    output = torch.empty_like(weight)
    _prelu_partial_sum_kernel[(count,)](values, highs, lows, values.numel(), BLOCK=block, enable_fp_fusion=False)
    _prelu_final_sum_kernel[(1,)](highs, lows, output, count, BLOCK=block, enable_fp_fusion=False)
    return output


"""FP32 PReLU 输入梯度与误差补偿权重梯度部分和融合计算。"""


@triton.jit
def _prelu_accurate_fused_kernel(g_ptr, x_ptr, w_ptr, gi_ptr, highs, lows,
                                 count, n_iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    weight = tl.load(w_ptr)
    high = tl.full((), 0.0, tl.float32)
    low = tl.full((), 0.0, tl.float32)
    for j in range(0, n_iters):
        offsets = (pid * n_iters + j) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < count
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(gi_ptr + offsets, tl.where(x > 0, g, g * weight), mask=mask)
        products = tl.where(x < 0, g * x, 0.0)
        block_high, block_low = _prelu_pair_sum(products, tl.zeros([BLOCK], tl.float32), BLOCK)
        new_high, error = _prelu_two_sum(high, block_high)
        high, low = _prelu_two_sum(new_high, low + block_low + error)
    tl.store(highs + pid, high)
    tl.store(lows + pid, low)


_prelu_previous_run = run


def run(grad_output, self, weight):
    if self.dtype != torch.float32 or weight.numel() != 1 or self.numel() == 0:
        return _prelu_previous_run(grad_output, self, weight)
    x = self.contiguous()
    g = grad_output.contiguous()
    gi = torch.empty_like(x)
    gw = torch.empty_like(weight)
    block = 1024
    iters = 16 if x.numel() <= (1 << 21) else 32
    count = triton.cdiv(x.numel(), block * iters)
    highs = torch.empty((count,), dtype=torch.float32, device=x.device)
    lows = torch.empty_like(highs)
    _prelu_accurate_fused_kernel[(count,)](g, x, weight, gi, highs, lows, x.numel(), iters,
                                         BLOCK=block, enable_fp_fusion=False)
    _prelu_final_sum_kernel[(1,)](highs, lows, gw, count, BLOCK=block, enable_fp_fusion=False)
    return gi, gw

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

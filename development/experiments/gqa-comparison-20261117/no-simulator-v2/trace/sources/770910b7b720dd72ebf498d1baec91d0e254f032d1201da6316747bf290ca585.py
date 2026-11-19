"""Grouped-query attention (BSND) as a single pure-Triton kernel.

Public entrypoint: run(query, key, value, scaleValue=-1.0, is_causal=False)

Semantics (see task-description.md):
    y[b, i, hq, :] = softmax_j(Q[b, i, hq, :] . K[b, j, g, :] * scale) V[b, j, g, :]
    g = hq * N_kv // N_q,  scale = 1/sqrt(D) when scaleValue <= 0
    causal: j > i + (S_kv - S) is set to -inf before softmax (bottom-right aligned)

Every device computation below is Triton: QK^T and PV are tl.dot, the softmax
(max / sum / rescaling) runs in fp32 registers, and the only Python-side work is
metadata (shape / stride), the torch.empty output allocation and the launch.
KV heads are never materialised to N_q heads: each program indexes the shared
K/V head hq // G directly.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_fwd_kernel(
    Q, K, V, Out,
    q_sb, q_ss, q_sh, q_sd,
    k_sb, k_ss, k_sh, k_sd,
    v_sb, v_ss, v_sh, v_sd,
    o_sb, o_ss, o_sh, o_sd,
    S, SKV, D, NQ, G,
    qk_scale, delta,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    P_TERMS: tl.constexpr,   # 2 for fp16 probabilities, 3 for bf16 probabilities
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // NQ
    hq = pid_bh % NQ
    kvh = hq // G          # shared KV head; no materialisation of G copies

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < S
    d_mask = offs_d < D

    q_ptrs = Q + b * q_sb + hq * q_sh + offs_m[:, None] * q_ss + offs_d[None, :] * q_sd
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    k_base = K + b * k_sb + kvh * k_sh
    v_base = V + b * v_sb + kvh * v_sh

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    # Bottom-right aligned causal mask: valid j <= i + (SKV - S).  Rows of this
    # block are at most pid_m*BLOCK_M + BLOCK_M - 1, so later KV blocks are fully
    # masked and are skipped entirely.
    if IS_CAUSAL:
        hi = tl.minimum(SKV, pid_m * BLOCK_M + BLOCK_M + delta)
    else:
        hi = SKV

    for n0 in range(0, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < SKV

        k_ptrs = k_base + offs_n[:, None] * k_ss + offs_d[None, :] * k_sd
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale   # fp32 accumulation

        if IS_CAUSAL:
            qk = tl.where(offs_n[None, :] <= offs_m[:, None] + delta, qk, -float("inf"))
        qk = tl.where(n_mask[None, :], qk, -float("inf"))

        # Online softmax, all reductions in fp32.
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        # Fully masked row guard (cannot happen when S <= SKV, kept for safety).
        m_safe = tl.where(m_new > -float("inf"), m_new, 0.0)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v_ptrs = v_base + offs_n[:, None] * v_ss + offs_d[None, :] * v_sd
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # Precision-first PV: the softmax probabilities are fp32, and rounding
        # them directly to v.dtype before tl.dot injects an absolute error of
        # ~u * sum_j p_j |v_j| (u = 2^-11 fp16 / 2^-8 bf16).  For outputs formed
        # by cancellation (near zero) that error dominates the relative metric.
        # Represent p as a multi-term expansion in the dot dtype and accumulate
        # every product in fp32.  p_hi is the rounded high part, the residual
        # r = p - fp32(p_hi) is exact in fp32, and each successive residual is
        # scaled by 2048 before rounding (so it stays in the normal range, i.e.
        # no fp16 subnormal flush) and scaled back afterwards:
        #   P_TERMS = 2:  p = p_hi + p_lo/2048 + O(u^2 p)      (fp16: 2^-22)
        #   P_TERMS = 3:  p = p_hi + p_lo/2048 + p_lo2/2048^2 + O(u^3 p)  (bf16)
        # bf16 needs the third term: its unit roundoff u = 2^-8 leaves a
        # u^2 ~ 1.5e-5 relative probability error with two terms, which is far
        # too coarse for cancellation-dominated near-zero outputs (measured
        # BF16 MARE 1.33 vs the 3.9e-3 nearest-BF16 floor).  Three bf16 terms
        # carry 3*8 mantissa bits, so the residual u^3 ~ 6e-8 is at or below the
        # fp32 accumulation/rounding level of the surrounding arithmetic.
        # V is already exact in v.dtype (it is an input), so only p is split.
        # The low-order dot is added last so it is not rounded against the
        # O(sum_j p_j |v_j|) accumulator.  Everything stays strict Triton.
        p_hi = p.to(v.dtype)
        r1 = (p - p_hi.to(tl.float32)) * 2048.0
        p_lo = r1.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(p_hi, v) + tl.dot(p_lo, v) * (1.0 / 2048.0)
        if P_TERMS == 3:
            p_lo2 = ((r1 - p_lo.to(tl.float32)) * 2048.0).to(v.dtype)
            acc += tl.dot(p_lo2, v) * (1.0 / (2048.0 * 2048.0))
        m_i = m_safe

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]
    out = tl.where(l_i[:, None] > 0.0, out, 0.0)  # all-masked rows -> zeros

    o_ptrs = Out + b * o_sb + hq * o_sh + offs_m[:, None] * o_ss + offs_d[None, :] * o_sd
    tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :])


def run(query, key, value, scaleValue=-1.0, is_causal=False):
    B, S, NQ, D = query.shape
    SKV = key.shape[1]
    NKV = key.shape[2]
    G = NQ // NKV

    if scaleValue is None or scaleValue <= 0:
        scale = 1.0 / math.sqrt(D)
    else:
        scale = float(scaleValue)
    # exp2 based softmax keeps the scaling neutral in fp32.
    qk_scale = scale * 1.4426950408889634

    y = torch.empty((B, S, NQ, D), dtype=query.dtype, device=query.device)

    # Probability-representation term count, chosen per dot dtype only (a pure
    # metadata decision on the host): bf16's u = 2^-8 needs a third term to get
    # the residual down to the fp32 level, while fp16's two-term residual
    # (u^2 = 2^-22) is already at its nearest-representable floor, so its code
    # path is left exactly as it was (no extra dot, no perf cost).
    p_terms = 3 if query.dtype == torch.bfloat16 else 2

    # UB budget: acc is BLOCK_M x BLOCK_D fp32, plus q/k/v/p tiles.
    BLOCK_D = max(16, triton.next_power_of_2(D))
    if BLOCK_D <= 128:
        BLOCK_M, BLOCK_N = 64, 64
    elif BLOCK_D <= 256:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 16, 32

    grid = (triton.cdiv(S, BLOCK_M), B * NQ)
    _gqa_fwd_kernel[grid](
        query, key, value, y,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        S, SKV, D, NQ, G,
        qk_scale, SKV - S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=bool(is_causal),
        P_TERMS=p_terms,
        num_warps=1, num_stages=1,
    )
    return y

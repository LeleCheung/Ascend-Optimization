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

        # Precision-first PV with a two-term split of the fp32 probabilities.
        # Rounding p directly to v.dtype injects an absolute error of
        # ~u * sum_j p_j |v_j| (u = 2^-11 fp16 / 2^-9 bf16), which dominates the
        # relative metric for outputs formed by cancellation.  Instead keep
        #   p = p_hi + p_lo / 2048 + O(u^2 p)
        # where p_hi is p rounded to the dot dtype, r1 = 2048 * (p - p_hi) is the
        # residual pushed into the normal range before rounding (no subnormal
        # flush), and 1/2048 is an exact power of two so the rescaling is exact.
        # Two terms carry 2 * mantissa bits of p: 2^-22 relative for fp16 and
        # 2^-18 for bf16, both below the fp32 rounding of the surrounding
        # softmax/PV arithmetic.  A further bf16 residual term would contribute
        # only ~2^-29 * p (well under that floor) while costing a whole extra
        # PV tl.dot, i.e. one more cube->vector accumulator handoff (L0C movout
        # + flag round trip + fp32 read-back) per KV block.  So both dtypes use
        # the same two-dot form.
        # The low-order product is added last in fp32 so it is not rounded
        # against the O(sum_j p_j |v_j|) accumulator; V is an exact input, so
        # only p is split.  Everything stays strict Triton.
        p_hi = p.to(v.dtype)
        r1 = (p - p_hi.to(tl.float32)) * 2048.0
        p_lo = r1.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(p_hi, v) + tl.dot(p_lo, v) * (1.0 / 2048.0)
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

    # UB budget: acc is BLOCK_M x BLOCK_D fp32, plus q/k/v/p tiles.
    # KV 方向分块取 128：每个 KV 块固定要付一次完整交接链（QK dot 搬出、
    # 向量 softmax 与两项分解、p_hi/p_lo 搬入 L1、两次 PV dot 其中一次以
    # acc*alpha 作 L0C 初值、两次 L0C 搬出、向量合并），该链与算术量无关。
    # 分块由 64 提到 128 后，非因果 case2 退化为单迭代，case4/7 迭代数 4/2
    # 减半，每程序 PV 交接、acc 重缩放与 K/V 装入次数同比例减半。
    #
    # 资源约束（实际失败IR）：因果分支在循环内额外生成与 BLOCK_N 同宽的
    # 整型索引/掩码临时量，把 64/128/128 的峰值推到 231488B，超过 196608B
    # 预算；非因果编译的同一分块没有该临时量，故三项非因果 case 可以编译。
    # 因此分块按 is_causal 分流：因果仍取 BLOCK_M=64 但 BLOCK_N=64，使所有
    # BLOCK_N 线性相关的缓冲（k/v、qk/p、p_hi/p_lo 与因果索引临时量）减半，
    # 峰值估算约 150KB。代价仅限 case1：每程序 KV 迭代 2->3，PV 交接与
    # acc 重缩放次数 +50%，被掩码列减少令向量 softmax 元素 -25%，grid 与
    # 并行度不变；非因果三个 case 的分块与性能路径不变。
    #
    # 本轮机制（交接次数而非算术量）：case4 的每 program 成本由
    # “3 次 L0C->workspace->UB 交接 x 轮数”主导（L0C_to_GM 192KiB、
    # GM_to_UB 192KiB、Cube 每 program 等待 10.08us），而 SKV<=256 的非因果
    # case 完全可以一轮覆盖。故非因果把 BLOCK_N 提到 min(256, next_pow2(SKV))：
    # case4/7 变成单轮 256 列，交接 6->3 次、acc 重缩放与两次 PV 的向量合并
    # 各减半。为遵守 UB 上限（因果 231488B 的实测失败），把 BLOCK_M 同步减半
    # 到 32，使所有随 M×N 增长的张量仍是 8192 个元素，与当前可编译配置同量级。
    # 代价：grid 与并发的 program 数翻倍（case4 1024->2048），每个 program 仍
    # 完整装入其 KV 头，K/V 的 GM->L1 装入量约翻倍（MTE2 当前非瓶颈）；
    # BLOCK_M=32 使每次 dot 的 M 维变短。case2 的 SKV=128 仍取 BLOCK_N=128 /
    # BLOCK_M=64，与原分块一致，可作为不受机制影响的对照。
    # 两项概率分解的表达式、相加顺序与 dtype 保持不变；数值差异仅来自在线
    # softmax 由“两段 128 列 + 一次重缩放”合并为“一段 256 列”，max/sum 仍是
    # fp32 归约，重缩放由 2 次减为 1 次，误差不增。
    BLOCK_D = max(16, triton.next_power_of_2(D))
    if BLOCK_D <= 128:
        if is_causal:
            BLOCK_M, BLOCK_N = 64, 64
        else:
            kv_extent = triton.next_power_of_2(SKV)
            BLOCK_N = min(256, max(64, kv_extent))
            BLOCK_M = 64 if BLOCK_N <= 128 else 32
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
        num_warps=1, num_stages=1,
    )
    return y

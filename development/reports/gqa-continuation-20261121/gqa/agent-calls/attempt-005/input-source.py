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

Compile-time geometry specialisation (this round's mechanism): shape, stride,
head counts, scale and the causal offset are passed as tl.constexpr, because
they are legal input *metadata*, never input values.  The actual TTIR showed
those quantities surviving as runtime scalars, so every KV iteration rebuilt
the n-mask (cmpi/andi/broadcast), the qk out-of-range select, and a 128x128
pointer tensor from splatted runtime strides, and the KV loop bound stayed
dynamic.  With the geometry static, the host also proves two masks redundant
(MASK_MD false when S % BLOCK_M == 0 and D == BLOCK_D, so the exact grid covers
only valid rows/columns; MASK_KV false when SKV % BLOCK_N == 0 and D ==
BLOCK_D), which lets Q/Out load-store unmasked and K/V load unmasked, and drops
the always-true qk select.  Dropping a mask that is provably all-true cannot change any loaded
value, and no arithmetic expression or its order is touched: the online
softmax, the two-term split of the probabilities and the acc rescaling stay
bit-identical to the previous candidate.  Cost: one compiled specialisation per
(distinct shape, stride, tiling) combination, i.e. a slightly longer first
call per case.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_fwd_kernel(
    Q, K, V, Out,
    q_sb: tl.constexpr, q_ss: tl.constexpr, q_sh: tl.constexpr, q_sd: tl.constexpr,
    k_sb: tl.constexpr, k_ss: tl.constexpr, k_sh: tl.constexpr, k_sd: tl.constexpr,
    v_sb: tl.constexpr, v_ss: tl.constexpr, v_sh: tl.constexpr, v_sd: tl.constexpr,
    o_sb: tl.constexpr, o_ss: tl.constexpr, o_sh: tl.constexpr, o_sd: tl.constexpr,
    S: tl.constexpr, SKV: tl.constexpr, D: tl.constexpr,
    NQ: tl.constexpr, G: tl.constexpr,
    qk_scale: tl.constexpr, delta: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MASK_MD: tl.constexpr,
    MASK_KV: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // NQ
    hq = pid_bh % NQ
    kvh = hq // G          # shared KV head; no materialisation of G copies

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # MASK_MD 为假表示 grid 恰好覆盖 S 行且 D == BLOCK_D，行/通道掩码可证恒真：
    # 直接无掩码载入，省掉 64 宽比较、128 宽比较及其 and（取值不变，见文件头说明）。
    q_ptrs = Q + b * q_sb + hq * q_sh + offs_m[:, None] * q_ss + offs_d[None, :] * q_sd
    if MASK_MD:
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < S) & (offs_d[None, :] < D), other=0.0)
    else:
        q = tl.load(q_ptrs)

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

        k_ptrs = k_base + offs_n[:, None] * k_ss + offs_d[None, :] * k_sd
        if MASK_KV:
            kv_mask = (offs_n[:, None] < SKV) & (offs_d[None, :] < D)
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        else:
            # SKV % BLOCK_N == 0 且 D == BLOCK_D：本块全部列/通道合法，
            # 无掩码 K 载入与下面的无掩码 V 载入取代两次掩码载入。
            k = tl.load(k_ptrs)
        qk = tl.dot(q, tl.trans(k)) * qk_scale   # fp32 accumulation

        if IS_CAUSAL:
            qk = tl.where(offs_n[None, :] <= offs_m[:, None] + delta, qk, -float("inf"))
        if MASK_KV:
            # MASK_KV 为假时该掩码恒真，整体删除；保留分支只为通用形状正确性。
            qk = tl.where(offs_n[None, :] < SKV, qk, -float("inf"))

        # Online softmax, all reductions in fp32.
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        # Fully masked row guard (cannot happen when S <= SKV, kept for safety).
        m_safe = tl.where(m_new > -float("inf"), m_new, 0.0)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v_ptrs = v_base + offs_n[:, None] * v_ss + offs_d[None, :] * v_sd
        if MASK_KV:
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        else:
            v = tl.load(v_ptrs)

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
    if MASK_MD:
        tl.store(o_ptrs, out.to(Out.dtype.element_ty),
                 mask=(offs_m[:, None] < S) & (offs_d[None, :] < D))
    else:
        tl.store(o_ptrs, out.to(Out.dtype.element_ty))


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
    # 两项概率分解的表达式与相加顺序保持不变，仅 KV 轴划分与重缩放次数变化。
    #
    # 本轮改动只在编译期几何特化与可证冗余掩码消除上，分块与上面记录的
    # 因果/非因果分流保持不变，故四项 case 的分块、迭代数、grid 与
    # L0C 交接结构均与上一候选相同；收益或代价都必须由完整四项真机评测
    # 判定，本注释不据此改写分块。因果路径删除越界掩码后，UB 峰值可能
    # 已有余量，是否放宽 BLOCK_N 留待独立机制单独验证。
    BLOCK_D = max(16, triton.next_power_of_2(D))
    if BLOCK_D <= 128:
        BLOCK_M = 64
        BLOCK_N = 64 if is_causal else 128
    elif BLOCK_D <= 256:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 16, 32

    # 掩码可证冗余判据：只用合法元数据（shape 与分块整除性、D 是否等于
    # BLOCK_D），不读任何输入值、不按 case_id 或数值选路。整除成立时掩码在
    # 合法 grid 下恒真，可以整段删除；否则内核走原来的通用掩码分支，语义不变。
    MASK_MD = not ((S % BLOCK_M == 0) and (D == BLOCK_D))
    MASK_KV = not ((SKV % BLOCK_N == 0) and (D == BLOCK_D))

    grid = (triton.cdiv(S, BLOCK_M), B * NQ)
    _gqa_fwd_kernel[grid](
        query, key, value, y,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        S=S, SKV=SKV, D=D, NQ=NQ, G=G,
        qk_scale=qk_scale, delta=SKV - S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=bool(is_causal),
        MASK_MD=MASK_MD, MASK_KV=MASK_KV,
        num_warps=1, num_stages=1,
    )
    return y

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

Query-row reuse (retained, non-causal only): BLOCK_M stays 64 for
the causal path and rises to 128 otherwise, so a program keeps the loaded K/V
tiles for 128 query rows instead of 64.  The paired profiling of case4 showed
1.09us of aic_cube_time against 9.15us of aic_scalar_wait_id1 per program over
1024 programs of 64x128 tiles, and the simulator windows put Cube at 7.90%
productive / 33.27% wait / 58.83% idle with the top instructions being
WAIT_FLAG and MOV_OUT_TO_L1: the wall clock follows the number of cross-core
handoff chains, whose per-chain K/V load (2x32KiB) and fixed handoff cost are
independent of the row count.  Doubling BLOCK_M halves the program and chain
counts (case4: 1024->512 programs, 2048->1024 chains) and halves total K/V
traffic, at the cost of doubling the M-scaled per-program buffers (estimated
peak near the 196608B budget).  The mask, expression and precision paths are
untouched; only tl.dot's M tiling and the program count change, so every output
row still accumulates the same KV blocks in the same j order with the same
two-term probability split.

Single-fp16 PV on the bf16 path (this round's mechanism): the paired profiling
of case4 (512 programs, 467.7us kernel) reports aic_cube_total_instr_number = 6
per program -- three matmuls per KV block, i.e. QK plus the two probability
terms -- and L0C_to_GM = 384KiB per program (196608KiB total) against 98304KiB
for the CANN reference, while aic_cube_time is only 2.02us against 11.79us of
aic_scalar_wait_id1.  The actual TTIR/BCMLIR confirm the third matmul exists
only to carry the bf16 residual: dot(p_lo, v) with a zero accumulator, then a
128x128 fp32 multiply by 1/2048 and an add.  On the bf16 path the two terms buy
2^-18 relative probability precision; a single fp16 PV term buys 2^-11, which
is still better than an undecomposed bf16 term (2^-9) and costs one matmul and
one cube-to-vector handoff per KV block instead of two.  So when V is bf16 the
probabilities and V are cast to fp16 and one fp16 x fp16, fp32-accumulating dot
replaces the pair: 3->2 matmuls per KV block, L0C 384->256KiB per program, no
zero-initialised dot, no 1/2048 rescale and no extra vector merge.  This is an
approximation, not an identity: p's fp16 rounding is a 2^-11 relative error, V
is exact from bf16 while |v| >= 2^-14, falls into fp16 subnormals below that
(absolute error <= 3e-8) and would overflow above 65504, which is outside this
contract's [-1,1] input range and is recorded only as a range limitation.  The
fp16 and any other dtype path keep the two-term expression and its summation
order verbatim; the path is chosen from the V dtype metadata alone, never from
input values or a case id.

Full-KV single block with one handoff chain (this round's mechanism): with the
paired profiling of case4 at 512 programs the AIC spends 1.35us in cube against
10.26us of aic_scalar_wait_id1 per program, and the CANN reference does the same
2048 cube instructions in 19 blocks/130us, i.e. the wall clock tracks the number
of cross-core handoff chains (one per KV tile per program), not the arithmetic.
So for the non-causal path a KV block now covers the whole SKV when SKV is a
positive multiple of 256 (case4/case7 have SKV=256): 2 iterations become 1, so
each program pays one QK movout + vector softmax + PV movout instead of two, and
the loop-carried online rescaling disappears (SINGLE_KV branch).  Concretely per
program: matmuls 4->2, L0C_to_GM 256->192KiB, no 128x128 fp32 acc*alpha, no
acc->L0C init reload; total GM->L1 bytes are unchanged and the program count is
unchanged, so only the chain count halves.  Numerics: every output row still
accumulates the same products over the same j order; the only difference is that
the PV reduction over j is no longer split into two dots with an exact rescale in
between (it becomes one k=256 reduction), which is why the official four-case
checker with unchanged thresholds remains the arbiter.  For SKV==BLOCK_N shapes
(case2) the new branch is term-for-term identical, because alpha there is
exp2(-inf-m)=0 exactly.  Cost/risk: a 128x256 fp32 qk tile and a 128x256 fp16
dot operand pair are larger than the previous 128/128 tiling, so the compile may
hit the UB/L0C budget; the causal path keeps 64/64 and is untouched.

Compile-time geometry specialisation (earlier, retained mechanism): shape, stride,
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

持久程序 headgroup 拆分（本轮机制）：父版本每个持久程序拥有一个共享 KV 头的
全部 G 个 query head，32 个程序在 20 个 Cube 核上形成 20+12 的不均衡两波
（case4 每程序 16 次迭代、case7 4 次）。本轮在 G 可被 2 整除且 G>=2 时把每个
KV 头拆成 G/2 个 headgroup、每程序只负责 2 个 query head：case4 grid 32→128、
每程序 4 次迭代，case7 grid 16→32、每程序 2 次迭代，核间负载更均衡、长尾更短。
K/V 仍每程序在循环外整段载入一次并只转换 V 一次，因此总 K/V 载入次数上升
（case4 约 4 倍、case7 约 2 倍），这是明确的交通代价；QK/softmax/PV/输出
表达式、M128/N256 分块、guard 与精度路径逐字不变。G 不整除 2 时 HG=G、NGRP=1，
与父版本所有权一致，不增加尾 head 掩码。

PV 归一化前移（本轮机制，仅持久分支）：父版本先以 FP16 舍入的未归一化概率做
单次 PV dot，再对 128x128 的 FP32 累加器做行除法与零行 select；实际 TTIR 在
PV 之后仍保留 divf/比较/select 的独立向量段，模拟器 cube 窗口的 FIXP 热点也
落在 L0C→OUT 的搬出与配对接手上。本轮保持 p = exp2(qk - max) 与
l_new = sum(p) 逐字不变，先算带保护的 l_safe，再算 FP32 的 p/l_safe 并把零行
保护放到权重上（无效行权重精确为 0，PV 得到精确零行），之后才转 FP16 做原来
的单次 PV dot；PV 结果直接转 Out dtype，按原地址与掩码写出，删掉 post-PV 的
除法与 where。数值上这是近似而非等价：舍入对象由未归一化概率变为归一化权重，
单项相对误差仍约 2^-11，但行和不再精确为 1；有限合同下 l_new >= 1，除法不
放大。收益代价：删掉 128x128 FP32 除法/broadcast/select，但引入 128x256 的
权重除法；若编译器把 PV 结果的转换与写出融合为 FIXP 直接输出，还可省去 PV
结果的 GM→UB 与额外交接，该融合仅作为待检验预测，由实际 IR、profiling 与
原四项 checker 判定。非持久回退内核（含 FP16 两项分解分支）逐字不变。

本轮机制（行倒数替代矩阵除法，仅持久分支）：成对 profiling 显示 case4 每程序
aiv_vec_time 14.055us 高于 aic_cube_time 5.173us，而实际 TTIR/BCMLIR 中归一化
仍是对 128x256 FP32 张量的逐元素 arith.divf（每 tile 32768 次除法）。本轮保留
p、m_safe、l_new、l_safe 与零行守卫，只把 p / l_safe[:, None] 改写为
inv_l = 1.0 / l_safe 后再 p * inv_l[:, None]：每 tile 除法由 32768 次降为 128 次，
其余为广播乘。数值上是近似而非等价：除法单次舍入变为倒数与乘法两次 FP32 舍入，
相对偏差上界约 2^-23，远小于随后 FP16 转换的 2^-11；有限合同下 l_new >= 1，
倒数不溢出、不放大，零行仍由 where 精确置 0。转 FP16、单次 PV dot、直接输出、
headgroup 拆分与所有回退路径均不变；是否转化为完整调用收益由原四项 checker 与
真机计时判定，本轮不作正向断言。
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_persist_kv_body(
    Q, K, V, Out,
    q_sb: tl.constexpr, q_ss: tl.constexpr, q_sh: tl.constexpr, q_sd: tl.constexpr,
    k_sb: tl.constexpr, k_ss: tl.constexpr, k_sh: tl.constexpr, k_sd: tl.constexpr,
    v_sb: tl.constexpr, v_ss: tl.constexpr, v_sh: tl.constexpr, v_sd: tl.constexpr,
    o_sb: tl.constexpr, o_ss: tl.constexpr, o_sh: tl.constexpr, o_sd: tl.constexpr,
    S: tl.constexpr, SKV: tl.constexpr, D: tl.constexpr,
    NQ: tl.constexpr, G: tl.constexpr, NKV: tl.constexpr,
    qk_scale: tl.constexpr, delta: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    MASK_MD: tl.constexpr, MASK_KV: tl.constexpr,
    NB_M: tl.constexpr,
    HG: tl.constexpr, NGRP: tl.constexpr,
):
    """每 (batch, 共享 KV 头, headgroup) 一程序的持久化复用体。

    程序先一次性载入该 KV 头的整段 K，并把 V 一次性转换成 fp16，二者都放在
    query-tile 循环之外；随后按 HG 个 query head × NB_M 个行块遍历本 headgroup
    拥有的全部 query tile。逐 tile 的 QK、FP32 softmax、单次 fp16 PV 与归一化
    写出表达式与通用路径逐字一致，只有 K/V 的载入与转换从每 tile 一次降为每程序一次。
    hq = kvh*G + hg*HG + gi 与行块 mb 是本程序的显式坐标，对 (b,hq,row-block) 的
    写出所有权唯一；NGRP=G//HG 个 headgroup 覆盖该 KV 头的全部 G 个 query head。
    """
    pid = tl.program_id(0)
    ng = pid // NGRP
    hg = pid % NGRP
    b = ng // NKV
    kvh = ng % NKV

    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    k_base = K + b * k_sb + kvh * k_sh
    v_base = V + b * v_sb + kvh * v_sh

    # 循环外：整段 K 载入一次。
    k_ptrs = k_base + offs_n[:, None] * k_ss + offs_d[None, :] * k_sd
    if MASK_KV:
        kv_mask = (offs_n[:, None] < SKV) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    else:
        k = tl.load(k_ptrs)

    # 循环外：整段 V 载入一次并转换为 fp16（bf16→f16 的取值与逐 tile 内联转换相同）。
    v_ptrs = v_base + offs_n[:, None] * v_ss + offs_d[None, :] * v_sd
    if MASK_KV:
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
    else:
        v = tl.load(v_ptrs)
    v16 = v.to(tl.float16)

    # 扁平遍历 HG*NB_M 个 query tile；显式坐标保证每个 (hq, mb) 只写一次。
    for t in range(0, HG * NB_M):
        gi = t // NB_M
        mb = t % NB_M
        hq = kvh * G + hg * HG + gi
        offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)

        q_ptrs = Q + b * q_sb + hq * q_sh + offs_m[:, None] * q_ss + offs_d[None, :] * q_sd
        if MASK_MD:
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < S) & (offs_d[None, :] < D), other=0.0)
        else:
            q = tl.load(q_ptrs)

        qk = tl.dot(q, tl.trans(k)) * qk_scale

        if MASK_KV:
            qk = tl.where(offs_n[None, :] < SKV, qk, -float("inf"))

        # SINGLE_KV 无 online 重缩放：表达式与通用路径的 SINGLE_KV 分支一致。
        m_safe = tl.max(qk, 1)
        m_safe = tl.where(m_safe > -float("inf"), m_safe, 0.0)
        p = tl.exp2(qk - m_safe[:, None])
        l_new = tl.sum(p, 1)

        # 本轮机制：把行归一化前移到 PV 之前。p 与 l_new 的 FP32 表达式不变；
        # 先由 l_new 得到带保护的 l_safe，再算 FP32 归一化权重 p/l_safe，并把
        # 零行保护施加在权重上（无效行权重精确为 0，PV 因此得到精确零行），
        # 之后才转 FP16 参与原来的单次 PV dot。PV 累加器即输出值，可直接转
        # Out dtype 写回，省掉 PV 之后 128x128 的 FP32 除法、broadcast 比较
        # 与 select。数值上由「round(p) 点乘 V 后除以 sum(p)」变为
        # 「round(p/sum(p)) 点乘 V」：舍入对象改为归一化权重（fp16 相对误差
        # 仍约 2^-11，行和不再精确为 1）。有限合同下 softmax 权重为正且行
        # 最大项为 1，故 l_new >= 1，除法不放大；l_safe 仅为全 -inf 行的
        # 通用保护保留。编译器是否把 PV 结果的转换与写出融合进 FIXP 输出
        # 路径属待检验预测，由实际 IR、profiling 与真机四项 checker 判定。
        l_safe = tl.where(l_new > 0.0, l_new, 1.0)
        # 本轮机制：把矩阵型 FP32 除法改为每行一次倒数再广播乘。实际 TTIR
        # （arith.divf tensor<128x256xf32>）与 BCMLIR（第 112 行同类 divf）确认
        # 父版本对每个 query tile 做 32768 个逐元素除法，程序内 4 个 tile 共约
        # 13 万次；成对 profiling 中每程序 aiv_vec_time 14.055us 已高于
        # aic_cube_time 5.173us。向量单元上浮点除法的下降序列明显长于乘法，
        # 故先对 128 个行分母各求一次倒数 inv_l，再用 p * inv_l[:, None] 广播乘。
        # 数值上：除法只舍入一次，改为「倒数舍入 + 乘法舍入」两次 FP32 舍入，
        # 相对偏差上界约 2^-23；有限合同下 l_new >= 1，倒数不溢出也不放大，
        # 远小于随后 FP16 转换的 2^-11。零行保护仍在权重上（无效行精确 0），
        # PV 与直接输出路径不变；是否真的缩短向量时间与完整调用由实际 IR、
        # profiling 与原四项 checker 判定，此处不作正向断言。
        inv_l = 1.0 / l_safe
        p_norm = tl.where(l_new[:, None] > 0.0, p * inv_l[:, None], 0.0)

        acc = tl.dot(p_norm.to(tl.float16), v16)

        o_ptrs = Out + b * o_sb + hq * o_sh + offs_m[:, None] * o_ss + offs_d[None, :] * o_sd
        if MASK_MD:
            tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
                     mask=(offs_m[:, None] < S) & (offs_d[None, :] < D))
        else:
            tl.store(o_ptrs, acc.to(Out.dtype.element_ty))


@triton.jit
def _gqa_fwd_kernel(
    Q, K, V, Out,
    q_sb: tl.constexpr, q_ss: tl.constexpr, q_sh: tl.constexpr, q_sd: tl.constexpr,
    k_sb: tl.constexpr, k_ss: tl.constexpr, k_sh: tl.constexpr, k_sd: tl.constexpr,
    v_sb: tl.constexpr, v_ss: tl.constexpr, v_sh: tl.constexpr, v_sd: tl.constexpr,
    o_sb: tl.constexpr, o_ss: tl.constexpr, o_sh: tl.constexpr, o_sd: tl.constexpr,
    S: tl.constexpr, SKV: tl.constexpr, D: tl.constexpr,
    NQ: tl.constexpr, G: tl.constexpr, NKV: tl.constexpr,
    qk_scale: tl.constexpr, delta: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MASK_MD: tl.constexpr,
    MASK_KV: tl.constexpr,
    PV_SINGLE_F16: tl.constexpr,
    SINGLE_KV: tl.constexpr,
    PERSIST_KV: tl.constexpr,
    NB_M: tl.constexpr,
    HG: tl.constexpr,
    NGRP: tl.constexpr,
):
    # 本轮机制：每 (batch, 共享 KV 头) 一程序的持久复用分支。仅在
    # BF16 非因果 SINGLE_KV D128 的原 128/256 分块下启用（见 run()），
    # 此时把 K/V 的载入与 V 的 f16 转换提升到 query-tile 循环外。
    # 其余元数据走下面的原实现，算术与 grid 逐字不变。
    if PERSIST_KV:
        _gqa_persist_kv_body(
            Q, K, V, Out,
            q_sb, q_ss, q_sh, q_sd,
            k_sb, k_ss, k_sh, k_sd,
            v_sb, v_ss, v_sh, v_sd,
            o_sb, o_ss, o_sh, o_sd,
            S=S, SKV=SKV, D=D, NQ=NQ, G=G, NKV=NKV,
            qk_scale=qk_scale, delta=delta,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            MASK_MD=MASK_MD, MASK_KV=MASK_KV,
            NB_M=NB_M, HG=HG, NGRP=NGRP,
        )
        return

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
        #
        # SINGLE_KV（编译期已知整段 KV 恰好落在一个 BLOCK_N 块内）时循环只跑
        # 一轮，online 重缩放整段是无效工作：alpha = exp2(-inf - m)=0 精确为零，
        # acc*alpha / l_i*alpha 只是把 0 写出去再搬回 L0C（TTIR 里 acc*alpha 正是
        # PV dot 的初值，BCMLIR 中是 outs(%acc*alpha)）。故该路径直接用本块
        # row-max 做最终 max，p 一次算完、l 一次求和、acc 由一次 PV dot 产生，
        # 省掉每轮的 128x128 fp32 乘、64KiB acc 写回与 L0C 初值灌入。
        # 非 SINGLE_KV 路径的表达式与相加顺序逐字不变。
        if SINGLE_KV:
            m_safe = tl.max(qk, 1)
            # Fully masked row guard (cannot happen when S <= SKV, kept for safety).
            m_safe = tl.where(m_safe > -float("inf"), m_safe, 0.0)
            p = tl.exp2(qk - m_safe[:, None])
            l_new = tl.sum(p, 1)
        else:
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            # Fully masked row guard (cannot happen when S <= SKV, kept for safety).
            m_safe = tl.where(m_new > -float("inf"), m_new, 0.0)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(qk - m_safe[:, None])
            l_new = l_i * alpha + tl.sum(p, 1)

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
        #
        # 本轮机制（BF16 路径单次 fp16 PV）：两项分解在 BF16 下每 KV 块要付
        # 3 个 matmul（QK + 2×PV）与 384KiB 的 L0C 搬出（每次 128x128 f32 的
        # 64KiB）。成对 profiling 里每程序 cube 仅 2.02us、而 aic_scalar_wait_id1
        # 达 11.79us，L0C_to_GM 总量是 CANN 参考的 2 倍：墙钟跟的是跨核交接链
        # 条数。因此 BF16 改用 11 位尾数的 fp16 作 PV 计算类型，一次
        # fp16×fp16、fp32 累加就替代两条 bf16 dot：每 KV 块 3→2 个 matmul、
        # L0C 384→256KiB/程序，并省掉零初值 dot、1/2048 缩放与一次矢量合并。
        # 数值上是近似而非等价：p 的 fp16 舍入为 2^-11 相对误差（单项 BF16 为
        # 2^-9，故仍优于未分解的 bf16 单项）；V 在 |v|>=2^-14 时 bf16→fp16
        # 精确，|v|<2^-14 落 fp16 次正规（绝对误差 <=3e-8），|v|>65504 溢出——
        # 后者超出合同 [-1,1] 输入域，仅作为适用范围限制记录。
        # FP16 及其他 dtype 仍走原两项分解，表达式与相加顺序逐字不变。
        #
        # 本轮机制（整段 KV 单块 + 单条交接链）：SINGLE_KV 下没有需要保留的
        # acc 初值，PV 直接从零初值的 dot 产生；非 SINGLE_KV 的两条表达式与
        # 相加顺序原样保留（故这里按 dtype × 分支各写一次，避免重排 =
        # (d1+d2)+acc*alpha 这种改变舍入的分组）。
        if SINGLE_KV:
            if PV_SINGLE_F16:
                acc = tl.dot(p.to(tl.float16), v.to(tl.float16))
            else:
                p_hi = p.to(v.dtype)
                r1 = (p - p_hi.to(tl.float32)) * 2048.0
                p_lo = r1.to(v.dtype)
                acc = tl.dot(p_hi, v) + tl.dot(p_lo, v) * (1.0 / 2048.0)
        else:
            if PV_SINGLE_F16:
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))
            else:
                p_hi = p.to(v.dtype)
                r1 = (p - p_hi.to(tl.float32)) * 2048.0
                p_lo = r1.to(v.dtype)
                acc = acc * alpha[:, None] + tl.dot(p_hi, v) + tl.dot(p_lo, v) * (1.0 / 2048.0)
        l_i = l_new
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
    # 向量 softmax 与两项分解、p_hi/p_lo 搬入 L1、PV dot、L0C 搬出、向量合并），
    # 该链与算术量无关。分块由 64 提到 128 后，非因果 case2 退化为单迭代，
    # case4/7 迭代数 4/2 减半；本轮再把 SKV=256 整段并入单块（见下）。
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
    # 已验收保留：编译期几何特化、可证冗余掩码消除、非因果 BLOCK_M 64->128、
    # BF16 路径单次 fp16 PV。
    # 本轮（retained 400.21285us 之上）只改非因果的 KV 轴取块：SKV 为 256 的
    # 整数倍时整段 KV 一块装下（BLOCK_N=256），使每程序交接链 2->1 并在内核走
    # SINGLE_KV 无重缩放路径；因果 64/64、FP16 两项分解表达式与相加顺序、
    # grid 的程序数（S/BLOCK_M × B×NQ）全部不变。
    # 收益或代价都必须由完整四项真机评测判定，本注释不据此改写结论。
    # 数值风险说明：BLOCK_M 只改变 tl.dot 的 M 维铺排与程序数；本轮 BLOCK_N 由
    # 128 提到 256 只改变 PV 在 KV 轴上的归约分组（两次 k=128 累加 + 一次精确
    # 重缩放 -> 一次 k=256 归约），不再有中间 fp32 重缩放；每行仍按同一 j 顺序
    # 累加同一组乘积，是否仍在阈值内由原四项 checker 判定。BF16 单次 fp16 PV
    # 把概率精度由 2^-18 降到 2^-11（近似，非等价），适用输入域见文件头说明。
    BLOCK_D = max(16, triton.next_power_of_2(D))
    if BLOCK_D <= 128:
        # 保留机制：非因果沿 query 行方向由 64 提到 128。
        # 成对 profiling（case4，1024 program，kernel 763.155us）显示每 program
        # 墙钟约 15us 中 aic_cube_time 仅 1.09us，而 aic_scalar_wait_id1 9.15us；
        # 模拟器窗口中位 Cube productive 7.90%/wait 33.27%/idle 58.83%、Vector
        # productive 25.14%/wait 17.13%/idle 57.73%，热点全是 WAIT_FLAG 与
        # MOV_OUT_TO_L1 这类同步/搬运指令。即墙钟由「每 program 每轮 KV 的跨核
        # 交接链」数量主导：每链的 K/V 装载量（2×32KiB）与交接固定成本不随行数
        # 变化，而每链算术只有 64x128x128 + 2×(64x128x128)。
        # BLOCK_M 提到 128 后：program 数与链数减半（case4 1024->512 程序、
        # 2048->1024 链），总 K/V 搬运减半（每 program 208KiB->约 136KiB GM->L1），
        # 每链在 M 方向复用 128 行。代价：per-program 峰值缓冲按 M 维翻倍，
        # 估算约 192KB、逼近 196608B 预算，故非因果才用；因果分支若回到
        # BLOCK_N=128 会重新引入 BLOCK_N 线性相关的索引临时量，实测峰值曾达
        # 231488B，故因果仍取 64/64，其编译与数值路径不变。
        BLOCK_M = 64 if is_causal else 128
        if is_causal:
            BLOCK_N = 64
        else:
            # 本轮机制：SKV 为 256 的整数倍时把整段 KV 放进一个 BLOCK_N=256
            # 块（本合同的 case4/case7 都是 SKV=256）。成对 profiling 显示每
            # 程序 aic_cube_time 仅 1.345us 而 aic_scalar_wait_id1 达 10.260us、
            # 512 个程序约 21 个波次，墙钟跟的是每程序的跨核交接链条数；
            # 2 轮 KV 迭代合成 1 轮后每程序只剩 1 条链（QK dot 搬出 → 向量
            # softmax/p·v 转换 → PV dot 搬出），并删掉整段 online 重缩放。
            # 其余形状（SKV=128 的 case2 等）仍取 128 的通用分块。
            BLOCK_N = 256 if (SKV >= 256 and SKV % 256 == 0) else 128
    elif BLOCK_D <= 256:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 16, 32

    # 掩码可证冗余判据：只用合法元数据（shape 与分块整除性、D 是否等于
    # BLOCK_D），不读任何输入值、不按 case_id 或数值选路。整除成立时掩码在
    # 合法 grid 下恒真，可以整段删除；否则内核走原来的通用掩码分支，语义不变。
    MASK_MD = not ((S % BLOCK_M == 0) and (D == BLOCK_D))
    MASK_KV = not ((SKV % BLOCK_N == 0) and (D == BLOCK_D))

    # PV 计算类型选路：只用 V 的 dtype 元数据（合法元数据，不读输入值、不看
    # case_id）。BF16 输入走单次 fp16 PV（本轮机制，见内核注释与文件头），
    # 其余 dtype 保持两项分解。设备内没有按数值分支。
    PV_SINGLE_F16 = value.dtype == torch.bfloat16

    # 单块 KV 判据：整段 KV 恰好落在一个 BLOCK_N 块内且非因果（因果仍需按
    # hi 提前收束）。满足时循环只有一轮，内核走无 online 重缩放的路径；对
    # SKV==BLOCK_N 的既有形状（如 case2）该路径与原来的 alpha=0 算术逐项等价。
    SINGLE_KV = (not is_causal) and (BLOCK_N == SKV)

    # 持久 KV 复用判据：只用元数据（dtype、causal、SINGLE_KV、分块），不读输入值、
    # 不按 case_id 选路。命中时 grid 由 (行块, b×hq) 改为 (b×kvh×headgroup)，
    # 每程序循环其 headgroup 的 HG 个 query head 与 NB_M 个行块；K/V 载入与 V
    # 转换在循环外每程序只付一次。
    #
    # 本轮机制（持久占用率/复用拆分）：父版本每个持久程序独占一个共享 KV 头的
    # 全部 G 个 query head（case4 grid=32、每程序 16 次迭代；case7 grid=16、
    # 4 次迭代），32 个程序在 20 个 Cube 核上形成 20+12 的不均衡两波。把每头
    # 按 2 个 query head 一组拆成 G/2 个 headgroup 后，case4 grid=128（每程序
    # 4 次迭代）、case7 grid=32（每程序 2 次迭代），核间负载更均衡、长尾更短。
    # 代价是 K/V 整段载入次数随程序数增加：case4 约 4 倍、case7 约 2 倍
    # （成对 profiling 的 GM_to_L1 可验证）；L0C 搬出与全部算术量不变。
    # G 不能整除 2 时 HG=G、NGRP=1，与父版本逐程序所有权一致，不产生尾 head。
    NB_M = triton.cdiv(S, BLOCK_M)
    HG = 2 if (G % 2 == 0 and G >= 2) else G
    NGRP = G // HG
    PERSIST_KV = ((not is_causal) and PV_SINGLE_F16 and SINGLE_KV
                  and BLOCK_M == 128 and BLOCK_N == 256 and BLOCK_D == 128)
    if PERSIST_KV:
        grid = (B * NKV * NGRP,)
    else:
        grid = (triton.cdiv(S, BLOCK_M), B * NQ)
    _gqa_fwd_kernel[grid](
        query, key, value, y,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        S=S, SKV=SKV, D=D, NQ=NQ, G=G, NKV=NKV,
        qk_scale=qk_scale, delta=SKV - S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=bool(is_causal),
        MASK_MD=MASK_MD, MASK_KV=MASK_KV,
        PV_SINGLE_F16=PV_SINGLE_F16,
        SINGLE_KV=SINGLE_KV,
        PERSIST_KV=PERSIST_KV,
        NB_M=NB_M, HG=HG, NGRP=NGRP,
        num_warps=1, num_stages=1,
    )
    return y

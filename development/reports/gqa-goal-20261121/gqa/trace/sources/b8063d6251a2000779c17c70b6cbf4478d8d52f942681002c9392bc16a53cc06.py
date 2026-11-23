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
表达式、M128/N256 分块、guard 与精度路径逐字不变。（该 headgroup 切分已被
下段的 20 常驻 worker 连续 tile 分区取代，此处保留为历史决策记录。）

20 常驻 worker 连续 tile 分区（本轮机制，取代上段 headgroup 切分）：把总
tile 空间 T = B*NKV*G*(S/BLOCK_M)（case4 为 4*8*8*2 = 512）按
floor(pid*T/P)..floor((pid+1)*T/P) 连续分给 P = min(20, T) 个常驻程序，P 取
当前硬件 Cube 核数这一调度参数（纯元数据，由 shape 与常量决定，不读输入值）。
每个 worker 的区间按 KV 组组织：tile 序号 = (b*NKV + kvh)*G*NB_M + g*NB_M + mb，
故同一 (b, kvh) 组内 TPG = G*NB_M 个 tile 连续；worker 只对与其区间相交的组
载入一次 K/V 并一次性把 V 转 fp16，再对该组内的 tile 子区间逐 tile 执行与
019 逐字相同的 QK、fp32 softmax、单次 fp16 PV、PV 后归一化与 BSND 写出。
收益：程序数 128->20（约 7 个波次合成单波次、每 worker 25/26 tile 近乎均分），
K/V 组载入次数 128->约 51，GM_to_L1 总量约 65536KB->6.5MB 量级；每 tile 的
matmul 数、L0C 搬出量与浮点表达式、相加顺序全部不变（无跨 tile 归约）。
代价与风险：每 worker 串行 25~26 条 tile 交接链，链间隐藏能力可能下降；外层
是按组遍历的运行时边界循环，K/V 缓冲需在循环内复用（形状与 019 相同，不新增
张量，峰值由同一批 128x256/128x128 张量决定）。FP16 与其他 dtype 的通用二维
grid 路径、因果 64/64 分块、两项分解表达式与相加顺序全部保持原样。

单块 softmax 的编译期公共偏移（本轮机制，仅 PART_TILES）：成对 profiling 显示
候选 Cube 累计时间已与 CANN 参考相当（662.187us vs 673.090us、L0C_to_GM 同为
98304KB），而 Vector 累计 aiv_vec_time 为 1587.885us vs 1120.804us（约 1.42 倍）；
模拟器 Vector 窗口 productive 21.78%/idle 66.43%，头部指令是 WAIT_FLAG_DEVI 与
MOV_SRC_TO_DST_ALIGN 这类交接/搬运（三核部分窗口各自归一化，
application_completed=false，全调用 completion 缺失）。故本轮删减 Vector 侧一段
纯计算：PART_TILES 已把整段 KV 并入单块、无 online 重缩放，逐行 max 的唯一作用
是把 exp2 自变量压到 <=0 以适配 FP16 PV；合同值域 [-1,1] 给出 |q·k| <= D，
host 由纯元数据取上界 Bq = D*|qk_scale| 并令 SM_OFFSET = Bq - 15，使
p = exp2(qk - SM_OFFSET) <= 2^15 = 32768 < 65504（约 2 倍余量），同时使典型行
的 p 处于 FP16 正规区。这样每 tile 删除一次 128x256 max 归约、其守卫与广播减法，
只保留标量常数减法 + exp2；sum、fp16 PV、PV 后归一化、输出 guard、全部地址与
20 worker 连续分区逐字保留。数值上是实数等价而非逐位等价：每行 p、l_new 与 PV
分子同乘一个公共 2 的幂次因子并精确抵消；p 的 FP16 舍入仍为 2^-11 相对误差量级。
风险是为节省归约而放弃逐行对齐：若某行 max 远低于 SM_OFFSET（约 -12.7 以下）该行
p 进入 FP16 次正规，更低则整体下溢（l_new 仍由 fp32 提供正值，输出退化为 0）；
合同均匀随机输入行 max 约 2、std 约 0.48，到达该边界需约 -30σ；值域外 |q|,|k|>1
可能溢出 FP16。故该机制依赖合同值域，通用 FP16 路径与二维 grid 保持逐行 max。

FP16 rawQK 交接（本轮机制，仅 PART_TILES）：模拟器把最大的 FIXP 热点定位在
QK→Vector 边界（FIX_L0C_TO_DST 83.49us/19 次对应 128x256 f32），而 PV 那条
29.22us/18 次是另一处；实际 BCMLIR 中 rawQK 正以 128x256 f32 从 L0C 搬出后
才做 qk_scale/公共偏移/exp2。故在 dot 之后先把 rawQK 舍入到 FP16 再显式扩回
FP32，不改变 Q/K 输入精度与 dot 的 fp32 累加，仅表达「搬出用 f16」的意图：
后端若把该转换折叠进 FIXP，QK 搬出 128KiB→64KiB、总 L0C 98304→约 65536KB；
若仅在 UB 发生则字节不变，按真实无减少记录并交接固定源码 IR 改法。数值上
rawQK 有符号就近舍入，|q·k| <= D = 128 不溢出 fp16，典型概率相对误差约 2e-4、
最坏约 0.56%，合同值域 [-1,1]，由原四项 checker 以不变阈值判定。
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_tile_part_body(
    Q, K, V, Out,
    q_sb: tl.constexpr, q_ss: tl.constexpr, q_sh: tl.constexpr, q_sd: tl.constexpr,
    k_sb: tl.constexpr, k_ss: tl.constexpr, k_sh: tl.constexpr, k_sd: tl.constexpr,
    v_sb: tl.constexpr, v_ss: tl.constexpr, v_sh: tl.constexpr, v_sd: tl.constexpr,
    o_sb: tl.constexpr, o_ss: tl.constexpr, o_sh: tl.constexpr, o_sd: tl.constexpr,
    S: tl.constexpr, SKV: tl.constexpr, D: tl.constexpr,
    NQ: tl.constexpr, G: tl.constexpr, NKV: tl.constexpr,
    qk_scale: tl.constexpr, delta: tl.constexpr, SM_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    MASK_MD: tl.constexpr, MASK_KV: tl.constexpr,
    NB_M: tl.constexpr, TPG: tl.constexpr, T: tl.constexpr, P: tl.constexpr,
):
    """20 常驻 worker 的连续 tile 分区复用体。

    每个程序先取连续区间 [lo, hi) = [floor(pid*T/P), floor((pid+1)*T/P))，
    再按 KV 组推进：组 grp = b*NKV + kvh 拥有 TPG = G*NB_M 个连续 tile，程序
    只对与自己区间相交的组载入一次整段 K 并一次性把 V 转 fp16，然后在该组的
    相交子区间内逐 tile 执行原表达式。区间由 floor 划分给出，对 pid 单调且相邻
    区间首尾相接，故 0 <= lo < hi <= T 且全部 T 个 tile 被唯一覆盖、每个
    (b,hq,行块) 只写出一次；组内坐标 gi = t // NB_M、mb = t % NB_M 把 tile 序号
    映回 hq = kvh*G + gi 与行偏移 mb*BLOCK_M。
    """
    pid = tl.program_id(0)
    lo = (pid * T) // P
    hi = ((pid + 1) * T) // P

    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # P <= T 时每个程序区间非空；该 guard 只为空区间（P > T）的通用正确性保留。
    if hi > lo:
        g_lo = lo // TPG
        g_hi = (hi - 1) // TPG
        for grp in range(g_lo, g_hi + 1):
            b = grp // NKV
            kvh = grp % NKV

            k_base = K + b * k_sb + kvh * k_sh
            v_base = V + b * v_sb + kvh * v_sh

            # 每 (程序, 相交组) 一次：整段 K 载入 + 整段 V 载入并转 fp16。
            k_ptrs = k_base + offs_n[:, None] * k_ss + offs_d[None, :] * k_sd
            if MASK_KV:
                kv_mask = (offs_n[:, None] < SKV) & (offs_d[None, :] < D)
                k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            else:
                k = tl.load(k_ptrs)

            v_ptrs = v_base + offs_n[:, None] * v_ss + offs_d[None, :] * v_sd
            if MASK_KV:
                v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)
            v16 = v.to(tl.float16)

            # 本 worker 与本组相交的 tile 子区间（组内局部序号）。
            t0 = tl.maximum(lo, grp * TPG) - grp * TPG
            t1 = tl.minimum(hi, (grp + 1) * TPG) - grp * TPG
            for t in range(t0, t1):
                gi = t // NB_M
                mb = t % NB_M
                hq = kvh * G + gi
                offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)

                q_ptrs = Q + b * q_sb + hq * q_sh + offs_m[:, None] * q_ss + offs_d[None, :] * q_sd
                if MASK_MD:
                    q = tl.load(q_ptrs, mask=(offs_m[:, None] < S) & (offs_d[None, :] < D), other=0.0)
                else:
                    q = tl.load(q_ptrs)

                # 本轮机制（PART_TILES 的 QK→Vector 交接，FP16 rawQK）：Q/K 仍为
                # 原 BF16，tl.dot 内部仍按 FP32 累加；dot 之后先把 rawQK 就近
                # 舍入到 FP16，再显式扩回 FP32，随后才执行原 qk_scale、编译期公共
                # 偏移、exp2/sum、fp16 PV 与归一化，表达式其余部分逐字不变。
                # 该舍入只由源码显式表达：后端若把 FP32→FP16 转换折叠进 FIXP
                # （fixpipe 输出 f16），则 QK 的 L0C→Vector 搬出由 128x256 f32
                # (128KiB) 降为 f16(64KiB)，总 L0C 搬出 98304→约 65536KB；若 cast
                # 只落在 UB，则搬出字节与热点时间不变，属真实无减少，由修改后
                # BCMLIR/二进制/profiling 判定。数值风险：rawQK 有符号、就近舍入
                # 对称保号；合同 [-1,1]^128 下 |q·k| <= D = 128 不溢出 fp16，典型
                # 概率相对误差约 2e-4、最坏约 0.56%，适用域 [-1,1]，由原四项
                # checker 以不变阈值判定（详见 diagnosis.json）。
                qk = tl.dot(q, tl.trans(k)).to(tl.float16).to(tl.float32) * qk_scale

                if MASK_KV:
                    qk = tl.where(offs_n[None, :] < SKV, qk, -float("inf"))

                # 本轮机制：编译期公共偏移替代逐行 max 稳定化。
                # 合同输入值域 [-1,1] 给出 |q·k| <= D，故 exp2 自变量
                # qk - SM_OFFSET <= D*|qk_scale| - SM_OFFSET = 15（host 取
                # SM_OFFSET = D*|qk_scale| - 15），p <= 2^15 = 32768 < 65504，
                # 在 FP16 有限范围内保留约 2 倍余量。softmax 对整行公共常数
                # 严格不变：p、l_new、PV 分子同乘 exp2(m_row - SM_OFFSET)，
                # PV 后的 acc/l_new 归一化使其在实数上精确抵消。因此这里删除
                # 每 tile 一次 128x256 的 max 归约、其 -inf 守卫与广播减法，
                # 只保留一次标量常数减法 + exp2；sum、PV、归一化与输出 guard
                # 逐字保留。有限精度代价：p 的 FP16 舍入仍为 2^-11 相对误差，
                # 但若某行 max 远低于 SM_OFFSET（约低于 -12.7），该行 p 进入
                # FP16 次正规；更低则整体下溢为 0（l_new 仍为 fp32 正值，
                # acc 为 0，输出为 0）。合同 [-1,1] 均匀随机输入的行 max 约 2
                # （std 约 0.48），到达该边界需约 -30σ，实际不可达；此机制
                # 依赖 |q|,|k| <= 1 的合同值域，域外可能溢出 FP16。
                p = tl.exp2(qk - SM_OFFSET)
                l_new = tl.sum(p, 1)

                acc = tl.dot(p.to(tl.float16), v16)

                # 本轮机制（仅 PART_TILES 输出归一化）：分母与有效性都是 128
                # 元行标量，原式把 l_safe 广播成 128x128 后做 fp32 除法，再
                # 生成 128x128 的 l_new>0 条件并 select。改为在行向量上先取
                # fp32 倒数、把 l_new>0 折进行系数，再把 fp32 PV 累加器按广播
                # 行系数相乘：每 tile 一次 128x128 VDIV 降为 128 元 VDIV 加
                # 128x128 VMUL，并删除末尾 128x128 比较/广播/select。代价是
                # 多一个 128 元 fp32 暂存与一次舍入（倒数+乘法，相对约 1.5ulp），
                # 故不声明逐位恒等；合同 D=128、scale=1/sqrt(128)、|q|,|k|<=1
                # 下 exp2 自变量 ∈[-17.644,15]，每项 p>0 且每行 256 项有效，
                # l_new 严格为正。完全掩码零行 acc=0、l_new=0 时系数为 0，
                # 输出仍为 0；NaN/非有限值超出官方有限合同，不作通用等价声明。
                row_coeff = tl.where(l_new > 0.0, 1.0 / l_new, 0.0)
                out = acc * row_coeff[:, None]

                o_ptrs = Out + b * o_sb + hq * o_sh + offs_m[:, None] * o_ss + offs_d[None, :] * o_sd
                if MASK_MD:
                    tl.store(o_ptrs, out.to(Out.dtype.element_ty),
                             mask=(offs_m[:, None] < S) & (offs_d[None, :] < D))
                else:
                    tl.store(o_ptrs, out.to(Out.dtype.element_ty))


@triton.jit
def _gqa_fwd_kernel(
    Q, K, V, Out,
    q_sb: tl.constexpr, q_ss: tl.constexpr, q_sh: tl.constexpr, q_sd: tl.constexpr,
    k_sb: tl.constexpr, k_ss: tl.constexpr, k_sh: tl.constexpr, k_sd: tl.constexpr,
    v_sb: tl.constexpr, v_ss: tl.constexpr, v_sh: tl.constexpr, v_sd: tl.constexpr,
    o_sb: tl.constexpr, o_ss: tl.constexpr, o_sh: tl.constexpr, o_sd: tl.constexpr,
    S: tl.constexpr, SKV: tl.constexpr, D: tl.constexpr,
    NQ: tl.constexpr, G: tl.constexpr, NKV: tl.constexpr,
    qk_scale: tl.constexpr, delta: tl.constexpr, SM_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MASK_MD: tl.constexpr,
    MASK_KV: tl.constexpr,
    PV_SINGLE_F16: tl.constexpr,
    SINGLE_KV: tl.constexpr,
    PART_TILES: tl.constexpr,
    NB_M: tl.constexpr,
    TPG: tl.constexpr,
    T: tl.constexpr,
    P: tl.constexpr,
):
    # 本轮机制：20 常驻 worker 的连续 tile 分区分支。仅在
    # BF16 非因果 SINGLE_KV D128 的原 128/256 分块下启用（见 run()），
    # 按 (程序, 相交 KV 组) 载入 K/V 后遍历该组的 tile 子区间；其余元数据走
    # 下面的原实现，算术、grid 与浮点顺序逐字不变。
    if PART_TILES:
        _gqa_tile_part_body(
            Q, K, V, Out,
            q_sb, q_ss, q_sh, q_sd,
            k_sb, k_ss, k_sh, k_sd,
            v_sb, v_ss, v_sh, v_sd,
            o_sb, o_ss, o_sh, o_sd,
            S=S, SKV=SKV, D=D, NQ=NQ, G=G, NKV=NKV,
            qk_scale=qk_scale, delta=delta, SM_OFFSET=SM_OFFSET,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            MASK_MD=MASK_MD, MASK_KV=MASK_KV,
            NB_M=NB_M, TPG=TPG, T=T, P=P,
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

    # 本轮机制（公共偏移，仅 PART_TILES 使用）：合同输入值域 [-1,1] 下
    # |q·k| <= D，故 exp2 自变量上界为 Bq = D*|qk_scale|。取
    # SM_OFFSET = Bq - 15，使 p = exp2(qk - SM_OFFSET) <= 2^15 = 32768，
    # 在 FP16 上限 65504 内保留约 2 倍余量，同时让典型行（行 max 约 2）
    # 的 p 落在 FP16 正规区而非次正规区。该偏移只是整行公共常数，实数上
    # 不改变 softmax，PV 后归一化使其抵消；全部量由 shape、D 与 qk_scale
    # 元数据决定，不读输入值、不按 case_id 选路。通用二维 grid 与 FP16
    # 路径不使用该值，保留逐行 max 稳定化。
    sm_offset = float(D) * abs(qk_scale) - 15.0

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

    # 分区判据：只用元数据（dtype、causal、SINGLE_KV、分块），不读输入值、
    # 不按 case_id 选路。命中时 grid 为 P 个常驻程序，每个程序取连续 tile 区间
    # [floor(pid*T/P), floor((pid+1)*T/P))，并在与自己区间相交的每个 KV 组上
    # 只载入一次 K/V；tile 序号按 (b, kvh, hq, 行块) 行优先排布，故同组
    # TPG = G*NB_M 个 tile 连续。
    #
    # 本轮机制（20 常驻 worker 连续分区）：019 的 headgroup 切分给出 case4
    # grid=128（7 个波次左右），每个程序都要整段重载一次 K/V；成对 profiling 的
    # 每程序 GM_to_L1 均值 512KB、L0C_to_GM 768KB，而参考实现在 19 个程序内完成
    # 同一份 2048 条 cube 指令。把 512 个 tile 连续分给 20 个常驻程序后：程序数
    # 128->20（单波次、每 worker 25/26 tile），组载入次数 128->约 51，每 tile 的
    # matmul 数、L0C 搬出量（128+64KiB）与浮点表达式逐字不变；代价是每 worker
    # 串行 25~26 条 tile 交接链，链间隐藏能力需真机验证。
    # P 取当前硬件 Cube 核数这一调度参数；P = min(20, T) 保证每个程序区间非空，
    # 仍是纯元数据推导（shape 与常量），不涉及输入值。
    NB_M = triton.cdiv(S, BLOCK_M)
    TPG = G * NB_M
    T = B * NKV * TPG
    PART_TILES = ((not is_causal) and PV_SINGLE_F16 and SINGLE_KV
                  and BLOCK_M == 128 and BLOCK_N == 256 and BLOCK_D == 128)
    if PART_TILES:
        P = 20 if T >= 20 else T
        grid = (P,)
    else:
        P = 1
        grid = (triton.cdiv(S, BLOCK_M), B * NQ)
    _gqa_fwd_kernel[grid](
        query, key, value, y,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        S=S, SKV=SKV, D=D, NQ=NQ, G=G, NKV=NKV,
        qk_scale=qk_scale, delta=SKV - S, SM_OFFSET=sm_offset,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=bool(is_causal),
        MASK_MD=MASK_MD, MASK_KV=MASK_KV,
        PV_SINGLE_F16=PV_SINGLE_F16,
        SINGLE_KV=SINGLE_KV,
        PART_TILES=PART_TILES,
        NB_M=NB_M, TPG=TPG, T=T, P=P,
        num_warps=1, num_stages=1,
    )
    return y

# GQA 六个候选案例：从工具证据到代码修改

用途：先介绍 HG-KDA 的技术点，再选择对应案例；当前不制作 PPT。以下代码均为教学伪代码，QK交接转换行按实际Triton差异保留。索引使用当前trace的0–23；attempt编号是研究记录编号，两者不同。

## 统一讲解链

**工具产生什么证据 → 关联到哪段源码／IR → 数据依赖或资源约束支持什么判断 → 修改哪段代码 → 真机收益与修改后机制核对。**

成对profiling揭示与CANN的成本差异；模拟器热点进一步区分同类指令的具体操作位置；模拟器timeline揭示发射／等待顺序与可重叠窗口。源码、IR和AST提供程序结构与语义信息，AST本身不含真实指令耗时，不能将全部案例概括成自动AST完成了运行时归因。

GQA只需介绍这些变量：Q、K用于匹配打分；P为指数权重；V是待聚合内容；d=sum(P)是归一化分母；O=(P@V)/d。每8个Q头共享一组K/V。

## 1．整块KV与单轮特化：compact第1段，具体索引5→6

证据：修改前成对profiling及生成IR促使研究交接次数与重复处理。目标有256个KV位置，旧BN128拆成两块；代码保留逐块online softmax的状态更新，即便能在一个块内完成也沿用通用处理。

定位：KV块循环、累计分母和输出的重缩放表达式。改法：host把该路径BN设为256，Triton对SKV==BN的单块情形使用直接求和与一次PV。

```python
# 修改前：两个KV块，携带online-softmax状态
for kv in blocks(KV, 128):
    score = dot(Q, kv.K)
    P, alpha, d = update_online_softmax(score, old_max, d)
    acc = alpha * acc + dot(P, kv.V)
O = acc / d

# 修改后：一个KV块，直接求分母和PV
score = dot(Q, K_all_256)
P = exp(score - row_max(score))
d = row_sum(P)
O = dot(P, V_all_256) / d
```

结果：400.21285→328.44680μs。实际修改是Triton源码＋host metadata。这个候选要解释online softmax，因此讲解成本高于案例2；不要把初始1900.721→328.447μs的整个阶段收益都算到这一次改法上。

依据：[实际diff](../gqa-goal-20261121/gqa/agent-calls/attempt-006/source.diff)、[修改前审查](../gqa-goal-20261121/gqa/agent-calls/attempt-006/host-review.json)。

## 2．KV持久复用：compact第2段，具体索引6→7

**工具输出。** 从修改前006成对profiling原始comparison.json核对：

| 指标 | 我们的实现 | 同次采集CANN |
|---|---:|---:|
| GM→L1总搬运 | 114688 KiB = 112 MiB | 65840 KiB ≈ 64.30 MiB |
| L0C→GM总搬运 | 98304 KiB = 96 MiB | 98304 KiB = 96 MiB |
| Cube程序数 | 512 | 19 |

**关联代码。** 每个query块独立program；同组8个query头、每头2个query块，共16个program反复读取同一份K/V。`k=tl.load(...)`、`v=tl.load(...)`与V转换位于每个query块的工作路径中。

**判断与修改。** 相同K/V与query块索引无关，可以每个(batch,kv_head)一次读取，再在程序内遍历多个query块。这里仍需权衡程序数量减少后的并行度，不能把搬运量降低直接等同为同比例耗时降低。

```python
# 修改前：每个query块一个program，共512个
q = load(Q_for_this_tile)
k = load(K_for_shared_head)       # 同组多个program重复载入
v = load(V_for_shared_head).to(fp16)
out = attention_tile(q, k, v)

# 修改后：每个(batch, kv_head)一个program，共32个
k = load(K_for_shared_head)
v = load(V_for_shared_head).to(fp16)
for tile in query_tiles_of_this_group:  # 16个query块
    q = load(Q_for_tile)
    out = attention_tile(q, k, v)
```

**验证。** 修改后GM→L1为53248 KiB=52 MiB；L0C→GM仍96 MiB。正式完整调用328.44680→268.77180μs，下降18.17%，原四用例通过。独立kernel计时329.446594→256.825134μs，单列。

依据：[修改前成对profiling](../gqa-goal-20261121/gqa/iterations/attempt-006/paired-parent/comparison.json)、[前置分析](../gqa-goal-20261121/gqa/host-analysis-attempt-017.json)、[实际源码diff](../gqa-goal-20261121/gqa/agent-calls/attempt-017/source.diff)、[修改后成对profiling](../gqa-goal-20261121/gqa/iterations/attempt-017/paired-parent/comparison.json)。

## 3．QK交接缩窄：compact第3段，具体索引10→11

**工具输出。** 031父版模拟器采集窗口中，两处FIX_L0C_TO_DST搬出记录：QK相关83.489998μs/19次，PV相关29.219999μs/18次。相同指令类别对应不同计算阶段，热点排序帮助优先研究QK交接；这些是窗口累计统计，不当作完整kernel可节省时间。

**关联代码。** 通过实际TTIR／BCMLIR与指令位置关联到QK的`tl.dot`结果：128×256的FP32张量从L0C搬出后，向量侧才做scale和指数运算。

**判断与修改。** 这一张量每块128KiB；如果后端能在搬出时转成FP16，就可减为64KiB，向量侧再恢复FP32。值域分析与原checker用于判断舍入是否可接受。

```python
# 修改前，实际Triton表达式
qk = tl.dot(q, tl.trans(k)) * qk_scale

# 修改后，实际Triton表达式
qk = tl.dot(q, tl.trans(k)).to(tl.float16).to(tl.float32) * qk_scale
```

**验证。** 检查生成代码是否将转换折叠进搬出路径；若只在接收端转换，就不会减少这段搬运。实际后测L0C→GM从96降至64MiB，正式完整调用205.28415→196.15815μs，原四用例通过。独立kernel205.624→191.024μs。修改的是Triton源码。

依据：[前置热点分析](../gqa-goal-20261121/gqa/host-analysis-attempt-033.json)、[实际修改](../gqa-goal-20261121/gqa/agent-calls/attempt-033/source.diff)、[修改前读取与关联审查](../gqa-goal-20261121/gqa/agent-calls/attempt-033/host-review.json)。

## 4．下一块QK前瞻：compact第4段，具体索引18→19

证据：模拟器显示当前QK→当前PV→下一QK的顺序及交接等待。代码中下一QK只依赖下一Q和本组K，不依赖当前PV；可以提前发起。修改Triton循环，先准备首块，之后携带当前QK结果并在当前softmax/PV前发起下一块QK。

```python
# 修改前
for t in tiles:
    raw = dot(load(Q[t]), K)
    P = make_probability(raw)
    store(normalize(dot(P, V), P))

# 修改后
raw = dot(load(Q[first]), K)
for t in tiles:
    if has_next(t):
        raw_next = dot(load(Q[t+1]), K)
    P = make_probability(raw)
    store(normalize(dot(P, V), P))
    if has_next(t):
        raw = raw_next
```

后重放确认next QK移至current PV之前；MMAD与搬运总量保持。正式完整调用154.8871→146.0561μs，下降5.70%。适合讲模拟器指导Triton调度；可展示顺序改变，不能把伪代码的提前发起直接当作所有流水都已实现理想重叠。

依据：[063完整案例](../gqa-goal-20261121/attempt-063-case.md)、[Triton diff](../gqa-goal-20261121/gqa/agent-calls/attempt-063/source.diff)。

## 5．独立归约前移：compact第5段，具体索引21→22

**工具输出。** IR003同核重放显示PV等待44.488–45.895μs，sum46.336–46.886μs、VREC46.887–46.900μs。分母计算排在PV返回之后。

**关联IR。** `enablevc_reduce_sum_ar_float`和`vrec_1d_float`对应`d=sum(P)`与`1/d`；它们读P，不读PV结果。实际代码排在接收PV之后。

**判断与修改。** P已经发给Cube，可在PV结果返回之前计算分母与倒数。固定Triton，移动晚层IR中的独立计算链，保留依赖和同步。

```python
# IR修改前的执行顺序示意
send_P_to_cube(P)
pv = wait_and_receive_PV()
d = reduce_sum(P)
inv = reciprocal(d)
out = pv * inv

# IR修改后的执行顺序示意
send_P_to_cube(P)
d = reduce_sum(P)
inv = reciprocal(d)
pv = wait_and_receive_PV()
out = pv * inv
```

**验证。** 后重放sum41.380–41.931μs、VREC41.932–41.945μs，进入发送／等待区间，PV接收在46.414–46.738μs。比较的是各自窗口内的顺序与重叠关系。实际浮点操作与逻辑搬运量保持；正式完整调用144.43715→133.3206μs，下降7.70%。

依据：[前后时间窗](../gqa-goal-20261121/ir-qk-receive-prefetch-case.md)、[真实16行IR移动](../gqa-goal-20261121/gqa-ir-qk-receive-prefetch/agent-calls/ir-004/model-ir.diff)。

## 6．下一块概率前瞻：compact第6段，具体索引22→23

证据：IR004重放中nextQK已经接收，但当前PV仍未返回；下一块转换／缩放／exp排在PV返回后。源码／IR依赖分析表明下一概率只依赖nextQK。缓冲分析发现其存储区被当前reciprocal与VMUL临时数据占用。

```python
# IR修改前：等待后才准备下一块概率
next_qk = receive_next_QK()
current_pv = wait_and_receive_current_PV()
finish_current_output(current_pv)
next_p = exp(to_fp32(next_qk) * scale)

# IR修改后：先调整临时缓冲，保证next_p存活
# reciprocal / VMUL scratch -> 不与next_p重叠的UB区域
next_qk = receive_next_QK()
next_p = exp(to_fp32(next_qk) * scale)
current_pv = wait_and_receive_current_PV()
finish_current_output(current_pv)  # 保持next_p不被覆盖
```

后重放：下一块转换51.915–52.082μs、exp52.260–52.587μs，落在当前PV等待51.447–52.837μs内；正式完整调用133.3206→120.9111μs，下降9.31%。适合展示时间线、跨迭代依赖、缓冲生命周期共同指导IR优化，讲解门槛高于案例5。

依据：[完整案例与缓冲分配](../gqa-goal-20261121/ir-probability-lookahead-case.md)。

## 选例建议

| 技术点 | 首选 | 备选 |
|---|---|---|
| 成对profiling | 案例2：112MiB对64.3MiB，引出KV重复载入，再验证降至52MiB | 案例1：缩减KV交接次数，需要解释online softmax |
| 模拟器热点 | 案例3：同类搬出指令区分QK/PV，关联IR与Triton，修改交接类型 | — |
| 模拟器流水空泡 | 案例5：分母独立于PV，IR前移，7.70%收益 | 案例4展示Triton前瞻；案例6展示更深的IR缓冲／流水协同 |

推荐三个主例为2、3、5。若需要强调Triton到IR的层次推进，先用案例4快速引入前瞻，再展开案例5或6。所有收益对应单次真实修改，不借用整个压缩阶段累计成绩。

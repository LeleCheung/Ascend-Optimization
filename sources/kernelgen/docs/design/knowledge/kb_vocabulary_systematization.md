# KB 检索词汇体系化 TODO

> 本文件聚焦 Source、Source → Concept 审核和检索词汇归一化。活动 Catalog
> 已在四个 commit-pinned Source 上发布首批 82 个审核 Concept，仍没有
> Observation；Distiller、Candidate body、Static Publisher 和运行时检索使用
> 同一词汇与正文契约。

## 背景

旧 Catalog 的自由标签存在同义重复和层次混乱。仓库内置 `kb/` 是当前
Source-backed 基线：Source 保留完整原文，Concept 只保存逐条审核后的原子
claim；正式可写运行应使用单独的 Catalog 路径。

外部参考源：
- `bottleneck_headroom_kernelstructure.yaml`
  — `method_catalog`（~20 个 CUDA 优化方法）+ `bottleneck_taxonomy`
- `KernelWiki/data/tags.yaml`
  — 28 个受控 technique tag，validator 拒绝未知值
- `KernelWiki/data/aliases.yaml`
  — canonical → synonyms 映射，查询时自动归一化
- `cannbot-skills/ops/ops-profiling/references/optimization_quickref.md`
  — Ascend 瓶颈分类（VEC/MTE2/CUBE/SCALAR/BankConflict/核间不均衡/DB未生效/流水气泡/L2）

---

## 1. 别名表（最高优先级）

**目标**：建一张 exact alias 映射表，`query_sources` 查询时自动展开。

- [x] 在 Catalog 根目录创建 `aliases.yaml`，格式：

```yaml
techniques:
  coalesced_access:
    - coalesced_global_access
    - coalesced access
    - coalesced global access
    - coalescing
  double_buffering:
    - double_buffer
    - double buffering
    - double-buffering
    - ping_pong_buffer
    - ping pong buffering
  shared_memory_tiling:
    - shared_mem_tiling
    - shared memory tiling
    - smem_tiling
  kernel_fusion:
    - op_fusion
    - operator_fusion
    - kernel fusion
    - operator fusion

symptoms:
  memory_bandwidth:
    - bandwidth_bound
    - memory bandwidth bound
    - memory_bandwidth_bound
  pipeline_stall:
    - pipeline stalls
    - pipeline_stalls
  resource_exhaustion:
    - resource exhaustion
    - resource_limit
```

- [x] `query_sources` 查询时把 canonical 或 alias 展开为同一组 exact terms
- [x] 未知词原样搜索，不阻塞
- [x] 查询结果返回 `expanded_terms`
- [x] 根据真实 Source 查询补充全部 canonical 项的第一版 exact aliases

`vectorized_access`、`wide_load`、`transfer_compute_overlap`、`ub_tiling` 等相关但
不等价的技术不能作为 alias 强行折叠。

**验收**：查询 "coalescing" 和 "coalesced_global_access" 命中相同 Source 集合。
完整审计报告写入 Catalog 根目录 `vocabulary_audit.json`；当前 32 个词族均有
Source 命中，canonical/alias expansion 全部一致。

---

## 2. 规范 symptom 类别（合并 CUDA + Ascend）

**目标**：~10 个平台无关瓶颈类别，覆盖两个平台的诊断术语。

| 规范 symptom | CUDA 表现 | Ascend 表现 |
|---|---|---|
| `memory_bandwidth` | DRAM throughput 饱和 | MTE2 Bound |
| `memory_latency` | L2 miss / long scoreboard stall | L2 Cache 命中率低 |
| `compute_bound` | tensor core 利用率低 | VEC Bound / CUBE Bound |
| `pipeline_stall` | warp stall / mbarrier wait | 流水线气泡 / DB 未生效 |
| `resource_exhaustion` | register spilling / SMEM 超限 | UB 溢出 / L1 溢出 |
| `bank_conflict` | shared mem bank conflict | UB bank/bankgroup conflict |
| `load_imbalance` | SM utilization 不均 / tail effect | 核间负载不均衡 |
| `launch_overhead` | kernel launch 占比高 | SCALAR Bound / 头开销 |
| `precision_error` | ULP 超标 / NaN | 精度校验失败 |
| `correctness_failure` | wrong result | 结果错误 / 数据损坏 |
| `compile_error` | 编译失败 / unsupported feature | 编译失败 / API 不支持 |
| `runtime_failure` | runtime error / timeout | 运行失败 / 超时 |

- [x] 将上表固化为 Catalog 根目录的 `canonical_symptoms.yaml`
- [x] 根据 Source 查询补充 12 类 symptom 的 exact aliases
- [x] 接入 Distiller contract；失败状态由 Python 映射到 canonical symptom

**验收**：canonical symptom 共 12 个（≤ 15）；查询 "MTE2 bound" 与
`memory_bandwidth` 使用同一 expansion。Publisher 将已知 alias 写成 canonical，
未知值保留并写入 `audit_warnings`。

---

## 3. 规范 technique 列表

**目标**：~25 个平台无关优化方法名。来源：YAML method_catalog + KernelWiki tags。

```
# 内存访问优化
coalesced_access          # 合并/向量化访存
shared_memory_tiling      # 共享内存/UB tiling
cache_policy              # L1/L2 cache 策略
software_prefetching      # 软件预取

# 流水与并行
double_buffering          # 双缓冲/多缓冲
pipeline_stages           # 多级流水
warp_specialization       # warp/核角色分工
persistent_kernel         # 持久化 kernel / 核常驻
tile_scheduling           # tile 调度策略

# 计算优化
kernel_fusion             # 算子/子图融合
mixed_precision           # 混合精度
register_blocking         # 寄存器分块
loop_unrolling            # 循环展开
instruction_scheduling    # 指令重排

# 结构优化
epilogue_fusion           # 后处理融合
split_k                   # K 轴分核
workload_balancing        # 负载均衡
redundant_computation     # 冗余计算换搬运

# 数据格式
data_layout_optimization  # 布局/NZ 格式
quantization              # 量化
```

- [x] 固化为 Catalog 根目录的 `canonical_techniques.yaml`
- [x] 根据 Source 查询为 20 个 canonical 值补充 exact aliases
- [x] Distiller 明确字段归属：API/编译器名放 keywords，诊断步骤放 body，
  retrieval.techniques 只放可复用优化方法

**验收**：canonical technique 共 20 个（≤ 30）；Publisher 对未知值告警，
定期审计报告统计实际 Concept 唯一值。

---

## 4. diagnostic 类 Concept body 结构化

**目标**：采用 KernelWiki pattern 页的固定结构，模型检索后能直接按步骤执行。

Diagnostic body 使用以下固定结构：

```markdown
## Symptom

怎么判断是这个瓶颈（profile 指标、阈值）

## Likely Causes

为什么会这样（2-4 条）

## Candidate Techniques

| 方法 | 预期收益 | 适用条件 |
|---|---|---|
| ... | ... | ... |

## Diagnosis Checklist

1. 确认步骤...
2. ...

## Caveats

- 别踩的坑
```

- [x] Diagnostic Candidate 强制使用上述五段结构；首批 17 个 Diagnostic
  已全部通过正文审计
- [x] Method Candidate 的 Limits 强制包含 `### Mechanism Requirements`
- [x] Method Candidate 的 Action 强制包含 `### Expected Metric Change`

**验收**：Distiller 输出和 Publisher 接收的新 Diagnostic 必须通过五段结构校验；
新 Method 必须通过两个约束子节校验。当前 82 个 Concept 正文长度为
623–1093 字符，未序列化空 metadata；旧弃用 Catalog 不迁入活动 Catalog。

---

## 5. Publisher 校验

- [x] validation.py 增加 technique/symptom 推荐检查：
  - 值在 canonical 列表或 aliases 表中 → pass
  - 值不在 → warning（不阻塞），写入 `audit_warnings`
- [x] `scripts/kb/audit_source_vocabulary.py` 统计词表、Source 命中和 Concept 唯一值；
  Publisher 在 symptom > 15 或 technique > 30 时写入 `audit_warnings`

---

## 6. motifs 扩展（低优先级）

当前 motifs：reduction、elementwise、matrix_multiply、normalization、attention。

参考 cannbot-skills ascendc-tiling-design 的算子分类，补充：

- [x] 增加 `sort`、`broadcast`、`conversion`（transpose/concat/split）
- [x] OperatorSignature 测试确认三个新 motif 进入通用 QueryContext motif 路由

---

## 7. 四源 KB 初始化与首批 Concept

当前仓库基线 Catalog：`kb/`。

| Source package | revision | manifest entries |
|---|---|---:|
| `source:cannbot-skills` | `7fedd2ff5e1f8a828aa7b02abbaaed785c98dc1c` | 2615 |
| `source:triton-ascend` | `a60006ac63f3f5068cc4b217ebd4e14ce0801f3e` | 2855 |
| `source:kernelwiki` | `2777d18ffb3a3d682d8f25a3e3b8864d925a5ff1` | 3520 |
| `source:agent-skills` | `6a6e9af256d866316cccc4b53966b146c4f536f2` | 2145 |

`agent-skills/official/CANNBot` 已从 snapshot 和 manifest 排除，因为
`cannbot-skills` 已作为独立 Source package 注册。所有 package 都来自干净工作树的
精确 Git commit；未跟踪文件、`.git` 和 submodule 内容不进入 Source。

- [x] 新建独立 Source-backed Catalog
- [x] 生成四个 content snapshot 与 Git manifest
- [x] 四个 ingestion plan 保留全量 `indexed` Source，并加入逐条审核的
  `extract` entries
- [x] 发布 82 个 Concept：triton-ascend 29、cannbot-skills 29、
  agent-skills 12、KernelWiki 12
- [x] Catalog validation：4 packages、11135 entries、82 Concept、0 Observation
- [x] Concept/Observation derived index 可以从事实文件重建
- [x] `query_sources` canonical/alias 查询返回相同结果
- [x] 未知查询词原样搜索
- [x] 每个 Concept 只有一个精确 Source locator；完整纳入与淘汰理由记录在
  `reviews/ascend910b-source-promotion.md`

首批 Concept 由 Static Publisher 发布，不依赖未运行的 campaign 或伪造
Observation。后续新增经验知识必须来自实际 round，不把 Source 支持的通用方法
冒充已经测得的性能结论。

---

## 8. 多芯片 Source → Concept 约束

多芯片知识不增加新的词表维度，也不按“算子 × workload × 芯片”复制。复用单位仍是
原子 claim，硬件差异由 `scope.target` 表达：

- [x] `device` 和 `architecture` 匹配同时核对 backend
- [x] 多个 `devices` 使用 OR 语义；`capabilities` 使用 AND 语义
- [x] 查询按 `exact > device > architecture > backend > portable` 优先保留
  同一 claim 的具体变体
- [x] Static Publisher 在发布前验证正文、target level 必填字段以及 Source
  `allowed_backends`
- [x] Source wrapper 保持固定；首批 Method/Diagnostic 使用审核后的
  `extract` 正文

`cannbot-skills` 和 `triton-ascend` 都只允许 Ascend backend，因此不能据此发布
CUDA 或 `portable` Concept。适用于整个 Ascend 后端的 claim 使用 backend scope；
只适用于某一架构或设备的优化分别使用 architecture/device scope。实现步骤、能力
要求或正确性边界不同的芯片变体必须拆分，并使用 `refines` 等关系连接。

首批 82 个 Concept 中，70 个使用 `DAV_2201` architecture scope，12 个
KernelWiki Concept 使用 portable scope。映射全部 13 个 Definition 的 22 个通用
Concept 不绑定 ID；其余 60 个专用 Concept 在现有 motif/dtype 无法无歧义表达
边界时绑定已审核的 `definition_ids`，避免专用知识误召回。未来新增算子经审核后
扩展适用集合，不按“算子 × workload × 芯片”复制正文。

---

## 不做

- 不把 `decision_table`、`headroom_tiers`、`ncu_predicates` 搬进 KB（属于决策引擎）
- 不把 `kernel_structures`（S0-S4）加入 Schema（motif 够用，结构分类写 body）
- 不把 `code_features`（has_reuse、is_pointwise）加入 Schema（等 QueryContext 能传入时再接）
- 不搬 KernelWiki 的 Source/PR 追踪基础设施（KB 的 Source 概念已够用）
- 不搬 PyPTO 调优状态机 / 25 个 Triton 优化点（是流程编排，不是知识）
- 不搬 KernelWiki 的 confidence/reproducibility 字段（KB 已有 evidence_state + usage_score）

---

## 实施顺序

```
1. 新建 Source-only Catalog             ← 已完成
2. 导入四个 commit-pinned Source         ← 已完成
3. aliases + canonical 列表              ← 已完成第一版
4. query_sources 查询展开与未知词透传    ← 已完成
5. Source 查询质量审计与词表迭代         ← 已完成第一版
6. Distiller/Publisher/body 契约接入      ← 已完成
7. motifs 扩展                           ← 已完成
8. 多芯片 scope 匹配与 Static 发布门禁    ← 已完成
9. 四源 Source → Concept 逐条审核与发布   ← 已完成
10. 13 算子生产检索回归                  ← 已完成
11. 真实 round 知识采用与效果闭环         ← 代码闭环已完成，真实 run 待验证
```

生产检索回归覆盖 13 个 Definition 的 `architecture_selection`、
`implementation`、`next_experiment` 和 `diagnosis`，共 52 次查询：52/52
有 direct 结果且无 task coverage gap，召回 68 个不同 Concept，返回项全部位于
对应 Definition 的审核适用集合，四个 Source 包均有实际召回。

活动 Catalog 仍不自动生成 Concept。Epoch Publisher 已会把
`knowledge_uses` 非空的已评测 round 发布为 Observation，即使本 epoch 没有新
Candidate；这些事实直接进入现有 metrics 和 `usage_score`。下一步不增加新 Schema：
选择一个真实算子运行，用现有 retrieval log、KnowledgeApplication、Observation 和
usage_score 验证“检索—采用—测量结果—后续排序提升”的完整运行记录。

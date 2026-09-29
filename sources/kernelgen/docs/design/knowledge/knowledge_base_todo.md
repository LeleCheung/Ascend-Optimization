# KernelGen Knowledge Base TODO

本文记录 KB 从“可查询、可发布的知识目录”演进为“根据实验结果持续学习的知识系统”所需的工作。它不是字段契约，也不描述已经上线的行为：

- 当前运行流程见 [`knowledge_base_implementation.md`](knowledge_base_implementation.md)；
- 字段语义见 [`knowledge_concept_contract.md`](knowledge_concept_contract.md)；
- 本文只维护优先级、实施切片和验收条件。

更新日期：2026-08-08。

## 0. TODO 管理规则

本文是 KB 改动的唯一实施 backlog。设计讨论只有进入本文并写清依赖和验收条件后，才进入代码修改。

状态约定：

| 状态 | 含义 |
|---|---|
| `TODO` | 设计边界已明确，可以按依赖实施 |
| `BLOCKED` | 依赖项尚未完成，不能提前修改 |
| `DEFERRED` | 字段可能有长期价值，但当前没有完整生产/消费流程 |
| `DONE` | 按改动范围完成验收并记录独立 commit；需要部署到 hw 时另行同步和验证运行副本 |

每个实施切片必须遵守：

1. 一个切片只解决一个可验证问题，不混入相邻重构；
2. 先写兼容策略和验收测试，再修改 Schema；
3. 旧 Ledger、旧 Concept 和旧 retrieval log 只做显式单向迁移，不猜测缺失事实；
4. 使用统计放在可重建派生索引，不因统计变化重写 Concept；
5. A100 修改和测试通过后再同步 hw；hw 通过后才提交；
6. 单元测试不得写公共 KB；涉及发布时只使用临时 Catalog；
7. 公共 KB 数据迁移必须是单独切片，并在执行前保存可回滚基线。

## 1. 长期目标

KB 的核心闭环应该是：

```text
Source / Concept
-> Query
-> ExperimentPlan 中声明如何使用
-> Round / Ledger
-> Eval
-> Observation
-> 使用效果索引
-> 下一次检索排序
-> Distiller 产生或演进 Concept
```

长期模型分工：

| 对象 | 职责 |
|---|---|
| Source | 保存原始文档、API、硬件和编译器资料 |
| Concept | 保存可行动、可检索的 claim |
| Query | 记录当时为什么查、返回了什么 |
| ExperimentPlan | 记录 Agent 决定采用、改写或拒绝什么 |
| Ledger | 保存每个 round 的计划、代码和真实结果 |
| Observation | 从 Ledger 冻结被知识引用的 round 事实 |
| Knowledge-use view | 关联“使用了什么”和“结果怎样”，供检索和统计使用 |

Source、Ledger 和 Observation 保存事实；Concept 保存解释和行动建议；使用统计是可从
Observation 重建的派生数据，不应每次使用都改写 Concept。

## 2. 当前基础

以下能力已经存在，本 TODO 不重复实现：

- [x] `ExperimentPlan.knowledge_uses` 统一记录实际落实到当前评测候选的 Concept 或精确 Source ref、query event、role、disposition、application note 和 affected parts。
- [x] `KnowledgeUse` 会随 ExperimentPlan 写入 Ledger round；v2.1 Ledger 单向迁移到
  v2.2。
- [x] Coder prompt 区分检索、读取、分析和真正落实到当前候选的知识，并要求记录具体应用方式。
- [x] Query、Source 和 Concept 的 detail-read provenance 已写入 workspace retrieval
  log。
- [x] `SourceReference` 新写入使用 `resource + revision + locator`；旧
  `id/author/usage_count/last_modified` 可读取但不再发布。完成提交：
  `4695e01`。
- [x] `get_source` provenance 保存实际 Source revision 和读取行范围。
- [x] Observation 从真实 Ledger round 生成。
- [x] Concept 的 target、software、workload、dtype、layout、numerics 和 evidence
  使用 Python 权威输入。
- [x] Runtime Concept 正文和 round 引用已有结构化校验。
- [x] Publisher 使用 staging 和 batch audit 发布。

当前缺口：

- Source 不能根据实际使用结果自动提炼为 Concept；
- measured evidence 仍累计复制进 Concept，长期可能使单个 Concept 过长。

## 3. P0：先实现可追踪的知识使用闭环

这是当前最值得实现的部分。完成后系统必须能回答：

> R5 使用了哪些 Concept/Source，它们被怎样应用，R5 相对 baseline 的结果是什么？

### 3.0 先补齐不可丢失的 lineage

这些是后续 usage 字段的依赖，不能在应用关联写入后再回头改变引用语义。

- [x] Source detail-read 保存 `resource + revision + locator`，并兼容读取旧
  SourceReference。Commit：`4695e01`。
- [x] Concept 使用稳定 ID，引用不带 `@N`；一次 Query 的 Catalog 状态由
  `catalog_snapshot` 固定。
- [x] 每个 Concept ID 只有一个 `concepts/<kind>--<slug>.md` 当前文件，
  Publisher 在事务中原子覆盖；历史和回滚交给 Catalog Git。
- [x] 纯 retrieval/usage 统计不修改 Concept 正文；SourcePackage revision 继续
  标识外部资料快照。
- [x] Query/Retrieval 使用唯一 event ID；相同请求重复执行也产生不同 ID，不发布
  新的 fingerprint。
- [x] Query event 保存足以重建当时 phase、task、question、operator、target、
  findings/errors 和返回结果的 context。
- [x] Run Archive 保存 retrieval event；旧 query log 仅做只读兼容。
- [x] Ledger Evaluation 写入 `evaluated_at`，Observation 原样复制。
- [x] Round 明确保存 `experiment_parent_round_num`；Observation
  同时保存 parent status，不能用“历史最佳”冒充本次实验父 round。
- [x] 性能比较继续使用 previous-best baseline；正确性恢复/回退使用 experiment
  parent。两个概念不得共用一个含糊的 `baseline_round_num`。

### 3.1 补充 Concept 使用描述

- [x] 在现有 `KnowledgeUse` 增加 `application_note`。
- [x] 增加 `affected_parts: list[str]`，例如 `input_load`、`reduction`、
  `boundary_mask`。
- [x] v2.1 Ledger 的 `adaptation` 单向迁移为 `application_note`；新写入不再输出
  旧字段。
- [x] Coder contract 要求新提交的 `adopted` 和 `adapted` 必须说明实际应用方式和当前候选中的作用位置；`rejected` 只保留历史读取兼容。旧 Ledger 的缺失内容不伪造。
- [x] 开放描述不做技术枚举，只做空白、长度、空值和重复值等轻量校验。

建议形状：

```yaml
knowledge_uses:
  - concept_ref: kg:method:vectorized-load
    query_event_id: query:0123456789abcdef0123456789abcdef
    role: implementation
    disposition: adapted
    application_note: 将连续加载改成宽度为 8 的向量加载
    affected_parts:
      - input_load
```

### 3.2 增加 Source 使用声明

- [x] 不增加平行 `SourceUse/source_uses`；现有 `KnowledgeUse` 只允许设置一个
  `concept_ref` 或 `source_ref`。
- [x] 一次 `get_source` 不会自动写入 `knowledge_uses`，只有 Agent 明确声明才算。
- [x] Source 使用记录 `resource`、locator、revision、query event、role、
  disposition、application note 和 affected parts。
- [x] Source ref 使用已有 revision/locator，不增加新的 fingerprint 机制。
- [x] Coder prompt 明确区分 read 与实际落实到当前候选的 adopted/adapted；新提交不再把 rejected 写入 `knowledge_uses`。
- [x] Source/Concept 的 query event 缺失不得阻塞 Eval。
- [x] Observation 为每条 application 派生 `lineage_status`；完整链路、
  Query 缺失、detail read 缺失和 legacy 未知均可区分，且不阻塞 Eval。

建议形状：

```yaml
knowledge_uses:
  - source_ref:
      resource: source:triton-ascend-docs
      locator: tl-load/mask-semantics
      revision: "3"
    query_event_id: source-query:0123456789abcdef0123456789abcdef
    role: constraint
    disposition: adopted
    application_note: 根据 mask 语义处理越界元素
    affected_parts:
      - boundary_mask
```

### 3.3 将使用关联冻结到 Observation

Observation 已经保存 round outcome 和 baseline comparison，不再复制一份相同的性能
字段。只需要把使用关联冻结进 Observation：

- [x] 新增 `KnowledgeApplication`，支持 Concept ref 或 Source fragment ref。
- [x] `KernelGenRunFactReader` 从 `round.plan` 复制应用声明；历史 adopted/adapted/rejected 均可读取，新 round 只写 adopted/adapted。
- [x] Observation 保存 `applications` 和 `usage_mode`：
  `none/single/combined`。
- [x] `single` 只表示该 round 声明了一项被采用知识，不自动声称严格因果。
- [x] 使用效果由 `Observation.applications + outcome + comparison` 组合得到，不重复
  存储 geo mean、status 或 workload 测量。
- [x] 历史 rejected/considered 记录可继续用于兼容统计，但新 round 不再产生 rejected 使用声明。

派生视图示例：

```yaml
ref: kg:method:vectorized-load
observation_ref: kg:observation:run-x:agent1:round-5
usage_mode: single
status_before: PASSED
status_after: PASSED
geo_mean_delta_pct: 18.68
effect: performance_improved
```

`effect` 第一版只做确定性分类：

| 条件 | effect |
|---|---|
| baseline 非 PASSED，当前 PASSED | `correctness_recovered` |
| baseline PASSED，当前非 PASSED | `correctness_regressed` |
| 两轮 PASSED，delta 为正 | `performance_improved` |
| 两轮 PASSED，delta 为负 | `performance_regressed` |
| 两轮 PASSED，无明显变化 | `no_material_change` |
| 没有可比较 baseline | `unclassified` |

原始数值仍来自 Observation。是否达到“明显变化”的阈值属于检索/统计配置，不写死在
Concept Schema 中。

### 3.4 Ledger 和兼容性

- [x] ExperimentPlan 磁盘形状升级为 Ledger v2.2，并提供 v2.0/v2.1 单向 migration。
- [x] 旧 Ledger 缺少新字段时恢复为空列表，不推断历史 Source 使用。
- [x] migration 不修改旧 round 的事实内容，也不伪造 application note。
- [x] trajectory round summary 显示 Concept/Source disposition、作用位置和应用说明。

### 3.5 P0 验收条件

- [x] Query event 保存 `catalog_snapshot`，可结合 Catalog Git 定位当时内容。
- [x] workspace 删除后，Observation 中的 Query event、Concept ref 和 Source ref
  仍可解析。
- [x] Observation 的时间是 Eval 执行时间，不是发布执行时间。
- [x] correctness recovery 使用 experiment parent，performance delta 使用明确标注
  的 performance baseline。
- [x] Concept query -> adopted plan -> Eval -> Ledger -> Observation 链路测试通过。
- [x] Source query -> detail read -> applied plan -> Eval -> Ledger -> Observation
  链路测试通过。
- [x] 只读取但未进入 plan 的 Source 不计为使用。
- [x] rejected 项不计为使用。
- [ ] 每个新 ExperimentPlan 显式输出 `knowledge_uses`；有 detail read 但列表为空时只进行一次整体确认，不逐项保存未使用记录。A100 已实现并通过测试，待 hw 同步验证后标记完成。
- [x] 一个 round 使用多项知识时标记 `combined`。
- [x] 没有 baseline 时保留使用记录，但效果为 `unclassified`。
- [x] P0 不因使用统计修改 Concept 正文。
- [x] A100 与 hw 全量测试通过。

## 4. P1：让检索利用真实使用效果

P0 完成后再做本阶段。

### 4.1 建立可重建的使用索引

- [x] 在派生 SQLite index 中增加统一 application 和 detail retrieval 表。
- [x] application 只保存可从 Observation 重建的字段；retrieval 只保存 Run Archive
  中成功 detail read 的 ref、event 和 scope。
- [x] 索引按 ref 和 exact evaluation scope 查询，并保留 workspace、effect 和 usage
  mode 供聚合。
- [x] 删除 `.derived` 后能够从 Catalog Observation 和 Run Archive retrieval event
  完整重建。
- [x] 不把 usage count、平均提升等高频变化写入 Concept Markdown。

`retrieved_count` 第一版明确指成功的 `get_knowledge/get_source` detail read，不把只在
query 结果列表中出现但未展开的条目算作已读取。

每个 ref 至少能查询：

```text
retrieved_count
considered_count
applied_count
evaluated_count
single_success_count
combined_success_count
correctness_recovery_count
regression_count
last_used_at
```

### 4.2 检索排序

- [x] 保持 scope 兼容作为前置条件。
- [x] 相同 Definition/workload/target/software 下的成功记录提高排序。
- [x] `single` 成功权重大于 `combined` 成功。
- [x] 相同 scope 下的 correctness/performance regression 降低排序。
- [x] Source 和 Concept 查询结果都返回简短 usage summary。
- [x] 返回可读的推荐理由，不只返回一个不可解释的 score。
- [x] 新知识没有使用历史时仍可通过语义和 scope 被检索，避免冷启动锁死。

### 4.3 多路召回与可解释融合

当前检索基线必须先明确：

- Concept 使用结构化 route 和精确 token 匹配，当前没有 BM25；
- Source 使用 aliases 扩词、正文子串频次和路径加权，当前没有 BM25；
- 历史 round 已使用 SQLite FTS5/BM25。

| ID | 状态 | 项目 | 目标 |
|---|---|---|---|
| `KB-R01` | `TODO` | Concept 多路召回 | 在不改变外部查询契约的前提下，增加结构化召回与 BM25 的可解释融合 |

`KB-R01` 按以下顺序实施：

1. 抽出内部 `ConceptRetriever`，保留当前结构化 route 召回作为第一路；
2. 使用 SQLite FTS5 为 Concept title、summary 和 body 增加 BM25 召回；
3. 使用 RRF 融合各路 rank，不直接相加不同量纲的原始 score；
4. 融合后继续执行现有 scope、evidence、usage 重排和 `claim_key` 去重；
5. 在 retrieval audit 中记录召回通道、各通道原始 rank 和融合理由，使每条结果可解释；
6. Concept 双路召回稳定后再评估 Source FTS5；relation 和 embedding 保持 `DEFERRED`，只有 committed Golden Query 证明存在语义漏召回时才启用。

本切片不修改五个 MCP 工具、`KnowledgeBundle`、Concept/Source/Observation 磁盘格式或 Publisher。第一阶段不引入向量数据库、embedding 服务或关系图检索。

`KB-R01` 验收条件：

- [ ] 现有 Golden Query 不回退，`direct/analogies/conflicts` 语义不变；
- [ ] 增加同义表达和正文语义命中的 Golden Query，证明 BM25 提供结构化 route 之外的增量召回；
- [ ] 多路融合的结果顺序确定性可重复，并能解释每条结果来自哪些召回通道；
- [ ] scope 明确冲突的结果仍不能进入 `direct`；
- [ ] 删除 `.derived` 后可以从当前事实源完整重建新索引；
- [ ] A100 全量测试和 Golden Query 通过后同步同一源码状态到 hw；不修改公共 Catalog 数据。

### 4.4 P1 已有能力验收条件

- [x] 同 scope 下，有独立成功记录的 Concept 排在无历史 Concept 前。
- [x] 不同 scope 的成功不能错误提升当前查询。
- [x] combined 使用不会被显示为单条知识的独立收益。
- [x] 负向 Observation 能在结果中看到，不被正向统计覆盖。
- [x] 查询输出通过 `observation_refs` 解释统计来自哪些 Observation。

## 5. P2：Source 晋升和 Concept 演进

### 5.1 Source -> Concept

- [x] Source detail read 会进入漏斗，但不会进入 Distiller 的 application result，
  因此不直接证明有用。
- [x] Source 首次实际应用并产生 correctness/performance 收益时，可生成
  `evidence_state=observed` 的 Candidate。
- [x] Source 支持的稳定事实可生成 `reference`；实测方法生成 `method`；失败到修复
  生成 `diagnostic`；单任务轨迹生成 `experience`。
- [x] Source-guided measured Concept 同时引用 exact Source fragment 和 Observation。
- [x] Distiller contract 要求同一 Source 的不同可行动 claim 分开生成 Concept，不把
  整篇文档压成一个条目。
- [x] 记录 Source fragment 到晋升 Concept 的映射，后续查询优先展示可行动 Concept，
  同时保留返回原文的入口。

### 5.2 evidence state

- [x] 单 workspace 的首次测量证据为 `observed`。
- [x] 至少两个独立 workspace 在兼容 scope 下重复成功后才变为 `corroborated`。
- [x] 相同 scope 下存在支持和反驳时变为 `contested`。
- [x] 反复失败且无支持证据时允许变为 `falsified`。
- [x] evidence state 从 Observation 确定性计算，不让 Agent直接填写。

### 5.3 scope 演进

| 项目 | 状态 | 当前处理与启用条件 |
|---|---|---|
| 首次生成保持 exact scope | `DONE` | Runtime scope 由 Python 绑定 Definition、target、software 和 workload |
| 多 workload 只放宽 workload | `DEFERRED` | 需要跨 Concept 聚合器先证明至少两个独立 workspace 的同 claim 逐 workload 成功；不得顺带改 target/software |
| 多设备兼容判断 | `DEFERRED` | 需要聚合器同时检查 architecture、software 与 revalidation 结果；设备别名规范化本身不足以证明可移植 |
| 部分 workload 拆分 Concept | `DEFERRED` | 需要把逐 workload 支持/反驳证据物化为可查询事实；当前 Observation 只引用 ledger 中的 workload 结果 |
| scope/claim 改变的写入规则 | `DEFERRED` | 需要先定义跨 exact Concept 的稳定 claim identity；当前不能把同 claim_key 的不同 exact scope 自动合并 |
| 使用次数不改 Concept | `DONE` | 使用统计只在可重建 index 和 metrics 中变化 |

Concept evidence、Source 或正文变化会原子更新同一当前文件；Git 保存内容历史。
这里的 `DEFERRED` 只针对跨 scope 聚合、扩展、收缩和拆分。启用前必须先实现一个
可单独预览和审核的 scope-evolution plan，不能在普通 publish 中隐式改写。

### 5.4 relations

| 项目 | 状态 | 当前处理与启用条件 |
|---|---|---|
| `adapted` 新 claim 提议 `refines` | `DEFERRED` | KnowledgeUse 记录 adaptation，但 Candidate 尚未声明“由哪个 ref 派生的新 claim” |
| 相反结果提议 `contradicts` | `DEFERRED` | 当前先在同 Concept 上派生 `contested/falsified`；只有形成两个独立 claim 后才有 relation target |
| 必要前提提议 `requires` | `DEFERRED` | 同一 round 使用多个 Concept 不能证明必要性，需要 Agent 明确 prerequisite 声明和 Observation 支持 |
| 应用链路优先于文本相似度 | `DEFERRED` | 先要求 Query 能沿 relation 图召回、Publisher 能检查方向与 scope；不引入纯文本猜测 |

现有 `relations` 只读兼容，Publisher 继续检查显式 target 是否存在。上述生产者和
图查询消费者同时具备前，不让 Runtime Agent 自动生成 relation。

## 6. P3：多 campaign 运行和公共 KB 运维

- [x] campaign 默认写 workspace 内的 candidate outbox；只有持锁的 Publisher
  能合并到公共 Catalog，Agent 不直接写公共 Catalog。
- [x] 每个 publish batch 保存新增/更新 Concept、Observation、使用关联和前后
  snapshot。
- [x] Observation-only batch 可按 audit 回滚；包含 Concept 变化的 batch 明确要求
  Git revert，避免伪装成可恢复的内容审计。
- [x] Concept 使用一个当前文件；Catalog Git 保存可审计历史。
- [x] Catalog transaction lock 串行化多个 campaign；Publisher 在最新 snapshot
  上合并，并通过 `rejected`、`contested` 和 audit 报告冲突。
- [x] 多算子 campaign 按 epoch 设置 barrier；发布或 synthesis 失败的算子退出
  后续 epoch，并可从完整 Agent Ledger 幂等重放 finalize。
- [x] 新 TargetContext 将 `Ascend910B`、`Ascend 910B`、`910B4-1` 等别名
  规范为稳定 family 名；旧 Concept 在索引和 scope match 时即时兼容，不迁移文件。
- [x] 环境版本变化时可从当前 TargetContext 生成待复验列表；只报告已知精确版本
  差异，不删除 Concept，也不自动写 `stale_after`。
- [x] Catalog validation、derived-index rebuild 和 publish replay 均有独立 CLI；
  rebuild 只重建 `.derived`，replay 复用 batch idempotency。

## 7. P4：衡量 KB 是否真的有用

至少记录以下系统指标：

- [x] query -> detail read 比例；
- [x] retrieved -> considered -> applied -> evaluated 漏斗；
- [x] 使用知识后的 correctness recovery rate；
- [x] single/combined performance improvement rate；
- [x] 每条 Concept/Source 在 exact evaluation scope 下的成功、回退和未分类
  结果分布；
- [x] 从 R1 到首次 PASSED 的 round 编号；
- [x] 从 R1 到 best kernel 的 round 编号；
- `DEFERRED`：`no KB / Source only / Concept / feedback ranking` 对照需要 Run
  Archive 先记录 campaign variant，当前不能从 mode 可靠反推；
- [x] 报告 exact duplicate、contested/falsified、never retrieved/applied 和
  query 无结果；“长期未使用”需要先确定时间窗口，不在第一版伪设阈值。

这些指标用于判断系统是否更好用，不作为单次 Eval 的阻塞门禁。

## 8. 当前先不做

以下工作暂不应挡住 P0/P1：

- 不建立覆盖所有 technique/domain 的封闭枚举；
- 不根据代码相似度猜测 Agent 是否用了某条知识；
- 不给多个同时采用的知识计算复杂的贡献分摊；
- 不根据一次成功自动声称 portable；
- 不为缺失字段重新引入 target fingerprint；
- 不要求每次发布人工审核；
- 不因为 usage count 变化重写 Concept；
- 不先建设新的多层目录或独立服务。

## 9. 字段取舍与简化 backlog

空字段不能统一“补满”或统一删除。判断标准是：是否有可靠生产者、是否参与决策、
是否能够验证。以下决策用于约束后续切片。

### 9.1 保留的条件字段

| 字段 | 决定 | 填写条件 |
|---|---|---|
| `capabilities` | 保留 | Definition/编译器事实明确要求，且 target 报告支持 |
| `software` | 保留 | 与 claim 相关的语言、编译器、runtime、library、driver 事实可采集 |
| `dataflow/layouts` | 保留 | Definition 或验证后的分析能够可靠确定 |
| `numerics` | 保留 | Definition/Eval contract 给出真实数值语义 |
| workload predicates | 保留 | 实际 workload 或多点证据支持范围 |
| `sources` | 保留并使用精确 ref | 正文包含 Source 支持的外部事实 |
| `relations` | Schema 兼容保留 | 只有明确已有 Concept 关系且图功能已启用时才生成 |

未满足条件时省略字段，不让 Agent 猜测，也不写空占位。

### 9.2 明确精简项

| ID | 状态 | 项目 | 决定与验收 |
|---|---|---|---|
| `KB-S01` | `DONE` | SourceReference 旧字段 | 新写入改为 revision/locator；旧字段只读兼容；commit `4695e01` |
| `KB-S02` | `DONE` | `Concept.type` | 已删除；只保留 `kind`，旧文件读取时迁移且新写入不再输出 |
| `KB-S03` | `DONE` | `Concept.generated` | 已合并进 `managed.created_by/created_at`，保留实际使用的 content hash |
| `KB-S04` | `DONE` | target/query/evaluation/profile fingerprint | 已改用结构化 target、event ID、solution SHA 和 Archive manifest；旧字段只读丢弃 |
| `KB-S05` | `DONE` | OperatorSignature 占位抽象 | 已删除 `semantic_names/structure_paths/extensions` 和强制 fallback validator；旧字段只读丢弃 |
| `KB-S06` | `DONE` | Query 重复字段 | 已删除始终复制 signature 的 `workload_summary`；scope matcher 直接读取 `operator_signature.workload_features` |
| `KB-S07` | `DONE` | Query result 冗余 | 已删除按 kind 硬编码的 `usage`；摘要不再携带完整 evidence，详情通过 `get_knowledge` 获取 |
| `KB-S08` | `DONE` | Observation 重复字段 | context refs 使用 Archive ref；已删除伪 baseline SHA、fingerprints 和无消费者 supersedes，旧字段只读丢弃 |
| `KB-S09` | `DONE` | ExperimentPlan 字段税 | 已删除未校准的 estimated range 和无消费者 risks/target_workloads；PlanSource 只保留 origin + parent_round_num，旧 Ledger 由 v2.5 迁移 |
| `KB-S10` | `DONE` | RoundConclusion 字段税 | 已删除无消费者的 architecture_tag、suggestion_followed、key_numbers、composition 和 diagnostic 占位字段；保留进入轨迹/决策的 optimization_level、debug_lesson、strategy_evolution |
| `KB-S11` | `DONE` | 无消费者代码 | 已删除 `PublishResult.bundle`、`KnowledgeLayout.usage_log`、零引用 `LocalSnapshotStore`、重复 WorkspaceMaterializer Protocol 和 3 个 publish dead helpers |
| `KB-S12` | `DONE` | Concept 多版本文件 | 删除 Concept revision 和 `@N` 引用；每个 ID 单文件原子覆盖，Query 用 catalog snapshot，历史由 Git 保存；Source revision 保留 |

### 9.3 暂缓启用的生命周期字段

| 字段 | 状态 | 当前处理 | 启用条件 |
|---|---|---|---|
| `verified` | `DEFERRED` | 取消自动排序奖励；旧自动生成值不代表真实审核 | 有 review method、reviewer、result 和 evidence |
| `stale_after` | `DEFERRED` | 不做静默过滤 | 有待复验队列、原因和 revalidation 结果 |
| `relations` | `DEFERRED` | 不要求 Runtime Agent 生成 | Query 能沿图召回，Publisher 能检查方向/scope |
| `status=draft` | `DEFERRED` | Runtime 信息不足仍拒绝发布 | 有独立 staging/review/promotion 流程 |
| software version constraint | `DEFERRED` | 新写入只允许精确版本 | Matcher 真正支持版本区间比较 |

公共 KB 中已有的 legacy `verified`、`exact: false` 等数据不在代码切片中顺手改写；
必须通过单独、可回滚的数据迁移处理。

### 9.4 检索字段接线

- [x] `domain` 和 `technique` 不能只建索引却永远没有 Query route；第一版已交叉索引
  为 keyword，不增加封闭枚举。
- [x] target backend/architecture/device/capability 结构化 route 已进入候选召回，
  scope matcher 继续负责最终兼容判断。
- [x] `numerics.exact: false` 不再作为有意义约束输出；只有真实要求 exact 时写入。
- [x] coverage gap 已按 query task 生成，不再对所有任务固定要求 reference 和 method。

### 9.5 架构简化

| ID | 状态 | 项目 | 目标 |
|---|---|---|---|
| `KB-A01` | `DONE` | KnowledgeMode 收敛 | 已收敛为“无配置即禁用 / 配置即 `read_write_v1`”；旧 mode 不再受支持 |
| `KB-A02` | `DONE` | Query log + Retrieval log | 已合并为 retrieval-log durable event；每次 query 只记录一次，旧 query-log 只读兼容；commit `09f1617` |
| `KB-A03` | `DONE` | Runtime/Static 共用 Candidate | RuntimeCandidate 只提交 scope_hints.motifs，完整 scope 由 Python 绑定；Static Candidate 继续提供完整 scope |
| `KB-A04` | `DEFERRED` | Source federated/allowed_hosts | 无真实 federated fetcher 前删除运行入口，不为未来空实现保留分支 |
| `KB-A05` | `DONE` | Source 可搜索性 | SourcePackage 显式 search_mode；searchable 强制本地 manifest，显式查询 static_only 返回错误，旧 package 按 manifest 只读推断 |
| `KB-A06` | `BLOCKED` | Legacy reducer + 双 Markdown report | V1 完成替换后收敛为一个结构化 run summary，删除平行 KB 写入链 |

## 10. 建议实施切片

依赖顺序优先于“改动看起来简单”。每片通过后更新状态和 commit。

### Slice 0：Source 精确溯源

状态：`DONE`，commit `4695e01`。

完成 `SourceReference revision/locator`、detail-read 行范围、旧字段兼容、Schema 和文档。

### Slice 1：单一当前 Concept 文件

状态：`DONE`。

主要位置：

```text
knowledge/catalog.py
knowledge/publishing/publisher.py
knowledge/index.py
knowledge/validation.py
tests/test_knowledge_*.py
```

Concept 引用改为稳定 ID；删除 Concept revision、`@N` 文件和 SQLite 复合主键。
Publisher 原子覆盖同一路径，Query event 用 `catalog_snapshot` 记录当时状态，
Catalog Git 负责内容历史。迁移工具选择旧文件最高 revision，归一为一个当前文件；
SourcePackage revision 保持不变。

### Slice 2：持久 Query event

状态：`DONE`。

主要位置：

```text
knowledge/contracts/runtime.py
knowledge/query.py
knowledge/sources.py
knowledge/records.py
knowledge/run_archive.py
```

完成唯一 Query event、单一 retrieval log、完整 QueryContext、旧 query log
只读兼容和 Run Archive 持久化。相同 Query 连续执行产生不同 event ID；
archive 后仍能解析每次 context 和结果。

### Slice 3：记录 Concept/Source 使用

状态：`DONE`。

主要位置：

```text
data/experiment_plan.py
data/ledger_migrations.py
.kernelgen/agents/kernel-coder.md
tests/test_ledger.py
tests/test_knowledge_*.py
```

完成统一 KnowledgeUse、精确 Source ref、Ledger v2.2 单向迁移、Coder contract 和
trajectory summary。未增加平行 SourceUse，也不把读取自动算作使用。

### Slice 4：Evaluation lineage 与 Observation 使用关联

状态：`DONE`。

主要位置：

```text
data/optimization_history.py
data/experiment_plan.py
data/ledger.py
knowledge/models/
knowledge/run_facts.py
kb/schemas/
tests/test_knowledge_*.py
```

完成 `evaluated_at`、experiment parent、performance baseline 和 3.3；能够从
Observation 确定性生成第一版使用效果视图。Ledger v2.2 单向迁移到 v2.3；旧
round 的缺失时间和 parent 不反推。

### Slice 5：使用效果进入检索

状态：`DONE`。

主要位置：

```text
knowledge/index.py
knowledge/query.py
knowledge/sources.py
mcp_server/knowledge_tools.py
tests/test_knowledge_golden_queries.py
```

完成 P1 的索引、排序和可读解释。

### Slice 6：Source 晋升

状态：`DONE`。

主要位置：

```text
agents/distiller/__init__.py
workflows/coder_and_distill.py
knowledge/publishing/publisher.py
tests/test_distiller_reports.py
tests/test_knowledge_*.py
```

完成 Python 生成的 Source application outcome、Distiller promotion contract，以及
Source query/detail read -> plan -> Eval -> Observation -> Candidate -> observed Concept
端到端链路。读取未使用的 Source 不会成为正向 application evidence；combined
成功不会被当作单项独立归因。

### Slice 7：字段精简

按 `KB-Sxx` 一项或一组强相关字段拆分提交。每次先从 Agent contract、Schema、
writer、reader 和 migration 全链路删除，不能只删 Pydantic 字段。

### Slice 8：架构收敛

KnowledgeMode 已完成收敛；P0/P1 稳定后再处理 Candidate 分离和 legacy
reducer。不得与 usage 闭环功能混在同一个提交。

## 11. 第一阶段完成定义

P0 和 P1 完成时，应能从任意成功或失败 round 回答：

1. 当时检索了哪些 Concept/Source？
2. Agent 实际采用、改写或拒绝了哪些？
3. 每项知识具体用于代码哪里？
4. 对比的 baseline round 是什么？
5. 正确性和性能发生了什么变化？
6. 这是单项使用还是组合使用？
7. 这些结果是否参与了下一次检索排序？

七个问题都能由结构化数据回答，且不依赖重新阅读 Agent 日志，才算形成了第一版知识演进闭环。

## 12. 保功能架构收敛目标

本轮架构审计不以“文件越少越好”或“删除行数最多”为目标，而以减少无效维护成本为目标。任何重构都必须保持下列外部行为：

1. `query_knowledge`、`get_knowledge`、`query_sources`、`get_source` 和 `search_rounds` 的 MCP 名称、参数和返回契约保持兼容；
2. `validate`、`rebuild`、`import`、`publish`、`revalidate`、`metrics` 和 `rollback` 的统一 CLI 保持可用；
3. Concept、Observation、Source、workspace state、retrieval log、Run Archive、Solution Registry 和发布审计的当前磁盘格式保持不变，已经承诺的旧格式读取兼容不得顺手删除；
4. direct、analogy、conflict 的 scope 判定、检索排序、usage feedback、Source 搜索和 Source 导入语义保持不变；
5. Candidate 修复、Observation 物化、幂等发布、并发发布锁、Catalog Git 提交和索引重建保持不变；
6. KnowledgeConfig 和 workspace state 只允许 `read_write_v1`，不保留旧 mode 运行分支；
7. 仓库内 `kb/` 继续作为可复现的初始 Source/Concept Catalog；不能仅加 `.gitignore` 后让安装者得到空 KB。

### 12.1 当前边界

模块已经改为按真实功能命名，不再保留 `domain/application/adapters/ingestion` 通用分层，但以下独立行为仍然分开维护：

- `models/` 和 `contracts/`：前者是持久知识模型和旧格式兼容，后者是 workspace、检索、Source 和发布的运行时契约；
- `query.py`、`sources.py`、`scope.py` 和 `index.py`：分别负责 Concept 检索、Source 检索、适用性判断和派生索引，不能合并成一个检索巨石；
- `publishing/`：事务编排、事实物化、Concept 合并、Candidate 校验和生命周期操作分别维护，但共用一个 Publisher 写入入口；
- `catalog.py`、`run_archive.py`、`run_facts.py` 和 `round_index.py`：分别处理权威 Catalog、不可变运行归档、Ledger 事实解释和历史 round 检索；
- knowledge 行为测试与 Golden Query：按职责整理后再依据重复的行为断言删除，不能按文件行数裁剪。

### 12.2 当前精简

| ID | 状态 | 项目 | 验收 |
|---|---|---|---|
| `KB-A07` | `DONE` | 删除无引用历史报告、无消费者接口和与当前代码冲突的旧操作说明 | A100 全量测试通过；hw 除镜像缺失依赖的测试文件外全部通过；当前文档不再声称 preflight receipt 阻塞 `eval_round` |
| `KB-A08` | `DONE` | 整理超大测试文件 | 原有 36 个运行时测试按职责拆分，共享 fixture 集中维护，测试数量和行为覆盖不变 |
| `KB-A09` | `DONE` | 收敛 `BatchPublisher` 内部职责 | 保留 `BatchPublisher` 公共入口和事务顺序；候选构建、事实物化、Catalog 写入及生命周期测试全部通过 |
| `KB-A12` | `DONE` | 扁平化 `domain`、`application`、`adapters` 和 `ingestion` 通用分层 | 删除通用分层，保留按功能命名的独立模块，没有把发布、查询、Source 搜索和持久化重新混进单个大文件 |

### 12.3 有条件收敛

| ID | 状态 | 项目 | 开始条件 |
|---|---|---|---|
| `KB-A10` | `BLOCKED` | 移除 legacy Markdown reducer | 已确认所有正式 campaign 只使用 V1，并提供旧 Markdown 的迁移说明和回滚点 |
| `KB-A11` | `DEFERRED` | 将初始 KB 数据拆到独立数据仓 | 有固定 revision、自动获取、离线回退和 Golden Query 验证；不能用裸 symlink 或静默空目录 |

### 12.4 每个切片的验收矩阵

每次架构切片都必须先在 A100 运行全量测试。涉及检索时还必须运行 committed Golden Query；涉及发布时必须覆盖幂等发布、并发锁、Observation lineage、Run Archive 和索引重建；涉及数据模型时必须运行 committed Schema 与 seed Catalog 校验。需要部署到 hw 时，再同步同一源码状态并运行 knowledge、MCP 和 workflow 相关测试；A100 源码提交与 hw 部署提交分开记录，不能把尚未同步的状态写成已经部署。

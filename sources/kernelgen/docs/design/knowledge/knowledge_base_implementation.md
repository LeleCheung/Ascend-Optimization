# KernelGen KB 完整设计

本文讲清楚 KernelGen 当前知识库为什么存在、里面保存什么、一次优化怎样使用和产生知识、知识有没有用怎样统计、历史 best code 怎样复用，以及发生错误时怎样恢复。

本文描述已经接入 `KernelGenWorkflow` 的 V1 流程。V1 KB 当前是显式启用能力，
不是 SimpleOpt/BatchSimpleOpt 的生产默认项；未传入 `KnowledgeConfig` 时
KernelGenWorkflow 保持无 V1 的普通路径。传入配置时允许 `read_only_v1` 或
`read_write_v1`；旧 workspace state mode 不再受支持。只读模式只保证 Catalog
及 run archive 不变，workspace 运行记录仍正常写入；Catalog 内索引只能读取，
缺失或过期时 fail closed，显式外置 derived root 才允许重建。

`OperatorOptimizeWorkflow` 的 simple_opt 模式默认不读取或发布 V1 KB；提供 `knowledge_catalog_path` 时会生成 Knowledge workspace state，复制 Skill，并使用 Knowledge Coder/Profile/Distiller，但 CandidateDraft 只留在本地，不执行 Publisher、Solution promotion 或 fork。新 YAML Batch 为各子任务复用此配置，legacy Python Batch 的历史能力不变。它们与 Knowledge-enabled KernelGenWorkflow 共用一个 Python `CoderAgent`，并通过 native Agent role 和工具白名单隔离权限。

如果只想启动任务，读 [`KB_QUICKSTART.md`](../../guides/KB_QUICKSTART.md)。如果要查某个字段的严格约束，读 [`knowledge_concept_contract.md`](knowledge_concept_contract.md)。如果要看还没完成的改进，读 [`knowledge_base_todo.md`](knowledge_base_todo.md)。本文是理解整个系统的主文档，不把未来计划写成已经上线的能力。

## 1. 先用一句话说明 KB 是什么

KernelGen KB 不是“一堆 Markdown”，而是一条从资料和历史实验出发，经过检索、实际使用、真实评测，再回到可复用知识的闭环。

```text
Source / Concept
        ↓ 检索并展开详情
ExperimentPlan 声明本轮实际采用、改造或拒绝了什么
        ↓
代码 + Eval + Ledger
        ↓
Observation 冻结真实 round
        ↓
Candidate 提议可复用结论
        ↓
Publisher 校验并更新 Concept
        ↓
下一次检索根据适用范围、证据和历史效果排序
```

这里最重要的原则是：模型负责提出问题、选择资料、做实验和解释结果；Python 负责保存真实上下文、评测结果、引用关系、发布事务和可重建统计。模型不能把一句看起来合理的话直接写成正式知识。

## 2. 这个系统解决什么问题

优化算子时，Agent 经常需要回答下面几类问题：

- 这个硬件、编译器或 DSL 到底支持什么？
- 对这个算子和 shape，哪些优化方法值得先试？
- 出现编译失败、精度错误或性能回退时，过去怎样定位过？
- 同一个算子以前跑过什么，哪些参数成功过，哪些失败过？
- 过去的 best kernel 能不能作为本次起点？
- 某条知识被检索后到底有没有真正用到，结果变好还是变差？

如果只保存最终回答，这些问题无法可靠回答；如果把所有日志都塞进检索，又会得到大量噪声。因此系统把原始资料、完整运行事实、可复用结论和 best code 分开保存。

## 3. 先分清这些对象

| 对象 | 说人话 | 谁产生 | 是否进入正式 Catalog |
|---|---|---|---|
| Source | 原始文档、仓库、博客、论文等资料快照 | 导入工具或维护者 | 是 |
| Query / Retrieval event | 当时为什么搜、返回了什么、真正打开了什么 | 查询服务 | 否，随运行归档 |
| ExperimentPlan | 本轮准备改什么，以及实际声明采用、改造或拒绝了哪些知识 | Coder | 否，写入 Ledger |
| Ledger | 一个 Agent 每个 round 的计划、代码身份、评测、profile 和结论 | Python 工具链 | 否，进入 Run Archive |
| Candidate | Distiller 提交的“这可能值得成为知识”的建议 | Distiller | 否，是发布输入 |
| Observation | Python 从真实 Ledger round 冻结出的不可变事实 | Publisher | 是 |
| Evidence | 一条 Observation 对某个 Concept claim 是支持、反驳还是仅作案例 | Distiller 提议方向，Publisher 绑定事实 | 是，位于 Concept 内 |
| Concept | 面向以后检索的可复用结论 | Publisher 创建或更新 | 是 |
| Solution | 某个精确环境和 benchmark 当前最好的已评测 kernel | Workflow 提升 | 是，独立于 Concept |
| Run Archive | 完整 Ledger、上下文、检索记录和大体积 artifact 的归档 | Workflow | 否，与 Catalog 同级保存 |

可以把它们理解成：

- Source 是“原始教材”。
- Ledger 是“实验记录本”。
- Observation 是“从实验记录本里冻结的一条事实”。
- Concept 是“以后可以直接拿来行动的知识卡片”。
- Solution 是“以后可以直接继续优化的代码起点”。
- Candidate 是“申请成为知识的草稿”，不是正式知识。

## 4. 哪些内容是权威事实

不同字段必须由真正知道答案的一方填写，不能让一个 Agent 自由补全所有内容。

| 内容 | 权威来源 |
|---|---|
| definition、op type、dtype、静态 axes、显式 dataflow/layout、数值 contract | Definition |
| backend、architecture、device、compiler、runtime、driver 和版本 | Eval service `/status` |
| 每轮代码、状态、速度、误差、父 round、性能基线 | Ledger / Evaluation |
| profile finding | 已记录的 ProfileAnalysis |
| 本轮实际落实到当前候选中的知识 | ExperimentPlan.knowledge_uses |
| claim、适用建议、检索词、正文解释 | Distiller Candidate |
| Observation ID、真实 scope、evidence state、managed metadata | Publisher |
| 当前最佳代码是否应替换 | Solution Registry 根据真实 geo mean 判断 |

如果 `/status` 不可达，或者没有报告 backend/device，Knowledge bridge 直接失败，
不会使用 orchestrator 的硬件名 fallback。输入 `target_hardware` 非空时还必须与
Server 实际 device 归一化一致。软件版本等可选 scope 字段没有可靠来源时保持为空，
不得由 Agent 补写。

系统不为缺失字段猜值。没有可靠 dataflow、layout、capability、software 版本或 numerics 约束时，保持为空并在 Markdown 中省略，比写一个看起来完整但错误的值更好。

## 5. Source：原始资料层

Source 保存外部资料的原貌和精确身份，适合回答“原文到底怎么说”“这个 API 在哪里定义”“有没有可参考代码”。

一个 SourcePackage 至少说明：

- 它是谁，例如 `source:triton-ascend...`；
- 原始来源在哪里；
- 当前快照 revision 是什么；
- 内容在 Catalog 中的逻辑位置；
- 是否允许搜索；
- 权威性、许可和使用限制是什么。

Source 有两种搜索模式：

| 模式 | 含义 |
|---|---|
| `searchable` | 内容和 Git manifest 已放入 Catalog，可以用 `query_sources` 和 `get_source` 搜索、读取 |
| `static_only` | 只作为已发布 Concept 的来源记录，不向 Agent 暴露原文搜索 |

Agent 不应该猜 Source package ID、路径或 revision。正确流程是先调用 `query_sources`，再把返回的 `query_id`、`source_package` 和 `path` 原样交给 `get_source`。`get_source` 每次最多读取 400 行，防止一次把整座仓库灌进上下文。

Source 不是越多越好。真正有用的 Source 应该有稳定 revision、可追溯 locator、明确使用范围，并能被具体问题命中。

## 6. Concept：可复用知识层

Concept 是 Agent 可以查询的知识卡片。它不追求记录整个实验过程，只回答一个清晰、可行动的 claim。

Concept 有四类：

| kind | 回答什么 | 典型内容 |
|---|---|---|
| `reference` | 有什么稳定事实或约束 | 某 API 的语义、某硬件能力、某 dtype 限制 |
| `method` | 应该怎样执行一种优化方法 | 分块归约、向量化加载、流水调度 |
| `diagnostic` | 出现某种症状怎样定位 | 精度异常、编译失败、带宽利用率低 |
| `experience` | 在特定条件下实际发生过什么 | 某 tile 在某 shape 上回退，某 mask 写法修复正确性 |

一个 Concept 只表达一个原子 claim。不要把“加载、归约、调度和精度处理”全塞进同一条，因为检索、证据合并和后续修正都会变得含糊。

### 6.1 正文应该怎么写

`reference`、`method` 和 `experience` 使用固定结构：

```markdown
## Claim

一句话讲清可复用结论。

## Evidence

说明 Source 或 R<n> 证据，以及证据实际证明到什么程度。

## Applicability

说明适用硬件、软件、算子、shape、dtype、layout 和数值条件。

## Action

说明以后遇到该条件时具体做什么。

## Limits

说明未验证范围、失败边界和不能推广的地方。
```

`method` 的 `Action` 内还必须有 `### Expected Metric Change`，`Limits` 内必须有 `### Mechanism Requirements`，这样方法不只说“做什么”，还说明预期看到什么指标变化以及成立所需机制。

`diagnostic` 使用另一套结构：

```markdown
## Symptom

## Likely Causes

## Candidate Techniques

## Diagnosis Checklist

## Caveats
```

正文不能泛泛地写“合理设置参数”“注意内存访问”“根据场景调优”。有用的正文至少应说清条件、动作、预期信号和限制。实验型正文引用的所有 `R<n>` 必须与 `observation_intents` 完全一致。

### 6.2 Concept 文件怎样演进

每个 Concept ID 在 `concepts/` 中只有一个当前文件：

```text
concepts/<kind>--<slug>.md
```

引用使用稳定 ID，例如 `kg:method:vectorized-load`，不再带 `@1`、`@2`。Publisher 原子覆盖当前文件，Catalog 的 Git commit 保存历史、差异和回滚点。因此知识演进不会为每次 revision 新建一个 Markdown 文件。

Source revision 仍然保留，因为它表示外部资料的具体快照；Concept revision 被删除，因为 Concept 历史已经由 Git 管理。

Runtime Candidate 的合并键是 `kind + claim_key + scope`。新 Concept 创建时，同组 Candidate 的 evidence 和 Source 会合并，正文由确定性排序后的第一条 Candidate 提供；已有 Runtime Concept 再收到同 claim、同 scope 的证据时，当前实现只追加 evidence/Source 并重算 evidence state，不自动重写 title、summary、retrieval、relations 或正文。审核过的静态导入可以更新这些内容。也就是说，当前系统已经能让证据持续累积，但“根据新实验自动重写旧正文”仍不是已上线能力。

## 7. Scope：这条知识到底适用于哪里

Concept 的 scope 不是装饰，它决定查询结果属于直接可用、类比参考还是不兼容。

Scope 分四部分：

| 部分 | 表达什么 |
|---|---|
| `target` | backend、architecture、device、capability 和 software |
| `operator` | definition、op type、motif、dataflow、dtype 和 layout |
| `workloads` | shape、轴值或其他 workload 特征的条件 |
| `numerics` | exact、TF32、累加 dtype 和误差要求 |

### 7.1 多芯片和多软件怎样表达

Target 有五个层级：

| level | 含义 |
|---|---|
| `exact` | 精确到 backend、architecture 和一个 device |
| `device` | 适用于一组明确 device |
| `architecture` | 适用于某个架构族 |
| `backend` | 只要求某个 backend |
| `portable` | 不依赖具体硬件身份 |

Runtime Candidate 默认由 Python 绑定为当前一次实验的 `exact` scope，因为一次实验只能证明它实际跑过的环境。它不能因为正文写了“通用”就自动扩大到所有 Ascend 芯片。

Source 导入产生的静态 Concept 可以使用更宽 scope，但必须由审核过的 ingestion plan 明确填写。将多个 exact Observation 汇总成 architecture 或 portable 知识，需要单独的 scope 演进流程；当前 Publisher 不会自动做这种推广。

软件名称会参与匹配。当前版本字段可以保存精确版本；看起来像 `>=`、`<` 的约束目前还没有真正的版本区间比较器，因此只会产生 scope gap，不能当成已完整支持的能力。

### 7.2 Workload 多了怎么办

Workload 不应该把所有 shape 复制成一个巨大列表。常见做法是：

- 当前 Runtime Observation 使用 Definition axes 生成精确 `eq` 条件；
- 多个离散值确实共享结论时，审核后可以使用 `in`；
- 只有证据证明存在范围边界时才使用 `gte`、`lt` 等条件；
- 有明确反例时放进 `excludes`；
- 无法可靠概括时保留多个 exact Concept，而不是过早合并。

### 7.3 查询时怎样判定

Scope matcher 的结果只有三种：

| 结果 | 含义 |
|---|---|
| `direct` | 已声明约束全部匹配，可以直接考虑 |
| `analogy` | 没有发现冲突，但当前上下文缺少部分字段，只能类比 |
| `incompatible` | 至少一项明确冲突，不应作为普通建议返回 |

“缺少信息”和“明确相反”不是一回事。比如 Concept 需要某 capability，而 `/status` 没报告任何 capability，这是 gap；如果服务报告了 capability 集合但明确不包含它，才是冲突。

## 8. Observation 和 Evidence：事实与解释分开

Observation 是从真实 round 冻结的记录，包含测量事实与 Agent 声明：

- run、workspace、definition 和 round 身份；
- target/operator context 引用；
- 被评测代码的 SHA；
- Eval status、geo mean、workload result 引用和评测时间；
- 本轮 `experiment_plan`、`code_changes` 和 `agent_conclusion`；
- experiment parent 和 performance baseline；
- 本轮声明的知识 applications；
- 相关 artifact 引用。

当前持久化契约是 Observation v2。实验计划、声明的代码变化和 Agent 结论与测量结果一起冻结，Reviewer 直接读取同一条 Observation，不再使用平行的 `observation_facts`。Reviewer decision audit 只保存 `observation_refs`，避免复制事实。v1 Observation 和旧冗余字段会被严格拒绝；上线 Catalog 必须先迁移旧记录，不能依赖兼容读取。

“冻结”表示不可重写记录，不表示记录中的解释都正确。`experiment_plan` 是实验意图，`code_changes` 直接来自 `record.plan.code_changes`，不是从源码计算得到的 diff；`agent_conclusion` 是 Agent 的解释。测量只证明被评测代码在对应 workload/条件下的结果，Python `automatic_checks` 只证明其列明的结构与 provenance 条件，二者都不自动证明计划中的参数真正生效、瓶颈归因或编译器根因。Reviewer 对实现相关 claim 应核对可访问的源码绑定证据，包括 host dispatch 和有效参数；诊断 claim 还要排除错误 control、源码错误及未测 correctness 等混杂。必要证据不可访问或不足时 defer，证据明确矛盾时 reject，不修改原 Observation。该边界不改变 Schema、ledger 的测量权威性或 Publisher 的精确引用/详情阅读门禁；提示规则也不保证模型每次都作出正确语义判断。

Observation 不写“这证明了什么”。同一事实可能支持一个负面经验，同时反驳一个正面方法，因此解释放在 Concept 的 Evidence 中。

### 8.1 `claim_stance` 到底指什么

Distiller 在 Candidate 中填写 `claim_stance`，它永远表示“该 Observation 相对于当前 Concept claim 的方向”：

| 值 | 含义 |
|---|---|
| `supports` | 实验结果与当前 Concept claim 一致 |
| `refutes` | 实验结果与当前 Concept claim 相反 |
| `illustrates` | 是相关案例，但不足以独立证明方向 |

不要把“原始实验假设失败”直接写成 `refutes`。例如一轮实验原本假设“减小 BLOCK_N 会提升性能”，实际发生回退；这条结果反驳原实验假设，但支持“减小 BLOCK_N 在该条件下会回退”的负面 Concept，所以对这个负面 Concept 应写 `supports`。原实验假设为什么失败写进 rationale 或运行报告。

旧 Candidate 的 `stance` 仍能读取，新 Candidate 只输出 `claim_stance`。Catalog 内部 Evidence 继续使用 `stance`，因为它已经明确位于 Concept 语境中。

### 8.2 `evidence_state` 怎样算

Publisher 根据 Source 和 Evidence 自动计算：

| 状态 | 含义 |
|---|---|
| `source_supported` | 当前只有可追溯 Source 支持 |
| `observed` | 一个 workspace 的实验提供了支持或案例 |
| `corroborated` | 多个独立 workspace 提供了支持或案例 |
| `contested` | 同时存在支持和反驳证据 |
| `falsified` | 当前只有反驳证据 |

`contested` 不是发布错误。它表示知识存在真实冲突，查询时会进入 `conflicts`，提醒 Agent 不要把它当成稳定建议。

## 9. Agent 怎样检索知识

知识检索结果不会预先整包塞进 Analyzer 或 Coder prompt。Agent 的角色说明只告诉它何时应该调用知识工具；是否检索、查询什么和展开哪条详情，由 Agent 按当前问题决定。

这样做有三个原因：

- 不相关知识不会占用上下文；
- 每次查询都有明确问题和 phase；
- 系统可以区分“返回过”“打开过”和“实际采用过”。

当前五个知识工具是：

| 工具 | 用途 |
|---|---|
| `query_knowledge` | 搜索已发布 Concept |
| `get_knowledge` | 展开查询返回的 Concept 正文 |
| `query_sources` | 搜索原始 Source |
| `get_source` | 读取命中的 Source 片段 |
| `search_rounds` | 搜索 Run Archive 中的历史实验 |

Analyzer 与 Epoch Summary 的推荐顺序是：

```text
先查 Concept
→ 有合适结果就展开详情
→ Concept 不足以回答 API、编译器、硬件或代码细节时再查 Source
→ 需要比较过去具体实验时查历史 round
```

Analyzer 用 `initial` phase 做冷启动分析；Coder 根据时机使用 `post_error`、`post_evaluation`、`post_profile` 或 `plateau`；Profile Analyzer 使用 `post_profile`；Epoch Summary 使用 `post_evaluation + next_experiment`。

### 9.1 Coder 和 Profile Analyzer 的双路检索

Coder 和 Profile Analyzer 在提出会影响非 baseline 实验方案的知识建议时，都围绕同一个具体技术问题查询 Concept 和 Source。Concept 用来寻找已验证、可复用的经验；Source 查询固定使用 max_results=4，用来寻找原始实现、API、编译器或硬件事实。

它们只展开真正相关的少量内容：最多读取两个 Source 局部片段，每次最多 200 行。精确 scope 且有测量支持的 Concept 用来判断已观察到的结果；Source 用来补充更具体的实现依据。两者可以共同影响一个方案。

Source 不会全文预注入 Coder 或 Distiller prompt。Profile Analyzer 只在紧凑建议中交回实际参考的精确 Concept ID 或 Source ref；Coder 只有重新读取并把它用于冻结 ExperimentPlan 后，才能声明实际使用。

### 9.2 Concept 查询怎样排序

查询先用 phase、task、definition、op type、backend、architecture、device、capability、motif、dataflow、dtype、layout、symptom 和问题关键词召回候选，再用 scope matcher 判定。

结果分成：

- `direct`：scope 完整匹配；
- `analogies`：没有冲突，但有 scope gap；
- `conflicts`：Concept 已被标为 contested 或 falsified。

排序综合考虑词法命中、scope、target 精度、evidence state、已有 verified 记录和同 scope 历史使用效果。相同 `claim_key` 只保留分数最高的一条，避免结果里重复出现同一个结论。

如果 `stale_after` 已存在且日期已过，当前查询会跳过该 Concept。系统暂时没有自动复验队列，因此 Runtime Publisher 不会随意生成 `stale_after`。

### 9.3 Source 查询怎样排序

Source 查询使用开放关键词和 aliases 扩展，按 package、path、文件类型、大小和 ingestion 规则过滤。命中片段如果已经被提升成可行动 Concept，会返回 `promoted_concept_refs` 并获得排序加权，但原始 Source 仍可以单独阅读。

Source 查询不会把 `static_only` 伪装成“零命中”。显式查询或读取 `static_only` package 会报清晰错误。

## 10. 怎样判断知识是否真的被使用

“搜到过”不等于“使用过”。系统把使用过程拆成一条漏斗：

```text
query returned
→ detail read
→ applied to the submitted solution
→ evaluated
→ correctness 或 performance outcome
```

只有 Coder 可以在 `ExperimentPlan.knowledge_uses` 中声明实际使用。这里的“使用”不是“帮助思考”，而是知识已经落实到本轮 `eval_round` 提交的 kernel 或其启动、分派配置中。每个 ExperimentPlan 都必须显式包含该列表；没有实际应用时写 `[]`。

| 字段 | 含义 |
|---|---|
| `concept_ref` 或 `source_ref` | 二选一，说明用了哪条知识 |
| `query_event_id` | 它来自哪次查询 |
| `role` | constraint、hypothesis、implementation 或 diagnostic |
| `disposition` | 新提交只能是 adopted 或 adapted；rejected 仅为历史读取兼容 |
| `application_note` | 具体怎样采用或改造并落实到当前候选 |
| `affected_parts` | 当前 kernel 或执行配置中受影响的具体部分 |

Source 使用必须记录精确的 `resource + revision + locator`。只调用 `get_knowledge` 或 `get_source` 仍然只是“读过”；仅影响分析、假设、Profile 解释、诊断方向或最终被放弃的知识不进入 `knowledge_uses`。知识即使导致精度错误或性能回退，只要确实落实到该候选中，仍然属于实际应用，效果由 Evaluation 判断。

当本轮存在成功的 detail read 而 Coder 提交空列表时，`eval_round` 会在真正评测前返回一次 `KNOWLEDGE_USE_REVIEW_REQUIRED`。Coder 只需补上实际应用项，或者以 `confirm_no_knowledge_applied: true` 明确确认本轮没有采用；系统不会为每条未采用知识保存拒绝记录。

知识使用声明是 provenance，不是权限凭证。Query log 缺失、损坏、detail read 缺失或历史 lineage 不完整不会阻止真实 Eval。Publisher 仍会把链路标记为 `complete`、`query_missing`、`detail_not_read` 或 `legacy_unknown`，供离线检查。

Profile Analyzer 可以把精确 Source ref 放进建议交给 Coder，但只有 Coder 重新读取并将其用于冻结方案后，才能记录实际使用。

## 11. 使用结果怎样进入统计和下一次检索

Publisher 会把 round 中实际应用的 `knowledge_uses` 冻结到 Observation 的 `applications`。即使 Distiller 没有产生任何新 Candidate，只要一个已评测 round 声明过知识使用，Publisher 也会生成对应 Observation。历史 Ledger 中的 `rejected` 仍可读取，但不计入 applied 或 usage mode。

Python 根据父 round 和性能基线把整轮结果分类为：

| effect | 含义 |
|---|---|
| `correctness_recovered` | 父 round 未通过，当前通过 |
| `correctness_regressed` | 父 round 通过，当前未通过 |
| `performance_improved` | 父/当前都通过，并且相对性能基线提升 |
| `performance_regressed` | 父/当前都通过，并且相对性能基线下降 |
| `no_material_change` | 变化没有超过 materiality 阈值 |
| `unclassified` | 缺少可比较条件 |

`usage_mode` 按本轮 adopted/adapted 数量分成 `none`、`single` 和 `combined`。`single` 表示只有一项知识被声明应用，关联更清楚，但仍不等于严格因果证明；`combined` 更不能把整轮提升归功于其中任意单项。

客观 effect 只回答“应用知识的这一轮发生了什么”，不能解释知识机制是否真的成立。因此，Coder 在评测后通过 `RoundConclusion.knowledge_assessments` 对本轮每个 `knowledge_uses` 项逐条复盘。引用必须与冻结 Plan 中的 Concept 或 Source 精确一致；没有应用知识时写空列表。

| assessment | 含义 |
|---|---|
| `confirmed` | 评测和可用 Profile 证据支持该知识应用的预期机制 |
| `partially_confirmed` | 只验证了部分机制或部分 workload |
| `not_confirmed` | 结果没有支持该机制，或出现与预期相反的证据 |
| `inconclusive` | 当前实验不足以判断 |

Agent 提供机制判断和理由，Python 继续拥有状态、性能差值和正确性转换等客观事实。多项知识同时使用时可以逐条复盘，但除非实验隔离了某一项，否则只能说明组合证据，不能把整轮变化独立归因给每一项。每个新 conclusion 都必须显式输出 `knowledge_assessments`，没有使用知识时输出 `[]`。

同 scope 历史反馈会进入下一次排序。这里的“同 scope”要求 definition、target backend/architecture/device、software 和 workload features 一致。单项成功比组合成功的加权更强，正确性或性能回退会降低推荐。Agent assessment 会随 Observation 保存并出现在 usage summary 和 metrics 中，但暂不重复修改排序分数，避免与同一轮客观 effect 重复计权。跨设备、跨软件和跨 workload 的效果不会直接混进当前加权。

运行：

```bash
python3 -m kernelgen.tools.kb metrics \
  --kb /path/to/catalog \
  --run-archive /path/to/run-archive
```

可以得到：

- query 到 detail read 的比例；
- retrieved、considered、applied、evaluated 漏斗；
- correctness recovery 和 performance improvement；
- 每个 Concept/Source 在精确 scope 下的结果分布；
- Agent 对每条实际应用知识的 confirmed、partially confirmed、not confirmed 或 inconclusive 判断；
- 从未检索、从未应用、contested、falsified 和重复 claim；
- 每个 workspace 的 first passed round 和 best round。

当前 Run Archive 不记录 campaign variant，因此 `campaign_comparison.available=false`。系统可以统计“某条知识在哪些 round 被声明使用以及结果怎样”，但不能直接自动完成 no-KB、Source-only、Concept-only 等 campaign 级因果对照。

## 12. 一次完整运行发生什么

### 12.1 运行开始

`KernelGenKnowledgeBridge` 做三件事：

1. 从 Definition 构造 `OperatorSignature`；
2. 从 Eval service `/status` 构造严格 `TargetContext`，并校验 orchestrator
   输入与实际 device 一致；
3. 给每个隔离 workspace 写入 `state.json`、`operator-signature.json` 和 `target-context.json`。

Workspace 只保存中央 Catalog 的路径和本地运行上下文，不复制整份公共 KB。

### 12.2 Analyzer 和 Coder 工作

Analyzer prompt 不预注入检索结果。它需要资料时按需调用 Concept 或 Source 工具。

Coder 每轮执行：

```text
提出 ExperimentPlan
→ 修改 kernel
→ preflight_kernel
→ eval_round（评测并原子记录 immutable round）
→ 必要时 profile
→ finalize_round 原子写入结论、candidate transition 和 STOP verdict
```

每个 Agent 只写自己的 workspace。并行 Agent 之间不会直接覆盖 Ledger、代码或 Candidate。

### 12.3 Distiller 产生运行报告和 Candidate

Coder 结束后，Distiller 读取真实 Ledger trajectory、profile、可用 Source/Concept provenance 和 Source application outcome，产生：

```text
.new_experience.md
.new_detailed.md
.kernelgen/knowledge/candidates.jsonl
```

两份 Markdown 是单次运行报告，不是正式 Concept。`read_write_v1` 不把它们合并进 Catalog；它们用于人工阅读和 epoch synthesis。`candidates.jsonl` 才是 V1 Publisher 的结构化输入。

Distiller 的源码证据采用「完整初始实现 + 每轮 diff」，改写比例较大时重新提供完整实现。不能只截取 JIT 函数：host 选参、launch/grid、全局常量与 helper 都可能改变实际执行，函数体中的空行也不能截断后续计算。Epoch synthesis 仍使用有预算上限、不重复源码的各 Agent 轨迹，仅对唯一权威 best 提供一次完整源码。计划与结论是待核对的模型叙述，不能用它们代替实际源码证明某个参数已生效；测量与源码均来自原始 ledger，不改写历史事实。

Distiller 的逐轮轨迹和 epoch synthesis 的 best/final 证据还会从原始 `evaluation.workloads` 派生 timing 配对：分别显示候选时延与 reference 时延的几何平均变化，口径为各测例 `after/before` 的几何平均减一。候选变化为负表示变快；reference 变慢可能提高 headline speedup，即使候选完全没有提速。性能 incumbent 与实验 parent 分别按 ledger 中的 round ID 配对，两者相同则只展示一次，不把相邻 round 默认当作实验父版本。配对要求两轮均 PASSED，timing UUID/axes 集合一致且不重复，全部 timing 通过、时延为正有限值；基线不在当前轨迹或数据不足时明确显示 unavailable，不悄悄筛选有利子集。该摘要只提供观察证据，不是因果归因、统计显著性或噪声下限，也不改变 ledger、best 选优、Observation effect 分类和持久化 Schema。

如果某一条 Candidate 不符合契约，Distiller 最多进行两次单条修复，不重新生成整份报告。仍然无效时只跳过该条，其他 Candidate 和已完成 Ledger 保留。如果整个 Distiller 输出连报告 contract 都无法解析，Distillation 会被跳过，优化结果仍然有效，但本 Agent 不会产生新的报告或 Candidate。

### 12.4 epoch 结束时发布

固定顺序是：

```text
归档每个 Agent workspace
→ 重建历史 round 索引
→ 收集 usage Observation 和 Candidate
→ 从真实 workspace 绑定 scope、round 和 provenance
→ 在 staging 中创建 Observation v2 和 Candidate review unit
→ Reviewer 对照附近 Concept、Source 和同批 Candidate 做语义审核
→ 只把批准的 Candidate 合并为 Concept
→ 写正式 Observation
→ 原子更新 Concept
→ 重建索引
→ 写 Reviewer decision audit 和 batch audit
→ 提交 Catalog Git
```

Reviewer mode 为 `off` 时不调用模型且 Candidate 保持 deferred；`shadow` 只记录
决定；`enforce` 才允许批准项进入 merge。同一 epoch 无论被切成多少 packet，都只
创建一个 `<epoch>R/knowledge-reviewer` 工作区。由于 runtime log 与 retrieval audit
是共享追加文件，真实 Reviewer 调用在该目录内串行执行，每个 decision 仍只绑定
本次调用新增的 detail-read events；无 Candidate 时目录保持未创建。

本 epoch 发布成功后，下一 epoch 可以检索到这些新知识。同一 epoch 中并行 Agent 看不到彼此尚未发布的 Candidate。其他 campaign 若中途更新公共 Catalog，后续按需查询会看到当前最新内容；每个 Query event 的 snapshot 记录它当时看到的 Catalog 状态。Catalog 本身是一个独立 Git 仓库时，Workflow 才会自动提交本次变化；没有独立 `.git` 时发布仍然有效，但不会凭空创建版本历史。

发布和 epoch synthesis 共用一个 finalize 路径。V1 发布结果必须为 `published`
或 `noop`；多 Agent epoch 还必须写出可解析的 `synthesis.json`。任一步失败都使
该 epoch 保持 incomplete，不能进入下一 epoch。

### 12.5 运行结束时提升 best solution

Workflow 会把全局 best workspace 的已评测代码尝试提升到 Solution Registry。只有新 geo mean 严格大于同一 slot 的现有值才替换。

Solution slot 按 target、definition 和 benchmark 区分，保存：

```text
solutions/<target>/<definition>/<benchmark>/
├── manifest.json
└── best_kernel.py
```

Concept 保存“应该怎样做”，Solution 保存“当前最好的代码是什么”，两者不能互相替代。

benchmark identity 由 Server 内置 Catalog manifest 的 `name + api_version` 生成，例如 `flaggems-adapter-definitions` 在 API v6.2 下对应 `flaggems-adapter-definitions-v6.2`。它不使用旧 `trace_set_key`，也不增加内容哈希。

这里的“精确匹配”按当前代码具体指 definition、benchmark、backend、architecture、device 和 implementation language。Manifest 会记录更完整的 evaluation scope，但 compiler/runtime/driver 版本目前不参与 `resolve()` 的拒绝条件；环境版本明显变化时应先检查 `revalidate` 结果并选择 fresh，不能假设 fork 已经完成版本兼容判断。

## 13. Source 怎样变成 Concept

Source 不会因为“被搜了很多次”就自动变成 Concept。检索频率只能说明大家关心它，不能证明其中某句话正确、可行动或适用于当前硬件。

当前有两条 Source→Concept 路径。

### 13.1 静态、审核过的导入

维护者先建立 SourcePackage 和 ingestion plan。plan 为每个路径指定是仅索引、直接保留原文、人工抽取、只作为 Source，还是不进入本地搜索。

`kb import` 会先构造 Source-backed Candidate 并校验 scope、正文、locator 和 package policy。默认 dry-run，只有显式 `--publish` 才走与 Runtime Candidate 相同的 Publisher 事务。

```bash
python3 -m kernelgen.tools.kb import \
  --kb /path/to/catalog \
  --plan /path/to/plan.yaml

python3 -m kernelgen.tools.kb import \
  --kb /path/to/catalog \
  --plan /path/to/plan.yaml \
  --publish
```

这条路径适合硬件手册、API 语义、编译器约束和已经人工整理好的方法。

### 13.2 运行中经过实验的提升

如果 Coder 真正读取并在 ExperimentPlan 中采用了某个 Source，Eval 得到结果后，Python 会把 Source application 与 Observation 连接起来。Distiller 可以据此提出同时引用 Source 和实测 round 的 Candidate。

只有单项应用产生的 correctness recovery 或 performance improvement 才适合作为该 Source 的正向 measured evidence。组合成功只能说明整组修改有效，不能自动把其中每个 Source 都提升为独立成功 Concept。回退、拒绝、无明显变化或无法分类的结果也可以形成负面经验或限制，但不能伪装成正向证明。

## 14. Publisher 怎样避免把 Catalog 写乱

Publisher 是正式 Catalog 的唯一写入者，主要机制都服务于一致性和可恢复性，不是为了阻止 Agent 做优化。

### 14.1 单写和幂等

多个 campaign 共用 Catalog 时，publish transaction lock 会把归档、索引、Concept 更新和 Git commit 串行化。`batch_id` 是幂等键；同一个 batch 重放会返回原结果，不会再写一份。

### 14.2 先 staging，再落盘

Publisher 先复制需要校验的 Catalog 内容到 staging，在 staging 中物化 Observation、合并 Concept 并验证引用。Observation 会先于 Concept 落盘，避免出现 Concept 指向不存在 Observation 的状态。

Concept 更新前会保存当前文件内容。如果更新或索引重建失败，Concept 会恢复并重建索引。完整运行事实仍在 Run Archive，不会因 Concept 写入失败而丢失。

### 14.3 单条失败隔离

Candidate 的 scope、round、Source、relation 或正文不合法时，只把该 Candidate 写进 `rejected`，其他合法 Candidate 继续发布。`PublishResult.status` 只有 `published` 和 `noop`；是否存在部分拒绝要看 `rejected` 列表，不存在单独的 `rejected` status。

### 14.4 Agent 不能覆盖权威字段

Runtime Candidate 只允许提供 `scope_hints.motifs` 这类可检索提示。Publisher 用 workspace 中的 OperatorSignature 和 TargetContext 绑定完整 exact scope，并从 Ledger 创建 Observation。Agent 写出的冲突 hardware、workload、numerics 或 measurement 不会成为权威事实。

## 15. 文件都在哪里

### 15.1 Agent workspace

```text
<run>/<epoch>/agentN/
├── .ledger.json
├── .best_kernel.py
├── .new_experience.md
├── .new_detailed.md
└── .kernelgen/knowledge/
    ├── state.json
    ├── context/
    │   ├── operator-signature.json
    │   └── target-context.json
    ├── retrieval-log.jsonl
    ├── candidates.jsonl
    └── publish-result.json
```

`retrieval-log.jsonl` 是当前统一的 durable event log，记录 query 和 detail read 的成功或失败。`query-log.jsonl` 只保留为旧 workspace 的兼容路径，新流程不再维护平行 query log。系统没有独立 `usage-log.jsonl`；实际使用以 Ledger round 中的 `knowledge_uses` 为准。

### 15.2 Catalog

```text
catalog/
├── sources/
│   ├── packages/
│   ├── content/
│   ├── manifests/
│   └── ingestion/
├── concepts/
├── observations/by_run/
├── solutions/
├── publish/audit/
├── schemas/
├── aliases.yaml
└── .derived/
    ├── knowledge.db
    └── rounds.db
```

`observations/by_run/<run>.jsonl` 的文件名只是存储分桶，不是 Observation 的完整身份。Observation ID 编码 run/workspace/round，记录中的 target context 和 evaluation scope 区分不同硬件、软件和 workload，因此同算子在不同芯片上的实验不会仅靠文件名混为一条。

`.derived/knowledge.db` 和 `.derived/rounds.db` 是可重建索引，不是事实来源。删除后可以从 Concept、Observation、Run Archive 和 retrieval events 重建。

### 15.3 Run Archive

```text
<catalog-parent>/run-archive/
└── <run>/<workspace>/snapshots/<snapshot-id>/
    ├── manifest.json
    ├── ledger.json
    ├── context/
    ├── knowledge/
    └── artifacts/
```

Snapshot ID 来自 Ledger 和知识日志内容。Archive 保存完整运行证据和大文件，默认不进入 Catalog Git。随着 round 增长，主要增加的是 Run Archive，而不是 Concept 文件数量。

## 16. Fresh、fork 和 resume 有什么区别

| 模式 | 从哪里开始 | 是否使用旧代码 | 是否能查询 KB |
|---|---|---|---|
| `fresh` | 新 workspace 和冷启动分析 | 否 | 能 |
| `fork` | 新 workspace | 使用精确 scope 的历史 best solution | 能 |
| `resume` | 原 workspace 的后续 epoch | 延续上一 epoch checkpoint 和 ledgers | 能 |

`fork` 适合“用过去 best code 作为新实验起点，但保留一条新的运行 lineage”。它要求 Catalog 中存在 definition、benchmark、backend、architecture、device 和 implementation language 匹配的 Solution。compiler/runtime/driver 版本当前不参与 Solution resolve，版本变化时需要人工决定是否仍适合作为起点。

`resume` 适合“原任务中断后继续”。它要求 `--start-epoch > 1`，并校验旧 Ledger 的 definition、hardware 和 implementation language。`--clean` 不能与 resume 同时使用。

`fresh` 只表示不继承旧代码，不表示忽略知识。Agent 仍可按需查询过去的 Concept、Source 和 round。

## 17. 可选字段什么时候填写

空字段不应为了“看起来完整”而输出。它们不是没用，而是只有在真实生产环节存在时才有意义。

| 字段 | 什么时候填写 | 当前生产者或限制 |
|---|---|---|
| `capabilities` | claim 明确依赖某硬件能力，且 Definition 要求、Eval service 确认 | Runtime exact scope 只写 required capability |
| `symptoms` | 失败、异常或诊断知识 | Distiller 填写；Python从失败 status 补充 |
| `sources` | 正文使用了真实 Source 事实 | 必须来自当前 workspace 成功读取的精确 Source ref |
| `relations` | 与一个已存在 Concept 有明确关系 | target 必须被当前 workspace 读取且 Catalog 中存在；当前查询不沿关系图扩展 |
| `stale_after` | 有明确复验周期 | 手工策略可写；过期后查询跳过，当前没有自动复验队列 |
| `verified` | 真正执行了审核 | 必须有 reviewer 和时间；Publisher 写文件不算审核 |
| `dataflow/layouts` | Definition 或分析能可靠确定 | Runtime identity 以 Definition 为准，不猜测 |
| `software` | Eval service 实际报告 | 稳定 Runtime 发布要求关键名称和版本完整 |
| `numerics` | Definition/Eval contract 给出要求 | 某轮测得误差属于 Observation，不是 scope tolerance |

Markdown writer 会省略空字符串、空列表、空字典和 `null`，但保留有意义的 `false` 和 `0`。省略表示“当前没有可靠值”，不表示这个字段永远无用。

Relation 支持 `requires`、`addresses`、`contradicts`、`supersedes`、`refines`、`broader_than`、`narrower_than` 和 `related`。当前系统只验证 target 存在并保存关系，还没有沿图召回或自动推导关系，因此不要为了填字段生成大量 `related`。

## 18. 失败后怎样恢复

### 18.1 Query 或 provenance 不完整

知识查询是建议性能力。查询失败、profile artifact 缺失或 provenance 不完整不应阻断 Eval。检查 `retrieval-log.jsonl`，修正下一次工具调用即可。

### 18.2 Distiller 失败

单条 Candidate 会独立修复和跳过。整个 Distiller contract 失败时，Ledger 和 best kernel 仍有效，只是没有新的报告和 Candidate。可以从完成的 Ledger 重新执行 finalize，而不必重跑 Coder。

### 18.3 Agent 已完成，但 epoch 发布或 synthesis 失败

使用 `--finalize-epoch <N>`。它要求该 epoch 的 Agent Ledger 数量与
`--n-parallel` 一致，只执行幂等 KB 发布和 epoch synthesis，不再启动 Analyzer
或 Coder；已有合法 synthesis checkpoint 时直接复用。

多个算子使用 `examples/kernel_gen/run_campaign.py`。它等待同一 epoch 的所有
活跃算子，再根据 ledger、`publish-result.json` 和 `synthesis.json` 判断哪些算子
可以进入下一 epoch；单个算子 incomplete 不阻塞其他算子继续。

### 18.4 索引损坏或删除

```bash
python3 -m kernelgen.tools.kb rebuild \
  --kb /path/to/catalog \
  --run-archive /path/to/run-archive
```

### 18.5 想撤销一次公共 KB 写入

先查看 batch audit 和 Catalog Git。只新增 Observation、没有改 Concept 的 batch 可以先 dry-run 再应用：

```bash
python3 -m kernelgen.tools.kb rollback \
  --kb /path/to/catalog \
  --batch-id <batch-id>

python3 -m kernelgen.tools.kb rollback \
  --kb /path/to/catalog \
  --batch-id <batch-id> \
  --apply
```

只要 batch 创建或更新了 Concept，就必须对 Catalog Git 执行 revert，再运行 `kb rebuild`。公共 Catalog 做批量清理或迁移前，应先建立 Git tag 或明确备份点。

## 19. 常用检查命令

验证 Catalog：

```bash
python3 -m kernelgen.tools.kb validate --kb /path/to/catalog
```

重建两个派生索引：

```bash
python3 -m kernelgen.tools.kb rebuild \
  --kb /path/to/catalog \
  --run-archive /path/to/run-archive
```

手动重放 workspace 发布：

```bash
python3 -m kernelgen.tools.kb publish \
  --kb /path/to/catalog \
  --run-archive /path/to/run-archive \
  --run-id <run-id> \
  --batch-id <batch-id> \
  <workspace>...
```

检查环境变化后哪些 Concept 需要复验：

```bash
python3 -m kernelgen.tools.kb revalidate \
  --kb /path/to/catalog \
  --target-context /path/to/target-context.json
```

`revalidate` 只生成候选列表，不删除、不改写 Concept，也不自动填写 `stale_after`。

## 20. 当前明确没有自动做什么

为了避免把未实现能力写成设计承诺，下面这些边界必须说清楚：

- 不把检索结果预注入所有 Agent prompt。
- 不把“读过”自动算成“使用过”。
- 不因为一个 Source 被频繁检索就自动提升为 Concept。
- 不把组合实验的提升自动归因到其中某一条知识。
- 不根据一个 exact 实验自动推广到整个芯片族或 portable scope。
- 不自动生成 relations、verified 或 stale policy。
- 不把 `.new_experience.md` 和 `.new_detailed.md` 当成正式 V1 Concept。
- 不为每次 Concept 更新新建一个 revision 文件。
- 不用 fingerprint 代替可读的 target、solution SHA、event ID 和 Archive manifest。
- 不把 `.derived` SQLite 当作不可替代的事实源。

当前仍存在的过渡设计也要诚实保留：legacy Markdown reducer 尚未移除；版本区间匹配不完整；relation 尚未参与图检索；campaign variant 尚未进入 metrics。这些项目以 [`knowledge_base_todo.md`](knowledge_base_todo.md) 为准，不能在本文中假装已经完成。

## 21. 代码从哪里看

| 文件 | 主要职责 |
|---|---|
| `knowledge/models/` | Concept、Observation、Candidate、Query、Source 和使用记录的持久模型 |
| `knowledge/contracts/` | workspace、查询、Source 和发布的运行时契约 |
| `knowledge/context.py` | 从 Definition 和 Eval service 构造并写入权威上下文 |
| `knowledge/scope.py` | direct、analogy、incompatible 判定 |
| `knowledge/query.py` | Concept 检索、排序和详情读取 |
| `knowledge/sources.py` | Source 搜索、读取和审计 |
| `knowledge/publishing/publisher.py` | 发布锁、staging、提交和失败恢复 |
| `knowledge/publishing/materialize.py` | Candidate 和 Ledger 转为 Observation/Evidence |
| `knowledge/publishing/merge.py` | Concept 查重、证据合并、状态和 supersede |
| `knowledge/publishing/lifecycle.py` | deprecate、delete、revalidate 和 rollback |
| `knowledge/publishing/static_import.py` | 审核过的 Source→Concept 导入 |
| `knowledge/catalog.py` | Catalog Markdown/JSONL 读写和 Git 提交 |
| `knowledge/index.py` | 可重建 Concept/usage 索引 |
| `knowledge/run_archive.py` | 不可变运行归档 |
| `knowledge/round_index.py` | 历史 round 索引 |
| `knowledge/metrics.py` | 使用漏斗和效果统计 |
| `knowledge/solutions.py` | exact-scope best kernel registry |
| `workflows/knowledge_bridge.py` | KernelGen workflow 与 KB 的业务接入点 |
| `mcp_server/knowledge_tools.py` | Agent-facing 知识查询、读取和审计工具 |
| `agents/distiller/__init__.py` | 运行报告、Candidate 生成和单条修复 |

目录按真实功能组织，不再用 `domain/application/adapters/ingestion` 通用分层。模块仍然多于简单的“读写文件”，因为系统确实承担 scope 匹配、旧格式兼容、检索审计、使用反馈、不可变归档、并发发布和 Git 回滚；这些行为不能通过合并成巨型文件来隐藏。

## 22. 修改 KB 后怎样验收

代码或 Schema 修改至少运行：

```bash
cd /data/akg_kernel_bench_lite/kernelgen

python3 -m kernelgen.scripts.dev.export_knowledge_schemas --check

python3 -m kernelgen.tools.kb validate --kb kb

python3 -m pytest -q \
  tests/test_knowledge_schema.py \
  tests/test_knowledge_*.py \
  tests/test_knowledge_golden_queries.py \
  tests/test_knowledge_distiller_reports.py
```

发布、检索或存储逻辑改变时还应检查：

- 每个 Observation 都能回到 Run Archive 的 Ledger；
- Candidate 无法覆盖 Python 权威 scope 和 measurement；
- 一条无效 Candidate 不影响其他 Candidate；
- 同一 batch 重放保持幂等；
- 并发 campaign 不会绕过 transaction lock；
- 删除 `.derived` 后索引可以重建；
- fresh、fork、resume 的代码起点和 lineage 正确；
- Concept 和 Source 的 scope 匹配保持正确，Solution 的 definition、benchmark、backend、architecture、device 和 language 匹配保持正确；若修改了 compiler/runtime/driver 语义，还要补充 Solution 版本兼容测试；
- 新写 Markdown 使用自然换行，不按固定字符宽度硬换行。

## 23. 最后再总结一次

KernelGen KB 的核心不是“让 Agent 写更多 Markdown”，而是把下面这件事做完整：

```text
知道资料从哪里来
→ 知道 Agent 实际用了什么
→ 知道它改了代码哪里
→ 知道真实评测发生了什么
→ 知道结论适用于什么范围
→ 知道下一次为什么推荐它
→ 出错时还能从 Ledger、Archive 和 Git 恢复
```

只要 Source、Concept、Observation、Solution 和 Run Archive 各自守住边界，知识库就能持续演进，同时不会把猜测、日志噪声和历史代码混成一团。

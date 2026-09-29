# KernelGen Concept Field Contract

本文是 KernelGen Knowledge Base V1 的字段级规范。它定义 Concept 和 Candidate 每个字段的语义、数据所有者、权威输入、生产规则、必填条件、空值含义、匹配行为和发布要求。

`knowledge/models/` 和 `kb/schemas/` 定义机器可验证的数据形状；本文定义这些字段应该表达什么。两者冲突时，必须先修正文档或实现，不能依赖宽松 Schema 把不完整知识发布为 `stable`。

知识使用反馈、Source 晋升和 Concept 长期演进尚未全部实现，具体优先级和验收条件见 [`knowledge_base_todo.md`](knowledge_base_todo.md)。

规范词：

- **MUST**：违反时必须拒绝发布或保留为 `draft`。
- **SHOULD**：一般必须满足；不满足时必须记录原因。
- **MAY**：按知识内容选择。

## 1. 核心模型

一条 Concept 由六部分组成：

```text
scope      什么时候允许使用
retrieval  什么查询容易召回
evidence   为什么相信它
relations  它与其他知识的关系
body       实际应该知道或执行什么
managed    如何保证完整性和可审计性
```

必须区分两类判断：

- `scope` 是适用性约束。Scope 不兼容时，即使文本相似也不能直接应用。
- `retrieval` 是召回路由。它影响能否搜到，不会扩大 `scope`。

## 2. 数据所有权

Agent 只能提出 Candidate。Python 和审核流程负责把 Candidate 变成正式 Concept。

| 数据 | 权威所有者 | Agent 是否可建议 | Publisher 行为 |
|---|---|---:|---|
| definition、operator identity | Definition/Python | 可补充 motif 标签 | Python 强制绑定 |
| target、software | Eval service/Python | 否 | Python 强制绑定 |
| workload、numerics | Definition、Ledger、Evaluation | 可提出更窄约束 | Python 验证，不能扩大 |
| measurement | Ledger/Observation | 否 | Python 从真实 round 生成 |
| title、summary、body | Agent/人工 | 是 | 质量校验后接受 |
| retrieval | 调用上下文 + Agent | 是 | Python 补全并验证 |
| relations | Agent/人工/后处理 | 是 | 验证目标和关系语义 |
| status | 发布策略/审核者 | 否 | 根据质量门决定 |
| verified | 审核者 | 否 | 通过审核后写入 |
| managed | Publisher | 否 | Python 生成 |

任何 Agent 提交的 target identity、版本号或测量数字都不能覆盖 Python 权威数据。

### 2.1 生产链路

正式 Concept 只能由以下六类输入产生：

```text
Eval service /status ───────────────> TargetContext ─────┐
Definition ─────────────────────────> OperatorSignature ─┤
Ledger + Eval + Profile ────────────> Observation ───────┤
Distiller ──────────────────────────> Candidate ─────────┤
Source ingestion / 查询返回的 ref ──> Source / Candidate ┤
明确的审核或维护策略 ─────────────────────────────────────┤
                                                          v
                                                      Publisher
                                                          |
                                                          v
                                                       Concept
```

每一项非空 metadata 都必须能回答：

1. 原始输入是什么；
2. 哪个组件做了转换；
3. 输入缺失或冲突时怎样处理。

“模型认为大概如此”不是权威输入。没有可靠来源的字段保持未声明；如果它是当前
kind 的 `stable` 必填字段，则 Concept 必须降为 `draft` 或拒绝。

### 2.2 逐字段生产表

| 字段 | 权威输入 | 生产者 | 生产规则 | 无法取得时 |
|---|---|---|---|---|
| `target.backend` | Eval `/status.target.backend`，兼容 `/status.backend` | `build_target_context()` | `npu/cann -> ascend` 等统一规范化 | Runtime Experience 降为 `draft` |
| `target.devices` | Eval `/status.target.device` | `build_target_context()` | 使用执行评测机器报告的设备型号；orchestrator 值只用于一致性校验，不作 fallback | 缺失时拒绝启动 Knowledge runtime |
| `target.architecture` | Eval `/status.target.architecture` | Eval service + Python | 使用后端采集或受控映射，例如 `910B* -> DAV_2201` | Runtime Experience 降为 `draft` |
| `target.software` | Eval `/status.software` | Eval service + Python | 读取实际安装的 compiler/runtime/driver/library 名称与版本 | 相关 claim 降为 `draft` |
| `target.capabilities` | Definition 的显式能力要求，或经过验证的后端分析 | Python/分析器 | 只记录应用 claim 所必需的能力 | 省略；不得复制 profiler 能力 |
| `operator.definition_ids` | Definition identity | `build_operator_signature()` | 绑定真实 definition name/ID | Runtime Candidate 拒绝 |
| `operator.op_types` | Definition `op_type` | `build_operator_signature()` | 规范化并与 Agent 标签合并 | 保留已知 identity，不能猜 |
| `operator.dtypes` | Definition inputs/outputs | `build_operator_signature()` | 规范化 `FLOAT32/torch.float32 -> float32` | 省略未知项 |
| `operator.motifs` | Definition 结构 + Agent 语义标签 | Python + Distiller | Python 值不可被 Agent 删除 | 无可靠依据的标签不写 |
| `operator.dataflow/layouts` | Definition、stride/layout 描述、验证后的分析 | Python/分析器 | 只写能够从输入或 artifact 证明的结构 | 省略 |
| `workloads` | Definition axes + 实际 Evaluation workloads | Python | 实测标量转 `eq`；多 workload 用 `in/any`；推广范围需多点 evidence | Runtime Experience 降为 `draft` |
| `numerics` | Definition/Eval correctness contract | Python | 写允许的精度、近似和累加约束，不写观测结果冒充阈值 | 数值 claim 降为 `draft` |
| measurement | Ledger round | Run fact reader | 原样冻结 status、latency、speedup、error 等事实 | Observation 拒绝 |
| `evidence` | `observation_intents` + 已验证 Observation | Publisher | 校验 round 后生成 ref；stance/rationale 来自 Candidate | Experience 拒绝 |
| `evidence_state` | evidence + sources | Publisher | 确定性计算 | 无依据时拒绝 |
| `retrieval.phases` | 调用阶段、round 状态、profile 使用情况 | Python + Distiller | Python 补事实阶段，Agent 补语义路由 | Runtime Concept 降为 `draft` |
| `retrieval.tasks` | Candidate 的预期使用场景 | Distiller | 从枚举中选择，不由错误码机械决定 | Runtime Concept 降为 `draft` |
| `retrieval.symptoms` | Evaluation status、profile findings | Python + Distiller | 错误码规范化；分析器可补经过验证的症状 | 失败知识降为 `draft` |
| `retrieval.techniques/keywords` | 实验计划、代码变化、claim | Distiller | 使用规范化、可区分的短标签 | 不满足 kind 质量门则 `draft` |
| `sources` | 已摄取 Source 或查询返回的真实 Source ref | Ingestion/Distiller + Publisher | Publisher 验证 ref 可解析 | 外部事实无 Source 时拒绝或收窄正文 |
| `relations` | 已检索到的 Concept ID | Distiller/人工 + Publisher | Agent 提议方向，Publisher 验证 target 和 scope 关系 | 省略，不得编造 |
| `status` | 本文质量门 | Publisher | 完整为 `stable`，可修复缺口为 `draft`，失效为 `deprecated` | 不得默认 `stable` |
| `verified` | 一次具体审核动作 | 审核器/人工 | 记录实际检查主体和时间；写文件不算审核 | 省略 |
| `stale_after` | 明确维护策略 | 审核者/维护任务 | 只在有版本更新或复验周期策略时设置 | 省略，不使用任意默认天数 |
| `managed` | 发布事务和规范化 Concept | Publisher | 生成写入者、时间和内容 hash | 发布失败 |

### 2.3 冲突与降级顺序

同一字段有多个候选来源时，优先级固定：

```text
执行机器的结构化事实
> Definition / Ledger / Evaluation
> 已验证的分析 artifact
> Agent 建议
```

具体规则：

1. Eval service 报告的实际 backend/device 是唯一执行身份；orchestrator 的显示名称
   只做归一化一致性校验，不提供 fallback。
2. Definition 的 operator identity、dtype 和静态 axes 优先于 Candidate。
3. Ledger 的测量值优先于正文或 rationale 中出现的数字。
4. Runtime Agent 不提交 scope；Static ingestion 的审核 scope 不能覆盖权威事实或把
   单点结果扩大成范围结论。
5. `/status`、Definition 或 Ledger 缺少关键字段时，Publisher 只能 `draft` 或拒绝；
   不能用当前机器的值回填历史运行。
6. 旧 archive 没有保存的环境版本不可事后恢复；必须保留未知或重新运行。

### 2.4 Retrieval 的事实映射

Python 可以确定性补充以下路由：

| 已验证事实 | 可补充 phase | 可补充 symptom |
|---|---|---|
| `COMPILE_ERROR`、`RUNTIME_ERROR`、`INCORRECT_NUMERICAL` | `post_error` | 对应规范化错误码 |
| 存在 PASSED evaluation 或 round 比较 | `post_evaluation` | 无 |
| claim 使用已记录 ProfileAnalysis | `post_profile` | 分析中的规范化 finding |
| 多轮达到已定义 plateau 条件 | `plateau` | 无 |

`tasks`、`techniques` 和 `keywords` 仍需要理解 claim 的用途，由 Distiller 提议并
由 Schema/质量门校验。Python 不应仅凭一个错误码猜测完整查询意图。

### 2.5 条件生产和人工生产字段

以下字段有价值，但只能在对应输入真实存在时产生：

- Concept `capabilities`：只能来自 Definition 的显式要求，并且 Eval
  `target.capabilities` 必须确认目标支持。`/status.profile.capabilities` 只描述
  profiler 可观测能力，不进入 Concept。
- `relations`：Distiller 只能使用当前 workspace 通过 `get_knowledge` 实际读取过的
  Concept ID；Publisher 还要确认 target 当前存在。
- `stale_after`：必须来自明确维护策略，不能机械设置 `created_at + 90 days`。
- `verified`：只有具体机器验证或人工审核实际发生后才能写。

字段有用不等于每条 Concept 都必须产生。缺少生产依据时省略，比填入相同默认值
或模型猜测更准确。

## 3. 空值语义

空值不能同时表示“不适用”“未知”和“未采集”。

| 表示 | 语义 |
|---|---|
| `""` | 未知或未采集，不是通配符 |
| `[]`（scope） | 未声明该维度约束 |
| `[]`（retrieval） | 没有该类召回路由，不表示所有情况 |
| `null`（numerics） | 不声明该数值属性 |
| `level: portable` | 明确声明不受硬件 target 限制 |

规则：

1. `stable` Concept 的核心适用字段不得用空值表达未知。
2. `draft` MAY 暂时包含未知字段，但必须说明缺口。
3. 空 software 字段不能被解释为“适用于所有软件版本”。
4. `portable` 必须显式使用 `level: portable`，不能靠空 target 模拟。
5. 数值正确性、编译器行为和平台实现差异相关的结论，software 版本未知时
   MUST NOT 发布为 `stable`。
6. Markdown 序列化 SHOULD 省略可选的 `""`、`[]`、`{}` 和 `null`；读取时由
   Schema 恢复默认值。
7. `false` 和 `0` 通常是有效值，MUST NOT 被通用空值压缩删除；唯一例外是
   `numerics.exact: false`，它不表达约束，应省略。`allow_tf32: false` 仍需保留。
8. 省略可选字段表示“未声明或未知”，不表示该维度已被证明无约束。
9. 空值压缩不能掩盖 `stable` 必填字段缺失；质量门在序列化之前执行。

## 4. Candidate 与 Concept

### 4.1 Candidate

Candidate 是发布建议，不是正式知识。

Runtime 与 Static 使用不同输入边界：

- `RuntimeCandidate` 不提交完整 `scope`，只可提交 `scope_hints.motifs`；Publisher 从
  Definition、TargetContext 和已测 workload 绑定 exact scope。
- Static `CandidateConcept` 来自审核后的 ingestion plan，继续提交完整 scope。
- 旧 runtime Candidate 的 scope 读取时只保留 `operator.motifs`，其余被 Python
  覆盖的字段不再传播。

| 字段 | 作用 | 所有者 |
|---|---|---|
| `candidate_id` | 本批次内稳定标识 | Python |
| `proposed_kind` | 建议的 Concept 类型 | Agent |
| `proposed_id` | 静态知识可建议固定 ID | Agent/摄取器 |
| `claim_key` | 声明身份和合并键 | Agent，Publisher 校验 |
| `title`、`summary`、`body` | 建议内容 | Agent |
| `domains` | 粗粒度领域标签 | Agent |
| `scope_hints.motifs` | Definition 外的可复用结构标签 | Runtime Agent，Python 合并 |
| `scope` | 完整适用范围 | 仅 Static ingestion；Runtime 由 Python 绑定 |
| `retrieval` | 召回标签建议 | Agent + Python |
| `relations` | 与已有 Concept 的关系建议 | Agent/人工 |
| `observation_intents` | 希望引用哪些 round 及理由 | Agent |
| `source_refs` | 希望引用哪些 Source | Agent/摄取器 |
| `created_by` | Candidate 来源 | Python |

Candidate 引用的 round 不存在、Python 无法权威绑定 scope 或正文不满足对应 kind 的契约时，
Publisher MUST 拒绝该 Candidate 或保留为 `draft`。

### 4.2 Concept

Concept 是正式可查询知识。除 Candidate 内容外，它增加：

- 稳定且唯一的 ID；
- 生命周期状态；
- 权威 scope；
- 真实 Source/Observation evidence；
- 创建者、审核和内容完整性信息。

## 5. 顶层字段

| 字段 | 语义 | 发布要求 |
|---|---|---|
| `schema_version` | 数据格式版本 | MUST 与当前 Schema 一致 |
| `id` | Concept 的稳定身份和检索引用 | MUST 唯一 |
| `kind` | `reference/method/diagnostic/experience` | MUST |
| `title` | 可扫描的结论标题 | MUST，不写空泛主题 |
| `summary` | 一句话说明条件、动作或结论 | MUST |
| `claim_key` | 同类声明的稳定合并键 | MUST |
| `domains` | 粗粒度领域 | MUST 至少一个 |
| `status` | `draft/stable/deprecated` | MUST |
| `verified` | 真正完成审核的记录 | 未审核时 MUST 为空 |
| `stale_after` | 强制重新验证日期 | 易漂移知识 SHOULD 设置 |
| `sources` | 外部资料依据 | 按 kind 要求 |
| `scope` | 硬适用约束 | MUST |
| `retrieval` | 软召回路由 | `stable` 时按 kind 完整 |
| `evidence_state` | 当前证据状态 | Python 计算 |
| `evidence` | Observation 与 claim 的联系 | 运行知识 MUST |
| `relations` | 与其他 Concept 的有向关系 | 有已知关系时 SHOULD |
| `managed` | hash 和维护时间 | Python 生成 |
| `body` | 结构化、可执行正文 | MUST 满足 kind 模板 |

### 5.1 `kind`

`kind` 是唯一的 Concept 类型字段。旧文件中的 `type` 只在读取时迁移为 `kind`；
新文件不得再次输出两套等价类型。

### 5.2 `claim_key`

`claim_key` 表示“这条知识在声明什么”，不是标题 slug。

要求：

- 使用稳定、小写、可读的领域词；
- 不包含 run、epoch、agent、日期等偶然身份；
- 同一 `kind + claim_key + scope` 的 Candidate 可合并 evidence；
- claim 语义或 scope 实质变化时不能强行复用。

正确：

```yaml
claim_key: gelu.reference_formula_matches_tanh_on_ascend
```

错误：

```yaml
claim_key: agent1_round5_good_result
```

### 5.3 `status`、`verified`、`stale_after`

生命周期：

```text
Candidate -> draft -> stable -> deprecated
```

| 状态 | 含义 | 是否参与默认查询 |
|---|---|---:|
| `draft` | 结构合法，但 scope、证据或语义审核未完成 | 否 |
| `stable` | 通过结构、证据、scope 和语义质量门 | 是 |
| `deprecated` | 被替代、失效或不再建议使用 | 否 |

`verified` 不是“Publisher 成功写文件”。它表示某个审核主体检查了 claim、scope、
evidence 和正文。单纯的 `kernelgen/publisher-v1` 写入动作 MUST NOT 自动视为
语义审核。

以下知识 SHOULD 设置 `stale_after`：

- 编译器 lowering、runtime 行为、driver 行为；
- 性能阈值和调优参数；
- 未被多个独立环境复现的经验；
- 依赖快速演进库版本的 API 行为。

## 6. `kind` 及其质量门

| kind | 回答的问题 | 最低依据 | `stable` 的额外要求 |
|---|---|---|---|
| `reference` | 稳定事实或约束是什么 | Source，或明确测量 | 来源可定位；区分事实与推断 |
| `method` | 应该怎样执行 | Source 或 Observation | Applies When、步骤、成功信号、失败模式 |
| `diagnostic` | 出现症状时怎样定位 | Source 或 Observation | 非空 symptoms/tasks；检查与解释闭环 |
| `experience` | 特定条件下实际发生了什么 | Observation | 精确 scope；非空 phases/tasks；限制与重验条件 |

单 workspace 产生的 Runtime Experience 默认 MUST 使用 `level: exact`。只有后续
汇总任务基于多个独立设备、版本或 workspace 的一致 Observation，才能把 scope
扩大为 `device`、`architecture` 或 `backend`。

## 7. `scope`

### 7.1 `scope.target`

#### `level`

| level | 含义 | MUST 提供 |
|---|---|---|
| `exact` | 仅当前环境组合 | backend、architecture、device、相关 software |
| `device` | 同一设备型号 | backend、device；architecture SHOULD |
| `architecture` | 同一架构 | backend、architecture |
| `backend` | 整个后端 | backend |
| `portable` | 不依赖硬件 target | 无硬件 identity；software 约束仍 MAY 存在 |

Publisher 只能保持或收窄 Candidate scope，MUST NOT 把单次实验从 `exact` 自动
扩大为 `device`。

Concept scope 不使用合成 `target_fingerprint`。`exact` 直接比较可读的 backend、
architecture、device 和已声明 software 字段。Run Archive 或派生索引内部 MAY
保留自己的 identity/hash，但它们不是 Concept metadata，也不参与本文字段契约。

#### `backend`

规范化执行后端，例如：

```text
ascend
cuda
cpu
```

`npu`、`cann` 等输入别名应由 Python 规范化为 `ascend`。

#### `architecture`

硬件架构或 ISA/核心架构，例如 `DAV_2201`。它不是 vendor，也不是产品型号。

#### `devices`

具体产品或设备型号，例如 `Ascend910B`。同一 Concept MAY 包含多个有明确 Source
依据或已独立验证的设备，但单次运行通常只能绑定一个。

`devices` 使用 OR 语义：当前设备命中列表中的任意一个即可。`capabilities` 使用
AND 语义：声明的能力必须全部存在。`device` 和 `architecture` 层级仍必须填写
`backend`；匹配时同时检查 backend，避免不同后端出现同名设备或架构时误用。

#### 多芯片 Concept

多芯片通过“共享 claim + 硬件 scope 变体”表达，不把任意平台条件堆进一篇正文：

1. 机制和动作对整个后端一致时，使用 `level: backend`。
2. 对同一架构一致时，使用 `level: architecture`。
3. 多个设备的机制、动作和限制完全一致时，使用 `level: device` 和多个
   `devices`。
4. 实现步骤、能力要求或正确性边界不同，必须拆成不同 scope 的 Concept；
   特化项 SHOULD 用 `refines` 指向通用项。
5. claim 相同但 scope 不同的变体 MAY 共用 `claim_key`；查询去重时优先保留
   更具体的可用项，顺序为
   `exact > device > architecture > backend > portable`。
6. 只有少量设备参数不同且不改变步骤或边界时，正文 MAY 在 Limits 下增加
   `### Device-specific Parameters`；否则必须拆分。
7. Source package 声明 `allowed_backends` 时，Static Publisher MUST 拒绝
   `portable` 或 backend 不匹配的 Candidate。

当前 schema 有意只允许一个 `architecture` 和一个 `backend`。真正跨架构但仍属于
同一 backend 的结论，应使用 backend scope；否则拆成关联变体，不增加联合 scope
表达式。

#### `capabilities`

表示“应用这条知识所必需的能力”，不是当前环境所有可用能力的复制。

正确示例：

```yaml
capabilities:
  - supports_bf16_cube
  - supports_block_pointer
```

错误示例：

```yaml
capabilities:
  - arithmetic_utilization
  - cache_counters
  - kernel_profile
  - source_hotspots
```

后一组是 Profiler 可观测能力，应保存在 TargetContext 或 Observation profile
上下文；除非 claim 明确依赖某项观测能力，否则不应复制进每条 Concept scope。

生产规则：

1. Definition 明确声明的能力要求可以由 Python 绑定。
2. 验证后的编译或 profile 分析可以提出能力要求，但必须能定位到 artifact。
3. Distiller 只能使用输入中已出现的规范化能力名，不能从模型记忆猜测。
4. 当前没有可靠依据时省略 `capabilities`；空值不影响使用 backend、architecture、
   device 和 software 做 scope 匹配。

#### `software`

| 字段 | 含义 | 示例 |
|---|---|---|
| `language` | 实现语言/DSL | `triton` |
| `language_version` | 语言规范版本；未知时留空 |  |
| `compiler` | 实际编译器实现 | `triton-ascend` |
| `compiler_version` | 编译器版本或约束 | `3.2.0` |
| `runtime` | 执行 runtime | `CANN` |
| `runtime_version` | runtime 版本或约束 | `8.5.0` |
| `library` | 直接影响语义的库 | `torch-npu` |
| `library_version` | 库版本或约束 | `>=2.6,<2.7` |
| `driver_version` | 驱动版本或约束 | `25.2.0` |

规则：

1. 名称必须来自统一词表，不能同一组件有多个拼写。
2. 版本必须来自运行环境，不从模型记忆猜测。
3. 若 claim 涉及 compiler/runtime/library 行为，对应名称和版本 MUST 非空。
4. 版本变化后是否仍适用未知时，使用 `exact` 或设置 `stale_after`，不能省略版本
   后扩大范围。

### 7.2 `scope.operator`

| 字段 | 语义 | 来源 |
|---|---|---|
| `definition_ids` | 精确 Definition | Python，Runtime Experience MUST 非空 |
| `op_types` | 算子大类 | Definition |
| `motifs` | 可复用计算结构 | Python + Agent 标签 |
| `dataflow` | 数据流结构 | Definition/分析器 |
| `dtypes` | 输入、输出或累加 dtype | Definition |
| `layouts` | 内存布局 | Definition |

Python MUST 绑定真实 `definition_id`。Agent 可以建议 motif 等检索标签；dataflow、
dtype、layout 和 op type 只能使用 Definition/验证后分析提供的权威值。

### 7.3 `scope.workloads`

Workload predicate：

```yaml
- field: num_elements
  op: gte
  value: 1048576
```

支持：

```text
eq ne lt lte gt gte in not_in
```

组合语义：

- `all`：所有条件都必须满足；
- `any`：至少一个条件满足；为空时不增加条件；
- `excludes`：任意条件满足即排除。

规则：

1. 字段名必须来自 Definition axes 或稳定派生特征词表。
2. `in/not_in` 的 value 必须是非空列表；其他操作符使用标量。
3. 单次实验只能声明已测 workload 或其更窄子集。
4. 从一个 shape 推广到范围条件（如 `gte`）必须有多个边界 Observation 或外部
   Source 支持。
5. `[]` 表示未声明该维度，不代表已证明所有 workload 都适用。

### 7.4 `scope.numerics`

| 字段 | 语义 |
|---|---|
| `exact` | 是否要求精确语义/位级要求 |
| `allow_tf32` | 是否允许 TF32 |
| `accumulation_dtypes` | 允许的累加 dtype |
| `max_abs_error` | 最大绝对误差 |
| `max_rel_error` | 最大相对误差 |

`exact` 只有在 claim 要求精确语义或位级一致时才写 `true`。不要求 exact 时省略该
字段；若 claim 依赖某种近似模式，必须写出对应的 `allow_tf32`、误差阈值或其他明确
约束，不能用 `exact: false` 表达“宽松”。

数值正确性 Concept MUST 从 Definition/Evaluation 写入适用的容差、近似模式和累加
语义。未知时只能是 `draft`，不能把 `null` 当成无限容差。

必须区分：

- `scope.numerics` 是 Definition/Eval contract 规定的允许误差和语义约束；
- Observation 中的 error 是某次 round 实际测得的结果。

例如某轮观测到 `max_abs_error=4.7e-7`，不能据此把
`scope.numerics.max_abs_error` 写成 `4.7e-7`。只有 benchmark correctness
contract 明确使用该阈值时才能写入 scope。

## 8. `retrieval`

### 8.1 字段

| 字段 | 作用 | 示例 |
|---|---|---|
| `phases` | 优化生命周期阶段 | `post_error`, `post_profile` |
| `tasks` | 当前查询意图 | `diagnosis`, `implementation` |
| `symptoms` | 可观察问题 | `INCORRECT_NUMERICAL`, `UB_OVERFLOW` |
| `techniques` | 方法或变换 | `autotune`, `double_buffering` |
| `keywords` | 补充全文检索词 | `gelu`, `tanh`, `erf` |

合法 phase：

```text
initial
post_error
post_evaluation
post_profile
plateau
```

合法 task：

```text
constraint_check
architecture_selection
implementation
diagnosis
next_experiment
portability_analysis
```

### 8.2 必填规则

1. `stable` Runtime Experience MUST 有至少一个 phase 和一个 task。
2. 由失败产生的 Experience/Diagnostic MUST 有 symptom。
3. Method SHOULD 有 technique。
4. keywords SHOULD 包含 3～12 个有区分度的词，不复制整段标题。
5. phase/task 必须来自枚举，不能互相串列。
6. Python SHOULD 从运行上下文补充 phase/task，从 Evaluation status 补充 symptom；
   Agent 只补充语义标签。

空 retrieval 不表示“任何时候都适用”，只表示缺少结构化召回路由。此类 Runtime
Concept MUST 保留为 `draft`。

## 9. `sources`、`evidence` 和 `evidence_state`

### 9.1 `sources`

SourceReference 指向外部资料或已摄取知识源：

| 字段 | 作用 |
|---|---|
| `resource` | Source ID、URL 或资源定位符 |
| `revision` | 实际读取的 SourcePackage revision；URL 来源可省略 |
| `locator` | revision 内的精确文件、heading 或行范围 |
| `title` | 可选的可读标题 |

旧文件中的 `id` 会在读取时迁移为 `locator`；`author`、`usage_count` 和
`last_modified` 只为兼容旧数据读取，不再发布。来源作者和更新时间属于
SourcePackage，使用统计从 Observation 派生，不能复制进 Concept。

Reference 若没有测量 evidence，MUST 至少有一个 Source。

Runtime Experience 可以只由 Observation 支持。正文若出现外部 API、框架默认
行为、硬件规格或编译器通用语义，则必须：

1. 引用查询/摄取流程返回的真实 Source ref；或
2. 把正文收窄为“在本次 benchmark 和记录环境中观察到”，不作外部事实声明。

Publisher 必须验证 Source ref 可解析，不能接受 Agent 手写但不存在的资源 ID。

SourcePackage 必须声明 `search_mode`：

| 值 | 语义 |
|---|---|
| `searchable` | 本地快照可全文搜索；必须有 `kb://` content root 和 manifest root |
| `static_only` | 只提供来源身份或支持已提炼 Concept，不开放原文搜索 |

显式查询 `static_only` package 必须报错，不能把缺少 manifest 静默解释为零命中。
旧 package 缺字段时按 manifest 是否存在只读推断；新写入必须显式声明。

### 9.2 `evidence`

ConceptEvidence 说明某个不可变 Observation 为什么支持当前 claim：

| 字段 | 作用 |
|---|---|
| `observation_ref` | 指向真实 run/workspace/round |
| `stance` | `supports/refutes/illustrates` |
| `rationale` | 该 round 与 claim 的逻辑联系 |
| `confidence` | `high/medium/low` |

`rationale` 必须解释“为什么这个 round 能支持或反驳该 claim”，不能只重复测量值。

### 9.3 `evidence_state`

由 Python 根据 evidence 和 sources 计算：

| 状态 | 含义 |
|---|---|
| `source_supported` | 只有 Source 支持 |
| `observed` | 一个独立 workspace 的证据 |
| `corroborated` | 多个独立 workspace/run 支持 |
| `contested` | 同时存在支持与反驳 |
| `falsified` | 当前只有反驳 |

`contested` 和 `falsified` 是证据结论，不等同于 Schema 错误。默认检索可以返回它们，
但正文必须清楚说明不确定性或失败结论。

## 10. `relations`

Relations 是 Concept 之间的有向语义边。它不替代 scope，也不替代 evidence。

```yaml
relations:
  - type: refines
    target: kg:diagnostic:gelu-numerical-mismatch
```

`target` 必须是已存在、无 `@revision` 后缀的 Concept ID。

| type | `A -> B` 的精确定义 | 对 B 的生命周期影响 |
|---|---|---|
| `requires` | 应用 A 前必须满足或理解 B | 无 |
| `addresses` | A 提供解决 B 所描述问题的方法或证据 | 无 |
| `contradicts` | A 和 B 在重叠 scope 内给出不兼容结论 | B 不自动失效 |
| `supersedes` | A 是 B 的明确替代版本 | B SHOULD 同事务 deprecated |
| `refines` | A 在不否定 B 的前提下增加精度或条件 | 无 |
| `broader_than` | A 的有效 scope 严格包含 B | 无 |
| `narrower_than` | A 的有效 scope 严格包含于 B | 无 |
| `related` | 有关联，但以上关系均不准确 | 无 |

规则：

1. MUST NOT 指向自身。
2. target MUST 存在，否则 Catalog validation 失败。
3. `contradicts` 只有在 scope 有重叠时才成立；无重叠只是不同条件下的经验。
4. `supersedes` 需要说明替代原因，并同步处理旧 Concept 状态。
5. `refines` 不表示旧知识错误。
6. `broader_than` 和 `narrower_than` 必须能由 scope 比较或多环境证据支持。
7. `related` 是最后选择，不能用于逃避更精确关系。
8. Relation 不会自动扩大适用范围。

当前实现只存储、校验和比较 relations；`query_knowledge` 尚不会沿关系图扩展召回。
在实现关系图检索前，relations 的主要作用是维护、冲突表达和演进审计。

关系的生产顺序固定：

```text
query_knowledge 返回已有 Concept ref
-> Distiller/人工判断有向关系
-> Candidate 提交 relation
-> Publisher 验证 target、方向和 scope
-> Concept 写入
```

Distiller 没有收到真实 Concept ID、或没有精确关系时，`relations` 应省略。

## 11. `body`

正文必须让另一个 Agent 在不阅读原始日志的情况下知道：

1. 什么时候适用；
2. 观察或依据是什么；
3. 应采取什么行动；
4. 哪些情况不能泛化；
5. 什么时候必须重新验证。

仅设置字符最小长度不能保证质量。`stable` Concept 必须满足对应结构。

### 11.1 Reference

```markdown
## Fact
## Scope and Constraints
## Source Interpretation
## Revalidate When
```

### 11.2 Method

```markdown
## Applies When
## Procedure
## Success Signal
## Failure Modes
## Revalidate When
```

### 11.3 Diagnostic

```markdown
## Trigger
## Checks
## Interpretation
## Next Action
## Limits
```

### 11.4 Experience

```markdown
## Applies When
## Observation
## Action
## Constraints and Non-goals
## Revalidate When
```

每个必需章节必须包含实际内容。正文中的测量值必须能回到 evidence 引用的
Observation；外部 API 或硬件事实必须能回到 Source。

## 12. `managed`

| 字段 | 作用 |
|---|---|
| `content_hash` | 规范化内容 hash，检测变化 |
| `created_by` | 首次创建该 Concept 的 Publisher |
| `created_at` | 第一版创建时间 |
| `updated_at` | 当前内容更新时间 |

`managed` 由 Publisher 生成。人工修改正文或 metadata 后，必须通过正式发布流程
更新 hash，不能只手改 Markdown。每个 Concept ID 只有一个
`concepts/<kind>--<slug>.md` 当前文件；内容历史和回滚由 Catalog Git 记录，
不在目录中保存 `@N` 副本。SourcePackage revision 不受此规则影响。
旧 Ledger/Observation 的 `kg:...@N` 引用只读时归一为稳定 ID，新记录不得输出
该后缀。

## 13. 发布质量门

### 13.1 所有 `stable` Concept

- Schema 合法；
- claim_key 和 ID 合法且无冲突；
- scope 足以支持正文中的推广范围；
- evidence/source 可解析；
- retrieval 满足对应 kind；
- body 满足对应结构；
- relations target 存在；
- managed 由 Python 生成；
- 若写入 verified，它必须对应一次真实质量审核，不能由 Publisher 冒充。

### 13.2 Runtime Experience

额外 MUST：

- 至少一个 Observation；
- 单 workspace 默认 `level: exact`；
- definition ID 非空；
- backend、architecture、device 非空；
- 与 claim 相关的软件名称和版本非空；
- phases/tasks 非空；
- 失败经验的 symptoms 非空；
- 不把单次 Observation 写成跨版本或跨设备事实；
- 正文包含限制和重新验证条件。

### 13.3 不满足质量门

按顺序选择：

1. Python 可从权威上下文确定性补全时，补全；
2. 仍缺字段但结构合法时，发布为 `draft`；
3. evidence、identity 或正文逻辑无效时，拒绝 Candidate；
4. MUST NOT 用空字符串补齐后发布为 `stable`。

## 14. 完整示例：GELU 数值经验

以下示例故意把结论限定为当前评测环境，不声明通用 PyTorch 默认行为。

```yaml
---
schema_version: "1.0"
id: kg:experience:gelu-reference-matches-tanh-in-ascend-eval
kind: experience
title: Tanh GELU matches the abl_t1_gelu reference in this Ascend environment
summary: >-
  For abl_t1_gelu on the recorded Ascend910B software stack, erf-based
  implementations failed numerical evaluation and the tanh formulation passed.
claim_key: gelu.reference_formula_matches_tanh_on_ascend
domains: [gelu, numerical_correctness, ascend]
status: stable
verified:
  - by: kernelgen/knowledge-review
    at: "2026-07-30T00:10:00Z"
stale_after: "2026-10-30"
scope:
  target:
    level: exact
    backend: ascend
    architecture: DAV_2201
    devices: [Ascend910B]
    software:
      language: triton
      compiler: triton-ascend
      compiler_version: 3.2.0
      runtime: CANN
      runtime_version: 8.5.0
      driver_version: 25.2.0
  operator:
    definition_ids: [abl_t1_gelu]
    op_types: [elementwise]
    motifs: [gelu]
    dtypes: [float32]
  workloads:
    all:
      - field: num_elements
        op: eq
        value: 16777216
retrieval:
  phases: [post_error, post_evaluation]
  tasks: [diagnosis, implementation]
  symptoms: [INCORRECT_NUMERICAL]
  techniques: [gelu_tanh_approximation]
  keywords: [gelu, tanh, erf, ascend910b, numerical]
evidence_state: observed
evidence:
  - observation_ref: kg:observation:abl_t1_gelu:1R.agent1:round-5
    stance: supports
    rationale: >-
      Earlier erf-based rounds failed numerical evaluation; after changing the
      formula to tanh, R5 passed in the same recorded environment.
    confidence: high
managed:
  content_hash: "sha256:<publisher-generated>"
  created_by: kernelgen/publisher-v1
  created_at: "2026-07-30T00:00:00Z"
  updated_at: "2026-07-30T00:00:00Z"
---

## Applies When

- Definition is `abl_t1_gelu`.
- Target and declared software fields match the recorded environment.
- Input dtype and workload match the recorded scope.

## Observation

Erf-based implementations failed numerical evaluation in the preceding rounds.
R5 changed the implementation to the tanh formulation and passed under the same
evaluation service and target.

## Action

When this exact benchmark reports `INCORRECT_NUMERICAL` for an erf-based GELU,
test the tanh formulation while keeping launch and workload conditions fixed.

## Constraints and Non-goals

This evidence does not establish the default semantics of PyTorch on other
backends or software versions. It does not establish correctness for other
dtypes, shapes, devices, compiler versions, or tolerance policies.

## Revalidate When

Re-run the comparison after any change to the benchmark reference, PyTorch or
torch-npu library, Triton compiler, CANN runtime, driver, dtype, shape, or
numerical tolerance.
```

## 15. 当前实施状态

| 领域 | 当前实现 | 尚未实现/边界 |
|---|---|---|
| 环境采集 | Eval `/status` 提供 target 和 software；信息不完整时拒绝 runtime stable 发布 | 每种新平台仍需实现自己的采集器 |
| Concept/round/evaluation identity | target exact 比较 backend/device/architecture/software；evaluation/profile 复用 solution SHA；Archive manifest 保存文件 lineage | 旧 fingerprint 只读兼容，不再写入 |
| target level | 单 workspace Runtime Concept 绑定为 exact | 跨机器证据需单独归纳 |
| 多芯片适用性 | device/architecture 匹配同时检查 backend；查询优先具体 scope；Static Source 受 allowed_backends 门禁 | 跨 backend 结论仍需独立 Source 和关联变体 |
| capabilities | 只绑定 Definition 明确要求且 target 报告支持的能力 | 验证后分析自动提升为能力约束尚未实现 |
| workload/numerics | 静态 axes 绑定为 `eq`；numerics 只来自显式 Definition contract | 多 workload 范围归纳暂不自动做 |
| 空字段 | Markdown 省略 `""`、`[]`、`null`，保留 `false/0` | 模型读取时仍恢复默认空值 |
| retrieval/symptoms | Runtime Candidate 必须有 phase/task；失败 Observation 状态自动进入 symptoms | 自由文本错误根因不做脆弱正则猜测 |
| body | Runtime Candidate 固定 Claim/Evidence/Applicability/Action/Limits；正文轮次必须等于 intents | 各 kind 的更细模板可后续增加 |
| status | 通过门禁的 runtime knowledge 创建为 stable；信息不完整直接拒绝 | 自动保存 draft 候选尚未实现 |
| verified/stale | Publisher 不再伪造 verified；过期 Concept 不参与检索 | 真实审核和维护策略入口尚未实现 |
| Sources/relations | 只接受本 workspace 实际读取过的 ref/Concept ID，并在发布时验证 Catalog | query 尚不沿关系图扩展召回 |

## 16. Review Checklist

发布或人工审核前逐项回答：

### Identity

- claim_key 是否描述稳定声明，而不是某次 run？
- title/summary 是否同时包含条件和结论？

### Scope

- 推广范围是否不超过 evidence？
- exact scope 是否有 backend、architecture、device 和相关 software？
- 相关 hardware/software 版本是否完整？
- workload 和 numerics 是否来自权威输入？

### Retrieval

- 这条知识会在什么 phase/task 被使用？
- 失败知识是否有 symptom？
- 方法是否有 technique？

### Evidence

- 每个测量结论是否能回到 Observation？
- 每个外部事实是否能回到 Source？
- rationale 是否解释逻辑关系，而非只复制数值？

### Relations

- 是否存在更一般、更新、冲突或被替代的 Concept？
- relation target 是否存在且方向正确？

### Body

- 是否写明 Applies When？
- 是否有可执行 Action？
- 是否明确 Constraints and Non-goals？
- 是否说明 Revalidate When？

### Lifecycle

- 是否真的达到 stable？
- verified 是否来自真实审核？
- 是否需要 stale_after？

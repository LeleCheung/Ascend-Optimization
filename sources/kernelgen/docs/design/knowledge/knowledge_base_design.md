# KernelGen 知识库设计

状态：Proposed。本文同时描述长期目标架构和可维护的 Core V1；标记为 Later 的能力不属于首版实现范围。

适用范围：KernelGen 多 Agent 算子生成与优化流程

目标后端：CUDA、Ascend，以及未来新增的其他加速器后端

> **当前实现边界（2026-07-23）**
>
> 本文的大部分 Concept、Ontology、TargetContext、索引和
> `query_knowledge` 内容仍是演进设计，不能作为当前文件格式或 API 使用。
> 当前运行中的稳定基线仍是 legacy Markdown KB：
>
> ```text
> agent Distiller
>   → .new_experience.md + .new_detailed.md（候选）
>   → epoch knowledge reducer（唯一 canonical writer）
>   → kb/experience/by_definition/{op_type}/{definition}/{target}/
>        experience.md + detailed.md
> ```
>
> reducer 以权威 ledger 校验和排序候选，同时归并两份文档，执行大小限制和
> 原子写入；只有 run-level `kb/` 保存 Git 历史。agent 与 synthesis workspace
> 只获得不含 `.git` 的内容快照。以下章节出现“未来”“应”“Core V1”时，均指
> 迁移目标，不表示这些能力已经落地。

## 1. 背景

KernelGen 当前已经具备完整的运行证据链：Coder 先通过服务端 preflight，
再在评测前提交 `ExperimentPlan`；Python 写入 immutable solution 与完整
per-workload evaluation；ProfileAnalyzer 通过后端中立的 `ProfileAnalysis`
记录 profile 证据；Coder 最后提交 `RoundConclusion`。Distiller 随后从
trajectory 生成候选，epoch reducer 再统一调用 MergeAgent 合并经验。

当前知识库仍采用以文件路径为主的组织方式：

```text
kb/experience/by_definition/{op_type}/{definition}/{target_hardware}/experience.md
kb/experience/by_definition/{op_type}/{definition}/{target_hardware}/detailed.md
kb/experience/by_op_type/{op_type}/{target_hardware}/insights.md
```

`query-kb` 主要通过 Agent 手工执行 Glob、Read 和 Grep 查找固定目录，路径和内容明显偏向 Triton CUDA、NCU 与 SASS。该方式适合早期小规模知识库，但无法稳定支持以下场景：

- 新算子尚无相同 definition、相同 `op_type` 或相同算子名称。
- 同一算子需要运行在 H100、H200、B200、Ascend 910B4 等不同芯片上。
- 相同方法具有通用原理、后端实现和芯片观测三个不同层级。
- 相同知识在不同 workload、dtype、layout、编译器版本下结论不同。
- 多个并行 Agent 同时产生相容、冲突或部分重叠的经验。
- 知识库持续增长后，Agent 无法靠浏览目录精准定位有限且相关的内容。

本设计将知识库从“按目录积累 Markdown”升级为“开放世界的结构化 Concept 集合”，同时保留 Markdown 的可读性和 Git 的可维护性。

## 2. 设计目标

### 2.1 功能目标

- 新算子无需预先注册算子家族，也能通过计算结构、数据流、workload 和目标平台检索相关知识。
- 同一知识可以表达通用、后端级、架构级、设备级和精确软件栈级适用范围。
- CUDA 与 Ascend 使用同一 Concept、Query 和 Evidence 数据模型，后端差异通过 scope、capability 和 backend detail 表达。
- 所有经验性结论能够追溯到权威 ledger round、solution hash、evaluation fingerprint、profile artifact 或外部来源。
- 查询结果能够解释为什么匹配、哪些条件不匹配、应当直接采用还是只作为待验证假设。
- 未来的 Distiller 和 Concept-aware Merge 能够增量写入小粒度知识，而不是
  不断重写无法精准检索的大文档；当前 Markdown 基线由 epoch reducer 做
  有界归并。
- Ontology 可以演进，但新增算子或新硬件不会导致旧 Concept 和旧查询失效。

### 2.2 非目标

- 知识库不替代 `.ledger.json`、源码 snapshot、NCU report、msprof report、SASS 或后端指令报告。
- 第一版不要求图数据库、向量数据库或在线外部服务。
- 第一版不要求完整 computation DAG、通用子图同构或数学表达式规范化。
- 第一版不自动把一次运行结果泛化为跨芯片或跨后端事实。
- Skill 不承担知识正文；Skill 只描述何时查询、怎样使用结果和怎样提交候选知识。
- Ontology 不试图预先穷举未来所有算子、编程语言、编译器和硬件能力。
- 普通新增算子、芯片或文档不应要求修改公共 Schema。

### 2.3 可维护性目标

- 新增普通算子不修改 Schema，也不强制新增 ontology term。
- 新增芯片型号只增加规范化 target registry/alias 和 service capability，不复制一套知识目录。
- 新增文档仓库只增加一个 SourcePackage manifest，文件级 locator 由工具生成。
- 新增运行经验由 Distiller 生成 candidate，target、solution、evaluation 和 profile evidence 由工具权威回填。
- 作者只维护 claim、适用条件和必要关系；revision、时间戳、反向边、checksum 和索引由工具维护。
- 删除全部派生索引后能够从 Markdown/YAML、ontology 和 source manifests 完整重建。

### 2.4 规范化交付物

本文解释设计原则和目标边界。实现时还必须提供机器可验证规范：

```text
kb/SPEC.md
kb/schemas/concept.schema.json
kb/schemas/operator_signature.schema.json
kb/schemas/target_context.schema.json
kb/schemas/query_context.schema.json
kb/schemas/knowledge_bundle.schema.json
kb/schemas/source_package.schema.json
kb/ontology/*.yaml
kb/retrieval-tests/golden/*
kernelgen/knowledge/*.py
```

`SPEC.md` 使用 MUST/SHOULD/MAY 描述字段语义和所有权；JSON Schema/Pydantic 约束结构；golden queries 约束查询行为。仅有本文而没有 Schema 和回归测试，不视为完成规范实现。

## 3. 核心设计决策

1. `kind` 是小而封闭的知识类型集合：`reference`、`method`、`diagnostic`、`experience`。
2. `domains` 是受控但可扩展的主题标签，初始包含 `operator`、`hardware`、`language`、`compiler`、`runtime`、`profiler`、`optimization`、`numerics`。
3. `kind` 和 `domains` 只负责分类；适用性由 `scope` 表达，可信度由 `evidence` 和 lifecycle 表达，检索入口由 `retrieval` 表达。
4. 算子分类采用开放世界模型。算子家族是可选候选，不是检索前置条件；未知原语使用 namespaced extension 保留，不能压缩成 `other`。
5. 新算子的主表示是组合式 `OperatorSignature`。Core V1 使用语义候选、计算 motif、结构 path/features、数据流和 workload 特征；完整 computation DAG 仅作为 Later 能力。
6. 设备名称不能继续作为自由字符串路径身份。TargetContext 由 eval/profile service 规范化并返回。
7. 原始运行证据保留在 workspace/artifact store；公共 KB 只保存经过蒸馏和合并的 Concept，并以 locator 引用原始证据。
8. 查询采用“结构化过滤与召回为主，文本检索为补充”的混合模型；第一版不依赖 embedding。
9. Agent 不再通过猜路径完成主要检索。确定性的 Python query engine 负责索引、过滤、ontology 扩展、排序和结果解释。
10. `query-kb` 保留为稳定 Skill，但内部调用 `query_knowledge` MCP/CLI，而不是承载知识分类和路径规则。
11. 自动写回只能产生 candidate Concept；公共 Concept 和公共 ontology 的更新必须通过结构校验、证据校验和 Merge 决策。
12. 跨目标泛化必须由多目标证据支持。其他芯片上的经验默认是 analogy，不是当前目标的已验证事实。
13. Frontmatter 区分作者字段和工具管理字段；Agent 不手工生成权威 revision、target fingerprint、solution hash 或 evaluation fingerprint。
14. Workload 适用条件使用受控的结构化表达式，不保存或执行任意 predicate 字符串。
15. SourcePackage 是外部资料的默认管理单位；只有需要独立版本和生命周期的单个文档才提升为独立 SourceDocument。
16. 图关系是逻辑模型，Markdown/YAML 是权威存储，SQLite/JSON 是可重建查询投影；Core V1 不引入外部图数据库。

### 3.1 Core V1

Core V1 只实现能够通过真实检索回归验证价值的最小闭环：

```text
Concept
OperatorSignature-lite
TargetContext
SourcePackage
结构化 scope
Evidence locator
少量强 relation
SQLite/FTS 或等价本地索引
确定性 query_knowledge
Candidate → Merge → Public
Golden retrieval tests
```

### 3.2 Later

以下能力保留在数据模型的演进空间中，但不进入 Core V1：

```text
完整 computation DAG
复杂子图匹配
Embedding recall/reranker
Learned ranker
自动 scope promotion
自动 ontology promotion
外部图数据库
跨仓库实时共享服务
自动知识质量评分
```

增加 Later 能力前必须用 golden query、false-transfer rate、查询延迟或维护成本证明其必要性。

## 4. 总体架构

```text
Definition + Workloads                 Eval/Profile Service
          │                                     │
          ▼                                     ▼
OperatorSignatureBuilder              Canonical TargetContext
  ├── deterministic extraction                  │
  └── Analyzer classification                   │
          └──────────────────┬──────────────────┘
                             ▼
                         QueryContext
                             │
                             ▼
                     Knowledge Query Engine
                  ┌──────────┼──────────┐
                  │          │          │
             scope filter  ontology   multi-route
                           expansion    recall
                  └──────────┼──────────┘
                             ▼
                    rank + diversify + explain
                             │
                             ▼
                       KnowledgeBundle
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
           Analyzer         Coder      ProfileAnalyzer
                                            │
                                            ▼
ledger + solution + evaluation + profile + conclusion
                             │
                             ▼
                         Distiller
                             │
                             ▼
                     Candidate Concepts
                             │
                             ▼
                          Merge
                             │
                             ▼
                       Public Concepts
```

## 5. 数据分层

### 5.1 运行事实层

运行事实层由 Python 和 service 权威写入，不属于公共知识正文：

- `.ledger.json`
- immutable solution snapshot
- evaluation workload measurements
- evaluation fingerprint
- profile analysis
- NCU/msprof 原始报告
- SASS 或后端 instruction listing
- 编译器输出
- eval/profile service capability

运行事实是蒸馏知识的证据来源，不能直接作为公共 Concept 的自然语言结论。

### 5.2 外部来源层

外部来源层保存不可由 KernelGen 运行产生的权威资料和参考代码：

- 厂商硬件与 profiler 文档
- Triton、Triton-Ascend、CUDA、CANN 文档
- 编译器源码
- Tutorials
- Reference kernels
- 第三方项目

大体积源码和文档无需全部转换为 Concept。Reference Concept 可以指向这些来源中的具体文件、符号、版本和段落。

外部来源默认以 SourcePackage 管理，而不是要求每个文件都维护完整 metadata：

```yaml
schema_version: "1.0"
id: kg:source-package:triton-ascend
source_type: upstream_repository
authority: primary
origin:
  repository: https://github.com/triton-lang/triton-ascend
  revision: abc123
content_root: sources/code/triton-ascend
domains:
  - compiler
  - language
  - profiler
scope:
  targets:
    backends:
      - ascend
lifecycle:
  status: active
```

Concept 对具体内容使用稳定 locator：

```yaml
source:
  package: kg:source-package:triton-ascend
  path: docs/en/debug_guide/profiling.md
  locator:
    kind: heading
    value: Profiling
```

源码可以使用 `symbol`、`line` 或 commit-stable anchor。只有需要独立版本、授权或生命周期管理的大型文档才建立单独 SourceDocument；普通仓库文件不建立一份 manifest。

SourcePackage 与 Concept 是两层数据：

- SourcePackage 保留原始资料、来源、revision、authority 和 checksum。
- Concept 保存可复用的单一事实、方法、诊断或经验。
- 一份 SourcePackage 可以支持多个 Concepts。
- 尚未蒸馏的 Source 可以作为 `raw_source_reference` 回退结果，但不能被当成已验证的行动建议。

### 5.3 候选知识层

Distiller 从一个已完成 trajectory 中生成 candidate Concept。候选知识必须保留：

- 原始 claim
- 适用条件
- 证据 locator
- 被验证、部分验证、证伪或不可评估的假设
- 与现有 Concept 的潜在关系
- 不确定性

Candidate 不直接参与默认公共查询，除非查询显式请求 candidate。

### 5.4 公共知识层

通过 Merge 的 Concept 进入公共知识层。公共 Concept 应当小粒度、可单独寻址、可独立验证，并且不能依赖读者先读一整段运行历史。

## 6. Knowledge Concept 模型

### 6.1 核心字段

每个 Concept 由 Markdown 文件承载，YAML frontmatter 用于索引与过滤，正文用于 Agent 阅读。

```yaml
---
schema_version: "1.0"
id: kg:concept:experience:softmax-hopper-large-row
kind: experience
subtype: optimization_result
title: Large contiguous row reductions benefit from a separate Hopper configuration
summary: A concise retrieval-oriented summary.
domains:
  - operator
  - optimization
  - hardware

scope: {}
retrieval: {}
lifecycle: {}
evidence: []
relations: {}
---
```

作者或 Distiller 负责 `kind`、`title`、`summary`、claim、scope、retrieval cue 和必要的正向 relations，并可以提出 `proposed_id`。Candidate writer 负责分配或确认最终稳定 `id`。以下字段由管理工具或权威 service 生成并校验，不能由 Agent 自由声明：

```text
revision
created_at
updated_at
content_hash
target fingerprint
solution hash
evaluation fingerprint
source checksum
inverse relations
index terms
```

工具可以将 `revision` 持久化到 frontmatter，但必须自动递增；并发合并使用 `base_revision + base_content_hash`，不依赖人工记忆 revision。

### 6.2 `kind`

`kind` 是封闭枚举：

| kind | 含义 | 示例 |
|---|---|---|
| `reference` | 权威事实、API、ISA、硬件、编译器或 profiler 语义 | H100 shared-memory 限制、msprof 指标含义 |
| `method` | 可执行的实现或优化方法 | blocked reduction、persistent kernel、double buffering |
| `diagnostic` | 从症状和证据到判定与下一实验的过程 | 区分带宽不足与并行度不足 |
| `experience` | 从一次或多次运行得到的条件化观测 | 某 tile 在 H100 大 shape 上提升、在 910B4 上回退 |

以下内容不增加为新的顶层 kind：

- `anti_pattern`：作为 `experience.subtype` 或 relation/tag。
- `failure_pattern`：作为 `experience.subtype`。
- `hardware_characteristic`：通常属于 `reference`，若来自测量则属于 `experience`。
- `workload_sensitivity`：属于 `experience.subtype`。
- `optimization_pattern`：属于 `method`。

### 6.3 `domains`

初始受控词表：

```text
operator
hardware
language
compiler
runtime
profiler
optimization
numerics
```

Domains 可以通过 ontology 增长，例如未来新增：

```text
communication
distributed
memory_management
```

Domain 是多选主题标签，不承担适用范围判断。

Domains 可以由 `kind`、scope、retrieval cue 和引用 term 自动推导并由作者覆盖；普通 Concept 不要求人工枚举所有相关 domain。Domain 只用于粗粒度浏览和召回，不能成为 scope hard filter。

### 6.4 `scope`

`scope` 表达 Concept 在什么条件下可直接适用、仅可类比或明确不适用。

```yaml
scope:
  operators:
    families:
      - operator.softmax
    definitions: []
    required_motifs:
      - motif.reduction
      - motif.elementwise
    required_dataflow:
      - dataflow.reduce_then_broadcast

  targets:
    backends:
      - cuda
    architectures:
      - nvidia.hopper
    devices:
      - nvidia.h100-sxm-80gb
    required_capabilities: []

  software:
    languages:
      - triton
    language_version: ">=3.2,<4"
    compilers: []
    compiler_revisions: []
    runtimes:
      - cuda
    runtime_version: ">=12.4"

  workloads:
    dtypes:
      - fp16
      - bf16
    layouts:
      - contiguous
    conditions:
      all:
        - field: reduction_size
          op: ge
          value: 2048

  numerics:
    accumulation_dtypes:
      - fp32
    required_properties: []

  excludes:
    devices: []
    capabilities: []
    workload_conditions:
      any:
        - field: reduction_size
          op: lt
          value: 128
```

Scope 字段为空表示“未知或未声明限制”，不自动等价于“已证明全局适用”。明确可移植必须使用显式 `portable: true`，并满足 lifecycle 和 evidence 的 promotion 规则。

Workload condition 使用受控表达式：

```yaml
conditions:
  all:
    - field: reduction_size
      op: ge
      value: 2048
    - any:
        - field: dtype
          op: eq
          value: fp16
        - field: dtype
          op: eq
          value: bf16
```

Core V1 允许的组合节点是 `all`、`any`、`not`，比较操作是 `eq`、`ne`、`lt`、`le`、`gt`、`ge`、`in`、`not_in`。`field` 必须来自 OperatorSignature/workload 的受控字段注册表。Query engine 只能解释该 DSL，禁止执行任意 Python、SQL 或字符串表达式。

### 6.5 `retrieval`

`retrieval` 描述何时应当召回该 Concept：

```yaml
retrieval:
  phases:
    - initial
    - post_eval
    - post_profile
  tasks:
    - architecture_selection
    - diagnosis
    - implementation
  symptoms:
    - symptom.large_shape_regression
    - profile.resource_pressure
  techniques:
    - technique.blocked_reduction
  keywords:
    - row reduction
    - tile size
```

结构化 retrieval cue 用于精确召回，`keywords` 只作为 lexical fallback。

### 6.6 `lifecycle`

```yaml
lifecycle:
  status: validated
  confidence: medium
  ontology_version: "1.0"
  owner: kernelgen
```

`created_at`、`updated_at` 和 revision 由工具管理，可以出现在规范化文件或索引中，但不属于 Agent 输出契约。

`status` 使用：

| status | 含义 |
|---|---|
| `candidate` | Distiller 刚生成，尚未进入公共 KB |
| `validated` | 至少有直接证据支持，但适用范围仍由 scope 限制 |
| `contested` | 存在无法消解的相反证据 |
| `falsified` | claim 已被更强证据否定 |
| `deprecated` | 因版本、硬件或新知识失效 |

`confidence` 使用 `high`、`medium`、`low`。`validated` 与 `confidence` 是两个维度；单次可靠测量可以是 `validated + low confidence`。

### 6.7 `evidence`

```yaml
evidence:
  - id: evidence.run.kernelgen-001.r4
    type: measured_run
    claim_role: supports
    run_id: kernelgen-001
    round_num: 4
    definition_id: definition.softmax.v1
    solution_sha256: 4e6c...
    evaluation_fingerprint: 9ad2...
    profile_analysis_path: .kernelgen/profile-analysis/round-0004.json
    workload_uuids:
      - workload-17
      - workload-18
    observed_target_fingerprint: target:cuda:nvidia:h100-sxm-80gb:...
    result_summary: "Latency decreased for reduction_size >= 4096 and regressed below 256."

  - id: evidence.source.triton-docs-001
    type: authoritative_source
    claim_role: supports
    uri: sources/docs/triton/...
    revision: abc123
    locator: "section or symbol"
```

`claim_role` 使用 `supports`、`contradicts`、`qualifies`。Evidence 不能只保存自然语言来源名称，必须包含可解析 locator。

### 6.8 `relations`

```yaml
relations:
  broader:
    - kg:concept:method:reduction
  implements:
    - kg:concept:method:blocked-reduction
  requires:
    - kg:concept:reference:triton-program-id
  contradicts:
    - kg:concept:experience:softmax-single-config
  supersedes: []
```

Core V1 只允许作者维护 `broader`、`implements`、`requires`、`contradicts` 和 `supersedes` 五种强关系。`narrower`、`implemented_by`、`required_by` 和 `superseded_by` 由索引生成器构建；`contradicts` 自动构建对称边。`related` 容易退化成无意义弱边，不进入 Core V1，只有后续检索评测证明必要时才加入。

### 6.9 正文模板

```markdown
# Title

## Claim

一条明确、条件化且可验证的结论。

## Mechanism

解释为什么该结论可能成立，并区分证据支持的因果关系和仍未验证的推断。

## Action

未来 Agent 可以执行的具体动作、参数和验证方式。

## Applicability

对 scope 的人类可读解释。

## Limitations and counterexamples

已知不适用条件、冲突证据、版本风险和不可迁移部分。

## Evidence summary

对 frontmatter evidence 的精炼解释，不复制完整 report。
```

### 6.10 Kind-specific 正文要求

通用模板不要求每类知识维护无意义段落。各 kind 的最小正文为：

```text
reference
  Fact
  Applicability and versions
  Authoritative source
  Limitations

method
  Goal and preconditions
  Mechanism
  Procedure and parameters
  Validation
  Risks and rollback

diagnostic
  Trigger
  Required observations
  Procedure
  Judgment criteria
  Next experiments
  Blind spots

experience
  Claim
  Intervention
  Observed result
  Applicability
  Counterexamples or uncertainty
  Evidence summary
```

一个 Concept 只表达一个主要 claim 或 procedure。包含多个独立事实和方法的大文档保留为 SourcePackage 内容，并蒸馏成多个 Concepts，而不是复制成一个巨型 Concept。

## 7. TargetContext

### 7.1 目标

TargetContext 为同一算子跨芯片运行提供规范化身份，取代自由字符串 `target_hardware` 作为知识检索主键。

```yaml
schema_version: "1.0"
backend: cuda
vendor: nvidia
architecture: nvidia.hopper
architecture_id: sm90a
device_model: nvidia.h100-sxm-80gb
device_revision: ""
memory_bytes: 85899345920
capabilities:
  - capability.tensor_core
  - capability.tma

software:
  language: triton
  language_version: 3.3.0
  compiler: triton
  compiler_revision: abc123
  runtime: cuda
  runtime_version: "12.8"
  driver_version: "570.x"
  framework: pytorch
  framework_version: "2.x"

profiler:
  name: ncu
  version: "..."
  capabilities:
    - profile.metrics
    - profile.instruction_listing

fingerprint: target:cuda:nvidia:h100-sxm-80gb:...
```

Ascend 使用同一结构：

```yaml
backend: ascend
vendor: huawei
architecture: huawei.ascend-910b
architecture_id: ascend-910b
device_model: huawei.ascend-910b4
software:
  language: triton-ascend
  compiler_revision: abc123
  runtime: cann
  runtime_version: "..."
profiler:
  name: msprof
```

### 7.2 所有权

- Requested target 可以来自 workflow 输入。
- Canonical TargetContext 必须由 eval/profile service 或可信的本地探测器返回。
- Agent 不得自行声明 device、runtime version 或 capability。
- Ledger 后续版本应保存 canonical TargetContext 或 fingerprint，使导出的运行记录可独立理解。
- Knowledge query 使用 canonical fields，不使用用户提供字符串做等值匹配。

### 7.3 目标层级

```text
portable
  → backend
    → architecture family
      → device model
        → exact target fingerprint
```

H100 和 H200 可以共享 Hopper 架构知识，但设备带宽、容量和某些性能阈值仍需设备级证据。CUDA 与 Ascend 可以共享抽象 method 或 diagnostic，但不能共享后端专属实现事实。

## 8. OperatorSignature

### 8.1 设计原则

OperatorSignature 使用开放世界、组合式表示。新算子不要求先归入已有 family，也不要求所有计算节点均存在于核心 ontology。

OperatorSignature 描述算子的逻辑语义，不描述当前 candidate kernel 的实现表现。`uncoalesced load`、`low occupancy` 或 `pipeline idle` 属于 eval/profile observation，不属于 OperatorSignature。

### 8.2 Core V1 结构

Core V1 不要求生成完整 computation DAG，而使用稳定、可校验的结构指纹。结构指纹表达 motif、短 path、dataflow feature 和 logical access，足以支持新算子的多路径召回，同时避免把首版演变成通用计算图规范化项目。

```yaml
schema_version: "1.0"
ontology_version: "1.0"
definition_id: definition.fused-rmsnorm-gate.v1

semantics:
  canonical_name: fused_rmsnorm_gate
  family_candidates:
    - term: operator.rms_norm
      confidence: 0.88
    - term: operator.normalization
      confidence: 0.97
  axis_roles:
    - axis: 0
      role: axis.batch
    - axis: -1
      role: axis.reduction
  numerical:
    accumulation_dtype: fp32
    deterministic: true
    tolerance: {}

structure:
  motifs:
    - id: motif.elementwise
      operations:
        - square
      confidence: 1.0
    - id: motif.reduction
      operation: sum
      axis_role: axis.reduction
      confidence: 1.0
    - id: motif.elementwise
      operations:
        - rsqrt
        - multiply
      confidence: 1.0
  paths:
    - motif.elementwise>motif.reduction
    - motif.reduction>edge.broadcast>motif.elementwise
  features:
    - feature.last_axis_reduction
    - feature.row_local_dependency
    - dataflow.reduce_then_broadcast
  logical_access:
    - access.contiguous
    - access.broadcast

workload_features:
  dtypes:
    - fp16
  layouts:
    - contiguous
  rank:
    - 2
  dimensions:
    batch_size:
      min: 1
      max: 4096
    reduction_size:
      values:
        - 1024
        - 4096
        - 8192

unknown_terms: []

confidence:
  deterministic_fields: high
  semantics: high
  structure: medium
```

Structure path 不是任意字符串。每个 token 必须是 ontology term 或合法 extension，分隔符和最大长度由 Schema 固定。Core V1 只索引长度为 2 到 4 的 path，不执行通用子图同构。

### 8.3 固定结构与开放词表

固定内容：

- OperatorSignature 的字段结构
- Term ID 格式
- Evidence 和 relation 类型
- 核心 motif、structure path 和 dataflow feature 维度

开放内容：

- operator family
- 领域特有 primitive
- operation
- hardware capability
- compiler behavior term
- 新的 structure/dataflow term

### 8.4 核心 motif

第一版核心 motif：

```text
motif.elementwise
motif.reduction
motif.contraction
motif.scan
motif.selection_ordering
motif.gather_scatter
motif.layout_transform
motif.neighborhood_stencil
motif.histogram_atomic
motif.random_sampling
motif.collective_communication
```

一个算子可以同时具有多个 motif。`fusion` 是 dataflow/topology 特征，不是计算 motif。

### 8.5 核心 dataflow 维度

Dependency scope：

```text
scope.independent
scope.row
scope.segment
scope.tile
scope.sequence
scope.global
scope.sequential
```

Topology：

```text
dataflow.chain
dataflow.fan_in
dataflow.fan_out
dataflow.diamond
dataflow.reduce_then_broadcast
dataflow.multi_stage
dataflow.producer_consumer
dataflow.fused_chain
```

Logical access：

```text
access.contiguous
access.strided
access.blocked
access.transposed
access.broadcast
access.gather
access.scatter
access.indirect
access.irregular
```

Shape structure：

```text
shape.static
shape.dynamic
shape.ragged
shape.sparse
shape.variable_length
```

### 8.6 未知结构

未知节点不能被转换成 `other`。使用 namespaced extension：

```yaml
structure:
  motifs:
    - id: extension:mamba.selective_scan
      broader:
        - motif.scan
      confidence: 0.92

unknown_terms:
  - id: extension:project.selective_state_update
    description: Conditionally updates recurrent state using selected channels.
    source_locator:
      symbol: run
    candidate_broader:
      - motif.scan
      - motif.selection_ordering
```

Query engine 可以通过 `broader` 和已知属性召回通用知识。未知 term 被保留后，未来 ontology 更新可以重新映射，不需要重新读取原始 definition。

### 8.7 生成职责

确定性 Python 提取：

- definition identity
- inputs/outputs
- dtype
- workload axes 和具体取值
- 静态可识别的 shape、broadcast、reduction 和 API 调用
- reference source locator

Analyzer 推断：

- family candidates
- semantic tensor/axis roles
- motifs
- normalized structure paths/features
- dataflow topology 和 logical access
- numerical properties
- unknown extension 与 broader 候选

Python 校验：

- Schema
- ontology term ID
- extension namespace
- structure path token 和长度
- confidence
- workload 字段

Analyzer 可以返回多个候选和置信度，不能被强迫选择唯一 family。

### 8.8 Later：完整 computation DAG

只有当 golden query 证明 motif、structure path 和 feature 无法区分关键算子结构时，才增加可选 `computation_graph`。完整图必须由确定性 extractor 与 Analyzer 共同生成，定义 operation 等价、edge 语义、动态控制流和 canonicalization，并提供独立回归测试。缺少该字段的旧 Signature 仍然合法，查询必须继续支持 Core V1 表示。

## 9. Ontology

### 9.1 目录

```text
kb/ontology/
  manifest.yaml
  operator_families.yaml
  computation_motifs.yaml
  operations.yaml
  dataflow_terms.yaml
  axis_roles.yaml
  tensor_roles.yaml
  hardware_targets.yaml
  capabilities.yaml
  profile_terms.yaml
  aliases.yaml
```

### 9.2 Term 模型

```yaml
- id: motif.reduction
  label: reduction
  description: Combines multiple logical elements into fewer outputs.
  aliases:
    - reduce
    - fold
  broader: []
  status: active
  introduced_in: "1.0"
```

Term ID 一经公开不能改变语义。名称变化通过 alias 表达；语义替换通过 deprecation 和 supersedes 表达。

Core V1 的 ontology 只维护 `broader` 层级和 alias；弱 `related` 边不作为默认作者字段。新增 term 必须满足跨多个 definitions、documents 或 Concepts 的稳定复用需求，普通新算子使用 extension 而不修改公共 ontology。

### 9.3 Namespace

```text
operator.*          公共算子家族
motif.*             公共计算 motif
op.*                公共逻辑 operation
dataflow.*          公共数据流拓扑
access.*            公共逻辑访问模式
axis.*              公共 axis role
tensor.*            公共 tensor role
target.*            公共目标层级
capability.*        公共硬件/软件能力
profile.*           后端中立 profile 症状
symptom.*           Eval/error 层面的可观察症状
technique.*         用于检索的优化技术标签
extension:<ns>.*    项目或领域扩展
```

### 9.4 演进规则

- 新算子第一次出现时不自动创建公共 family。
- 新结构先以 namespaced extension 保存。
- 同一 extension 在多个 definitions 或 concepts 中复用，或具备独立优化/诊断价值时，才进入 ontology promotion review。
- Ontology 更新生成新版本，并维护 alias/deprecation/migration。
- 公共 Concept 记录其创建时的 `ontology_version`。
- Index builder 在不修改 Concept 正文的情况下解析旧 alias。

## 10. QueryContext

```yaml
schema_version: "1.0"
phase: initial
task: architecture_selection

operator_signature: {}
target_context: {}

evaluation:
  status: ""
  aggregate: {}
  workload_observations: []

profile:
  dominant_bound: unknown
  findings: []

error:
  category: ""
  message: ""

question: Select an initial architecture and avoid known invalid strategies.
limits:
  max_results: 12
  max_tokens: 8000
  include_candidates: false
```

### 10.1 Phase

```text
initial
post_eval
post_profile
error_recovery
plateau
implementation_lookup
```

### 10.2 Task

```text
architecture_selection
diagnosis
implementation
debugging
parameter_selection
workload_dispatch
portability
```

### 10.3 阶段性证据边界

- `initial` 只能基于 operator semantics、workload、TargetContext 和 Reference/Method 形成假设，不能声称已知运行瓶颈。
- `post_eval` 可以使用 correctness、per-workload latency、speedup 和 shape 分化。
- `post_profile` 可以使用后端中立 findings、backend detail 和 artifact evidence。
- `error_recovery` 可以使用编译、运行和 correctness 错误。
- Query engine 不能把后续阶段的经验以“已观测事实”注入早期阶段。

## 11. 查询流程

### 11.1 规范化

1. 校验 QueryContext。
2. 解析 ontology alias。
3. 规范化 TargetContext。
4. 将 workload axes 映射到 semantic roles。
5. 构建 operator、motif、structure path/feature、dataflow、target、symptom 和 lexical query facets。

### 11.2 硬过滤

以下条件命中时，Concept 不能作为当前目标的直接 action：

- `scope.requires` 不满足。
- `scope.excludes` 命中。
- 后端专属 implementation 与当前 backend 不同。
- 必需 capability 缺失。
- dtype、layout 或 numerical constraint 不兼容。
- 明确的软件版本范围不兼容。

被硬过滤的 Concept 可以在用户要求 portability 或 conflict analysis 时作为 rejected/analogy 返回，但必须说明原因。

### 11.3 多路径召回

Query engine 对以下路径并行召回并取并集：

1. 相同 definition ID。
2. 相同 operator family 或 alias。
3. 相同 motif 组合。
4. 相同 structure path 或 feature。
5. 相同 dataflow topology、dependency scope 或 logical access。
6. 相同结构化 workload condition、dtype、layout 或 semantic axis。
7. 相同 target fingerprint、device、architecture 或 backend。
8. 相同 eval observation。
9. 相同后端中立 profile finding。
10. 相同错误类别或错误特征。
11. BM25/关键词 fallback。
12. Concept coverage 不足或需要权威细节时，按 SourcePackage metadata 和 locator 搜索原始 source。
13. 可选 embedding recall，属于 Later。

查询采用 Concept-first、Source-fallback：

```text
Concepts
  → 已蒸馏、带 scope、可执行、低上下文成本

Sources
  → 原始文档、源码、论文和教程
  → 仅在覆盖不足、需要查证 API/版本或用户明确要求时读取
```

尚未蒸馏的 source result 标记为 `raw_source_reference`，不能自动提升为 `directly_applicable`。

### 11.4 Open-world 回退

新算子查询顺序：

```text
exact definition
  → family candidates
    → structure path/features
      → motif + dataflow
        → workload + target
          → generic method/diagnostic/reference
```

Family 无匹配时，Query engine 使用长度为 2 到 4 的规范化 structure paths、features 和 motif 组合做局部匹配。未知 extension 通过 `broader`、candidate broader、同一 path 中的已知 token 和文本描述参与召回。

### 11.5 匹配等级

```text
exact             相同 definition、目标和关键软件栈
family            相同 operator family 且 scope 兼容
structural        相同 structure path、feature 或 dataflow 结构
target            相同 target/architecture 上的通用知识
portable          与后端无关的 method/reference/diagnostic
analogy           其他设备、架构或 backend 上的相似经验
conflict          对相同 claim 存在相反证据
```

### 11.6 排序

第一版使用可解释规则评分：

```text
score =
    semantic_match
  + structure_match
  + dataflow_match
  + workload_match
  + target_match
  + phase_task_match
  + symptom_match
  + evidence_quality
  + freshness
  - scope_gap
  - contradiction_penalty
  - version_risk
```

建议初始权重：

| 条件 | 分值 |
|---|---:|
| 相同 definition | +100 |
| 相同 operator family | +50 |
| 相同关键 structure path | +45 |
| 相同 dataflow topology | +30 |
| workload condition 精确匹配 | +35 |
| 相同 target fingerprint | +45 |
| 相同 architecture | +30 |
| 相同 backend | +15 |
| phase/task 精确匹配 | +20 |
| profile symptom 精确匹配 | +35 |
| 多次独立证据 | +20 |
| 仅其他 device | -10 |
| 仅其他 architecture | -20 |
| 仅其他 backend | -30 |
| 软件版本风险 | -10 到 -40 |
| excludes 命中 | 不进入 direct result |

权重必须通过 golden query 校准，不能长期依赖主观常量。

### 11.7 结果去重与多样化

不能简单返回总分最高的 N 条，否则可能全部是同一个 definition 的重复经验。结果按槽位多样化：

- 最多若干 exact/family experience
- 至少一个适用 Method
- 至少一个相关 Diagnostic
- 必要的 Target Reference
- 最多若干 analogy
- 所有重要 conflict

同一 claim 的多个 Concept 优先合并为一个 result，并在 evidence 中列出多条来源。

## 12. KnowledgeBundle

```yaml
schema_version: "1.0"
query_id: query-...
operator_signature_id: definition.fused-rmsnorm-gate.v1
target_fingerprint: target:...

results:
  - concept_id: kg:concept:method:rowwise-blocked-reduction
    revision: 4
    kind: method
    match_level: structural
    score: 128.5
    matched_on:
      - motif.reduction
      - dataflow.reduce_then_broadcast
      - axis.reduction
      - target.architecture:nvidia.hopper
    scope_gaps:
      - Concept evidence includes H200; current target is H100.
    evidence_status: validated
    confidence: medium
    usage: candidate_experiment
    summary: ...
    action: ...
    source_path: concepts/method/rowwise-blocked-reduction.md

analogies: []
conflicts: []
rejected:
  - concept_id: kg:concept:method:cuda-tma-copy
    reason: Current target lacks capability.tma.

coverage:
  exact_definition: false
  operator_family: true
  structural: true
  target_specific: true
  profile_specific: false
  gaps:
    - No measured experience exists for this definition on H100.
```

`usage` 使用：

```text
authoritative_reference
directly_applicable
candidate_experiment
diagnostic_procedure
analogy_only
do_not_apply
```

Agent 必须根据 `usage` 使用内容，不能把 `analogy_only` 改写成当前目标事实。

## 13. 新算子检索示例

假设新算子 `segmented_selective_state_update` 从未进入 KB：

```yaml
semantics:
  canonical_name: segmented_selective_state_update
  family_candidates: []
structure:
  motifs:
    - motif.scan
    - motif.selection_ordering
    - motif.elementwise
  paths:
    - motif.selection_ordering>access.gather>motif.scan
  features:
    - scope.segment
    - scope.sequential
    - shape.ragged
  logical_access:
    - access.indirect
    - access.gather
unknown_terms:
  - id: extension:project.selective_state_update
    candidate_broader:
      - motif.scan
```

Query engine 不会因 family 为空而失败，而会召回：

- Scan 的通用并行化与数值约束。
- Segmented/ragged workload 的 Diagnostic。
- Indirect gather 的硬件和编译器知识。
- 相同 structure path 的其他算子经验。
- 当前 target 上支持或不支持的实现 capability。
- 其他 backend 上的 analogy，并标注迁移风险。

如果没有任何结构知识，KnowledgeBundle 仍应返回通用 target/compiler/reference 和明确的 coverage gap，而不是错误归类到最近的旧算子。

## 14. 多芯片知识与泛化

### 14.1 三层表达

同一优化思想分成：

```text
抽象 Method
  ├── CUDA implementation Concept
  └── Ascend implementation Concept

每个 implementation
  └── 多个设备/workload Experience
```

例如“异步搬运与计算重叠”是抽象 Method；TMA 实现和 Ascend 对应实现是不同 implementation Concept；H100、B200、910B4 上的测量分别是 Experience。

### 14.2 默认泛化规则

- 单个 exact target 的一次观测只能生成 exact target Experience。
- 同一 architecture 多个 devices 的一致证据可以候选提升为 architecture scope。
- 同一 backend 多个 architectures 的一致证据可以候选提升为 backend scope。
- 不同 backend 的一致结果只有在 claim 不依赖后端实现，并且机制和条件均兼容时，才可以候选提升为 portable scope。
- 不一致结果保留为条件分支或 `contested`，不能由 Merge 选择更大 speedup 的一方覆盖另一方。
- 扩大 scope 必须新增 promotion evidence，不能只编辑 frontmatter。
- Core V1 只生成 scope promotion proposal，不自动修改公共 Concept；Merge/curator 必须显式批准。

### 14.3 跨目标查询

查询优先级：

```text
exact target fingerprint
  → device
    → architecture
      → backend
        → portable
```

其他 backend 经验只进入 `analogies`，除非查询目标明确是 portability analysis。

## 15. 知识生产与写回

本章描述 Concept-aware 目标形态。当前实现的对应关系是：

| 目标概念 | 当前 legacy 实现 |
|---|---|
| candidate | agent workspace 的 `.new_experience.md` / `.new_detailed.md` |
| candidate validation | reducer 校验 ledger 的 definition、target、status、best score |
| merge | 高分候选优先的有界 `MergeAgent` 文本归并 |
| public/canonical | run-level KB 的 `experience.md` / `detailed.md` |
| revision history | run-level `kb/.git` 的每 epoch commit |

### 15.1 Distiller 输入

Distiller 使用：

- OperatorSignature
- TargetContext
- 全部 ledger rounds
- plan、solution、evaluation、profile、conclusion
- best round
- 当前检索使用过的 Concept IDs
- 现有相关 Concepts

### 15.2 Distiller 输出

未来 `DistillOutput` 应由自由文本升级为：

```yaml
candidate_concepts:
  - proposed_id: kg:concept:experience:rmsnorm-large-reduction
    kind: experience
    title: ...
    summary: ...
    scope: {}
    retrieval: {}
    proposed_relations: {}
    body: |
      ...
    evidence_intents:
      - round_num: 4
        claim_role: supports
skip_reason: ""
```

每个 candidate 只能表达一个主要 claim。若一次 trajectory 得到“一个成功方法”和“一个独立失败模式”，应生成两个 Concepts。

Distiller 只声明 `round_num`、workload UUID 或 SourcePackage locator 等 evidence intent；Candidate writer 从 ledger/profile/service 回填 target fingerprint、solution hash、evaluation fingerprint、latency 和 artifact path。Agent 不能把自己生成的数字直接写成权威 evidence。

### 15.3 Distiller 规则

- 直接测量优先于 round narrative。
- 代码 diff 与测量不支持的因果解释必须标为 hypothesis。
- 成功、失败、部分成功和 workload 分化均可成为知识。
- 未 profile 的 round 不能生成 profiler 结论。
- 只在目标设备上测量的经验不能声明其他设备适用。
- 必须保留 dtype、shape、layout、软件栈和数值约束。
- 如果没有新增、可靠、可执行的知识，返回 `skip_reason`。

### 15.4 Merge 决策

Concept-aware Merge 内部决策应比当前文本 KEEP/DISCARD/MERGE 更具体：

```text
reject_candidate
create_concept
update_concept
add_evidence
add_counterexample
mark_contested
supersede_concept
keep_separate
```

现有 `KEEP`、`DISCARD`、`MERGE` API 可以在迁移阶段映射：

- `KEEP` → `reject_candidate` 或 `add_evidence` 但无需改正文。
- `DISCARD` → `reject_candidate`。
- `MERGE` → `update_concept`、`add_evidence`、`add_counterexample` 或 `mark_contested`。

### 15.5 Merge 约束

- 先比较 scope，再比较性能数字。
- 不兼容 target、workload、baseline 或 metric 的数字不能直接比较。
- 兼容 claim 的新运行优先追加 evidence，不反复重写正文。
- 相反结果优先增加 counterexample 或 contested，不静默删除。
- Scope 扩大只能产生 promotion proposal，Core V1 不自动批准。
- Ontology term promotion 与普通 Concept merge 分开执行。
- 并行 Agent 对同一 Concept 的更新使用 revision 检查，冲突时重新基于最新 revision 合并。

## 16. Agent 与 Workflow 集成

本章各项是迁移到 Concept KB 时的新增职责。当前 workflow 仍遵循“agent
只产候选、epoch reducer 单点写回、EpochSummary 只做跨 agent 策略综合”的
边界。

### 16.1 Analyzer

新增职责：

- 生成 OperatorSignature 的推断部分。
- 使用 `phase=initial` 查询 KB。
- 将 KnowledgeBundle 中的 exact、structural、target 和 coverage gap 转换成初始架构候选。

不负责：

- 声称尚未 profile 的瓶颈。
- 写入公共 ontology。
- 直接修改 KB。

### 16.2 Coder

调用时机：

- 初始实现前。
- Eval 显示 workload 分化后。
- ProfileAnalyzer 给出 finding/next experiment 后。
- 编译、运行或 correctness 错误后。
- Plateau 时。

Coder 的 ExperimentPlan 应记录使用过的 `concept_id@revision`，建议未来在 `PlanSource` 增加：

```yaml
knowledge_refs:
  - kg:concept:method:rowwise-blocked-reduction@4
```

这样 Distiller 可以比较“知识建议的预期”与“实际测量”，评估 Concept 的有效性。

### 16.3 ProfileAnalyzer

- 使用 `phase=post_profile` 和后端中立 findings 查询 Diagnostic、Method 和 Experience。
- Backend detail 用于寻找 backend implementation；通用检索使用 `ProfileFinding.category`。
- 下一实验引用具体 Concept ID。
- 不将 NCU 指标名硬编码为全局瓶颈分类。

### 16.4 Distiller

- 从已完成且通过 ledger gate 的 trajectory 生成 candidate Concepts。
- 引用 immutable evidence。
- 保留已使用知识与实际效果之间的 calibration。

### 16.5 Merge

- 执行 Concept identity、scope、claim 和 evidence 级合并。
- 处理新增证据、反例、冲突和 scope promotion proposal。
- 不从外部搜索新证据。

### 16.6 EpochSummary

- 查询跨 Agent 已验证知识和本 epoch 新候选。
- 使用 retrieval coverage 避免多个并行方向重复探索同一 Concept。
- 把 next direction 绑定到目标 Concept 或明确的 coverage gap。

### 16.7 eval/profile service

- 返回 canonical TargetContext。
- 保持 backend-neutral evaluation/profile contract。
- 暴露 backend capabilities。
- Artifact path 和 fingerprint 保持可验证。

### 16.8 `query-kb` Skill

Skill 仅规定：

- 哪些阶段必须查询。
- 怎样构造问题。
- 怎样区分 direct、candidate、analogy 和 conflict。
- 怎样将 Concept 引用写入 plan。

Skill 不再维护硬编码目录清单，也不直接承载知识正文。

## 17. 存储布局

```text
kb/
  README.md

  ontology/
    manifest.yaml
    operator_families.yaml
    computation_motifs.yaml
    operations.yaml
    dataflow_terms.yaml
    axis_roles.yaml
    tensor_roles.yaml
    hardware_targets.yaml
    capabilities.yaml
    profile_terms.yaml
    aliases.yaml

  concepts/
    reference/
      hardware/
      languages/
      compilers/
      runtimes/
      profilers/
      operators/
    method/
      portable/
      cuda/
      ascend/
    diagnostic/
      portable/
      cuda/
      ascend/
    experience/
      operator/
      workload/
      hardware/
      compiler/

  sources/
    packages/
      triton-ascend.yaml
      triton.yaml
    docs/
    code/
    reference-kernels/
    reference-projects/

  indexes/
    catalog.jsonl
    term-postings.json
    structure-path-postings.json
    lexical-index/
    knowledge.db

  candidates/
    pending/
    rejected/

  retrieval-tests/
    golden/
    fixtures/
```

目录只用于人类浏览和大类管理，查询不依赖目录表达完整 scope。`sources/packages/` 为来源包 manifest；普通 source 文件通过 package ID、相对路径和 locator 引用。`indexes/` 是从 ontology、Concept frontmatter 和 SourcePackage manifests 生成的派生数据，不能手工编辑。

原始运行文件继续保留在 run workspace/profile-results，不复制进 `kb/concepts/`。

## 18. Query API

### 18.1 MCP

建议新增一个公开工具：

```text
query_knowledge
```

请求：

```json
{
  "phase": "post_profile",
  "task": "diagnosis",
  "round_num": 4,
  "question": "Find applicable diagnostics and a next experiment for the recorded resource-pressure finding.",
  "max_results": 12
}
```

在 Agent workspace 中，tool 根据 `.kernelgen/tool-context.json`、OperatorSignature、ledger 和 profile analysis 获取权威上下文。Agent 不允许通过参数伪造 definition、target 或 evaluation fingerprint。

响应为 KnowledgeBundle。

### 18.2 CLI

保留人工诊断入口：

```bash
python -m kernelgen.tools.query_knowledge --workspace PATH --phase initial --task architecture_selection
```

CLI 与 MCP 调用同一个 Python query engine。

### 18.3 内部组件

```text
OperatorSignatureBuilder
OntologyRegistry
ConceptLoader
IndexBuilder
ScopeMatcher
StructureMatcher
KnowledgeRanker
ResultDiversifier
KnowledgeQueryService
```

## 19. 索引

### 19.1 `catalog.jsonl`

每行是一个 Concept revision 的规范化 metadata：

```json
{"id":"kg:concept:method:rowwise-blocked-reduction","revision":4,"kind":"method","domains":["operator","optimization"],"terms":["motif.reduction","dataflow.reduce_then_broadcast"],"path":"concepts/method/portable/rowwise-blocked-reduction.md"}
```

### 19.2 Term postings

```text
term → concept IDs
```

覆盖 family、motif、operation、dataflow、target、capability、profile symptom、task 和 phase。

### 19.3 Structure path postings

对 OperatorSignature-lite 的 structure 生成规范化 path：

```text
motif.elementwise > motif.reduction > motif.elementwise
motif.reduction > edge.broadcast > motif.elementwise
motif.selection_ordering > access.gather > motif.scan
```

Core V1 使用 path overlap 和 feature overlap，不实现通用子图同构。完整 computation DAG 和复杂 graph matching 属于 Later。

### 19.4 Lexical index

BM25 覆盖 title、summary、claim、action、limitations、aliases 和 extension description。Lexical recall 用于发现 ontology 尚未覆盖的新术语，不负责最终 scope 判定。

### 19.5 Embedding

Embedding 是可选后续能力，只参与 candidate recall 或 rerank。它不能绕过硬 scope filter，也不能改变 evidence status。

### 19.6 SQLite 投影

Core V1 可以先使用 JSONL/postings；当需要统一事务、FTS5 和 relation 查询时，生成本地 SQLite `knowledge.db`。SQLite 仍是可删除、可重建的派生投影，不是公共知识 source of truth。只有实际规模和多跳查询证明本地投影不足时，才评估外部图数据库。

### 19.7 Source 索引

SourcePackage manifest 进入 package-level metadata index；文件列表、heading、symbol、line range 和 lexical chunks 由 IndexBuilder 自动生成。维护者不为每个普通文件手写 metadata。Source result 必须携带 package ID、package revision、相对路径和 locator，确保 Reference Concept 和查询结果可追溯。

## 20. 一致性与并行写入

- Concept `id` 稳定，`revision` 由工具单调递增，作者和 Agent 不手工维护。
- Candidate 记录其读取的 base revision 和 base content hash。
- Merge 使用 optimistic revision check。
- 两个并行 Agent 修改同一 Concept 时，第二个提交必须重新对最新 revision 合并。
- Index 在 Concept commit 后原子重建。
- Git commit 保存 Concept、ontology 和 index source 的变更；派生索引可选择提交或 CI 重建。
- Concept 迁移后仍保持当前单写者原则：isolated agent 只提交 candidate；
  epoch reducer 统一合并 Concept 文件和 evidence entry。不要恢复每个
  workspace 的嵌套 Git diff merge。
- 删除 Concept 使用 `deprecated` 或 `superseded`，不直接删除历史文件。
- 直接人工编辑 Concept 时，validator 根据 Git/base hash 检测 revision 漂移并通过管理工具补齐元数据。

## 21. 安全与可信边界

- KB Markdown 是数据，不是高权限系统提示；query tool 返回正文时应使用明确 data boundary。
- Raw conversation 和未知外部文档默认不直接注入 Agent prompt。
- Authoritative source 和 measured evidence 分别标注，不能把 Agent narrative 升级为事实。
- Agent 生成的 target、latency、speedup、profile metric 不作为 evidence，必须与 Python/service 记录一致。
- `artifact_path`、solution hash、fingerprint 和 workload UUID 必须通过现有 ledger/profile validation。
- Concept body 中的代码示例不能自动执行；Coder 仍需通过 eval gate 验证。

## 22. 质量评测

### 22.1 Golden query

```yaml
id: query.rmsnorm.h100.resource-pressure
query:
  operator_signature_fixture: fixtures/rmsnorm.yaml
  target_context_fixture: fixtures/h100.yaml
  phase: post_profile
  findings:
    - profile.resource_pressure
expected:
  must_retrieve:
    - kg:concept:diagnostic:resource-pressure
    - kg:concept:method:rowwise-blocked-reduction
  should_retrieve:
    - kg:concept:experience:rmsnorm-hopper-large-hidden
  must_not_directly_apply:
    - kg:concept:method:ascend-vector-pipeline
```

### 22.2 指标

| 指标 | 目的 |
|---|---|
| Recall@K | 相关知识能否被召回 |
| Precision@K | 返回内容是否真正相关 |
| nDCG | 最有价值知识是否位于前面 |
| false-transfer rate | 其他芯片经验是否被错误当成当前事实 |
| duplicate rate | 返回结果是否重复 |
| conflict recall | 冲突证据是否被暴露 |
| coverage accuracy | coverage gap 是否真实 |
| actionability | 结果是否能形成具体 ExperimentPlan |
| citation validity | evidence locator 是否存在且匹配 |

### 22.3 测试层级

- Ontology schema 和 alias/deprecation 测试。
- Concept frontmatter 与正文模板测试。
- Scope filter 单元测试。
- Target hierarchy 测试。
- Structure path/feature 生成与匹配测试。
- Query rank/dedup 测试。
- Golden retrieval 回归。
- Distiller candidate schema 测试。
- Merge conflict/scope promotion 测试。
- MCP 使用权威 workspace context 的集成测试。
- CUDA 与 Ascend 端到端检索测试。

## 23. 可观测性

每次 query 保存紧凑 retrieval trace：

```yaml
query_id: query-...
context_fingerprint: ...
candidate_counts:
  exact: 0
  family: 4
  structural: 19
  lexical: 8
filtered:
  incompatible_backend: 3
  missing_capability: 2
returned:
  - concept@revision
latency_ms: 14.2
```

ExperimentPlan 保存实际使用的 Concept refs。Distiller 后续可以计算：

- Concept 被召回次数。
- 被用于计划次数。
- 对应实验成功、失败和不可评估次数。
- 在哪些 target/workload 上结果冲突。

这些数据用于改进 retrieval cue 和评分，但不能自动把相关性变成因果性。

## 24. 迁移方案

### Phase 0：冻结设计与术语

- 评审本设计。
- 确定 Concept、OperatorSignature-lite、TargetContext、SourcePackage 和 KnowledgeBundle v1 schema。
- 确定 core ontology 初始 term。
- 写入 `kb/SPEC.md` 并明确作者字段、工具字段和所有权。

### Phase 1：结构与只读索引

- 增加 Pydantic models。
- 增加 ontology registry。
- 增加 Concept loader/validator。
- 增加 SourcePackage loader/validator。
- 从现有 Markdown 构建 legacy adapter 和 `catalog.jsonl`。
- 不改变当前“Distiller 产候选、epoch reducer 调用 MergeAgent 并写回”的
  单写者流程。

### Phase 2：确定性查询

- 实现结构化 condition evaluator、ScopeMatcher、ontology expansion、term/structure-path postings、规则评分和 KnowledgeBundle。
- 增加 `query_knowledge` CLI/MCP。
- 修改 `query-kb` Skill，移除硬编码主要路径。
- Analyzer/Coder/ProfileAnalyzer 逐步改用新工具。

### Phase 3：OperatorSignature 与 TargetContext

- 扩展 Analyzer 输出或增加独立 signature builder。
- eval/profile service 返回 canonical TargetContext。
- workspace 保存 `.kernelgen/operator-signature.json` 和 `.kernelgen/target-context.json`。
- query tool 从 workspace 权威读取。

### Phase 4：Concept-aware 写回

- Distiller 输出 candidate Concepts。
- Candidate writer 从 ledger/profile/service 回填权威 evidence。
- Merge 实现 evidence append、counterexample、contested 和 scope promotion proposal。
- ExperimentPlan 增加 `knowledge_refs`。
- 将 legacy `experience.md` 拆分成小粒度 Concepts。

### Phase 5：跨算子与跨芯片提升

- 基于多个 definitions/targets 生成并人工/规则审核 promotion proposal。
- 增加 architecture/backend/portable scope 规则。
- 建立 CUDA/Ascend golden retrieval suite。

### Phase 6：可选高级检索

- 根据 golden query 数据决定是否增加 embedding、learned ranker 或更复杂 graph matching。
- 高级检索必须保留 scope hard filter、evidence status 和结果解释。

## 25. Legacy 兼容

迁移阶段 LegacyAdapter 将现有路径映射为临时候选 metadata：

```text
experience/by_definition/{op_type}/{definition}/{target_hardware}/experience.md
experience/by_definition/{op_type}/{definition}/{target_hardware}/detailed.md
```

映射规则：

- `kind=experience`
- `scope.operators.definitions=[definition]`
- `scope.operators.families` 从 `op_type` 生成低置信度候选
- `target_hardware` 尝试解析为 target alias；解析失败则记录 unresolved target
- evidence 缺失时 `status=candidate` 或 `confidence=low`
- 原文件保留为 source locator

Legacy 内容不能因被导入就自动获得 `validated` 状态。只有能关联 ledger/evaluation/profile 证据的 claim 才能升级。

## 26. 验收标准

V1 完成需要满足：

1. 一个没有 family 的新算子可以通过 motif、structure path/features 和 dataflow 检索至少通用 Method/Diagnostic/Reference。
2. H100 的后端专属实现不会作为 910B4 的直接 action 返回。
3. H100/H200 的 Hopper 共同知识可以按 architecture 匹配，并保留 device scope gap。
4. CUDA 与 Ascend profile findings 可以通过同一后端中立 category 检索通用 Diagnostic。
5. 每条 Experience result 至少包含一个有效 evidence locator，或明确标为 candidate/low confidence。
6. Query result 包含 `matched_on`、`scope_gaps`、`usage` 和 coverage。
7. Distiller 不能把单设备结果自动写成 portable Concept。
8. Merge 不会用不兼容 workload 上的更大 speedup 覆盖兼容条件下的反向证据。
9. 新 ontology extension 不会使旧 Concept 或旧 Signature 校验失败。
10. Golden query suite 能检测 false transfer、漏召回、重复和冲突遗漏。
11. 新增一个未注册 family 的普通算子不修改公共 Schema 或 ontology。
12. 新增一个文档仓库只需要一个 SourcePackage manifest，普通文件不需要逐个维护 metadata。
13. 删除 `kb/indexes/` 后可以确定性重建等价索引。
14. Agent 输出不包含权威 revision、target fingerprint、solution hash 或 evaluation fingerprint。
15. Workload condition 不允许任意字符串表达式或动态代码执行。

## 27. 与 Learned Skills 的关系

KernelGen 不需要为每类增长知识创建 `learned-*` Skill。

```text
Skill
  = 稳定工作流和工具使用规则

Concept
  = 持续增长、带 scope 和 evidence 的知识

Ledger/Artifacts
  = 单次运行的权威事实
```

`query-kb` 是稳定 Skill；硬件、编译器、优化方法、诊断步骤和历史经验都是 KB Concept 或外部 source。Distiller/Merge 维护 Concept，不生成新的 Skill。只有当工作流本身发生稳定且跨任务的变化时，才考虑更新 Skill。

## 28. 最终边界

这套设计明确区分：

```text
分类：kind + domains
适用：scope
发现：retrieval + ontology + structure index
可信：evidence + lifecycle
演进：relations + revision + promotion
运行事实：ledger + artifacts
```

其开放性来自：

- family 可选。
- motif 多标签组合。
- Core V1 的 structure path/features 支持局部结构匹配，完整 computation graph 可以后续兼容增加。
- 未知 term 被完整保留为 namespaced extension。
- 查询能从 exact 逐级回退到 structural、target 和 portable。
- 跨芯片经验被条件化，而不是复制或错误泛化。

其鲁棒性来自：

- 确定性字段与 Agent 推断分离。
- Scope hard filter。
- 后端中立 profile finding。
- 权威 evidence locator。
- 可解释排序与 coverage gap。
- Ontology 版本化。
- Golden retrieval regression。

这使新算子即使没有名称、family 或完整历史经验，也可以从其 structure path/features、数据流、workload、目标平台和运行症状出发找到可验证的相关知识；同时系统会明确承认知识缺口，而不会通过错误归类制造虚假的确定性。

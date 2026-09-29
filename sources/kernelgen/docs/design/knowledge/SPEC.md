# KernelGen Knowledge Base V1 Specification

状态：Approved target；Schema 与 runtime 主链路已实现。实际接入范围见 [实现说明](knowledge_base_implementation.md)。

本文是 KernelGen KB V1 唯一规范性定义。实现结构、当前状态、改动顺序和验证
命令见 `docs/design/knowledge/knowledge_base_implementation.md`。其他文档不得重复定义字段或
行为。

规范词：

- MUST：实现必须满足。
- MUST NOT：实现禁止执行。
- SHOULD：默认应满足；偏离时必须记录理由。
- MAY：可选能力。

## 1. V1 范围

V1 必须闭合以下流程：

```text
读取稳定 Concept
  -> 构造 QueryContext
  -> scope 硬过滤和确定性检索
  -> Agent 引用知识执行实验
  -> 写入不可变 Evidence
  -> 产生 Candidate
  -> 单 Publisher 校验、合并、发布
  -> 重建派生索引
```

V1 不引入 PostgreSQL、外部图数据库、向量数据库、Cognee、Graphiti 或
Hindsight。SQLite 只允许作为可删除、可重建的本地查询投影。

V1 不实现自动跨设备泛化、自动 ontology 晋升、学习排序或完整 computation
DAG。

## 2. 权威边界

系统包含四类数据，权威级别不同。

### 2.1 Run Fact

`.ledger.json`、immutable solution、evaluation、profile 和 artifact 是单次运行
事实源。延迟、speedup、correctness、solution hash、target fingerprint 和
evaluation fingerprint 必须来自 Python 或 eval/profile service。

Agent 文本不得覆盖这些字段。

### 2.2 Evidence

Evidence 是 Run Fact 的不可变结构化引用。每条 Evidence 只描述一次实验结果，
必须能够解析回权威 Run Fact。

Evidence MUST append-only。纠错必须追加 correction 记录，不得静默改写原记录。

### 2.3 Candidate

Candidate 是 Agent 或 Distiller 提交的知识提案，不是公共知识。Candidate 可以
被拒绝、合并、拆分或保留待审。

Candidate 中的权威测量字段必须由 Candidate writer 从 ledger 回填。

### 2.4 Concept

Concept 是可供后续任务使用的稳定知识单元。V1 只允许四种 `kind`：

- `reference`：事实、约束、接口或平台能力。
- `method`：可复用优化动作及其机制。
- `diagnostic`：如何根据观测判断问题并选择下一实验。
- `experience`：特定上下文中的已测结果。

一个 Concept MUST 只有一个主要 claim。

### 2.5 数据平面

实现必须分离三个数据平面：

- Query Plane：只读固定 snapshot，负责 context、scope、召回和排序。
- Learning Plane：记录 query/usage，接收 Candidate，物化 Evidence。
- Publish Plane：单写者，负责 merge、revision、校验和原子发布。

Query Plane MUST NOT 修改公共 KB。Agent MUST NOT 访问 Publish Plane。
Publisher MUST NOT 修改 Ledger。

### 2.6 依赖边界

KB 核心只能依赖 Contract、Domain 和 Port。它 MUST NOT import Agent、Workflow、
MCP 或 Runtime。

Workflow、MCP 和 CLI MUST 通过同一 application service 使用 KB。Skill 只能
定义调用时机和结果使用规则，不得实现检索算法。

## 3. 存储布局

```text
kb/
  README.md
  SPEC.md

  concepts/
    reference/
      hardware/
      language/
      compiler/
      runtime/
      library/
      profiler/
      evaluation/
      numerics/
    method/
    diagnostic/
    experience/

  evidence/
    by_run/

  sources/
    packages/
    content/
    ingestion/

  registries/
    aliases.yaml
    targets.yaml
    terms.yaml

  schemas/
    concept.schema.json
    evidence.schema.json
    candidate.schema.json
    query_context.schema.json
    knowledge_bundle.schema.json
    operator_signature.schema.json
    target_context.schema.json
    source_package.schema.json

  retrieval-tests/
    fixtures/
    golden/

  .derived/
    catalog.jsonl
    knowledge.db
```

目录只负责浏览。适用范围 MUST 来自 `scope`，查询不得通过目录名推断完整
scope。

`.derived/` 是派生数据。删除后必须能从 Concept、registry 和 source manifest
确定性重建。

`sources/packages/` 保存 SourcePackage manifest；`sources/content/` 保存许可
允许纳入仓库的原始来源快照。`content_root` 使用 `kb://sources/content/...`
时，validator MUST 校验文件或目录存在且 checksum 一致。目录 checksum 必须
根据排序后的相对路径与逐文件 SHA-256 确定性计算。仓库型 SourcePackage
MUST 保存固定 revision 的完整 Git tree，并在 `sources/manifests/*.json`
记录每个 tracked entry 的 path、mode、object ID、size 和 SHA-256。Gitlink
只记录固定 commit，不得伪装成已导入的 submodule 内容。仓库内相对 symlink
可以保留，但必须解析在 Source root 内。
`sources/ingestion/` 保存经过 review 的静态摄取计划，不属于查询语料。
受限来源必须在 manifest 标明许可和分发边界。

`usage_policy` MUST 明确来源是 `redistributable`、`restricted`、
`federated_only` 或 `unknown`。`restricted`/`federated_only` MUST 提供
`usage_notes`；查询和导出层不得丢失该边界。CANN OSL 来源只能用于华为 AI
处理器和/或 CANN 相关软件，不得被推断为 portable 或 CUDA/NVIDIA 知识。
受限来源还 MUST 声明 `allowed_backends`；validator MUST 拒绝 portable、
缺少 backend 或 backend 不在许可集合内的 Concept。`query_sources` MUST
把 `usage_policy` 和 `usage_notes` 随每个命中返回。

### 3.1 静态来源摄取

静态文档、博客和仓库必须先注册为不可变 SourcePackage，再生成
source-backed Candidate。SourcePackage ID SHOULD 包含固定 revision；新
revision MUST 新建 package，不得覆盖仍被 Concept 引用的旧快照。

摄取模式：

- `source_only`：只保留原文，不生成 Candidate。
- `direct`：从已 review 的文件或 Markdown 章节确定性生成 Candidate。
- `extract`：使用人工或 Distiller 审核后的正文生成 Candidate。
- `federated`：只注册外部索引，不复制或发布正文。

带 Git manifest 的 SourcePackage，其 ingestion plan MUST 用规则覆盖全部
tracked entry。`source_only` 基线规则可以被后续更具体的 `direct` 或
`extract` 规则覆盖。完整 Source 可以通过独立 `query_sources` 接口进行有界
原文检索；该接口不改变 Concept 的适用性、证据或发布语义。

`direct` 与 `extract` MUST 经过同一个 Publisher；不得直接写 canonical
Concept。静态 Candidate MUST 包含 SourceReference，MUST NOT 请求实验
Evidence，MUST NOT 发布 Experience。发布结果的 `evidence_state` 必须为
`source_supported`。

SourceReference 的 `resource` 必须是精确 SourcePackage ID；`id` 使用
`<relative-path>#<heading-or-symbol>`。这样来源 revision 更新后，旧 Concept
仍可解析回原始内容。

静态 Candidate ID 必须由 SourcePackage revision、entry 和正文确定性生成。
默认 static batch ID 还必须包含 ingestion plan hash：同一 plan 重放幂等，
同一 source revision 下的 reviewed plan 修订可以生成新的 Concept revision。

每个 run 使用：

```text
.kernelgen/knowledge/
  state.json
  context/
    operator-signature.json
    target-context.json
  candidates.jsonl
  query-log.jsonl
  usage-log.jsonl
  publish-result.json
```

并行 Agent 写自己的 workspace；不得共同写一个 JSONL 文件。

`state.json` MUST 固定 `snapshot`、`catalog_ref` 和运行模式。同一 epoch 或
batch 的全部 Agent MUST 使用同一 snapshot。

## 4. Concept 文档

Concept 是带 YAML frontmatter 的 UTF-8 Markdown。

V1 Concept SHOULD 保持 OKF v0.2 基础兼容：提供 `type`，使用顶层
`status`、`generated`、`verified`、`stale_after` 和 `sources`。KernelGen
字段属于允许的扩展。`kind` 用于内部稳定枚举，validator 必须确认它与
`type` 一致。

### 4.1 必填字段

```yaml
---
schema_version: "1.0"
id: kg:method:two-stage-reduction
revision: 1
type: Method
kind: method
title: Two-stage reduction
summary: Split a large reduction into partial and final stages.
claim_key: method.reduction.two_stage
domains: [operator, optimization]
status: stable
generated:
  by: kernelgen/publisher-v1
  at: 2026-07-26T10:00:00Z
verified:
  - by: process:benchmark-validator
    at: 2026-07-26T10:00:00Z
stale_after: null
sources: []

scope:
  target:
    level: backend
    backend: cuda
  operator:
    motifs: [reduction]
  workloads:
    all: []
  numerics:
    exact: true

retrieval:
  phases: [initial, post_profile]
  tasks: [architecture_selection, diagnosis]
  symptoms: []
  techniques: [two_stage_reduction]
  keywords: [partial reduction]

evidence_state: corroborated

evidence_refs:
  - kg:evidence:run-123:agent-2:round-4

relations: []

managed:
  content_hash: sha256:...
  created_at: 2026-07-26T10:00:00Z
  updated_at: 2026-07-26T10:00:00Z
---
```

含义：

- `id`：跨 revision 稳定。移动文件不得改变 ID。
- `revision`：Publisher 单调递增。
- `type`：OKF 类型；只允许 `Reference`、`Method`、`Diagnostic`、
  `Experience`。
- `kind`：内部小写枚举，必须与 `type` 一一对应。
- `claim_key`：用于候选去重和 merge；相同 kind、同一语义 claim 使用同一值。
- `summary`：单句检索摘要，不保存多个独立结论。
- `retrieval.phases/tasks`：非空时是查询 eligibility 白名单；当前 phase/task
  不在列表中则不得进入召回。symptoms、techniques、keywords 用于加权路由。
- `sources`：外部或内部来源引用；Reference 无内部 Evidence 时必须至少包含一个
  SourcePackage locator。
- `managed`：工具字段。Agent MUST NOT 填写。

`managed.content_hash` 对规范化 frontmatter 和正文计算，但排除整个 `managed`
对象，避免自引用。snapshot hash 对排序后的 `id@revision:content_hash` 计算。

### 4.2 正文

正文 SHOULD 使用以下固定章节：

```markdown
# Claim

# Mechanism

# Action

# Applicability

# Limitations and counterexamples

# Evidence summary
```

`reference` 可以省略 `Action`；`experience` 必须包含 measured outcome 和
context，不得把假设写成已证实机制。

## 5. Scope

### 5.1 Target scope

```yaml
target:
  level: exact | device | architecture | backend | portable
  backend: cuda
  architecture: sm80
  devices: [NVIDIA-A100-SXM4-80GB]
  capabilities: [tensor_core]
  software:
    language: triton
    language_version: ">=3.2,<3.4"
    compiler: triton
    compiler_version: "3.3.1"
    runtime: cuda
    runtime_version: "12.6"
```

规则：

- `exact` 必须包含完整 target fingerprint。
- `device` 必须包含至少一个 device。
- `architecture` 必须包含 architecture。
- `backend` 必须包含 backend。
- `portable` 必须显式声明；字段缺失不表示 portable。
- Experience 的 target level 不得宽于其 Evidence。

### 5.2 Operator scope

```yaml
operator:
  definition_ids: []
  op_types: [normalization]
  motifs: [reduction, elementwise]
  dataflow: [reduce_then_broadcast]
  dtypes: [float16, bfloat16]
  layouts: []
```

空列表表示该维度不限制；未知值必须记录在查询 coverage gap 中，不能当作匹配。

### 5.3 Workload scope

任意条件必须使用受控表达式：

```yaml
workloads:
  all:
    - field: reduction_size
      op: gte
      value: 4096
    - field: batch_size
      op: in
      value: [1, 2, 4, 8]
  any: []
  excludes: []
```

V1 只允许：

```text
eq ne lt lte gt gte in not_in
```

禁止自由 predicate、Python、SQL、正则表达式和动态代码。

## 6. Lifecycle

`status`：

- `draft`：未进入默认查询结果。
- `stable`：可正常使用。
- `deprecated`：仅为历史链接保留。

`evidence_state`：

- `source_supported`：由外部来源支持，未被内部实验验证。
- `observed`：一个兼容 Evidence。
- `corroborated`：至少两个独立兼容 Evidence。
- `contested`：存在兼容反向 Evidence。
- `falsified`：claim 已被权威反证。

“独立”至少要求不同 run；Publisher MAY 增加更严格规则。

查询默认只返回 `stable`。`contested` 可以返回，但必须进入 conflicts 或带警告。
`falsified` 和 `deprecated` 不得作为 direct action。

## 7. Evidence

Evidence JSONL 最小结构：

```json
{
  "schema_version": "1.0",
  "id": "kg:evidence:run-123:agent-2:round-4",
  "run_id": "run-123",
  "workspace_id": "agent-2",
  "definition_id": "flashinfer:rmsnorm:v1",
  "round_num": 4,
  "target_context_ref": ".kernelgen/knowledge/target_context.json",
  "operator_signature_ref": ".kernelgen/knowledge/operator_signature.json",
  "ledger_locator": {"path": ".ledger.json", "round_num": 4},
  "solution_sha256": "...",
  "baseline_sha256": "...",
  "evaluation_fingerprint": "...",
  "profile_fingerprint": "...",
  "outcome": {
    "status": "passed",
    "geo_mean_speedup": 1.23,
    "workload_results_ref": "artifact://sha256/..."
  },
  "artifact_refs": [],
  "recorded_at": "2026-07-26T10:00:00Z"
}
```

`id` MUST 唯一。Publisher 必须以 ID 幂等处理。

Evidence MUST append-only。纠错记录必须设置 `supersedes_evidence_id`，并保留
原 Evidence。

公共 Concept 只能引用满足以下条件的 Evidence：

- ledger 可解析。
- definition、round、target 与 locator 一致。
- solution hash 和 evaluation fingerprint 校验成功。
- artifact locator 在发布后的保留期内可解析。

若证据不能长期解析，Candidate 可以保留，但不得发布成 `stable` Experience。

Publisher 只把被公共 Concept 引用的紧凑 EvidenceRecord 复制到
`kb/evidence/by_run/{run_id}.jsonl`。原始日志、代码和 profile 不复制进 Git，
继续通过 content-addressed artifact locator 引用。这样公共 Concept 跨 run
可解析，同时避免保存所有无价值实验。

## 8. Candidate

Candidate JSONL：

```json
{
  "schema_version": "1.0",
  "candidate_id": "run-123:agent-2:candidate-1",
  "base_snapshot": "sha256:...",
  "proposed_kind": "method",
  "proposed_id": null,
  "claim_key": "method.reduction.two_stage",
  "title": "Two-stage reduction",
  "summary": "...",
  "domains": ["operator", "optimization"],
  "scope": {
    "target": {
      "level": "backend",
      "backend": "cuda"
    }
  },
  "retrieval": {},
  "body": "...",
  "relations": [],
  "evidence_intents": [
    {"round_num": 4, "claim_role": "supports"}
  ],
  "source_refs": [],
  "created_by": "agent-2"
}
```

Candidate MUST NOT 包含 revision、content hash 或自行生成的权威测量值。

## 9. SourcePackage

外部博客、文档和仓库先注册 SourcePackage：

```yaml
schema_version: "1.0"
id: source:triton-docs
source_type: documentation
origin: https://triton-lang.org/
revision: "3.3.1"
  content_root: kb://sources/content/triton-docs/index.html
license: Apache-2.0
checksum: sha256:...
retrieved_at: 2026-07-26T10:00:00Z
```

原始来源不是直接 action。来源可产生 Reference、Method 或 Diagnostic
Candidate；来源报告的性能只能标记 `source_reported`，不能伪装成内部
Evidence。

Concept MUST 通过 `source:<id>` 引用已注册 SourcePackage，不得只保存易漂移
URL。manifest 的 `origin` 保留原 URL；`checksum` 固定本次原文快照。

## 10. QueryContext

```json
{
  "schema_version": "1.0",
  "phase": "post_profile",
  "task": "diagnosis",
  "round_num": 4,
  "question": "Find the next experiment for low utilization.",
  "operator_signature": {},
  "target_context": {},
  "workload_summary": {},
  "findings": [],
  "errors": [],
  "knowledge_snapshot": "sha256:...",
  "max_results": 12
}
```

`operator_signature` 和 `target_context` 必须由工具读取或构造。Agent 请求不得
覆盖权威 target、definition、evaluation 或 profile 字段。

## 11. 查询

查询顺序必须固定：

1. 加载并校验 snapshot、TargetContext 和 OperatorSignature。
2. 排除 deprecated、falsified 和默认不可见 draft。
3. 执行 target、numerics 和 workload 硬过滤。
4. 按 definition、motif/dataflow、finding/symptom、technique 和 lexical
   路径召回。
5. 使用固定规则排序。
6. 按 claim key 去重并限制同类结果数量。
7. 输出 direct、analogies、conflicts 和 coverage gaps。

匹配等级：

- `direct`：所有硬约束满足。
- `analogy`：结构相关，但 target 或 workload 存在明确 gap。
- `incompatible`：至少一个硬约束冲突；不得返回给 Agent。
- `unknown`：查询上下文缺字段；只可作为待验证候选。

V1 排序优先级：

```text
direct
  > exact definition
  > target specificity
  > motif/dataflow overlap
  > phase/task/symptom
  > corroborated
  > observed
  > source_supported
  > lexical-only
```

性能数字不得参与跨不兼容 scope 排序。

正式查询接口：

```text
query_knowledge
  输入：phase、task、question、optional round_num
  输出：KnowledgeBundle

get_knowledge
  输入：query_id、concept_refs、detail_level
  输出：已授权 Concept 正文或来源片段
```

Agent 只能提供查询意图。Definition、OperatorSignature、TargetContext、snapshot、
evaluation error 和 profile finding 必须由工具提供。

`get_knowledge` 只能读取同一 `query_id` 已返回的固定 revision。MCP、CLI 和
Workflow MUST 调用同一个 QueryService。

## 12. KnowledgeBundle

```json
{
  "schema_version": "1.0",
  "query_id": "query:run-123:agent-2:4",
  "snapshot": "sha256:...",
  "direct": [],
  "analogies": [],
  "conflicts": [],
  "coverage": {
    "matched_routes": [],
    "gaps": []
  }
}
```

每个结果必须包含：

```json
{
  "concept_ref": "kg:method:two-stage-reduction@4",
  "kind": "method",
  "summary": "...",
  "usage": "direct_action",
  "matched_on": ["motif.reduction", "target.backend.cuda"],
  "scope_gaps": [],
  "evidence_state": "corroborated",
  "evidence_refs": []
}
```

`usage` 只允许：

```text
constraint direct_action hypothesis diagnostic_context analogy warning
```

## 13. 知识使用反馈

ExperimentPlan SHOULD 记录：

```yaml
knowledge_uses:
  - concept_ref: kg:method:two-stage-reduction@4
    query_id: query:run-123:agent-2:4
    role: hypothesis
    disposition: adopted
```

`role`：

```text
constraint hypothesis implementation diagnostic
```

`disposition`：

```text
adopted adapted rejected
```

实验结束后，工具追加 outcome 和 attribution confidence。该反馈可以调整检索
排序，但不得直接修改 Evidence 事实或自动扩大 scope。

新 V1 查询产生的 `knowledge_uses` MUST 包含 `query_id`。Ledger 2.0 迁移记录
和未启用 V1 查询的 2.1 记录 MAY 为空。

eval tool 必须校验：

- Concept ref 属于当前 snapshot。
- revision 固定。
- ref 被当前 workspace 的 `query_id` 返回过。
- `disposition=adapted` 时 adaptation 非空。
- analogy 不得伪装成无适配说明的 direct implementation。

## 14. Publisher

只有 Publisher 可以写公共 Concept。

处理顺序：

1. 收集各 workspace 的 Candidate。
2. 校验 ledger 和 evidence intent。
3. 生成权威 Evidence。
4. 以 `kind + claim_key` 查找 merge 候选。
5. 比较 scope、claim 和已有 evidence。
6. 执行 merge decision。
7. 原子写 Concept 和 Evidence 引用。
8. 单调增加 revision。
9. 重建索引。
10. 写 publish result；成功后提交 Git。

允许的 merge decision：

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

Publisher MUST 保留反向结果。它不得因某个候选 speedup 更高而覆盖兼容 scope
中的回归证据。

Publisher MUST 处理一个完整 epoch 或 batch，不能按 Agent 完成顺序发布。
发布请求必须携带 `expected_snapshot`。snapshot 已变化时必须重新计算 merge，
不得覆盖新状态。

运行时只有三种状态：

```text
disabled
read_only_v1
read_write_v1
```

- `disabled`：Workflow 不传 `KnowledgeConfig`，不创建 V1 workspace state，也不
  读取或发布 V1 Catalog。
- `read_only_v1`：允许检索 Concept 和 Source，并在当前 workspace 写入运行态
  state、retrieval log、Candidate 和 `status=noop` 的 publish result；不得修改
  Catalog、run archive 或 Catalog 内 `.derived`。Catalog 索引缺失或过期时必须
  fail closed；只有显式提供 Catalog 外部的 `derived_root` 时，才可在那里重建
  可删除索引。Reviewer mode 必须为 `off`。
- `read_write_v1`：V1 读写均为权威路径。

`KnowledgeConfig` 和 workspace state 的 mode MUST 为 `read_only_v1` 或
`read_write_v1`；`legacy`、`shadow` 和 `read_v1` 不再是有效运行状态。启用
选择必须集中在 bootstrap/config，不得散落在 Agent 或 MCP。

## 15. 并发

V1 使用隔离写入和单 Publisher：

- 每个 Agent 只写自己的 `.kernelgen/knowledge/`。
- Publisher 是公共 KB 唯一写者。
- Candidate 记录读取时的 snapshot hash。
- 发布前执行 optimistic revision check。
- 所有文件使用临时文件加 `os.replace` 原子落盘。
- Publisher crash 后可凭 candidate ID 和 evidence ID 幂等重放。
- Publisher 必须使用 catalog lock、staging、全量校验和原子替换。
- stale snapshot 必须显式返回，不得使用 last-writer-wins。

因此 V1 不需要 PostgreSQL。SQLite WAL MAY 用于派生索引，但不能成为唯一
事实源。

## 16. Legacy 迁移

迁移期：

- `.new_experience.md` 映射为一个或多个 Candidate。
- `.new_detailed.md` 保留为 evidence narrative，不进入默认检索。
- `experience.md` 转换为 `experience` Concept。
- `detailed.md` 转换为 artifact 或 Evidence 补充材料。
- 历史 legacy Markdown 可以离线迁移为 Candidate/Evidence；新运行不得同时启动
  legacy reducer 和 V1 Publisher。

### 16.1 Ledger 兼容边界

KB V1 需要读取现有 ledger，因为历史运行是 Experience 和 Evidence 的来源。
兼容规则必须单向、有限：

- 新 Writer MUST 只写当前 `schema_version: "3.0"`。
- Reader MUST 原生读取 `3.0`，并在内存中迁移受支持的 `2.0` 至 `2.6`；不得覆盖
  源文件。
- `2.1` 至 `2.6` 只解释为 jiabei lineage；同名但来源可能不同的 `2.0` 必须先
  根据分支标记判定，无法判定时由调用方显式提供 origin，禁止猜测。
- 旧 ledger 缺少 `knowledge_uses` 时迁移为空列表；该 Evidence 仍可用于性能
  结论，但不能用于知识归因。
- `1.x`、缺失版本和未知未来版本 MUST 明确拒绝。
- 兼容逻辑只能存在于 ledger load/migration 边界，业务代码不得散布旧字段分支。

支持窗口结束后 MAY 使用显式离线工具升级归档 v2 ledger；删除 v2 reader 必须
单独做 major migration，不能在普通 V1 更新中静默删除。

## 17. 合规与验收

V1 必须满足：

1. 所有 Concept 通过 JSON Schema 和 Pydantic 双重校验。
2. 任一 Experience 都有至少一个可解析 Evidence。
3. A100 专属 Method 不作为 Ascend direct action 返回。
4. 缺失 target 字段不会被解释为 portable。
5. 相反 Evidence 被返回为 conflict，不被覆盖。
6. 同一 Candidate 重放不会重复 Evidence。
7. 删除 indexes 后可以重建等价查询结果。
8. 并行 Agent 不直接修改公共 Concept。
9. QueryBundle 包含匹配原因、scope gap 和 coverage。
10. Golden query 固定 false-transfer、冲突召回、去重和排序行为。
11. MCP、CLI 和 Workflow 对同一 QueryContext 返回相同 Bundle。
12. 伪造或未查询到的 knowledge ref 被 eval tool 拒绝。
13. 同 epoch/batch 的全部 Agent 使用同一 snapshot。
14. Publisher 失败时旧 snapshot、Concept 和 index 完全不变。
15. 删除 legacy adapter 后，KB core、QueryService 和 Publisher 无需修改。

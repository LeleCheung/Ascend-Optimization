# KernelGen V1 KB 快速使用

V1 KB 的完整读写闭环由多 Agent、多 epoch 的 `KernelGenWorkflow` 负责；`read_only_v1` 可用于只消费已发布知识而不修改 Catalog。`kg run --mode simple_opt` 默认不接入，但可以显式查询并生成本地 CandidateDraft；它不执行 Publisher、Solution promotion 或 fork。新 YAML Batch 为每个子任务使用同一入口和 Knowledge 参数；legacy Python Batch 的能力不变。

## 1. 验证 Catalog

仓库内置 Catalog 可用于离线验证：

```bash
cd /data/akg_kernel_bench_lite/kernelgen
python3 -m kernelgen.tools.kb validate --kb kb
```

读写运行建议将可写 Catalog 和 run archive 放在独立运行目录。只读运行可以直接
挂载基线 Catalog，但必须满足下文的派生索引约束。

## 2. 启动 Knowledge-enabled KernelGenWorkflow

先按 `docs/ONBOARDING.md` 和多设备 runbook 确认 Eval Server。Server
`/status.target.backend` 和 `/status.target.device` 必须存在；输入的
`--target-hardware` 必须与 Server 实际设备一致，不提供 fallback。Server
提供 `metadata` 时会原样保留 `complete/missing`；信息不完整时优化可以继续，
但不得发布 stable runtime knowledge，自相矛盾的元数据会在启动时直接拒绝。

启用 V1 读写闭环的运行示例：

```bash
python3 examples/kernel_gen/run_example.py \
  --definition my_operator \
  --workspace runs/kernel_gen_kb/my_operator \
  --eval-server http://127.0.0.1:8000 \
  --target-hardware Ascend910B \
  --catalog-path /path/to/native-catalog \
  --knowledge-mode read_write_v1 \
  --knowledge-catalog-path /path/to/catalog \
  --knowledge-reviewer-mode enforce \
  --n-parallel 2 \
  --n-epoch 1
```

只读消费生产 Catalog 的示例：

```bash
python3 examples/kernel_gen/run_example.py \
  --definition my_operator \
  --workspace runs/kernel_gen_kb_read_only/my_operator \
  --eval-server http://127.0.0.1:8000 \
  --target-hardware Ascend910B \
  --catalog-path /path/to/native-catalog \
  --knowledge-mode read_only_v1 \
  --knowledge-catalog-path /path/to/production-catalog \
  --knowledge-derived-path /path/to/disposable-derived/my_operator \
  --knowledge-reviewer-mode off \
  --n-parallel 2 \
  --n-epoch 1
```

未提供 `KnowledgeConfig` 或 `--knowledge-catalog-path` 时禁用 V1。仅提供 Catalog
而省略 `--knowledge-mode` 时，为保持兼容仍默认 `read_write_v1`；生产只读必须
显式传 `read_only_v1`。该模式要求 Reviewer 为 `off`，不会获取 Catalog lock、
写 run archive、发布 Concept/Solution 或修改 Catalog 内索引；workspace 本身仍会
写运行记录。若省略 `--knowledge-derived-path`，Catalog 自带 `.derived` 必须已经
存在且与 Catalog 同步，否则立即失败；外部 derived path 必须位于 Catalog 之外，
可按需重建且可以删除。

`read_write_v1` 可通过 `--knowledge-run-archive-path /path/to/run-archive`
指定不可变运行归档目录。旧 workspace state 中的 `legacy`、`shadow` 和
`read_v1` 不再受支持。

Reviewer mode 为 `off` 时不调用模型并延后所有 Runtime Candidate；`shadow` 会
记录审核决定但不合并 Catalog；`enforce` 只发布 Reviewer 批准的 Candidate。
同一个 epoch 的所有 Reviewer packet 复用 `<epoch>R/knowledge-reviewer`，为避免
共享 runtime log 和 retrieval audit 串包，packet 会串行执行；没有 Candidate 时
不会创建该目录。Reviewer 决策保存在独立 audit 中，只引用已经持久化的
Observation v2 ID。

### 2.1 区分固定知识与跨 epoch 经验

KernelGen 的 `--no-cross-epoch-knowledge` 关闭新 epoch 知识的复用及 synthesis 建议传递，但保留当前 run 内已测 best-code 的继承、共享 Analyzer、评测与 ledger。默认 `--cross-epoch-knowledge` 保持原行为。关闭后只能不配置 KB，或使用 `read_only_v1` 固定基础知识；可写 KB 会被拒绝，避免一边声明关闭一边读到本轮新增知识。该参数由 `kg run`、Python launcher 和 Batch YAML 的共享参数契约处理。

| 对照目的 | Knowledge 配置 | cross_epoch_knowledge |
|---|---|---|
| 无知识，仅保留本次 best-code 继承 | 不配置 KB | false |
| 固定基础知识，不传递 epoch 建议 | read_only_v1，外部 derived_root | false |
| 固定基础知识，传递 epoch 建议 | read_only_v1，外部 derived_root | true |
| 基础知识及本轮审核发布的新知识 | read_write_v1，reviewer=enforce | true |

关闭开关不禁用 epoch synthesis 的归档生成，只是不把其 next_directions 注入下一 epoch；因此可以保留相同的分析预算和审计证据。实验应为每组复制独立 Catalog，并固定初始代码策略、测试集、模型及搜索预算。`start_mode=fresh` 且不传 seed 表示从零开始；不要把历史 Solution fork 与知识效果混在一起。

当前推荐组合和协议兼容条件见 [`ONBOARDING.md`](../ONBOARDING.md) 的“安装”一节。KB workflow 与其他 KernelGen workflow 共用 v6.2 binding、EvaluationSettings 和 `/status` 目标环境校验，不维护单独的 Server 版本基线。

### 2.2 启动查询 KB 的 SimpleOpt

```bash
kg run --mode simple_opt \
  --definition-name flaggems_rsqrt \
  --workspace runs/simple_opt_kb/flaggems_rsqrt \
  --eval-server http://127.0.0.1:8000 \
  --target-hardware Ascend910B \
  --knowledge-catalog-path /path/to/catalog
```

提供 `--knowledge-catalog-path` 会复制 Knowledge Skill、materialize workspace state，并选择 Knowledge Coder/Profile/Distiller。产生的 CandidateDraft 留在 `stages/optimize/work`，不写公共 Catalog；因此不需要 run archive，也不支持 KernelGen 的 fork。中断任务使用 `kg resume <workspace>`，不改变知识快照。

## 3. 多算子 campaign 与 epoch 恢复

多个算子共享一个可写 Catalog 时，使用按 epoch 设置 barrier 的 campaign 入口：

```bash
python3 examples/kernel_gen/run_campaign.py \
  --definitions flaggems_rsqrt flaggems_gelu \
  --workspace-root runs/kernel_gen_campaign \
  --knowledge-mode read_write_v1 \
  --knowledge-catalog-path /path/to/catalog \
  --knowledge-run-archive-path /path/to/run-archive \
  --catalog-name flaggems-adapter-definitions \
  --eval-server http://127.0.0.1:8000 \
  --target-hardware Ascend910B \
  --max-operators 2 \
  --agents-per-operator 2 \
  --n-epoch 3
```

同一 epoch 的算子可以并发运行；只有写出全部 Agent ledger、
`publish-result.json`（状态为 `published` 或 `noop`）以及多 Agent
`synthesis.json` 的算子才能进入下一 epoch。某个算子 incomplete 时，其他已完成
算子继续。

如果 Agent 已结束但发布或 synthesis 失败，只重放收尾：

```bash
python3 examples/kernel_gen/run_example.py \
  --definition flaggems_rsqrt \
  --workspace runs/kernel_gen_campaign/flaggems_rsqrt \
  --knowledge-catalog-path /path/to/catalog \
  --eval-server http://127.0.0.1:8000 \
  --target-hardware Ascend910B \
  --n-parallel 2 \
  --n-epoch 1 \
  --finalize-epoch 1
```

该入口要求本 epoch 已有全部 Agent ledger，不会重新运行 Analyzer 或 Coder。
Publisher 通过 batch ID 幂等重放，已有合法 synthesis checkpoint 也会直接复用。

## 4. fresh、fork 与 resume

- `fresh` 创建不带公共 Solution 种子的全新运行；
- `fork` 只用于 Knowledge-enabled KernelGen，从 Solution Registry 读取
  Definition、benchmark、backend、architecture、device、language 完全一致且
  checksum 正确的 best code；
- `resume` 只续跑当前 workspace，要求 `--start-epoch > 1`，不查询公共 Solution。

benchmark identity 由 Catalog manifest 自动生成，例如 `flaggems-adapter-definitions` 在 API v6.2 下对应 `flaggems-adapter-definitions-v6.2`，无需单独传入。运行结束或 `--finalize-epoch` 收尾完成后，只有严格高于当前 slot `geo_mean` 的权威 best 才会替换 Solution；SimpleOpt 和 BatchSimpleOpt 不读取或发布该 Registry。

## 5. 运行时边界

Knowledge-enabled KernelGen 和显式 KB-enabled SimpleOpt 使用独立角色：

- `kernel-knowledge-coder`
- `kernel-knowledge-profile-analyzer`

默认 SimpleOpt 和 BatchSimpleOpt 使用：

- `kernel-coder`
- `kernel-profile-analyzer`

Python 实现仍共用一个 `CoderAgent`。Workflow 只选择不同的 native role，从而
隔离 KB prompt 和工具权限。

对于 KB-guided 的非 baseline 实验，Knowledge 角色按 K-08 使用同一问题查询
Concept 和 Source：Source 查询使用 `max_results=4`，最多读取两个片段，每个
不超过 200 行。这是 role/Skill 指导和 lineage 审计规则，不是 Python/MCP 的
eval 门禁。存在 TargetContext 时，匹配 `allowed_backends` 的 Source 获得优先级；
跨后端受限或 federated-only Source 不可查询和读取，可再分发的跨后端资料仅作为
降权类比，未声明 backend 的通用资料保持中性。

## 6. 检查知识使用记录

启用 V1 KB 后，workspace 的主要记录为：

```text
.kernelgen/knowledge/state.json
.kernelgen/knowledge/retrieval-log.jsonl
.kernelgen/knowledge/candidates.jsonl
.kernelgen/knowledge/publish-result.json
.ledger.json
```

`state.json` 是知识运行上下文，包含 Catalog 引用、mode、run/workspace ID、
TargetContext 和 OperatorSignature 引用。它不是优化生命周期状态。显式
KB-enabled SimpleOpt 会生成 state、查询/检索记录和本地 Candidate；只有完成
KernelGen epoch publication 的 workspace 才生成 `publish-result.json`。默认
SimpleOpt 和 BatchSimpleOpt 不生成这些文件。`read_only_v1` 仍写 workspace 内
这些运行记录，但其 `publish-result.json` 固定为 `status=noop`，不会写 Catalog。

`retrieval-log.jsonl` 是当前统一的查询和详情读取日志；旧 workspace 的
`query-log.jsonl` 仅用于兼容读取，新流程不再写入。

“检索到”不等于“使用”。只有实际影响 ExperimentPlan 的 Concept/Source 才写入
`knowledge_uses`；查询 lineage 缺失只影响审计结果，不阻塞 preflight、eval 或
`finalize_round`。

## 7. 维护入口

```bash
python3 -m kernelgen.tools.kb --help
python3 -m kernelgen.tools.kb validate --kb /path/to/catalog
python3 -m kernelgen.tools.kb rebuild \
  --kb /path/to/catalog \
  --run-archive /path/to/run-archive
```

Schema 和数据流见
[`knowledge_base_implementation.md`](../design/knowledge/knowledge_base_implementation.md)，
字段合同见
[`knowledge_concept_contract.md`](../design/knowledge/knowledge_concept_contract.md)。

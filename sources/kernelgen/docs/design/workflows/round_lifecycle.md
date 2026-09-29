# Round lifecycle and ledger schema v3

KernelGen 的 `.ledger.json` 使用 `schema_version: "3.0"`。数据模型以 jiabei
`2.6` 为基础，只叠加当前执行链路需要的 `candidate_path`、workload
`phase/skip_reason`、`api_version`、anti-hack、`timing_skipped`、evaluation
fingerprint 和 `next_verdict`。支持的 v2 ledger 只在内存中迁移，源文件不会被
覆盖；无法判断分支来源的同名 `2.0` 必须由调用方显式指定来源。

## Round 之前：preflight

`preflight_kernel` 是正式 round 的前置资格检查，但它本身不属于 ledger
round。KernelGen 先校验目标 backend，服务端再对每个 workload specialization
执行 compile/smoke。它不做静态源码策略、数值对拍或性能计时。

- `FAILED` 或 `ERROR`：Coder 修复当前候选并再次 preflight；不调用
  `finalize_round`，也不增加 plateau 计数。
- `PASSED`：保存一次性 receipt 和该候选的不可变 snapshot。Coder 不得在
  preflight 与 `eval_round` 之间修改 `tmp/main.py`。

receipt 绑定 kernel、Implementation、Definition、有序 Workload、target、
Catalog name、Server API version、backend 和计时策略。上述任意内容变化后，
`eval_round` 返回 `PREFLIGHT_REQUIRED`。receipt 在一次正式 eval 前被消费，
不能用于第二个 round。

preflight 诊断保存在：

```text
.kernelgen/preflight/
├── attempts.jsonl
├── candidates/<attempt-id>.py
├── receipt.json                 # 尚未消费的通过 receipt
└── last-consumed-receipt.json
```

## Workload 模式

v5 catalog 分别提供 correctness 与 timing Workload。服务端按请求顺序返回扁平
`per_workload`，其中 `uuid` 的值就是全局唯一的 `Workload.name`。profile 只允许
选择 timing workload，避免对只用于正确性覆盖的小 shape 做性能归因。

preflight receipt 和 immutable eval snapshot 都记录完整 workload 名称、内容
哈希以及允许 profile 的 workload。`ToolContext` 只保存 `catalog_name`，
不会保存本地 Catalog 路径或复制 workload。

## 每轮结构

```json
{
  "round_num": 2,
  "experiment_parent_round_num": 1,
  "plan": {},
  "solution": {},
  "evaluation": {},
  "profile": {},
  "conclusion": {},
  "next_verdict": {}
}
```

| 阶段 | 写入者 | 时机 | 核心内容 |
|---|---|---|---|
| `plan` | Coder 通过 `eval_round` 提交，Python 校验和冻结 | eval 之前 | 实验类型、策略、实际代码改动、可证伪假设、预期效果、idea source、参数和 Knowledge usage |
| `solution` | Python | eval 记录时和 snapshot 完成时 | 完整源码、源码 SHA-256、候选相对路径、immutable snapshot 相对路径 |
| `evaluation` | Python | eval 完成后 | 权威 aggregate、完整 per-workload candidate/reference latency、speedup、correctness、相对上一权威最佳 round 的差值、evaluation fingerprint |
| `profile` | Python 根据 ProfileAnalyzer 的已验证提交更新 | new best eval 后按需；最终 best 在 Distill 前 best effort 补齐 | required/status、analysis path、compact summary |
| `conclusion` | Coder 通过 `finalize_round` 提交，Python 校验 | eval 后；不等待 pending Profile | expectation status、root cause、perf gap、下一建议、优化层级、debug lesson 和 strategy evolution；`null` 表示尚未完成 |
| `next_verdict` | Python 在同一次 `finalize_round` 中计算并写入 | 与 conclusion 原子提交 | CONTINUE/STOP、稳定 reason code 和原因；eval 后、finalize 前与 conclusion 一同为 `null` |

## 顺序和所有权

`eval_round` 的 `experiment_plan` 是必填参数，并且当前候选必须已经取得有效
preflight receipt。新 workspace 的第一轮必须使用 `kind: "baseline"`、
`source.origin: "baseline"` 和
`expected_effect.direction: "establish_baseline"`；后续轮不能伪装成
baseline。模型在同一次 tool call 中先提交 plan，测量结果随后才返回，因此
plan 无法写入事后解释。

当 plan 来源是 `profile_next_experiment` 时，必须给出
`source.parent_round_num`。Ledger 从该 round 的 ProfileRecord 解析
`analysis_path`，并校验 profile 状态、文件和 `next_experiment`，避免在 plan
中重复保存路径或声称采用了不存在的 profile 建议。

`solution` 与 `evaluation` 完全由 Python 写入。浮点数按服务端返回值原样存储，不做小数位截断。每个 workload 保存 `latency_ms`、`reference_latency_ms` 和 `speedup`；`evaluation.comparison` 使用新 round 写入前的权威 best round 作为 baseline，记录 aggregate geo_mean 变化和每个 workload 的 latency/speedup 绝对及百分比变化。

ProfileAnalyzer 不直接编辑 ledger。它只能读取 immutable eval context、采集 artifacts，并通过 `record_profile_analysis` 提交后端中立 schema。工具校验 fingerprint、solution identity、workload UUID、profile ID、manifest 和 artifact path，再写 analysis 文件并更新 `profile`。analysis 中的 eval latency、reference latency、speedup 和 axes 由 Python 从 snapshot 回填。

`profile_enabled` 是总开关。Python 只在 `PASSED`、非 hack 且刷新
`best_geo_mean` 的 round 上写 `profile.required=true`；普通 passing non-best
round 为 `not_required`。Coder 根据 Profile 是否能区分当前性能解释来决定是否
立即调用 Analyzer。pending/collecting 是可审计的待办状态，不阻塞 conclusion、
candidate transition、STOP 或下一轮。Coder 收尾后，Workflow 只检查最终 best：
若仍 pending/collecting，则做一次 best-effort Profile；Analyzer 失败会尽量写成
terminal `failed`，但不会使有效优化结果失败。

`finalize_round` 只接受 `RoundConclusion`，其 schema `extra="forbid"`，因此不能夹带或覆盖测量字段。baseline 的 `expectation_status` 必须是 `baseline`；后续 round 不能使用 `baseline`，且必须提供非空 `perf_gap_analysis`。校验通过后，Python 在一次 ledger 原子写入中同时保存 conclusion 和 stop-policy verdict，并按 `best_round` 对候选执行 KEEP、REVERT 或 REPAIR 转换，再把转换结果和 verdict 直接返回给 Coder。不再保存 `narrative_recorded`，因为 `conclusion == null` 已经无歧义地表达 pending 状态。

Server 返回的 `is_hack` 和 `hack_reason` 会原样写入 evaluation。`is_hack=true`
的 measured round 仍允许用于调试和结论记录，但无论其 `status` 或
`geo_mean` 如何都不能刷新 best 或覆盖 `.best_kernel.py`。存在历史 best 时，
`finalize_round` 会把候选回退到该 best；没有历史 best 时保留候选供下一轮修复。

## 停止计数

`rounds_without_improvement` 只表示“已经完整 `PASSED`，但没有刷新
`best_geo_mean`”的性能平台期。`INCORRECT_NUMERICAL`、编译失败、运行失败和
部分通过仍会记录为 measured round，供后续诊断使用，但不会消耗
`early_stop_rounds`。因此在获得第一个正确候选之前，系统不会把连续纠错误判
成性能已经收敛。该计数是派生状态，加载 ledger 时会按 round 历史重新计算，
以修复旧版本写入的失败轮次计数。

`max_round` 默认 15，统计所有已经记账的 measured round，不区分 `PASSED`、
数值错误、编译错误或运行错误。preflight 失败和未记账的
transport/config/hardware mismatch 不计入。达到上限是硬 STOP，不受
`min_rounds` 或 `soft_stop_disabled` 影响。

## 状态机

```text
edit candidate
  -> preflight_kernel
       -> failed: edit and retry (no round)
       -> passed: do not edit candidate
  -> eval_round(experiment_plan; consumes receipt)
  -> Python writes plan + solution + evaluation
  -> if new best and diagnosis is useful:
       Agent(kernel-profile-analyzer)
       -> Python writes terminal profile state
  -> finalize_round(conclusion)
       -> Python atomically persists conclusion + next_verdict
       -> Python applies KEEP / REVERT / REPAIR to the candidate
       -> CONTINUE: next candidate is eligible
       -> STOP: preflight/eval remain blocked
  -> before Distill: best-effort Profile of final best if still pending
```

`preflight_kernel` 和 `eval_round` 继续检查 pending conclusion；`finalize_round`
仍是每轮必需的原子 conclusion/verdict/candidate transition。pending Profile
只作为 recommendation，不阻塞这些工具或 stop policy。此外，`eval_round`
单独检查当前精确候选的一次性 preflight receipt。

`finalize_round` 返回 CONTINUE 时解除下一轮门禁；返回 STOP 后，
preflight/eval 返回 `RUN_STOPPED`，不实际执行。不存在独立的 `next` MCP 或
CLI；Coder 直接使用 `finalize_round` 返回的已持久化 verdict。
`SingleCoderOptimizationWorkflow` 收尾要求最新 verdict 为 STOP，因此 Coder 在 CONTINUE 后提前返回不能直接形成有效 workflow 结果。Claude Runtime 已取得 session ID 时，Workflow 会在同一个 Coder 对话中继续执行；如果 Runtime 不支持 resume 或没有取得 session ID，则直接失败，不会使用 ledger 摘要启动新的 Coder context。续接后若没有新增 measured round，或续接次数达到 `max_coder_sessions`，Workflow 只在已经满足 `min_rounds` 且存在权威 PASSED best 时写入 `coder_no_progress` 或 `coder_session_limit_reached` STOP，并保留该 best 进入收尾；否则仍然失败。这个 supervisor STOP 是 Python 持久化的终态，不会把 Debug Job 当作 measured round，也不会接受缺少有效 best 的提前返回。

## 跨 agent 使用

每个并行 Coder 保持独立 ledger。Distiller 按
`plan → solution → evaluation → profile → conclusion → next_verdict` 重建
trajectory；非 best round 允许 `profile=not_required`，被后续 best 取代的旧
round 也可能保留 pending。Distiller 提取被验证、部分验证、证伪或不可评估的假设，并输出
`.new_experience.md` 与 `.new_detailed.md` 候选。

epoch reducer 以 ledger 校验并排序候选，再统一写 canonical KB。EpochSummary
使用有界、无源码的比较 trajectory 横向分析各 agent 的预期增益、实际
aggregate/per-workload latency、speedup 和 perf gap；全局最佳 kernel 只作为
一份代码锚点传入。种子仍由 Python 的权威 `geo_mean` 选择，模型只负责解释
和设计互不重复的下一批方向。

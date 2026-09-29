# ProfileAnalyzeAgent 设计与实现

> 状态：Implemented
>
> 日期：2026-07-22
>
> 依赖：`kernelgen_server` 的统一 eval/profile service

## 1. 背景

KernelGen 当前的 `kernel-coder` 在一次 Claude Code 长会话中执行完整优化内循环：修改 `tmp/main.py`、调用 `preflight_kernel` 直到当前精确候选通过、调用 `eval_round`，再调用 `finalize_round` 原子保存 conclusion 和 Python stop-policy verdict，并由 Python 自动 KEEP/REVERT/REPAIR，直到 verdict 返回停止。

`kernelgen_server` 已经提供 backend-neutral 的 profile、artifact 和 status
接口。CUDA 由 NCU 产生 `.ncu-rep`、`ncu-details.txt`、`sass.json` 等 artifact；
Ascend 由 msprof 和 simulator 产生 `msprof-details.txt`、
`ascend-instructions.json`、源码映射和 vendor archive。两种后端通过统一的
`ProfileResult`、`capabilities`、`metrics` 和 `artifacts` 暴露，不需要
KernelGen 直接执行或解析 vendor profiler 命令。

ProfileAnalyzeAgent、可信 MCP 工具、ledger 状态和 final-best Workflow
收尾链路均已实现。

## 2. 目标

1. ProfileAnalyzeAgent 必须是 Claude Code 原生 subagent，而不是 Python 启动的第二个独立 Claude 会话。
2. Coder 的主会话和上下文必须持续存在；profile 原始报告和长分析留在 subagent 的独立上下文中，只把紧凑结论返回 Coder。
3. Python 根据权威 eval 决定该轮是否具备 Profile 资格：只标记非 hack 的 new
   best；Coder 根据诊断价值决定是否立即执行。
4. pending Profile 是可审计待办，不阻塞 conclusion、candidate transition、
   STOP 或下一轮；Workflow 在 Distill 前 best effort 补最终 best。
5. ProfileAnalyzeAgent 可以选择一个或多个 workload，不设置 workload 数量上限，并根据证据充分性决定是否继续采集。
6. ProfileAnalyzeAgent 输出必须区分直接证据、推断和下一步实验。
7. CUDA、Ascend 和未来后端共享同一 Agent、MCP 工具和数据契约；backend 差异只存在于 `ProfileResult` 的 capabilities、metrics 和 artifacts 中。
8. eval latency 和 correctness 始终是性能与正确性的权威真值；profiler timing 仅用于诊断。

## 3. 非目标

1. 不在 KernelGen 中重新实现 NCU、msprof、simulator 或 artifact 下载逻辑。
2. 不把 ProfileAnalyzeAgent 改成外层 `Workflow` 顺序调用的独立 LLM session。
3. 不通过 Claude hooks 启动 subagent；hooks 不是本设计的权威状态机。
4. 不让 ProfileAnalyzeAgent 修改 kernel、调用 eval、决定 KEEP/REVERT 或决定停止。
5. 不恢复旧 `primary`、`repr` 或 representative evaluation 逻辑；eval 结果覆盖完整 workload 集合。
6. 不为 profile workload 数量设置固定上限。
7. 第一版不实现 Codex provider 的 native agent 定义，但 Python 数据契约和 MCP 工具必须保持 provider-neutral。

## 4. 核心决策

### 4.1 原生 subagent，Python 管理资格和可信状态

`kernel-coder` 继续由 `claude --agent kernel-coder` 作为 Claude Code 主线程运行。
当 `eval_round` 对 new best 返回 `profile_required=true`，且 Profile 证据能区分
竞争解释时，Coder 在当前会话中调用原生 `Agent(kernel-profile-analyzer)`。

ProfileAnalyzeAgent 使用独立上下文，不继承 Coder 的完整对话。它只接收 `round_num` 和任务要求，然后通过 MCP 读取权威 round snapshot、完整 eval 结果和 profile capabilities。subagent 完成后，Claude Code 把它的紧凑返回值作为 Agent tool result 放回 Coder 的原上下文，Coder 会话不会重建或重置。

Python 在内循环中不启动第二个 LLM session，只保存
`profile_required/profile_status` 和已验证 analysis。Coder 未立即执行时可以继续
`finalize_round` 和后续 round。Coder 结束后，Workflow 可使用同一 Runtime 显式
调用 native Profile Analyzer，确保最终 best 得到一次 best-effort 分析。

### 4.2 不依赖自动 delegation

Agent frontmatter 的 `description` 有助于 Claude 选择 subagent，但它不是调用
保证。是否立即执行由以下两层共同控制：

1. `kernel-coder` prompt 要求只在 evidence 能指导下一实验时调用
   `Agent(kernel-profile-analyzer)`，并且只传 `round_num`。
2. `SingleCoderOptimizationWorkflow` 在 Distill 前检查最终 best；若仍是
   `pending/collecting`，则调用 Analyzer 并在失败时尽量记录 terminal evidence。

Claude Code 2.1.198 起默认把 subagent 放到后台，因此实际调用仍要求
`run_in_background=false`，并由 `ClaudeRuntime` 设置
`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`。这是为了避免 Profile 与候选修改并发，
不是 lifecycle gate。

### 4.3 ProfileAnalyzeAgent 独占 profile 工具

Coder 有 `preflight_kernel`、`eval_round`、`finalize_round` 和
`Agent(kernel-profile-analyzer)`，但没有 `get_profile_context`、
`profile_workloads` 或 `record_profile_analysis`。ProfileAnalyzeAgent 有这
三个 profile MCP 工具，但没有 `Write`、`Edit`、`Bash`、
`preflight_kernel`、`eval_round` 或 `finalize_round`。

因此 Coder 不能自行伪造 profile analysis，也不能绕过 subagent 直接清除 pending 状态。

## 5. 总体架构

```text
Claude Code main thread: kernel-coder
  |
  | edit tmp/main.py
  v
mcp__kernelgen__preflight_kernel
  |
  |-- FAILED/ERROR --> edit and retry (no measured round)
  |
  `-- PASSED (do not edit candidate)
          |
          v
mcp__kernelgen__eval_round
  |
  |-- non-PASSED -----------------------------> finalize_round(finalize)
  |
  `-- PASSED + new best + profile_required=true
          |
          | if diagnosis is useful
          v
      Agent(kernel-profile-analyzer)  [native subagent, isolated context]
          |
          |-- get_profile_context(round_num)
          |-- select one or more workloads
          |-- profile_workloads(round_num, workload_uuids, ...)
          |-- Read normalized local artifacts
          `-- record_profile_analysis(round_num, analysis)
                    |
                    v
          compact Agent result returns to kernel-coder
                    |
                    v
          finalize_round(auto KEEP/REVERT/REPAIR + finalize)

Coder stop
  -> Workflow profiles final best if it is still pending
  -> Distiller
```

外层 `KernelGenWorkflow` 和 `SingleCoderOptimizationWorkflow` 不拆分 Coder 的长会话。它们只负责写入包含 profile policy 的可信 `ToolContext`。

## 6. 触发策略

### 6.1 第一版触发条件

当前采用简单、确定且容易验证的资格策略：

```python
profile_required = (
    context.profile_enabled
    and status == "PASSED"
    and not is_hack
    and geo_mean is not None
    and geo_mean > previous_best
)
```

相同 evaluation fingerprint 已有 terminal analysis 时，snapshot 绑定阶段会复用
已有结果并把当前 round 改为不再 required。terminal 状态包括 `completed`、
`inconclusive`、`unsupported` 和 `failed`；`pending/collecting` 表示仍可分析，
但不会阻塞 lifecycle。

### 6.2 不触发的情况

- `PARTIAL_PASS`
- `COMPILE_ERROR`
- `RUNTIME_ERROR`
- `INCORRECT_NUMERICAL`
- `INCORRECT_SHAPE`
- `INCORRECT_DTYPE`
- `TIMEOUT`
- transport/config/hardware mismatch
- `profile_enabled=false`
- `PASSED` 但没有刷新 best
- `is_hack=true`
- 相同 evaluation fingerprint 已有 terminal analysis

`KernelGenInput.profile_enabled` 默认是 `true`；示例入口提供 `--no-profile`，
用于没有 profiler 的调试环境显式关闭该阶段。`SimpleOptInput` 与底层
`CoderInput` 默认是 `false`，因此只有 KernelGen 默认启用 profile。

### 6.3 Evaluation fingerprint

Fingerprint 用于阻止对完全相同的评测对象重复采集，至少包含：

```text
solution_sha256
definition_sha256
ordered workload UUID + workload_sha256 集合
catalog_name + target_hardware + server_backend
eval schema version
profile analysis schema version
```

ProfileAnalyzeAgent 仍然可以被调用来读取已有分析，但 `profile_workloads` 应优先复用完全匹配的本地 manifest，而不是重复占用设备。

### 6.4 后续可选策略

未来可以增加 `all_passed`、`new_best`、`plateau` 等策略，但第一版只实现 `all_passed`。先保证控制链正确，再根据真实 profile 开销决定是否增加策略复杂度。

## 7. Provider-neutral Agent 定义

新增 `.kernelgen/agents/kernel-profile-analyzer.md`，运行时再生成 Claude/Codex provider 文件：

```yaml
---
name: kernel-profile-analyzer
description: Analyze an authoritative fully-passing kernel evaluation, select representative workloads, collect backend-native profiles, and record an evidence-backed next experiment. Use whenever eval_round returns profile_required=true.
capabilities: [read, search]
mcp_tools: [get_profile_context, profile_workloads, record_profile_analysis]
subagents: []
model: inherit
approval: no_prompts
---
```

Agent body 只包含稳定职责和决策规则：

1. 先调用 `get_profile_context`，不信任 delegation prompt 中复制的测量数字。
2. 基于完整 `per_workload` 选择 workload，并记录每个选择原因。
3. 先使用 `metrics` level；只有当下一步代码修改需要源码或指令定位时，才升级到 `source` 或 `instruction`。
4. 优先读取 `ncu-details.txt`、`msprof-details.txt`、`sass.json`、`ascend-instructions.json` 等本地可读 artifact；不要求读取 vendor binary archive。
5. 每个结论使用 `evidence -> inference -> optimization action`。
6. 每个 evidence 必须携带 artifact 路径和可用的 metric/line/PC/kernel locator。
7. 只返回一个本轮最推荐的 `next_experiment`，避免 Coder 在一轮混合多个无法归因的改动。
8. profile unsupported、failed 或 inconclusive 时仍必须调用 `record_profile_analysis`。
9. 不修改 kernel，不决定 KEEP/REVERT，不调用
   preflight/eval/finalize。

## 8. Coder 集成

### 8.1 Tool allowlist

`kernel-coder` frontmatter 包含：

```yaml
mcp__kernelgen__preflight_kernel, mcp__kernelgen__eval_round,
mcp__kernelgen__finalize_round,
Agent(kernel-profile-analyzer)
```

Coder 不增加三个 profile MCP 工具。

`ClaudeRuntime.allowed_tools` 同时增加 `Agent(kernel-profile-analyzer)`，否则 CLI 全局 allowlist 会阻止 frontmatter 中的 Agent tool。

### 8.2 内循环执行顺序

Coder 的单轮顺序调整为：

```text
edit
-> preflight_kernel
-> if failed: fix and repeat without creating a round
-> eval_round
-> if profile_required and diagnosis is useful:
     Agent(kernel-profile-analyzer)
     verify record_profile_analysis returned recorded=true
-> finalize_round (auto candidate transition + persisted CONTINUE/STOP)
```

Coder prompt 明确：不要 Profile 每个 passing candidate；只有 new best 会请求，
并由 Agent 判断证据能否指导下一实验。已经调用 Analyzer 时必须前台等待
`recorded=true`；跳过或失败不阻止 `finalize_round`。

同样，`preflight_kernel` 返回 `PASSED` 后到紧接着的 `eval_round` 之间不得编辑
`tmp/main.py`。通过 receipt 绑定精确代码、Catalog、Server API、backend、
timing 和 target device；代码或这些服务身份字段变化都会令 `eval_round` 返回
`PREFLIGHT_REQUIRED`。preflight 失败不创建 ledger round，也不进入本节描述的
profile 阶段。

Profile 工具从 immutable round snapshot 读取 candidate，而不是读取可能已被 Coder 恢复过的 `tmp/main.py`，从而保证 profile 对象与 eval 对象一致。

## 9. Eval snapshot

`eval_round` 在权威测量落入 ledger 的同时保存不可变 snapshot：

```text
.kernelgen/evals/round-0004/
  main.py
  result.json
  solution.json
  definition.json
  workloads.json
  identity.json
```

Snapshot 保存实际提交给 eval 的完整 `Implementation`、`Definition` 和有序
`Workload` 集合，不能在 profile 阶段仅根据扁平 `per_workload` 结果猜测或重建。
`kernelgen.tools.kernelgen_server_adapter.prepare_evaluation_bundle()` 是
preflight、eval 和 snapshot 共用的唯一构造入口。

`identity.json` 至少包含：

```json
{
  "round_num": 4,
  "evaluation_fingerprint": "...",
  "solution_sha256": "...",
  "definition_name": "...",
  "definition_sha256": "...",
  "workload_uuids": ["..."],
  "catalog_name": "flaggems-v5",
  "target_hardware": "...",
  "server_backend": "cuda"
}
```

Snapshot 由 Python 写入，Agent 只读。`profile_workloads` 不接受任意 kernel code 或绝对路径，只接受 `round_num` 和 workload UUID。

## 10. MCP 工具

三个工具注册到现有 `kernelgen` stdio MCP server，不启动新的 server。

### 10.1 `get_profile_context`

输入：

```json
{"round_num": 4}
```

输出：

```json
{
  "round_num": 4,
  "evaluation_fingerprint": "...",
  "eval_status": "PASSED",
  "solution_sha256": "...",
  "candidate_path": ".kernelgen/evals/round-0004/main.py",
  "per_workload": [
    {
      "uuid": "...",
      "axes": {"M": 4096},
      "status": "PASSED",
      "speedup": 0.82,
      "latency_ms": 0.31
    }
  ],
  "service": {
    "backend": "cuda",
    "profile": {
      "supported": true,
      "profiler": "ncu",
      "levels": ["metrics", "source", "instruction"]
    }
  },
  "profile_state": "pending",
  "cached_profiles": []
}
```

职责：读取可信 snapshot 和 `/status`，不执行 profile，不生成优化建议。

### 10.2 `profile_workloads`

输入：

```json
{
  "round_num": 4,
  "workload_uuids": ["uuid-a", "uuid-b"],
  "level": "metrics",
  "backend_options": {}
}
```

职责：

1. 校验 round 存在、eval 为 `PASSED`、profile state 为 `pending/collecting`。
2. 从 immutable eval snapshot 读取代码和按 UUID 固化的完整 workload。
3. 重建与 eval 一致的 `Solution`、`Definition` 和 `Workload`。
4. 对每个 workload 调用 KernelGen Server profile client。
5. 让 client 下载并校验全部 artifacts。
6. 在调用 client 前生成本地 `request-id`，把输出写到 `.kernelgen/profiles/round-0004/<workload-uuid>/<request-id>/`；server 返回的 `profile-id` 保存在 manifest 中。
7. 返回紧凑 result、manifest 路径和本地 artifact 路径，不把完整报告内联进 tool result。

服务端 `/profile` 一次只处理一个精确 workload，因此 MCP 工具内部逐个调用。接口不设置 `workload_uuids` 数量上限；是否继续采集由 Agent 根据证据决定。

### 10.3 `record_profile_analysis`

输入为完整 `ProfileAnalysis`。工具通过 Pydantic 校验、验证引用的 profile IDs 和 artifact paths 确实属于当前 round、原子写入 analysis 文件并更新 ledger。

只有该工具可以把 `profile.status` 从 `pending/collecting` 改为 terminal status。

## 11. ProfileAnalysis 数据契约

```json
{
  "schema_version": "1.0",
  "round_num": 4,
  "evaluation_fingerprint": "...",
  "status": "completed",
  "solution_sha256": "...",
  "backend": "cuda",
  "profiler": "ncu",
  "dominant_bound": "memory",
  "profiled_workloads": [
    {
      "uuid": "...",
      "axes": {"M": 4096},
      "eval_speedup": 0.82,
      "eval_latency_ms": 0.31,
      "selection_reason": "worst speedup in the large-M regime",
      "profile_ids": ["..."],
      "manifest_paths": [".../manifest.json"],
      "status": "completed"
    }
  ],
  "findings": [
    {
      "category": "memory_latency",
      "label": "long scoreboard stalls",
      "backend_detail": "ncu:latency-bound",
      "workload_uuids": ["..."],
      "confidence": "high",
      "evidence": [
        {
          "metric": "long_scoreboard",
          "value": 4.2,
          "unit": "ratio",
          "artifact_path": ".../ncu-details.txt",
          "artifact_kind": "agent_report",
          "locator": {
            "kernel": "kernel_name",
            "source_file": "main.py",
            "source_line": 37,
            "pc": "0x..."
          }
        }
      ],
      "inference": "load-to-use latency prevents issue progress"
    }
  ],
  "performance_interpretation": "The large-M workload regresses because additional parallelism does not hide memory latency.",
  "next_experiment": {
    "action_category": "memory_pipeline",
    "action_description": "Increase independent loads per iteration without changing tile shape.",
    "expected_impact": "Reduce long-scoreboard stalls on the selected workload.",
    "risks_and_rollback": "May increase register pressure; revert if occupancy or full-set geo_mean regresses.",
    "validation_workloads": ["..."],
    "success_criteria": [
      "full eval remains PASSED",
      "geo_mean improves",
      "selected workload speedup improves",
      "no new register spill evidence"
    ]
  },
  "warnings": [],
  "open_questions": []
}
```

### 11.1 Status

```text
completed
inconclusive
unsupported
failed
```

`completed` 至少要求一个 profiled workload、一个 evidence-backed finding 和一个 next experiment。`inconclusive` 要求说明已采集内容和无法得出结论的原因。`unsupported` 要求保存 `/status` capabilities。`failed` 要求保存错误和已成功下载的部分 artifact。

### 11.2 Dominant bound

```text
compute
memory
latency
launch
mixed
unknown
```

`dominant_bound` 是跨后端粗分类；具体问题放在 `findings[].category`，避免旧设计把 bound、症状、根因和优化手段混入同一个 `ncu_bottleneck` 枚举。

### 11.3 Finding category

```text
launch_parallelism
memory_bandwidth
memory_latency
memory_access_efficiency
cache_behavior
resource_pressure
compute_pipeline
synchronization
control_divergence
instruction_efficiency
unknown
```

CUDA 的 register spill、long scoreboard、tensor instruction absence，Ascend 的 MTE2/MTE3 dominance、cube underutilization、scalar overhead 等保存在 `label` 和 `backend_detail`，不扩散到公共顶层 schema。

### 11.4 Evidence 原则

- `value` 必须来自 `ProfileResult.metrics` 或已下载 artifact。
- `artifact_path` 必须是 client 工作区内已校验文件。
- `locator` 可以为空，但只要报告提供 kernel、source line、PC 或 instruction mapping 就应记录。
- `confidence` 仅允许 `high/medium/low`。
- `inference` 是 Agent 推断，不能写进 evidence。
- 不保存重复的 `ncu_key_metrics` 字符串；紧凑展示由 renderer 从 evidence 生成。

## 12. Workload 选择

ProfileAnalyzeAgent 从完整 eval `per_workload` 开始，不使用旧的 primary/repr 概念。

首轮优先考虑：

1. 最低 speedup workload。
2. 绝对 latency 最大 workload。
3. axes 或 speedup 出现明显突变的边界 workload。
4. 不同性能区间中能检验不同瓶颈假设的代表 workload。
5. 必要时选择一个表现较好的对照 workload。

Agent 可以在读取第一批报告后继续选择更多 workload。停止条件不是数量，而是：已有 evidence 可以解释所有重要性能区间，新增 workload 不再改变 dominant finding，或者 profiler 已无法提供更强证据。

如果同一 round 的不同 workload 呈现不同 bottleneck，`dominant_bound` 使用 `mixed`，每个 finding 用 `workload_uuids` 明确作用范围，不能把一个 workload 的结论推广到全集。

## 13. Profile level 策略

第一步默认使用 `metrics`，以较低开销获得 performance counters 和 backend report。

当某个 finding 将直接驱动下一步源码修改，并且 service capabilities 支持时，Agent 应升级：

- `source`：需要定位到源码行或 source hotspot。
- `instruction`：需要确认 SASS、Ascend simulator instruction、load/store width、spill、tensor/cube 指令或 pipeline mapping。

不要求每个 workload 都采集 instruction level，也不禁止为多个 workload 采集。每次升级必须在 `selection_reason` 或 finding 中说明目的。

## 14. Ledger 状态机与门禁

每个 round 的 profile 阶段位于 ledger schema v3 的嵌套 `profile` 对象中；整体 schema 见 [round_lifecycle.md](workflows/round_lifecycle.md)：

```text
profile.required: bool
profile.status: not_required | pending | collecting | completed | inconclusive | unsupported | failed
profile.analysis_path: str
profile.summary: str
evaluation.fingerprint: str
conclusion: RoundConclusion | null
```

状态转移：

```text
preflight PASSED -> one-time receipt outside ledger
eval non-PASSED -> not_required
eval PASSED + new best -> pending
eval PASSED + non-best/hack -> not_required
first profile_workloads call -> collecting
record_profile_analysis -> completed | inconclusive | unsupported | failed
finalize_round -> atomically writes conclusion + next_verdict
```

### 14.1 `preflight_kernel` / `eval_round` 门禁

两个工具继续检查上一轮 conclusion，但不检查 pending Profile。当
`conclusion=null` 时返回：

```json
{
  "status": "ROUND_CONCLUSION_REQUIRED",
  "round_num": 4,
  "required_tool": "finalize_round"
}
```

该门禁在解析候选路径和调用 KernelGen Server 之前执行，因此不会产生额外评测或
新 round。

上一轮 conclusion 和 verdict 完成后，`preflight_kernel` 才会调用服务端
preflight。
`eval_round` 还会验证当前候选的一次性 receipt；receipt 缺失、已消费、代码
变化或 Server API/backend/timing/target identity 变化时返回：

```json
{
  "status": "PREFLIGHT_REQUIRED",
  "required_tool": "preflight_kernel"
}
```

该返回同样不创建新 round。

### 14.2 `finalize_round` 门禁

pending/collecting Profile 不阻止 `finalize_round`。Profile analysis 由
`record_profile_analysis` 独立落盘，`finalize_round` 不接受模型复制的 profile
metrics。成功调用后，`conclusion` 和 Python 计算的 `next_verdict` 在同一次
ledger 原子写入中落盘，候选按权威 best 自动 KEEP/REVERT/REPAIR，并直接返回
转换结果、CONTINUE/STOP，以及“Profile final best if useful”的非阻塞
recommendation；同一 round 的第二次写入返回
`ROUND_CONCLUSION_ALREADY_RECORDED`。

### 14.3 Verdict 的单一出口

不存在独立的 `next` MCP 或 CLI。`finalize_round` 原子持久化 conclusion 和
verdict，并在同一次调用中返回 CONTINUE/STOP。pending Profile 不阻塞该过程，
只通过返回值中的 `recommendations` 提示最终 best 可按需 Profile。

### 14.4 正常路径的门禁边界

`finalize_round` 把 conclusion 与 verdict 原子变成终态；下一次
`preflight_kernel` 与 `eval_round` 要求上一轮已有 conclusion 和 CONTINUE
verdict，并在 STOP 后返回 `RUN_STOPPED`。它们不检查 pending Profile。
`eval_round` 再增加 preflight receipt gate。提示词负责描述正确顺序，Python
状态机负责强制 conclusion、verdict、STOP 和 receipt；Profile 是否立即执行由
Agent 判断，最终 best 由 Workflow 收尾。

## 15. 本地文件布局

```text
agent-workspace/
  .kernelgen/
    tool-context.json
    preflight/
      attempts.jsonl
      candidates/
      receipt.json | last-consumed-receipt.json
    evals/
      round-0004/
        main.py
        result.json
        solution.json
        definition.json
        workloads.json
        identity.json
    profiles/
      round-0004/
        <workload-uuid>/
          <request-id>/
            manifest.json
            ncu-details.txt | msprof-details.txt
            sass.json | ascend-instructions.json
            <vendor artifacts>
    profile-analysis/
      round-0004.json
```

所有路径都在当前 isolated workspace 内。MCP handler 使用既有 workspace path guard，拒绝绝对路径和目录逃逸。server 端 artifact 生命周期不影响已下载的 client 文件。

## 16. 与 KernelGen Server 的边界

KernelGen 通过 `kernelgen.tools.kernelgen_server_adapter` 统一构造请求，并调用
`kernelgen_server.client` 的公开 status、preflight、evaluate 和 profile 接口。
Agent 和 MCP 不直接拼装 Server schema。

KernelGen 负责：

- round snapshot 与 evaluation fingerprint
- workload 选择的 Agent 上下文
- 多 workload 的逐次 client 调用
- profile analysis schema 和 ledger 状态
- Coder/subagent 工具权限

KernelGen Server 负责：

- backend/device capability detection
- ProfileTarget 内容哈希校验
- vendor profiler 执行
- normalized metrics/report 生成
- artifact 注册、下载、size/SHA-256 校验
- CUDA/Ascend backend 差异

KernelGen 不检查 `target_hardware` 字符串来决定 NCU 或 msprof，也不直接解析 server 本地路径。

## 17. 失败与降级

| 场景 | ProfileAnalyzeAgent 行为 | 是否阻塞 lifecycle |
| --- | --- | --- |
| `/status` 返回 unsupported | 记录 `unsupported` 和 capabilities | 否 |
| `/status` 不可达 | 尽量记录 `failed` 和 transport error | 否 |
| 部分 workload profile 成功 | 保留已下载 artifact，记录 `inconclusive` 或 `completed` 并附 warnings | 否 |
| 所有 workload profile 失败 | 记录 `failed` | 否 |
| normalized artifact 缺失但 raw report 存在 | 读取可用 metrics，记录 warning | 否 |
| subagent 未调用 `record_profile_analysis` | 状态保持 pending/collecting，最终 best 由 Workflow best effort 重试 | 否 |
| subagent 启动失败 | Coder 仍可完成 `finalize_round`，最终 best 由 Workflow best effort 重试 | 否 |
| analysis schema 校验失败 | MCP 返回字段错误，subagent 可修复后重试 | 否 |

Profile 是可审计、可补偿的诊断待办，不是 conclusion、candidate transition 或
STOP 的门禁；terminal failure 也不能伪装成成功。

## 18. 历史字段清理

本次实现同步删除 backend-specific narrative 字段：

```text
ncu_analysis
ncu_bottleneck
ncu_key_metrics
```

替代方式：

- 完整分析存入 `profile-analysis/round-N.json`。
- ledger 的嵌套 `profile` 只保存 status、analysis path 和可渲染的紧凑摘要。
- raw/normalized metrics 保存在 profile manifest 和 artifacts 中。
- `key_numbers` 如果仍被其他流程使用，只保留非 profile 的通用诊断数字；profile 数字通过 evidence 引用，不再复制。

不提供旧 ledger 文件迁移逻辑。

## 19. 实现文件

新增：

```text
.kernelgen/agents/kernel-profile-analyzer.md
data/profile_analysis.py
tools/profile_round.py
tests/test_profile_analysis.py
tests/test_profile_round.py
```

修改：

```text
.kernelgen/agents/kernel-coder.md
framework/runtime/claude.py
mcp_server/server.py
mcp_server/contract.py
data/tool_context.py
tools/preflight.py
tools/eval_round.py
tools/finalize_round.py
data/ledger.py
data/optimization_history.py
data/round_conclusion.py
workflows/optimization/single_coder/profiling.py
workflows/optimization/kernelgen/
agents/coder/__init__.py
tests/test_mcp_server.py
tests/test_cli_runtime.py
tests/test_inner_loop_flow.py
tests/test_ledger.py
tests/test_finalize_round.py
examples/kernel_gen/run_example.py
DESIGN.md
```

`data/profile_analysis.py` 定义 provider-neutral Pydantic contract，供 MCP 校验和未来非 Claude runtime 使用。第一版 Claude 路径不由 Python 直接调用该 Agent class。

## 20. 测试计划

### 20.1 Host unit tests

- `PASSED` eval 创建 `pending` profile state。
- 非 PASSED eval 创建 `not_required`。
- 相同 fingerprint 的 terminal analysis 不重复 pending。
- snapshot 保存 eval 时的代码，即使 `tmp/main.py` 后续被恢复或修改。
- `profile_workloads` 拒绝未知 round、非 PASSED round、未知 workload UUID、绝对路径和目录逃逸。
- 多 workload 输入不做数量截断。
- `record_profile_analysis` 校验 profile ID、artifact path、solution hash 和 schema。
- completed analysis 缺少 evidence 或 next experiment 时被拒绝。
- failed/unsupported analysis 可以没有 finding，但必须包含 error/capability evidence。
- terminal status 原子完成 evidence 状态，不影响 lifecycle gate。

### 20.2 Gate integration tests

- `eval_round` 没有当前有效 preflight receipt 时返回 `PREFLIGHT_REQUIRED`，
  不执行测量、不新增 round。
- preflight 通过后修改代码、修改上下文或重启 service 会使 receipt 失效；
  receipt 在一次 eval 前被消费。
- pending Profile 时 `finalize_round` 仍能原子完成 conclusion/verdict/candidate
  transition，并在返回值中携带 advisory recommendation。
- pending Profile 不阻止下一次 `preflight_kernel`/`eval_round`；pending
  conclusion 仍阻止。
- `finalize_round` 成功后原子写入嵌套 `conclusion` 和 `next_verdict`，并返回
  CONTINUE/STOP；重复写入被拒绝。
- MCP/Coder 工具集中不存在独立的 `next`。
- Coder frontmatter 有 `Agent(kernel-profile-analyzer)`，没有 profile MCP tools。
- ProfileAnalyzer frontmatter 有 profile MCP tools，没有
  edit/preflight/eval tools。
- ClaudeRuntime 全局 allowlist 包含指定 Agent tool。
- ClaudeRuntime 强制 `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`，避免 Coder 在依赖 subagent 完成前继续执行。

### 20.3 Fake service tests

- CUDA fixture 返回 `ncu-details.txt` 和 `sass.json`。
- Ascend fixture 返回 `msprof-details.txt` 和 `ascend-instructions.json`。
- 两种 fixture 产生同一个 ProfileAnalysis schema。
- artifact download partial failure 保留成功文件和 warnings。
- service unsupported 和 transport failure 正确落 terminal state。

### 20.4 Opt-in hardware tests

- CUDA：真实 eval -> native subagent -> `/profile` -> 本地 artifact -> analysis -> Coder 恢复。
- Ascend：真实 eval -> native subagent -> msprof/simulator artifact -> analysis -> Coder 恢复。
- 验证被 profile 的 `solution_sha256/workload_sha256` 与 eval snapshot 一致。

### 20.5 已实现的可重复 E2E

- `tests/test_mcp_stdio.py` 启动真实 stdio 子进程并使用官方 `ClientSession` 完成 initialize、list_tools、tool call、advisory Profile、terminal analysis 和 exactly-once `finalize_round` 状态转换，不依赖函数内直接调用。
- `tests/run_mcp_cuda_e2e.py` 在 opt-in CUDA 环境执行真实 `preflight_kernel -> eval_round -> profile_workloads -> record_profile_analysis -> finalize_round(finalize)`，校验 NCU report/details/SASS 下载、相同 profile 请求缓存和相同 eval fingerprint 的 terminal analysis 复用，并保留完整 workspace。
- 使用 `claude -p --agent kernel-profile-analyzer` 的真机验证确认 Claude Code 能从 workspace 中生成的 `.claude/agents/kernel-profile-analyzer.md` 加载独立工具集、连接同一个 `kernelgen` MCP server、读取权威 eval context、采集 NCU、提交通过 schema/evidence identity 校验的 analysis，并返回 `recorded: true`。
- schema v3 改动后的原生 Coder→ProfileAnalyzer 真机链需要重新执行；`tests/run_mcp_cuda_e2e.py` 已更新为 plan/profile/conclusion 协议并保留可重复入口。

## 21. 验收标准

1. Coder 的 Claude session ID 在 subagent 调用前后保持不变。
2. ProfileAnalyzeAgent 以 Claude Code 原生 `Agent(kernel-profile-analyzer)` 形式出现，而不是新的 `claude -p` 进程。
3. profile 长报告不进入 Coder prompt，Coder 只收到紧凑 subagent result 和 analysis path。
4. 任何 measured round 在 Coder conclusion 和 verdict 完成前无法 preflight
   下一候选或进入下一 measured round；pending Profile 不阻塞。
5. Coder 无法直接调用 profile 或清除 pending state。
6. ProfileAnalyzeAgent 无法修改 kernel 或调用
   preflight/eval/finalize。
7. 多 workload profile 不被固定数量限制。
8. CUDA 与 Ascend 使用同一 Agent prompt、MCP 工具和 ProfileAnalysis schema。
9. 所有 measurement evidence 都能追溯到 `ProfileResult.metrics` 或本地已校验 artifact。
10. profile failure 不伪装成成功，也不会阻塞优化。
11. Coder 返回后，`SingleCoderOptimizationWorkflow` 在 Distiller 运行前对最终 best
    做 best-effort Profile；失败尽量记录 terminal evidence，但不丢弃有效 best。

## 22. 已采用的第一版决策

1. 只对非 hack 的 new best 请求 Profile；Coder 按诊断价值决定是否立即执行，
   Workflow 在 Distill 前 best effort 补最终 best。
2. 默认先采集 `metrics`，由 Agent 按证据需要升级 `source/instruction`。
3. `failed`、`unsupported` 和 `inconclusive` 都是 terminal evidence 状态；
   pending 本身也不阻塞 lifecycle。
4. 每次 analysis 只给 Coder 一个 `next_experiment`。
5. 直接删除 `ncu_*` ledger 字段且不迁移旧 ledger。
6. 第一版只实现 Claude native subagent，Codex native agent 定义留作后续独立改动。
7. `kernelgen_server_adapter.prepare_evaluation_bundle()` 是
   preflight、eval 和 snapshot 共用的请求构造入口。

# KernelGen 设计文档

> 版本: 1.2 | 日期: 2026-07-23

## 1. 概述

KernelGen 是一个 **AI Agent 编排框架**,用于自动化 GPU/NPU kernel 优化。它将优化过程分解为多个专业 Agent(分析、编码、蒸馏、合并、总结),通过 Workflow 编排它们的协作,最终产出高性能的 Triton/Gluon/CUDA kernel + 可复用的优化经验(KB)。

**核心设计理念:**
- **Agent 拥有判断,Python 拥有协调** —— Agent 做需要推理的事(分析瓶颈、写代码、比较方案),Python 做确定性的事(选 best、合并 KB、停止决策)
- **一切 Runnable** —— Agent 和 Workflow 共享同一个 `run(inp)` 契约,可组合、可嵌套、可并行
- **隔离 + 可插拔** —— Workspace 隔离各 agent 的文件空间;Runtime 可替换 LLM 脚手架(claude/codex/...)
- **权威测量 + 红线** —— eval 的数字是权威的(agent 不能伪造);停止决策由 Python 账本裁定(agent 不能绕过)

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│  KernelGenWorkflow (per-definition, multi-epoch)            │
│                                                             │
│  AnalyzerAgent → 1R/shared_analysis (仅首次运行)             │
│                                                             │
│  for epoch in start_epoch..n_epoch:                         │
│    run_parallel(SingleCoderOptimizationWorkflow, N agents)       │
│      ├── CoderAgent: 写码→preflight→eval→按需 profile→record(finalize) │
│      └── DistillerAgent: trajectory → 两类 KB 候选           │
│    pick_best (Python 按权威 geo_mean 选赢家)                 │
│    epoch KB reducer (唯一 canonical KB 写者)                 │
│    EpochSummaryAgent → synthesis.json + 下一 epoch 方向      │
└─────────────────────────────────────────────────────────────┘
```

这里有三个必须保持的顺序关系：

1. `preflight_kernel` 通过后才能进入正式 `eval_round`；失败的 preflight
   不产生 ledger round。
2. epoch KB reducer 在 EpochSummary 之前运行，因此 synthesis 和下一 epoch
   读取的是本 epoch 已合并的 KB。
3. 多 agent 时最后一个 epoch 也会生成 `synthesis.json`，使后续命令可以从
   `--start-epoch` 继续，而不必重跑上一 epoch。

### 2.1 实现语言与 Agent 角色解耦

Analyzer、Coder 和 EpochSummary 的角色文档只定义稳定的分析、实验和状态
转换职责，不硬编码具体编程语言。`KernelGenInput.implementation_language`
选择受信的 `ImplementationProfile`，由 Python 将同一个 profile 注入
Analyzer、Coder、Distiller 和 EpochSummary task prompt，并写入
`ToolContext`、preflight receipt、eval `Solution.spec.language`、snapshot 和
ledger。

当前端到端 profile 只有 `triton`。它明确规定 DSL、入口、JIT marker、launch
形式、wrapper 白名单和禁止的 framework fallback。添加 TileLang/CUDA 等新
profile 时，必须同时提供 builder、eval-service preflight policy 和测试；不能
只在 prompt 中增加一个语言名字。旧 Analyzer checkpoint 中的
`triton_grid` 仍可读取，新输出统一使用语言无关的 `program_grid`。

## 3. 包结构

```
kernelgen/
├── framework/              ← 公共框架(领域无关)
│   ├── runnable.py         Runnable 基类(bind + run + I/O 校验)
│   ├── base.py             BaseAgent(preprocess → invoke → postprocess + repair-retry)
│   ├── workflow.py         Workflow 基类
│   ├── runtime/            LLM 脚手架抽象
│   │   ├── base.py         Runtime(Protocol) + CLIRuntime(spawn/pump/idle-timeout)
│   │   ├── claude.py       ClaudeRuntime(stream-json)
│   │   ├── codex.py        CodexRuntime(JSONL events)
│   │   └── factory.py      CLI runtime 构造与 provider 环境解析
│   ├── parallel.py         Workspace + IsolatedDirectory + run_parallel
│   ├── contract.py         render_contract / extract_json / append_repair
│   └── models.py           DefinitionModel(嵌套 pydantic)
│
├── agents/                 ← 每个 agent 一个 package
│   ├── analyzer/           冷启动分析(grid/pitfalls/strategy)
│   ├── coder/              内循环优化(写码→preflight→eval→profile→finalize)
│   ├── distiller/          蒸馏(trajectory → candidate experience)
│   ├── merge/              Judge+Merge KB
│   └── epoch_summary/      跨 agent 对比 + 出方向
│
├── workflows/              ← 编排(组合 agents)
│   ├── optimization/       OperatorOptimizeWorkflow(新任务统一入口)
│   │   ├── kernelgen/     KernelGenWorkflow(multi-epoch 编排)
│   │   └── single_coder/  SingleCoderOptimizationWorkflow(共享执行引擎)
│   ├── knowledge_reducer.py  epoch 级 KB 候选收集与归并
│   ├── legacy/            旧 SimpleOpt/Python Batch 续跑与结果读取
│   └── optimize.py         Python 控制的逐轮优化
│
├── mcp_server/             ← 单一 stdio MCP server(Claude/Codex 可共用)
│   ├── server.py           FastMCP 注册 preflight/eval/profile/finalize 工具
│   ├── context.py          ToolContext 旧导入路径兼容层
│   └── contract.py         稳定 server/tool 名称
│
├── tools/                  ← 确定性领域函数 + 人工 CLI
│   ├── preflight.py        preflight receipt 的生成、验证与单次消费
│   ├── eval_round.py       完整 eval round 生命周期 + 权威评测记账
│   ├── profile_round.py    immutable snapshot + profile + analysis 落盘
│   └── finalize_round.py     评测后结论落账本(schema 阻止测量字段)
│
├── data/                   ← 数据层(纯 Python,host 可测)
│   ├── implementation.py   implementation language/profile 注册表
│   ├── tool_context.py     工作区工具配置 + 路径边界
│   ├── ledger.py           Ledger(worktree 权威状态)
│   ├── stop_policy.py      StopConfig + next_verdict(trigger/veto 纯函数)
│   ├── selection.py        pick_best(分数) / pick_seed(种子)
│   ├── experiment_plan.py  ExperimentPlan(eval_round 输入 schema)
│   ├── round_conclusion.py RoundConclusion(finalize_round 输入 schema)
│   ├── profile_analysis.py backend-neutral ProfileAnalysis schema
│   ├── optimization_history.py  OptimizationHistory 数据模型
│   └── trajectory.py       完整蒸馏 trajectory + 有界 synthesis trajectory
│
├── .kernelgen/             ← provider-neutral 运行定义
│   ├── agents/             角色正文与 capability/tool/subagent 契约
│   ├── skills/             通用 Agent Skills 及其脚本、引用和资源
│   └── mcp.json            MCP transport/command/timeout 唯一真值
├── kb/                     ← repository base KB；运行时只有 run-level 副本保存 Git 历史
├── tests/                  ← host 测试
└── TODO.md
```

## 4. 核心抽象

### 4.1 Runnable — 统一契约

```python
class Runnable(ABC):
    InputModel: Type[BaseModel]
    OutputModel: Type[BaseModel]

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "Runnable": ...
    def run(self, inp, runtime=None) -> BaseModel: ...
    def _execute(self, inp) -> Any: ...  # 子类实现
```

**设计理念:**
- `run(inp)` 是唯一入口:校验输入 → `_execute` → 校验输出
- `bind(path, factory)` 让 `run_parallel` 能统一调度(allocate → bind → run)
- Agent 和 Workflow 都是 Runnable → 可组合、可嵌套

### 4.2 BaseAgent — 契约层

```python
class BaseAgent(Runnable):
    name: str                 # 逻辑名;Claude 映射到 kernel-<name>
    InputModel / OutputModel  # Pydantic I/O 契约

    def preprocess(inp, runtime) -> str   # 只拼动态 input + output contract
    def postprocess(raw, runtime) -> BaseModel  # 解析 JSON → OutputModel
    def _execute(inp): native agent invoke + repair-retry(_MAX_RETRIES=2)
```

**设计理念:**
- Agent = 契约层(要什么、吐什么)
- Runtime = 执行层(怎么跑)
- 两者正交:换 LLM 脚手架不改 agent;换 agent 不改 runtime

### 4.3 Workflow — 编排层

```python
class Workflow(Runnable):
    @classmethod
    def bind(cls, path, runtime_factory) -> "Workflow":
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp):
        # 编排内部 Agent/Workflow + 直接调 Python(pick_best 等)
```

**设计理念:**
- Workflow 是 Runnable(可被 run_parallel 并行、可嵌套)
- 内部自造 runtime(`self._runtime_factory(cwd)`)—— 不用外部传
- 轻 Python 直接调(pick_best),重步走 run_parallel

### 4.4 Runtime — 执行层

```python
class Runtime(Protocol):
    def invoke(self, prompt: str, *, model: str = "inherit", agent: str | None = None) -> str: ...

class CLIRuntime(Runtime, ABC):
    # 通用: spawn + stdin/stderr threads + idle-timeout pump + resume
    def _command(prompt, session_id) -> list: ...   # 子类填
    def _parse_line(line, state) -> str: ...        # 子类填

class ClaudeRuntime(CLIRuntime):
    # stream-json 特化 + settings 覆写 + 完整运行日志

class CodexRuntime(CLIRuntime):
    # codex exec --json 特化 + MCP 映射 + thread resume + 完整运行日志
```

**设计理念:**
- Runtime 是"prompt 进,text 出"的纯传输
- worktree/env/resume 全是构造时的属性(不进 invoke)
- 接新脚手架 = 加一个 CLIRuntime 子类并在 provider-neutral factory 注册

ClaudeRuntime 和 CodexRuntime 在每个 workspace 分别保存一份人类可读日志：

- `.kernelgen/claude-runtime.log`：给人阅读，包含 prompt、完整 thinking、
  assistant text、工具输入/结果、token usage 和 resume 信息。
- `.kernelgen/codex-runtime.log`：保留 prompt、reasoning summary、assistant text、工具输入/结果、token usage，并记录可用于精确 resume 的 Codex thread ID。

Claude partial stream 和最终聚合消息可能携带同一 thinking/tool 信息。ClaudeRuntime 按 message ID 和已完成 thinking block 去重；CodexRuntime 直接消费 `codex exec --json` 的 completed item。两者都只写人类可读证据，不额外落盘 provider 原始 stream。

### 4.5 Workspace — 隔离

```python
class Workspace(ABC):
    def allocate(name) -> str: ...   # 返回隔离路径
    def cleanup(name): ...

class IsolatedDirectory(Workspace):
    # mkdir + materialize .kernelgen/provider config + copytree kb(排除 .git)
```

**设计理念:**
- Workspace 只管"给路径"(不管"怎么跑"——那是 Runnable.bind 的事)
- `run_parallel` 用它:allocate → bind → run → 收集
- Agent 和 Workflow 统一:换隔离方式 = 换 workspace 参数
- agent、shared analysis 和 synthesis 只拿 KB 内容快照，不包含嵌套 `.git`；
  只有 run-level `kb/` 维护 epoch 历史

## 5. Agent 详解

### 5.1 AnalyzerAgent (agents/analyzer/)

| | |
|---|---|
| **职责** | 冷启动分析:给定 kernel 定义,产出 grid/loop/pitfalls 优化计划 |
| **输入** | `AnalyzerInput{definition, target_hardware}` |
| **输出** | `AnalyzerOutput{core_math, program_grid, inner_loop, key_pitfalls, ...}` |
| **工具** | query_kb(查 KB 找相关经验) |
| **何时调** | epoch 1 开头,一次 |

Analyzer 与 Coder 使用同一个可读 definition 布局：Axes、Inputs、Outputs、
DPS 模式分别成段，reference 源码只在自身段内缩进。不能对包含多行源码的
整块 prompt 使用 `dedent`，否则 reference 的缩进会改变外围字段布局，模型
看到的 definition 会出现层级错乱。

### 5.2 CoderAgent (agents/coder/)

| | |
|---|---|
| **职责** | 内循环优化:自主 LOOP(写码→preflight 通过→提交 plan 并 eval→按诊断价值调用 ProfileAnalyze→finalize_round 自动 KEEP/REVERT/REPAIR 并 finalize→continue/stop) |
| **输入** | `CoderInput{definition, analysis, seed_code, eval_server_url, ...}` |
| **输出** | `CoderReport{status, summary}`(极简;真实产出在 ledger) |
| **工具** | preflight_kernel(编译门禁), eval_round(冻结计划+评测+记账), Agent(kernel-profile-analyzer), finalize_round(候选转换+结论+停止裁决), query_kb |
| **何时调** | 每 epoch 每 agent slot |
| **特殊** | 一次长 invoke,agent 自己循环(不是外部多次调) |

### 5.2.1 ProfileAnalyzeAgent (.kernelgen/agents/kernel-profile-analyzer.md)

| | |
|---|---|
| **职责** | 对一个权威 PASSED round 选择任意数量 workload，调用统一 profile service，读取本地 artifact，并记录 evidence-backed 下一实验 |
| **输入** | Coder 只委派 `round_num`；其余上下文由 `get_profile_context` 从 immutable snapshot 获取 |
| **输出** | 紧凑 subagent result；完整结果在 `.kernelgen/profile-analysis/round-N.json` |
| **工具** | get_profile_context, profile_workloads, record_profile_analysis, Read/Glob/Grep |
| **权限** | 无 Write/Edit/Bash/preflight/eval/finalize，Coder 也没有三个 profile MCP 工具 |
| **触发** | 仅 new best 返回 `profile_required=true`；Coder 判断证据能否指导下一实验，Workflow 在 Distill 前 best effort 补最终 best |

Coder 需要诊断时在当前会话中前台调用 native subagent；pending Profile
不是 ledger gate，不阻塞 `finalize_round`、STOP 或下一轮。无论是否立即 Profile，
Coder 都必须调用一次 `finalize_round` 原子写入 `conclusion` 和 verdict；
`conclusion != null` 本身就是完成标志，不再维护容易漂移的
`narrative_recorded` 布尔字段。CUDA 与 Ascend 使用相同工具和 schema，
后端差异由 KernelGen Server 的 capabilities、metrics 和 artifacts 表达。
Coder 结束后，`SingleCoderOptimizationWorkflow` 对尚未完成的最终 best 运行一次
best-effort Profile；失败写 terminal evidence，但不丢弃有效 best。

### 5.3 DistillerAgent (agents/distiller/)

| | |
|---|---|
| **职责** | 从 trajectory(初始 kernel + per-round diff + plan/measurement/profile/conclusion)生成候选 KB experience |
| **输入** | `DistillInput{definition, rounds(含嵌套 solution), best_geo_mean, best_round}` |
| **输出** | `DistillOutput{candidate_experience, candidate_detailed, skip_reason}` |
| **工具** | 无(信息全在 prompt) |
| **何时调** | 每个 CoderAgent 跑完后(SingleCoderOptimizationWorkflow 内) |

Distiller 除了 rounds 还接收完整 canonical definition 和权威 best 摘要，不读取
KB。shape、常量和 `op_type` 必须来自 definition，不能从失败代码或较差候选中
反推。

### 5.4 MergeAgent (agents/merge/)

| | |
|---|---|
| **职责** | Judge+Merge:比较候选 vs 已有 KB,决定 KEEP/DISCARD/MERGE |
| **输入** | `MergeInput{mode, candidate, existing, label, ...}` |
| **输出** | `MergeOutput{verdict, merged_text}` |
| **工具** | 无 |
| **何时调** | epoch reducer 合并多个 agent 候选时 |
| **特殊** | Python 负责候选校验、排序、大小限制和落盘；MergeAgent 只做语义合并 |

### 5.5 EpochSummaryAgent (agents/epoch_summary/)

| | |
|---|---|
| **职责** | 跨 N agent 对比:选种子 + 出下一 epoch 方向 + synthesis report |
| **输入** | `EpochSummaryInput{agent_results(含有界无源码 trajectory + new_experience), fixed_best_geo, fixed_best_kernel, ...}` |
| **输出** | `EpochSummaryOutput{selected_author, next_directions, synthesis_report}` |
| **工具** | query_kb |
| **何时调** | 每个多 agent epoch 末，包括当前命令的最后一个 epoch |

EpochSummary 的 prompt 只放一份全局最佳 kernel，作为下一 epoch 的代码锚点；
每个 agent 的 trajectory 只保留轮次表和 best/final 关键证据，避免重复源码使
synthesis 上下文随 round 数无界增长。

Synthesis 是 best-effort：失败不会抹掉本 epoch 已完成的 ledger、最佳 kernel
或 KB commit，但不会生成 `synthesis.json`，因此下一次 `--start-epoch` 续跑
前必须先补齐该 checkpoint。

## 6. Workflow 详解

### 6.1 SingleCoderOptimizationWorkflow (workflows/optimization/single_coder/)

一个 agent slot 的完整生命周期:
```
CoderAgent(优化) → DistillerAgent(蒸馏)
                 → .new_experience.md + .new_detailed.md
```
是 `run_parallel` 的并行单元。

Coder 返回结构合法的 `CoderReport` 后，workflow 还会检查 ledger
postcondition：不得存在 missing conclusion，最新 round 必须有持久化 verdict，
且 ledger 的 definition/hardware/language 必须与 `SingleCoderOptimizationInput`
一致。pending Profile 不属于 lifecycle blocker；Workflow 在 Distiller 前只对
最终 best 做一次 best-effort Profile。workflow 最终从 ledger 组装权威
`SingleCoderOptimizationOutput`，包含 `best_code`、`best_geo_mean` 和 round 数。

该 workflow 不修改 canonical KB。并行 agent 只提交候选，统一由外层
`KernelGenWorkflow` 在 epoch 边界归并，避免并发覆盖和较差候选抢先写入。

### 6.2 KernelGenWorkflow (workflows/optimization/kernelgen/)

主编排(per-definition, multi-epoch):
```
for epoch:
  build_inputs(cold_start / directions + seed_code)
  run_parallel(SingleCoderOptimizationWorkflow × N, IsolatedDirectory)
  pick_best
  knowledge.merge_epoch_kb (候选校验/排序 → 有界语义合并 → git commit)
  EpochSummaryAgent → synthesis.json + next_directions
```

`start_epoch > 1` 时，workflow 从
`1R/shared_analysis/analysis.json`、上一 epoch 的 `synthesis.json` 和此前
`agent*/.ledger.json` 重建状态。`n_epoch` 表示最终 epoch 编号，不是追加数量。

package 按实际步骤划分职责，公开导入仍统一从 `kernelgen.workflows.optimization.kernelgen` 获取：

| 模块 | 职责 |
|---|---|
| [workflow.py](workflows/optimization/kernelgen/workflow.py) | 公开 run/finalize 入口、共享生命周期、epoch 顺序与全局 best 更新 |
| [preparation.py](workflows/optimization/kernelgen/preparation.py) | fresh/fork/resume、冻结评测合同、KGS 目标身份、初始 seed、共享 Analyzer checkpoint |
| [epoch.py](workflows/optimization/kernelgen/epoch.py) | Coder 输入、隔离 workspace、并发与部分失败、单次 ledger 结果读取、综合输入 |
| [recovery.py](workflows/optimization/kernelgen/recovery.py) / [finalization.py](workflows/optimization/kernelgen/finalization.py) | 完成状态恢复 / Knowledge 发布与综合 checkpoint、finalize-only 业务步骤 |
| [knowledge.py](workflows/optimization/kernelgen/knowledge.py) | V1 Bridge、最佳 Solution 发布与仍在使用的 legacy KB 路径 |
| [contracts.py](workflows/optimization/kernelgen/contracts.py) | 稳定公开输入输出与内部只读 `EpochResult` |

`EpochResult` 在内存中将最佳 workspace、round、geo、code 和所属 epoch 的结果元数据一起传递；全局更新、历史恢复复用相同选择规则，同分保留先前赢家。综合直接使用赢家身份，agent ID 带 epoch 路径，不再根据相同分数反推作者；公开 `KernelGenOutput` 从结果派生，不增加内部身份字段或状态文件。`PreparedRun` 只是准备步骤的只读返回值，不是可持续修改的运行状态仓库。

内部模块直接调用各自依赖，不再通过 Workflow 私有方法传递 `geo_reader`、`code_reader`、`collector`、`epoch_loader`、`coder_builder` 等回调。原 `_collect`、`_read_best_*`、`_load_completed_*`、`_coder_input` 等私有转发方法已移除；依赖这些私有 override 的测试或扩展需改用真实 ledger fixture 或对应模块的数据边界。公开 Workflow 导入、`run`/`bind`/`finalize_completed_epoch`、Runtime factory、Workspace 隔离和 Knowledge reviewer 串行化保持不变。SimpleOpt 继续复用单 Coder、ledger、Catalog/snapshot primitives 和共享生命周期，不引入多 epoch 或新的通用 Workflow 基类。

## 7. 工具(Tools)

| 工具 | 位置 | 职责 | 谁调 |
|---|---|---|---|
| `preflight_kernel` | mcp_server/server.py → tools/preflight.py → KernelGen Server `/preflight` | 运行服务端版本化静态策略和每 workload 目标机 compile/smoke；生成绑定精确候选的一次性 receipt；失败不创建 round | CoderAgent |
| `eval_round` | mcp_server/server.py → tools/eval_round.py | 先冻结 ExperimentPlan，再评测并写入 immutable solution/evaluation；仅 new best 创建 advisory Profile 请求，仍阻止越过上一轮 conclusion | CoderAgent |
| `finalize_round` | mcp_server/server.py → tools/finalize_round.py | exactly-once 原子写入 RoundConclusion 和 Python stop-policy verdict，并按权威 best 自动 KEEP/REVERT/REPAIR；schema 禁止测量字段 | CoderAgent |
| `get_profile_context` | mcp_server/server.py → tools/profile_round.py | 读取 immutable eval snapshot、完整 workload 结果和 profiler capabilities | ProfileAnalyzeAgent |
| `profile_workloads` | mcp_server/server.py → tools/profile_round.py | 对 Agent 选中的精确 workload 采集并下载 backend-native artifact | ProfileAnalyzeAgent |
| `record_profile_analysis` | mcp_server/server.py → tools/profile_round.py | 校验证据引用并落盘 terminal ProfileAnalysis；不控制 round lifecycle | ProfileAnalyzeAgent |

Coder 和 ProfileAnalyzeAgent 通过 Claude Code 原生 MCP tool call 调用它们。上述
lifecycle/Profile 工具与 status、debug、knowledge 工具共用一个 stdio server
进程，不是每个工具各启一个 server。不存在独立的 `next` MCP 或 CLI；正常 round
的 conclusion 和 verdict 由 `finalize_round` 原子写入并直接返回。评测地址、
definition、hardware 与 DPS 模式由 `.kernelgen/tool-context.json` 提供，而不是作为
MCP 调用参数暴露；首次权威 eval 同时把 definition 和 hardware 固化到 ledger，使
导出的 ledger 可以独立理解，后续 round 若元数据不一致则拒绝写入。

`ToolContext` 只有一套 schema，但有两种准备方式：

- Workflow 托管：`SingleCoderOptimizationWorkflow` 在 Coder 启动前写入
  `<workspace>/.kernelgen/tool-context.json`，Runtime 设置
  `KERNELGEN_WORKSPACE`。
- 独立 MCP/E2E：没有 Workflow 代为准备，启动脚本必须显式构造并写入
  `ToolContext`，同时设置 `KERNELGEN_WORKSPACE`；缺失时工具 fail fast。

工具显式收到 workspace 时优先使用它；否则按
`KERNELGEN_WORKSPACE → CLAUDE_PROJECT_DIR → cwd` 解析。Claude 原生 Agent
与 inline fallback、legacy workloads 与 phased workloads 都复用同一套
初始化策略。

preflight 位于 ledger round 之外。通过结果绑定 kernel/solution/definition/
workloads/target/policy，以及 Server API/backend/timing/device identity；候选
代码、上下文或这些服务身份字段变化都会使 receipt 失效。`eval_round` 在正式
评测前验证并消费 receipt，因此同一通过结果不能复用。详细状态机见
[round_lifecycle.md](docs/design/workflows/round_lifecycle.md)，当前 v5 Catalog 与历史 v4
Definition/Workload 约定见
[flaggems_definition_workload_guide.md](docs/guides/flaggems_definition_workload_guide.md)。

ledger 不再保留旧的跨 agent 历史拼接字段 `source_agent`、`best_source_run`、`seed_round_count`，因为当前 multi-epoch 流程通过独立 workspace、`seed_code` 和 EpochSummary 传递信息，不合并各 agent 的 rounds。全 workload eval 取代 regime/representative 评测后，`per_regime_geo_means` 也已删除。`best_code` 与每轮 `solution.code` 保留，前者让单独导出的 ledger 自包含，后者供 Distiller 和 EpochSummary 构造 trajectory。

`Ledger.best` 的 code、round、geo_mean 均来自同一次加载的 ledger，供结果输出、恢复 seed、候选回退和 Knowledge promotion 使用。`.best_kernel.py` 仅是便于交付的代码导出产物；缺失、损坏或写入中断导致它与 ledger 不一致时，不改变优化事实，也不作为读取 fallback。

`Ledger.coder_completed` 是正常监督与 epoch 恢复共用的 measured Coder 完成判定：全部 measured round 已记录 conclusion，最后一轮有非取消的 STOP verdict；pending Profile 不阻止 Coder 完成。`user_cancelled` 仍须显式续跑，不能通过 finalize-only 当作正常结束。Measured ledger 的 definition、target 和 implementation identity 必须匹配，缺字段不能作为匹配；空 ledger 不能证明完成，恢复零轮失败还需已有 epoch completion manifest 证明该 invocation 已返回，且不能带有无 measured round 的 best 摘要。

停止策略 `.stop_config.json` 使用公共原子写入工具持久化，成功提交后才更新内存配置。只有文件不存在时才使用默认 `StopConfig`；已有文件无法解析或构造配置时明确报错并保留原文件，不静默改用默认轮数上限。

优化 Workflow 的开始、取消、失败和最终输出由 `workflows.lifecycle.run_workflow` 统一处理。同一调用链显式传递同一个 RunControl 时（SimpleOpt → KernelOptimization），最外层拥有生命周期；函数调用上下文只记录临时所有权，退出即释放，不新增持久状态。独立 Coder 的 RunControl 仍分别维护其 scope。KernelGen 正常执行与 finalize-only 使用同一边界；输出校验及最后安全点通过后才写成功终态。Epoch 综合原样传播模型异常和取消，在发布、综合及最终 promotion 前检查协作取消；不改变 Runtime 完整模型输出后的安全点，不轮询或 kill Agent。

ledger schema v3 将每轮固定为 `plan / solution / evaluation / profile /
conclusion / next_verdict`。Coder 在看到 eval 结果前提交 `plan`，Python 写入源码
hash、immutable snapshot、完整 workload candidate/reference latency、speedup
以及相对上一权威最佳轮的差值，ProfileAnalyzer 只能经校验工具写 profile
analysis，Coder 提交 `conclusion` 后由 Python `finalize_round` 在同一原子写入
中持久化最终裁决。非 baseline round 必须给出 `expectation_status` 和非空
`perf_gap_analysis`，从而让
Distiller 与 EpochSummary 能比较并行探索的预期和实际差距。完整字段和所有权
见 [round_lifecycle.md](docs/design/workflows/round_lifecycle.md)。

`mcp_server/stdio.py` 只替换 FastMCP 的标准流 transport，协议消息、初始化、session 和 tool dispatch 仍使用官方 MCP SDK。在部分托管环境中，AnyIO 对阻塞 stdin 的 worker-thread 包装无法把完成事件可靠送回 asyncio loop，会令官方 FastMCP stdio initialize 卡住；Linux agent workspace 因此使用 asyncio 原生 pipe reader，并由真实 MCP ClientSession 握手测试覆盖。

角色与 provider 配置分层维护，不会产生第二份角色真值：

- `.kernelgen/agents/*.md`：KernelGen 自有的角色正文和中立 `capabilities`、`mcp_tools`、`subagents` 契约。
- `.kernelgen/skills/<name>/SKILL.md`：KernelGen 自有的通用 Agent Skill 定义；同目录下可包含 `scripts/`、`references/` 和 `assets/`。
- `.kernelgen/mcp.json`：KernelGen 自有的 MCP transport、command 和秒制 timeout 契约。
- workspace `.claude/agents/*.md`：运行前从中立定义生成，供 Claude Code 原生 agent 使用。
- workspace `.claude/skills/<name>/`：从中立 Skill 完整复制，供 Claude Code 原生发现。
- workspace `.codex/agents/*.toml`：运行前从同一中立定义生成，供 Codex 项目级 custom agent 使用。
- workspace `.agents/skills/<name>/`：从同一中立 Skill 完整复制，供 Codex 按官方仓库级路径发现。
- workspace `.mcp.json`：从中立 MCP 定义生成，仅供 Claude Code 注册项目级 MCP server。

ClaudeRuntime 使用生成的 YAML frontmatter 选择原生 agent、限制工具并读取生成的 `.mcp.json`；CodexRuntime 使用生成的 custom agent 配置支持子 agent，同时从 workspace 的中立快照生成当前 `codex exec` 的 `mcp_servers.*` 临时配置并设置 MCP `enabled_tools`。两个 runtime 加载同一份 Skill 内容，仅使用各自的原生发现目录。生成物只存在于运行 workspace，不提交到仓库，也不修改用户 `CODEX_HOME`。

ProfileAnalyzeAgent 被调用时必须作为 Coder 的前台证据收集步骤，不能成为与候选
修改并发的旁路任务。由于 Claude Code 2.1.198 起默认后台运行 subagent，Coder
role 明确要求 `run_in_background=false`，`ClaudeRuntime` 同时设置
`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`。是否立即调用由 Coder 根据诊断价值
决定；pending 状态本身不再形成 MCP gate。

## 8. 数据流

下图描述未启用 V1 Knowledge 的普通优化路径。显式启用 V1 时，Knowledge Coder/Distiller、发布与恢复的契约以 [V1 实现说明](docs/design/knowledge/knowledge_base_implementation.md) 为准，不把 legacy Markdown reducer 当成唯一知识路径。

```
KernelGenInput{definition, n_parallel, n_epoch, ...}
    │
    ▼ preparation.prepare_analysis(cold_start)
AnalyzerOutput{program_grid, key_pitfalls, ...}
    │
    ▼ epoch.build_epoch_inputs(分发 directions)
N × CoderInput{definition, analysis, seed_code}
    │
    ▼ run_parallel(SingleCoderOptimizationWorkflow)
    │   ├── CoderAgent → ledger(.ledger.json + .best_kernel.py)
    │   └── DistillerAgent → .new_experience.md + .new_detailed.md
    │
    ▼ pick_best(ledger geo_mean)
best_result{best_code, best_geo_mean}
    │
    ▼ epoch KB reducer(校验候选 → 分数排序 → MergeAgent → 原子写入)
run-level canonical KB 更新 + git commit
    │
    ▼ EpochSummaryAgent(单份 best kernel + 有界 trajectory × N)
synthesis/synthesis.json{next_directions, synthesis_report}
    │
    ▼ (next epoch: directions → coder_inputs with seed_code)
```

## 9. KB 更新流程（legacy 路径）

**三级视角:**

1. **Per-agent**(`SingleCoderOptimizationWorkflow` 内):
   - DistillerAgent 读取完整 schema v3 ledger 和已有 KB 快照
   - 只生成 `.new_experience.md` 与 `.new_detailed.md`
   - 不修改 canonical KB

2. **Per-epoch**(`kernel_gen.knowledge.merge_epoch_kb`):
   - 以 agent ledger 校验 definition、op_type、target 和权威分数
   - PASSED 和高 `best_geo_mean` 候选优先，避免较差结果主导合并
   - 同时合并 `experience.md` 与 `detailed.md`
   - canonical 路径严格使用真实
     `{op_type}/{definition}/{target_hardware}`，不创建 `elementwise` 占位目录
   - `experience.md`/`detailed.md` 分别限制为 12,000/30,000 字符
   - 多候选合并失败时保留旧 KB；成功后原子写入并提交 run-level KB Git
   - agent 和 synthesis 只复制不含 `.git` 的 KB 内容快照

3. **跨算子**(外层编排,TODO):
   - 读各算子 canonical KB 和权威运行证据
   - cross-def 蒸馏 → 共享 KB

上述是保留的 legacy Markdown 路径，不代表 V1 尚未实现。V1 已作为显式能力接入 KernelGenWorkflow，支持只读与读写模式；SimpleOpt 可显式消费知识并产生本地 CandidateDraft，但不发布公共 Catalog。使用方法见 [Knowledge 快速使用](docs/guides/KB_QUICKSTART.md)，当前实现与后续计划分别见 [实现说明](docs/design/knowledge/knowledge_base_implementation.md) 和 [待办](docs/design/knowledge/knowledge_base_todo.md)，不在本总览重复维护 V1 字段和状态机。

## 10. 如何添加新 Agent

### 步骤

**1. 写 Python 契约与动态 prompt:**
```bash
mkdir -p kernelgen/agents/my_agent/
```

```python
# kernelgen/agents/my_agent/__init__.py
from pydantic import BaseModel, Field
from kernelgen.framework.base import BaseAgent
from kernelgen.framework.models import DefinitionModel

class MyAgentInput(BaseModel):
    """Agent 需要的输入。"""
    definition: DefinitionModel
    target_hardware: str
    # ... 你的字段

class MyAgentOutput(BaseModel):
    """Agent 的结构化输出。"""
    result: str = Field(description="...")
    # ... 你的字段

class MyAgent(BaseAgent):
    name = "my_agent"
    InputModel = MyAgentInput
    OutputModel = MyAgentOutput

    # 可选:覆盖 preprocess(自定义 prompt 组装)
    def preprocess(self, inp, runtime) -> str:
        from kernelgen.framework.contract import render_contract
        contract = render_contract(self.OutputModel)
        return f"<input>\n{inp.model_dump_json()}\n</input>\n\n{contract}"
```

**2. 写 provider-neutral agent role:**

文件名和 `name` 都使用 `kernel-<Python name>` 约定,例如
`.kernelgen/agents/kernel-my-agent.md`:

```markdown
---
name: kernel-my-agent
description: Explains when the runtime should use this agent.
capabilities: [read, search, skill]
mcp_tools: []
subagents: []
model: inherit
---

You are an expert at [whatever your agent does].

Given the input, produce a structured JSON output following the contract below.
```

Workspace materializer 会由这一个文件生成 Claude `.md` 和 Codex `.toml`；ClaudeRuntime 使用 `--agent kernel-my-agent`，CodexRuntime 为主 turn 注入同一中立角色并让 Codex 发现生成的 custom subagents，FakeRuntime 使用 inline fallback。

**3. (可选)增加确定性工具:**

先把领域函数放到 `tools/`,再在 `mcp_server/server.py` 暴露薄 MCP adapter,并把稳定名称加入 `mcp_server/contract.py`。最后在中立 frontmatter 的 `mcp_tools` 中写 `my_tool`，materializer 会生成 provider 需要的完整工具名。同一 server 可注册多个工具,不需要新增进程。

如果只是知识、操作指南或检索流程，应新增 `.kernelgen/skills/<name>/SKILL.md`，而不是做 MCP tool。Skill 使用通用 `SKILL.md` 格式，workspace materializer 会完整生成 Claude `.claude/skills/<name>/` 和 Codex `.agents/skills/<name>/`；不要直接维护 provider 目录。Skill 和 MCP 是两个独立层次，不会因同名描述自动冲突。

**4. 校验并在 workflow 里调用:**

```bash
PYTHONPATH=/path/to/project-parent python3 -m kernelgen.tools.doctor
```

```python
from kernelgen.agents.my_agent import MyAgent
result = MyAgent().run({"definition": ..., "target_hardware": "..."}, runtime)
```

### 要点

- **InputModel/OutputModel 是单一真源** —— prompt 契约从它派生(render_contract),校验也用它
- **中立 Agent Role 是角色与权限的单一真源** —— provider frontmatter/TOML 均从 `.kernelgen/agents` 生成
- **不需要维护 init/role 对照表** —— `my_agent` 自动映射 `kernel-my-agent`;doctor 检查文件名、name、工具与 MCP 配置
- **原生 subagent 引用会前置校验** —— `Agent(kernel-foo)` 指向的 Markdown、frontmatter 和间接 MCP 依赖会在 Runtime 启动前与 doctor 中检查
- **隔离工作区自动物化中立角色、Skill 与 MCP 配置** —— `.claude/agents`、`.claude/skills`、`.codex/agents`、`.agents/skills` 和 `.mcp.json` 都是运行时生成物
- **FakeRuntime 可测** —— `MyAgent().run(inp, FakeRuntime(["scripted reply"]))`

## 11. 如何添加新 Workflow

### 步骤

**1. 在 `workflows/` 创建文件:**
```python
# kernelgen/workflows/my_workflow.py
from kernelgen.framework.workflow import Workflow
from kernelgen.framework.parallel import IsolatedDirectory, run_parallel
from kernelgen.agents.my_agent import MyAgent
from pydantic import BaseModel

class MyWorkflowInput(BaseModel):
    # ...

class MyWorkflowOutput(BaseModel):
    # ...

class MyWorkflow(Workflow):
    name = "my_workflow"
    InputModel = MyWorkflowInput
    OutputModel = MyWorkflowOutput

    def __init__(self, *, cwd, runtime_factory):
        self._cwd = Path(cwd)
        self._runtime_factory = runtime_factory

    @classmethod
    def bind(cls, path, runtime_factory):
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp):
        main_rt = self._runtime_factory(str(self._cwd))

        # 非并行步骤:直接用 main_rt
        step1_out = MyAgent().run(step1_input, main_rt)

        # 并行步骤:用 run_parallel
        results = run_parallel(
            SomeAgent, inputs,
            workspace=IsolatedDirectory(base=self._cwd / "parallel",
                                        claude_source=self._cwd / ".claude"),
            runtime_factory=self._runtime_factory,
        )

        # Python 权威裁决:直接调(不套 agent)
        best = pick_best(results, key=lambda r: r[0].score)

        return {"result": best, ...}
```

**2. 使用:**
```python
wf = MyWorkflow(cwd="/path/to/workspace", runtime_factory=make_rt)
out = wf.run(input_dict)
# 或通过 run_parallel 并行多个:
run_parallel(MyWorkflow, inputs, workspace=Docker(...), runtime_factory=make_rt)
```

### 要点

- Workflow 也是 Runnable —— 可以被 run_parallel 并行(每个 workflow 一个隔离空间)
- `bind(path, factory)` 让 run_parallel 能统一调度
- 内部 `main_rt = self._runtime_factory(cwd)` —— 非并行 agent 用这个
- 轻 Python(pick_best/should_stop)直接调,不套壳
- 并行步用 `run_parallel(AgentOrWorkflow, inputs, workspace=..., runtime_factory=...)`

## 12. Tutorial: 用框架实现不同复杂度的 Pipeline

下面通过三个真实 example 展示如何用本框架构建 agent pipeline。从最简单到最复杂,递进演示框架的核心能力。

### 12.1 auto_gen — 最简模式:一个 Agent 直接 fan-out

**场景**: 批量生成 FlagGems 算子。每个算子 = 1 个 agent 一次性完成(实现→注册→测试→benchmark)。Agent 内部自主控制流程,Python 不参与迭代。

**为什么不需要 Workflow**: 没有 Python 需要控制的步骤(停止条件/版本保存/跨 agent 对比)。`run_parallel(Agent, inputs)` 直接 fan-out 即可。

#### Step 1: 定义 Agent (`agents/auto_gen/__init__.py`)

```python
from kernelgen.framework.base import BaseAgent
class AutoGenInput(BaseModel):
    operator: str   # "relu", "gelu", ...

class AutoGenOutput(BaseModel):
    operator: str
    status: str     # "success" / "failed"
    accuracy_passed: bool | None = None
    error_message: str | None = None
    files_created: list[str] = []
    files_modified: list[str] = []
    implementation_mode: str | None = None
    # ...

class AutoGenAgent(BaseAgent):
    name = "auto_gen"
    InputModel = AutoGenInput
    OutputModel = AutoGenOutput
```

`BaseAgent.preprocess()` 会把 `AutoGenInput` 序列化为动态任务输入并从 `AutoGenOutput` 派生输出契约，因此 Python 中不需要模板替换。

#### Step 2: 写中立 Agent Role (`.kernelgen/agents/kernel-auto-gen.md`)

```markdown
---
name: kernel-auto-gen
description: "Use this agent when a complete FlagGems Triton operator must be implemented, registered, tested, formatted, cataloged, committed, and benchmarked in one autonomous session."
capabilities: [shell, read, write, edit, search]
mcp_tools: []
subagents: []
model: inherit
---

You are a FlagGems Triton operator implementation expert.
Use the `operator` field from the dynamically injected input; the static instructions below refer to it as `<operator>`.
```

长角色指令只保存在这个原生 agent 文件中，使用 `<operator>` 表示动态输入里的值。内容包含:
- 算子语义查询步骤
- 实现模式选择(pointwise_dynamic / manual kernel)
- 注册步骤(ops/__init__.py + _FULL_CONFIG)
- 测试 + pre-commit + benchmark 流程
- 最终 JSON 输出格式

ClaudeRuntime 通过 `--agent kernel-auto-gen` 使用该文件；FakeRuntime 等非原生 runtime 自动内联同一正文，不维护第二份 `role.md`。

#### Step 3: 入口脚本 (`examples/auto_gen/run_example.py`)

```python
from kernelgen.agents.auto_gen import AutoGenAgent
from kernelgen.framework.parallel import IsolatedDirectory, run_parallel
from kernelgen.framework.runtime.claude import ClaudeRuntime

inputs = [{"operator": "relu"}, {"operator": "gelu"}, {"operator": "silu"}]
ws = IsolatedDirectory(
    base=Path("/tmp/auto_gen/agents"),
    claude_source=KERNELGEN_ROOT / ".claude",
)

def make_rt(path):
    return ClaudeRuntime(workspace=path, model="deepseek-v4-pro[1m]", ...)

results = run_parallel(AutoGenAgent, inputs, workspace=ws,
                       runtime_factory=make_rt, max_workers=4)

for result, ws_name in results:
    print(result.model_dump_json())
```

**关键点**: 不需要 Workflow 类 —— 直接 `run_parallel(Agent, inputs)` 完成并行调度。

---

### 12.2 optimize_loop — 中等模式:Agent + Python 控制循环

**场景**: 迭代优化 kernel 性能。每轮 agent 做一个优化方向,Python 负责版本保存 + 性能历史更新 + 停止条件判断。

**为什么需要 Workflow**: Python 要控制"何时停"(target speedup / max iters),并在每轮之间更新 PERFORMANCE.md 让下一轮 agent 看到历史。

#### Step 1: 定义单轮 Agent (`agents/optimize/__init__.py`)

```python
class OptimizeInput(BaseModel):
    operator: str

class OptimizeOutput(BaseModel):
    operator: str
    status: str
    speedup: float | None = None
    test_passed: bool | None = None
    kernel_path: str | None = None
    optimization_direction: str | None = None

class OptimizeAgent(BaseAgent):
    name = "optimize"
    InputModel = OptimizeInput
    OutputModel = OptimizeOutput
```

静态单轮优化流程位于 `.kernelgen/agents/kernel-optimize.md`，frontmatter 声明中立 capabilities。动态 `operator` 与输出契约仍由 `BaseAgent.preprocess()` 注入；各 Runtime 使用同一角色定义。

#### Step 2: 定义 Workflow (`workflows/optimize.py`)

```python
class OptimizeWorkflow(Workflow):
    name = "optimize_loop"
    InputModel = OptimizeWorkflowInput   # operator + max_iters + target_speedup
    OutputModel = OptimizeWorkflowOutput # best_speedup + iterations_run + history

    def _execute(self, inp):
        rt = self._runtime_factory(str(self._cwd))
        agent = OptimizeAgent()

        for i in range(inp.max_iters):
            # 1. Agent 做一轮优化
            result = agent.run({"operator": inp.operator}, rt)

            # 2. Python 保存版本 + 更新历史(agent 下轮能看到)
            self._save_version(i, result.model_dump())
            self._update_performance_md()

            # 3. Python 判断停止条件
            if result.speedup and result.speedup >= inp.target_speedup:
                return {"status": "success", "target_reached": True, ...}

        return {"status": "success", "target_reached": False, ...}
```

#### Step 3: 入口脚本 (`examples/optimize_loop/run_example.py`)

```python
from kernelgen.workflows.optimize import OptimizeWorkflow
from kernelgen.framework.parallel import IsolatedDirectory, run_parallel

# 多个算子并行,每个算子内部迭代
inputs = [
    {"operator": "softmax", "max_iters": 20, "target_speedup": 1.5},
    {"operator": "layernorm", "max_iters": 20, "target_speedup": 1.5},
]
ws = IsolatedDirectory(
    base=Path("/tmp/optimize/operators"),
    claude_source=KERNELGEN_ROOT / ".claude",
)

results = run_parallel(OptimizeWorkflow, inputs, workspace=ws,
                       runtime_factory=make_rt, max_workers=4)
```

**关键点**:
- Agent 只做"单轮",Workflow 控制循环 —— 分离关注点
- Workflow 也是 Runnable,可以被 `run_parallel` 并行(多算子同时优化)
- 每轮更新 PERFORMANCE.md → agent 下轮看到历史 → 避免重复方向

---

### 12.3 kernel_gen — 最复杂模式:多 Agent 编排 + 多 Epoch

**场景**: 最完整的 kernel 优化:冷启动分析 → N 路并行(Coder→Distiller)
→ 选 best → epoch KB reducer → EpochSummary 出方向 → 下一 epoch。

**架构**:
```
KernelGenWorkflow
  AnalyzerAgent → 1R/shared_analysis (仅首次运行)
  for epoch in start_epoch..n_epoch:
    run_parallel(SingleCoderOptimizationWorkflow × N)   ← 嵌套 Workflow
      ├── CoderAgent (写码→preflight→eval→ProfileAnalyze→finalize→...)
      └── DistillerAgent (trajectory → experience/detailed 候选)
    pick_best (Python 权威选择)
    epoch KB reducer (候选校验/排序 → MergeAgent → git commit)
    EpochSummaryAgent → synthesis.json + next_directions
```

#### 入口脚本 (`examples/kernel_gen/run_example.py`)

```bash
python3 examples/kernel_gen/run_example.py \
  --definition flaggems_rsqrt \
  --n-parallel 3 --n-epoch 2 \
  --catalog-name flaggems-v5 \
  --eval-server "$KERNELGEN_SERVER_URL" \
  --target-hardware "$TARGET_HARDWARE"
```

脚本做的事:
1. `setup_workspace`: 快照 `.kernelgen` 中立定义并生成 `.claude/agents`、`.claude/skills`、`.codex/agents`、`.agents/skills` 和 Claude `.mcp.json`，从 repository KB 复制
   不含 `.git` 的初始内容，并只在 run-level `kb/` 初始化 Git
2. `load_definition`: 按名称从 Server 内置 Catalog 加载 Definition 和 Workload
3. 由 Workflow 写 `.kernelgen/tool-context.json`;Runtime 设置 PYTHONPATH 与工作区边界
4. 构造 `KernelGenWorkflow(cwd, runtime_factory)` → `wf.run(inp)`
5. 输出结果 + 验证(shared analysis / agent ledgers / synthesis / kb git log)

**关键点**:
- SingleCoderOptimizationWorkflow 本身是 Runnable → 被外层 `run_parallel` 并行
- Agent(Coder)内循环是长 invoke,不是 Python loop —— 因为需要连续推理上下文
- 每次编辑后的候选必须先通过服务端 `preflight_kernel`；失败不消耗 round
- 停止决策由原生 MCP `finalize_round` 调用底层纯函数并原子持久化 —— 不是 agent 自己决定
- verdict 按 round 持久化；缺失 verdict（仅旧 workspace）或 STOP 状态都会在
  preflight/eval 入口被硬拒绝，Workflow 只接受终态 STOP
- canonical KB 只有 epoch reducer 能写；并行 agent 只产生候选
- 多 agent 时每个 epoch 都持久化 `synthesis.json`，包括最后一个 epoch
- 可用 `--start-epoch 2 --n-epoch 2` 从 1R checkpoint 直接运行 2R；
  `--clean` 不能与 `--start-epoch > 1` 同时使用

---

### 框架复用总结

| 复杂度 | 模式 | Agent 数 | Python 控制 | 核心抽象 |
|--------|------|----------|-------------|----------|
| 简单 | Agent 直接 fan-out | 1 per task | 无 | `run_parallel(Agent, inputs)` |
| 中等 | Agent + Python loop | 1 per task × N rounds | 停止/版本 | `Workflow._execute` 内循环 |
| 复杂 | 多 Agent 编排 + 嵌套 Workflow | 5+ types | 选择/合并/方向 | `run_parallel(Workflow)` 嵌套 |

无论哪种复杂度,核心不变:
- `BaseAgent`: 契约(InputModel/OutputModel) + 动态 prompt + `preprocess/postprocess`
- `Workflow`: 编排(`_execute` 里组合 Agent/Python/run_parallel)
- `run_parallel`: 统一调度(Agent 和 Workflow 都是 Runnable)
- `ClaudeRuntime` / `CodexRuntime`: 执行层(provider event stream + MCP config + spawn/pump/idle-timeout/resume)

## 13. 运行示例(CLI 入口)

```bash
cd /path/to/kernelgen

# auto_gen: 批量生成算子(最简)
python3 examples/auto_gen/run_example.py \
  --input ops.jsonl --max-workers 4

# optimize_loop: 迭代优化(中等)
python3 examples/optimize_loop/run_example.py \
  --input ops.jsonl --max-iters 20 --target-speedup 1.5

# kernel_gen: 多 epoch 多 agent(最复杂)
python3 examples/kernel_gen/run_example.py \
  --definition flaggems_rsqrt --n-parallel 3 --n-epoch 2 \
  --eval-server "$KERNELGEN_SERVER_URL" \
  --target-hardware "$TARGET_HARDWARE"

# 已完成 1R 后，只运行 2R
python3 examples/kernel_gen/run_example.py \
  --definition flaggems_rsqrt --n-parallel 3 \
  --start-mode resume --start-epoch 2 --n-epoch 2 \
  --eval-server "$KERNELGEN_SERVER_URL" \
  --target-hardware "$TARGET_HARDWARE"
```

## 14. 测试

```bash
cd /path/to/kernelgen
# 全量 host 测试(FakeRuntime,无 LLM/GPU)
python3 -m pytest -q tests

# 真实 stdio initialize/list_tools/tool-call/gate 测试
python3 -m pytest -q tests/test_mcp_stdio.py

# CUDA MCP e2e；保留 workspace 供检查 ncu-details.txt、report.ncu-rep、sass.json、manifest 和 analysis
python3 tests/run_mcp_cuda_e2e.py \
  --server "$KERNELGEN_SERVER_URL" \
  --catalog-name flaggems-v5 \
  --kernel tests/fixtures/cuda_gelu.py \
  --hardware "$TARGET_HARDWARE" \
  --level instruction

# 完整工作流 e2e(需要 KernelGen Server + LLM API)
python3 tests/run_e2e_full.py
```

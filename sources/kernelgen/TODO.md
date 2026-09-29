# KernelGen TODO

## 近期

- [x] **Per-definition KB 双文档归并**：DistillerAgent 产出
  `.new_experience.md` 和 `.new_detailed.md`，epoch reducer 统一校验、排序并
  写入 canonical `experience.md` / `detailed.md`；只有 run-level KB 保存 Git
  历史。
- [ ] **跨 definition KB 文件**：
  - `optimization_patterns/{gpu}/{op_type}.md`
  - `debug_patterns/{op_type}.md`

- [x] **Coder round-finalize 与 STOP 硬护栏（P0）**：2026-07-23 的 910B-6
  `SimpleOptWorkflow + deepseek-v4-pro[1m]` 干净 e2e 中，Coder 没有保证每轮
  `eval_round → finalize_round → next`，最终跑了 13 rounds；离线检查 ledger
  确认当时应在 best round 10、连续 3 轮无提升后 STOP。Prompt 中的
  “STOP NOW” 当时不是系统级约束。现在 `finalize_round` 原子写入 conclusion
  和 Python 权威 `next_verdict`，并直接返回 CONTINUE/STOP；正常路径不再要求
  单独调用 `next`。STOP 后 preflight/eval 返回 `RUN_STOPPED`。Workflow 收尾
  还要求最新 round 存在终态 STOP verdict，不能接受在 CONTINUE 后提前返回的
  CoderReport。独立的 `next` MCP 和 CLI 均已删除。

- [x] **Cold-start 失败与性能 plateau 分开计数（P0）**：
  `Ledger.record_eval()` 现在只有在 round 完整 `PASSED`、存在 `geo_mean`
  且没有刷新 best 时才增加 `rounds_without_improvement`。编译、runtime、
  数值错误和 partial pass 仍作为 measured round 保留，但不消耗 performance
  plateau budget；加载 schema-v2 ledger 时也会从历史重新计算该派生计数。
  因此 `early_stop_rounds=1, min_rounds=1` 下首轮 `RUNTIME_ERROR` 不再触发
  plateau STOP，已有 ledger 和 inner-loop 回归测试覆盖。

- [x] **绝对 `max_round` 硬上限（P0）**：`max_round` 默认 15，统计所有已记账
  的 measured round。达到上限后 `finalize_round` 持久化并返回
  `max_round_reached` STOP，后续 preflight/eval 被硬拒绝。该硬触发器不受
  `min_rounds` 或 `soft_stop_disabled` 影响；preflight 失败以及未记账的
  transport/config/hardware mismatch 不计入上限。

- [x] **CLIRuntime wall-clock timeout 与子进程回收（P0）**：
  `_pump()` 已使用 reader thread + queue 定期检查 hard/idle deadline；
  CLI 在独立 process group 中运行，timeout、解析异常和 `KeyboardInterrupt`
  都会终止并回收进程组。host 测试覆盖静默超时、中断清理和 CLI 后代进程清理。
- [ ] **CLIRuntime 持续输出的 hard-timeout 回归**：补充一个 CLI 持续写 stdout
  但永不结束的测试，确认 output activity 不能延后 wall-clock hard deadline。

- [ ] **Workspace 单写者锁与 ledger 并发保护（P0）**：两个 E2E/Coder
  共用同一路径时会同时追加 `.ledger.json`，产生表面 PASSED、实际不可采信的
  混合结果。
  - Workflow 启动时在 workspace 建立包含 host/PID/run_id 的独占锁。
  - 已有活锁时 fail fast；陈旧锁必须先验证 PID/host，再显式恢复。
  - Ledger 写入增加文件锁或 compare-and-swap，避免并发 read-modify-write。
  - e2e 默认生成唯一 `WT`，并断言退出后没有该 workspace 的活动进程。

- [x] **标注并排除框架算子回退（P0）**：KernelGen Server 对 Triton
  Implementation 做保守 AST 检查，标记禁止的 Torch API、缺少
  `@triton.jit` kernel 或没有 Triton launch 的候选。`is_hack/hack_reason`
  原样进入 ledger；hack round 可以保留调试证据，但不能刷新 best 或覆盖
  `.best_kernel.py`。

- [x] **E2E LLM 配置统一（P0）**：仓库根目录的 `env.sh` 统一保存组内共享
  endpoint、token 和 `MODEL=deepseek-v4-pro[1m]`。e2e 从环境变量读取配置，
  缺失时 fail fast，不回退到远端 `.claude` 中的模型端点，也不在日志中
  打印 token。

- [x] **ProfileAnalyzeAgent 接入**:Claude 原生 subagent、immutable eval
  snapshot、统一 profile MCP、backend-neutral ProfileAnalysis，以及
  preflight/eval/finalize 对 pending profile/conclusion 的门禁已完成。
  STOP verdict 现由上面的原子 round-finalize 门禁强制。
- [x] **Profile 下游紧凑消费**：完整 profile analysis 保留在 artifact path；
  trajectory、Distiller 和 EpochSummary 使用 ledger 中的状态、摘要和路径，
  不把长报告重复注入 synthesis prompt。

- [ ] **Profile 证据升级与超时预算（P0）**：910B-6 真机 E2E 中，metrics
  已足够支持下一步 block-size 实验，但 ProfileAnalyzer 仍升级到 source；
  source 的 Ascend simulator 超时 600 秒后又继续请求 instruction，单轮额外
  消耗超过 20 分钟且没有新增证据。需要禁止 terminal timeout 后继续升级，
  并为 source/instruction 设置独立短预算；只有下一步修改确实依赖源码或
  指令定位时才允许升级。

- [x] **服务端 preflight gate**：Coder 自主调用 MCP `preflight_kernel`；
  KernelGen Server 统一负责版本化静态策略和逐 workload compile/smoke，精确候选
  receipt 通过后才允许 `eval_round`，失败不消耗 round。

- [ ] **Claude Code Hooks 硬护栏**:在 IsolatedDirectory 生成的 `.claude/settings.local.json` 里注入 hooks:
  - `PreToolUse(Write|Edit)` matcher `.ledger.json|.stop_config.json` → exit 2 阻止 agent 篡改
  - 确认当前 Claude Code hooks 对
    `mcp__kernelgen__eval_round` 和 `mcp__kernelgen__finalize_round` 的 matcher
    支持；若可用，用 `PostToolUse` 提醒完成 round-finalize。最终停止仍必须由
    Python 硬护栏保证，hook 只能作为辅助
  - `PostToolUse(Write)` matcher `tmp/main.py` → 可选自动记录 kernel 版本
  - 改动点: `framework/parallel.py` IsolatedDirectory._create() 追加写入 hooks 配置

- [ ] **独立 MCP workspace 准备 helper**：Workflow 模式已经自动写
  `ToolContext`；直接运行 MCP/E2E 时仍由脚本手工创建
  `.kernelgen/tool-context.json` 并设置 `KERNELGEN_WORKSPACE`。提供一个
  `prepare_workspace()` 入口，统一路径校验、上下文落盘和必要环境说明，减少
  测试/集成脚本重复代码。该项是易用性改进，不阻塞现有 workflow。

- [x] **Agent 完整日志写入 workspace**：人类可读日志保存到
  `.kernelgen/claude-runtime.log`；thinking、工具输入/结果完整保留并去重，
  `_cleanup()` 关闭文件。不额外保存体积较大的 provider 原始 stream。

- [x] **Epoch synthesis checkpoint / resume**：每个多 agent epoch（包括最后
  一个）保存 `synthesis.json`；`--start-epoch` 从 shared analysis、上一
  synthesis 和历史 agent ledgers 恢复。

- [x] **统一 op_type 枚举值**:Extractor role 使用固定枚举，postprocess 对常见别名做确定性归一化并拒绝未知类型。

## 中期

- [ ] **移除旧 910B 直传脚本**：`sync_to_910b6.sh` 会直接覆盖远端工作树，
  多设备任务已改用本地 Agent + 远端 Server，不再需要同步 KernelGen 工作树，
  禁止继续使用该脚本。Server 开发代码必须通过 Gitee 临时 test 分支同步；后续
  删除脚本，并把仍需保留的只读检查迁入部署/验收工具。

- [ ] **跨算子 KB 同步**:所有算子跑完后 cross-def 蒸馏 → 写共享 KB(by_op_type/insights + optimization_patterns)
- [ ] **对拍验证(#11)**:新框架 vs 旧 pipeline 比 geo_mean(同一个算子跑一次)
- [ ] **Docker workspace**:IsolatedDirectory → DockerWorkspace(每个 workflow 一个容器)

## 后续

- [ ] **Multi-workflow 并行**:批量优化 N 个算子(run_parallel(KernelGenWorkflow, ops, workspace=Docker()))
- [ ] **Epoch-level early stop**:连续 epoch 无提升则停(当前只有 round-level)
- [ ] **Seed 从 pick_seed 继承**(EpochSummary 提案 + 权威兜底)

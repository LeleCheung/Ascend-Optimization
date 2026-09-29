# Stage 编排重构：昇腾 CLI E2E 验证

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

## 结论与范围

2026-09-13，公共 StageRunner 重构完成 host 回归和真实 `kg run --mode extract_and_optimize` 后台验证。首次运行因最终独立复验的 reference profiler 计时无效而正确失败；显式重测通过后，CLI resume 完成优化结果恢复和真实代码审核。再次 resume 已完成任务，五个阶段全部复用，20 个业务文件哈希不变，没有新的模型或 Server 执行调用。本轮是“保留失败证据后恢复通过”，不是首次运行直接通过。

使用用户授权的已准备 Catalog，跳过源码抽取与 Catalog Review；Bundle 上传、目标 readiness、SimpleOpt 两轮搜索、独立复验及 Code Review 均真实执行。因此不能据此声明真实抽取、Catalog Review 或上游 Gems 源码契约全部通过。架构边界见 [Stage 编排设计](../design/runtime/stage_orchestration.md)。

## 代码与环境

KG 包版本为 v6.2.1，实际被测重构 commit 为 `48bcb5d25e8b1fa16313405e1a46d1403da45eb8`，分支 `refactor/stage-orchestration`，不是新发布 tag。纯重构 worktree 的相关 host 测试为 **267 passed**。

真实 CLI 测试使用独立临时 worktree `worktrees/kg-stage-orchestration-e2e`，在上述 commit 上叠加尚未提交的 `fix/catalog-e2e-followup` CLI 接入改动，包括 extract_and_optimize mode、输入 Catalog/skip_review、嵌套 ledger 状态及后台结果处理、KGS 锁定清单。新 workspace 接受 CLI 管理元数据的现有适配迁移到公共 StageRunner。该组合的相关 host 测试为 **268 passed**，差异保存在验证 JSON 的 `integration_diff` 中；这些业务改动不混入本次重构提交。两套测试都通过临时父目录的 `kernelgen` 链接隔离导入，并核对实际模块路径。

实际配套为 **KG v6.2.1@48bcb5d2 + 上述 CLI 集成快照 / KGS v6.3.1@c309cb351d2a473e79dcad80a90df4363d5b9fb7 / Protocol v6.2**。协议和功能依据目标 `/status.api_version`、`capabilities` 判断；Bundle evaluation binding 和 operation cancellation 均启用。

| 项目 | 实测配置 |
| --- | --- |
| Agent | 本地 Python 3.12，Claude Runtime，`deepseek-v4-flash[1m]`，Coder lease 上限 1 |
| KGS 实例 | 复用 `ascend-20260910`，容器 `codex_fib_ascend_20260821` |
| 连接 | 本地 SSH stdio proxy `127.0.0.1:19606` → 远端 loopback `18306` |
| 目标 | Ascend910B4-1，backend `npu`，timing `profiler`，可见设备 `npu:0`～`npu:7` |
| 调度 | 8 个设备 slot，16 个请求 workers；单个请求由 KGS 分配 slot，未固定独占某张卡 |
| Server 解释器 | `/home/secure/xuyao/kernelgen_e2e_ascend_20260821/deployments/ascend-managed-20260910/venv/bin/python`，Python 3.11.15 |
| 核心包 | Torch `2.9.0+cpu`、torch-npu `2.9.0.post2`、flagtree `0.6.0+ascend3.5`；Triton 模块 `3.5.1` 由 flagtree 提供 |
| 目标运行时 | CANN `9.0.0`，driver `25.2.0` |
| 搜索 | relu，`max_round=min_rounds=2`，warmup/benchmark 各 100 ms，1 trial，Eval timeout 1500 s |

通过实际 Server PID 的工作目录和解释器核对远端 checkout 为精确 c309cb3 且 clean。首次环境诊断使用普通 `python3` 命中了旧 PATH 解释器，不代表实际 KGS；纠正后按 Server 自身解释器读取环境，前后快照一致。本轮没有安装包、修改受保护运行时、同步远端代码或重启 Server。

## 实测结果与失败恢复

输入保留 fixture 的 18 个 correctness 和 15 个 timing workload，native oracle 为 `torch.relu`。运行结束校验 definition、correctness、timing 和 oracle 文件与输入 fixture 字节一致。下列数值来自 `verification-before-resume.json` 对 readiness、ledger 和最终独立复验的读取。

| 测量 | 结果 | workload | 几何平均加速比 |
| --- | --- | --- | --- |
| readiness / reference-as-solution | PASSED | 33/33 | 0.9996476776385238 |
| measured round 1 | PASSED | 33/33 | 0.7236414684073575 |
| measured round 2 | PASSED | 33/33 | 0.9207372684916427 |
| 首次最终独立复验 | PARTIAL_PASS | 32/33 | 不发布有效最终加速比 |
| 显式重新独立复验 | PASSED / VERIFIED | 33/33 | 0.9158924371596344 |

首次最终复验中，`benchmark/test_relu.py::test_relu::core::float16::3` 的 reference profiler 返回 `inf us`，KGS 报 `Ascend profiler produced invalid timing (inf us); walltime fallback is disabled`。这是无效计时，不是已证实的候选数值错误，也不能据单次重测推断其根因已修复。CLI 正确显示 FAILED，优化结果为 NEEDS_RETEST，已完成 3/5 阶段，Code Review 尚未执行；没有提前报告 SUCCEEDED。KGS slot 未进入 broken，incidents 保持 0。

保留首次输出、失败 receipt、原始 retest 及 ledger 哈希后，在确认后台进程退出并持有 workspace 独占的条件下，通过现有 `request_retest` API 显式申请第二次完整复验。未修改候选、reference、workload 或计时设置，也未启用 walltime fallback；搜索 ledger 哈希保持不变。第二次复验全部通过且没有 drift。随后真正执行 CLI resume，前三阶段复用，optimize 创建 attempt 02，复用已确认的最终复验并保持两个 measured round，之后真实 Code Review 完成。

最终结果为 SUCCEEDED、5/5 阶段完成、进程 exit 0。Code Review 检查 8 个文件，0 个 P0、4 个 P1；其中应保留的边界包括缺少完整上游 Gems 源码证据，以及最终性能未超过 baseline。最终加速比约 **0.916×**，达到当前 0.8 合格线，但不是性能提升。best code、最终输出和 ledger 的候选一致，SHA-256 为 `65f4febdba5a6af6ed6960f5dba23fdda08461ef69a471eef123e1820f08d8f8`。

## Resume、状态和资源验收

恢复成功后再次执行 CLI resume，收到按顺序排列的五个 `LIFECYCLE_STAGE_REUSED` 事件：extract_catalog、review_catalog、distribute_catalog、optimize、code_review。该次执行没有新的阶段启动、Server operation 或模型调用事件；包括失败旧 receipt、成功 receipt、ledger、最终输出及 best code 在内的 20 个业务文件哈希不变。

纯重构版本还完成跨进程 dummy FAILED、WAITING、CANCELLED 后恢复检查：成功步骤复用、历史 receipt 不覆盖，三种场景均最终 SUCCEEDED。该检查不等同于真机在途 operation cancellation 测试，后者本轮未重新注入。

真实后台测试持续采样 CLI JSON 和 KGS status，汇总时已有 110 份观测。根阶段、完成数、失败与恢复终态符合执行事实；结束时无活跃 operation，Coder lease 使用数为 0，KGS `healthy=available=device_slots=8`、`active=waiting=checking=broken=incidents=0`。复用的常驻 KGS 和 proxy 保留运行，未创建需回收的临时 Server。

仍有明确限制：模型轮内执行 Debug 调参时，优化 scope 有时停留在最近写入的 `FINALIZING_ROUND`，并非实时 Debug 子阶段；这属于现有进度写入粒度，本次没有修改。`max_round=2` 限制 measured round，不限制轮内 Debug 调用次数，因此不能据此估算整轮耗时。独立的 `refactor/stage-progress-schema` 新 JSON Schema 未纳入本轮测试；此处验证的是现有 Schema。未注册 CLI 后台请求的前台 dummy campaign 仍需通过 campaign 状态 API 查询，不作为 `kg status` 已接入证据。

## 原始证据与查看命令

本机忽略目录中的证据根路径为 `/data/akg_kernel_bench_lite/kernelgen/.kernelgen/experiments/stage-orchestration-e2e-0rv5Si`，不随 Git 分发。保留了 `input.json`、`run/`、`observations/`、`first-failure-evidence.json`、`explicit-timing-retest.json`、`verification-before-resume.json`、`verification-after-resume.json`、`cli-resume-checks.json`、Server 身份与前后环境快照，以及验证脚本。运行目录包含请求、进程和事件记录、每阶段 receipt、Agent 日志、ledger、最终输出与 best kernel。

在当前测试机器、临时导入链接仍存在时，无需激活环境即可查看实际后台任务；以下命令依赖上述 CLI 集成 worktree，不表示纯重构分支已包含新的 mode：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/kg-stage-e2e-import-br0EAm:/data/akg_kernel_bench_lite/kernelgen_server python3 -m kernelgen.cli.main status /data/akg_kernel_bench_lite/kernelgen/.kernelgen/experiments/stage-orchestration-e2e-0rv5Si/run --json
```

去掉 `--json` 可看文本摘要。通用启动、监控和回收要求以 [多设备实验手册](../operations/multiple_device_experiment_runbook.md) 为准，不从本报告的历史容器、端口或路径推导新实验配置。

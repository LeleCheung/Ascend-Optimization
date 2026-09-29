# Workflow build 重构：昇腾 CLI 下游 E2E

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

## 结论与范围

2026-09-13，在 `refactor/stage-orchestration` 的原 worktree `worktrees/kg-stage-orchestration` 完成真实 CLI 后台 ExtractAndOptimize 下游 E2E，首次直接 SUCCEEDED，五个步骤均为 attempt 01，进程 exit 0。随后真正执行 CLI resume，五个步骤全部复用，19 个业务文件哈希不变，没有新增模型调用或 Server operation。相关 host 回归为 **289 passed**。

使用与[前一轮 Stage 编排 E2E](stage_orchestration_ascend_20260913.md)相同的 prepared relu Catalog 和参数：跳过源码抽取与 Catalog Review；Bundle 分发、目标 readiness、SimpleOpt 两轮、最终独立复验、Code Review 均真实执行。该结果不覆盖真实抽取、Catalog Review、KernelGen 多 Coder 或真机在途取消。设计见 [Workflow 编排设计](../design/runtime/stage_orchestration.md)。

## 被测代码与环境

实际组合为 **KG v6.2.1@6b2c83b6 + 未提交 CLI 集成快照 / KGS v6.3.1@c309cb351d2a473e79dcad80a90df4363d5b9fb7 / Protocol v6.2**，不是新发布 tag。CLI 集成快照包含 extract_and_optimize mode、输入 Catalog/skip_review、嵌套 ledger 查询、后台结果处理、CLI 管理元数据适配和 KGS lock；完整差异保存在 `verification-before-resume.json` 的 `integration_diff`。按用户要求，集成代码已迁入原分支开发和测试，本次真实运行没有从临时 E2E worktree 导入代码。

通过临时父目录中的 `kernelgen` 链接隔离导入，核对 KG 实际路径为 `/data/akg_kernel_bench_lite/worktrees/kg-stage-orchestration/__init__.py`，KGS client 为 `/data/akg_kernel_bench_lite/kernelgen_server/kernelgen_server/__init__.py`。host 回归覆盖 Workflow composition、OperatorDevelopment、CatalogOptimize、ExtractAndOptimize、CLI/lifecycle/history/resume、run control/cancellation、SimpleOpt、KernelGen 及 evaluation snapshot，共 289 passed。

提交 PR 前追加 `tests/test_cli_catalog.py`，覆盖 launcher 的 runtime/model、resume 和退出码、缺少 input 时的前置拒绝、外层不重复申请 lease、等待状态保留以及 Catalog Review bypass 不移除 Code Review。整理时补上 runtime/model 传递并避免该模式 resume 拼接 KernelGen epoch 参数；完整相关 host 门禁为 **298 passed，1 个现有 Starlette 弃用警告**。上述真机证据对应整理前快照，整理后的两项 launcher 改动由 host 测试验证，没有将其冒称为另一轮完整真机 E2E。

| 项目 | 实测配置 |
| --- | --- |
| Agent | 本地 Python 3.12，Claude Runtime，`deepseek-v4-flash[1m]`，Coder lease 上限 1 |
| KGS | 复用 `ascend-20260910`，容器 `codex_fib_ascend_20260821`，实际 PID 4116011 |
| 连接 | SSH stdio proxy `127.0.0.1:19606` → 远端 loopback `18306` |
| 设备 | Ascend910B4-1，backend `npu`，timing `profiler`，可见 `npu:0`～`npu:7` |
| 调度 | 8 个独立设备 slot，16 个请求 workers；单个请求由 KGS 分配 slot |
| Server Python | `/home/secure/xuyao/kernelgen_e2e_ascend_20260821/deployments/ascend-managed-20260910/venv/bin/python`，3.11.15 |
| 核心包 | Torch `2.9.0+cpu`、torch-npu `2.9.0.post2`、flagtree `0.6.0+ascend3.5`；Triton 模块 `3.5.1` 由 flagtree 提供 |
| 运行时 | CANN `9.0.0`，driver `25.2.0` |
| 参数 | relu，`max_round=min_rounds=2`，warmup/benchmark 各 100 ms，1 trial，Eval timeout 1500 s |

Server 实际 checkout 为精确 c309cb3 且 clean；协议与 Bundle evaluation binding、operation cancellation 以目标 `/status` 为准。前后实际 Server 解释器、包版本和 checkout 快照一致，没有安装依赖、替换运行时、远端同步或重启 KGS。

## 结果与状态

输入保留 18 个 correctness 和 15 个 timing workload，oracle 为 `torch.relu`。结束时逐字节核对 definition、correctness、timing、oracle 与输入 fixture 相同。以下数值来自验证脚本读取的 readiness、ledger 和最终输出。

| 测量 | 结果 | workload | 几何平均加速比 |
| --- | --- | --- | --- |
| readiness / reference-as-solution | PASSED | 33/33 | 1.0012920110047216 |
| measured round 1 | PASSED | 33/33 | 0.3120512376625977 |
| measured round 2 | PASSED | 33/33 | 0.8652470207470194 |
| 首次最终独立复验 | PASSED / VERIFIED，无 drift | 33/33 | 0.8635692376429928 |

最终复验约 0.864×，达到当前 0.8 合格线，但不表示超过 baseline。best code、ledger 和最终输出一致，候选 SHA-256 为 `9ee5ffe05860f5ea7854255efa4cf5fdc1b837d03c6c28bf07d25999ddfe1565`。本轮没有重试最终复验或修改候选来恢复失败；与上次结果比较的是流程语义，而不是要求模型生成完全相同的代码或加速比。

真实 Code Review 检查 8 个文件，返回 0 个 P0、3 个 P1，涉及 fp32 tile 计算与设计意图差异、连续内存假设及特殊值语义覆盖。这是审核 Agent 的原始结论，不是本次人工复核的三项已证实缺陷；例如报告中关于 `view` 会复制的表述不应当作事实。没有为使本次 E2E 通过而修改 Catalog 或 candidate，也没有扩大 review skill 的覆盖规则。

持续采样得到 69 份 CLI JSON/KGS status 观测，看到 distribute_catalog、optimize 的 PREFLIGHT/EVALUATING/RETESTING、code_review，以及最终 COMPLETED。Code Review 执行时根任务仍为 RUNNING、完成 4/5；只有全部完成才出现 SUCCEEDED、5/5。CLI 文本的 `best_geo_mean` 为 ledger 中的 0.8652470207470194，最终独立复验为 0.8635692376429928，两者归属不同，不应强求数值相同。

仍有现存粒度限制：模型 Debug 调参时 scope 有时保持最近的 `FINALIZING_ROUND`，不是实时 Debug 子阶段；`max_round=2` 不限制轮内 Debug 调用次数。完整真机生成轮次时尚未整合新进度 Schema；后续 Schema 验证见下节，不能将展示改动宣称为再次完整生成。

## 后续进度 Schema v2 验证

按用户要求将 refactor/stage-progress-schema 整合到 !109 后，对同一个已完成任务再次执行真实 CLI resume（PID 1320185）。全部五个调用复用，19 个业务文件哈希保持不变，没有新增模型或 Server operation，终态 SUCCEEDED、exit 0。`verification-progress-v2.json` 保存了本次实际 `--detail` 和文本输出。首次完整 E2E 和之前 resume 的证据文件保持不动。

`kg status --detail` 返回 Schema 2.0，Extract、Catalog Review、分发和 Code Review 的 `progress_kind=basic`、`progress={}`，没有 round/epoch/performance 字段。Optimize 的 rounds 详情保留两轮和 ledger 最佳加速比，根 tasks 详情为 5/5。文本仍显示 tasks 和 best_geo_mean。原 `kg status --json` 删除，其他命令的 --json 不变；这不是仅过滤 null 的展示改动，类型由 Workflow 显式声明。

整合后的相关 host 门禁为 **307 passed，1 个现有 Starlette 弃用警告**；包括独立进度类型、未知类型拒绝、ledger 指标禁止直接写入、CLI detail/text、Batch 缺失子任务、非 status 命令参数不变，以及既有 Workflow/CLI/Service 回归。没有新增依赖或重跑 GPU/NPU 生成评测。

## Resume 与资源释放

已完成任务的真实 CLI resume 启动了新后台进程 1191357（首次进程 1001475），最终 exit 0。依次复用 extract_catalog、review_catalog、distribute_catalog、optimize、code_review，未创建新的业务 attempt。19 个业务文件哈希不变，事件中没有新增步骤执行、模型调用或 Server operation。

结束时无活跃 Agent 或登记的 operation，Coder lease 使用数为 0，KGS `device_slots=healthy=available=8`、`active=waiting=checking=broken=incidents=0`。复用的常驻 KGS 和 SSH proxy 保持运行，本次没有创建临时 Server。

## 原始证据与查看

证据根目录为 `/data/akg_kernel_bench_lite/kernelgen/.kernelgen/experiments/workflow-build-e2e-SWhnCa`，属于本机忽略数据，不随 Git 分发。保留 input、run、observations、Server 身份、前后环境快照、`verification-before-resume.json`、`verification-after-resume.json` 及验证脚本。运行目录保留 CLI 请求/进程/事件、各步骤 receipt、Agent 日志、ledger、最终输出和 best kernel。

在当前机器且导入链接仍存在时查看：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/kg-stage-orchestration-import-mxW0cs:/data/akg_kernel_bench_lite/kernelgen_server python3 -m kernelgen.cli.main status /data/akg_kernel_bench_lite/kernelgen/.kernelgen/experiments/workflow-build-e2e-SWhnCa/run --detail
```

去掉 `--detail` 查看文本摘要。后续实验配置仍以[多设备实验手册](../operations/multiple_device_experiment_runbook.md)、当前 host inventory 和 KG lock 为准，不复用报告中的历史 PID 或假定环境永久不变。

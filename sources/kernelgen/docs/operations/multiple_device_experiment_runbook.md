# 多设备实验与 E2E 验收手册

本文维护当前 CLI 实验流程，不保存机器部署快照。首次远端部署见 [远端 KGS 部署](deployment/remote_server.md)；Kernel Todo V2 的状态流转与专用工具见 [实验手册](experiments/kernel_todo_v2_experiment_runbook.md)；原始验证证据见 [验证记录](../validation/multiple_device_e2e_validation_report.md)。

## 1. 确定输入与环境

新实验以当前 KG 的 [KGS lock](../../deployment/kgs.lock.yaml) 选择配套 repository、release 和 exact commit。KGS 的 compatibility.yaml 提供自身描述和 FlagGems 默认 revision，不使用历史 latest_validated_kgs 反向矩阵选版本。KG、KGS release 和 Protocol 独立记录；协议只检查 /status.api_version，可选能力只检查 /status.capabilities。

连接入口、已有容器和远端 Server 端口取自 [机器清单](../../tests/hosts.md)；卡号必须现场确认并获得授权，清单不分配设备。历史实验获准使用全部卡不代表新实验也获准。镜像不是 KG 自动选择或创建的资源，见 [容器与镜像边界](deployment/remote_server.md#容器镜像与解释器)。

KG、模型 Runtime、Catalog 和 workspace 留在本地；远端 KGS 负责编译、执行、计时和 Profile。新任务默认使用 kg；只有旧 Python campaign 原地续跑、专用抽取/回写工具或共享 Knowledge campaign 才使用相应 Python 入口。不要把旧 campaign 的状态目录转换成 CLI run。

启动前记录 KG commit、锁定及实际 KGS commit、Catalog manifest/摘要、实际 Gems commit、容器和解释器、核心运行时版本、设备授权、backend/timing、模型、run name 和 workspace。远端事实通过 KGS 获取，不用 Agent 本机 Torch/Triton 推断。禁止安装或替换 Torch、Triton、torch_npu 和厂商运行时。

## 2. 启动并检查 KGS

首次部署按 [远端指南](deployment/remote_server.md) 配置命名实例；后续从同一 CLI 工作目录运行：

```bash
kg server start <name>
kg server status <name>
kg server status <name> --json
```

完整 JSON 的 server 是远端 /status；目标硬件名称取自 server.target.device。任务 endpoint 是本地 loopback URL，远端模式使用本地代理端口，不是 SSH 端口或远端 Server 端口。

必须满足 scheduler.device_slots == healthy == 已授权设备数、checking=broken=0、max_active <= device_slots；空闲时 active=waiting=0、available=device_slots。请求线程 max-workers、Agent Coder 配额与设备 slot 是不同资源，不互相替代。

跑 Gems 代码前必须在 KGS 实际执行环境准备 FlagGems，只有本地 Agent 安装不够。在未提交任务、实例可以停止时执行：

```bash
kg server stop <name>
kg server install-flaggems <name>
kg server start <name>
kg server status <name>
```

Gems 来源策略来自锁定 KGS 的 compatibility.yaml。branch policy 下，新实验的上述准备步骤自动解析默认分支最新 commit，安装到独立 checkout；KGS 自身仍使用 lock 中的 exact commit。默认分支实例的后续 start 也会检查更新，但不会替换仍在运行的服务；旧实验结束后再 stop/start。显式完整 commit 选择及旧 campaign 续跑保留原快照，不重新追踪分支。多机同一实验复用一次解析的 commit，更新后重新 inspect 和验证 baseline；FlagGems-backed native 的冻结 oracle 继续遵守 manifest exact revision。完整边界见 [Gems 来源策略](../design/gems_source_policy.md)。

## 3. 先 Smoke，再 Batch

先选择已经通过 inspect、reference 和 baseline 门禁的简单算子，例如 Catalog 实际包含的 relu、add 或 square。不要用复杂 Attention 或矩阵乘判断部署健康；不裁剪 workload 来伪装通过。下面是占位模板，替换后执行：

```bash
kg run --mode simple_opt --definition <definition> \
  --catalog-name <catalog> --target-hardware "<server.target.device>" \
  --eval-server <local-loopback-url> --min-rounds 1 --max-round 1
kg status <workspace>
kg history <workspace>
```

SUBMITTED 只表示独立后台进程已提交；SUCCEEDED 表示 Workflow 完成，不自动表示性能达标。Smoke 要核对 ledger 中 correctness 和有效 headline timing，以及任务前后 scheduler。昇腾正式性能使用 profiler，其他设备按已验证 backend/timing 配置；计时失败不得回退 wall time。

新 Batch 使用 [CLI YAML 格式](../design/kg_cli.md#yaml-batch)：

```bash
kg config set run.max-workers <coder-limit> --eval-server <local-loopback-url>
kg run --batch-file <batch.yaml>
kg status <batch-workspace>
kg status <batch-workspace> --detail
```

每个算子独立 workspace 和后台进程，Batch 根只保存子任务索引。SimpleOpt 占 1 个 Coder lease，KernelGen 占 n_parallel 个；不要把 n_parallel 个并发 Coder 错当成同一 SimpleOpt workspace 的重复进程。配额以实际空闲资源和当次模型限流为依据；2N 只能作为已授权实验的调度窗口，不要求拆分 campaign，也不能增加每卡 Eval 并发。

当前默认不为昆仑芯启动批量优化；只有明确授权诊断时做单算子低并发实验。Kernel Todo V2 的具体并发和验收约定由其 [专用手册](experiments/kernel_todo_v2_experiment_runbook.md) 管理，不把某次模型 endpoint 限额当作项目永久配置。

## 4. 同一候选的跨芯片复测

已有 Gems pytest 导出的 Definition 和一份生成候选时，可调用 [MultipleDeviceTest Workflow](../design/workflows/multiple_device_test.md) 在每个目标 KGS 上先跑候选无关的 core `reference-only`。目标 reference 失败时记录逐 case 原因并停止该目标的候选测试；只有通过后才继续 Preflight 和 Eval。各目标保留独立结果，不把一个芯片的 API/dtype 能力推断到另一芯片。目标地址必须是本地 SSH stdio HTTP proxy 的 loopback URL；当前 Workflow 使用 Python API，不属于 `kg run` 的优化模式。

## 5. 监控、取消与续跑

```bash
kg status <single-run-workspace>
kg status <single-run-workspace> --detail
kg history <single-run-workspace> --json
kg logs <single-run-workspace> --follow
kg logs <single-run-workspace> --raw
kg cancel <workspace>
kg resume <single-run-workspace>
```

任务 status 使用 --detail 输出 v2 JSON；history、list、server status 仍使用 --json。普通调用只展示自己的 progress 类型，不要求所有 scope 都有 epoch/round 字段。性能由 ledger 投影，日志不是状态真源。

取消是协作式请求，Runtime 在完整模型输出后退出；KGS 声明 operation_cancel 时同时转发活动 operation。不要直接 kill Agent。Batch 可以整批取消，但不支持根目录 resume；根据 status 选择尚未完成的子 workspace 续跑。组合 Workflow 已完成且产物有效的调用复用 receipt，普通未完成调用新建 attempt 重跑，只有 Optimize 恢复内部 session/ledger。

启动初期每 30 秒检查推进，稳定后按实验窗口每 15 分钟检查；窗口边界记录 scheduler 和 incidents。TIMEOUT 或 worker 异常后原 slot 先 checking，强探针通过才恢复；不跨卡重试，不自动复位物理卡。broken>0 或 incidents 在同一窗口反复增长时，保留证据并在当前窗口结束后人工检查，不开启新窗口。具体事故记录见 KGS 的 device_slot_isolation_incident.md。

## 6. 验收、归档与停止

流程状态、数值计时结果和性能目标分别判定：PASSED 要求完整正确性及有效计时，但不等于加速比大于 1。Kernel Todo V2 的 >=0.8 是该实验的合格线，不是所有 Workflow 的通用成功定义。

保留启动命令、Server/Agent 日志、输入快照、最终 JSON、ledger、best kernel 和 scheduler 前后快照；按 [结果整理手册](experiments/experiment_result_maintenance.md) 使用汇总工具生成报告，不手抄加速比。新 CLI 的产物在本地，不要调用远端回收脚本去找它们。

tests/check_multi_device_batch_results.sh 和 collect_multi_device_batch_results.sh 面向旧的远端 Python campaign 布局；只有该类实验才按原 run name 和部署目录使用。历史命令见 [历史 E2E 报告](../validation/multiple_device_e2e_validation_report.md)，不能套用于新 CLI Batch。

确认本轮 Agent/Coder 全部结束、Server slot idle、证据归档后，只关闭自己创建的临时实例：

```bash
kg server stop <name>
```

共享 Server 不因本轮结束而关闭。操作方法修改本文或专用手册；实际版本、镜像、端口及结果新建带日期的 validation 报告，不回填为永久部署默认值。

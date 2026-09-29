# Kernel Todo V2：KG CLI 实验手册

本手册只维护当前 Kernel Todo V2 的一条操作流程：NVIDIA 使用 Gems adapter 优化，复测同一候选在各目标芯片上的可移植性；不能直接复用或性能未达标时，在目标芯片再运行 Gems adapter 优化，最后回到 FlagGems 验收和归档。正式优化统一使用 `kg run` 默认的 KernelGen；只有明确需要 SimpleOpt 模式时才显式选择。本流程**不做 Native 二阶段优化，也不做 Catalog 抽取**。历史 Native 实验及其证据保留在 Git 历史和对应 validation 报告中，不作为新任务的启动步骤。

`kernel_todo_v2/` 是本地忽略的实验输入，不随新 clone 分发。开始前从授权归档恢复清单或重新生成；不能把脚本默认路径当成仓库自带数据。首次安装和两类算子输入见 [ONBOARDING](../../ONBOARDING.md)，远端 KGS、SSH stdio proxy、设备 slot 与 Batch 资源门禁见 [多设备实验手册](../multiple_device_experiment_runbook.md)，当前机器与容器只以 [hosts.md](../../../tests/hosts.md) 为准。

## 1. 流程与事实来源

```text
NVIDIA kg run（默认 kernelgen）
  → MultipleDeviceTest：每台先 reference-only，再测同一份 NV 候选
  → 不能复用或低于本实验门槛的目标：kg run --reference-code-path <NV best>（默认 kernelgen）
  → FlagGems 原 pytest/benchmark 最终验收，归档结果
```

每个 `kg run` 有独立 workspace。`kg status` 是进度，`kg history` 和该 workspace 的 ledger 是逐轮性能事实；最终独立复验可能与搜索时 best geo mean 略有不同。跨芯片 Workflow 的 `workflow_result.json` 是按目标解析的入口，逐 case 的原始 KGS 响应仍保存在 `targets/<name>/`。`SUBMITTED`、Workflow `SUCCEEDED`、正确性和性能达标不是同一个结论。本实验仍以**完整正确性、有效计时且最终 geo mean ≥0.8**为达标线；不满足时保留有效失败或阻塞证据，不伪造成功。

## 2. 准备输入和目标 KGS

记录 `KG release@commit / KGS release@commit / Protocol`、实际 Gems commit、容器与解释器、backend/timing、可见物理卡、模型、run name 和 workspace。协议由 KGS `/status.api_version` 判断，功能由 `/status.capabilities` 判断；不能按软件 release 或 Agent 本机 Torch/Triton 推断目标能力。不得安装或替换 Torch、Triton、torch_npu 或厂商运行时。已配置的命名实例在任务开始前执行：

```bash
kg server start <name> --install-gems
kg server status <name>
kg server status <name> --json
```

首次 `start` 的 remote、backend、container、设备和 loopback 端口参数见 [远端部署](../deployment/remote_server.md)。只用一个本地 SSH stdio HTTP proxy 连接远端 loopback KGS，不开放远端端口。新实验在启动前固定并记录实际 Gems commit；已经运行的 Server 不会原地替换 checkout，旧 campaign 续跑也不能中途换 KGS、Gems 或测试契约。空闲时要求 scheduler `device_slots=healthy=available=已授权卡数`，`active=waiting=checking=broken=0`；Agent Coder lease、KGS 请求线程和设备 slot 不能混为一项并发配置。

算子若已在 KGS 的 Gems adapter Definition 中，直接用 `--definition <operator>`；新的 Gems pytest 先在本地**导出一次** Definition，再把同一份 `catalog_path` 交给各目标。`kg definition` 不是 Native Catalog 抽取，不执行 pytest，也不能代替后续目标验证：

```bash
kg definition --flaggems-repo /path/to/FlagGems \
  --pytest-path tests/test_<operator>.py --workspace <definition-workspace>
```

来源 checkout 必须是已提交且干净的实际 Gems 版本，并与目标 KGS 使用的版本一致；若 Definition、原 pytest、benchmark、ABI 或 case list 无法被当前协议准确表达，记录阻塞，不删测例、不改 reference 语义。下文的 `<catalog-path>` 指导出命令返回的路径；使用 KGS 已安装 Definition 的单机优化可以省略 `--catalog-path`。

## 3. NVIDIA：生成并确认候选

先用一个简单算子做 1 round Smoke，再对清单中已过目标门禁的算子提交正式任务。下面命令使用本地导出的 Definition；已安装在 KGS 的算子可去掉 `--catalog-path`：

```bash
kg run --definition <operator> \
  --catalog-path <catalog-path> --eval-server <nv-local-kgs-url> \
  --min-rounds 1 --max-round 1
kg status <nv-workspace>
kg status <nv-workspace> --detail
kg history <nv-workspace> --json
```

Smoke 通过仅证明链路可用，正式任务使用新 workspace；未显式设置时，KernelGen 使用 1 个 Coder、1 个 epoch、最多 10 rounds，可按当次预算调整。`kg run` 默认后台提交；只有 `status` 终态、获胜 Coder 的 ledger 有全量正确且有效计时的 measured round、最终复验通过，才能冻结该 Coder 的 `.best_kernel.py` 作为跨芯片输入。获胜路径从 `stages/optimize/attempts/<编号>/result.json` 的 `result.data.candidate` 读取，不要假定它位于 `stages/optimize/work/` 根目录。正式多算子任务用 [CLI YAML Batch](../../design/kg_cli.md#yaml-batch)，每个算子仍有独立进程和子 workspace；`run.max-workers` 是本地 Coder lease，不是 Server device slot。旧 Python campaign 只在原 workspace 用原版本续跑，不把历史目录转换成新 CLI run。

## 4. 同一 NV 候选的跨芯片复测

使用 [MultipleDeviceTest Workflow](../../design/workflows/multiple_device_test.md) 冻结一份 Gems Definition Bundle 和一份 NV best code。即使 NVIDIA 优化时用了 KGS 内置 Definition，跨目标复测仍需从**同一原 pytest 和 Gems commit**导出一份本地 Definition，以便上传同一内容寻址 Bundle。目标 URL 必须是各自本地 SSH stdio proxy 的 loopback 地址；下例中的路径和端口均为占位符：

```python
from kernelgen.workflows.multiple_device_test import MultipleDeviceTestWorkflow

result = MultipleDeviceTestWorkflow(cwd="<new-test-workspace>").run({
    "operator": "<operator>",
    "catalog_path": "<catalog-path>",
    "candidate_path": "<nv-best-kernel.py>",
    "targets": [
        {"name": "ascend", "server_url": "http://127.0.0.1:<ascend-proxy-port>"},
        {"name": "hygon", "server_url": "http://127.0.0.1:<hygon-proxy-port>"},
    ],
})
print(result.model_dump_json(indent=2))
```

每台先读取 `/status`、上传并 inspect，再运行候选无关的 core `reference-only`。reference 失败、全 skip 或任一 core case 被跳过时，不提交该目标的候选；通过后才执行 Preflight 和完整 Eval。KGS/Gems 不可用、来源不匹配或设备故障属于环境阻塞，不能记为 API/dtype 不支持；候选编译、正确性或硬件网格失败应保留为候选结果。顶层 `state=FAILED` 只表示至少一个目标未通过，仍要逐个读取 `output.results`；全部原报告和最终 `workflow_result.json` 保留在新 workspace。本次真实 `relu6_` 结果和阻塞记录见 [跨芯片验证](../../validation/multiple_device_relu6_20260925.md)。

## 5. 目标芯片：必要时用 NV 代码作 reference 再优化

若目标候选不能运行、全量正确但最终加速比 `<0.8`，或用户直接指定该芯片的待优化算子，在该芯片自己的 KGS 上启动新的默认 KernelGen workspace。先确认该目标的原 pytest/reference 可运行；不能拿 NVIDIA 的 baseline 代替目标 baseline。NV best 只作为只读代码参考，不是 `--seed-code-path`，不继承 NV ledger、速度或正确性：

```bash
kg run --definition <operator> \
  --catalog-path <same-catalog-path> --eval-server <target-local-kgs-url> \
  --reference-code-path <nv-best-kernel.py>
kg status <target-workspace> --detail
kg history <target-workspace> --json
```

已安装在目标 KGS 的 Definition 可省略 `--catalog-path`。同一测试契约已经独立审核且目标 reference 通过时，实验可以显式使用 `--skip-review` 跳过重复的模型审核；它不会跳过 baseline、候选 Preflight/Eval 或代码审核。正式实验应显式指定并记录 `--model`。参考代码是给 Agent 的设计材料，模型可能复写它；必须比较候选与参考文件 SHA、目标 ledger 和最终复验，不能仅因传了 `--reference-code-path` 就宣称完成芯片特化。[昇腾 CLI 实测](../../validation/ascend_nv_reference_cli_20260925.md) 是一次显式 `simple_opt`、1 round 的链路 Smoke，记录了 best code 与 NV 参考相同的边界；它尚不能证明默认 KernelGen 的正式优化效果。

## 6. 监控、验收与收尾

```bash
kg status <workspace>
kg status <workspace> --detail
kg history <workspace> --json
kg logs <workspace> --follow
kg cancel <workspace>
kg resume <single-run-workspace>
```

取消是协作式的，Runtime 在完整模型输出后响应；KGS 声明 operation cancellation 时同时取消已登记的执行请求，不直接 kill Agent。`kg resume` 只用于原 CLI 单任务 workspace，并沿用其 KGS/Gems/Catalog 快照；Batch 根目录不支持 resume，先用 `kg status <batch-workspace>` 找到需续跑的子任务。旧历史 `pipeline_results.json` 和旧更新脚本属于其原实验布局，不能把新 CLI 的 `stages/optimize/work` 误传给只读取 `definitions/*/optimize_definition_output.json` 的旧脚本。

归档每个算子的 RunRequest、终态 `status --detail`、各 Coder ledger、`kernelgen_output.json`、获胜 Coder 的 `optimize_definition_output.json` 与最终复验、best kernel SHA、代码审核、跨芯片 `workflow_result.json` 及逐目标 KGS 原报告。V2 的 `>=0.8` 是实验门槛，不是 KGS `PASSED` 的通用定义。若要提交或回写 FlagGems，仍需在实际目标环境确认仓库原生 correctness pytest 与 benchmark 覆盖了**这份候选**；当前测试若无法准确注入候选，应标记为待补验收，不能以 KGS benchmark 通过代替。完成后确认 Agent/Coder 已退出、各 Server scheduler 空闲健康，只停止本次自己创建的临时服务。

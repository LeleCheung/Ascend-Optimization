# KernelGen 项目上手指南

读完本文，你应能知道项目做什么、完成安装、跑通一个算子，并找到后续问题的文档。默认 Example 使用本地 Agent + 本地 KGS；目标设备在远端时，先看 [远端 KGS 部署](operations/deployment/remote_server.md)。

## 1. 项目做什么

KernelGen 是面向 GPU/NPU 单算子的 AI 优化框架：根据算子定义生成 Kernel，反复验证正确性、测量性能并迭代优化。

- **KernelGen（KG）**：运行模型 Agent 和 Workflow，管理 workspace、优化历史与产物。
- **KernelGen Server（KGS）**：在目标芯片上加载算子，执行 Preflight、编译、正确性验证、Benchmark 和 Profile。
- **Catalog**：提供算子定义、reference 和测试 workload，作为生成与评测的输入契约。

只做算子生成/优化，推荐使用 `kg run`：默认走 `kernelgen`，1 个 Coder、1 个 epoch、最多 10 rounds；也可显式选择 `--mode simple_opt`。输入分为两类：Gems adapter Definition 复用 Gems 原有 pytest，Native Catalog 自带 reference 和 workload。两类都优先使用 KGS 已安装的内容；本地新获得的 Definition 或 Catalog 用 `--catalog-path` 输入。共享 Knowledge campaign 等其他 Workflow 使用对应 Python launcher。完整命令见 [KG CLI 文档](design/kg_cli.md)。

优化默认先审核测试契约，再在目标设备验证 baseline；只有显式 `--skip-review` 才跳过模型审核。该选项不会跳过目标验证、候选正确性、计时或代码审核。详见 [统一测试契约审核](design/workflows/review_tests.md)。

## 2. 安装

前提：Linux、Python 3.10+、Git，以及已配好驱动、Torch、Triton 和厂商运行时的目标环境。使用项目专用 Python 环境；安装过程只准备 KG/KGS 轻量依赖，不安装、升级或替换上述核心运行时。需有两个 Gitee 仓库的读取权限。

直接安装你当前 clone 的 KG checkout，不在安装时切换分支或版本。KG 的包依赖从配套 KGS exact commit 安装纯客户端；只有启动 Server 时才准备完整 KGS checkout，提交与 [锁定清单](../deployment/kgs.lock.yaml) 保持一致。

最新发布组合为 KG v6.7.0 / KGS v6.5.0 / Protocol v6.2。KG 只安装从[锁定清单](../deployment/kgs.lock.yaml)中 KGS exact commit 构建的独立客户端，Server 仍在目标机器准备；两侧的验证范围见 [KG v6.7.0 发布说明](releases/v6.7.0.md)和 [KGS v6.5.0 发布说明](https://gitee.com/BaaiAC/kernelgen_server/blob/v6.5.0/docs/releases/v6.5.0.md)。KGS 的 `compatibility.yaml` 指定官方 Gems `kernelgen-dev` 分支；每次实验仍须记录实际 Gems commit，并让新导出的 Definition 与目标 KGS 所用 checkout 一致。旧 KGS 或 Gems 不具备所需能力时会明确失败。

在 KG 仓库根目录执行：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -e .
kg --help
```

这里的基础 Python 应已有本地 KGS 所需的核心运行时；`--system-site-packages` 复用它们而不重新安装。只运行远端 KG 的 CPU 环境可以省略此选项。后续终端重新执行 `source .venv/bin/activate` 即可使用 `kg`；更新 KG checkout 后重新运行 `python3 -m pip install -e .` 同步客户端依赖，`kg run` 不会在任务中途更新它。上述简化安装适用于包含 [部署修复](validation/deployment_fixes_20260913.md) 的 checkout；旧 KG v6.3.0 提交 `fef3e2ef` 仍需使用 `venv --copies` 并先安装 `setuptools>=69`、wheel，详见发布版验收报告。

安装 KG 时，其 KGS 依赖只有纯客户端包，不会在本机 clone 或运行完整 KGS。首次 `kg server start` 才按锁定清单准备目标机器上的 KGS checkout，安装 client 与 Server 并检查导入路径和受保护运行时版本；已有目录仅在来源、commit 和 clean 状态匹配时复用，不覆盖其他版本或未提交修改。本地 Server 需要另选目录时，在 `kg server start` 使用 `--kgs-root <path>`，该目录也必须匹配锁定提交。

只管理远端 KGS 时，`kg server start --target remote` 不要求本地 KGS checkout，也不启动本地 KGS；Workflow 所需的 `kernelgen_client` 已随 KG 安装，具体见 [远端部署](operations/deployment/remote_server.md)。

还需准备并认证 Claude Code 或 Codex CLI；下例默认使用已可用的 Claude Code，需要 Codex 时给 `kg run` 增加 `--runtime codex`。模型安装、认证变量和自定义 endpoint 见 [Runtime 配置指南](guides/runtime_and_workflow_examples.md#1-模型-runtime-与-catalog-准备)，不要把凭据写进命令或日志。

## 3. 跑通一个 Example

以下以 NVIDIA 的一张已授权空闲卡 `0` 为例。其他芯片先按 [部署指南](operations/deployment/remote_server.md) 选择 backend、设备和 timing。

### 启动并检查 KGS

在上一步的 KG 仓库目录执行；后续命令保持同一工作目录，CLI 状态默认保存在这里的 `.kernelgen/`：

```bash
kg server start example --target local \
  --backend cuda --devices 0 --timing triton --install-gems
kg server status example
```

`--install-gems` 在首次启动前准备 FlagGems。已配置默认分支的实例再次 `kg server start` 时会检查最新 commit；运行中的服务不会原地升级。KGS 自身仍按 lock 中的 exact commit 部署，见 [Gems 来源策略](design/gems_source_policy.md)。Native Example 不需要安装 Gems。

看到 `state: RUNNING`，以及 scheduler 的 `healthy=1`、`active=waiting=broken=0` 即可继续，无需再调用 HTTP 接口。默认端口为 8000，KGS 只监听 loopback，不要向公网开放。

### Gems adapter：先用 KGS 已有的 Definition

```bash
kg run --definition negative
```

这是 Gems adapter 参数最少的路径：KGS 已安装 `negative` 的 Definition，评测使用该实例 FlagGems checkout 中的原 pytest 和 benchmark。无需本地导出 Definition。

#### 新 Gems pytest 如何获得 Definition

确保本地有与目标 KGS 相同 commit 的、已提交且干净的 Gems checkout；算子须同时有 correctness pytest、benchmark 和可确定的公开 ABI。`kg server status example --json` 的 `config.flaggems_commit` 可用于核对目标版本。

```bash
kg definition --flaggems-repo /path/to/FlagGems \
  --pytest-path tests/test_relu6_.py --workspace runs/relu6-definition
kg run --definition relu6_ --catalog-path runs/relu6-definition/catalog
```

`kg definition` 在本地生成 Definition 和来源清单，不运行 pytest。`kg run` 会将其作为 Bundle 上传，并在 KGS 上核对原始 suite、源码 commit、core baseline 和候选评测；不需要用户填写 Bundle ID，也不会把 pytest 源码复制进 Bundle。该算子可以尚未列入 KGS 内置 Definition。更详细的输入与版本约束见 [Gems Definition 指南](design/workflows/gems_adapter_definition.md)。

### Native Catalog：先用 KGS 已有的 Catalog

```bash
kg run --catalog-name kernelgenbench --definition kernelgenbench_square
```

这是 Native Catalog 参数最少的路径：KGS 已安装 `kernelgenbench`，其中包含 `kernelgenbench_square` 的 reference 和 workload。只跑 Native 时，上面的 `kg server start` 可以省略 `--install-gems`。[A100 实测](validation/native_catalog_onboarding_a100_20260925.md)验证了默认审核、单轮生成和最终复验。

#### 新算子如何获得 Native Catalog

已有符合 [v6.2 Native Catalog 契约](design/v6.2.md)的本地目录时，直接提交，无需先复制到 KGS 的 `data/`：

```bash
kg run --definition <operator> --catalog-path /path/to/native-catalog
```

若还没有 Catalog，应先由算子提供方按 [KGS Native Catalog 编写规范](https://gitee.com/BaaiAC/kernelgen_server/blob/main/docs/v6.2_native_catalog_authoring.md)交付并验证；从 Gems 源码或 PR 生成时，可独立运行 `kg extract --flaggems-repo /path/to/FlagGems --operator <operator>`，取其输出的 `catalog_path` 再提交给 `kg run`。抽取需要预先配置的采集环境，详见 [抽取与采集指南](design/workflows/flaggems_pr_extract.md)，不要让每台目标机器各自重抽一份。

### 查看进度与结果

目标设备由 KGS `/status.target.device` 返回，无需填写硬件名称；CLI、Batch 和 Python launcher 均保留未指定状态，直到 Workflow 读取 KGS 后确定实际设备。显式传入 `--target-hardware` 时作为约束进行比对，不匹配即报错；不要凭 Agent 本机环境推断目标能力。

上述 `kg run` 命令都默认连接本机 8000 端口，创建独立 workspace 并在后台运行。复制输出的 workspace 路径：`kg status <workspace>` 查看摘要、`kg status <workspace> --detail` 查看结构化进度、`kg history <workspace>` 查看已计时轮次、`kg logs <workspace> --follow` 查看事件。后台提交成功只表示已接受任务。远端 KGS 使用本地 SSH stdio proxy 的 loopback URL，通过 `--eval-server` 指定；若设置过 `FIB_EVAL_SERVER` 或 `KERNELGEN_RUNTIME`，CLI 会沿用。

若环境配置了 HTTP 代理，先按 [实验手册的代理配置](operations/multiple_device_experiment_runbook.md) 排除 loopback 请求，不修改系统全局代理。

任务状态 `SUCCEEDED` 且 `kg history <workspace>` 中有正确性通过、计时有效的 measured round，表示流程跑通；是否更快另看 `best_geo_mean > 1`。性能事实以 ledger 为准。产物位于 `stages/optimize/work/`：SimpleOpt 直接保存 ledger 和 best kernel，KernelGen 按 epoch/Coder 分目录保存。失败时保留 workspace，先查看 `kg logs <workspace> --raw`；中断续跑使用 `kg resume <workspace>`。

确认任务结束后，关闭本例创建的 Server：

```bash
kg server stop example
```

`simple_opt/kernelgen` 都支持上述两类输入；自定义算子目录快捷参数 `--operator-dir` 尚未实现。本地 Definition/Catalog 由 OperatorOptimize 上传并建立 Bundle 执行绑定。是否可执行仍检查目标 KGS 的 `evaluation_binding` 和 evaluator capability，不能只看上传成功；Gems Definition 导出不等于生成新 pytest。

## 4. 下一步看什么

| 你要解决的问题 | 对应文档 |
|---|---|
| CLI 参数、极致优化、YAML Batch、状态、取消、续跑、Server 管理 | [KG CLI 文档](design/kg_cli.md) |
| 算子从 pytest 准备到优化、审核及 PR 的整体流程，当前 dummy 能力与待接入项 | [Lifecycle 整体流程](design/workflows/operator_lifecycle.md) |
| 模型配置、参考代码、Knowledge 查询、其他 Python Workflow | [Runtime 与进阶 Workflow 示例](guides/runtime_and_workflow_examples.md) |
| 其他芯片部署、设备探针、FlagGems 精确 revision、Server 自测 | [部署指南](operations/deployment/remote_server.md) |
| 远端 SSH stdio proxy、多设备实验、连接配置与故障排查 | [多设备实验与 E2E 验收手册](operations/multiple_device_experiment_runbook.md) |
| Kernel Todo V2 的 NV 优化、跨芯片复测和目标芯片特化 | [V2 单一实验手册](operations/experiments/kernel_todo_v2_experiment_runbook.md) |
| 同一份 Gems pytest 和生成候选在多种芯片上复测，并先排除目标 API/dtype 不支持 | [MultipleDeviceTest Workflow](design/workflows/multiple_device_test.md) |
| 独立抽取共享 Catalog，再交给各目标优化 | [Catalog 流程](design/workflows/catalog_optimize.md)、[PR 抽取与采集配置](design/workflows/flaggems_pr_extract.md) |
| Definition、workload 与 Gems adapter 输入契约 | [FlagGems 指南](guides/flaggems_definition_workload_guide.md) |
| 已有 pytest 尚无 Definition，用 `kg definition --flaggems-repo <repo> --pytest-path <path>` 导出后优化 | [Gems Definition CLI 与 Workflow](design/workflows/gems_adapter_definition.md) |
| 审核 Gems 原生 pytest 入口、baseline、skip 和输入污染 | [pytest review 设计](design/workflows/pytest_review.md)、[pytest review skill](../.kernelgen/skills/pytest-review/SKILL.md)、[检查脚本用法](../.kernelgen/skills/pytest-review/references/checks.md) |
| 多算子共享知识的完整 campaign | [Knowledge 快速使用](guides/KB_QUICKSTART.md) |
| 如何验收、归档 best kernel，已有硬件验证结果 | [结果整理手册](operations/experiments/experiment_result_maintenance.md)、[E2E 验证报告](validation/multiple_device_e2e_validation_report.md) |
| 修改 Agent/Workflow、开发约束和测试 | [AGENTS.md](../AGENTS.md)、[整体设计](../DESIGN.md) |
| 理解协议、运行控制和性能分析实现 | [Native Schema](design/v6.2.md)、[运行控制](design/runtime/run_control.md)、[Profiling 设计](design/profiling_execution_design.md) |

文档按用途存放：`guides/` 是使用指南，`operations/deployment/` 是部署，`operations/experiments/` 是专用实验流程，`operations/troubleshooting/` 是事故分析，`validation/` 是带版本的实测证据，`design/workflows/`、`design/runtime/`、`design/knowledge/` 分别维护领域设计；`releases/` 和 `archive/` 不作为当前部署配置。文档维护规则见 [维护约定](documentation.md)，不另建一份重复的总索引。

Agent 接手代码或实验任务时，先读 `AGENTS.md`，再按本表读取任务相关文档；本文不是部署安全约束或实验规范的替代品。

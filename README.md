# 昇腾算子优化复现仓库

本仓库把昇腾算子优化实验所需的项目文件、第三方源码、固定版本信息和复现记录集中保存。目标是在一台新的 910B 环境中，依据本文档重建 KernelGen、KernelGen Server（KGS）和 FlagGems 的运行环境，并复现实验。

## 目录

- `project/Ascend-Optimization-v1/`：本实验的脚本、候选算子、报告和说明文档。
- `third_party/FlagGems/`：实验固定版本的 FlagGems 源码。
- `sources/kernelgen/`：KernelGen 6.7.0 解压源码。
- `sources/kernelgen_server/`：KernelGen Server 6.5.0 源码。
- `runtime/catalogs/`：可复现所需的最小 Catalog；虚拟环境、Claude 二进制和原始运行日志留在 910B 本机，不提交到 Git。
- `manifests/`：组件提交号、镜像和软件版本。

## 目标

本项目针对华为昇腾 910B 上 FlagGems 的 Triton 算子开展性能优化。只选择正确性已经达标、但加速比不理想的代表性算子；精度失败的算子不进入优化流程。每个结论都必须绑定设备、软件版本、形状、数据类型、计时口径和完整 workload。

### 当前交付目标：narrow_copy 诊断到优化闭环

首个正式案例是 `narrow_copy`。最终交付不是一次 profiler 截图或单轮加速比，而是以下可复核的材料：

1. 依据 910B/CANN 官方资料、仓库 `docs/` 和实际 KG/KGS 源码，说明真机 `metrics`、模拟器 `instruction`、host 计时、编译 IR 各自能证明什么；记录 KG 原生分析的缺口。参考 [独立 Profiling Workflow 设计](docs/ascend-profiling-workflow.md)及固定版本的 `cuda-optimized-skill`，只迁移证据门禁、单项假设和消融方法，不照搬 CUDA 指标或峰值。
2. 交付可独立运行的 Profiling Agent：输入固定算子、候选源码、workload、KGS 评测、原始 `msprof` 产物及可用 IR；校验 case、源码和产物身份；输出逐 case 中文诊断、证据来源和置信度、受限类型、一个可证伪的修改建议。缺少同范围数据时明确标记 Roofline 或源码映射不可用。
3. 在隔离的 910B 工作区，以相同的 18 个 correctness、15 个 timing workload 和边界用例验证至少一项由诊断驱动的候选；保留失败尝试及重复测量波动，逐 case 对照 PyTorch、FlagGems、KG 无/有 profiler 和前一版候选。正确性失败的计时不得参与最优候选选择。
4. 提交候选源码、运行命令、环境版本、原始证据、中文报告和局限性，并同步 Windows、910B 新工作树与 GitHub。若仍低于 PyTorch，只给出限定设备、软件、workload 和实验预算的结论，不声称 Triton 的理论上限。

当前独立 workflow 入口是 `project/Ascend-Optimization-v1/scripts/ascend_profiling_workflow.py`。它会校验候选源码 SHA、Definition benchmark fingerprint、case UUID、profile 级别和每个原始 artifact 的大小/SHA，然后输出 Markdown 与 JSON 体检报告；`--profile RESPONSE REQUEST ARTIFACT_DIR` 可重复传入多个 case 的 `metrics` 或 `instruction` 采集。真实归档的运行示例和结果见 [narrow_copy 独立体检](project/Ascend-Optimization-v1/reports/ascend910b/narrow-copy-combined-20261006/独立体检.md)，拒绝篡改源码和 artifact 的测试见 `scripts/test_ascend_profiling_workflow.py`。

体检提出的第一项可证伪修改是把 persistent copy 的 `_PERSIST_ITERS` 从 `13` 改成 `8`。该候选正确性 `33/33`，但 geo mean 为 `0.5112×`，低于原候选首轮 `0.5349×`，已按门禁回退；失败尝试和证据见 [persist8 消融记录](project/Ascend-Optimization-v1/reports/ascend910b/narrow-copy-combined-20261006/诊断消融-persist8.md)。

编译链与运行时分层结论见 [narrow_copy 编译链报告](docs/narrow-copy-compiler-chain.md)。当前证据显示：首次编译约为百毫秒级，但 warm 调用仍受输出分配和 launcher 路径影响；FlagTree 的内部 compiled runner 不是稳定 ABI；公开的 `simt_only` 旁路在当前 CANN BiShengIR 版本上因编译参数不兼容而失败。因此在该问题完成前不切换到下一个算子。

## 分步计划

### 第 0 步：固定环境

1. 在 910B 上使用清单中的镜像、CANN、Python、torch-npu、KG 和 KGS 版本。
2. 检查设备占用，使用独立 Catalog 和运行工作区，不改动共享容器配置和他人任务。
3. 记录 KG、KGS、FlagGems 提交号、镜像摘要、设备号和全部运行命令。

### 第 1 步：最小算子复现

1. 先使用 `runtime/catalogs/catalog-square` 跑通一个手工构造的 Native Catalog 算子，验证编译、正确性、设备计时、最终复验、KGS HTTP 服务和 Claude MCP 调用链路。
2. 这一步只是 KG+KGS 的烟雾测试。`square` 不是导师所说的 Gems Definition 目录算子，也不能作为后续性能优化目标。
3. 保存运行请求、事件日志、逐用例结果和失败原因，确认新工作区可以独立重跑。

### 第 1.5 步：Gems Definition 流程

1. 先阅读 `sources/kernelgen/docs/ONBOARDING.md` 和 `sources/kernelgen/docs/design/workflows/operator_optimization.md`。
2. 在 `sources/kernelgen_server/data/flaggems-adapter-definitions/` 中选择一个已有算子名，使用 `GemsAdapterDefinitionWorkflow`（`kg definition`）导出 Definition。
3. 用 `OperatorOptimizeWorkflow`（`kg run`）完成测试审核、KernelGen 优化和代码审核。Native Catalog 已经准备好的算子可以直接进入 `kg run`，不必重复导出 Definition。
4. 给 Claude Code 的最小启动提示可以写成：`请先阅读 docs/ONBOARDING.md 和 docs/design/workflows/operator_optimization.md，然后从 kernelgen_server/data/flaggems-adapter-definitions 中选择一个算子，完成 kg definition 和 kg run 的最小闭环。` 实际运行时还要补充目标设备、Catalog 路径、模型、预算和工作区。

### 第 2 步：筛选代表性算子

1. 读取 `project/Ascend-Optimization-v1/验收看板_结果表.xlsx` 的华为 Ascend910B 列，并在 FlagGems 华为后端找到对应实现。
2. 过滤掉精度失败、没有有效计时、数据不完整的记录。
3. 从正确性达标但加速比不理想的算子中选择一个代表性算子。优先选择 `flaggems-adapter-definitions` 中已有 Definition 的算子；若只有 Native Catalog，则记录来源和转换方式。固定其 FlagGems 入口、华为实现、测试、benchmark、shape、dtype 和完整 workload。

### 第 3 步：建立基线

1. 在同一设备和同一 workload 上分别记录 PyTorch 原生实现与现有 FlagGems 实现。
2. 统一 warmup、重复次数、同步范围和统计方法，记录逐用例延迟、几何平均和波动。
3. 确认候选代码实际执行 Triton 内核，不允许以 fallback 代替优化结果。

### 第 4 步：KernelGen 对照实验

1. 对同一个算子和同一套 workload 运行完整 KernelGen，先关闭优化过程的 profiling 反馈。
2. 在独立工作区使用相同模型、Coder、epoch、round 和预算，再运行开启 profiling 反馈的版本。
3. 比较两组的正确性、搜索轨迹、最终候选、逐用例性能和总耗时。这里的 `--no-profile` 只影响优化反馈，KGS 的设备计时仍可能使用 NPU profiler。

### 第 5 步：独立 Profiling Agent

1. 先实现一个独立 workflow：输入算子、编译产物、workload 和 profiling 数据，输出结构化体检报告。
2. 优先使用 Msopprof 的 `metrics` 和 `instruction` 两类数据；明确区分实测采集与模拟器分析。
3. 围绕 Roofline、访存带宽、计算吞吐、流水线空泡、Kernel 启动开销、缓存和资源冲突判断受限类型。
4. 将指令映射回 LLVM IR，再关联到 Triton 源码；无法精确映射时保留证据和不确定性。
5. 参考 `KernelFlow-ops/cuda-optimized-skill` 的分析流程，但适配 CANN、Ascend 910B 和 Msopprof 数据格式。

导师要求这个分析器先作为独立 workflow：输入算子、编译产物、workload 和 profiling 数据，输出结构化体检报告；稳定后再与 KernelGen 融合。profiling 默认使用 `msprof`，`metrics` 用于硬件指标，`instruction` 才会使用模拟器。两种模式由 agent 根据分析问题选择。

### 第 6 步：反馈优化与结论

1. 把 profiling agent 的瓶颈报告转为明确的候选修改，例如 tile、访存布局、融合、向量化或减少启动次数。
2. 在相同 workload 上重新验证正确性和性能，保留每次修改的证据。
3. 输出优化后的 FlagGems/Triton 实现，或给出边界清楚的结论：在指定设备、软件、workload 和预算下，Triton 候选没有超过 PyTorch。
4. 只有在独立 workflow 稳定后，再考虑把 profiling agent 接入 KernelGen 主流程。

## 仓库角色与保留策略

`Ascend-Optimization` 是当前主仓库。`project/Ascend-Optimization-v1` 是其中的实验项目目录，不是新的 Git 仓库。原来的 Windows `workplace` 和 GitHub `KernelGen` 仓库暂时保留，作为历史基线和故障回溯依据；等第 1 步在 910B 上能从新仓库独立复现并完成一次验收后，再考虑归档它们。

910B 上与本仓库当前提交同步的工作树是 `/data/hanle/ascend-optimization/Ascend-Optimization-current`。旧 `/data/hanle/ascend-optimization/Ascend-Optimization` 停在历史提交，含未跟踪实验文件，暂时保留且未覆盖。910B 的 GitHub SSH 暂不可用；本次通过 Windows 生成 Git bundle 导入新提交。容器内的 Git LFS 3.0.2 可执行文件位于 `/data/hanle/ascend-optimization/runtime/kg-controller/git-lfs-package/usr/bin/git-lfs`，使用时把该目录加入 `PATH`；新工作树的 LFS 文件已从本地对象缓存恢复。

## 复现原则

本仓库不提交 Python 虚拟环境、容器层或 Claude 原生二进制。它们依赖机器架构和驱动，无法作为跨机器的可靠复现材料。请按照 `manifests/components.lock.yaml` 创建环境，并使用 `project/Ascend-Optimization-v1/experiments/` 下的脚本运行实验。

仓库中的压缩包和测试二进制由 Git LFS 管理。Windows 等开发机克隆后需要安装 Git LFS 并执行 `git lfs pull`。910B 上的展开源码已经可以直接使用；如果需要读取这些大文件，再安装 Git LFS 或使用 `project/KernelGen` 中的原始文件。

FlagGems、KernelGen 和 KGS 的源码目录均为导出副本，不包含上游仓库的 `.git` 元数据；版本和来源记录在清单中。这样克隆本仓库后可以直接查看和构建全部源码，同时不会产生嵌套 Git 仓库。

## narrow_copy 最新实验

2026-10-05 已在 910B 上完成 `narrow_copy` 的隔离 FlagGems case API 适配、专用 KGS `19652`、Claude Code 驱动的 KernelGen 正式正确性和 walltime 计时闭环。33/33 个正确性 workload 与 15/15 个计时 workload 通过，整体生命周期成功；本轮相对 PyTorch 的加速比几何平均值为 `0.3532×`，因此仍需继续优化。运行产物、补丁和踩坑说明见 [narrow_copy 实验记录](project/Ascend-Optimization-v1/reports/ascend910b/narrow-copy-20261005/README.md)。

随后在开启 profiling 的一次两轮运行中，候选按完整 launch signature 缓存已编译 launcher，geo mean 从该运行 round 1 的 `0.3396×` 提升到 round 2 的 `0.4708×`，最终独立复验为 `0.4615×`；完整 correctness 与 15/15 timing 通过，但整体仍低于 PyTorch native。KGS 对 5/15 个 timing case 的 `msprof metrics` 分析是在 round 2 候选评测**之后**完成的，不能将上述提升归因于这次正式分析；原先单轮 `--no-profile` 运行也不构成同预算对照。新的两轮配对 A/B 见 [narrow_copy 实验记录](project/Ascend-Optimization-v1/reports/ascend910b/narrow-copy-20261005/README.md)。Excel 第 29 行 `G29=0.0138` 缺少 baseline、workload 和统计口径，不能与本次加速比直接比较。

新的配对 A/B 已完成：有 profiler 组在第一轮正式分析 `recorded=true` 后才修改第二轮候选，但第二轮有 9/18 个 correctness case 运行错误，KG 回退到第一轮。两组最终独立复验分别为无 profiler `0.3580×`、有 profiler `0.3498×`，均通过正确性且均未超过 PyTorch。逐 case 结果、失败边界和下一步见 [中文配对报告](project/Ascend-Optimization-v1/reports/ascend910b/narrow-copy-20261005/ab-report.md)。

2026-10-06 已修复上述第二轮候选少传 `EVEN` 参数的问题，并把 launch-plan 与已编译 launcher 缓存合并成新候选。新候选在专用 KGS 上三次独立评测均通过 18/18 correctness、15/15 timing，geo mean 分别为 `0.5349×`、`0.5120×`、`0.5152×`，原始结果见 [合并候选评测](project/Ascend-Optimization-v1/reports/ascend910b/narrow-copy-combined-20261006/)。它仍低于 PyTorch；三次运行使用不同 NPU，尚不能将与历史候选的差异视为严格同卡消融，也不能据此宣称独立 Profiling Agent 已完成。

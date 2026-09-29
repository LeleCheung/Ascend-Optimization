# Ascend 910B 算子优化实验

本仓库使用 **KernelGen（KG）+ KernelGen Server（KGS）**，在华为昇腾 910B 上优化 FlagGems 的 Triton 算子。目标是对已经正确性达标、但加速比不理想的代表性算子，比较自动优化、profiling 辅助优化和华为细粒度 profiling 辅助优化的效果，交付可复现的优化版本或有证据支持的性能结论。

## 范围与目标

- 输入：`验收看板_结果表.xlsx` 的 `华为-Ascend910B` 列，以及 [FlagGems](https://github.com/flagos-ai/FlagGems) 华为后端的实现、正确性测试和 benchmark。
- **不处理表中精度失败的算子**。先确认正确性通过和有效计时，再选择加速比不理想的算子；零值、空值或只有“精度通过”的记录不能直接作为有效性能数据。
- 先走通一个简单算子，再选择一个代表性算子完成全流程，不立即铺开批量实验。
- 最终产出是优化后的 Triton 实现及验证证据，或“在指定设备、软件版本、shape/dtype、工作负载与搜索预算下，尚未超过 PyTorch”的结论。有限实验不能证明所有 Triton 实现都打不过 PyTorch，也不能将一次运行的最好结果称为理论上限。

## 实验路线

1. **最小闭环**：在 910B 上准备兼容的 KG、KGS 和 FlagGems，选择简单算子（如适用的逐元素算子），完成生成、编译、正确性、设备计时、迭代和最终复验。具体算子以 Catalog 和目标环境支持情况为准。
2. **代表性算子筛选与复现**：定位 Excel 行、FlagGems 算子入口、华为实现、pytest 和 benchmark；重新确认正确性与低加速比。选择有代表性的计算或访存模式，保留完整 workload，不能只挑有利 shape。
3. **完整 KernelGen，无 profiling 分析反馈**：固定模型、初始实现、epoch/round/Coder 数与预算，运行完整 `kernelgen` workflow，记录最好实现与逐 case 性能。
4. **完整 KernelGen，有 profiling 分析反馈**：使用相同基线、workload、模型和预算，在独立 workspace 运行，比较 profiling 对优化过程与最终性能的贡献。
5. **华为细粒度分析**：验证并接入 Msopprof，分析流水线、搬运、缓存和资源冲突；在工具支持时使用 MindStudio Insight。将分析证据反馈给优化流程，验证新的候选实现。
6. **交付**：正确性复验、统一条件性能复测，归档候选源码、FlagGems 补丁、逐 case 数据、汇总、profiling 证据和结论。

“不开 profiler / 开 profiler”指是否向优化过程提供额外 profiling 分析反馈，不表示关闭正确性或设备计时。KGS 文档要求昇腾使用其支持的严格设备计时链路，可能由 `torch_npu.profiler` 实现；两组实验必须保持同一计时口径。具体开关以当前 KG/KGS 版本文档及实现为准，不猜测 CLI 参数。

## 目标环境与版本

- SSH：`ssh baai-910b`，目标地址 `10.0.0.8`，通过堡垒机转接。
- 主机：鲲鹏 920 / ARM64，系统可见 192 个逻辑 CPU。
- 加速卡：8 个 Ascend 910B4-1，每个 64 GiB HBM。
- 已观察到 CANN 和运行中的 `kernelgen-ascend-flagtree060` 容器；这不代表包版本兼容、设备空闲或服务可复用，实验前还需检查。
- KG 负责模型 Agent、工作流和历史；KGS 在 910B 上执行编译、正确性验证、benchmark 与 profile。第一阶段优先在 910B 的项目专用环境打通，之后可把 KG 控制端与设备端分开。
- 保留现有 Torch、torch_npu、Triton、CANN 和驱动；依据 KG 的 `deployment/kgs.lock.yaml` 与 KGS 的兼容性文档核对版本。压缩包文件名不能替代 exact commit 或源码摘要。
- 每次记录 KG/KGS/FlagGems 版本（无法取得提交号时记录源码摘要）、容器/镜像、解释器、核心依赖、设备号、模型/运行时、预算、命令和时间。
- 使用已确认空闲的设备和独立工作区，不覆盖现有实验或重启他人服务。

## 910B 上的镜像、容器与仓库

- **镜像**是容器的基础环境。当前 `tle_yy` 使用 `harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64:202606-torch2.9.0-base`；本实验没有构建或修改该共享镜像。
- **容器**是由镜像启动的运行环境，KG、KGS 和编译依赖在其中运行。`tle_yy` 将宿主机的 `/data` 以可写方式绑定到容器内的 `/data`，所以这一路径下的文件在宿主机和容器中是同一批文件，不是两份副本。
- **Git 仓库**位于 `/data/hanle/ascend-optimization/KernelGen`，用于保存文档、脚本、候选源码和精选结果；它可与 Windows 上的 `workplace` 通过 GitHub 同步。`/data/hanle/ascend-optimization/FlagGems` 是独立仓库。
- **运行时目录**位于 `/data/hanle/ascend-optimization/runtime/kg-controller`，存放虚拟环境、专用服务、Catalog 和原始运行工作区。它也经 `/data` 挂载可见，但不属于 `KernelGen` Git 仓库，不会随 `git push` 自动上传。

因此，“三处仓库提交一致”只表示 Git 跟踪的文件一致，不表示 910B 上的镜像、容器、虚拟环境和原始实验输出也存在于 Windows 或 GitHub。

## 仓库组织

本仓库是实验总仓库。当前已提交的材料为：

```text
README.md                         # 目标、范围、实验路线
验收看板_结果表.xlsx               # 原始筛选依据
kernelgen-dev.zip                 # KG 源码快照
kernelgen_server-v6.5.0.tar        # KGS 源码快照
doc/                              # 老师提供的要求与工具资料
```

后续按需建立如下目录（以下是约定，并非已经完成的部署）：

```text
sources/kernelgen/                # 解压后的 KG；修改可追踪
sources/kernelgen_server/         # 解压后的 KGS；修改可追踪
experiments/ascend910b/<operator>/ # 版本清单、配置、运行脚本、复现步骤
kernels/ascend910b/<operator>/    # 最终候选实现及入口
patches/flaggems/                 # 对固定 FlagGems commit 的补丁
reports/ascend910b/<run-id>/       # 逐 case 数据、汇总、分析与结论
runs/                            # 本地/服务器运行工作区，大型原始数据另行归档
```

FlagGems 建议独立放在服务器的同级目录，例如：

```text
/data/hanle/ascend-optimization/
├── KernelGen/                    # 本实验仓库
└── FlagGems/                     # 独立 checkout，固定实验 commit
```

这只是建议路径，使用前检查存储空间和已有内容。不要向 KG 框架包内部随意塞入整个 FlagGems 仓库。实际华为后端目录、测试入口与兼容分支应在固定 revision 上确认；最终算子修改在 FlagGems checkout 中验证，再以补丁或独立提交回收至本仓库。不要把开发分支、主分支或机器上现有 checkout 默认视作相同基线。

## 性能判定与证据

- 明确 baseline 是 PyTorch 原生 NPU 路径还是现有 FlagGems 实现。两者分别记录，不能混称。
- 使用统一定义：`speedup = baseline_duration / candidate_duration`；大于 1 表示候选更快。Excel 的历史口径仍需核实后才能直接比较。
- 正确性通过是前置条件。候选必须实际执行目标 Triton 实现，不能用 PyTorch fallback 伪装优化成功。
- 固定 shape、dtype、stride、容差、warmup、重复次数、同步与计时范围；记录设备占用和波动，避免 profiler 的采集开销污染最终性能比较。
- 保存逐 case 延迟与加速比、聚合方法、回退 case 和预算。完整 workload 的表现与局部 shape 的收益分别说明。
- 保留 KG ledger、最终 JSON、best kernel、日志、KGS 状态、命令和 profiler 原始数据位置。必要时重复独立运行，区分搜索随机性与测量噪声。

## Msopprof、MindStudio Insight 与 Roofline

要求来源见 [细粒度 Profiling 算子优化](doc/Fine-grained%20Profiling算子优化.md)、[Roofline 分析](doc/Roofline%20分析.md)，以及 `doc/Msopprof_MindStudio Insight 调优.zip` 内的说明。

- Msopprof 用于采集，MindStudio Insight 用于分析可视化数据。优先检查 CANN 已集成的工具及当前版本能力。
- 老师的材料注明 MindStudio Insight 当时仅支持仿真采集和 950 上板采集。因此 **910B 上板数据能否直接可视化必须实测确认**；必要时先分析 Msopprof CSV，再验证支持的仿真路径，并明确区分仿真与实测。
- Roofline 按老师提供的 `manifest.json + points.jsonl` 数据契约组织，每个 case × dtype 一条记录，保存 FLOPs、Bytes、时间及来源。
- FLOPs、Bytes、时间必须覆盖同一 kernel/range；区分硬件计数与算法估算，标记内存层级及算力/带宽 roof 的来源，不套用其他设备的峰值。

参考：[FlagGems](https://github.com/flagos-ai/FlagGems)、[老师的 Msopprof/MindStudio Insight 文档](https://jwolpxeehx.feishu.cn/wiki/Op60w1fwyidnQ7kvuLmckyOYnPe)。

## 当前状态

已完成 910B 上的隔离 KG 6.7.0 / KGS 6.5.0 部署和 `square` 算子的生成、编译、正确性与设备计时。首次运行的最终复验因计时漂移停在 `NEEDS_RETEST`，不能作为稳定加速结论；三次计时的第二轮在优化阶段取消，未完成正式候选评测。复现路径和踩坑记录见 [square 交接文档](experiments/ascend910b/square/PITFALLS.md)，首次结果见 [运行报告](reports/ascend910b/square-private-1/README.md)。代表性 FlagGems 算子和 profiler 对照实验尚未开始。

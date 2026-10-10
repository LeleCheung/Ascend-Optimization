# 昇腾算子优化复现仓库

本仓库用于优化华为昇腾 910B 上的 FlagGems 算子，并保存 KG/KGS 源码、实验候选、评测证据和复现方法。

**当前目标：从验收看板华为列选择加速比低于 0.8×、正确性通过的代表性算子，使用 FlagGems `master` 的固定提交建立基线，完成三版本优化对照，争取超过 0.8×。**

2026-10-09 导师确认：Excel 数字是加速比，绿色验收门槛为大于 0.8×，FlagGems 使用 `master`。筛选依据是华为列实际数字；黄色底色不作为立项依据。旧实验锁定的版本不是新的 master 基线。

## 当前状态

| 对象 | Excel 华为列 | 状态 |
|---|---:|---|
| [narrow_copy](project/Ascend-Optimization-v1/operators/narrow_copy/README.md) | G29：0.0138× | 历史 D2D 候选约 0.56～0.59×，仍未达标，保留在主线 |
| [matmul_bias_activation](project/Ascend-Optimization-v1/operators/matmul_bias_activation/README.md) | G611：0.5408× | 固定 master 复测 0.443×；流水候选 v4 为 0.759×，相对 master 1.71×，完整测试 42/42；KG 原生 profiler 第二轮 0.697×，待独立复验，两组尚未闭环 |
| [amin](project/Ascend-Optimization-v1/operators/amin/README.md) | G92：0.5536× | 原版大输入超过 coreDim 启动上限；启动兼容修复基线 0.296×，直接归约候选 v3 为 0.626×，两者均完整通过 27/27；继续调优 |

此前按颜色误选的 MSE backward、PReLU backward、t_copy、SmoothL1 backward、BatchNorm backward、native_layer_norm，Excel 数值均已超过 0.8×，已移入 [历史归档](project/Ascend-Optimization-v1/archive/README.md)。原结果保留为流程和优化方法参考，不计入新目标的成果。

现有 narrow_copy 分析使用过 msprof 指标、模拟器指令及编译产物。matmul 的 Msopprof 官方 Roofline 文本给出 `memory caused`，已核验 kernel、case 和物理卡身份；计数缩放与硬件 roof 尚未核实，数值 Roofline 仍待建立。新目标的完整四版本闭环尚未完成。

## 目录

```text
Ascend-Optimization/
├── project/Ascend-Optimization-v1/
│   ├── operators/          当前主线，按算子集中保存候选、专用脚本和报告
│   ├── tools/              多算子共用的环境、评测、profiling 和验证工具
│   ├── archive/            误选算子、联合对照、烟雾测试和旧阶段记录
│   └── 验收看板_结果表.xlsx
├── sources/                KG 6.7.0、KGS 6.5.0 源码及原始压缩包
├── third_party/FlagGems/    历史固定版本的源码副本
├── manifests/              组件版本与环境清单
└── docs/                   架构、profiling 资料和方法设计
```

[项目索引](project/Ascend-Optimization-v1/README.md)说明文件归属与历史状态。候选和专用脚本跟随算子保存，公共工具由多个算子复用；评测批次中的源码快照用于证明当次实际测了什么，继续保留。

## 分步计划

### 第 0 步：准备独立环境

1. 按 [组件清单](manifests/components.lock.yaml)准备 CANN、Python、torch_npu、Triton、KG 和 KGS，记录实际版本、设备号、容器和运行命令。
2. 检查设备占用，使用独立 KGS、Catalog、tmux 会话和运行工作区。
3. Claude Code 作为 KG 的模型调用运行时，已有实验通过 CC 接入 DeepSeek；配置和令牌保留在机器本地。环境工具见 [tools](project/Ascend-Optimization-v1/tools/README.md)。

square 的 KG+KGS 烟雾测试已完成，记录见 [烟雾测试归档](project/Ascend-Optimization-v1/archive/smoke-tests/square/README.md)。正式算子从 `sources/kernelgen_server/data/flaggems-adapter-definitions/` 选择；仓库内旧 square/softmax Catalog 已移除。

### 第 1 步：筛选并固定正式对象

1. 读取 [验收看板](project/Ascend-Optimization-v1/验收看板_结果表.xlsx)华为 Ascend910B 列，筛选有效加速比低于 0.8× 的算子；精度失败、无有效计时的记录不进入优化。
2. 拉取 FlagGems `master`，固定具体提交号。仓库内 `third_party/FlagGems/` 是历史源码副本，不能直接当作最新 master。
3. 在 `operators/<算子>/` 建立一个入口 README，记录 Excel 行号与数值、上游提交、实际调用实现、shape、dtype 和完整 workload。
4. 复测 master 的正确性和性能。若在相同评测范围已超过 0.8×，记录原因后选下一项；正确性失败的对象不进入本轮性能优化。

### 第 2 步：走通 Definition 与基线

先读 `sources/kernelgen/docs/ONBOARDING.md` 和 `sources/kernelgen/docs/design/workflows/operator_optimization.md`。

1. 用 `GemsAdapterDefinitionWorkflow`（`kg definition`）导出 Definition；已有合适 Native Catalog 时可直接优化。
2. 用 `OperatorOptimizeWorkflow`（`kg run`）执行测试审核、优化和代码审核。KG 默认配置为 1 个 Coder、1 个 epoch、最多 10 rounds；实验需明确记录实际预算。
3. 在同一设备、同一 workload 下评测 PyTorch + torch_npu 与原版 FlagGems，保存逐 case 延迟、正确性和统计口径。

给 CC 的启动提示词可用下列模板，替换算子名、提交与环境参数：

> 请先阅读 docs/ONBOARDING.md 和 docs/design/workflows/operator_optimization.md。从 kernelgen_server/data/flaggems-adapter-definitions 中使用指定算子的 Definition，完成 kg definition 和 kg run。优化对象来自验收看板华为列低于 0.8× 的算子，FlagGems 使用指定 master 提交。先复测完整正确性与基线，再优化性能；记录设备、Catalog、KGS 地址、模型、预算、候选源码和逐 case 结果，使用新的独立工作区。

### 第 3 步：完成 KG 两组对照

使用相同起点、模型、workload、轮数及预算，分别运行 KG 无 profiling 反馈、KG 有原生 profiling 反馈，保存各轮候选及最终独立复验。

`--no-profile` 关闭优化反馈，设备计时仍可能使用 NPU profiler。必须写清设备侧 kernel 计时还是完整调用计时，不能把历史 `timing=walltime` 请求直接解释为端到端耗时。

### 第 4 步：独立 Profiling Agent 与第三版本

1. 调研当前 KG/KGS 的 Ascend 采集与反馈代码。导师说明默认使用 `msprof`，`metrics` 用于真机指标，`instruction` 使用模拟器，由 agent 根据问题选择。
2. 先做独立 workflow：输入算子语义、候选、workload 和原始 profiling 产物，输出中文体检报告；公共入口见 [tools/profiling](project/Ascend-Optimization-v1/tools/README.md)。
3. 按 [Roofline 资料](docs/Roofline%20分析.md)、[细粒度 profiling 资料](docs/Fine-grained%20Profiling算子优化.md)和 [workflow 设计](docs/ascend-profiling-workflow.md)，分析计算、访存、流水线和启动开销；只有数据范围与设备峰值匹配时才计算 Roofline。
4. 参考 [cuda-optimized-skill](https://github.com/KernelFlow-ops/cuda-optimized-skill) 的语义分析、瓶颈判断和建议组织方式，适配 910B 架构与指标；指令必要时先映射回 LLVM IR，再关联 Triton 源码。
5. 把诊断转为具体修改，生成我们的优化版本。先确保至少一个正式目标完成“分析→修改→正确性→性能”的闭环，再考虑接入 KG 主流程。

### 第 5 步：交付

每个正式算子交付一个入口报告：PyTorch/FlagGems 基线、KG 无 profiler、KG 原生 profiler、我们分析指导的版本，附关键优化、完整正确性、逐 case 性能、版本、源码和复现命令。

目标是完整正确性通过并超过 0.8×；超过 1× 表示快于 PyTorch。未达标则直接写结果和已定位的问题，保留有效尝试。`narrow_copy` 的 CANN D2D 混合实现须与 Triton 版本分开标注。

## 复现与同步

虚拟环境、容器层和 Claude 二进制在服务器运行环境中；源码和评测证据在本仓库。910B 仓库外的 `/data/hanle/ascend-optimization/runtime/kg-controller/` 不受本次本地目录整理影响。

压缩包等大文件由 Git LFS 管理；克隆后安装 Git LFS 并执行 `git lfs pull`。KG/KGS 原包位于 `sources/archives/`。导出源码不含嵌套 `.git`，版本见清单。

目录整理已提交至 `a349822b`。2026-10-10 在 910B 新建独立实验副本 `/data/hanle/ascend-optimization/goal-20261010/Ascend-Optimization`，使用该提交并固定新的 FlagGems master；本轮新增工具、候选与评测结果待验证后提交同步。历史报告中的绝对路径、端口与版本是当时快照。

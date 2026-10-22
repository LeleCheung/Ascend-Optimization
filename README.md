# 昇腾算子优化复现仓库

本仓库用于优化华为昇腾 910B 上的 FlagGems 算子，并保存 KG/KGS 源码、实验候选、评测证据和复现方法。

**当前目标：按 2026-10-15 会议指定的 10 项算子与 5 项 FlagGems-sglang 候选推进优化，固定 FlagGems/FlagTree 基线，将 HG-KDA 接入 KG，形成指标、指令与热点分阶段体检；正确性失败对象先修复，再报告性能。此前低于 0.8× 的四算子阶段已完成。**

2026-10-09 导师确认：Excel 数字是加速比，绿色验收门槛为大于 0.8×，FlagGems 使用 `master`。筛选依据是华为列实际数字；黄色底色不作为立项依据。旧实验锁定的版本不是新的 master 基线。

## 当前状态

正在执行[六算子夜间连续队列](project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/README.md)：`selu_`、`rad2deg_`、`cumulative_trapezoid`、`scatter_add`、`uniform`、`gelu_and_mul`。服务器tmux串行运行，自动衔接搜索、补充语义、一次独立完整评测及最终采集；阶段结果不表示整轮完成。最终报告和三端同步验收完成后再更新交付成绩。

最新[缓冲与搬运批次](project/Ascend-Optimization-v1/operators/campaigns/buffer-pointwise-20261022/README.md)已闭环：`log10_` **0.875×**、`selu_` **0.898×**，各36项正式测试与42＋12＋9项补充检查通过，最终真机采集及48项证据核验通过。log10_相对同批父版实际加速**1.125×**；SELU保留上一轮源码，本次分数属于复测结果。新增源码、失败机制、中文简报和一次完整复现命令见该批次；历史成绩如下保留。

2026-10-22的[log10_、selu_成对诊断](project/Ascend-Optimization-v1/operators/campaigns/paired-pointwise-20261022/精简汇报.md)已完成方法接入、源码优化与完整验证：最终完整调用加速比分别为**0.737×、0.790×**，各通过36项正式测试、42＋12项补充检查，最终KGS profiling及成对硬件采集成功，48项证据核验通过；整组尚未达0.8×。小输入定位到启动参数处理，大FP16的SELU从81.6降至45.6微秒；固定后端的UB约束和失败候选一并归档。源码和复现入口见[批次README](project/Ascend-Optimization-v1/operators/campaigns/paired-pointwise-20261022/README.md)。

后续汇报遵循[实验归属与汇报口径](docs/实验归属与汇报口径.md)：FlagGems与FlagGems-SGLang分别成表，简表采用“当前优化版”，写清Codex参与，不将混合流程成绩称为纯原生KG独立效果。

2026-10-21 的[15项体检与四组优化](project/Ascend-Optimization-v1/operators/夜间大goal-20261021.md)已结束：固定原版12项完整通过、3项部分失败。`value_selecting_reduction_backward`有效最佳版相对PyTorch为 **1.593× / 1.679×**，`moe_sum_reduce`为 **1.628× / 1.619×**；均通过扩展语义和两次完整复验，来源为KG生成后的独立修复。HG-KDA已真实接入KG，原始组与额外修复分别记录。见[精简汇报](project/Ascend-Optimization-v1/operators/campaigns/night-20261021-v3/精简汇报.md)、[交付源码与逐case数据](project/Ascend-Optimization-v1/operators/campaigns/night-20261021-v3/交付清单.md)。本批使用KGS Native完整调用walltime，与下述历史设备kernel成绩分别报告。

2026-10-15 首项 `rad2deg_` 已完成历史批次闭环：固定 FlagGems 为0.253×，HG-KDA最终两次为0.803×/0.806×，均33/33，另通过21项补充语义检查；相对 FlagGems 为3.20×/3.19×。已验证 metrics→模拟器指令的阶段升级，并核查 CANN DMA 直接调用链。该批采用设备kernel计时，结果、失败记录与复现步骤见[会议后首项交付](project/Ascend-Optimization-v1/operators/会议后首项交付-20261015.md)；最新完整调用体检以上述夜间批次为准。

2026-10-15 会议后进入[下一阶段](project/Ascend-Optimization-v1/operators/下一阶段计划-20261015.md)：固定 FlagGems 与 FlagTree 版本，登记会议指定的 10 项算子与 5 项 FlagGems-sglang 候选；原记录精度失败的指定对象先修正确性。HG-KDA 新增[项目内 skill、分类知识与分阶段体检入口](project/Ascend-Optimization-v1/tools/profiling/hgkda/README.md)，评测继续使用 KGS。以下四项属于上一阶段已完成成果。

四算子的[统一工作流](project/Ascend-Optimization-v1/tools/profiling/四算子工作流.md)已完成本阶段验证与交付：历史最佳源码复现为 logaddexp2 1.262×、amin 0.934×、矩阵乘 0.889×、narrow_copy 1.105×，均完整通过并完成两次复验。从明确优化前源码调用 CC+DeepSeek 的四项生成实验也已完成，3 项在最终复验中取得有效提升；失败和未采纳代码保留。生成成绩、起点与方法见[最终交付](project/Ascend-Optimization-v1/operators/四算子工作流交付-20261014.md)和[最新进展](project/Ascend-Optimization-v1/operators/四算子工作流进展-20261014.md)。源码复验与方法生成分别记账，数值 Roofline 尚未建立。历史成绩保留如下。

2026-10-14：[持续优化闭环](project/Ascend-Optimization-v1/tools/profiling/README.md)已完成矩阵乘和amin各六轮、完整评测及两次最终复验。logaddexp2历史无profiler/有profiler为1.061×/1.267×；历史方法workflow已无人值守复现，两次1.269×/1.261×，均36/36。见[本阶段交付](project/Ascend-Optimization-v1/operators/一阶段交付-20261014.md)和[历史方法](project/Ascend-Optimization-v1/tools/profiling/历史复现知识-20261014.md)。

| 对象 | Excel 华为列 | 状态 |
|---|---:|---|
| [narrow_copy](project/Ascend-Optimization-v1/operators/narrow_copy/README.md) | G29：0.0138× | 四版本 0.010×、0.495×、0.806×、1.102×，均 33/33；最终相对 master 108.23× |
| [matmul_bias_activation](project/Ascend-Optimization-v1/operators/matmul_bias_activation/README.md) | G611：0.5408× | 本批四版本 0.439×、0.751×、0.713×、0.906×，均 42/42；新候选两次 0.908×、0.906×，相对 master 2.08× |
| [amin](project/Ascend-Optimization-v1/operators/amin/README.md) | G92：0.5536× | 本批四版本0.308×、0.742×、0.512×、0.936×，均27/27；新候选两次0.942×、0.936×，相对启动兼容master为3.04× |
| [logaddexp2](project/Ascend-Optimization-v1/operators/logaddexp2/README.md) | G122：0.3218× | master 0.231×；历史无/有profiler 1.061×/1.267×；方法workflow复现1.269×/1.261×，均36/36 |

此前按颜色误选的 MSE backward、PReLU backward、t_copy、SmoothL1 backward、BatchNorm backward、native_layer_norm，Excel 数值均已超过 0.8×，已移入 [历史归档](project/Ascend-Optimization-v1/archive/README.md)。原结果保留为流程和优化方法参考，不计入新目标的成果。

narrow_copy、matmul_bias_activation和amin均超过0.8×。新一轮矩阵乘把双tile复用A扩展到FP32；amin去掉整除维度的边界掩码，调整归约精度与缓冲配置。见[矩阵乘最新汇报](project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/master-closure-20261012-agent/精简汇报.md)、[本批交接](project/Ascend-Optimization-v1/tools/profiling/复现与交接-20261013.md)。[2026-10-12成果表](project/Ascend-Optimization-v1/operators/低于0.8算子四版本闭环-20261012.md)保留上批数据。

现有 narrow_copy 分析使用过 msprof 指标、模拟器指令及编译产物。matmul 原版的 Msopprof 官方 Roofline 文本给出 `memory caused`，已核验 kernel、case 和物理卡身份；计数缩放与硬件 roof 尚未核实，数值 Roofline 仍待建立。

## 目录

```text
Ascend-Optimization/
├── project/Ascend-Optimization-v1/
│   ├── operators/          当前主线，按算子集中保存候选、专用脚本和报告
│   ├── tools/              多算子共用的环境、评测、profiling 和验证工具
│   ├── archive/            误选算子、联合对照、烟雾测试和旧阶段记录
│   └── 验收看板_结果表.xlsx
├── sources/                KG 6.7.0、KGS 6.5.0 源码及原始压缩包
├── third_party/FlagGems/    本轮固定 master 源码，随总仓库提供
├── manifests/              组件版本与环境清单
└── docs/                   架构、profiling 资料和方法设计
```

[项目索引](project/Ascend-Optimization-v1/README.md)说明文件归属与历史状态。候选和专用脚本跟随算子保存，公共工具由多个算子复用；评测批次中的源码快照用于证明当次实际测了什么，继续保留。

控制端迁移与资料入口：[Linux 控制端迁移](docs/Linux控制端迁移.md)、[控制工具](project/Ascend-Optimization-v1/tools/control/README.md)、[导师要求原始记录](docs/项目要求与导师沟通.md)、[模型接入](docs/模型接入说明.md)、[服务器清单](docs/服务器清单.md)。这些内容已纳入仓库，不再依赖 Windows 外层临时文件。

## 分步计划

### 第 0 步：准备独立环境

1. 按 [组件清单](manifests/components.lock.yaml)准备 CANN、Python、torch_npu、Triton、KG 和 KGS，记录实际版本、设备号、容器和运行命令。
2. 检查设备占用，使用独立 KGS、Catalog、tmux 会话和运行工作区。
3. Claude Code 作为 KG 的模型调用运行时，已有实验通过 CC 接入 DeepSeek；配置和令牌保留在机器本地。环境工具见 [tools](project/Ascend-Optimization-v1/tools/README.md)。

square 的 KG+KGS 烟雾测试已完成，记录见 [烟雾测试归档](project/Ascend-Optimization-v1/archive/smoke-tests/square/README.md)。正式算子从 `sources/kernelgen_server/data/flaggems-adapter-definitions/` 选择；仓库内旧 square/softmax Catalog 已移除。

### 第 1 步：筛选并固定正式对象

1. 当前按[会议指定清单](project/Ascend-Optimization-v1/operators/下一阶段计划-20261015.md)推进十项 Excel 算子与五项 FlagGems-sglang 候选；同时记录 Excel 精度状态与加速比。指定的精度失败、无有效计时对象先修正确性或补齐评测，不再按上一阶段规则排除。
2. 本轮固定 FlagGems `master` 提交 `d6a8eec473517a3d68157b208eb9c057eb1d4c50`，源码已纳入 `third_party/FlagGems/`，逐文件 SHA 见 `manifests/flaggems-master-files.json`。下一轮更新上游时重新固定提交，不静默追随 master。
3. 在 `operators/<算子>/` 建立一个入口 README，记录 Excel 行号与数值、上游提交、实际调用实现、shape、dtype 和完整 workload。
4. 固定 FlagTree 和 FlagGems 后复测正确性与性能；两者的版本身份见[锁定清单](manifests/hgkda-stage2.lock.json)。正确性失败先修复，完整通过后再报告有效性能；是否继续优化由本阶段指定任务决定。

### 第 2 步：走通 Definition 与基线

先读 `sources/kernelgen/docs/ONBOARDING.md` 和 `sources/kernelgen/docs/design/workflows/operator_optimization.md`。

1. 用 `GemsAdapterDefinitionWorkflow`（`kg definition`）导出 Definition；已有合适 Native Catalog 时可直接优化。
2. 用 `OperatorOptimizeWorkflow`（`kg run`）执行测试审核、优化和代码审核。KG 默认配置为 1 个 Coder、1 个 epoch、最多 10 rounds；实验需明确记录实际预算。
3. 在同一设备、同一 workload 下评测 PyTorch + torch_npu 与原版 FlagGems，保存逐 case 延迟、正确性和统计口径。

给 CC 的启动提示词可用下列模板，替换算子名、提交与环境参数：

> 请先阅读 docs/ONBOARDING.md 和 docs/design/workflows/operator_optimization.md。从 kernelgen_server/data/flaggems-adapter-definitions 中使用指定算子的 Definition，完成 kg definition 和 kg run。优化对象使用会议指定清单，FlagGems 与 FlagTree 使用阶段锁定版本。先复测完整正确性与基线，再优化性能；记录设备、Catalog、KGS 地址、模型、预算、候选源码和逐 case 结果，使用新的独立工作区。

### 第 3 步：完成 KG 两组对照

使用相同起点、模型、workload、轮数及预算，分别运行 KG 无 profiling 反馈、KG 有原生 profiling 反馈，保存各轮候选及最终独立复验。

`--no-profile` 关闭优化反馈，设备计时仍可能使用 NPU profiler。必须写清设备侧 kernel 计时还是完整调用计时，不能把历史 `timing=walltime` 请求直接解释为端到端耗时。

### 第 4 步：独立 Profiling workflow 与分析优化版本

独立workflow的首要目的，是把此前Codex参与整理的瓶颈诊断与有效方法固化为可自动运行的流程。可从已验证候选复现历史成果，再按同一诊断接口优化；不要求重新发明候选。KG无profiler/有profiler标签表示运行开关，不表示工程没有Codex参与。复现同一源码时明确记录为复现；新增优化需提供实际修改和完整评测。

1. 调研当前 KG/KGS 的 Ascend 采集与反馈代码。导师说明默认使用 `msprof`，`metrics` 用于真机指标，`instruction` 使用模拟器，由 agent 根据问题选择。
2. 先做独立 workflow：输入算子语义、候选、workload 和原始 profiling 产物，输出中文体检报告；公共入口见 [tools/profiling](project/Ascend-Optimization-v1/tools/README.md)。
3. 按 [Roofline 资料](docs/Roofline%20分析.md)、[细粒度 profiling 资料](docs/Fine-grained%20Profiling算子优化.md)和 [workflow 设计](docs/ascend-profiling-workflow.md)，分析计算、访存、流水线和启动开销；只有数据范围与设备峰值匹配时才计算 Roofline。
4. 参考 [cuda-optimized-skill](https://github.com/KernelFlow-ops/cuda-optimized-skill) 的语义分析、瓶颈判断和建议组织方式，适配 910B 架构与指标；指令必要时先映射回 LLVM IR，再关联 Triton 源码。
5. 把诊断转为具体修改，生成我们的优化版本。先确保至少一个正式目标完成“分析→修改→正确性→性能”的闭环，再考虑接入 KG 主流程。

### 第 5 步：交付

每个正式算子交付一个入口报告：PyTorch/FlagGems 基线、KG 无 profiler、KG 原生 profiler、我们分析指导的版本，附关键优化、完整正确性、逐 case 性能、版本、源码和复现命令。

目标是完整正确性通过并超过 0.8×；超过 1× 表示快于 PyTorch。未达标则直接写结果和已定位的问题，保留有效尝试。`narrow_copy` 的 CANN D2D 混合实现须与 Triton 版本分开标注。

## 复现与同步

2026-10-12 已完成服务器目录收敛与三个最终候选完整复验，记录见 [目录整理验收](docs/910B目录整理验收-20261012.md)。

虚拟环境、容器层和 Claude 二进制在服务器运行环境中；源码和评测证据在本仓库。910B 收敛为唯一主仓库 `/data/hanle/ascend-optimization/Ascend-Optimization/`，运行环境在同级 `runtime/`，KG 工作区、日志和诊断在同级 `runs/`。完整目录、启动命令和历史恢复方式见 [910B 目录与复现](docs/910B目录与复现.md)。

压缩包等大文件由 Git LFS 管理；克隆后安装 Git LFS 并执行 `git lfs pull`。KG/KGS 原包位于 `sources/archives/`。导出源码不含嵌套 `.git`，版本见清单。

2026-10-10 的完整实验副本原位于 `goal-20261010/Ascend-Optimization`；本次整理将其作为唯一主仓库，原批次运行产物迁入 `runs/20261010/`。19655、19657 分别服务 amin/矩阵乘和 narrow 设备任务计时，当前均使用物理卡 7，评测串行执行。历史报告中的绝对路径、端口与版本是当时快照。旧 FlagGems 完整源码可从总仓库提交 `0008e8a8` 的 Git 历史恢复，无需额外保留完整历史 checkout。

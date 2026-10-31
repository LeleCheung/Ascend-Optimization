# 910B 算子优化项目

2026-10-31夜间六项仍在推进，当前入口为[30项验收总表](operators/campaigns/reference-closure-20261031/30项验收总表.md)和[CANN参考工具](tools/profiling/hgkda/reference_closure/README.md)。24项原参考稳定达标继续保留；CANN兼容组合、命名NPU分解与原参考条件分别报告。

2026-10-31完成[最后两项攻关交付](operators/campaigns/remaining-final-20261031/交付报告.md)：`fix` **0.862×**、`expm1_` **0.879×**新增稳定达标，各自同源码两次完整评测30/30、原扩展、24项原生路径与30项冻结新形状检查通过。30项目标累计**24项常规稳定达标、1项求逆正确性未通过、5项特殊参考条件单列**。本轮是Codex自写Ascend C＋Triton特殊布局回退，未新增CC/DeepSeek生成。新增执行路径诊断入口，分别核算候选/参考延迟变化；14份正式请求、12条设备路径、2份真实Triton IR已核验。[复现工具](tools/profiling/hgkda/remaining_final/README.md)和本批`delivery.json`为最新入口，下列旧日期内容保留为历史记录。

2026-10-30接续[剩余九项攻关交付](operators/campaigns/remaining-nine-20261030/交付报告.md)：新增稳定达标6项——`negative_` 0.937×、`log2_` 0.915×、`round_out` 1.412×、`special_round_out` 1.247×、`hardswish` 0.905×、`fmod_tensor_` 1.242×。均为自写Ascend C＋Triton布局回退，通过同源码两次完整KGS、原扩展及对齐路径检查。`fix`第三次0.766×未过线，`expm1_`0.697×及求逆5/7继续列为未完成。30项累计22项常规达标，另有5项特殊参考条件；37份正式请求、52条设备路径、9份实际IR已核验。[复现工具](tools/profiling/hgkda/remaining/README.md)和交付清单提供源码、附件身份及单次新复测入口。

2026-10-30完成[三路对照与新增达标](operators/三路对照与新增达标-20261030.md)：`cross_out` **3.891×**、`nan_to_num_` **0.866×**、`native_group_norm` **0.814×**新增达标，`log10_` **0.938×**恢复稳定达标；四项完整评测和补测通过，既有12项原样复验通过。30项累计16项常规达标，CANN三项直接对照、最终源码、实际IR和单命令复现均已保存。

当前新增任务见[2026-10-29新25算子计划](operators/下一阶段计划-20261029.md)，执行工具位于[tools/profiling/hgkda/next_phase](tools/profiling/hgkda/next_phase/README.md)。旧阶段成果和原始合同继续保留，以下2026-10-15目标属于历史阶段。

2026-10-30新25项首遍已完成：六项本批新达标、三项历史复验达标、十二项正确但性能未达标，累计积分与矩阵求逆未交付，两个原版高分因参考CPU回退单列。[精简汇报](operators/campaigns/next25-20261029-v2/本批精简汇报-20261030.md)、[25项全表](operators/campaigns/next25-20261029-v2/已核验进展.md)及[接续说明](operators/campaigns/next25-20261029-v2/夜间接续-20261030.md)已更新；本轮为KGS完整调用计时，不混入历史设备成绩。

当前按 2026-10-15 会议推进[下一阶段](operators/下一阶段计划-20261015.md)：固定 FlagGems/FlagTree，登记 10 项指定算子与 5 项 SGLang 候选，构建可插入 KG 的分阶段 HG-KDA 体检流程，评测使用 KGS。指定精度失败对象先修正确性。此前低于 0.8× 的四算子阶段已完成。完整步骤见 [总 README](../../README.md)。

2026-10-26最新[10项最终验收](operators/会议10算子最终验收-20261026.md)：7项默认达标、1项确定性条件达标、2项补充NPU分解对照通过。本轮池化较同批父版实际加速1.547×；默认扫描参考反例和两项CPU回退叶子算子均已定位。源码、两种参考、成对trace与实际IR、96项证据核验和复现见[本批交付](operators/campaigns/acceptance-three-20261026/README.md)。

## 文件结构

本阶段首项 `rad2deg_` 已完成两次33/33完整复验，相对 PyTorch 为0.803×/0.806×，相对固定 FlagGems 为3.20×/3.19×；另通过21项补充语义检查。成绩、真实阶段升级、失败记录及复现步骤见[会议后首项交付](operators/会议后首项交付-20261015.md)。

```text
Ascend-Optimization-v1/
├── operators/
│   ├── narrow_copy/
│   │   ├── README.md
│   │   ├── candidates/     实验候选、验证代码和提示词
│   │   ├── scripts/        该算子的专用运行、采集与分析工具
│   │   └── reports/        按批次保存结果、原始请求及源码快照
│   ├── amin/               启动兼容基线、直接归约候选与 KG 两组
│   ├── matmul_bias_activation/
│       ├── README.md
│       ├── candidates/     固定 master 和流水候选
│       ├── scripts/        分块、Roofline 采集与解读
│       └── reports/        新 master 复测、优化结果和历史证据
│   └── logaddexp2/          新正式对象，固定master与三种优化流程
├── tools/                  多个算子共用的工具
├── archive/                退出当前主线的实验与历史对照
└── 验收看板_结果表.xlsx
```

## 为什么按算子集中

一次优化需要同时查看候选、运行脚本和证据，按算子存放可直接找到完整实验。算子内部仍区分候选、脚本和报告：候选会继续修改，报告中的源码快照固定当次评测版本，两者用途不同。公共环境、评测和分析器由多个算子复用，放在 `tools/`。

只在实际产生文件时创建子目录，不预建空的三版本目录。详细实验按批次保存实际运行配置、输入版本、Codex参与和最终候选；旧批次名称保持原样。面向用户的简表遵循[实验归属与汇报口径](../../docs/实验归属与汇报口径.md)，将FlagGems与FlagGems-SGLang分表，不默认展开KG无／有profiler列，不将Codex参与后的结果称为纯原生KG独立效果。

## 阅读入口

最新[两算子收尾交付](operators/两算子收尾交付-20261024.md)：上采样**0.915×**、原合同24/24，累计积分**0.812×（官方确定性条件）**、33/33。默认累计积分参考不稳定的固定输入反例及复现步骤一并交付；不将确定性成绩混成默认合同通过。完整源码、门禁、原始证据和单次复现命令见[本批README](operators/campaigns/two-closure-20261024/README.md)。

[六算子夜间连续优化已结束](operators/夜间连续优化交付-20261023.md)：`gelu_and_mul` 2.271×、`uniform` 1.325×、`rad2deg_` 1.117×、`selu_` 0.837×；SELU 同批实际加速1.042×。`scatter_add` 0.419×未达标，`cumulative_trapezoid`未通过数值门禁。七份完整原始快照共4507个文件、62项离线证据核验通过；SELU/rad2deg_的KGS metrics缺口单列。最终源码与结果以[机器清单](operators/夜间连续优化交付-20261023.json)为准，uniform已替换为标准Philox修复版，旧1.497×不作为交付。

上轮结果：[log10_、selu_缓冲与搬运优化](operators/campaigns/buffer-pointwise-20261022/README.md)。最终0.875×、0.898×；log10_相对同批父版实际加速1.125×，SELU保留旧源码。各36项正式测试及42＋12＋9项补充检查通过，最终采集和48项证据核验闭环。源码、简报和一次完整复现命令均在该批次内；上轮0.737×、0.790×保留在[历史成对批次](operators/campaigns/paired-pointwise-20261022/README.md)。

| 入口 | 用途 |
|---|---|
| [四算子工作流交付](operators/四算子工作流交付-20261014.md) | 四项历史最佳复现、真实方法生成结果与证据 |
| [四算子工作流进展](operators/四算子工作流进展-20261014.md) | 最新成绩和工作流固化内容 |
| [主线算子](operators/README.md) | amin、matmul_bias_activation、narrow_copy 的四版本闭环 |
| [本批进展](operators/持续优化进展-20261013.md) | 新候选成绩、已完成四版本和待办 |
| [上批四版本成果表](operators/低于0.8算子四版本闭环-20261012.md) | 2026-10-12交付成绩与源码 SHA |
| [矩阵乘最新精简汇报](operators/matmul_bias_activation/reports/master-closure-20261012-agent/精简汇报.md) | 自动闭环两次复验与FP32双tile改进 |
| [持续优化交接](tools/profiling/复现与交接-20261013.md) | tmux队列、恢复、证据核验与未完成事项 |
| [复现与交接](tools/evaluation/复现与交接-20261010.md) | 真机复测、KG 两组及离线证据核验 |
| [公共工具](tools/README.md) | 环境、KG/KGS 评测和独立 profiling workflow |
| [历史归档](archive/README.md) | 六个误选对象及保留原因、联合复测与烟雾测试 |
| [独立 workflow 设计](../../docs/ascend-profiling-workflow.md) | 已有证据分析与后续 Roofline 工作 |

## 整理边界

原始评测 JSON、日志、压缩包、源码快照与 Excel 保留原内容；只更新可维护的运行入口、测试定位和 Markdown 路径。KG/KGS 源码在根目录 `sources/`，本轮固定 master 的 FlagGems 源码在 `third_party/FlagGems/`；历史完整源码从总仓库 Git 历史恢复。

仓库内旧 `runtime/` 和空 `kernels/` 已移除。910B 唯一仓库路径为 `/data/hanle/ascend-optimization/Ascend-Optimization/`，同级 `runtime/` 保存运行环境，`runs/<批次>/` 保存工作区和日志；见 [目录与复现](../../docs/910B目录与复现.md)。2026-10-14交付：amin 0.936×、矩阵乘0.906×；logaddexp2历史有profiler为1.267×，方法workflow已无人值守复现1.269×/1.261×。narrow_copy此前交付1.102×。详见[阶段报告](operators/一阶段交付-20261014.md)。

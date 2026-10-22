# 910B 算子优化项目

当前按 2026-10-15 会议推进[下一阶段](operators/下一阶段计划-20261015.md)：固定 FlagGems/FlagTree，登记 10 项指定算子与 5 项 SGLang 候选，构建可插入 KG 的分阶段 HG-KDA 体检流程，评测使用 KGS。指定精度失败对象先修正确性。此前低于 0.8× 的四算子阶段已完成。完整步骤见 [总 README](../../README.md)。

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

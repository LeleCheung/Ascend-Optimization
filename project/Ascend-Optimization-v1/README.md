# 910B 算子优化项目

当前任务是优化验收看板华为列低于 0.8×、正确性通过的算子，使用 FlagGems `master` 的固定提交，对照 KG 无 profiling 反馈、KG 原生反馈及我们分析指导的版本。完整步骤见 [总 README](../../README.md)。

## 文件结构

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

只在实际产生文件时创建子目录，不预建空的三版本目录。新报告用批次目录分别标明 `kg-no-profile`、`kg-profile`、`agent-optimized`，写清输入版本与最终候选；旧批次名称保持原样。

## 阅读入口

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

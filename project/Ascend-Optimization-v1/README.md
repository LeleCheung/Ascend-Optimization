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
│   └── matmul_bias_activation/
│       ├── README.md
│       └── reports/        已有历史实验；新一轮尚未开始
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
| [主线算子](operators/README.md) | narrow_copy、matmul_bias_activation 与下一对象 |
| [公共工具](tools/README.md) | 环境、KG/KGS 评测和独立 profiling workflow |
| [历史归档](archive/README.md) | 六个误选对象及保留原因、联合复测与烟雾测试 |
| [独立 workflow 设计](../../docs/ascend-profiling-workflow.md) | 已有证据分析与后续 Roofline 工作 |

## 整理边界

原始评测 JSON、日志、压缩包、源码快照与 Excel 保留原内容；只更新可维护的运行入口、测试定位和 Markdown 路径。KG/KGS 源码在根目录 `sources/`，历史 FlagGems 源码在 `third_party/FlagGems/`。

仓库内旧 `runtime/` 和空 `kernels/` 已移除。910B 仓库外运行环境保持原状。2026-10-09 本次整理仅在 Windows 完成，尚未提交或同步服务器。

# 首批六算子真实优化轨迹

先展示方法、分析信息与结果；不以独立因果贡献或超过历史最佳作为画图门槛。

打开[index.html](index.html)查看总览或单张；每张提供SVG、PNG。traces.json保存对应账本、源码身份、修改计划和原始评测信息。

每图是一段连续实验，不是该算子从首次优化到当前交付的完整生涯。百分比表示相邻两轮加速比的变化；历史候选不自动等同当前严格交付版本。

| 算子 | 本段成绩 | 图 |
| --- | --- | --- |
| `rad2deg_` | 0.313× → 0.861× → 1.129× → 1.134× | [SVG](rad2deg_.svg)／[PNG](rad2deg_.png) |
| `uniform` | 0.152× → 0.892× → 1.493× | [SVG](uniform.svg)／[PNG](uniform.png) |
| `negative_` | 0.160× → 0.283× → 0.413× | [SVG](negative_.svg)／[PNG](negative_.png) |
| `selu_` | 0.730× → 1.002× | [SVG](selu_.svg)／[PNG](selu_.png) |
| `log10_` | 0.915× → 0.751× → 0.834× | [SVG](log10_.svg)／[PNG](log10_.png) |
| `native_group_norm` | 0.592× → 0.582× → 0.547× | [SVG](native_group_norm.svg)／[PNG](native_group_norm.png) |

分析信息来源依据冻结的plan／conclusion及既有区域报告归纳；未将历史模型的机制判断自动升级为已证明因果。
SELU的成对profiling属于批次背景，本图修改方向直接来自宿主探测；本轮新的采集失败不改变已有测量节点的展示。
native_group_norm有partial模拟器材料，参与当前两次修改的具体贡献待核验。
log10结合成对指标、模拟器及IR；图显示双块预取失败和后续恢复，没有声称该轮净收益。

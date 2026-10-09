# narrow_copy 阶段索引

连续 dim=0 输入走 CANN D2D，其他布局走 Triton。三次完整评测为 0.5919×、0.5672×、0.5613×，正确性全部通过，相对原 Triton 候选提升约 9%～11%，仍未达到 0.8×。

| 阶段 | 入口 |
|---|---|
| 初始 KG 与 profiling 配对 | [初始报告](narrow-copy-20261005/README.md)、[配对结果](narrow-copy-20261005/ab-report.md) |
| launcher 修复及重复评测 | [逐例对照](narrow-copy-fixed-20261006/逐例对照.md) |
| 合并候选与独立诊断 | [独立体检](narrow-copy-combined-20261006/独立体检.md)、[persist8 失败尝试](narrow-copy-combined-20261006/诊断消融-persist8.md) |
| CANN D2D 验证 | [最终结果](narrow-copy-final-20261006/native-copy-result.md) |

候选与专用脚本见 [算子入口](../README.md)，公共分析器见 [工具索引](../../../tools/README.md)。此前已分析 profiling 指标，Roofline 尚无可用结果。

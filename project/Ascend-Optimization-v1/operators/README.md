# 当前主线算子

筛选依据是验收看板华为列实际加速比低于 0.8×，并且正确性通过；后续统一使用 FlagGems `master` 固定提交复测。

| 算子 | Excel 位置与加速比 | 当前记录 | 下一步 |
|---|---|---|---|
| [narrow_copy](narrow_copy/README.md) | G29：0.0138× | D2D 混合候选约 0.56～0.59×，未达标 | 新 master 基线复测后决定是否继续 |
| [matmul_bias_activation](matmul_bias_activation/README.md) | G611：0.5408× | 四版本均 42/42；master 0.443×、KG 无 profiler 0.740×、KG profiler 0.710×、我们的 v6 0.773× | 编译流水线实验，争取超过 0.8× |
| [amin](amin/README.md) | G92：0.5536× | 四版本均 27/27；兼容基线 0.296×、KG 无 profiler 0.769×、KG profiler 0.499×、我们的 v5 0.916× | 已超过 0.8×，继续 bf16 归约改进 |

一个算子一个目录，README 串起候选、专用脚本和报告。新对象只在正式复测时创建目录；通用工具见 [tools](../tools/README.md)。

Excel 与历史 KGS 实测对应不同版本或测试范围，分别记录。此前误选的六个算子已移至 [归档](../archive/README.md)。

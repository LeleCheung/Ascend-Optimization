# matmul_bias_activation

验收看板华为列 G611 为 0.5408×，符合低于 0.8× 的筛选条件，因此保留在当前主线。

历史候选约 0.49×，详情见 [阶段报告](reports/matmul-bias-activation-20261007/README.md)。这些记录对应旧版本；下一步先固定 FlagGems master 并复测正确性、PyTorch 与 FlagGems 基线，再开展 KG 无/有 profiling 及我们的优化版本。

当前只有已产生的报告，新候选与专用脚本产生后再加入本算子目录。

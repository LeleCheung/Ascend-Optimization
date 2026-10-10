# matmul_bias_activation

验收看板华为列 G611 为 0.5408×，符合低于 0.8× 的筛选条件，因此保留在当前主线。

2026-10-10 固定 FlagGems master 为 `d6a8eec473517a3d68157b208eb9c057eb1d4c50`，使用独立 KGS 19655、物理卡 7、单 worker。完整范围为 27 个正确性用例和 15 个性能用例，float16/float32/bfloat16。

| 版本 | 相对 PyTorch | 相对 master | 完整测试 |
|---|---:|---:|---|
| FlagGems master | 0.443× | 1.00× | 42/42 |
| 我们的 K128 流水候选 v3 | 0.747× | 1.69× | 42/42 |
| 我们的 K256 流水候选 v4 | 0.759× | 1.71× | 42/42 |
| KG 无 profiler | 待完成 | 待完成 | 待独立复验 |
| KG 原生 profiler | 待完成 | 待完成 | 待独立复验 |

候选将 K 分块由 32 增大到 128，启用多缓冲和流水调度；规则矩阵走优化路径，非整齐矩阵保留 master。Msopprof 官方 Roofline 文本对 1024³ fp16 case 给出 `memory caused`，CSV 与基本信息已核验；原始 FOP 计数与算法 FLOPs 相差 128 倍，数值 Roofline 需要先核实计数缩放和 roof。见 [官方采集解读](reports/diagnostics-20261010/matmul-roofline-20261010/官方Roofline解读.md)。

v4 使用 dot 的累加器参数，并将规则低精度大矩阵 K 分块增加到 256，已完成完整 42 用例评测，全部通过。整体相对 PyTorch **0.759×**，相同 case 实际延迟相对 master **1.710×**；PyTorch 参考几何均值漂移为 +0.14%。见 [v4 对比](reports/profiling-k256-dotacc-v4-20261010/版本对比.md)。更宽的 128×256/256×128 tile 在多缓冲下超出 UB 容量，失败原文保留在分块诊断记录。

原生 KG profiler 组第二轮已得到完整通过的 **0.697×** 候选，目前由原生分析器完成最后一轮 profiling 与收尾，之后独立复验；无 profiler 组仍按串行调度等待。原生组自行分析和选择参数，没有接受我们的优化建议。

候选通过完整测试，但整体仍低于 0.8×，继续优化。主要短板是半精度大矩阵，当前相对 PyTorch 约 0.55～0.62×。性能是设备 kernel 计时；原版与候选的 PyTorch 参考延迟几何均值漂移约 -0.22%。原始记录见 [master](reports/master-20261010/flaggems-master-1.result.json)、[v3](reports/profiling-k128-pipeline-v3-20261010/profiling-k128-pipeline-v3-1.result.json)。

v1 未通过正确性；v2 在完整测试中卡住，停止本轮对应测试进程后记录为 HTTP_ERROR。失败记录保留，不能作为成果。v3 已绕开该路径，但卡住的具体编译或运行时原因仍未证实。

历史候选约 0.49×，见 [旧阶段报告](reports/matmul-bias-activation-20261007/README.md)。旧版本数据不与本轮 master 混算。

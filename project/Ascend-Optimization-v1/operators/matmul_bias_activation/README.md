# matmul_bias_activation

验收看板华为列 G611 为 0.5408×，符合低于 0.8× 的筛选条件，因此保留在当前主线。

2026-10-10 固定 FlagGems master 为 `d6a8eec473517a3d68157b208eb9c057eb1d4c50`，使用独立 KGS 19655、物理卡 7、单 worker。完整范围为 27 个正确性用例和 15 个性能用例，float16/float32/bfloat16。

| 版本 | 相对 PyTorch | 相对 master | 完整测试 |
|---|---:|---:|---|
| FlagGems master | 0.443× | 1.00× | 42/42 |
| 我们的 K128 流水候选 v3 | 0.747× | 1.69× | 42/42 |
| 我们的 K256 流水候选 v4 | 0.759× | 1.71× | 42/42 |
| 我们的设备计时选块候选 v5 | 0.772× | 1.75× | 42/42 |
| 我们的分组调度候选 v6 | 0.773× | 1.74× | 42/42 |
| KG 无 profiler（独立复验） | 0.740× | 1.67× | 42/42 |
| KG 原生 profiler（独立复验） | 0.710× | 1.59× | 42/42 |

候选将 K 分块由 32 增大到 128，启用多缓冲和流水调度；规则矩阵走优化路径，非整齐矩阵保留 master。Msopprof 官方 Roofline 文本对 1024³ fp16 case 给出 `memory caused`，CSV 与基本信息已核验；原始 FOP 计数与算法 FLOPs 相差 128 倍，数值 Roofline 需要先核实计数缩放和 roof。见 [官方采集解读](reports/diagnostics-20261010/matmul-roofline-20261010/官方Roofline解读.md)。

v4 使用 dot 的累加器参数，并将规则低精度大矩阵 K 分块增加到 256，已完成完整 42 用例评测，全部通过。整体相对 PyTorch **0.759×**，相同 case 实际延迟相对 master **1.710×**；PyTorch 参考几何均值漂移为 +0.14%。见 [v4 对比](reports/profiling-k256-dotacc-v4-20261010/版本对比.md)。更宽的 128×256/256×128 tile 在多缓冲下超出 UB 容量，失败原文保留在分块诊断记录。

v5 根据三个 dtype、1024/2048/4096 三个尺寸的设备计时选择配置，并重新完成 **42/42** 评测，相对 PyTorch **0.772×**、相对 master **1.746×**。诊断共 72 组，其中 15 组编译运行成功。128×256/256×128 输出 tile 仍超出 UB，fp32 的 BK512 超出 CBUF；不采用失败配置。当前选中的均为 128×128、BK256，变化包括 fp32 大矩阵扩大 K 分块，以及方阵改为单维 grid 和指针递增。见 [v5 对比](reports/profiling-kwide-v5-20261010/版本对比.md)。

原生 KG profiler 组第二轮为 **0.697×**；无 profiler 组第二轮为 **0.741×**。随后独立复验均 **42/42** 通过，成绩分别为 **0.710×** 和 **0.740×**。四版本数据已经收齐，见 [独立四版本对比](reports/master-closure-20261010/版本对比.md)及同名 CSV。原生组自行分析和选择参数，没有接受我们的优化建议。

分组 v6 已通过完整 42 项，结果 **0.773×**，与 v5 的 0.772× 基本持平；2048 方阵采用 GROUP=4，4096 方阵保留 GROUP=0。编译流水线 v7 完成 108 组诊断，从逐值一致且设备计时更快的结果中选出 5 个配置，完整复验 **42/42、0.748×**，没有超过 v6，最终保留 v6。去掉规则矩阵 mask、ND→NZ 转换路径及缓冲选项未带来整体突破。当前 backend 暴露 `enable_preload`，但 CANN 9.0.0 的 BiSheng 不接受对应 `--enable-preload=True` 参数；关闭子块绑定的部分配置超出 UB。失败原文与逐配置数据见 [编译诊断](reports/diagnostics-20261010/matmul-compiler-pipeline-20261010/artifacts/compiler-pipeline.json)，完整结果见 [v7](reports/profiling-kcompiler-v7-20261010/profiling-kcompiler-v7-1.result.json)。

候选通过完整测试，但整体仍低于 0.8×，继续优化。主要短板是半精度大矩阵，当前相对 PyTorch 约 0.55～0.62×。性能是设备 kernel 计时；原版与候选的 PyTorch 参考延迟几何均值漂移约 -0.22%。原始记录见 [master](reports/master-20261010/flaggems-master-1.result.json)、[v3](reports/profiling-k128-pipeline-v3-20261010/profiling-k128-pipeline-v3-1.result.json)。

v1 未通过正确性；v2 在完整测试中卡住，停止本轮对应测试进程后记录为 HTTP_ERROR。失败记录保留，不能作为成果。v3 已绕开该路径，但卡住的具体编译或运行时原因仍未证实。

历史候选约 0.49×，见 [旧阶段报告](reports/matmul-bias-activation-20261007/README.md)。旧版本数据不与本轮 master 混算。

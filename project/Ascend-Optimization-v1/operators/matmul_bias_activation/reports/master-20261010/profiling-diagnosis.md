# matmul_bias_activation 独立 Profiling 体检

候选 SHA-256：`e6a28cc92428c9500473e7d7516cb2a1e4b2edd1dbe9edbd67b6124a414c51d8`；KGS 正式评测 42/42，geo mean 0.4434×；2/15 个 timing case 有独立采集。

计时范围：`device_kernel`。

| case | PyTorch us | 候选 us | 加速比 | 真机 kernel us | 诊断 | 置信度 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| bfloat16/0 | 12.09 | 11.08 | 1.091× | — | inconclusive | low |
| bfloat16/1 | 636.31 | 2522.95 | 0.252× | — | inconclusive | low |
| bfloat16/2 | 25.12 | 66.58 | 0.377× | — | inconclusive | low |
| bfloat16/3 | 100.13 | 353.12 | 0.284× | — | inconclusive | low |
| bfloat16/4 | 636.46 | 2524.95 | 0.252× | — | inconclusive | low |
| float16/0 | 11.65 | 11.17 | 1.042× | 11.25 | gemm_control_and_parallelism | medium |
| float16/1 | 614.15 | 2516.38 | 0.244× | — | inconclusive | low |
| float16/2 | 23.55 | 66.56 | 0.354× | 67.52 | gemm_data_pipeline | medium |
| float16/3 | 94.53 | 351.69 | 0.269× | — | inconclusive | low |
| float16/4 | 615.84 | 2511.91 | 0.245× | — | inconclusive | low |
| float32/0 | 13.10 | 13.65 | 0.960× | — | inconclusive | low |
| float32/1 | 2271.67 | 3677.41 | 0.618× | — | inconclusive | low |
| float32/2 | 49.87 | 91.21 | 0.547× | — | inconclusive | low |
| float32/3 | 283.15 | 496.04 | 0.571× | — | inconclusive | low |
| float32/4 | 2274.03 | 3682.33 | 0.618× | — | inconclusive | low |

## 逐 case 判断

### benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::0

小矩阵的 Scalar 活跃比例超过一半，原 tile 的 program 数较少；循环控制和并行覆盖应优先优化。

下一项实验：用较小 M/N tile 覆盖更多核，同时增大 K 分块以减少循环；完整复验。

反证条件：更高 program 数反而增加搬运，或总延迟没有改善。

- `instruction`：`Ascend-Optimization\project\Ascend-Optimization-v1\operators\matmul_bias_activation\reports\master-20261010\profile-instruction-float16-0\response.json`；profile ID `3d550d6ca6304067a3cb649459139efd`；原始 artifact 10 个，SHA 已校验。
  模拟器指令 978 条；映射源码行 [44, 73, 75, 76, 80, 84, 86, 93]。
- `metrics`：`Ascend-Optimization\project\Ascend-Optimization-v1\operators\matmul_bias_activation\reports\master-20261010\profile-metrics-float16-0\response.json`；profile ID `b45de17ea2714dd1b28b4dec223cdcc9`；原始 artifact 4 个，SHA 已校验。

### benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::2

矩阵乘的 AIC MTE2 活跃比例为 85.2%，Scalar 为 0.3525。优先增加 K 分块、减少循环和地址计算，再重叠搬运与计算。流水线比例不等于 HBM 带宽利用率，各比例不能相加。

下一项实验：增大 K 分块至 128/256，按 shape 调整 M/N tile，测试 multibuffer/unit_flag；完整正确性与所有 timing case 复验。

反证条件：MTE2 和总延迟不改善，或新增分块导致精度失败。

- `metrics`：`Ascend-Optimization\project\Ascend-Optimization-v1\operators\matmul_bias_activation\reports\master-20261010\profile-metrics-float16-2\response.json`；profile ID `b42b5cd4bca34558839db83cdda85927`；原始 artifact 4 个，SHA 已校验。

## 证据边界

Roofline：**unavailable**。未验证同一 kernel/range 的 FLOPs、实际搬运 Bytes、执行时间和硬件 roof
- profile 与正式 Eval 是独立采集，不能相减推断精确 host 开销
- profile evaluation_id 不是正式 Eval ID；通过 Definition 指纹、case 与完整候选源码绑定
- 未采集的 case 只保留正式计时，不推广其他 case 的瓶颈结论

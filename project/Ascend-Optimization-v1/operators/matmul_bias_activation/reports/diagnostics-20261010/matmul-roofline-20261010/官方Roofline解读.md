# matmul_bias_activation 官方 Roofline 采集解读

Msopprof 给出 **latency bound: memory caused**。这是该工具对本 case 的定性诊断；完整数值 Roofline 尚未建立。

固定 master、1024³ fp16、物理卡 7；kernel `matmul_bias_activation_kernel_mix_aic`；采集 task duration 为 73.441467 μs。

| 指标 | program 记录均值 |
| --- | ---: |
| aic_cube_ratio | 4.54% |
| aic_mte1_ratio | 5.01% |
| aic_mte2_ratio | 19.95% |
| aic_scalar_ratio | 7.52% |

上述均值用于描述 CSV，流水线比例不能相加，也不代表 HBM 峰值利用率。

原始 Cube FOP 计数总和 16777216，算法 GEMM FLOPs 2147483648，相差 128 倍。需要先确认计数单位和缩放，才能画可信的数值 Roofline。

优化方向：增加 K 分块、减少循环与搬运同步，启用 multibuffer/unit_flag；宽 tile 若超出 UB 容量则缩小分块。效果以完整评测结果为准。

早期 summary.json 仅按文件名寻找 roofline，漏掉了 stdout 中的官方结论。该原始 summary 保留，本报告补充正确解释。

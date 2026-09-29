# AKG Bench Lite 13 个比赛算子 Ascend 优化结果

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

> 自动汇总时间：2026-08-05T23:07:56+08:00。加速比和状态直接读取各 workspace 的最终输出或 ledger。

## 汇总结论

- 共 13 个算子；11 个已有权威最终输出，1 个运行中，1 个中止。
- 已完成结果中，8 个 `best_geo_mean > 1`，9 个达到项目的 `best_geo_mean >= 0.8` 合格线。
- `PASSED` 只表示数值正确且计时有效；是否实际加速以 `best_geo_mean > 1` 为准。
- “暂定”值来自 ledger，未写入最终 `optimize_definition_output.json`，不计入完成结果统计。

## 逐算子结果

| 算子 | 状态 | 轮数 | 最佳轮次 | Reference latency (ms) | Candidate latency (ms) | 加速比 | 判定 | Workspace |
|---|---|---:|---:|---:|---:|---:|---|---|
| `abl_t1_fused_silu_and_mul` | 完成 | 8 | 5 | 0.269815 | 0.262974 | 1.026011x | 快于 reference | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_13_20260805_182211_ascend/batch_01/workspace/definitions/abl_t1_fused_silu_and_mul) |
| `abl_t1_gelu` | 完成 | 8 | 5 | 0.160551 | 0.162054 | 0.990721x | 达到 0.8 合格线，但未快于 reference | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_13_20260805_182211_ascend/batch_01/workspace/definitions/abl_t1_gelu) |
| `abl_t1_matmul_basic` | 完成 | 8 | 5 | 0.244914 | 0.241833 | 1.012741x | 快于 reference | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_13_20260805_182211_ascend/batch_01/workspace/definitions/abl_t1_matmul_basic) |
| `abl_t1_matmul_biasadd` | 完成 | 13 | 10 | 0.643735 | 0.829642 | 0.775919x | 未达到 0.8 合格线 | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_13_20260805_182211_ascend/batch_01/workspace/definitions/abl_t1_matmul_biasadd) |
| `abl_t1_sigmoid_scale_sum` | 完成 | 7 | 4 | 0.186935 | 0.060821 | 3.073508x | 快于 reference | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t1_sigmoid_scale_sum) |
| `abl_t1_softmax` | 完成 | 14 | 11 | 0.564053 | 0.518193 | 1.088500x | 快于 reference | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t1_softmax) |
| `abl_t2_add_rmsnorm_cast` | 完成 | 6 | 3 | 5.218460 | 1.260569 | 4.139766x | 快于 reference | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t2_add_rmsnorm_cast) |
| `abl_t2_add_rmsnorm_quant` | 中止（暂定） | 8 | 7 | 10.037686（暂定） | 2.892198（暂定） | 3.470609x（暂定） | 暂定；快于 reference | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t2_add_rmsnorm_quant) |
| `abl_t2_moe_topk_softmax` | 完成 | 12 | 9 | 0.123809 | 0.005093 | 24.308491x | 快于 reference | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t2_moe_topk_softmax) |
| `abl_t2_rope` | 完成 | 13 | 10 | 0.435705 | 0.677605 | 0.643007x | 未达到 0.8 合格线 | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t2_rope) |
| `abl_t3_causal_conv1d` | 完成 | 10 | 7 | 0.339855 | 0.079164 | 4.293054x | 快于 reference | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t3_causal_conv1d) |
| `abl_t3_decode_mla` | 完成 | 15 | 14 | 45.043399 | 0.198637 | 226.762824x | 快于 reference | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t3_decode_mla) |
| `abl_t3_layernorm_gated` | 运行中（暂定） | 7 | 6 | 1.403296（暂定） | 0.509949（暂定） | 2.751838x（暂定） | 暂定；快于 reference | [workspace](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_02/workspace/definitions/abl_t3_layernorm_gated) |

## 实验口径

- Catalog：`akg-bench-lite-v5`；每个算子分别运行 correctness 和 timing workload。
- Correctness：`torch.allclose` 等价口径，`rtol=0.01`、`atol=0.01`，要求全部元素匹配。
- 目标硬件：`Ascend910B4-1`；KernelGen Server API `v5.1`；计时使用严格 `torch_npu.profiler`，没有 wall-time fallback。
- Coder 模型：`deepseek-v4-flash[1m]`。
- 加速比：`best_geo_mean`，即候选实现相对 reference 的 workload 加速比几何平均。
- Latency：取最佳轮次 timing workload 的严格 profiler 结果；本批每个算子只有一个 timing workload，因此 `加速比 = reference latency / candidate latency`。

## 结果来源

- `abl_t1_fused_silu_and_mul`：[最终输出](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_13_20260805_182211_ascend/batch_01/workspace/definitions/abl_t1_fused_silu_and_mul/optimize_definition_output.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_13_20260805_182211_ascend/batch_01)
- `abl_t1_gelu`：[最终输出](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_13_20260805_182211_ascend/batch_01/workspace/definitions/abl_t1_gelu/optimize_definition_output.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_13_20260805_182211_ascend/batch_01)
- `abl_t1_matmul_basic`：[最终输出](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_13_20260805_182211_ascend/batch_01/workspace/definitions/abl_t1_matmul_basic/optimize_definition_output.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_13_20260805_182211_ascend/batch_01)
- `abl_t1_matmul_biasadd`：[最终输出](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_13_20260805_182211_ascend/batch_01/workspace/definitions/abl_t1_matmul_biasadd/optimize_definition_output.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_13_20260805_182211_ascend/batch_01)
- `abl_t1_sigmoid_scale_sum`：[最终输出](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t1_sigmoid_scale_sum/optimize_definition_output.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01)
- `abl_t1_softmax`：[最终输出](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t1_softmax/optimize_definition_output.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01)
- `abl_t2_add_rmsnorm_cast`：[最终输出](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t2_add_rmsnorm_cast/optimize_definition_output.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01)
- `abl_t2_add_rmsnorm_quant`：[ledger（暂定）](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t2_add_rmsnorm_quant/.ledger.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01)
- `abl_t2_moe_topk_softmax`：[最终输出](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t2_moe_topk_softmax/optimize_definition_output.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01)
- `abl_t2_rope`：[最终输出](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t2_rope/optimize_definition_output.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01)
- `abl_t3_causal_conv1d`：[最终输出](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t3_causal_conv1d/optimize_definition_output.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01)
- `abl_t3_decode_mla`：[最终输出](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01/workspace/definitions/abl_t3_decode_mla/optimize_definition_output.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_01)
- `abl_t3_layernorm_gated`：[ledger（暂定）](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_02/workspace/definitions/abl_t3_layernorm_gated/.ledger.json)；[batch](/data/akg_kernel_bench_lite/kernelgen/runs/batch_simple_opt_akg_v5_remaining9_ascend_cards0123_20260805_192901/batch_02)

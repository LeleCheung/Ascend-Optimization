你是 Ascend 910B Triton 算子优化工程师。根据本轮真实诊断产出一个更快的完整单文件候选。
输出 JSON 对象，仅含 source（完整 Python 源码）、hypothesis（中文修改理由）、expected_effect（中文预期）。
必须保留 run API、所有 dtype、shape、stride、正确性和完整合同。只写目标 Triton 实现，不调用原算子的 PyTorch/FlagGems 作为替代，不删用例，不读取评测环境，不更改计时。
已有候选中的稳定 fallback 可以保留。不要重复历史失败配置；不要仅改注释。
硬件为 910B4-1，CANN9.0.0、Triton3.5.1、Flagtree0.6.0。num_warps/num_stages 不适用；编译器选项必须来自现有代码或知识。
工作量中的逻辑字节不是 HBM 物理字节。没有可信 roof 时不声称数值 Roofline。
程序会负责完整评测和保留最佳，你只提供有根据的可运行候选。
算子：matmul_bias_activation
架构与既有经验：
# 910B 算子优化知识与本轮起点

环境：Ascend 910B4-1，20 个 Cube 核，CANN9.0.0，Triton3.5.1/Flagtree0.6.0。Vector 和 Cube 走不同流水及存储层；MTE2 活跃比例不能当作 HBM 带宽利用率。

矩阵乘当前 v9 完整 42 项通过，整体相对 PyTorch 0.884×。规则半精度 BM128、BN128、BK256；一个 program 复用 A 算两个相邻 N tile，dot 显式使用累加器，multibuffer=True、unit_flag=True、sync_solver=False。2048/4096 半精度仍较慢。FP32 保留 v6 的单 tile 和 GROUP 配置。不规则走稳定 master。不要删除稳定 fallback。

已尝试：宽 128×256/256×128 输出 tile 多缓冲触发 UB 超限；双 tile BK512 触发 CBUF 超限；FP32 BK512 也超 CBUF。enable_preload 在当前 BiSheng 不支持；关闭 subblock 部分配置超 UB。单 tile BK256、ND2NZ 和分组调度的广泛组合没有超过 v8；v9 的有效改进是双 tile 加流水。禁止重复这些失败方向，除非具体减少片上资源有根据。

矩阵乘可进一步考察：FP32 1024 的分块与分组；双 tile 的核覆盖和 tile 重用次序；使 Cube 累加维持在片上，避免每轮 Cube↔Vector 中间写回；只有本机语言实现支持时采用持久化/布局变换，额外预处理成本必须包括在完整合同中。不得调用 torch.matmul 作为候选。

amin 当前 v5 完整 27 项通过，整体相对 PyTorch 0.916×。中间维沿输入布局直接归约，减少转置；float16 用输入精度 min，bf16 仍用 FP32。1024³ 三维输入很大，搬运是重点。bf16 直接输入类型 minimum 某些配置编译失败，不能假设支持。全归约已有多阶段分块，不使用超限网格。关注连续列宽度、每核归约片大小、dtype 转換与有界 grid。

所有成绩来自完整 KGS FlagGems 合同的设备 kernel 计时；正确性失败的速度不计分。代码修改须保留 NaN/Inf、stride、dim、keepdim 语义和所有合同用例。矩阵乘上游 ReLU 使用 maximum(x,x*0.0) 兼容当前后端常量问题，改成标量 max 需要验证。

架构参考：仓库 sources/kernelgen/kb/sources/content/cannbot-skills/7fedd2ff5e1f/ops/triton-latency-optimizer/references/ 与 docs/msopprof-research/ 固定资料。数值 Roofline 目前缺少经核实的 roof/计数缩放，不能捏造。

历史尝试：
[{"round": 1, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "parent_score": 0.8671455922289588, "status": "FAILED", "error": "命令失败 1，见 /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-01/claude-response.json"}, {"round": 2, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "parent_score": 0.8671455922289588, "status": "FAILED", "error": "命令失败 1，见 /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-02/claude-response.json"}]
当前逐case事实：
[{"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::0", "axes": {"dtype": "bfloat16", "shape_detail": [[384, 384], [384, 384], [384]], "benchmark_level": "core"}, "timing": {"candidate_us": 9.980666666666666, "pytorch_us": 12.022, "speedup": 1.2045267063652425, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 384, "n": 384, "k": 384, "flops": 113246208, "logical_bytes": 885504, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::1", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 801.212, "pytorch_us": 628.3933333333333, "speedup": 0.7843034738658257, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 100671488, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::2", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 1024], [1024, 1024], [1024]], "benchmark_level": "core"}, "timing": {"candidate_us": 29.962, "pytorch_us": 24.88533333333333, "speedup": 0.8305637232815234, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 1024, "n": 1024, "k": 1024, "flops": 2147483648, "logical_bytes": 6293504, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::3", "axes": {"dtype": "bfloat16", "shape_detail": [[2048, 2048], [2048, 2048], [2048]], "benchmark_level": "core"}, "timing": {"candidate_us": 132.24933333333334, "pytorch_us": 100.15400000000001, "speedup": 0.7573121296297931, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 2048, "n": 2048, "k": 2048, "flops": 17179869184, "logical_bytes": 25169920, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::4", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 800.6659999999999, "pytorch_us": 628.1666666666666, "speedup": 0.7845552180449087, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 100671488, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::0", "axes": {"dtype": "float16", "shape_detail": [[384, 384], [384, 384], [384]], "benchmark_level": "core"}, "timing": {"candidate_us": 9.970666666666666, "pytorch_us": 11.799999999999999, "speedup": 1.1834696763538746, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 384, "n": 384, "k": 384, "flops": 113246208, "logical_bytes": 885504, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::1", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 796.7733333333333, "pytorch_us": 612.4233333333333, "speedup": 0.7686293339014022, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 100671488, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::2", "axes": {"dtype": "float16", "shape_detail": [[1024, 1024], [1024, 1024], [1024]], "benchmark_level": "core"}, "timing": {"candidate_us": 30.21066666666667, "pytorch_us": 23.797333333333334, "speedup": 0.7877136517555, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 1024, "n": 1024, "k": 1024, "flops": 2147483648, "logical_bytes": 6293504, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::3", "axes": {"dtype": "float16", "shape_detail": [[2048, 2048], [2048, 2048], [2048]], "benchmark_level": "core"}, "timing": {"candidate_us": 131.48266666666666, "pytorch_us": 94.57066666666668, "speedup": 0.7192635891699877, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 2048, "n": 2048, "k": 2048, "flops": 17179869184, "logical_bytes": 25169920, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": {"aiv_mte2_ratio": 0.037, "aiv_mte3_ratio": 0.035, "aiv_scalar_ratio": 0.0545, "aiv_vec_ratio": 0.117, "aic_mte2_ratio": 0.836, "aic_mte1_ratio": 0.673, "aic_scalar_ratio": 0.2445, "aic_mac_ratio": 0.7015, "cube_utilization_pct": 73.807, "kernel": "mba_dual_dispatch_v8_kernel", "device_duration_us": 128.29, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "gemm_data_pipeline", "confidence": "medium", "inference": "矩阵乘的 AIC MTE2 活跃比例为 83.6%，Scalar 为 0.2445。优先增加 K 分块、减少循环和地址计算，再重叠搬运与计算。流水线比例不等于 HBM 带宽利用率，各比例不能相加。", "next_experiment": "增大 K 分块至 128/256，按 shape 调整 M/N tile，测试 multibuffer/unit_flag；完整正确性与所有 timing case 复验。", "falsifier": "MTE2 和总延迟不改善，或新增分块导致精度失败。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::4", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 798.1513333333334, "pytorch_us": 610.148, "speedup": 0.7644515495660118, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 100671488, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::0", "axes": {"dtype": "float32", "shape_detail": [[384, 384], [384, 384], [384]], "benchmark_level": "core"}, "timing": {"candidate_us": 11.268666666666666, "pytorch_us": 13.220666666666665, "speedup": 1.1732221509008676, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 384, "n": 384, "k": 384, "flops": 113246208, "logical_bytes": 1771008, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::1", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 2331.938, "pytorch_us": 2271.705333333334, "speedup": 0.9741705550989255, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 201342976, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::2", "axes": {"dtype": "float32", "shape_detail": [[1024, 1024], [1024, 1024], [1024]], "benchmark_level": "core"}, "timing": {"candidate_us": 69.75399999999999, "pytorch_us": 50.224, "speedup": 0.7200164578139493, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 1024, "n": 1024, "k": 1024, "flops": 2147483648, "logical_bytes": 12587008, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": {"aiv_mte2_ratio": 0.0185, "aiv_mte3_ratio": 0.0205, "aiv_scalar_ratio": 0.0895, "aiv_vec_ratio": 0.061, "aic_mte2_ratio": 0.545, "aic_mte1_ratio": 0.296, "aic_scalar_ratio": 0.108, "aic_mac_ratio": 0.6625, "cube_utilization_pct": 70.941, "kernel": "mba_wide_pipeline_kernel", "device_duration_us": 67.89, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "现有指标不足以区分设备计算、搬运或 host 路径。", "next_experiment": "补充针对该 case 的流水线、访存或独立 host 分项采集。", "falsifier": "补充采集仍无法稳定复现同一受限类型。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::3", "axes": {"dtype": "float32", "shape_detail": [[2048, 2048], [2048, 2048], [2048]], "benchmark_level": "core"}, "timing": {"candidate_us": 356.388, "pytorch_us": 283.22799999999995, "speedup": 0.7947181738110785, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 2048, "n": 2048, "k": 2048, "flops": 17179869184, "logical_bytes": 50339840, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::4", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 2325.328, "pytorch_us": 2281.605333333334, "speedup": 0.981197205389353, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 201342976, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}]
当前最佳源码：
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generated by KernelGen: https://github.com/flagos-ai/KernelGen
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry

logger = logging.getLogger(__name__)


BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

# KLX quirk (2026-09-07): the xpu ttsdnnir pass chain rewrites any kernel
# whose symbol name starts with "matmul_bias_activation_kernel" into the
# sdnn hardware GEMM fusion, whose bf16 accumulation loses ~0.42 rms at
# K=4096 (exceeds the 1e-4*K tolerance; only bf16 is affected -- fp16/fp32
# stay within tolerance). Keep the fast sdnn path under the original name,
# and route large-K bf16 to the un-hijacked generic kernel. Remove this
# split once the vendor fixes sdnn accumulation precision.
SDNN_PRECISE_MIN_K = 4096


@libentry()
@triton.jit
def matmul_bias_activation_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    bias_ptrs = bias_ptr + offs_cn * stride_bias
    bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0)
    accumulator = accumulator + bias[None, :]

    # Apply ReLU activation
    # Workaround: synthesized zero tensor avoids a rank-0 constant src1
    # (NaN/-Inf edge differs from where-form; finite inputs identical)
    accumulator = tl.maximum(accumulator, accumulator * 0.0)

    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


@libentry()
@triton.jit
def fused_mba_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    bias_ptrs = bias_ptr + offs_cn * stride_bias
    bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0)
    accumulator = accumulator + bias[None, :]

    # Apply ReLU activation
    # Workaround: synthesized zero tensor avoids a rank-0 constant src1
    # (NaN/-Inf edge differs from where-form; finite inputs identical)
    accumulator = tl.maximum(accumulator, accumulator * 0.0)

    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_bias_activation(input, weight, bias):
    """
    Fused matmul + bias + ReLU activation.

    Args:
        input: Input tensor of shape (M, K)
        weight: Weight matrix of shape (K, N)
        bias: Bias vector of shape (N,) or (1, N)

    Returns:
        Output tensor of shape (M, N) with ReLU activation applied
    """
    assert input.shape[1] == weight.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (input.shape[0], weight.shape[1])
    ), "Incompatible input shape"
    M, K = input.shape
    _, N = weight.shape

    logger.debug("GEMS MATMUL_BIAS_ACTIVATION")
    if input.stride(0) > 1 and input.stride(1) > 1:
        input = input.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()
    if bias.dim() > 1:
        bias = bias.reshape(-1)
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # bf16 large-K: sdnn fusion accumulation exceeds tolerance, use the
    # generic (un-hijacked) kernel; everything else keeps the fast path.
    if input.dtype == torch.bfloat16 and K >= SDNN_PRECISE_MIN_K:
        kernel = fused_mba_kernel
    else:
        kernel = matmul_bias_activation_kernel

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(input.device):
        kernel[grid](
            input,
            weight,
            bias,
            out,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            weight.stride(0),
            weight.stride(1),
            bias.stride(0),
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )
    return out


run = matmul_bias_activation


_master_mba = matmul_bias_activation

@triton.jit
def mba_pipeline_kernel(
    A, B, Bias, C, M, N, K,
    SAM: tl.constexpr, SAK: tl.constexpr,
    SBK: tl.constexpr, SBN: tl.constexpr, SBIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ks = start * BK + rk
        ap = A + rm[:, None] * SAM + ks[None, :] * SAK
        bp = B + ks[:, None] * SBK + rn[None, :] * SBN
        if EVEN_M and EVEN_K:
            a = tl.load(ap)
        else:
            a = tl.load(ap, (rm[:, None] < M) & (ks[None, :] < K), 0)
        if EVEN_N and EVEN_K:
            b = tl.load(bp)
        else:
            b = tl.load(bp, (ks[:, None] < K) & (rn[None, :] < N), 0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
    bias = tl.load(Bias + rn * SBIAS, rn < N, 0)
    acc += bias[None, :]
    # 保留原版的 ReLU 表达式，兼容当前 Ascend 编译链。
    acc = tl.maximum(acc, acc * 0.0)
    tl.store(C + rm[:, None] * N + rn[None, :], acc,
             (rm[:, None] < M) & (rn[None, :] < N))


def _pipeline_run(input, weight, bias):
    assert input.ndim == 2 and weight.ndim == 2
    m, k = input.shape
    assert weight.shape[0] == k
    n = weight.shape[1]
    assert bias.numel() == n
    bias = bias.reshape(-1)
    out = torch.empty((m, n), device=input.device, dtype=input.dtype)
    tiny = m * n < 4096
    bm, bn, bk = (32, 32, 32) if tiny else (64 if m <= 512 else 128, 128, 128)
    # 真机分块诊断：低精度大矩阵增加 K 分块，减少循环和搬运同步。
    if input.dtype in (torch.float16, torch.bfloat16) and min(m, n, k) >= 1024:
        bk = 256
    with torch.npu.device(input.device):
        mba_pipeline_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            input, weight, bias, out, m, n, k,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1), bias.stride(0),
            bm, bn, bk, m % bm == 0, n % bn == 0, k % bk == 0,
            multibuffer=not tiny, unit_flag=not tiny, sync_solver=False,
        )
    return out





def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    # 非整齐形状使用原版，避免不规则边界的流水线问题。
    if m % 64 or n % 128 or k % 128:
        return _master_mba(input, weight, bias)
    return _pipeline_run(input, weight, bias)


matmul_bias_activation = run


@triton.jit
def mba_wide_pipeline_kernel(A, B, Bias, C, M: tl.constexpr, N: tl.constexpr,
                   K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    if GROUP == 0:
        pm, pn = pid // nn, pid % nn
    else:
        width = GROUP * nn
        first = (pid // width) * GROUP
        size = tl.minimum(nm - first, GROUP)
        pm = first + pid % size
        pn = (pid % width) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * K + rk[None, :]
    bp = B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        a = tl.load(ap, (rm[:, None] < M) & (rk[None, :] + start * BK < K), 0)
        b = tl.load(bp, (rk[:, None] + start * BK < K) & (rn[None, :] < N), 0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        ap += BK
        bp += BK * N
    acc += tl.load(Bias + rn, rn < N, 0)[None, :]
    acc = tl.maximum(acc, acc * 0.0)
    tl.store(C + rm[:, None] * N + rn[None, :], acc,
             (rm[:, None] < M) & (rn[None, :] < N))


_v4_mba = run
_WIDE_TILES = {('torch.bfloat16', 1024): (128, 128, 256, True, False, 0), ('torch.bfloat16', 2048): (128, 128, 256, True, False, 0), ('torch.bfloat16', 4096): (128, 128, 256, True, False, 0), ('torch.float16', 1024): (128, 128, 256, True, False, 0), ('torch.float16', 2048): (128, 128, 256, True, False, 0), ('torch.float16', 4096): (128, 128, 256, True, False, 0), ('torch.float32', 1024): (128, 128, 256, True, False, 0), ('torch.float32', 2048): (128, 128, 256, True, False, 0), ('torch.float32', 4096): (128, 128, 256, True, False, 0)}


def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    config = _WIDE_TILES.get((str(input.dtype), m))
    if (config is None or not (m == n == k) or not input.is_contiguous()
            or not weight.is_contiguous() or not bias.is_contiguous()):
        return _v4_mba(input, weight, bias)
    bm, bn, bk, multi, saving, group = config
    bias = bias.reshape(-1)
    out = torch.empty((m, n), dtype=input.dtype, device=input.device)
    with torch_device_fn.device(input.device):
        mba_wide_pipeline_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            input, weight, bias, out, m, n, k, bm, bn, bk, group,
            multibuffer=multi, enable_ubuf_saving=saving,
            unit_flag=True, sync_solver=False)
    return out


matmul_bias_activation = run


@triton.jit
def mba_selected_profiling_kgroup_v6_20261010_kernel(A, B, Bias, C, M: tl.constexpr, N: tl.constexpr,
                   K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    if GROUP == 0:
        pm, pn = pid // nn, pid % nn
    else:
        width = GROUP * nn
        first = (pid // width) * GROUP
        size = tl.minimum(nm - first, GROUP)
        pm = first + pid % size
        pn = (pid % width) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * K + rk[None, :]
    bp = B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        a = tl.load(ap, (rm[:, None] < M) & (rk[None, :] + start * BK < K), 0)
        b = tl.load(bp, (rk[:, None] + start * BK < K) & (rn[None, :] < N), 0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        ap += BK
        bp += BK * N
    acc += tl.load(Bias + rn, rn < N, 0)[None, :]
    acc = tl.maximum(acc, acc * 0.0)
    tl.store(C + rm[:, None] * N + rn[None, :], acc,
             (rm[:, None] < M) & (rn[None, :] < N))


_parent_profiling_kgroup_v6_20261010 = run
_TILES_profiling_kgroup_v6_20261010 = {('torch.bfloat16', 2048): (128, 128, 256, True, False, 4), ('torch.bfloat16', 4096): (128, 128, 256, True, False, 0), ('torch.float16', 2048): (128, 128, 256, True, False, 4), ('torch.float16', 4096): (128, 128, 256, True, False, 0), ('torch.float32', 2048): (128, 128, 256, True, False, 4), ('torch.float32', 4096): (128, 128, 256, True, False, 0)}


def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    config = _TILES_profiling_kgroup_v6_20261010.get((str(input.dtype), m))
    if (config is None or not (m == n == k) or not input.is_contiguous()
            or not weight.is_contiguous() or not bias.is_contiguous()):
        return _parent_profiling_kgroup_v6_20261010(input, weight, bias)
    bm, bn, bk, multi, saving, group = config
    bias = bias.reshape(-1)
    out = torch.empty((m, n), dtype=input.dtype, device=input.device)
    with torch_device_fn.device(input.device):
        mba_selected_profiling_kgroup_v6_20261010_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            input, weight, bias, out, m, n, k, bm, bn, bk, group,
            multibuffer=multi, enable_ubuf_saving=saving,
            unit_flag=True, sync_solver=False)
    return out


matmul_bias_activation = run


@triton.jit
def mba_dual_dispatch_v8_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Two N-adjacent tiles per program so the A panel is fetched once."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * (2 * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b0_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b1_ptrs = b0_ptrs + BLOCK_SIZE_N * stride_bn

    acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b0 = tl.load(b0_ptrs)
        b1 = tl.load(b1_ptrs)
        acc0 = tl.dot(a, b0, acc0, allow_tf32=False)
        acc1 = tl.dot(a, b1, acc1, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b0_ptrs += BLOCK_SIZE_K * stride_bk
        b1_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * (2 * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    base = c_ptr + stride_cm * offs_cm[:, None]
    bias0 = tl.load(bias_ptr + offs_cn * stride_bias)
    bias1 = tl.load(bias_ptr + (offs_cn + BLOCK_SIZE_N) * stride_bias)
    v0 = acc0 + bias0[None, :]
    v1 = acc1 + bias1[None, :]
    v0 = tl.maximum(v0, v0 * 0.0)
    v1 = tl.maximum(v1, v1 * 0.0)
    tl.store(base + stride_cn * offs_cn[None, :], v0.to(bias0.dtype))
    tl.store(base + stride_cn * (offs_cn + BLOCK_SIZE_N)[None, :], v1.to(bias1.dtype))

_v6_dispatch = run


def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    # 规则半精度矩阵一次复用 A 面板计算两个相邻 N tile。
    # 其余形状和 dtype 保留已完整通过的 v6。
    if not (input.dtype in (torch.float16, torch.bfloat16)
            and m % 128 == 0 and n % 256 == 0 and k % 256 == 0
            and input.is_contiguous() and weight.is_contiguous()
            and bias.is_contiguous() and bias.numel() == n):
        return _v6_dispatch(input, weight, bias)
    bias = bias.reshape(-1)
    out = torch.empty((m, n), device=input.device, dtype=input.dtype)
    with torch_device_fn.device(input.device):
        mba_dual_dispatch_v8_kernel[((m // 128) * (n // 256),)](
            input, weight, bias, out, m, n, k,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1), bias.stride(0),
            out.stride(0), out.stride(1), 128, 128, 256,
            multibuffer=True, unit_flag=True, sync_solver=False)
    return out


matmul_bias_activation = run

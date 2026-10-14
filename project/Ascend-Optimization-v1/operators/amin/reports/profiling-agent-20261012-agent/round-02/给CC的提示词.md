你是 Ascend 910B Triton 算子优化工程师。根据本轮真实诊断产出一个更快的完整单文件候选。
输出 JSON 对象，仅含 source（完整 Python 源码）、hypothesis（中文修改理由）、expected_effect（中文预期）。
必须保留 run API、所有 dtype、shape、stride、正确性和完整合同。只写目标 Triton 实现，不调用原算子的 PyTorch/FlagGems 作为替代，不删用例，不读取评测环境，不更改计时。
已有候选中的稳定 fallback 可以保留。不要重复历史失败配置；不要仅改注释。
认真读当前源码，历史已采纳改动已经在当前源码中，不要把它重复当成新优化。必须具体说明修改了哪个 kernel、分块或调用参数，以及哪些慢 case 会因此改善。
硬件为 910B4-1，CANN9.0.0、Triton3.5.1、Flagtree0.6.0。num_warps/num_stages 不适用；编译器选项必须来自现有代码或知识。
工作量中的逻辑字节不是 HBM 物理字节。没有可信 roof 时不声称数值 Roofline。
只能把 PipeUtilization 当作相应流水活跃比例；不将 tile 逻辑字节除以全kernel时间解释为实测L1/HBM带宽，也不把片上容量估算当作已编译证明。容量与性能推测必须由完整测试验证。
程序会负责完整评测和保留最佳，你只提供有根据的可运行候选。
算子：amin
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
[{"round": 1, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "parent_score": 0.8997956723470408, "hypothesis": "修改点集中在两个 amin 设备内核 amin_native_middle_kernel / amin_native_last_kernel 以及驱动它们的 run 与 _NATIVE_TILES 配置表，没有改动任何其它 kernel 和 fallback 路径。\n\n1) 让 bfloat16 也走「输入精度 min」（NATIVE=True）。当前 bf16 三个形状走 NATIVE=False，即每次 tl.load 后都做 values.to(tl.float32)，累加器是 (BR,BC)/(BO,BR) 的 FP32 tile，再在 FP32 上做归约。与同 shape、同布局、同代码路径、仅 dtype 不同的 float16 对比（本轮可信诊断给出的同一合同计时）：(4096,4096) fp16 39.29us vs bf16 60.67us；(64,512,512) fp16 29.21us vs bf16 61.31us；(1024,1024,1024) fp16 1669.6us vs bf16 3658.8us。bf16 慢 1.5～2.2 倍，而两者唯一的差别就是「逐元素 fp32 转换 + FP32 累加器 + FP32 归约」这一组额外 Vector 指令。bf16 (4096,4096) 的真机指标显示 aiv_vec_ratio=0.795（MTE2 仅 0.1935），说明瓶颈在 Vector 流水线而不是搬运；而 min 选择操作在任何能表示原值的 dtype 上都是精确的，bf16 原生 min 不会改变结果，NaN/Inf、mask 用 inf 的语义都不变。\n\n2) BR 对齐 fp16。原生累加器只有 FP32 累加器一半的 UB 占用，因此 bf16 可以把归约片放大到与 fp16 相同：(64,512,512) BR 16→32（内层循环 32 次→16 次）、(1024,1024,1024) BR 8→16（64 次→32 次），(4096,4096) 保持 BO=8/BR=1024 只翻转 NATIVE。这直接针对这三个 bf16 慢 case，同时也减少每轮的 mask/地址生成开销。\n\n3) 编译失败安全网。知识中明确记载「bf16 直接输入类型 minimum 某些配置编译失败，不能假设支持」，所以新增 _BF16_NATIVE_FALLBACK：一旦原生路径抛异常，就把该 shape 记入 _native_bf16_disabled 并永久退回「原生配置未启用时」的原始分块（含 BR 一并还原，避免 FP32 累加器在放大后的 BR 下超片上容量），并立即用回退配置重跑同一 out。失败时候选与当前版本行为一致，不会因为编译问题丢掉 fallback；异常只对 bf16 原生路径吞掉，其余异常照常抛出。第一次调用之后不再走 try 分支里的失败路径，稳定 fallback 保持不变。\n\n未采用的方向：不动全归约（amin_global_chunk_kernel）与 v4 的路径判定，不动 fp16/fp32 的分块（它们已由设备计时选定），不做任何未经验证的 Roofline 推断，也不重复历史上已被否决的配置。", "expected_effect": "主要改善三个此前明显落后 PyTorch 的 bfloat16 用例：test_amin[dtype2] 的 (4096,4096) dim=1（当前 60.67us / 0.46×）、(64,512,512) dim=1（61.31us / 0.55×）、(1024,1024,1024) dim=1（3658.76us / 0.39×）。去掉逐元素 FP32 转换与 FP32 累加后，这三个 case 的 Vector 工作量按同 shape 的 fp16 对照大约降到原来的 2/3～1/2，目标是把 bf16 拉回到与 fp16 同一量级（约 39us / 29us / 1670us 附近），使 bf16 三个 case 从 0.39～0.55× 提升到接近或超过 1.0×。若 bf16 原生 min 在本机编译失败，则自动退回现有已验证配置，指标与当前版本持平，不会回退。\n\nfloat16 / float32 的 9 个用例配置保持不变，预期与当前逐 case 事实一致（其中 (1024,1024,1024) 的 fp32 3302.75us×(4.3GB 已计入逻辑字节) 与 fp16 1669.6us 路径不动，避免无依据的分块改动）。全归约与 (64,64) 两个已领先的用例（1.3～2.1×）沿用原路径，不受影响。所有改动只涉及 dtype 归约精度与归约片大小，输出 dtype、keepdim、dim 归一化、mask 用 +inf 的语义完全一致，NaN/Inf 行为与 float16 现有路径相同，不修改计时与用例。以上为基于同合同对照计时的预期，最终以完整测试的实测结果为准。", "source_sha256": "eca4ab44bbe16ae1095ea55deded8f676a97e2e83237e633f5c5f8978e8ede3d", "status": "PASSED", "score": 0.9229204558479288, "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "accepted": true}]
当前逐case事实：
[{"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::0", "axes": {"dtype": "float16", "shape_detail": [[1048576]], "benchmark_level": "core"}, "timing": {"candidate_us": 6.08, "pytorch_us": 8.468, "speedup": 1.3927566980806234, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1048576, "output_elements": 1, "reduction_axis": null, "reduction_length": 1048576, "minimum_comparisons": 1048575, "flops": null, "logical_bytes": 2097154, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::1", "axes": {"dtype": "float16", "shape_detail": [[64, 64], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 1.6713333333333333, "pytorch_us": 3.6146666666666665, "speedup": 2.1626747502143866, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 4096, "output_elements": 64, "reduction_axis": 1, "reduction_length": 64, "minimum_comparisons": 4032, "flops": null, "logical_bytes": 8320, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::2", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 39.124, "pytorch_us": 22.986666666666668, "speedup": 0.5875347079336436, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 4096, "reduction_axis": 1, "reduction_length": 4096, "minimum_comparisons": 16773120, "flops": null, "logical_bytes": 33562624, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::3", "axes": {"dtype": "float16", "shape_detail": [[64, 512, 512], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 29.230666666666668, "pytorch_us": 17.361333333333334, "speedup": 0.5939438240301599, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 32768, "reduction_axis": 1, "reduction_length": 512, "minimum_comparisons": 16744448, "flops": null, "logical_bytes": 33619968, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::4", "axes": {"dtype": "float16", "shape_detail": [[1024, 1024, 1024], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 1668.7766666666669, "pytorch_us": 1418.5553333333332, "speedup": 0.850057036788004, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1073741824, "output_elements": 1048576, "reduction_axis": 1, "reduction_length": 1024, "minimum_comparisons": 1072693248, "flops": null, "logical_bytes": 2149580800, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::0", "axes": {"dtype": "float32", "shape_detail": [[1048576]], "benchmark_level": "core"}, "timing": {"candidate_us": 5.946666666666667, "pytorch_us": 8.982000000000001, "speedup": 1.510417425715935, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1048576, "output_elements": 1, "reduction_axis": null, "reduction_length": 1048576, "minimum_comparisons": 1048575, "flops": null, "logical_bytes": 4194308, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::1", "axes": {"dtype": "float32", "shape_detail": [[64, 64], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 1.73, "pytorch_us": 3.6159999999999997, "speedup": 2.0901103982428757, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 4096, "output_elements": 64, "reduction_axis": 1, "reduction_length": 64, "minimum_comparisons": 4032, "flops": null, "logical_bytes": 16640, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::2", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 53.501333333333335, "pytorch_us": 37.51866666666667, "speedup": 0.7012665704283312, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 4096, "reduction_axis": 1, "reduction_length": 4096, "minimum_comparisons": 16773120, "flops": null, "logical_bytes": 67125248, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::3", "axes": {"dtype": "float32", "shape_detail": [[64, 512, 512], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 53.14333333333333, "pytorch_us": 30.636, "speedup": 0.5764795023279765, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 32768, "reduction_axis": 1, "reduction_length": 512, "minimum_comparisons": 16744448, "flops": null, "logical_bytes": 67239936, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::4", "axes": {"dtype": "float32", "shape_detail": [[1024, 1024, 1024], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 3304.0226666666667, "pytorch_us": 2827.218, "speedup": 0.8556896546001402, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1073741824, "output_elements": 1048576, "reduction_axis": 1, "reduction_length": 1024, "minimum_comparisons": 1072693248, "flops": null, "logical_bytes": 4299161600, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::0", "axes": {"dtype": "bfloat16", "shape_detail": [[1048576]], "benchmark_level": "core"}, "timing": {"candidate_us": 6.196666666666667, "pytorch_us": 8.393333333333334, "speedup": 1.3544859415931962, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1048576, "output_elements": 1, "reduction_axis": null, "reduction_length": 1048576, "minimum_comparisons": 1048575, "flops": null, "logical_bytes": 2097154, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::1", "axes": {"dtype": "bfloat16", "shape_detail": [[64, 64], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 1.6666666666666667, "pytorch_us": 3.7753333333333337, "speedup": 2.2651240925544465, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 4096, "output_elements": 64, "reduction_axis": 1, "reduction_length": 64, "minimum_comparisons": 4032, "flops": null, "logical_bytes": 8320, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::2", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 60.437333333333335, "pytorch_us": 28.02733333333333, "speedup": 0.4637429451835313, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 4096, "reduction_axis": 1, "reduction_length": 4096, "minimum_comparisons": 16773120, "flops": null, "logical_bytes": 33562624, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": {"aiv_mte2_ratio": 0.204, "aiv_mte3_ratio": 0.0345, "aiv_scalar_ratio": 0.121, "aiv_vec_ratio": 0.781, "aic_mte2_ratio": 0.0, "aic_mte1_ratio": 0.0, "aic_scalar_ratio": 0.0, "aic_mac_ratio": 0.0, "cube_utilization_pct": 0.0, "kernel": "amin_native_last_kernel", "device_duration_us": 64.26, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "reduction_vector_pipeline", "confidence": "medium", "inference": "最小值归约的 Vector 活跃比例为 78.1%。优先检查类型转换与归约指令，不能将它解释为 Cube GEMM 算力已达峰值。", "next_experiment": "固定布局与分块，比较 fp32 累积和可编译的输入精度 min，保存 IR/指标并完整复验。", "falsifier": "向量流水线变化不伴随延迟改善，或类型变化破坏 min 的结果。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::3", "axes": {"dtype": "bfloat16", "shape_detail": [[64, 512, 512], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 61.394666666666666, "pytorch_us": 33.69266666666666, "speedup": 0.5487889032887168, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 32768, "reduction_axis": 1, "reduction_length": 512, "minimum_comparisons": 16744448, "flops": null, "logical_bytes": 33619968, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::4", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 1024, 1024], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 3660.61, "pytorch_us": 1442.3866666666668, "speedup": 0.39402906271461874, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1073741824, "output_elements": 1048576, "reduction_axis": 1, "reduction_length": 1024, "minimum_comparisons": 1072693248, "flops": null, "logical_bytes": 2149580800, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": {"aiv_mte2_ratio": 0.348, "aiv_mte3_ratio": 0.001, "aiv_scalar_ratio": 0.045, "aiv_vec_ratio": 0.764, "aic_mte2_ratio": 0.0, "aic_mte1_ratio": 0.0, "aic_scalar_ratio": 0.0, "aic_mac_ratio": 0.0, "cube_utilization_pct": 0.0, "kernel": "amin_native_middle_kernel", "device_duration_us": 3667.13, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "reduction_vector_pipeline", "confidence": "medium", "inference": "最小值归约的 Vector 活跃比例为 76.4%。优先检查类型转换与归约指令，不能将它解释为 Cube GEMM 算力已达峰值。", "next_experiment": "固定布局与分块，比较 fp32 累积和可编译的输入精度 min，保存 IR/指标并完整复验。", "falsifier": "向量流水线变化不伴随延迟改善，或类型变化破坏 min 的结果。"}}]
当前最佳源码：
# Copyright 2026, The FlagOS Contributors.
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
#
# Generated by KernelGen: https://github.com/flagos-ai/KernelGen

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def amin_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=max_value)
    amin_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, amin_val)


@libentry()
@triton.jit
def amin_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    amin_val = tl.min(mid_val)
    tl.store(out, amin_val)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def amin_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)

    # Map the program id to the row of inp it should compute.
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    _all = tl.full([BLOCK_M, BLOCK_N], value=max_value, dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        a = tl.load(inp + cols, mask, other=max_value)
        _all = tl.minimum(_all, a)
    all = tl.min(_all, axis=1)[:, None]
    tl.store(out, all, row_mask)


def amin(inp, dim=None, keepdim=False):
    logger.debug("GEMS AMIN")
    assert inp.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), "amin only supports float dtypes"
    if dim is None or len(dim) == 0:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            amin_kernel_1[(mid_size, 1)](
                inp,
                mid,
                M,
                block_size,
            )
            amin_kernel_2[(1, 1)](
                mid, out, mid_size, block_mid
            )  # max block size is 128k, so mid does not requires int64 index
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            amin_kernel[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out


def amin_(inp, dim=None, keepdim=False):
    logger.debug("GEMS AMIN_")
    assert inp.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), "amin_ only supports float dtypes"
    if isinstance(dim, int):
        dim = [dim]
    if dim is None or len(dim) == 0:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            amin_kernel_1[(mid_size, 1)](
                inp,
                mid,
                M,
                block_size,
            )
            amin_kernel_2[(1, 1)](mid, out, mid_size, block_mid)
        inp.copy_(out.reshape(inp.shape) if keepdim else out)
        return inp
    else:
        result = amin(inp, dim=dim, keepdim=True)
        inp.copy_(result)
        return inp


_master_amin = amin
def run(inp, dim=None, keepdim=False):
    if isinstance(dim, int):
        dim = [dim]
    return _master_amin(inp, dim=dim, keepdim=keepdim)

# KGS 要求公开符号与 run 为同一个入口；原内核保持不变。
amin = run


# 连续单轴直接索引：outer × reduction × inner，读取相邻 inner 元素。
# 非连续或多轴输入仍使用上游通用实现，未替换为 torch.amin。
@triton.jit
def amin_direct_middle_kernel(X, Y, R: tl.constexpr, I: tl.constexpr,
                              BC: tl.constexpr, BR: tl.constexpr):
    outer = tl.program_id(0)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    reds = tl.arange(0, BR)
    acc = tl.full((BR, BC), float('inf'), tl.float32)
    for start in range(tl.cdiv(R, BR)):
        r = start * BR + reds
        values = tl.load(X + outer * R * I + r[:, None] * I + cols[None, :],
                         (r[:, None] < R) & (cols[None, :] < I), float('inf'))
        acc = tl.minimum(acc, values.to(tl.float32))
    value = tl.min(acc, axis=0)
    tl.store(Y + outer * I + cols, value, cols < I)


@triton.jit
def amin_direct_last_kernel(X, Y, O: tl.constexpr, R: tl.constexpr,
                            BO: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BO + tl.arange(0, BO)
    reds = tl.arange(0, BR)
    acc = tl.full((BO, BR), float('inf'), tl.float32)
    for start in range(tl.cdiv(R, BR)):
        r = start * BR + reds
        values = tl.load(X + rows[:, None] * R + r[None, :],
                         (rows[:, None] < O) & (r[None, :] < R), float('inf'))
        acc = tl.minimum(acc, values.to(tl.float32))
    value = tl.min(acc, axis=1)
    tl.store(Y + rows, value, rows < O)


def run(inp, dim=None, keepdim=False):
    dims = [dim] if isinstance(dim, int) else dim
    if dims is None or len(dims) != 1 or not inp.is_contiguous():
        return _master_amin(inp, dim=dims, keepdim=keepdim)
    axis = dims[0] % inp.ndim
    shape = list(inp.shape)
    reduction = shape[axis]
    outer = math.prod(shape[:axis])
    inner = math.prod(shape[axis + 1:])
    out_shape = shape[:]
    if keepdim:
        out_shape[axis] = 1
    else:
        out_shape.pop(axis)
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        if inner == 1:
            br = triton.next_power_of_2(min(reduction, 2048))
            bo = min(4, max(1, 2048 // br))
            # 大尺寸末轴采用已在真机逐值核对的多行分块。
            if outer >= 1024 and reduction >= 1024:
                bo, br = (16, 512) if inp.dtype == torch.float16 else (16, 256)
                if inp.dtype == torch.bfloat16:
                    bo, br = 32, 256
            amin_direct_last_kernel[(triton.cdiv(outer, bo),)](
                inp, out, outer, reduction, bo, br)
        else:
            bc = min(64, triton.next_power_of_2(inner))
            br = min(128, triton.next_power_of_2(reduction))
            # 32 行归约片减少 UB 压力，扩大连续列提高搬运粒度。
            if reduction >= 512 and inner >= 128:
                bc, br = (128, 32) if inp.dtype == torch.float32 else (256, 32)
            amin_direct_middle_kernel[(outer, triton.cdiv(inner, bc))](
                inp, out, reduction, inner, bc, br)
    return out


amin = run


_direct_axis_amin = run

@triton.jit
def amin_global_chunk_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    # 超大全归约也限制 coreDim；一个 program 可处理多个独立分块。
    for chunk in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = chunk * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(X + offsets, offsets < N, float('inf'))
        result = tl.min(values.to(tl.float32), axis=0)
        tl.store(Y + chunk, result)


def run(inp, dim=None, keepdim=False):
    dims = [dim] if isinstance(dim, int) else dim
    if not inp.is_contiguous() or (dims is not None and len(dims) != 0):
        return _direct_axis_amin(inp, dim=dims, keepdim=keepdim)
    count = inp.numel()
    if count == 0:
        return _master_amin(inp, dim=dims, keepdim=keepdim)
    out = torch.empty([1] * inp.ndim if keepdim else [],
                      dtype=inp.dtype, device=inp.device)
    current = inp
    # 阶段数随输入增长，避免第二阶段生成过大的单 program。
    with torch_device_fn.device(inp.device):
        while True:
            block = min(16384, triton.next_power_of_2(count))
            chunks = triton.cdiv(count, block)
            target = out if chunks == 1 else torch.empty(
                (chunks,), dtype=inp.dtype, device=inp.device)
            amin_global_chunk_kernel[(min(chunks, 65535),)](current, target, count, block)
            if chunks == 1:
                break
            current, count = target, chunks
    return out


amin = run


@triton.jit
def amin_native_middle_kernel(X, Y, R: tl.constexpr, I: tl.constexpr,
           BC: tl.constexpr, BR: tl.constexpr, NATIVE: tl.constexpr):
    outer = tl.program_id(0)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    reds = tl.arange(0, BR)
    if NATIVE:
        acc = tl.full((BR, BC), float('inf'), X.dtype.element_ty)
    else:
        acc = tl.full((BR, BC), float('inf'), tl.float32)
    for start in range(tl.cdiv(R, BR)):
        r = start * BR + reds
        values = tl.load(X + outer * R * I + r[:, None] * I + cols[None, :],
                         (r[:, None] < R) & (cols[None, :] < I), float('inf'))
        acc = tl.minimum(acc, values.to(acc.dtype))
    value = tl.min(acc, axis=0)
    tl.store(Y + outer * I + cols, value, cols < I)


@triton.jit
def amin_native_last_kernel(X, Y, O: tl.constexpr, R: tl.constexpr,
         BO: tl.constexpr, BR: tl.constexpr, NATIVE: tl.constexpr):
    rows = tl.program_id(0) * BO + tl.arange(0, BO)
    reds = tl.arange(0, BR)
    if NATIVE:
        acc = tl.full((BO, BR), float('inf'), X.dtype.element_ty)
    else:
        acc = tl.full((BO, BR), float('inf'), tl.float32)
    for start in range(tl.cdiv(R, BR)):
        r = start * BR + reds
        values = tl.load(X + rows[:, None] * R + r[None, :],
                         (rows[:, None] < O) & (r[None, :] < R), float('inf'))
        acc = tl.minimum(acc, values.to(acc.dtype))
    value = tl.min(acc, axis=1)
    tl.store(Y + rows, value, rows < O)


# 分块来自设备 kernel 计时；全归约及未覆盖形状沿用 v4。
# bfloat16 与 float16 同为 2 字节 dtype，但此前只有 float16 走输入精度归约。
# 同一 shape、同一布局下 fp16(原生 min) 明显快于 bf16(FP32 累加器 + 逐元素转换)：
#   (4096,4096)      39.3us vs 60.7us
#   (64,512,512)     29.2us vs 61.3us
#   (1024,1024,1024) 1669.6us vs 3658.8us
# 因此让 bf16 也走输入精度 min，并把归约片对齐到 fp16 的 BR
# （原生累加器只占 FP32 累加器一半 UB，才有余量放大 BR）。
_v4_amin = run
_NATIVE_TILES = {('torch.bfloat16', (64, 512, 512)): (512, 32, True), ('torch.bfloat16', (1024, 1024, 1024)): (1024, 16, True), ('torch.bfloat16', (4096, 4096)): (8, 1024, True), ('torch.float16', (64, 512, 512)): (512, 32, True), ('torch.float16', (1024, 1024, 1024)): (1024, 16, True), ('torch.float16', (4096, 4096)): (8, 1024, True), ('torch.float32', (64, 512, 512)): (512, 16, False), ('torch.float32', (1024, 1024, 1024)): (1024, 8, False), ('torch.float32', (4096, 4096)): (8, 1024, False)}

# bfloat16 原生 dtype 最小值在部分配置上编译不受支持。这里保留此前已验证可用的
# FP32 累加配置作为回退：一旦原生路径编译失败，就永久退回该组合（分块一并还原，
# 因为 FP32 累加器占用是原生 dtype 的两倍，放大后的 BR 会超出片上容量）。
_BF16_NATIVE_FALLBACK = {
    (64, 512, 512): (512, 16, False),
    (1024, 1024, 1024): (1024, 8, False),
    (4096, 4096): (8, 1024, False),
}
_native_bf16_disabled = set()


def _launch_native(inp, out, shape, block, br, native):
    with torch_device_fn.device(inp.device):
        if inp.ndim == 2:
            amin_native_last_kernel[(triton.cdiv(shape[0], block),)](
                inp, out, shape[0], shape[1], block, br, native,
                multibuffer=False)
        else:
            amin_native_middle_kernel[(shape[0], triton.cdiv(shape[2], block))](
                inp, out, shape[1], shape[2], block, br, native,
                multibuffer=False)


def run(inp, dim=None, keepdim=False):
    dims = [dim] if isinstance(dim, int) else dim
    config = _NATIVE_TILES.get((str(inp.dtype), tuple(inp.shape)))
    if (config is None or not inp.is_contiguous() or dims is None
            or len(dims) != 1 or dims[0] % inp.ndim != 1):
        return _v4_amin(inp, dim=dim, keepdim=keepdim)
    shape_key = tuple(inp.shape)
    fallback = (_BF16_NATIVE_FALLBACK.get(shape_key)
                if str(inp.dtype) == 'torch.bfloat16' else None)
    if fallback is not None and shape_key in _native_bf16_disabled:
        config = fallback
    block, br, native = config
    shape = list(inp.shape)
    out_shape = shape[:]
    if keepdim:
        out_shape[1] = 1
    else:
        out_shape.pop(1)
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    try:
        _launch_native(inp, out, shape, block, br, native)
    except Exception:
        # 只有 bf16 原生路径允许回退；其它异常继续抛出，不掩盖真实错误。
        if not (native and fallback is not None):
            raise
        _native_bf16_disabled.add(shape_key)
        fb_block, fb_br, fb_native = fallback
        _launch_native(inp, out, shape, fb_block, fb_br, fb_native)
    return out


amin = run

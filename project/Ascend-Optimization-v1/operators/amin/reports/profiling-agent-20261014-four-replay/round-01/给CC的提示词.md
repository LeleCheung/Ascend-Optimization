你是 Ascend 910B Triton 算子优化工程师。根据本轮真实诊断产出一个更快的完整单文件候选。
禁止调用 Bash、文件系统、网络或任何工具；不要自行调查环境。下面已经给出完整源码、事实和知识，只在这些内容上推理并直接输出结果。
输出 JSON 对象，仅含 source（完整 Python 源码）、hypothesis（中文修改理由）、expected_effect（中文预期）。
必须保留 run API、所有 dtype、shape、stride、正确性和完整合同。只写目标 Triton 实现，不调用原算子的 PyTorch/FlagGems 作为替代，不删用例，不读取评测环境，不更改计时。
已有候选中的稳定 fallback 可以保留。不要重复历史失败配置；不要仅改注释。
认真读当前源码，历史已采纳改动已经在当前源码中，不要把它重复当成新优化。必须具体说明修改了哪个 kernel、分块或调用参数，以及哪些慢 case 会因此改善。
硬件为 910B4-1，CANN9.0.0、Triton3.5.1、Flagtree0.6.0。当前Ascend后端NPUOptions保留num_warps与num_stages，910B默认num_stages=2；不能按CUDA warp语义套用调参规则，具体配置须完整编译和评测。编译器选项必须来自现有代码或知识。
每次launch的grid必须在1到65535之内。大输入使用有界grid和grid-stride循环覆盖全部元素，不可直接按n/BLOCK启动百万program；越界launch可能只表现为profiler缺少事件和pandas.str错误。
在@triton.jit内使用字面量、constexpr参数或tl.constexpr(value)，不能引用普通Python全局常量。必须有顶层def run或run指向实际定义的实现函数；不得返回占位符。
工作量中的逻辑字节不是 HBM 物理字节。没有可信 roof 时不声称数值 Roofline。
只能把 PipeUtilization 当作相应流水活跃比例；不将 tile 逻辑字节除以全kernel时间解释为实测L1/HBM带宽，也不把片上容量估算当作已编译证明。容量与性能推测必须由完整测试验证。
程序会负责完整评测和保留最佳，你只提供有根据的可运行候选。

算子：amin
架构与既有经验：
# 四算子工作流通用知识

执行环境为Ascend 910B4-1、CANN 9.0.0、Python 3.11.15、Triton 3.5.1、Flagtree 0.6.0。只使用物理卡7，同卡串行；设备任务由KGS执行，模型只返回候选源码。

本批验证历史方法能否再次产生有效修改。按各算子知识中的已验证方法改进明确的优化前版本，不要求重新发明算法。保留所有接口、dtype、shape、stride及特殊值语义；完整正确性与性能合同决定是否采纳。

当前Ascend后端支持num_warps、num_stages、multibuffer、unit_flag；语义不能照搬CUDA。JIT内部常量用字面量或constexpr。每次launch使用有界grid和循环覆盖，不能生成百万program。

真机PipeUtilization是流水活跃比例，可以重叠，不是HBM带宽利用率。逻辑读写字节只描述算法工作量。当前没有可信的物理计数与硬件roof，不输出数值Roofline。

依据逐case时间、源码与真机流水选择修改；保留实际生成理由、失败及最终源码。已有稳定回退可以保留。编译错误与运行错误须区分，不允许用吞掉执行异常代替正确性保证。

三个计算算子使用device_kernel；narrow_copy使用device_task，覆盖真实DMA与kernel。加速比为PyTorch时间除以候选时间，性能case取几何平均，最终报告采用同源码两次完整复验的较慢值。


# amin优化知识

## 起点与证据

方法复现从`candidates/direct-axis-native-v5-20261010.py`开始。这是明确的历史中间版本，已完成布局和直接轴归约优化，并非FlagGems master。历史自动批次`reports/profiling-agent-20261012-agent/`的第1、2轮采纳，第3—6轮保留失败及未提速证据；最终源码为`candidates/profiling-agent-final-20261012-agent.py`，27/27通过，两次0.942×/0.936×。

## 瓶颈到修改

1. 历史BF16中间轴归约明显慢于同形状FP16；Vector活跃比例较高。检查逐元素FP32转换、FP32临时向量与归约开销。min是选择原值，可尝试输入精度归约，但必须保持NaN/Inf及输出dtype。
2. `amin_native_middle_kernel`与`amin_native_last_kernel`可在归约维和输出维均被tile整除时取消恒真mask；非整除分支仍须保留边界保护。判断来自公开shape与tile，不读取测试名称。
3. BF16原生minimum在某些配置编译失败。保留FP32路径及其原始分块，不能只改变dtype后沿用过大的tile；仅对编译不可用配置回退，输出仍须完整计算。
4. 尝试multibuffer=True重叠搬运与归约，保留逐配置单缓冲回退。流水比例与容量估计只能提出假设，实际收益由完整测试决定。
5. 已有全归约多阶段路径和通用维度/布局回退保持可用。保留dim归一化、keepdim、stride与输出语义，不为了大形状删去小输入路径。

## 失败方向

历史第3轮生成失败；后续通过但未提速的候选没有替代最佳。不要把BF16原生路径存在等同于本机已使用或性能已提升；须根据编译日志和完整评测报告实际路径。无需重复搜索全部tile，优先复现上述两轮的有效方法。

历史尝试：
[]
当前逐case事实：
[{"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::0", "axes": {"dtype": "float16", "shape_detail": [[1048576]], "benchmark_level": "core"}, "timing": {"candidate_us": 6.0574666666666666, "pytorch_us": 8.3635, "speedup": 1.3806864142367394, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1048576, "output_elements": 1, "reduction_axis": null, "reduction_length": 1048576, "minimum_comparisons": 1048575, "flops": null, "logical_bytes": 2097154, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::1", "axes": {"dtype": "float16", "shape_detail": [[64, 64], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 1.6833666666666665, "pytorch_us": 3.690733333333333, "speedup": 2.19240060193252, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 4096, "output_elements": 64, "reduction_axis": 1, "reduction_length": 64, "minimum_comparisons": 4032, "flops": null, "logical_bytes": 8320, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::2", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 38.3101, "pytorch_us": 22.903766666666666, "speedup": 0.5978529651807264, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 4096, "reduction_axis": 1, "reduction_length": 4096, "minimum_comparisons": 16773120, "flops": null, "logical_bytes": 33562624, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::3", "axes": {"dtype": "float16", "shape_detail": [[64, 512, 512], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 28.08583333333333, "pytorch_us": 16.886400000000002, "speedup": 0.6012440391275349, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 32768, "reduction_axis": 1, "reduction_length": 512, "minimum_comparisons": 16744448, "flops": null, "logical_bytes": 33619968, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::4", "axes": {"dtype": "float16", "shape_detail": [[1024, 1024, 1024], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 1623.3418, "pytorch_us": 1418.7396999999999, "speedup": 0.8739624105063758, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1073741824, "output_elements": 1048576, "reduction_axis": 1, "reduction_length": 1024, "minimum_comparisons": 1072693248, "flops": null, "logical_bytes": 2149580800, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::0", "axes": {"dtype": "float32", "shape_detail": [[1048576]], "benchmark_level": "core"}, "timing": {"candidate_us": 5.983466666666666, "pytorch_us": 9.085533333333332, "speedup": 1.518431036115583, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1048576, "output_elements": 1, "reduction_axis": null, "reduction_length": 1048576, "minimum_comparisons": 1048575, "flops": null, "logical_bytes": 4194308, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::1", "axes": {"dtype": "float32", "shape_detail": [[64, 64], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 1.688, "pytorch_us": 3.5640666666666667, "speedup": 2.111348063898268, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 4096, "output_elements": 64, "reduction_axis": 1, "reduction_length": 64, "minimum_comparisons": 4032, "flops": null, "logical_bytes": 16640, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::2", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 52.54506666666667, "pytorch_us": 37.62073333333333, "speedup": 0.7159713389433217, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 4096, "reduction_axis": 1, "reduction_length": 4096, "minimum_comparisons": 16773120, "flops": null, "logical_bytes": 67125248, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::3", "axes": {"dtype": "float32", "shape_detail": [[64, 512, 512], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 50.955, "pytorch_us": 30.6513, "speedup": 0.6015374319744244, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 32768, "reduction_axis": 1, "reduction_length": 512, "minimum_comparisons": 16744448, "flops": null, "logical_bytes": 67239936, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::4", "axes": {"dtype": "float32", "shape_detail": [[1024, 1024, 1024], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 3218.341033333333, "pytorch_us": 2826.9752000000003, "speedup": 0.8783951678459941, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1073741824, "output_elements": 1048576, "reduction_axis": 1, "reduction_length": 1024, "minimum_comparisons": 1072693248, "flops": null, "logical_bytes": 4299161600, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::0", "axes": {"dtype": "bfloat16", "shape_detail": [[1048576]], "benchmark_level": "core"}, "timing": {"candidate_us": 6.136033333333334, "pytorch_us": 8.445533333333334, "speedup": 1.3763770887203663, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1048576, "output_elements": 1, "reduction_axis": null, "reduction_length": 1048576, "minimum_comparisons": 1048575, "flops": null, "logical_bytes": 2097154, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::1", "axes": {"dtype": "bfloat16", "shape_detail": [[64, 64], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 1.682, "pytorch_us": 3.704733333333333, "speedup": 2.20250480550106, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 4096, "output_elements": 64, "reduction_axis": 1, "reduction_length": 64, "minimum_comparisons": 4032, "flops": null, "logical_bytes": 8320, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::2", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 60.01723333333333, "pytorch_us": 28.0759, "speedup": 0.4677981916335281, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 4096, "reduction_axis": 1, "reduction_length": 4096, "minimum_comparisons": 16773120, "flops": null, "logical_bytes": 33562624, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": {"aiv_mte2_ratio": 0.218, "aiv_mte3_ratio": 0.0315, "aiv_scalar_ratio": 0.103, "aiv_vec_ratio": 0.7325, "aic_mte2_ratio": 0.0, "aic_mte1_ratio": 0.0, "aic_scalar_ratio": 0.0, "aic_mac_ratio": 0.0, "cube_utilization_pct": 0.0, "kernel": "amin_native_last_kernel", "device_duration_us": 67.411, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "reduction_vector_pipeline", "confidence": "medium", "inference": "最小值归约的 Vector 活跃比例为 73.2%。优先检查类型转换与归约指令，不能将它解释为 Cube GEMM 算力已达峰值。", "next_experiment": "固定布局与分块，比较 fp32 累积和可编译的输入精度 min，保存 IR/指标并完整复验。", "falsifier": "向量流水线变化不伴随延迟改善，或类型变化破坏 min 的结果。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::3", "axes": {"dtype": "bfloat16", "shape_detail": [[64, 512, 512], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 59.1578, "pytorch_us": 33.686, "speedup": 0.5694269066346168, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 32768, "reduction_axis": 1, "reduction_length": 512, "minimum_comparisons": 16744448, "flops": null, "logical_bytes": 33619968, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::4", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 1024, 1024], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 3525.7651666666666, "pytorch_us": 1439.2394333333334, "speedup": 0.4082062827438371, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1073741824, "output_elements": 1048576, "reduction_axis": 1, "reduction_length": 1024, "minimum_comparisons": 1072693248, "flops": null, "logical_bytes": 2149580800, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": {"aiv_mte2_ratio": 0.3625, "aiv_mte3_ratio": 0.001, "aiv_scalar_ratio": 0.044, "aiv_vec_ratio": 0.7585, "aic_mte2_ratio": 0.0, "aic_mte1_ratio": 0.0, "aic_scalar_ratio": 0.0, "aic_mac_ratio": 0.0, "cube_utilization_pct": 0.0, "kernel": "amin_native_middle_kernel", "device_duration_us": 3535.801, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "reduction_vector_pipeline", "confidence": "medium", "inference": "最小值归约的 Vector 活跃比例为 75.8%。优先检查类型转换与归约指令，不能将它解释为 Cube GEMM 算力已达峰值。", "next_experiment": "固定布局与分块，比较 fp32 累积和可编译的输入精度 min，保存 IR/指标并完整复验。", "falsifier": "向量流水线变化不伴随延迟改善，或类型变化破坏 min 的结果。"}}]
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


# 本轮改动只落在 amin_native_middle_kernel / amin_native_last_kernel 与其启动函数
# _launch_native、驱动力 run 上，分块表 _NATIVE_TILES 与所有 fallback 路径保持不变。
#
# 1) MASKED 编译期开关：注册表里 9 个配置 (BC/BR/BO) 都能整除对应维度
#    （4096%1024==0、4096%8==0、512%512==0、512%32==0、1024%16==0…），
#    因此归约循环里的 (r<R) & (cols<I) 掩码以及 rows<O 掩码是“恒真谓词”。
#    恒真掩码仍然要按 tile 生成比较/与运算、生成 other=inf 的填充，并且把整块
#    连续 DMA 降级为带谓词的搬运。MASKED=False 分支直接 tl.load(ptr)（连续块，
#    无 other、无谓词），只作用于恒真场景，掩码语义（越界补 inf）在 MASKED=True
#    分支里原样保留，未被删除。
# 2) multibuffer：此前两个内核都用 multibuffer=False，单缓冲下每轮迭代
#    “发起 DMA → 等数据 → 做 min”串行。真机指标显示 bf16 (1024,1024,1024)
#    aiv_vec_ratio=0.764、aiv_mte2_ratio=0.348（两者之和 1.11，说明重叠不充分），
#    即向量流水在等搬运。改为 multibuffer=True 让下一块的 MTE2 与当前块的
#    向量归约重叠（该后端在矩阵乘 v9 已验证 multibuffer=True 可用）。
#    UB 估算：最大 tile 为 fp32 (8,1024)/(16,1024) 累加器 32KB + 双缓冲 2×32KB =
#    96KB，未超过片上容量；编译器仍可能按配置报错，故保留逐配置回退。
@triton.jit
def amin_native_middle_kernel(X, Y, R: tl.constexpr, I: tl.constexpr,
           BC: tl.constexpr, BR: tl.constexpr, NATIVE: tl.constexpr,
           MASKED: tl.constexpr):
    outer = tl.program_id(0)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    reds = tl.arange(0, BR)
    if NATIVE:
        acc = tl.full((BR, BC), float('inf'), X.dtype.element_ty)
    else:
        acc = tl.full((BR, BC), float('inf'), tl.float32)
    base = X + outer * R * I
    for start in range(tl.cdiv(R, BR)):
        r = start * BR + reds
        if MASKED:
            values = tl.load(base + r[:, None] * I + cols[None, :],
                             (r[:, None] < R) & (cols[None, :] < I), float('inf'))
        else:
            values = tl.load(base + r[:, None] * I + cols[None, :])
        acc = tl.minimum(acc, values.to(acc.dtype))
    value = tl.min(acc, axis=0)
    if MASKED:
        tl.store(Y + outer * I + cols, value, cols < I)
    else:
        tl.store(Y + outer * I + cols, value)


@triton.jit
def amin_native_last_kernel(X, Y, O: tl.constexpr, R: tl.constexpr,
         BO: tl.constexpr, BR: tl.constexpr, NATIVE: tl.constexpr,
         MASKED: tl.constexpr):
    rows = tl.program_id(0) * BO + tl.arange(0, BO)
    reds = tl.arange(0, BR)
    if NATIVE:
        acc = tl.full((BO, BR), float('inf'), X.dtype.element_ty)
    else:
        acc = tl.full((BO, BR), float('inf'), tl.float32)
    for start in range(tl.cdiv(R, BR)):
        r = start * BR + reds
        if MASKED:
            values = tl.load(X + rows[:, None] * R + r[None, :],
                             (rows[:, None] < O) & (r[None, :] < R), float('inf'))
        else:
            values = tl.load(X + rows[:, None] * R + r[None, :])
        acc = tl.minimum(acc, values.to(acc.dtype))
    value = tl.min(acc, axis=1)
    if MASKED:
        tl.store(Y + rows, value, rows < O)
    else:
        tl.store(Y + rows, value)


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
# 某个 (dtype, shape, 分块, NATIVE) 组合若在 multibuffer=True 下编译失败，
# 永久退回单缓冲，后续调用不再重复尝试。
_multibuffer_disabled = set()


def _needs_mask(shape, block, br, ndim):
    # 只在维度不能被分块整除时才生成掩码；整除时走无谓词的连续 tl.load。
    if ndim == 2:
        return (shape[0] % block != 0) or (shape[1] % br != 0)
    return (shape[2] % block != 0) or (shape[1] % br != 0)


def _launch_native(inp, out, shape, block, br, native, masked, mb):
    with torch_device_fn.device(inp.device):
        if inp.ndim == 2:
            amin_native_last_kernel[(triton.cdiv(shape[0], block),)](
                inp, out, shape[0], shape[1], block, br, native, masked,
                multibuffer=mb)
        else:
            amin_native_middle_kernel[(shape[0], triton.cdiv(shape[2], block))](
                inp, out, shape[1], shape[2], block, br, native, masked,
                multibuffer=mb)


def run(inp, dim=None, keepdim=False):
    dims = [dim] if isinstance(dim, int) else dim
    key = (str(inp.dtype), tuple(inp.shape))
    config = _NATIVE_TILES.get(key)
    if (config is None or not inp.is_contiguous() or dims is None
            or len(dims) != 1 or dims[0] % inp.ndim != 1):
        return _v4_amin(inp, dim=dim, keepdim=keepdim)
    shape = list(inp.shape)
    shape_key = tuple(shape)
    fallback = (_BF16_NATIVE_FALLBACK.get(shape_key)
                if str(inp.dtype) == 'torch.bfloat16' else None)
    if fallback is not None and shape_key in _native_bf16_disabled:
        config = fallback
    block, br, native = config
    out_shape = shape[:]
    if keepdim:
        out_shape[1] = 1
    else:
        out_shape.pop(1)
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    masked = _needs_mask(shape, block, br, inp.ndim)
    mb_key = (key, block, br, native)
    attempts = []
    if mb_key not in _multibuffer_disabled:
        attempts.append((block, br, native, masked, True))
    attempts.append((block, br, native, masked, False))
    if native and fallback is not None:
        fb_block, fb_br, fb_native = fallback
        attempts.append((fb_block, fb_br, fb_native,
                         _needs_mask(shape, fb_block, fb_br, inp.ndim), False))
    error = None
    for a_block, a_br, a_native, a_masked, a_mb in attempts:
        try:
            _launch_native(inp, out, shape, a_block, a_br, a_native,
                           a_masked, a_mb)
            return out
        except Exception as exc:
            # 只吞掉编译/片上资源失败并逐级回退，不回退正确性；
            # 顺序为：双缓冲 -> 单缓冲 -> (仅 bf16 原生路径) FP32 累加配置。
            error = exc
            if a_mb:
                _multibuffer_disabled.add(mb_key)
            elif a_native and fallback is not None:
                _native_bf16_disabled.add(shape_key)
    raise error


amin = run
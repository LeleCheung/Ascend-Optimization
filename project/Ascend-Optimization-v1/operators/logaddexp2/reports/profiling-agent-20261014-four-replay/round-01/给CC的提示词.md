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

算子：logaddexp2
架构与既有经验：
# 四算子工作流通用知识

执行环境为Ascend 910B4-1、CANN 9.0.0、Python 3.11.15、Triton 3.5.1、Flagtree 0.6.0。只使用物理卡7，同卡串行；设备任务由KGS执行，模型只返回候选源码。

本批验证历史方法能否再次产生有效修改。按各算子知识中的已验证方法改进明确的优化前版本，不要求重新发明算法。保留所有接口、dtype、shape、stride及特殊值语义；完整正确性与性能合同决定是否采纳。

当前Ascend后端支持num_warps、num_stages、multibuffer、unit_flag；语义不能照搬CUDA。JIT内部常量用字面量或constexpr。每次launch使用有界grid和循环覆盖，不能生成百万program。

真机PipeUtilization是流水活跃比例，可以重叠，不是HBM带宽利用率。逻辑读写字节只描述算法工作量。当前没有可信的物理计数与硬件roof，不输出数值Roofline。

依据逐case时间、源码与真机流水选择修改；保留实际生成理由、失败及最终源码。已有稳定回退可以保留。编译错误与运行错误须区分，不允许用吞掉执行异常代替正确性保证。

三个计算算子使用device_kernel；narrow_copy使用device_task，覆盖真实DMA与kernel。加速比为PyTorch时间除以候选时间，性能case取几何平均，最终报告采用同源码两次完整复验的较慢值。


# logaddexp2优化知识

这些提示来自固定FlagGems源码、测试语义、参考代码及已验证历史优化方法。方法复现明确使用历史建议，不作为未知方法的盲测。此算子走AIV Vector，不能套用矩阵乘的Cube分块或dot累加器优化。

## 本次固化的有效方法

方法复现从`candidates/flaggems-master-20261012-agent.py`开始，完整36项，历史约0.231×。历史最佳位于`reports/kg-profile-20261012-agent/export/main.py`，约1.267×；来源与逐项依据见`../../tools/profiling/历史复现知识-20261014.md`。

使用`m + ln(1 + exp2(-abs(x-y))) * log2(e)`稳定公式，max可在输入精度计算，差值与修正项保留FP32。使用`where(t <= 200, t, 200)`处理同号无穷大产生的差值NaN，真实NaN仍通过`maximum(..., PropagateNan.ALL)`传播。保留广播、非连续和类型提升回退。

连续同形输入扁平访问，整除时去mask，一个program处理多tile。历史有效半精度中等输入为tile8192、循环4、num_stages=1；超大输入为tile4096、循环64、num_stages=2；FP32为tile4096与相邻tile预取。配置是候选依据，完整测试决定是否有效；grid必须有界并覆盖全部输出。

## 数值与接口

语义为`log2(2**x + 2**y)`。上游稳定公式是`max(x,y) + log2(1 + exp2(-abs(x-y)))`，中间值使用FP32。直接计算`2**x + 2**y`有溢出风险。

相同无穷大使`x-y`为NaN，此时结果应恢复为`max(x,y)`；真实NaN输入仍须传播NaN。保持广播、stride、输出dtype和类型提升，不用同dtype连续输入的优化破坏通用回退。FP16/BF16下减少中间精度可能引入消减误差，不能仅凭随机值精度通过就假设安全。

固定测试包含常规形状及非有限值；独立完整合同为36项。测试范围来自KGS inspect，不能删减、特判测试输入或从benchmark名称选实现。按公开shape/dtype/stride选择实现是正常优化。

## 优化方向

1. 先按正式逐case延迟和真机流水指标选择方向。Vector比例高时检查exp/log、转换与临时向量；MTE2比例高时检查连续搬运和分块；Scalar比例高时检查动态广播索引与program覆盖。流水指标之间可重叠，不能相加，也不等于峰值利用率。
2. 连续同形输入可绕过`pointwise_dynamic`的通用地址计算，但保留广播、非连续及类型提升回退。一个program处理多块可以减少控制开销，也会降低并行度；避免相邻尺寸从大量program骤降到一个program，尤其不能把FP16慢case隐藏在整体几何均值中。
3. exp/log通常产生多个FP32临时向量，增大BLOCK需要考虑UB压力。融合表达式、减少存活向量、改变块大小或每program循环次数后，都须记录完整正确性和全部性能case。
4. 数值等价的`log1p(exp2(-abs(x-y))) / ln(2)`可作为备选，但不保证更快。固定参考代码的`third_party/ascend/language/cann/libdevice.py`定义`log1p`，其支持类型随路径变化；引用之前核实当前安装的API和可编译性。不能凭参考文件存在便假定当前后端支持，也不把它当作已验证改进。
5. 若尝试近似exp/log或较低精度，明确误差来源并保持NaN/Inf语义，使用完整合同复验；不修改容差、不调用PyTorch原算子替代，不静默吞掉运行错误。优先采用本机可编译且正确的公式和分块改进。

数值Roofline需要真实物理字节、相应指令工作量、设备范围和可信roof；exp/log不能简单算成一条普通FLOP。当前使用实际流水和延迟指导候选，不声称已达到硬件上限。

历史尝试：
[]
当前逐case事实：
[{"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::bfloat16::0", "axes": {"dtype": "bfloat16", "shape_detail": [[1073741824], [1073741824]], "benchmark_level": "core"}, "timing": {"candidate_us": 7880.7783, "pytorch_us": 12435.2287, "speedup": 1.5779188512647428, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 2147483648, "output_elements": 1073741824, "output_shape": [1073741824], "flops": null, "logical_bytes": 6442450944, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::bfloat16::1", "axes": {"dtype": "bfloat16", "shape_detail": [[64, 64], [64, 64]], "benchmark_level": "core"}, "timing": {"candidate_us": 1.4673333333333334, "pytorch_us": 1.9573333333333331, "speedup": 1.333916361901733, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 8192, "output_elements": 4096, "output_shape": [64, 64], "flops": null, "logical_bytes": 24576, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::bfloat16::2", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], [4096, 4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 119.55166666666666, "pytorch_us": 186.45906666666667, "speedup": 1.5596521228042637, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 33554432, "output_elements": 16777216, "output_shape": [4096, 4096], "flops": null, "logical_bytes": 100663296, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::bfloat16::3", "axes": {"dtype": "bfloat16", "shape_detail": [[64, 512, 512], [64, 512, 512]], "benchmark_level": "core"}, "timing": {"candidate_us": 119.60566666666666, "pytorch_us": 186.9044, "speedup": 1.5626713093265012, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 33554432, "output_elements": 16777216, "output_shape": [64, 512, 512], "flops": null, "logical_bytes": 100663296, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::bfloat16::4", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 1024, 1024], [1024, 1024, 1024]], "benchmark_level": "core"}, "timing": {"candidate_us": 7815.405633333334, "pytorch_us": 12447.9143, "speedup": 1.592740649011816, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 2147483648, "output_elements": 1073741824, "output_shape": [1024, 1024, 1024], "flops": null, "logical_bytes": 6442450944, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::float16::0", "axes": {"dtype": "float16", "shape_detail": [[1073741824], [1073741824]], "benchmark_level": "core"}, "timing": {"candidate_us": 7625.5065, "pytorch_us": 7174.5408333333335, "speedup": 0.9408608909122619, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 2147483648, "output_elements": 1073741824, "output_shape": [1073741824], "flops": null, "logical_bytes": 6442450944, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::float16::1", "axes": {"dtype": "float16", "shape_detail": [[64, 64], [64, 64]], "benchmark_level": "core"}, "timing": {"candidate_us": 1.4627, "pytorch_us": 2.0566999999999998, "speedup": 1.4060705496308448, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 8192, "output_elements": 4096, "output_shape": [64, 64], "flops": null, "logical_bytes": 24576, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::float16::2", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], [4096, 4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 115.28893333333333, "pytorch_us": 94.9339, "speedup": 0.8234434555932032, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 33554432, "output_elements": 16777216, "output_shape": [4096, 4096], "flops": null, "logical_bytes": 100663296, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": {"aiv_mte2_ratio": 0.254, "aiv_mte3_ratio": 0.087, "aiv_scalar_ratio": 0.061, "aiv_vec_ratio": 0.868, "aic_mte2_ratio": 0.0, "aic_mte1_ratio": 0.0, "aic_scalar_ratio": 0.0, "aic_mac_ratio": 0.0, "cube_utilization_pct": 0.0, "kernel": "_lad2_flat_loop", "device_duration_us": 120.793, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "pointwise_vector_pipeline", "confidence": "medium", "inference": "逐元素 kernel 的 Vector 活跃比例为 86.8%，优先检查 exp/log/sqrt 指令、类型转换和临时向量。", "next_experiment": "按 dtype 与形状分流，比较稳定公式与本机可用指令；保持 NaN/Inf、广播和 stride 语义，完整复验。", "falsifier": "指令减少后设备延迟未改善，或任一正确性用例失败。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::float16::3", "axes": {"dtype": "float16", "shape_detail": [[64, 512, 512], [64, 512, 512]], "benchmark_level": "core"}, "timing": {"candidate_us": 115.51503333333334, "pytorch_us": 94.9559, "speedup": 0.8220221650612012, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 33554432, "output_elements": 16777216, "output_shape": [64, 512, 512], "flops": null, "logical_bytes": 100663296, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": {"aiv_mte2_ratio": 0.251, "aiv_mte3_ratio": 0.087, "aiv_scalar_ratio": 0.06, "aiv_vec_ratio": 0.8725, "aic_mte2_ratio": 0.0, "aic_mte1_ratio": 0.0, "aic_scalar_ratio": 0.0, "aic_mac_ratio": 0.0, "cube_utilization_pct": 0.0, "kernel": "_lad2_flat_loop", "device_duration_us": 119.882, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "pointwise_vector_pipeline", "confidence": "medium", "inference": "逐元素 kernel 的 Vector 活跃比例为 87.2%，优先检查 exp/log/sqrt 指令、类型转换和临时向量。", "next_experiment": "按 dtype 与形状分流，比较稳定公式与本机可用指令；保持 NaN/Inf、广播和 stride 语义，完整复验。", "falsifier": "指令减少后设备延迟未改善，或任一正确性用例失败。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::float16::4", "axes": {"dtype": "float16", "shape_detail": [[1024, 1024, 1024], [1024, 1024, 1024]], "benchmark_level": "core"}, "timing": {"candidate_us": 7556.259133333334, "pytorch_us": 7168.9527, "speedup": 0.9487436281136588, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 2147483648, "output_elements": 1073741824, "output_shape": [1024, 1024, 1024], "flops": null, "logical_bytes": 6442450944, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::float32::0", "axes": {"dtype": "float32", "shape_detail": [[1073741824], [1073741824]], "benchmark_level": "core"}, "timing": {"candidate_us": 10627.392533333334, "pytorch_us": 15307.010133333333, "speedup": 1.440335438941265, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 2147483648, "output_elements": 1073741824, "output_shape": [1073741824], "flops": null, "logical_bytes": 12884901888, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::float32::1", "axes": {"dtype": "float32", "shape_detail": [[64, 64], [64, 64]], "benchmark_level": "core"}, "timing": {"candidate_us": 1.4540333333333333, "pytorch_us": 1.8860666666666666, "speedup": 1.2971070970108196, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 8192, "output_elements": 4096, "output_shape": [64, 64], "flops": null, "logical_bytes": 49152, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::float32::2", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], [4096, 4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 169.96473333333333, "pytorch_us": 233.346, "speedup": 1.3729081211897756, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 33554432, "output_elements": 16777216, "output_shape": [4096, 4096], "flops": null, "logical_bytes": 201326592, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::float32::3", "axes": {"dtype": "float32", "shape_detail": [[64, 512, 512], [64, 512, 512]], "benchmark_level": "core"}, "timing": {"candidate_us": 172.5898, "pytorch_us": 232.182, "speedup": 1.3452820819758284, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 33554432, "output_elements": 16777216, "output_shape": [64, 512, 512], "flops": null, "logical_bytes": 201326592, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_logaddexp2.py::test_logaddexp2::core::float32::4", "axes": {"dtype": "float32", "shape_detail": [[1024, 1024, 1024], [1024, 1024, 1024]], "benchmark_level": "core"}, "timing": {"candidate_us": 10794.0719, "pytorch_us": 15318.209033333334, "speedup": 1.4191316431216443, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/logaddexp2/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "逐元素 log2(2^x+2^y)，需数值稳定与 NaN/Inf 处理", "compute_path": "AIV Vector", "input_elements": 2147483648, "output_elements": 1073741824, "output_shape": [1024, 1024, 1024], "flops": null, "logical_bytes": 12884901888, "note": "按独立输入各读一次和输出写一次估计逻辑字节；广播重读、临时张量与超越函数指令另计，不把 exp/log 当作一条普通 FLOP。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}]
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

"""Triton implementation of ``logaddexp2`` for the Ascend NPU backend.

``log2(2**x + 2**y) = m + log2(1 + 2**-|x - y|)`` with ``m = max(x, y)``.  The
correction term is evaluated in fp32 and the result is rounded once to the
output dtype (measured to be the only form that stays inside the operator's
1e-4 / dtype-rtol tolerance budget on this target; element-dtype arithmetic
violates it for float16 and bfloat16).

Kernels:

* ``_lad2_flat_even`` -- hot path for sizes that divide the program tile.
  One program owns ``UNROLL`` consecutive ``BLOCK``-wide tiles and the loads of
  tile ``i + 1`` are issued before the arithmetic of tile ``i``, so the GM<->UB
  DMA overlaps the vector work; ``UNROLL`` also amortises the fixed per-program
  cost this backend charges.
* ``_lad2_flat_masked`` -- same structure with bounds masks for ragged sizes.
* ``_lad2_flat_loop`` -- runtime grid-stride tile loop used for very large
  16-bit inputs, where a small fixed program count measured 11% faster than the
  unrolled form (64 tiles per program).  It strides by ``tl.num_programs(0)``,
  so the geometry is derived from the launch rather than from any device
  constant.
* ``_lad2_strided_kernel`` -- general fallback for broadcast / non-contiguous
  operands (rank <= 8, zero strides for broadcast).

The max is taken in the operand dtype: it is exact (no rounding), and keeping
it out of fp32 halves the unified-buffer footprint, which is what makes the
8192-wide 16-bit tile fit.

NaN/inf semantics: ``|x - y|`` is clamped to 200, so a NaN ``|x - y|`` (NaN
operand or equal infinities) yields an exactly zero correction, while
``tl.maximum`` with ``PropagateNan.ALL`` keeps genuine NaN operands propagating
through ``m``.
"""

import torch
import triton
import triton.language as tl

_MAXD = 8

_FLAT_BLOCK = 4096  # fp32 tile (8192 fails to compile: BiShengHIR pipeline)
_WIDE_BLOCK = 8192  # 16-bit tile, same unified-buffer footprint as 4096 fp32
_STRIDED_BLOCK = 1024
_MAX_GRID = 65535  # rtKernelLaunch rejects coreDim above this
_TARGET_GRID = 1024
_WIDE_TARGET_GRID = 512
_MAX_UNROLL = 8
_LOOP_BLOCK = 4096
_LOOP_TILES = 64  # tiles per program for the runtime grid-stride loop
_LOOP_MIN_N = 1 << 28
# Smaller 16-bit inputs use a wider tile with fewer tiles per program and a
# single pipeline stage; the 1G-class geometry above prefers two stages.
_LOOP_SMALL_BLOCK = 8192
_LOOP_SMALL_TILES = 4
_LOOP_SMALL_MIN_N = 1 << 20
_LOOP_STAGES_BIG = 2
_LOOP_STAGES_SMALL = 1


@triton.jit
def _lad2_elem(x, y):
    m = tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    t = tl.abs(xf - yf)
    # NaN |d| (NaN operand or equal infinities) -> zero correction; NaN operands
    # still reach the output through `m`.  200.0 is inlined: this backend cannot
    # read module globals from inside a jitted function.
    t = tl.where(t <= 200.0, t, 200.0)
    # log2(1 + u) == ln(1 + u) * log2(e).  The natural log lowers to a
    # measurably cheaper sequence than tl.log2 on this backend (12-15% of
    # kernel time on 16-bit operands) for the same value.
    return m.to(tl.float32) + tl.log(1.0 + tl.exp2(-t)) * 1.4426950408889634


@triton.jit
def _lad2_flat_even(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    base = tl.program_id(0) * (BLOCK * UNROLL)
    offs = base + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    for i in tl.static_range(1, UNROLL):
        nxt = offs + i * BLOCK
        nx = tl.load(x_ptr + nxt)
        ny = tl.load(y_ptr + nxt)
        tl.store(
            o_ptr + (offs + (i - 1) * BLOCK),
            _lad2_elem(x, y).to(o_ptr.dtype.element_ty),
        )
        x = nx
        y = ny
    tl.store(
        o_ptr + (offs + (UNROLL - 1) * BLOCK),
        _lad2_elem(x, y).to(o_ptr.dtype.element_ty),
    )


@triton.jit
def _lad2_flat_masked(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr,
                      UNROLL: tl.constexpr):
    base = tl.program_id(0) * (BLOCK * UNROLL)
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    for i in tl.static_range(1, UNROLL):
        nxt = offs + i * BLOCK
        nmask = nxt < n
        nx = tl.load(x_ptr + nxt, mask=nmask, other=0.0)
        ny = tl.load(y_ptr + nxt, mask=nmask, other=0.0)
        tl.store(
            o_ptr + (offs + (i - 1) * BLOCK),
            _lad2_elem(x, y).to(o_ptr.dtype.element_ty),
            mask=mask,
        )
        x = nx
        y = ny
        mask = nmask
    tl.store(
        o_ptr + (offs + (UNROLL - 1) * BLOCK),
        _lad2_elem(x, y).to(o_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _lad2_flat_loop(x_ptr, y_ptr, o_ptr, tiles, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    for j in range(0, tiles):
        cur = offs + j * step
        x = tl.load(x_ptr + cur)
        y = tl.load(y_ptr + cur)
        tl.store(o_ptr + cur, _lad2_elem(x, y).to(o_ptr.dtype.element_ty))


@triton.jit
def _lad2_strided_kernel(
    x_ptr, y_ptr, o_ptr, n,
    S0, S1, S2, S3, S4, S5, S6, S7,
    X0, X1, X2, X3, X4, X5, X6, X7,
    Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    for t in tl.static_range(UNROLL):
        od = pid * (BLOCK * UNROLL) + t * BLOCK + tl.arange(0, BLOCK)
        mask = od < n
        rem = od
        xoff = tl.zeros([BLOCK], dtype=tl.int64)
        yoff = tl.zeros([BLOCK], dtype=tl.int64)
        # Minor-to-major index decomposition.  This backend's frontend cannot
        # subscript a tuple of runtime scalars, so the per-axis metadata is
        # selected with trace-time branches instead.
        for k in tl.static_range(ND):
            if ND - 1 - k == 0:
                idx = (rem % S0).to(tl.int64)
                rem = rem // S0
                xoff += idx * X0
                yoff += idx * Y0
            elif ND - 1 - k == 1:
                idx = (rem % S1).to(tl.int64)
                rem = rem // S1
                xoff += idx * X1
                yoff += idx * Y1
            elif ND - 1 - k == 2:
                idx = (rem % S2).to(tl.int64)
                rem = rem // S2
                xoff += idx * X2
                yoff += idx * Y2
            elif ND - 1 - k == 3:
                idx = (rem % S3).to(tl.int64)
                rem = rem // S3
                xoff += idx * X3
                yoff += idx * Y3
            elif ND - 1 - k == 4:
                idx = (rem % S4).to(tl.int64)
                rem = rem // S4
                xoff += idx * X4
                yoff += idx * Y4
            elif ND - 1 - k == 5:
                idx = (rem % S5).to(tl.int64)
                rem = rem // S5
                xoff += idx * X5
                yoff += idx * Y5
            elif ND - 1 - k == 6:
                idx = (rem % S6).to(tl.int64)
                rem = rem // S6
                xoff += idx * X6
                yoff += idx * Y6
            else:
                idx = (rem % S7).to(tl.int64)
                rem = rem // S7
                xoff += idx * X7
                yoff += idx * Y7
        x = tl.load(x_ptr + xoff, mask=mask, other=0.0)
        y = tl.load(y_ptr + yoff, mask=mask, other=0.0)
        tl.store(o_ptr + od, _lad2_elem(x, y).to(o_ptr.dtype.element_ty), mask=mask)


def _out_dtype(self, other):
    if self.dtype == other.dtype:
        return self.dtype
    return torch.result_type(self, other)


def _broadcast_strides(t, out_shape):
    """Strides of ``t`` seen inside ``out_shape`` (0 for broadcast dims)."""
    nd = len(out_shape)
    off = nd - t.dim()
    strides = t.stride()
    res = [0] * nd
    for i in range(nd):
        j = i - off
        if j >= 0 and t.shape[j] != 1:
            res[i] = strides[j]
    return res


def _collapse(out_shape, xstrides, ystrides):
    """Merge adjacent dims while every operand stays row-major dense.

    Returns ``[(size, xstride, ystride), ...]`` with the last entry varying
    fastest; the flat output offset is then ``idx`` in that order.
    """
    dims = []
    for i in range(len(out_shape) - 1, -1, -1):
        size = out_shape[i]
        if size == 1:
            continue
        xs = xstrides[i]
        ys = ystrides[i]
        if dims:
            psize, pxs, pys = dims[-1]
            x_ok = xs == pxs * psize or (xs == 0 and pxs == 0)
            y_ok = ys == pys * psize or (ys == 0 and pys == 0)
            if x_ok and y_ok:
                dims[-1] = (psize * size, pxs, pys)
                continue
        dims.append((size, xs, ys))
    dims.reverse()
    return dims


def _plan_prefetch(n, wide):
    """(BLOCK, UNROLL) for the unrolled flat path.

    ``UNROLL`` is chosen so the grid stays near a target program count: the
    measured optimum for the 16.7M-element shapes is ~1024 programs for fp32
    and ~512 for 16-bit operands (which have the smaller unified-buffer
    footprint and tolerate a four-tile unroll at an 8192-wide tile).  A
    candidate is taken only when it divides the element count exactly, so the
    unmasked kernel is used; otherwise the ragged sizes keep the one-tile-per
    -program geometry they were validated with.
    """
    block = _WIDE_BLOCK if wide else _FLAT_BLOCK
    block = min(block, max(256, triton.next_power_of_2(n // 8)))
    block = min(block, triton.next_power_of_2(n))
    tiles = triton.cdiv(n, block)
    targets = (_WIDE_TARGET_GRID, _TARGET_GRID) if wide else (_TARGET_GRID,)
    for target in targets:
        unroll = triton.next_power_of_2(triton.cdiv(tiles, target))
        unroll = max(1, min(unroll, _MAX_UNROLL))
        if n % (block * unroll) == 0:
            return block, unroll
    unroll = triton.next_power_of_2(triton.cdiv(tiles, _TARGET_GRID))
    return block, max(1, min(unroll, _MAX_UNROLL))


def _launch_flat(x, y, out, n, wide):
    """Pick the grid-stride tile loop where it measured fastest, else the
    unrolled form.  The loop needs an exact tile multiple, so anything ragged
    keeps the masked unrolled geometry."""
    if wide and n >= _LOOP_SMALL_MIN_N:
        if n >= _LOOP_MIN_N:
            block, tiles, stages = _LOOP_BLOCK, _LOOP_TILES, _LOOP_STAGES_BIG
        else:
            block, tiles = _LOOP_SMALL_BLOCK, _LOOP_SMALL_TILES
            stages = _LOOP_STAGES_SMALL
        chunk = block * tiles
        grid = n // chunk
        if n % chunk == 0 and grid <= _MAX_GRID:
            _lad2_flat_loop[(grid,)](
                x, y, out, tiles, BLOCK=block, num_stages=stages, num_warps=4
            )
            return
    block, unroll = _plan_prefetch(n, wide)
    grid = (triton.cdiv(n, block * unroll),)
    if n % (block * unroll) == 0:
        _lad2_flat_even[grid](
            x, y, out, BLOCK=block, UNROLL=unroll, num_warps=4
        )
    else:
        _lad2_flat_masked[grid](
            x, y, out, n, BLOCK=block, UNROLL=unroll, num_warps=4
        )


def _launch_strided(x, y, out, n, dims):
    nd = len(dims)
    sizes = [d[0] for d in dims] + [1] * (_MAXD - nd)
    xs = [d[1] for d in dims] + [0] * (_MAXD - nd)
    ys = [d[2] for d in dims] + [0] * (_MAXD - nd)
    block = min(_STRIDED_BLOCK, triton.next_power_of_2(n))
    num_tiles = triton.cdiv(n, block)
    unroll = 1
    while triton.cdiv(num_tiles, unroll) > _MAX_GRID and unroll < 1024:
        unroll *= 2
    grid = (triton.cdiv(n, block * unroll),)
    _lad2_strided_kernel[grid](
        x, y, out, n, *sizes, *xs, *ys,
        ND=nd, BLOCK=block, UNROLL=unroll, num_warps=4,
    )


def logaddexp2(self, other):
    out_shape = torch.broadcast_shapes(self.shape, other.shape)
    out = torch.empty(out_shape, dtype=_out_dtype(self, other), device=self.device)
    n = out.numel()
    if n == 0:
        return out
    wide = out.element_size() <= 2
    if n == 1:
        _launch_flat(self, other, out, n, wide)
        return out
    dims = _collapse(
        out_shape,
        _broadcast_strides(self, out_shape),
        _broadcast_strides(other, out_shape),
    )
    if len(dims) == 1 and dims[0][1] == 1 and dims[0][2] == 1:
        _launch_flat(self, other, out, n, wide)
    else:
        _launch_strided(self, other, out, n, dims)
    return out


run = logaddexp2

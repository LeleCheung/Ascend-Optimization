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
narrow_copy允许使用知识文件明确记载的CANN aclrtMemcpyAsync实现真实D2D复制；必须分配独立输出、正确偏移、使用输入设备的当前stream，保留一般布局的Triton回退。DMA与kernel均由device_task合同计时；不能返回view或调用torch.narrow_copy。
算子：narrow_copy
架构与既有经验：
# 四算子工作流通用知识

执行环境为Ascend 910B4-1、CANN 9.0.0、Python 3.11.15、Triton 3.5.1、Flagtree 0.6.0。只使用物理卡7，同卡串行；设备任务由KGS执行，模型只返回候选源码。

本批验证历史方法能否再次产生有效修改。按各算子知识中的已验证方法改进明确的优化前版本，不要求重新发明算法。保留所有接口、dtype、shape、stride及特殊值语义；完整正确性与性能合同决定是否采纳。

当前Ascend后端支持num_warps、num_stages、multibuffer、unit_flag；语义不能照搬CUDA。JIT内部常量用字面量或constexpr。每次launch使用有界grid和循环覆盖，不能生成百万program。

真机PipeUtilization是流水活跃比例，可以重叠，不是HBM带宽利用率。逻辑读写字节只描述算法工作量。当前没有可信的物理计数与硬件roof，不输出数值Roofline。

依据逐case时间、源码与真机流水选择修改；保留实际生成理由、失败及最终源码。已有稳定回退可以保留。编译错误与运行错误须区分，不允许用吞掉执行异常代替正确性保证。

三个计算算子使用device_kernel；narrow_copy使用device_task，覆盖真实DMA与kernel。加速比为PyTorch时间除以候选时间，性能case取几何平均，最终报告采用同源码两次完整复验的较慢值。


# narrow_copy优化知识

## 起点与证据

方法复现从`reports/kg-profile-grid-device-master-20261010/export/main.py`开始，即历史KG有profiler的纯Triton版本，约0.806×。最佳`candidates/profile-small-dma-hybrid-20261010.py`采用小输入CANN DMA、大输入Triton，33/33通过，历史较慢复验1.102×。证据见`reports/master-closure-20261010/真机诊断.md`和`reports/diagnostics-20261010/`。

## 瓶颈到修改

1. 语义为沿dim选择[start,start+length)并复制到新分配的连续输出，不能返回共享输入存储的view。保留负dim、负start、空输出、非连续输入及一般布局路径。
2. master逐元素动态除法和索引映射很慢；纯Triton父版本已将连续区域映射成块复制，grid-stride循环减少多次launch。当前方法复现重点是在这个父版本上补小输入DMA。
3. 真机任务表显示小输入Triton设备任务慢于PyTorch的MEMCPY_ASYNC；大输入连续Triton复制已接近PyTorch。该结论来自device_task，不应再解释成纯host/launch开销。
4. 对连续输入、dim归一化后为0、正有效length且输出元素数不超过65536的单段复制，可通过ctypes调用`libacl_rt.so`的`aclrtMemcpyAsync`。使用新输出的data_ptr、正确起点字节偏移、D2D枚举3、输入设备的当前npu_stream；所有字节数用element_size计算。非适用输入回退父Triton实现。
5. 库不可用可回退；DMA调用返回非零应报错，不能假装输出已完成。使用异步当前stream，不在候选内人为同步来改变测量。

## 计时与采集

使用KGS19657、device_task，正式计时同时包含DMA与AIV kernel，排除host发射间隙。PyTorch小输入任务kernel_name可能是N/A；专用FlagGems计时副本已有字符串列兼容补丁。不能换到19655并遗漏DMA。

DMA任务没有AIV PipeUtilization，不把缺少Vector/MTE指标解释成零利用率或host瓶颈。保存原始附件和逐case正式时间；纯Triton分支可分析实际JIT符号的流水指标。

历史尝试：
[]
当前逐case事实：
[{"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::0", "axes": {"dtype": "bfloat16", "shape_detail": [[64, 64], 0, 16, 32], "benchmark_level": "core"}, "timing": {"candidate_us": 0.5593333333333336, "pytorch_us": 0.5600000000000003, "speedup": 1.0011916820592266, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 4096, "output_elements": 2048, "flops": 0, "logical_bytes": 8192, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::1", "axes": {"dtype": "bfloat16", "shape_detail": [[256, 256], 0, 64, 128], "benchmark_level": "core"}, "timing": {"candidate_us": 0.662, "pytorch_us": 0.6573333333333334, "speedup": 0.9929517192770478, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 65536, "output_elements": 32768, "flops": 0, "logical_bytes": 131072, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::2", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 1024], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 2.611333333333333, "pytorch_us": 4.413333333333332, "speedup": 1.6900425053929515, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 1048576, "output_elements": 524288, "flops": 0, "logical_bytes": 2097152, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::3", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], 0, 1024, 2048], "benchmark_level": "core"}, "timing": {"candidate_us": 10.521333333333335, "pytorch_us": 10.796000000000003, "speedup": 1.0261054419074718, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 16777216, "output_elements": 8388608, "flops": 0, "logical_bytes": 33554432, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::4", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 65536], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 82.93700000000003, "pytorch_us": 89.93566666666666, "speedup": 1.0843852349149734, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 67108864, "output_elements": 33554432, "flops": 0, "logical_bytes": 134217728, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::0", "axes": {"dtype": "float16", "shape_detail": [[64, 64], 0, 16, 32], "benchmark_level": "core"}, "timing": {"candidate_us": 0.5606666666666669, "pytorch_us": 0.5786666666666669, "speedup": 1.032098912203531, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 4096, "output_elements": 2048, "flops": 0, "logical_bytes": 8192, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::1", "axes": {"dtype": "float16", "shape_detail": [[256, 256], 0, 64, 128], "benchmark_level": "core"}, "timing": {"candidate_us": 0.6586666666666667, "pytorch_us": 0.6553333333333334, "speedup": 0.994940039467692, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 65536, "output_elements": 32768, "flops": 0, "logical_bytes": 131072, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::2", "axes": {"dtype": "float16", "shape_detail": [[1024, 1024], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 2.6033333333333335, "pytorch_us": 4.3373333333333335, "speedup": 1.6660435578658954, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 1048576, "output_elements": 524288, "flops": 0, "logical_bytes": 2097152, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::3", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], 0, 1024, 2048], "benchmark_level": "core"}, "timing": {"candidate_us": 10.536, "pytorch_us": 11.096666666666666, "speedup": 1.0532138710402015, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 16777216, "output_elements": 8388608, "flops": 0, "logical_bytes": 33554432, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::4", "axes": {"dtype": "float16", "shape_detail": [[1024, 65536], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 82.54700000000001, "pytorch_us": 89.70566666666666, "speedup": 1.0867222066755422, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 67108864, "output_elements": 33554432, "flops": 0, "logical_bytes": 134217728, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::0", "axes": {"dtype": "float32", "shape_detail": [[64, 64], 0, 16, 32], "benchmark_level": "core"}, "timing": {"candidate_us": 0.6873333333333332, "pytorch_us": 0.67, "speedup": 0.9747854337390295, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 4096, "output_elements": 2048, "flops": 0, "logical_bytes": 16384, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::1", "axes": {"dtype": "float32", "shape_detail": [[256, 256], 0, 64, 128], "benchmark_level": "core"}, "timing": {"candidate_us": 0.882, "pytorch_us": 0.8819999999999999, "speedup": 0.9999999999999999, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 65536, "output_elements": 32768, "flops": 0, "logical_bytes": 262144, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::2", "axes": {"dtype": "float32", "shape_detail": [[1024, 1024], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 3.1866666666666665, "pytorch_us": 4.726666666666666, "speedup": 1.4832484336265597, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 1048576, "output_elements": 524288, "flops": 0, "logical_bytes": 4194304, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::3", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], 0, 1024, 2048], "benchmark_level": "core"}, "timing": {"candidate_us": 17.596666666666668, "pytorch_us": 18.005999999999997, "speedup": 1.023261849241243, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 16777216, "output_elements": 8388608, "flops": 0, "logical_bytes": 67108864, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::4", "axes": {"dtype": "float32", "shape_detail": [[1024, 65536], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 214.29799999999992, "pytorch_us": 217.72999999999996, "speedup": 1.016015074328704, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-replay/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 67108864, "output_elements": 33554432, "flops": 0, "logical_bytes": 268435456, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}]
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
"""Triton implementation of ``torch.narrow_copy(inp, dim, start, length)``.

For a plain row-major input the narrowed slab is ``pre = prod(shape[:dim])``
contiguous runs of ``chunk = length * prod(shape[dim+1:])`` elements and the
result is a freshly allocated contiguous tensor, so the operator is a pure block
copy:

    out[r * chunk + i] = inp[r * dim_size * post + start * post + i]

The previous candidate mapped every flat output index through two integer
divisions plus a modulo by *runtime* divisors.  Integer division on this target
is bit-serial and costs roughly 14 us of device time per 1024-element program,
so the kernel was ALU-bound at ~1.4 Gelem/s instead of bandwidth-bound; it also
split large launches into 8192-program slices, paying one extra kernel per
8.4M elements.  This implementation never divides per element:

* ``pre == 1`` -- the slab is one contiguous block, so the kernel walks a flat
  range with only a per-program block index.
* ``pre > 1`` -- one tile covers a BLOCK-sized piece of one run; the run index is
  derived once per tile with a single scalar division and each run is itself
  contiguous, so again there is no per-element index arithmetic.

Both use a grid-striding ``tl.range`` loop whose program count is capped by the
device's vector-core count, queried at runtime and never hard-coded, so each
program keeps a deep pipelined copy loop.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Element count at or below which a single program copies the whole slab.
_SINGLE_PROGRAM_ELEMS = 4096
# Element count at or below which a 4K tile is used instead of a 16K tile.
_MEDIUM_ELEMS = 65536
# Blocks of pipelined work targeted per program once the grid has to be capped.
_BLOCKS_PER_PROGRAM = 12
# Conservative cap used only when the vector-core count cannot be queried.
_FALLBACK_PROGRAM_CAP = 256
_MIN_BLOCK = 16

_VECTOR_CORES = None
_CORES_PROBED = False


def _vector_cores():
    """Vector core count of the current device, queried once (never hard-coded)."""
    global _VECTOR_CORES, _CORES_PROBED
    if not _CORES_PROBED:
        _CORES_PROBED = True
        try:
            from triton.runtime import driver as _driver

            props = _driver.active.utils.get_device_properties(
                torch.npu.current_device()
            )
            for key in ("num_vectorcore", "num_aicore"):
                value = None
                try:
                    value = props[key]
                except Exception:
                    value = getattr(props, key, None)
                if value:
                    _VECTOR_CORES = int(value)
                    break
        except Exception:  # pragma: no cover - property query is best effort
            _VECTOR_CORES = None
    return _VECTOR_CORES


@triton.jit
def narrow_copy_flat_kernel(
    out_ptr,
    inp_ptr,
    base_in,
    n_elements,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Copy one contiguous run of ``n_elements`` starting at ``base_in``."""
    grid = tl.num_programs(0)
    for block in tl.range(tl.program_id(0), nblocks, grid):
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if EVEN:
            tl.store(out_ptr + offsets, tl.load(inp_ptr + base_in + offsets))
        else:
            mask = offsets < n_elements
            tl.store(
                out_ptr + offsets,
                tl.load(inp_ptr + base_in + offsets, mask=mask),
                mask=mask,
            )


@triton.jit
def narrow_copy_runs_kernel(
    out_ptr,
    inp_ptr,
    base_in,
    chunk,
    src_stride,
    nblocks_per_run,
    ntiles,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Copy ``ntiles`` BLOCK-sized tiles of ``pre`` contiguous runs of ``chunk``."""
    grid = tl.num_programs(0)
    for tile in tl.range(tl.program_id(0), ntiles, grid):
        run = tile // nblocks_per_run
        block = tile - run * nblocks_per_run
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        src = inp_ptr + run * src_stride + base_in + offsets
        dst = out_ptr + run * chunk + offsets
        if EVEN:
            tl.store(dst, tl.load(src))
        else:
            mask = offsets < chunk
            tl.store(dst, tl.load(src, mask=mask), mask=mask)


def _plan_block(unit, block_hint):
    """Tile size, tile count and exactness flag for a contiguous run of ``unit``."""
    if unit <= _SINGLE_PROGRAM_ELEMS:
        block = triton.next_power_of_2(unit)
        if block < _MIN_BLOCK:
            block = _MIN_BLOCK
        elif block > _SINGLE_PROGRAM_ELEMS:
            block = _SINGLE_PROGRAM_ELEMS
    elif unit <= _MEDIUM_ELEMS:
        block = 4096
    else:
        block = 16384
    if block_hint and block > block_hint:
        block = block_hint
    n_blocks = (unit + block - 1) // block
    even = (unit % block) == 0
    return block, n_blocks, even


def _grid_for(n_blocks):
    """Program count: one program per block when they fit, else core-sized."""
    cores = _vector_cores()
    cap = cores if cores else _FALLBACK_PROGRAM_CAP
    if n_blocks <= cap:
        return n_blocks
    grid = n_blocks // _BLOCKS_PER_PROGRAM
    if grid < cap:
        grid = cap
    if grid > n_blocks:
        grid = n_blocks
    return grid


def narrow_copy(inp, dim, start, length):
    logger.debug("GEMS NARROW_COPY")
    assert (
        dim >= -inp.ndim and dim < inp.ndim
    ), f"Invalid dim: {dim} for tensor with {inp.ndim} dimensions"
    dim = dim % inp.ndim

    # Handle negative start
    if start < 0:
        start = start % inp.size(dim)

    # Clamp length to valid range
    length = min(length, inp.size(dim) - start)

    # Compute output shape
    out_shape = list(inp.shape)
    out_shape[dim] = length
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    if out.numel() == 0:
        return out

    # Make input contiguous for simpler indexing
    inp = inp.contiguous()

    dim_size = inp.size(dim)
    post = 1
    for d in range(dim + 1, inp.ndim):
        post *= inp.size(d)

    n_total = out.numel()
    chunk = length * post
    pre = n_total // chunk
    base_in = start * post

    if pre == 1:
        # Single contiguous slab: out[i] = inp[base_in + i]
        block, n_blocks, even = _plan_block(n_total, 0)
        narrow_copy_flat_kernel[(_grid_for(n_blocks),)](
            out,
            inp,
            base_in,
            n_total,
            n_blocks,
            BLOCK_SIZE=block,
            EVEN=even,
        )
    else:
        # `pre` contiguous runs of `chunk` elements, source stride dim_size * post
        block, n_blocks, even = _plan_block(chunk, 4096)
        narrow_copy_runs_kernel[(_grid_for(pre * n_blocks),)](
            out,
            inp,
            base_in,
            chunk,
            dim_size * post,
            n_blocks,
            pre * n_blocks,
            BLOCK_SIZE=block,
            EVEN=even,
        )

    return out


run = narrow_copy


# 小输入复用已验证的 CANN DMA，大输入保留 KG profiler 循环复制。
import ctypes as _ctypes

try:
    _acl_rt = _ctypes.CDLL("libacl_rt.so")
    _acl_rt.aclrtMemcpyAsync.argtypes = [
        _ctypes.c_void_p, _ctypes.c_size_t, _ctypes.c_void_p,
        _ctypes.c_size_t, _ctypes.c_int, _ctypes.c_void_p,
    ]
    _acl_rt.aclrtMemcpyAsync.restype = _ctypes.c_int
except OSError:
    _acl_rt = None

_triton_contiguous_copy = run


def run(inp, dim, start, length):
    # 只替换单段连续复制；一般维度和布局沿用父候选。
    if not (-inp.ndim <= dim < inp.ndim):
        return _triton_contiguous_copy(inp, dim, start, length)
    axis = dim % inp.ndim
    if _acl_rt is None or axis != 0 or not inp.is_contiguous():
        return _triton_contiguous_copy(inp, dim, start, length)
    size = inp.shape[0]
    normalized_start = start % size if start < 0 and size else start
    count = min(length, size - normalized_start)
    if count <= 0 or normalized_start < 0:
        return _triton_contiguous_copy(inp, dim, start, length)
    post = inp.numel() // size
    elements = count * post
    if elements > 65536:
        return _triton_contiguous_copy(inp, dim, start, length)
    with torch.npu.device(inp.device):
        out = inp.new_empty((count, *inp.shape[1:]))
        amount = elements * inp.element_size()
        source = inp.data_ptr() + normalized_start * post * inp.element_size()
        stream = torch.npu.current_stream(inp.device).npu_stream
        status = _acl_rt.aclrtMemcpyAsync(
            _ctypes.c_void_p(out.data_ptr()), amount,
            _ctypes.c_void_p(source), amount, 3, _ctypes.c_void_p(stream))
        if status != 0:
            raise RuntimeError(f"aclrtMemcpyAsync failed: {status}")
    return out


narrow_copy = run

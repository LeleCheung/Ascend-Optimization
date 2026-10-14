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

Triton启动grid必须为tuple或返回tuple的函数。例如`kernel[(grid,)](...)`，不能写`kernel[grid](...)`并传入整数。旧建议驱动批次第2轮已出现`TypeError: object of type 'int' has no len()`，在真正执行内核前失败，原结果保留。

只使用起点源码已有或知识明确确认的导入；不要为了新kernel添加无用兼容模块。`20261014-four-direct`的logaddexp2前两轮添加了`flag_gems.utils.triton_lang_utils`，但本机固定FlagGems没有这个模块，导致零用例执行。此处只需`torch`、`triton`、`triton.language`与起点已有的`pointwise_dynamic`，无需引入该模块或`CodeGenConfig`。

有界grid不能只写`min(cdiv(n, span), limit)`：截断grid后，kernel必须用grid-stride循环处理剩余数据。logaddexp2的正式正确性形状最大约787万元素，性能形状包含约10.7亿元素，不能假设小形状正确便证明大形状完整覆盖。循环次数很大时采用运行时`range`；`tl.static_range(64)`会展开64份exp/log和访存代码，与历史运行时64轮循环不是同一种实现。

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

## 首批生成错误与明确修复

`20261014-four-direct`没有复现DMA收益。第1轮把`aclrtMemcpyAsync`第二个参数误写成枚举3，返回107000；这是本次调用参数错误，不能据此否定已验证DMA方法。六个参数顺序为`dst, destMax, src, count, kind, stream`，ABI依次为`c_void_p, c_size_t, c_void_p, c_size_t, c_int, c_void_p`；`destMax`与`count`均为输出字节数，只有第五个参数为D2D枚举3。输入设备的当前stream用`torch.npu.current_stream(inp.device).npu_stream`，并进入该设备上下文。

第2轮将DMA替换为`out.copy_(view)`，适用条件又要求尾维全为1，实际没有命中普通dim=0的小输入；未落实上述CANN方案。不要继续用这个错误条件：连续输入、归一化dim=0、正有效length、输出元素数≤65536即可形成一段连续区域，源字节偏移为`normalized_start * prod(inp.shape[1:]) * inp.element_size()`。严格保留父Triton路径，不同时重写大输入分块。

第3轮的`tl.range(..., num_stages=3)`被本机编译器拒绝，日志为`Only range iterator supports num_stages <= 2`。本批优先只添加正确ABI的小输入DMA分流，其他分块与循环不变；不要重复尝试该流水参数。

## 计时与采集

使用KGS19657、device_task，正式计时同时包含DMA与AIV kernel，排除host发射间隙。PyTorch小输入任务kernel_name可能是N/A；专用FlagGems计时副本已有字符串列兼容补丁。不能换到19655并遗漏DMA。

DMA任务没有AIV PipeUtilization，不把缺少Vector/MTE指标解释成零利用率或host瓶颈。保存原始附件和逐case正式时间；纯Triton分支可分析实际JIT符号的流水指标。

历史尝试：
[{"round": 1, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/seed-evaluation/seed-1.result.json", "parent_score": 0.7981353104113093, "hypothesis": "本轮只改一个点：把「小输入连续复制」从 AIV Triton kernel 换成知识中已验证 ABI 的 CANN `aclrtMemcpyAsync` D2D 拷贝，其余分块、grid 策略与大输入内核完全不动。\n\n依据逐 case 事实：flat_kernel 在小形状上显著慢于 PyTorch——float16::0 的 8 KiB 复制，候选设备任务 2.46 us / kernel 2.35 us，而 PyTorch 只有 0.571 us（0.232×）；float16::1、bfloat16::0/1、float32::0/1 同样是 0.23~0.59×。这些 case 的输出只有 2048~32768 个元素、单段连续，PyTorch 走的正是 MEMCPY_ASYNC，而我们的 kernel 即使把整个 slab 放进一个 program，也仍要付一次 AIV kernel 的设备时间（2.07~2.35 us），任何分块/循环/流水调整都消不掉这笔启动成本。语义上这些 case 就是一次 D2D 拷贝，因此直接调用 CANN runtime 才是对症的改法。\n\n具体改动：新增 `_AclRuntime`（惰性 `ctypes.CDLL(\"libacl_rt.so\")`，缓存函数指针）与 `_try_acl_memcpy_d2d`。严格按知识记载的六参数 ABI 调用：`dst, destMax, src, count, kind, stream`；destMax 与 count 都传输出字节数，kind=3（DEVICE_TO_DEVICE），stream 用 `torch.npu.current_stream(inp.device).npu_stream`；源指针写入正确字节偏移 `base_in * item_size = start*post*element_size`（这里 dim 归一化后为 0 才走该路径，post = prod(shape[1:])）。不改用错误枚举位置、不改用 `out.copy_`。触发条件按知识修正为：`pre == 1`（连续输入、归一化 dim=0 的等价条件）、`n_total <= 65536`（有效 length 为正由 `out.numel() != 0` 与 length=min(...) 保证）。取消原 `tl.range(..., num_stages=3)` 一类流水实验，本机编译器只支持 ≤2，本轮不重复。\n\n稳健性：库不可用（OSError/AttributeError）返回 False，回落到原有 Triton flat 路径；`aclrtMemcpyAsync` 返回非 0 直接 raise，不吞异常、不假装完成；不调用 `torch.narrow_copy`、不返回 view；输出始终 `torch.empty` 独立分配。大输入（float32::4 的 214 us 级）与 runs 分支（pre>1）保持原实现，受益的正是 0.23~0.59× 的 6 个小 case 与 1.0× 附近的 float32::2/bfloat16::2 量级，几何平均应明显改善。", "expected_effect": "预期小形状 6 个 case 从 0.23~0.59× 提升到接近或超过 1×：候选设备时间应降至 PyTorch MEMCPY_ASYNC 同一量级（0.5~0.9 us），因为这些 case 的输出只有 4~64 KiB，一次 D2D 拷贝的设备时间由 DMA 引擎决定，不再包含 AIV kernel 的 2.07~2.35 us 固定开销。中大型 case（float16/bfloat16::3/4、float32::3/4 与 pre>1 分支）走原路径，预期维持现有 1.006~1.074×，不引入回退。风险点：若该 CANN 版本对不足一个 cacheline 对齐的源偏移（start*post*element_size）要求更严格，非对齐小 case 会在运行时报错而非静默出错——届时可观察到明确的 RuntimeError 错误码，便于下一轮收窄触发条件；若 libacl_rt 不可用则自动回落原 Triton 实现，结果与当前最佳持平。", "source_sha256": "2a6e306866509369b33bdfc9419010f97019e95b88c6999e661fe3c37525d070", "status": "PASSED", "score": 1.1249550656359035, "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "accepted": true}]
当前逐case事实：
[{"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::0", "axes": {"dtype": "bfloat16", "shape_detail": [[64, 64], 0, 16, 32], "benchmark_level": "core"}, "timing": {"candidate_us": 0.5606666666666669, "pytorch_us": 0.5626666666666669, "speedup": 1.0035665458003922, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 4096, "output_elements": 2048, "flops": 0, "logical_bytes": 8192, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::1", "axes": {"dtype": "bfloat16", "shape_detail": [[256, 256], 0, 64, 128], "benchmark_level": "core"}, "timing": {"candidate_us": 0.6633333333333334, "pytorch_us": 0.6673333333333333, "speedup": 1.0060292418228407, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 65536, "output_elements": 32768, "flops": 0, "logical_bytes": 131072, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::2", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 1024], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 2.6413333333333333, "pytorch_us": 4.361999999999999, "speedup": 1.6514140050225257, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 1048576, "output_elements": 524288, "flops": 0, "logical_bytes": 2097152, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::3", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], 0, 1024, 2048], "benchmark_level": "core"}, "timing": {"candidate_us": 11.551333333333332, "pytorch_us": 11.430000000000001, "speedup": 0.9894962529902496, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 16777216, "output_elements": 8388608, "flops": 0, "logical_bytes": 33554432, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": {"aiv_mte2_ratio": 0.696, "aiv_mte3_ratio": 0.6915, "aiv_scalar_ratio": 0.133, "aiv_vec_ratio": 0.001, "aic_mte2_ratio": 0.0, "aic_mte1_ratio": 0.0, "aic_scalar_ratio": 0.0, "aic_mac_ratio": 0.0, "cube_utilization_pct": 0.0, "kernel": "narrow_copy_flat_kernel", "device_duration_us": 14.03, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "现有指标不足以区分设备计算、搬运或 host 路径。", "next_experiment": "补充针对该 case 的流水线、访存或独立 host 分项采集。", "falsifier": "补充采集仍无法稳定复现同一受限类型。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::4", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 65536], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 75.51233333333333, "pytorch_us": 85.47566666666668, "speedup": 1.131942951558088, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 67108864, "output_elements": 33554432, "flops": 0, "logical_bytes": 134217728, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::0", "axes": {"dtype": "float16", "shape_detail": [[64, 64], 0, 16, 32], "benchmark_level": "core"}, "timing": {"candidate_us": 0.5606666666666669, "pytorch_us": 0.5600000000000002, "speedup": 0.9988111513998691, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 4096, "output_elements": 2048, "flops": 0, "logical_bytes": 8192, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::1", "axes": {"dtype": "float16", "shape_detail": [[256, 256], 0, 64, 128], "benchmark_level": "core"}, "timing": {"candidate_us": 0.6666666666666666, "pytorch_us": 0.6633333333333334, "speedup": 0.9950007498875171, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 65536, "output_elements": 32768, "flops": 0, "logical_bytes": 131072, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::2", "axes": {"dtype": "float16", "shape_detail": [[1024, 1024], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 2.6360000000000006, "pytorch_us": 4.335333333333334, "speedup": 1.644639176561334, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 1048576, "output_elements": 524288, "flops": 0, "logical_bytes": 2097152, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::3", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], 0, 1024, 2048], "benchmark_level": "core"}, "timing": {"candidate_us": 10.498666666666667, "pytorch_us": 11.144666666666668, "speedup": 1.0615310369789468, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 16777216, "output_elements": 8388608, "flops": 0, "logical_bytes": 33554432, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::4", "axes": {"dtype": "float16", "shape_detail": [[1024, 65536], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 75.82900000000001, "pytorch_us": 88.47966666666667, "speedup": 1.1668312912413132, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 67108864, "output_elements": 33554432, "flops": 0, "logical_bytes": 134217728, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::0", "axes": {"dtype": "float32", "shape_detail": [[64, 64], 0, 16, 32], "benchmark_level": "core"}, "timing": {"candidate_us": 0.6766666666666666, "pytorch_us": 0.6753333333333331, "speedup": 0.9980298478057427, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 4096, "output_elements": 2048, "flops": 0, "logical_bytes": 16384, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::1", "axes": {"dtype": "float32", "shape_detail": [[256, 256], 0, 64, 128], "benchmark_level": "core"}, "timing": {"candidate_us": 0.8926666666666666, "pytorch_us": 0.8873333333333336, "speedup": 0.9940260613075462, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 65536, "output_elements": 32768, "flops": 0, "logical_bytes": 262144, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::2", "axes": {"dtype": "float32", "shape_detail": [[1024, 1024], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 3.0973333333333333, "pytorch_us": 4.683999999999999, "speedup": 1.5122520796788737, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 1048576, "output_elements": 524288, "flops": 0, "logical_bytes": 4194304, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::3", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], 0, 1024, 2048], "benchmark_level": "core"}, "timing": {"candidate_us": 18.41533333333334, "pytorch_us": 19.05866666666667, "speedup": 1.0349344661995463, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 16777216, "output_elements": 8388608, "flops": 0, "logical_bytes": 67108864, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::4", "axes": {"dtype": "float32", "shape_detail": [[1024, 65536], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 216.19199999999998, "pytorch_us": 215.378, "speedup": 0.9962348300423559, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-refined-method/round-01/evaluation/trial-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 67108864, "output_elements": 33554432, "flops": 0, "logical_bytes": 268435456, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}]
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

The narrowed slab of a row-major input is ``pre = prod(shape[:dim])`` contiguous
runs of ``chunk = length * prod(shape[dim+1:])`` elements and the result is a
freshly allocated contiguous tensor, so the operator is a pure block copy:

    out[r * chunk + i] = inp[r * dim_size * post + start * post + i]

The parent version already removed the per-element integer division of the
original master kernel.  What it still pays for on the small shapes is the AIV
kernel itself: the device-task table shows ``narrow_copy_flat_kernel`` taking
2.35 us for the 8 KiB case and 2.07 us for the 128 KiB case, where the PyTorch
reference finishes the whole device task in ~0.57 us and ~0.67 us.  No tiling of
a single-block copy can recover that -- a kernel launch on this device costs
more than the copy.

The operator semantics are exactly a device-to-device memcpy for the common
case (contiguous input, normalised ``dim == 0``, positive effective length),
because the selected region is one contiguous byte span and the output is an
independent contiguous buffer.  So for such inputs we issue a real CANN
``aclrtMemcpyAsync`` on the input device's current stream, using the verified
ABI:

    aclrtMemcpyAsync(dst, destMax, src, count, kind, stream)
    c_void_p, c_size_t, c_void_p, c_size_t, c_int, c_void_p

with ``kind = 3`` (ACL_MEMCPY_DEVICE_TO_DEVICE), ``destMax == count ==`` output
bytes, and the source byte offset ``normalized_start * prod(shape[1:]) *
item_size``.  The library handle is resolved lazily and cached; if libacl_rt is
unavailable the code falls back to the Triton kernels below, and a non-zero
return code from the copy is raised instead of being silently ignored.

Every other layout -- non-contiguous input, ``dim != 0``, negative ``start``
that survives normalisation, empty output -- keeps the parent Triton path with
its per-run / per-tile index arithmetic and grid-striding loops, unchanged.
"""

import ctypes
import logging

import torch
import triton
import triton.language as tl

try:  # pragma: no cover - torch_npu is always present on this target
    import torch_npu  # noqa: F401
    _HAS_TORCH_NPU = True
except Exception:  # pragma: no cover
    _HAS_TORCH_NPU = False

logger = logging.getLogger(__name__)

# --- small-copy DMA fast path ---------------------------------------------
# The slab is a single contiguous span and the output is independent, so the
# region can be handled by the CANN runtime's D2D memcpy.  Limit matches the
# historically verified threshold.
_DMA_MAX_ELEMS = 65536
_ACL_MEMCPY_DEVICE_TO_DEVICE = 3
_ACL_SUCCESS = 0
_ACL_LIB_CANDIDATES = ("libacl_rt.so", "libacl_rt.so.1")


class _AclRuntime(object):
    """Lazily resolved handle onto the CANN runtime's async memcpy."""

    _instance = None

    def __init__(self):
        self.memcpy_async = None
        self.lib = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self):
        for name in _ACL_LIB_CANDIDATES:
            try:
                lib = ctypes.CDLL(name)
                fn = lib.aclrtMemcpyAsync
            except (OSError, AttributeError):
                continue
            fn.restype = ctypes.c_int
            fn.argtypes = [
                ctypes.c_void_p,  # dst
                ctypes.c_size_t,  # destMax
                ctypes.c_void_p,  # src
                ctypes.c_size_t,  # count
                ctypes.c_int,     # kind
                ctypes.c_void_p,  # stream
            ]
            self.lib = lib
            self.memcpy_async = fn
            return
        self.memcpy_async = None


# --- Triton fallback path --------------------------------------------------
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


def _try_acl_memcpy_d2d(out, inp, byte_offset, nbytes):
    """Device-to-device memcpy of an independent contiguous slab.

    Returns ``True`` when the copy was issued on the current stream, ``False``
    when the CANN runtime is unavailable so the caller must use Triton, and
    raises when the runtime rejects the copy (so a failed copy can never be
    mistaken for a completed one).
    """
    rt = _AclRuntime.instance()
    if rt.memcpy_async is None or not _HAS_TORCH_NPU:
        return False
    stream = torch.npu.current_stream(inp.device).npu_stream
    ret = rt.memcpy_async(
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_size_t(nbytes),
        ctypes.c_void_p(inp.data_ptr() + byte_offset),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(_ACL_MEMCPY_DEVICE_TO_DEVICE),
        ctypes.c_void_p(stream),
    )
    if ret != _ACL_SUCCESS:
        raise RuntimeError("aclrtMemcpyAsync (D2D) failed with error code %d" % ret)
    return True


# Historical fallback for the DMA-candidate's small-copy contract: a plain
# tensor copy of the contiguous 1-D view.  Only reachable when acl_rt is
# missing on the host; it never shadows the CANN path above.
def _copy_via_triton_fallback(out, inp, base_in, n_total):
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
    item_size = inp.element_size()

    # Fast path: contiguous input narrowed along dim 0 is one contiguous byte
    # span in the source and one contiguous buffer at the destination, i.e. a
    # plain device-to-device memcpy.  This is the case the device-task table
    # shows losing to the PyTorch reference on every small shape.
    if pre == 1 and n_total <= _DMA_MAX_ELEMS:
        if _try_acl_memcpy_d2d(out, inp, base_in * item_size, n_total * item_size):
            return out

    if pre == 1:
        # Single contiguous slab: out[i] = inp[base_in + i]
        _copy_via_triton_fallback(out, inp, base_in, n_total)
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

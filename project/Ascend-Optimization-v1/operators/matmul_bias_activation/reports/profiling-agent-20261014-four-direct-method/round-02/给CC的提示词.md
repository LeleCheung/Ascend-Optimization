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

算子：matmul_bias_activation
架构与既有经验：
# 四算子工作流通用知识

执行环境为Ascend 910B4-1、CANN 9.0.0、Python 3.11.15、Triton 3.5.1、Flagtree 0.6.0。只使用物理卡7，同卡串行；设备任务由KGS执行，模型只返回候选源码。

本批验证历史方法能否再次产生有效修改。按各算子知识中的已验证方法改进明确的优化前版本，不要求重新发明算法。保留所有接口、dtype、shape、stride及特殊值语义；完整正确性与性能合同决定是否采纳。

当前Ascend后端支持num_warps、num_stages、multibuffer、unit_flag；语义不能照搬CUDA。JIT内部常量用字面量或constexpr。每次launch使用有界grid和循环覆盖，不能生成百万program。

Triton启动grid必须为tuple或返回tuple的函数。例如`kernel[(grid,)](...)`，不能写`kernel[grid](...)`并传入整数。旧建议驱动批次第2轮已出现`TypeError: object of type 'int' has no len()`，在真正执行内核前失败，原结果保留。

真机PipeUtilization是流水活跃比例，可以重叠，不是HBM带宽利用率。逻辑读写字节只描述算法工作量。当前没有可信的物理计数与硬件roof，不输出数值Roofline。

依据逐case时间、源码与真机流水选择修改；保留实际生成理由、失败及最终源码。已有稳定回退可以保留。编译错误与运行错误须区分，不允许用吞掉执行异常代替正确性保证。

三个计算算子使用device_kernel；narrow_copy使用device_task，覆盖真实DMA与kernel。加速比为PyTorch时间除以候选时间，性能case取几何平均，最终报告采用同源码两次完整复验的较慢值。


# matmul_bias_activation优化知识

## 起点与证据

方法复现从`candidates/profiling-dual-acc-v9-20261012.py`开始，是已优化中间版本，非master。历史`reports/profiling-agent-20261012-agent/`第3、5轮采纳，最终`candidates/profiling-agent-final-20261012-agent.py`完整42/42通过，两次0.908×/0.906×。

## 瓶颈到修改

1. 矩阵乘使用AIC Cube，bias和ReLU使用AIV。结合AIC MTE2、Scalar和MAC以及逐case时间，检查K循环控制、重复搬运和核覆盖；这些指标不能转换为未测量的HBM带宽。
2. 双N tile方案在同一次K循环中加载一次A，与相邻两个B tile做两路累积，复用A并减少循环与寻址成本。历史有效修改把该方案扩展到FP32规则方阵，重点是1024、2048、4096方阵，按公开形状和dtype分流。
3. 最终配置FP32采用BK128，半精度采用BK256。保持高精度累积与最终转换；不能为速度丢掉BF16大K情况下的稳定路径。
4. 起点已有半精度双tile、pipeline及master回退。新增FP32双tile路径后，清理被覆盖的旧分流，保留不规则形状、stride、bias广播与一般路径。
5. ReLU保留本机可编译的`maximum(x, x*0.0)`形式；不能默认替换成标量常量就一定可编译且语义相同。

## 失败方向

历史第1、2、4轮失败，第6轮半精度1024/2048的BK128候选通过但整体下降（约0.868×），没有采纳。优先复现FP32双N tile复用A与BK128，半精度沿用BK256，不再次把失败方向当作默认最佳配置。

历史尝试：
[{"round": 1, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "parent_score": 0.8707021663617351, "hypothesis": "本轮只改一处：把 v8 的“一次 K 步复用 A 面板、并行计算两个相邻 N tile”的双 N tile 结构从半精度扩展到 fp32 规则方阵（1024/2048/4096）。当前源码里 fp32 全部落到 v6（单 N tile、GROUP=4/GROUP=0 的 mba_selected_profiling_kgroup_v6_20261010_kernel），这正是三个 fp32 方阵明显落后的原因：1024→69.95us(0.717×)、2048→357.6us(0.792×)、4096→2326.0us(0.979×)，而 fp16 同形状已经是 30.0/132.3/790.5us，说明 fp32 路径的循环与寻址开销没有按同样的方式被摊薄。具体修改：新增 mba_dual_ntile_v9_kernel，保留 v8 的双累加器结构（acc0/acc1）、每次 K 迭代只加载一次 A、两个 B tile，并在规则形状下走无 mask 的 EVEN 分支；fp32 使用 BM=BN=128、BK=128（fp32 元素 4 字节，BK=256 的 A 面板字节数是 fp16 的两倍，按历史已采纳的 fp32 K 分块保持 128；用 min(256, 512//element_size) 的规则表达），半精度仍然转发到已验证的 v8，参数完全不变，因此 fp16/bf16 的 6 个 case 行为逐字节一致。grid 为 (m/128)*(n/256)，4096 时为 32*16=512 个 program，远小于 65535。预期 1024 fp32 与 2048 fp32 受益最大（A 复用把每 K 步的 A 搬运与地址计算减半，MTE2 与 scalar 开销同步下降），4096 fp32 已有 0.979×，回收的地址计算与 MTE2 压力应使其回到 1.0× 以上。风险：如果 fp32 的 cube 受限主要来自 MAC 而非取数，增益会变小；此时 v8 半精度路径与全部回退路径不变，不会比当前回归。", "expected_effect": "预期 fp32 三个规则方阵 case 一起改善：1024 从 69.95us（0.717×）降到 55-60us 区间（约 0.85-0.91×），2048 从 357.6us（0.792×）降到 300us 以内（约 0.94× 以上），4096 从 2326us（0.979×）提升到 1.0× 以上。384 小方阵与所有非方阵、非连续、bias 非一维形状仍走原 v6/v4/master 回退，行为不变。fp16/bf16 的 6 个 case 因转发到未修改的 v8 路径，时间应保持在当前水平（384→9.9us、1024→30.0us、2048→132.3us、4096→790.5us 附近）。整体几何平均预期从当前约 0.82× 提升到 0.85× 以上，且不牺牲任何已验证通过的路径。", "source_sha256": "210f603fd20c2d6c9bf2ae51dc79fa0e6936176ecbb6688bd1b49cddb35e2146", "status": "PASSED", "score": 0.796492681843334, "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/round-01/evaluation/trial-1.result.json", "accepted": false}]
当前逐case事实：
[{"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::0", "axes": {"dtype": "bfloat16", "shape_detail": [[384, 384], [384, 384], [384]], "benchmark_level": "core"}, "timing": {"candidate_us": 9.851533333333332, "pytorch_us": 12.000866666666667, "speedup": 1.2181702526484612, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 384, "n": 384, "k": 384, "flops": 113246208, "logical_bytes": 885504, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::1", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 792.7811666666666, "pytorch_us": 642.7941999999999, "speedup": 0.8108091437411193, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 100671488, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::2", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 1024], [1024, 1024], [1024]], "benchmark_level": "core"}, "timing": {"candidate_us": 30.152633333333334, "pytorch_us": 24.719266666666666, "speedup": 0.8198051696805817, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 1024, "n": 1024, "k": 1024, "flops": 2147483648, "logical_bytes": 6293504, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::3", "axes": {"dtype": "bfloat16", "shape_detail": [[2048, 2048], [2048, 2048], [2048]], "benchmark_level": "core"}, "timing": {"candidate_us": 131.66933333333333, "pytorch_us": 99.6473, "speedup": 0.7567997938271172, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 2048, "n": 2048, "k": 2048, "flops": 17179869184, "logical_bytes": 25169920, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::4", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 791.2918000000001, "pytorch_us": 640.1081333333333, "speedup": 0.8089407124391558, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 100671488, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::0", "axes": {"dtype": "float16", "shape_detail": [[384, 384], [384, 384], [384]], "benchmark_level": "core"}, "timing": {"candidate_us": 9.920200000000001, "pytorch_us": 11.634299999999998, "speedup": 1.1727871132929446, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 384, "n": 384, "k": 384, "flops": 113246208, "logical_bytes": 885504, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::1", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 792.8918666666667, "pytorch_us": 622.8644666666668, "speedup": 0.7855604456243913, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 100671488, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::2", "axes": {"dtype": "float16", "shape_detail": [[1024, 1024], [1024, 1024], [1024]], "benchmark_level": "core"}, "timing": {"candidate_us": 30.039233333333335, "pytorch_us": 23.529133333333334, "speedup": 0.7832808096051843, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 1024, "n": 1024, "k": 1024, "flops": 2147483648, "logical_bytes": 6293504, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::3", "axes": {"dtype": "float16", "shape_detail": [[2048, 2048], [2048, 2048], [2048]], "benchmark_level": "core"}, "timing": {"candidate_us": 132.25799999999998, "pytorch_us": 95.58923333333334, "speedup": 0.7227484239780653, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 2048, "n": 2048, "k": 2048, "flops": 17179869184, "logical_bytes": 25169920, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": {"aiv_mte2_ratio": 0.0375, "aiv_mte3_ratio": 0.038, "aiv_scalar_ratio": 0.0625, "aiv_vec_ratio": 0.1145, "aic_mte2_ratio": 0.8315, "aic_mte1_ratio": 0.6575, "aic_scalar_ratio": 0.2475, "aic_mac_ratio": 0.685, "cube_utilization_pct": 74.004, "kernel": "mba_dual_dispatch_v8_kernel", "device_duration_us": 131.012, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "gemm_data_pipeline", "confidence": "medium", "inference": "矩阵乘的 AIC MTE2 活跃比例为 83.2%，Scalar 为 0.2475。优先增加 K 分块、减少循环和地址计算，再重叠搬运与计算。流水线比例不等于 HBM 带宽利用率，各比例不能相加。", "next_experiment": "增大 K 分块至 128/256，按 shape 调整 M/N tile，测试 multibuffer/unit_flag；完整正确性与所有 timing case 复验。", "falsifier": "MTE2 和总延迟不改善，或新增分块导致精度失败。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::4", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 790.4864666666666, "pytorch_us": 619.0277333333332, "speedup": 0.7830972206696153, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 100671488, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::0", "axes": {"dtype": "float32", "shape_detail": [[384, 384], [384, 384], [384]], "benchmark_level": "core"}, "timing": {"candidate_us": 11.404200000000001, "pytorch_us": 12.969566666666669, "speedup": 1.137261091576569, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 384, "n": 384, "k": 384, "flops": 113246208, "logical_bytes": 1771008, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::1", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 2326.0272000000004, "pytorch_us": 2277.221566666667, "speedup": 0.9790176008109047, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 201342976, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::2", "axes": {"dtype": "float32", "shape_detail": [[1024, 1024], [1024, 1024], [1024]], "benchmark_level": "core"}, "timing": {"candidate_us": 69.9481, "pytorch_us": 50.13766666666666, "speedup": 0.7167842298539013, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 1024, "n": 1024, "k": 1024, "flops": 2147483648, "logical_bytes": 12587008, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": {"aiv_mte2_ratio": 0.0175, "aiv_mte3_ratio": 0.019, "aiv_scalar_ratio": 0.1055, "aiv_vec_ratio": 0.0585, "aic_mte2_ratio": 0.552, "aic_mte1_ratio": 0.2825, "aic_scalar_ratio": 0.127, "aic_mac_ratio": 0.6325, "cube_utilization_pct": 71.7355, "kernel": "mba_wide_pipeline_kernel", "device_duration_us": 70.261, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "现有指标不足以区分设备计算、搬运或 host 路径。", "next_experiment": "补充针对该 case 的流水线、访存或独立 host 分项采集。", "falsifier": "补充采集仍无法稳定复现同一受限类型。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::3", "axes": {"dtype": "float32", "shape_detail": [[2048, 2048], [2048, 2048], [2048]], "benchmark_level": "core"}, "timing": {"candidate_us": 357.6218333333333, "pytorch_us": 283.2923333333333, "speedup": 0.7921562044386167, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 2048, "n": 2048, "k": 2048, "flops": 17179869184, "logical_bytes": 50339840, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::4", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 2325.9998333333333, "pytorch_us": 2277.4528666666665, "speedup": 0.9791285605941118, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 201342976, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}]
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

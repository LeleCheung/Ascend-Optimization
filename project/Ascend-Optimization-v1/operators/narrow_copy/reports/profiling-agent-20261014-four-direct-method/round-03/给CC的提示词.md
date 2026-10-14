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
[{"round": 1, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "parent_score": 0.8106880512187921, "hypothesis": "本轮只改一处结构性瓶颈：上一版把 runs 分支的 tile 上限定死为 4096，并且 grid 上限只取设备核数，导致 (1) 4096x4096/1024x65536 两个大 stride shape 的 tile 数是实际需要的 2-4 倍，(2) 16M 元素的 shape 只拿到核数量级的 program，每个 program 串行跑十几个 tile，尾部和外层循环开销没被摊掉。修改点：(a) 新增 _tile_hint(total_elements)，按总拷贝量决定 tile 上限，小体量仍 4096，超过 65536 元素允许 16384，runs kernel 的 BLOCK_SIZE 由 4096 提升为 16384（EVEN 判定同步重算），使这两类 shape 的 tile 数量减半；(b) _grid_for 的 cap 从 cores 改为 cores*4（_GRID_OVERSUBSCRIBE=4），并把 n_blocks<=cap 时的直接返回提前，让 <=64K 元素的小 launch 完全不跨步、程序数就是 tile 数；(c) 补充小输入 D2D 快速路径：输入连续、dim 归一化后为 0、有效 length>0 且输出元素数不超过 32768（≤64KiB）时，用 ctypes 调 libacl_rt.so 的 aclrtMemcpyAsync，D2D 枚举 3、源地址加 start*post*element_size 字节偏移、目标为新建的连续 out、stream 取输入设备当前 npu_stream，返回非零直接抛错而不是假装成功；不满足条件（非连续、负 dim、超限、库缺失）一律走原 Triton 通用回退。合同、dtype、shape/stride、空输出与负 start/负 dim 语义全部保留，grid 仍受 65535 约束。", "expected_effect": "小 shape（64x64、256x256 的 6 个 case，现 1.20-2.44us vs PyTorch 0.56-0.89us，几何平均仅 0.4x）改由单次 D2D DMA 完成，预期设备任务时间落到 0.6-0.9us 量级，把这 6 个 case 从 0.23-0.57x 拉回 0.9-1.1x；1024x1024（0/1 两分支，现 1.5-1.7x）因 grid 不再被核数卡住、tile 数减少，预期小幅提升到 1.7-1.9x；4096x4096 与 1024x65536（现 1.01-1.14x，bfloat16/float32 的 1024x65536 为 0.99-1.13x）因 tile 数减半、grid 4 倍过订阅，预期提升到 1.1-1.3x。整体 15 个 case 的几何平均应从当前约 0.9x 提升到 1.05-1.15x。所有分支的空输出、负 start、负 dim、非连续输入行为不变，DMA 仅在严格可控条件下启用且失败即回退或报错。", "source_sha256": "817e62a807c61fca195b661f1f1347fda2d241bdeb7e3fa631e9a56b9c18c65f", "status": "FAILED", "error": "命令失败 1，见 /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/round-01/evaluation.log", "evaluation_status": "RUNTIME_ERROR", "failure_detail": "     before = registry.call_counts().get(key, 0) if overridden else 0\n                with dispatch:\n                    metric.latency = self.get_latency(op, *args, **kwargs)\n                if overridden:\n                    if registry.call_counts().get(key, 0) <= before:\n                        raise RuntimeError(\n                            \"Benchmark did not invoke the injected candidate\"\n                        )\n                    metric.candidate_source = \"override\"\n            if \"speedup\" in self.to_bench_metrics:\n                if Config.skip_native:\n                    if metric.latency_base is not None and metric.latency is not None:\n                        metric.speedup = (metric.latency_base + self.EPS) / (\n                            metric.latency + self.EPS\n                        )\n                else:\n                    metric.speedup = (metric.latency_base + self.EPS) / (\n                        metric.latency + self.EPS\n                    )\n    \n            if \"gbps\" in self.to_bench_metrics:\n                if Config.skip_native:\n                    if metric.latency_base is not None:\n                        metric.gbps_base = self.get_gbps(\n                            args, latency=metric.latency_base\n                        )\n                    if metric.latency is not None:\n                        metric.gbps = self.get_gbps(args, latency=metric.latency)\n                else:\n                    metric.gbps_base = self.get_gbps(args, latency=metric.latency_base)\n                    metric.gbps = self.get_gbps(args, latency=metric.latency)\n    \n            if \"tflops\" in self.to_bench_metrics:\n                metric.tflops = (\n                    self.get_tflops(self.torch_op, *args, **kwargs)\n                    / metric.latency\n                    / 1e12\n                    * 1e3\n                )\n                # utilization = metric.tflops / metric.latency / 1e12 * 1e3\n        except (RuntimeError, Exception) as e:\n            metric.error_msg = str(e)\n>           pytest.fail(str(e))  # raise exception again\n            ^^^^^^^^^^^^^^^^^^^\nE           Failed: aclrtMemcpyAsync failed with code 107000\n\nbenchmark/base.py:801: Failed\n----------------------------- Captured stdout call -----------------------------\n[2026-10-14 19:00:56] [WARNING] [1131694] profiler.py: Incorrect schedule: Stop profiler while current state is RECORD which may result in incomplete parsed data.\n[2026-10-14 19:00:56] [INFO] [1132172] profiler.py: Start parsing profiling data in sync mode at: /root/.triton/profile_results/prof_20261014_110056_MainProcess-1131694/bm-ctyun-wq-zone1-d-910b4-64g-0-8_1131694_20261014190056260_ascend_pt\n|\b/\b-\b[2026-10-14 19:00:59] [INFO] [1132184] profiler.py: CANN profiling data parsed in a total time of 0:00:02.040321\n\\\b[2026-10-14 19:01:00] [INFO] [1132172] profiler.py: All profiling data parsed in a total time of 0:00:03.260139\n=============================== warnings summary ===============================\n../../../../../usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/npu/__init__.py:539\n  /usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/npu/__init__.py:539: UserWarning: You can set the device capability via the environment variable TORCH_NPU_DEVICE_CAPABILITY. The return value of get_device_capability is only for compatibility with PyTorch and does not represent the actual capability of the NPU device.\n    warnings.warn(f\"You can set the device capability via the environment variable TORCH_NPU_DEVICE_CAPABILITY. {warning_str}\")\n\nsrc/flag_gems/utils/device_info.py:58\n  /data/hanle/ascend-optimization/runtime/flaggems-narrow-device/src/flag_gems/utils/device_info.py:58: UserWarning: [device_info] torch_device_fn.get_device_capability returned None for device_id=0, fallback to (0, 0).\n    warnings.warn(\n\n../../../../../usr/local/python3.11.15/lib/python3.11/site-packages/torch/backends/__init__.py:46\n  /usr/local/python3.11.15/lib/python3.11/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)\n    self.setter(val)\n\n-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n=========================== short test summary info ============================\nFAILED benchmark/test_narrow_copy.py::test_narrow_copy_perf - Failed: aclrtMe...\n1 failed, 3 warnings in 6.42s\n[W1014 19:01:01.162798482 ToKernelNpu.cpp:41] Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())\n", "failed_cases": []}, {"round": 2, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "parent_score": 0.8106880512187921, "hypothesis": "第1轮失败的直接原因是 _copy_with_device_dma 里用 ctypes 调 libacl_rt.so 的 aclrtMemcpyAsync 并传入手工拼出来的 npu_stream，返回码 107000（参数非法），整个 perf 用例因此 pytest.fail。也就是说那条路径的参数组装（stream 句柄或地址/长度）不对，不是算法问题。本轮不再碰 acl_rt，改用 PyTorch 自身的设备拷贝原语：新增 _copy_with_device_dma()，仅当满足严格结构条件时用 inp.reshape(-1).narrow(0, start*out.numel(), out.numel()) 构造一个不拷贝数据的连续视图，再 out.copy_(view) 一次完成 D2D，然后把新输出返回；条件不满足（非连续、storage_offset!=0、dim 后面存在 size>1 的维度导致不是单段、start<0、字节数>4096）一律返回 False 走原 Triton 分支。这套条件是按当前源码 narrow_copy_runs_kernel 的语义反推的：post==out.numel()/length 且 pre==1 时，切片在底层存储里就是 start*post 起的 out.numel() 个连续元素，所以视图偏移写成 start*out.numel() 正好等价，不需要再做 stride 乘法。同时把 _grid_for 收口到 1..65535，消除了第1轮核数×块数在大 shape 上无上限的风险。不改 narrow_copy_flat_kernel / narrow_copy_runs_kernel 的签名与分块（4096/16384、_BLOCKS_PER_PROGRAM=12 全部保留），不改 dtype/shape/stride/负 dim/负 start/空输出语义，不删用例。", "expected_effect": "第1轮在性能用例上直接抛异常、不可用；本轮把整条脆弱的 ctypes/DMA 路径删掉，评测不会再因 107000 中断，回到稳定的 Triton 主干，因此至少恢复 seed 的 0.81x 基线。真正受益的是 6 个最小 case（64x64 各 dtype，候选 1.20-2.44us vs PyTorch 0.56-0.70us，仅 0.23-0.57x）与 3 个 256x256 的 bf16/fp16 case（1.46us vs 0.67us，0.46x）：它们逻辑字节只有 8-131KiB，全被 AIV launch 固定开销主导，改走设备拷贝引擎后不再起 kernel，预期落到 0.6-0.9us 量级、逐 case 回到 0.9-1.1x。256x256 的 float32 是 128KiB，超出 4096 字节阈值仍走 Triton，预期与现在持平（0.57x 附近），不假装改善。1024x1024、4096x4096、1024x65536 六个大 case（现 0.99-1.67x）分块与 grid 策略未改动，预期持平；其中 1024x65536 float32 的 0.9965x 仍是几何平均的主要拖累项，本轮不动它以免引入未验证的回归。综合预期几何平均从 seed 的 0.81x 提升到 0.95-1.05x，且不再有 runtime error 型失败。所有分支的空输出、负 start、负 dim、非连续输入行为保持不变，DMA 分支失败即静默回退 Triton。", "source_sha256": "0899f03320a8378c380faf3d8f850e598438b8e85af6743f8adf40ea7d4a9c94", "status": "PASSED", "score": 0.8063732289206017, "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/round-02/evaluation/trial-1.result.json", "accepted": false}]
当前逐case事实：
[{"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::0", "axes": {"dtype": "bfloat16", "shape_detail": [[64, 64], 0, 16, 32], "benchmark_level": "core"}, "timing": {"candidate_us": 1.1966666666666665, "pytorch_us": 0.5620000000000002, "speedup": 0.469682199259115, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 4096, "output_elements": 2048, "flops": 0, "logical_bytes": 8192, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::1", "axes": {"dtype": "bfloat16", "shape_detail": [[256, 256], 0, 64, 128], "benchmark_level": "core"}, "timing": {"candidate_us": 1.4633333333333334, "pytorch_us": 0.6699999999999999, "speedup": 0.45789581577568716, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 65536, "output_elements": 32768, "flops": 0, "logical_bytes": 131072, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": {"aiv_mte2_ratio": 0.1635, "aiv_mte3_ratio": 0.169, "aiv_scalar_ratio": 0.1475, "aiv_vec_ratio": 0.008, "aic_mte2_ratio": 0.0, "aic_mte1_ratio": 0.0, "aic_scalar_ratio": 0.0, "aic_mac_ratio": 0.0, "cube_utilization_pct": 0.0, "kernel": "narrow_copy_flat_kernel", "device_duration_us": 2.06, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "现有指标不足以区分设备计算、搬运或 host 路径。", "next_experiment": "补充针对该 case 的流水线、访存或独立 host 分项采集。", "falsifier": "补充采集仍无法稳定复现同一受限类型。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::2", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 1024], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 2.6086666666666662, "pytorch_us": 4.358666666666666, "speedup": 1.6708150722563666, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 1048576, "output_elements": 524288, "flops": 0, "logical_bytes": 2097152, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::3", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], 0, 1024, 2048], "benchmark_level": "core"}, "timing": {"candidate_us": 10.485999999999997, "pytorch_us": 11.118, "speedup": 1.060270262538027, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 16777216, "output_elements": 8388608, "flops": 0, "logical_bytes": 33554432, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::bfloat16::4", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 65536], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 93.84499999999998, "pytorch_us": 105.7, "speedup": 1.1263251890615495, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 67108864, "output_elements": 33554432, "flops": 0, "logical_bytes": 134217728, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::0", "axes": {"dtype": "float16", "shape_detail": [[64, 64], 0, 16, 32], "benchmark_level": "core"}, "timing": {"candidate_us": 2.4420000000000006, "pytorch_us": 0.558666666666667, "speedup": 0.2288058092079222, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 4096, "output_elements": 2048, "flops": 0, "logical_bytes": 8192, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": {"aiv_mte2_ratio": 0.1575, "aiv_mte3_ratio": 0.104, "aiv_scalar_ratio": 0.3545, "aiv_vec_ratio": 0.0055, "aic_mte2_ratio": 0.0, "aic_mte1_ratio": 0.0, "aic_scalar_ratio": 0.0, "aic_mac_ratio": 0.0, "cube_utilization_pct": 0.0, "kernel": "narrow_copy_flat_kernel", "device_duration_us": 2.41, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "现有指标不足以区分设备计算、搬运或 host 路径。", "next_experiment": "补充针对该 case 的流水线、访存或独立 host 分项采集。", "falsifier": "补充采集仍无法稳定复现同一受限类型。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::1", "axes": {"dtype": "float16", "shape_detail": [[256, 256], 0, 64, 128], "benchmark_level": "core"}, "timing": {"candidate_us": 1.4600000000000006, "pytorch_us": 0.6693333333333332, "speedup": 0.45848457868182513, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 65536, "output_elements": 32768, "flops": 0, "logical_bytes": 131072, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::2", "axes": {"dtype": "float16", "shape_detail": [[1024, 1024], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 2.6406666666666667, "pytorch_us": 4.345333333333333, "speedup": 1.6455196092043978, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 1048576, "output_elements": 524288, "flops": 0, "logical_bytes": 2097152, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::3", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], 0, 1024, 2048], "benchmark_level": "core"}, "timing": {"candidate_us": 11.429999999999998, "pytorch_us": 11.568000000000001, "speedup": 1.0120733851847319, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 16777216, "output_elements": 8388608, "flops": 0, "logical_bytes": 33554432, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::4", "axes": {"dtype": "float16", "shape_detail": [[1024, 65536], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 93.56766666666667, "pytorch_us": 106.70866666666667, "speedup": 1.140443664182074, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 67108864, "output_elements": 33554432, "flops": 0, "logical_bytes": 134217728, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::0", "axes": {"dtype": "float32", "shape_detail": [[64, 64], 0, 16, 32], "benchmark_level": "core"}, "timing": {"candidate_us": 1.2193333333333336, "pytorch_us": 0.6966666666666668, "speedup": 0.571385616269852, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 4096, "output_elements": 2048, "flops": 0, "logical_bytes": 16384, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::1", "axes": {"dtype": "float32", "shape_detail": [[256, 256], 0, 64, 128], "benchmark_level": "core"}, "timing": {"candidate_us": 1.5620000000000003, "pytorch_us": 0.8853333333333334, "speedup": 0.566822439877942, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 65536, "output_elements": 32768, "flops": 0, "logical_bytes": 262144, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::2", "axes": {"dtype": "float32", "shape_detail": [[1024, 1024], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 3.098666666666667, "pytorch_us": 4.7040000000000015, "speedup": 1.5180555704957892, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 1048576, "output_elements": 524288, "flops": 0, "logical_bytes": 4194304, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::3", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], 0, 1024, 2048], "benchmark_level": "core"}, "timing": {"candidate_us": 17.618000000000002, "pytorch_us": 18.066000000000006, "speedup": 1.0254283946623077, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 16777216, "output_elements": 8388608, "flops": 0, "logical_bytes": 67108864, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float32::4", "axes": {"dtype": "float32", "shape_detail": [[1024, 65536], 0, 256, 512], "benchmark_level": "core"}, "timing": {"candidate_us": 219.3106666666667, "pytorch_us": 218.54333333333335, "speedup": 0.9965011597697817, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/narrow_copy/reports/profiling-agent-20261014-four-direct-method/seed-evaluation/seed-1.result.json", "scope": "设备任务计时，包括 DMA 与 kernel"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D", "input_elements": 67108864, "output_elements": 33554432, "flops": 0, "logical_bytes": 268435456, "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}]
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

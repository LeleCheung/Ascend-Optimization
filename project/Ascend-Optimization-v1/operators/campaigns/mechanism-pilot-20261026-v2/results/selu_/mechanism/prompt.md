你是Ascend 910B算子优化工程师。只根据提供的源码和测量，生成一个更快的完整单文件候选。

输出JSON，仅含source（完整Python源码）、hypothesis（中文）、expected_effect（中文）。

不得调用工具、读取文件或联网。保留def run(A)的原地返回别名和完整shape/dtype/stride语义。

KGS要求候选中若存在与算子同名的公共符号，则必须与run是同一函数对象；可只定义run，或在文件末尾写算子名=run。

不能调用目标算子的PyTorch或FlagGems作为新捷径。原代码已有的非连续fallback可以原样保留。允许torch分配、设备上下文和stream接口。

不能删用例、读取评测模式、缓存计算结果、缓存输入Tensor或复用旧输入指针。浮点运算和特殊值必须正确。

缓存编译产物时每次重新绑定本次输入、当前stream及配置。使用当前后端实际接口；不新增CUDA专有选项。

只允许一次候选机会。输出完整有效源码，不要Markdown，不要占位符。

算子：selu_

共享接口知识：
# 两组共享的固定后端接口知识

固定环境：910B4-1、CANN9.0、FlagTree0.6.0+ascend.gitc286cba6、Triton3.5.1。先导入 torch、torch_npu，再导入 Triton，避免当前安装版本的循环导入问题。

以下为本机已核对的 JITFunction.run 提交语义，供两组同等使用。它只描述编译和调用接口，不给任何算子的最佳实现。

1. 正常调用 `compiled = kernel[grid](*args, **kwargs)` 返回已编译对象。若是 future，使用 `.result()`。
2. 本机 `kernel.device_caches[device]` 是 `(kernel_cache, kernel_key_cache, target, backend, binder)`。
3. 绑定前，选项执行 `kwargs['debug'] = kwargs.get('debug', kernel.debug) or knobs.runtime.debug`，然后 `triton.runtime.jit._flagprism.apply_compile_options(kwargs)`；`binder(*args, **kwargs)` 返回 `(bound_args, specialization, options)`。直接查缓存必须与正常 JIT 采用同样的选项，否则可能命中不了真实编译产物。
4. `driver.active.get_current_device()`、`driver.active.get_current_stream(device)` 获取本次设备和当前 stream。
5. `grid(bound_args)` 或固定 tuple 得到实际 grid；不足三维用 1 补齐。
6. `metadata = compiled.launch_metadata(grid, stream, *bound_args.values())`。
7. 提交接口为 `compiled.run(gx, gy, gz, stream, compiled.function, compiled.packed_metadata, metadata, knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())`。
8. 所有输入指针和标量应来自本次调用。可以缓存编译对象与配置，不能缓存旧输入 Tensor、旧数据指针或固定 stream。新 dtype、shape、stride、设备、编译选项需要匹配正确的特化。
9. 直接调用是版本受限的优化途径，必须保留未知布局／版本的正常入口。禁止修改共享运行库、关闭正确性检查或缓存结果。

小输入、较大输入、连续与非连续输入都需要正确。grid 不超过 65535；容量受表达式和 dtype 影响，不假定 block 越大越好。不从活跃比例推导独占流水时间或定量 Roofline。


原始执行观测：
{"schema": "hgkda-execution-slice-v1", "source_sha256": "b87724ca16e0c7b21512857da9757113dbe3e2384203c3b3c54f2d07e19ab373", "operator": "selu_", "probe_sha256": "9ab0cce3f5c701fafe5d32a02522efd280833b80bfc4b62f996ce465036a92e9", "physical_card": 7, "torch": "2.9.0+cpu", "triton": "3.5.1", "jit_runtime_sha256": "d4b9f4cc94c7a5beeb06d8c9404442cdfa26aabc4a474697a090639deb72d2db", "timing_scope": "预热后批量同步walltime/调用，非纯设备时间；相同样本交替测量", "cases": [{"id": "float16-1024", "shape": [1024], "dtype": "float16", "status": "passed", "launches": [{"symbol": "selu_kernel_", "grid": [1], "programs": 1, "parameters": ["x_ptr", "n_elements", "BLOCK_SIZE"], "constants": {"n_elements": 1024, "BLOCK_SIZE": 1024}}], "timings": {"wrapper_us": [100.98508333333332, 90.74175000000001, 89.12675, 91.42508333333335, 89.26091666666667, 90.31925, 95.60425], "direct_us": [47.531666666666666, 43.385083333333334, 43.46333333333333, 44.316750000000006, 42.84758333333334, 41.762499999999996, 44.281749999999995]}, "direct_semantics": "passed", "trace": "candidate-trace.json"}, {"id": "float16-1048576", "shape": [1048576], "dtype": "float16", "status": "passed", "launches": [{"symbol": "selu_kernel_", "grid": [1024], "programs": 1024, "parameters": ["x_ptr", "n_elements", "BLOCK_SIZE"], "constants": {"n_elements": 1048576, "BLOCK_SIZE": 1024}}], "timings": {"wrapper_us": [160.72433333333333, 114.99758333333334, 129.11183333333335, 99.23591666666665, 93.69841666666667, 93.08841666666666, 93.43425], "direct_us": [55.778416666666665, 69.46841666666667, 84.00841666666666, 51.374249999999996, 48.211666666666666, 45.86175, 47.33416666666667]}, "direct_semantics": "passed", "tile_interventions": [{"block": 512, "grid": [2048], "timings": {"parent": [99.583875, 67.252625, 66.75, 68.035125, 66.631375], "variant": [104.532625, 96.076375, 82.18125, 93.202625, 83.87375]}, "status": "passed", "scope": "同输入同JIT分派路径，仅分块与grid改变；不等同完整候选"}, {"block": 2048, "grid": [512], "timings": {"parent": [294.98775, 69.2275, 68.77125, 66.42, 67.156375], "variant": [68.250125, 65.267625, 64.142625, 65.26, 67.641375]}, "status": "passed", "scope": "同输入同JIT分派路径，仅分块与grid改变；不等同完整候选"}, {"block": 4096, "grid": [256], "timings": {"parent": [147.841375, 71.42375, 69.253875, 69.02875, 70.447625], "variant": [69.732625, 67.7475, 67.953875, 67.24125, 66.893875]}, "status": "passed", "scope": "同输入同JIT分派路径，仅分块与grid改变；不等同完整候选"}]}, {"id": "float32-1024", "shape": [1024], "dtype": "float32", "status": "passed", "launches": [{"symbol": "selu_kernel_", "grid": [1], "programs": 1, "parameters": ["x_ptr", "n_elements", "BLOCK_SIZE"], "constants": {"n_elements": 1024, "BLOCK_SIZE": 1024}}], "timings": {"wrapper_us": [95.61425000000001, 91.60175, 90.56675, 95.41675, 91.39841666666666, 90.27425, 90.33341666666666], "direct_us": [44.096666666666664, 43.19675, 43.55675, 42.089166666666664, 44.14083333333334, 42.03925, 43.884166666666665]}, "direct_semantics": "passed"}, {"id": "float32-1048576", "shape": [1048576], "dtype": "float32", "status": "passed", "launches": [{"symbol": "selu_kernel_", "grid": [1024], "programs": 1024, "parameters": ["x_ptr", "n_elements", "BLOCK_SIZE"], "constants": {"n_elements": 1048576, "BLOCK_SIZE": 1024}}], "timings": {"wrapper_us": [134.86758333333333, 94.60425, 94.37341666666667, 92.39675, 91.11091666666665, 93.60175, 95.20341666666667], "direct_us": [45.719249999999995, 49.83175, 48.560833333333335, 45.221666666666664, 46.65091666666667, 44.14166666666667, 45.498416666666664]}, "direct_semantics": "passed", "tile_interventions": [{"block": 512, "grid": [2048], "timings": {"parent": [93.145, 66.827625, 67.0225, 66.40125, 66.207625], "variant": [96.490125, 90.772625, 92.6225, 87.78, 92.445125]}, "status": "passed", "scope": "同输入同JIT分派路径，仅分块与grid改变；不等同完整候选"}, {"block": 2048, "grid": [512], "timings": {"parent": [100.23875, 71.13875, 69.2, 74.25125, 148.015125], "variant": [75.573875, 70.132625, 87.2775, 73.515, 69.93375]}, "status": "passed", "scope": "同输入同JIT分派路径，仅分块与grid改变；不等同完整候选"}, {"block": 4096, "grid": [256], "timings": {"parent": [153.977625, 70.047625, 67.441375, 70.29625, 67.562625], "variant": [68.0125, 67.92, 64.98625, 65.13125, 65.835125]}, "status": "passed", "scope": "同输入同JIT分派路径，仅分块与grid改变；不等同完整候选"}]}]}

自动机制诊断（待验证的原因与方向）：
{"schema": "hgkda-mechanism-diagnosis-v1", "policy": "launch-mechanism-v1", "source_sha256": "b87724ca16e0c7b21512857da9757113dbe3e2384203c3b3c54f2d07e19ab373", "graphs": [{"case": "float16-1024", "source_sha256": "b87724ca16e0c7b21512857da9757113dbe3e2384203c3b3c54f2d07e19ab373", "scope": "warm_cache_single_stream_full_call", "nodes": [{"id": "wrapper", "kind": "python_wrapper"}, {"id": "jit", "kind": "jit_dispatch"}, {"id": "submit", "kind": "runtime_submission"}, {"id": "device", "kind": "device_execution"}, {"id": "completion", "kind": "synchronization"}], "edges": [{"source": "wrapper", "target": "jit", "basis": "捕获到的JIT调用与同stream完成语义"}, {"source": "jit", "target": "submit", "basis": "捕获到的JIT调用与同stream完成语义"}, {"source": "submit", "target": "device", "basis": "捕获到的JIT调用与同stream完成语义"}, {"source": "device", "target": "completion", "basis": "捕获到的JIT调用与同stream完成语义"}], "observations": {"wrapper_us": {"median": 90.74175000000001, "minimum": 89.12675, "maximum": 100.98508333333332, "n": 7}, "direct_us": {"median": 43.46333333333333, "minimum": 41.762499999999996, "maximum": 47.531666666666666, "n": 7}, "paired_ratio": {"median": 2.091542600087204, "minimum": 2.050619295958279, "maximum": 2.1626878180185574, "n": 7}, "launches": [{"symbol": "selu_kernel_", "grid": [1], "programs": 1, "parameters": ["x_ptr", "n_elements", "BLOCK_SIZE"], "constants": {"n_elements": 1024, "BLOCK_SIZE": 1024}, "compiled_artifact_sha256": {"source": "35974591a0d5308d64a0d492f9b89f9f9d02d179e9e3f111538bbea737312a99", "ttir": "b663c05da9eb1af865a33aaf143fbf58c8493356125bd03967b39151aa8fcc43", "ttadapter": "996882172cf8365cee9e0bdfbcd434275471b4e3a92dd0e0a6734c37cff931dd", "mlirbc": "a2357330309314ce30bbe0ded58a17a73ba9bfc84fb9de4dc76234b1e96f21b5", "bcmlir": "d6dbe7fe468b1c65c483cda0b317beefe5391f4e8e9fecc005938f99a8a13fb4", "npubin": "09d4879c2fb53827d133cf7d129fe32409d244b88f783dd456558de64db8d2be"}, "source_file": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/mechanism-pilot-20261026-v2/seeds/selu_.py"}]}, "unknown": ["未测量每条边的独占耗时", "不含设备内部指令依赖", "不从同步walltime推导纯设备耗时"]}, {"case": "float16-1048576", "source_sha256": "b87724ca16e0c7b21512857da9757113dbe3e2384203c3b3c54f2d07e19ab373", "scope": "warm_cache_single_stream_full_call", "nodes": [{"id": "wrapper", "kind": "python_wrapper"}, {"id": "jit", "kind": "jit_dispatch"}, {"id": "submit", "kind": "runtime_submission"}, {"id": "device", "kind": "device_execution"}, {"id": "completion", "kind": "synchronization"}], "edges": [{"source": "wrapper", "target": "jit", "basis": "捕获到的JIT调用与同stream完成语义"}, {"source": "jit", "target": "submit", "basis": "捕获到的JIT调用与同stream完成语义"}, {"source": "submit", "target": "device", "basis": "捕获到的JIT调用与同stream完成语义"}, {"source": "device", "target": "completion", "basis": "捕获到的JIT调用与同stream完成语义"}], "observations": {"wrapper_us": {"median": 99.23591666666665, "minimum": 93.08841666666666, "maximum": 160.72433333333333, "n": 7}, "direct_us": {"median": 51.374249999999996, "minimum": 45.86175, "maximum": 84.00841666666666, "n": 7}, "paired_ratio": {"median": 1.9434801396619077, "minimum": 1.5368916408177358, "maximum": 2.881478947203294, "n": 7}, "launches": [{"symbol": "selu_kernel_", "grid": [1024], "programs": 1024, "parameters": ["x_ptr", "n_elements", "BLOCK_SIZE"], "constants": {"n_elements": 1048576, "BLOCK_SIZE": 1024}, "compiled_artifact_sha256": {"source": "35974591a0d5308d64a0d492f9b89f9f9d02d179e9e3f111538bbea737312a99", "ttir": "b663c05da9eb1af865a33aaf143fbf58c8493356125bd03967b39151aa8fcc43", "ttadapter": "996882172cf8365cee9e0bdfbcd434275471b4e3a92dd0e0a6734c37cff931dd", "mlirbc": "a2357330309314ce30bbe0ded58a17a73ba9bfc84fb9de4dc76234b1e96f21b5", "bcmlir": "d6dbe7fe468b1c65c483cda0b317beefe5391f4e8e9fecc005938f99a8a13fb4", "npubin": "09d4879c2fb53827d133cf7d129fe32409d244b88f783dd456558de64db8d2be"}, "source_file": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/mechanism-pilot-20261026-v2/seeds/selu_.py"}]}, "unknown": ["未测量每条边的独占耗时", "不含设备内部指令依赖", "不从同步walltime推导纯设备耗时"]}, {"case": "float32-1024", "source_sha256": "b87724ca16e0c7b21512857da9757113dbe3e2384203c3b3c54f2d07e19ab373", "scope": "warm_cache_single_stream_full_call", "nodes": [{"id": "wrapper", "kind": "python_wrapper"}, {"id": "jit", "kind": "jit_dispatch"}, {"id": "submit", "kind": "runtime_submission"}, {"id": "device", "kind": "device_execution"}, {"id": "completion", "kind": "synchronization"}], "edges": [{"source": "wrapper", "target": "jit", "basis": "捕获到的JIT调用与同stream完成语义"}, {"source": "jit", "target": "submit", "basis": "捕获到的JIT调用与同stream完成语义"}, {"source": "submit", "target": "device", "basis": "捕获到的JIT调用与同stream完成语义"}, {"source": "device", "target": "completion", "basis": "捕获到的JIT调用与同stream完成语义"}], "observations": {"wrapper_us": {"median": 91.39841666666666, "minimum": 90.27425, "maximum": 95.61425000000001, "n": 7}, "direct_us": {"median": 43.55675, "minimum": 42.03925, "maximum": 44.14083333333334, "n": 7}, "paired_ratio": {"median": 2.1205704132834065, "minimum": 2.0584512257647973, "maximum": 2.267014473241333, "n": 7}, "launches": [{"symbol": "selu_kernel_", "grid": [1], "programs": 1, "parameters": ["x_ptr", "n_elements", "BLOCK_SIZE"], "constants": {"n_elements": 1024, "BLOCK_SIZE": 1024}, "compiled_artifact_sha256": {"source": "82e8f75d2623da9c52e6c2ff6ae9a27d42be4d4fb1c946d49a5cd6b90af7a004", "ttir": "c7c652f5521b8b62a7096896ae715b0c645004bc59850cd9518e62bc547cfcbb", "ttadapter": "a176bfb6cedb7b5391b1d02e105f5cde1ae65dd73fec5f0d7342cf7e830bee27", "mlirbc": "6e9ab2071eef21a2a761775f17eb53b725c1a542e3206a6dc4597277a2b1e5de", "bcmlir": "28a95875681c804b0e7cc5ec9665ed697d0f87dd16629dd80f1185d1cfb0be1b", "npubin": "5d73e3f42b3089e214a914b3c9c327c41768c2bf1497cc45e3e3728bbe6da2cd"}, "source_file": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/mechanism-pilot-20261026-v2/seeds/selu_.py"}]}, "unknown": ["未测量每条边的独占耗时", "不含设备内部指令依赖", "不从同步walltime推导纯设备耗时"]}, {"case": "float32-1048576", "source_sha256": "b87724ca16e0c7b21512857da9757113dbe3e2384203c3b3c54f2d07e19ab373", "scope": "warm_cache_single_stream_full_call", "nodes": [{"id": "wrapper", "kind": "python_wrapper"}, {"id": "jit", "kind": "jit_dispatch"}, {"id": "submit", "kind": "runtime_submission"}, {"id": "device", "kind": "device_execution"}, {"id": "completion", "kind": "synchronization"}], "edges": [{"source": "wrapper", "target": "jit", "basis": "捕获到的JIT调用与同stream完成语义"}, {"source": "jit", "target": "submit", "basis": "捕获到的JIT调用与同stream完成语义"}, {"source": "submit", "target": "device", "basis": "捕获到的JIT调用与同stream完成语义"}, {"source": "device", "target": "completion", "basis": "捕获到的JIT调用与同stream完成语义"}], "observations": {"wrapper_us": {"median": 94.37341666666667, "minimum": 91.11091666666665, "maximum": 134.86758333333333, "n": 7}, "direct_us": {"median": 45.719249999999995, "minimum": 44.14166666666667, "maximum": 49.83175, "n": 7}, "paired_ratio": {"median": 2.0431964766151918, "minimum": 1.8984733628660442, "maximum": 2.949908043840031, "n": 7}, "launches": [{"symbol": "selu_kernel_", "grid": [1024], "programs": 1024, "parameters": ["x_ptr", "n_elements", "BLOCK_SIZE"], "constants": {"n_elements": 1048576, "BLOCK_SIZE": 1024}, "compiled_artifact_sha256": {"source": "82e8f75d2623da9c52e6c2ff6ae9a27d42be4d4fb1c946d49a5cd6b90af7a004", "ttir": "c7c652f5521b8b62a7096896ae715b0c645004bc59850cd9518e62bc547cfcbb", "ttadapter": "a176bfb6cedb7b5391b1d02e105f5cde1ae65dd73fec5f0d7342cf7e830bee27", "mlirbc": "6e9ab2071eef21a2a761775f17eb53b725c1a542e3206a6dc4597277a2b1e5de", "bcmlir": "28a95875681c804b0e7cc5ec9665ed697d0f87dd16629dd80f1185d1cfb0be1b", "npubin": "5d73e3f42b3089e214a914b3c9c327c41768c2bf1497cc45e3e3728bbe6da2cd"}, "source_file": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/mechanism-pilot-20261026-v2/seeds/selu_.py"}]}, "unknown": ["未测量每条边的独占耗时", "不含设备内部指令依赖", "不从同步walltime推导纯设备耗时"]}], "decisions": [{"case": "float16-1024", "competing_hypotheses": [{"id": "dispatch", "evidence": "相同编译产物直接提交的配对实验", "supported": true}, {"id": "device_or_other", "evidence": "直接提交未改变设备程序；剩余耗时尚需分块探针", "supported": null}], "action": "cache_compiled_dispatch", "tile_evidence": [], "recommended_tile": null, "prediction": {"metric": "same_case_wrapper_latency", "direction": "decrease", "mechanism": "减少Python/JIT分派；设备程序和grid保持不变"}, "limits": "直接提交是干预探针，候选必须对动态指针、当前stream、布局及数值完整复验"}, {"case": "float16-1048576", "competing_hypotheses": [{"id": "dispatch", "evidence": "相同编译产物直接提交的配对实验", "supported": true}, {"id": "device_or_other", "evidence": "直接提交未改变设备程序；剩余耗时尚需分块探针", "supported": null}], "action": "cache_compiled_dispatch", "tile_evidence": [{"block": 512, "grid": [2048], "paired_ratio": {"median": 0.7944246561051581, "minimum": 0.6999912829766943, "maximum": 0.9526583207874098, "n": 5}, "supported": false, "scope": "同JIT分派的完整任务组织差异，未单独辨识设备耗时"}, {"block": 2048, "grid": [512], "paired_ratio": {"median": 1.0606713512250525, "minimum": 0.9928298323326514, "maximum": 4.322156919126522, "n": 5}, "supported": false, "scope": "同JIT分派的完整任务组织差异，未单独辨识设备耗时"}, {"block": 4096, "grid": [256], "paired_ratio": {"median": 1.0531251927026206, "minimum": 1.0191306235295632, "maximum": 2.1201177354215477, "n": 5}, "supported": false, "scope": "同JIT分派的完整任务组织差异，未单独辨识设备耗时"}], "recommended_tile": null, "prediction": {"metric": "same_case_wrapper_latency", "direction": "decrease", "mechanism": "减少Python/JIT分派；设备程序和grid保持不变"}, "limits": "直接提交是干预探针，候选必须对动态指针、当前stream、布局及数值完整复验"}, {"case": "float32-1024", "competing_hypotheses": [{"id": "dispatch", "evidence": "相同编译产物直接提交的配对实验", "supported": true}, {"id": "device_or_other", "evidence": "直接提交未改变设备程序；剩余耗时尚需分块探针", "supported": null}], "action": "cache_compiled_dispatch", "tile_evidence": [], "recommended_tile": null, "prediction": {"metric": "same_case_wrapper_latency", "direction": "decrease", "mechanism": "减少Python/JIT分派；设备程序和grid保持不变"}, "limits": "直接提交是干预探针，候选必须对动态指针、当前stream、布局及数值完整复验"}, {"case": "float32-1048576", "competing_hypotheses": [{"id": "dispatch", "evidence": "相同编译产物直接提交的配对实验", "supported": true}, {"id": "device_or_other", "evidence": "直接提交未改变设备程序；剩余耗时尚需分块探针", "supported": null}], "action": "cache_compiled_dispatch", "tile_evidence": [{"block": 512, "grid": [2048], "paired_ratio": {"median": 0.7362090167602842, "minimum": 0.7161829788212195, "maximum": 0.9653319445901847, "n": 5}, "supported": false, "scope": "同JIT分派的完整任务组织差异，未单独辨识设备耗时"}, {"block": 2048, "grid": [512], "paired_ratio": {"median": 1.0143460336754826, "minimum": 0.7928733064077225, "maximum": 2.116504906429299, "n": 5}, "supported": false, "scope": "同JIT分派的完整任务组织差异，未单独辨识设备耗时"}, {"block": 4096, "grid": [256], "paired_ratio": {"median": 1.0377791455884897, "minimum": 1.0262397922081865, "maximum": 2.26396066899467, "n": 5}, "supported": false, "scope": "同JIT分派的完整任务组织差异，未单独辨识设备耗时"}], "recommended_tile": null, "prediction": {"metric": "same_case_wrapper_latency", "direction": "decrease", "mechanism": "减少Python/JIT分派；设备程序和grid保持不变"}, "limits": "直接提交是干预探针，候选必须对动态指针、当前stream、布局及数值完整复验"}]}

固定FlagGems起点：
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

logger = logging.getLogger(__name__)


@triton.jit
def selu_kernel_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    x_f32 = x.to(tl.float32)
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    y_f32 = scale * tl.where(x_f32 > 0, x_f32, alpha * (tl.exp(x_f32) - 1.0))
    y = y_f32.to(x.dtype)

    tl.store(x_ptr + offsets, y, mask=mask)


def selu_(*args, **kwargs):
    logger.debug("GEMS SELU_")
    x = None
    if len(args) > 0 and torch.is_tensor(args[0]):
        x = args[0]
    elif "input" in kwargs and torch.is_tensor(kwargs["input"]):
        x = kwargs["input"]
    elif "self" in kwargs and torch.is_tensor(kwargs["self"]):
        x = kwargs["self"]
    elif "x" in kwargs and torch.is_tensor(kwargs["x"]):
        x = kwargs["x"]
    else:
        raise ValueError(
            "selu_ expects a Tensor as the first argument or under 'input'/'self'/'x' keyword."
        )

    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if (not x.is_contiguous()) or (x.dtype not in supported_dtypes):
        torch.ops.aten.selu_(x)
        return x

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        selu_kernel_[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x

_baseline=selu_
def run(A):
    return _baseline(A)
selu_=run

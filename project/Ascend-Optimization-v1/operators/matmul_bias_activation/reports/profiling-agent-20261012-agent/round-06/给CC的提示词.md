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
[{"round": 1, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "parent_score": 0.8671455922289588, "status": "FAILED", "error": "命令失败 1，见 /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-01/claude-response.json"}, {"round": 2, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "parent_score": 0.8671455922289588, "status": "FAILED", "error": "命令失败 1，见 /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-02/claude-response.json"}, {"round": 3, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "parent_score": 0.8671455922289588, "hypothesis": "本轮真机诊断（fp16 2048：aic_mte2=0.836、aic_mac=0.7015、cube_util=73.8%，判定 GEMM 数据流水受限）说明瓶颈是每字节 L1 填充能换到的乘加数，而不是 HBM 带宽。把两条已通过路径折算一下：fp16 dual(128×256, BK=256) 每步搬运 (128+256)×256×2=192KB、产出 16.8 MFLOP（87 FLOP/byte），实测 172 TFLOPS；fp32 单 tile(128×128, BK=256) 每步 (128+128)×256×4=256KB、产出 8.4 MFLOP（32 FLOP/byte），4096 实测 59 TFLOPS。两者折算的 L1 填充带宽都约 1.9TB/s，正好对应 mte2 接近打满——所以真正的杠杆是提高填充强度。\n\n输出累加器受 L0C 128KB 限制，BM/BN 不能再放大（宽 128×256 输出 tile 多缓冲已因 UB 超限失败）。但 v9 已验证的“一个 program 复用 A 面板算两个相邻 N tile”结构，在同样的 L0C 预算（两个 128×128 fp32 累加器 = 128KB）下就把强度从 2·BM·BN/((BM+BN)·esize) 的 32 提到 42.7 FLOP/byte（+33%）。此前只用在半精度上，fp32 仍走单 tile，这正是知识库列出的“FP32 1024 的分块与分组”未探索方向。\n\nfp32 每步片上字节必须与已验证可编译的 fp16 dual 对齐，因此 BK 取 128：(128+256)×128×4=192KB，与 fp16 BK=256 完全相同；而历史溢出的 fp16 dual BK=512 与单 tile BK=512 恰好都是 384KB/步。这不是重试失败配置，而是由失败边界反推出的合规点。同时 fp32 的 K 迭代数由 4/8/16 增至 8/16/32，缩短流水填充/排空占比（fp32 1024 单 tile 时 MTE2 仅 0.545、MAC 0.6625、cube_util 70.9%，属没吃饱而非打满）。\n\n实现上不新增 kernel 代码，直接以新 constexpr 复用已通过 42 项验收的 dual-dispatch kernel，只改 dispatch 条件；其余 12 个 case 的路径与参数一字未动，稳定 fallback（master / v4 / v6 / v9 链）全部保留，run API、dtype、stride、bias 广播语义不变。", "expected_effect": "fp32 规则方阵三个 case（1024/2048/4096）走双 N tile 路径：L1 填充强度 32→42.7 FLOP/byte。按 1.9TB/s 填充带宽量级推算，4096 fp32 有从 0.97× 提升到 1.2× 左右的空间（当前 2331us → 约 1.7-1.9ms），2048/1024 期望从 0.79×/0.72× 各上浮 0.05~0.15（1024 网格由 64 个 program 变 32 个，尾波占比同为 80%，但每 program 每步搬运量不变而乘加数翻倍，净收益为正）。如果 L0C/UB 或 MTE2 在 192KB/步下仍是硬上限，则最多与现状持平，不会低于现有 fallback。半精度、非方阵、小 shape（384）与不规则形状的 dispatch 与参数完全不变，预期成绩不退化；整体相对 PyTorch 的 15 case 均值有望从约 0.88 提升到 0.90 以上。", "source_sha256": "b157ef3a23ad01ef7ea74a5a8fd93dd05780c053eafd958030a7c66ecd9fd2e3", "status": "PASSED", "score": 0.9009200337814591, "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-03/evaluation/trial-1.result.json", "accepted": true}, {"round": 4, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-03/evaluation/trial-1.result.json", "parent_score": 0.9009200337814591, "status": "FAILED", "error": "缺少 run 入口"}, {"round": 5, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-03/evaluation/trial-1.result.json", "parent_score": 0.9009200337814591, "hypothesis": "上一轮（round-04）提交的 fp32 dual 路径从未被真正测到：那次失败的原因是响应被输出长度截断、落盘文件不完整，评测只报「缺少 run 入口」，并非编译或数值失败。本轮以完整可导入的单文件重新交付，并把改动收敛到一条：把半精度上已验证有效的「一个 program 复用 A 面板算两个相邻 N tile」结构（BM=128、有效 BN=256、两个 128x128 fp32 累加器=128KB，正好是 L0C 上限）推广到 fp32 规则方阵 1024/2048/4096，BK 按元素宽度取 128，使每步片上搬运量 (128+256)*128*4=192KB 与已完整通过 42 项验收的第 9 版半精度配置 (128+256)*256*2=192KB 完全一致；历史溢出的 fp16 dual BK=512 与 fp32 BK=512 分别对应 384KB、512KB 每步，192KB 是本机实测可接受点，因此这是按失败边界反推的合规点，不是重试失败配置。强度折算：fp32 单 tile 128x128 每步 (128+128)*BK*4 字节换 16.4K MAC = 16 MAC/byte，dual 每步 (128+256)*BK*4 字节换 32.8K MAC = 21.3 MAC/byte，同样乘加数下 L1 填充字节少 25%；L0C 固定 128KB 下 128x256 是还能提高填充强度的最后一档（256x256 需 256KB 累加器）。代价是 fp32 的 K 迭代数从 4/8/16 变为 8/16/32，这是唯一副作用，靠缩短流水填充占比抵消，需实测确认净收益。同时收敛掉已被 dual 完全遮蔽的 v5(_WIDE_TILES)/v6(kgroup) 两级 dispatch 与重复 kernel 体（二者只匹配 m==n==k 且 contiguous 的方阵，半精度三条早被 v9 拦截、fp32 三条现由 v10 拦截，行为不变），既减少文件体积也减少遮蔽层数，避免再次截断丢 run。master 通用 kernel（含 sdnn 命名绕行与大 K bf16 精度分流）、pipeline(v4)、半精度 dual 的语义，以及 run API、全部 dtype/shape/stride/bias 广播与 ReLU(maximum(x, x*0.0)) 合同均原样保留。", "expected_effect": "只影响 fp32 三个规则方阵 case（1024/2048/4096，round-03 真机为 0.875/0.859/1.024）：每步 L1 填充字节数下降 25%，若这三条确实是 MTE2/填充受限，期望 1024 由 0.875 上浮到 0.93~1.0、2048 由 0.859 上浮到 0.92~1.0、4096 由 1.024 抬到 1.05~1.12，折算到 15 条均值约 +0.015~0.03（基线 0.9009）。半精度 6 条、384 三条以及所有不规则形状的 dispatch 分支与 tile 参数一字未改，预期不退化；若 fp32 dual 在 192KB/步下仍被 L0C/UB 容量或 K 迭代数增加拖累，最坏与现状持平，不会低于现有 fallback。文件整体缩短并去掉遮蔽层，确保本次候选能正常导入并暴露 run 入口。定量上限不做 Roofline 声明：缺少可信 roof 与物理字节计数。", "source_sha256": "3e1180c39fe3565cf5045a3023199794b4f26bd60d5b7a6dcff5569df403917b", "status": "PASSED", "score": 0.9097039517345447, "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "accepted": true}]
当前逐case事实：
[{"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::0", "axes": {"dtype": "bfloat16", "shape_detail": [[384, 384], [384, 384], [384]], "benchmark_level": "core"}, "timing": {"candidate_us": 9.459333333333333, "pytorch_us": 11.948666666666666, "speedup": 1.263158822057699, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 384, "n": 384, "k": 384, "flops": 113246208, "logical_bytes": 885504, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::1", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 791.9633333333334, "pytorch_us": 638.0113333333334, "speedup": 0.8056071864934168, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 100671488, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::2", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 1024], [1024, 1024], [1024]], "benchmark_level": "core"}, "timing": {"candidate_us": 29.275333333333332, "pytorch_us": 24.659333333333333, "speedup": 0.8423251349538805, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 1024, "n": 1024, "k": 1024, "flops": 2147483648, "logical_bytes": 6293504, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::3", "axes": {"dtype": "bfloat16", "shape_detail": [[2048, 2048], [2048, 2048], [2048]], "benchmark_level": "core"}, "timing": {"candidate_us": 127.21733333333331, "pytorch_us": 100.34466666666667, "speedup": 0.7887658478672867, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 2048, "n": 2048, "k": 2048, "flops": 17179869184, "logical_bytes": 25169920, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::bfloat16::4", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 791.1226666666666, "pytorch_us": 638.0833333333334, "speedup": 0.8065542545588057, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 100671488, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::0", "axes": {"dtype": "float16", "shape_detail": [[384, 384], [384, 384], [384]], "benchmark_level": "core"}, "timing": {"candidate_us": 9.502, "pytorch_us": 11.803333333333335, "speedup": 1.242192076839155, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 384, "n": 384, "k": 384, "flops": 113246208, "logical_bytes": 885504, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::1", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 789.2213333333333, "pytorch_us": 618.3966666666666, "speedup": 0.7835529302021405, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 100671488, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::2", "axes": {"dtype": "float16", "shape_detail": [[1024, 1024], [1024, 1024], [1024]], "benchmark_level": "core"}, "timing": {"candidate_us": 29.247333333333334, "pytorch_us": 23.96066666666667, "speedup": 0.8192433979961775, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 1024, "n": 1024, "k": 1024, "flops": 2147483648, "logical_bytes": 6293504, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::3", "axes": {"dtype": "float16", "shape_detail": [[2048, 2048], [2048, 2048], [2048]], "benchmark_level": "core"}, "timing": {"candidate_us": 127.784, "pytorch_us": 95.07533333333333, "speedup": 0.7440317952963892, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 2048, "n": 2048, "k": 2048, "flops": 17179869184, "logical_bytes": 25169920, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": {"aiv_mte2_ratio": 0.045, "aiv_mte3_ratio": 0.039, "aiv_scalar_ratio": 0.056, "aiv_vec_ratio": 0.1155, "aic_mte2_ratio": 0.837, "aic_mte1_ratio": 0.671, "aic_scalar_ratio": 0.243, "aic_mac_ratio": 0.7, "cube_utilization_pct": 72.932, "kernel": "mba_dual_dispatch_v8_kernel", "device_duration_us": 130.12, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "gemm_data_pipeline", "confidence": "medium", "inference": "矩阵乘的 AIC MTE2 活跃比例为 83.7%，Scalar 为 0.243。优先增加 K 分块、减少循环和地址计算，再重叠搬运与计算。流水线比例不等于 HBM 带宽利用率，各比例不能相加。", "next_experiment": "增大 K 分块至 128/256，按 shape 调整 M/N tile，测试 multibuffer/unit_flag；完整正确性与所有 timing case 复验。", "falsifier": "MTE2 和总延迟不改善，或新增分块导致精度失败。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::4", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 788.0053333333333, "pytorch_us": 616.6926666666667, "speedup": 0.7825996377436145, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 100671488, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": {"aiv_mte2_ratio": 0.029, "aiv_mte3_ratio": 0.0195, "aiv_scalar_ratio": 0.0185, "aiv_vec_ratio": 0.0695, "aic_mte2_ratio": 0.918, "aic_mte1_ratio": 0.734, "aic_scalar_ratio": 0.145, "aic_mac_ratio": 0.768, "cube_utilization_pct": 85.9215, "kernel": "mba_dual_dispatch_v8_kernel", "device_duration_us": 804.3, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "gemm_data_pipeline", "confidence": "medium", "inference": "矩阵乘的 AIC MTE2 活跃比例为 91.8%，Scalar 为 0.145。优先增加 K 分块、减少循环和地址计算，再重叠搬运与计算。流水线比例不等于 HBM 带宽利用率，各比例不能相加。", "next_experiment": "增大 K 分块至 128/256，按 shape 调整 M/N tile，测试 multibuffer/unit_flag；完整正确性与所有 timing case 复验。", "falsifier": "MTE2 和总延迟不改善，或新增分块导致精度失败。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::0", "axes": {"dtype": "float32", "shape_detail": [[384, 384], [384, 384], [384]], "benchmark_level": "core"}, "timing": {"candidate_us": 10.937333333333333, "pytorch_us": 13.120666666666667, "speedup": 1.1996202643520875, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 384, "n": 384, "k": 384, "flops": 113246208, "logical_bytes": 1771008, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::1", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 2217.6746666666663, "pytorch_us": 2279.0946666666664, "speedup": 1.02769567518339, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 201342976, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::2", "axes": {"dtype": "float32", "shape_detail": [[1024, 1024], [1024, 1024], [1024]], "benchmark_level": "core"}, "timing": {"candidate_us": 57.036, "pytorch_us": 50.00933333333334, "speedup": 0.8768031708572874, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 1024, "n": 1024, "k": 1024, "flops": 2147483648, "logical_bytes": 12587008, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::3", "axes": {"dtype": "float32", "shape_detail": [[2048, 2048], [2048, 2048], [2048]], "benchmark_level": "core"}, "timing": {"candidate_us": 328.55, "pytorch_us": 283.70866666666666, "speedup": 0.8635175173182619, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 2048, "n": 2048, "k": 2048, "flops": 17179869184, "logical_bytes": 50339840, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float32::4", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], [4096, 4096], [4096]], "benchmark_level": "core"}, "timing": {"candidate_us": 2217.9546666666665, "pytorch_us": 2269.4826666666663, "speedup": 1.023232214098505, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/round-05/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector", "m": 4096, "n": 4096, "k": 4096, "flops": 137438953472, "logical_bytes": 201342976, "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}]
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
# and route large-K bf16 to the un-hijacked generic kernel.
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
    accumulator = tl.maximum(accumulator, accumulator * 0.0)

    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


def _master_mba(input, weight, bias):
    """稳定 fallback：任意 shape/stride/dtype，带完整 mask 与广播合同。"""
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
    """带 mask 的通用流水线路径：384 等小/中形状与大部分非方阵走这里。"""
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
    with torch_device_fn.device(input.device):
        mba_pipeline_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            input, weight, bias, out, m, n, k,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1), bias.stride(0),
            bm, bn, bk, m % bm == 0, n % bn == 0, k % bk == 0,
            multibuffer=not tiny, unit_flag=not tiny, sync_solver=False,
        )
    return out


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


def _dual_run(input, weight, bias, bm, bn, bk):
    """A 面板复用两个相邻 N tile；调用方负责保证尺寸整除（kernel 无 mask）。"""
    m, k = input.shape
    n = weight.shape[1]
    bias = bias.reshape(-1)
    out = torch.empty((m, n), device=input.device, dtype=input.dtype)
    with torch_device_fn.device(input.device):
        mba_dual_dispatch_v8_kernel[((m // bm) * (n // (2 * bn)),)](
            input, weight, bias, out, m, n, k,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1), bias.stride(0),
            out.stride(0), out.stride(1), bm, bn, bk,
            multibuffer=True, unit_flag=True, sync_solver=False)
    return out


# ---------------------------------------------------------------------------
# v11 (2026-10-12)
#
# 1) 半精度规则形状（m%128==0, n%256==0, k%256==0）继续走第 9 版的 dual
#    dispatch：BM=128、两个相邻 N tile（有效 BN=256）、BK=256、两个
#    128x128 fp32 累加器 = 128KB，正好压住 L0C 上限。该路径 42 项正确性
#    全过，整体 0.9009。
#
# 2) 本轮新增：把同一结构推广到 fp32 规则方阵 1024/2048/4096，BK 按元素
#    宽度取 128。上一轮该改动因响应被输出长度截断、落盘文件不含 run 入口
#    而未被测到，不是编译或数值失败，本轮以完整单文件重新交付。
#    依据：
#      * fp16 dual(BM128/BN256/BK256) 每步片上 192KB -> 32.8K MAC/步；
#        fp32 单 tile(BM128/BN128/BK256) 每步 (128+128)*256*4=256KB ->
#        16.4K MAC/步，即 16 MAC/byte。fp32 改成 dual 后每步
#        (128+256)*128*4=192KB -> 32.8K MAC/步，21.3 MAC/byte，
#        相同乘加数下 L1 填充字节数减少 25%。
#      * 192KB/步是已被本机验证可编译的点（fp16 BK=256 就是这个量级），
#        而历史溢出的 fp16 dual BK=512 与 fp32 BK=512 分别对应 384KB、
#        512KB 每步。所以取 BK=128 是由失败边界反推的合规点，不是重试
#        已失败配置。
#      * fp32 1024 单 tile 时 MTE2 仅 0.545、MAC 0.6625、cube_util 70.9%，
#        属于没吃饱；L0C 固定 128KB 下，128x256 是仍能提高填充强度的最后
#        一档（256x256 需要 256KB 累加器，保留位留给未来验证）。
#
# 3) 收敛：删掉已被上面两条完全遮蔽的 v5(_WIDE_TILES) 与 v6(kgroup) 两级
#    dispatch 及其重复 kernel 体（二者只匹配 m==n==k 且 contiguous 的方阵，
#    半精度 1024/2048/4096 早已被 dual 拦截，fp32 三条现在由 fp32 dual
#    拦截，行为不变），减少文件体积与遮蔽层数，避免再次因截断丢 run。
#    master（含 sdnn 命名与大 K bf16 精度绕行）、pipeline(v4) 与 dual 的
#    全部语义、run API、dtype/shape/stride/广播/ReLU 合同保持不变。
# ---------------------------------------------------------------------------


def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    if (input.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()
            and bias.numel() == n):
        # 半精度规则形状：v9 dual（本轮未改）。
        if (input.dtype in (torch.float16, torch.bfloat16)
                and m % 128 == 0 and n % 256 == 0 and k % 256 == 0):
            return _dual_run(input, weight, bias, 128, 128, 256)
        # fp32 规则方阵：本轮新增的 dual，BK 减半保持每步 192KB。
        if (input.dtype == torch.float32 and m == n == k
                and m >= 1024 and m % 256 == 0):
            return _dual_run(input, weight, bias, 128, 128, 128)
    # 与历史一致的稳定分流：不规则走 master，其余走带 mask 的流水线。
    if m % 64 or n % 128 or k % 128:
        return _master_mba(input, weight, bias)
    return _pipeline_run(input, weight, bias)


matmul_bias_activation = run

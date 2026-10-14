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
[{"round": 1, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/seed-evaluation/seed-1.result.json", "parent_score": 0.8997956723470408, "hypothesis": "修改点集中在两个 amin 设备内核 amin_native_middle_kernel / amin_native_last_kernel 以及驱动它们的 run 与 _NATIVE_TILES 配置表，没有改动任何其它 kernel 和 fallback 路径。\n\n1) 让 bfloat16 也走「输入精度 min」（NATIVE=True）。当前 bf16 三个形状走 NATIVE=False，即每次 tl.load 后都做 values.to(tl.float32)，累加器是 (BR,BC)/(BO,BR) 的 FP32 tile，再在 FP32 上做归约。与同 shape、同布局、同代码路径、仅 dtype 不同的 float16 对比（本轮可信诊断给出的同一合同计时）：(4096,4096) fp16 39.29us vs bf16 60.67us；(64,512,512) fp16 29.21us vs bf16 61.31us；(1024,1024,1024) fp16 1669.6us vs bf16 3658.8us。bf16 慢 1.5～2.2 倍，而两者唯一的差别就是「逐元素 fp32 转换 + FP32 累加器 + FP32 归约」这一组额外 Vector 指令。bf16 (4096,4096) 的真机指标显示 aiv_vec_ratio=0.795（MTE2 仅 0.1935），说明瓶颈在 Vector 流水线而不是搬运；而 min 选择操作在任何能表示原值的 dtype 上都是精确的，bf16 原生 min 不会改变结果，NaN/Inf、mask 用 inf 的语义都不变。\n\n2) BR 对齐 fp16。原生累加器只有 FP32 累加器一半的 UB 占用，因此 bf16 可以把归约片放大到与 fp16 相同：(64,512,512) BR 16→32（内层循环 32 次→16 次）、(1024,1024,1024) BR 8→16（64 次→32 次），(4096,4096) 保持 BO=8/BR=1024 只翻转 NATIVE。这直接针对这三个 bf16 慢 case，同时也减少每轮的 mask/地址生成开销。\n\n3) 编译失败安全网。知识中明确记载「bf16 直接输入类型 minimum 某些配置编译失败，不能假设支持」，所以新增 _BF16_NATIVE_FALLBACK：一旦原生路径抛异常，就把该 shape 记入 _native_bf16_disabled 并永久退回「原生配置未启用时」的原始分块（含 BR 一并还原，避免 FP32 累加器在放大后的 BR 下超片上容量），并立即用回退配置重跑同一 out。失败时候选与当前版本行为一致，不会因为编译问题丢掉 fallback；异常只对 bf16 原生路径吞掉，其余异常照常抛出。第一次调用之后不再走 try 分支里的失败路径，稳定 fallback 保持不变。\n\n未采用的方向：不动全归约（amin_global_chunk_kernel）与 v4 的路径判定，不动 fp16/fp32 的分块（它们已由设备计时选定），不做任何未经验证的 Roofline 推断，也不重复历史上已被否决的配置。", "expected_effect": "主要改善三个此前明显落后 PyTorch 的 bfloat16 用例：test_amin[dtype2] 的 (4096,4096) dim=1（当前 60.67us / 0.46×）、(64,512,512) dim=1（61.31us / 0.55×）、(1024,1024,1024) dim=1（3658.76us / 0.39×）。去掉逐元素 FP32 转换与 FP32 累加后，这三个 case 的 Vector 工作量按同 shape 的 fp16 对照大约降到原来的 2/3～1/2，目标是把 bf16 拉回到与 fp16 同一量级（约 39us / 29us / 1670us 附近），使 bf16 三个 case 从 0.39～0.55× 提升到接近或超过 1.0×。若 bf16 原生 min 在本机编译失败，则自动退回现有已验证配置，指标与当前版本持平，不会回退。\n\nfloat16 / float32 的 9 个用例配置保持不变，预期与当前逐 case 事实一致（其中 (1024,1024,1024) 的 fp32 3302.75us×(4.3GB 已计入逻辑字节) 与 fp16 1669.6us 路径不动，避免无依据的分块改动）。全归约与 (64,64) 两个已领先的用例（1.3～2.1×）沿用原路径，不受影响。所有改动只涉及 dtype 归约精度与归约片大小，输出 dtype、keepdim、dim 归一化、mask 用 +inf 的语义完全一致，NaN/Inf 行为与 float16 现有路径相同，不修改计时与用例。以上为基于同合同对照计时的预期，最终以完整测试的实测结果为准。", "source_sha256": "eca4ab44bbe16ae1095ea55deded8f676a97e2e83237e633f5c5f8978e8ede3d", "status": "PASSED", "score": 0.9229204558479288, "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "accepted": true}, {"round": 2, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-01/evaluation/trial-1.result.json", "parent_score": 0.9229204558479288, "hypothesis": "改动只落在两个 amin 设备内核 amin_native_middle_kernel / amin_native_last_kernel、启动函数 _launch_native 与驱动力 run；_NATIVE_TILES 分块表、_direct_axis_amin、全归约路径、bf16 回退表全部保留不动，run API、dim/keepdim/stride 语义不变。\n\n1) 新增 MASKED 编译期开关，取消“恒真掩码”。注册表 9 个配置的分块都能整除对应维度：last 路径 (4096,4096) 用 BO=8、BR=1024（4096%8==0、4096%1024==0），middle 路径 (64,512,512) 用 BC=512、BR=32（512%512==0、512%32==0）、(1024,1024,1024) 用 BC=1024、BR=16（1024%16==0），fp32 的 (512,16)/(8,1024) 与 bf16 回退配置同样整除。也就是说循环里 (r<R)&(cols<I) 和 rows<O 是恒真谓词，但它们仍要按整块 tile 生成比较+与运算、生成 other=inf 填充，并把本可整块连续的搬运降级为带谓词的访问——这些形状的 tile 本身是连续的（BC==I 时整块连续，last 路径每行 BR 个连续元素）。MASKED=False 分支直接 tl.load(ptr)，无掩码、无 other、无谓词；越界补 inf 的语义完整保留在 MASKED=True 分支，未被删除，非整除场景自动回落到该分支。\n\n2) 把两个内核的启动参数 multibuffer 由 False 改为 True（带逐配置回退）。此前单缓冲下每轮迭代是“发起 MTE2 → 等数据 → 向量 min”串行；真机指标显示 bf16 (1024,1024,1024) aiv_vec_ratio=0.764、aiv_mte2_ratio=0.348（之和 1.11，重叠不充分），说明向量流水大量时间在等搬运，而 bf16 (4096,4096) 的 mte2 仅 0.204 而 vec 0.781。multibuffer=True 让下一分块的搬运与当前分块的归约重叠，该后端在矩阵乘 v9 已验证可用（multibuffer=True/unit_flag=True）。UB 估算：最大 tile 是 fp32 (16,1024)/(8,1024) 的 32KB 累加器 + 双缓冲 2×32KB = 96KB，未超片上容量；若某配置编译报错，_multibuffer_disabled 永久退回单缓冲，只吞掉启动异常，其它异常照常抛出。\n\n3) multibuffer 与掩码删除都用 try/except 分级回退：双缓冲 → 单缓冲同分块 → 仅 bf16 原生路径 → FP32 累加回退配置（分块一并还原），不会因为编译问题丢掉稳定 fallback，也不改变最终结果值（min 的选择集合完全相同）。", "expected_effect": "主要改善 3 个只有 4 轮迭代、每轮固定等待搬运的 (4096,4096) dim=1 用例和 2 个 program 数只有 64 的 (64,512,512) dim=1 用例。\n\n依据：fp16 (4096,4096) 现在 39.1us / 33.5MB（约 0.86 TB/s），而同样 512 个 program、同样 4 轮迭代、仅 dtype 不同的 fp32 (4096,4096) 却做到 67MB/53.5us（约 1.25 TB/s），说明该 fp16 用例不是被搬运字节卡住，而是每轮 ~0.3us 的“发起-等待”固定开销（512/20≈26 轮/核 × 4 轮）。双缓冲把这段等待与向量归约重叠、去掉恒真掩码的逐块比较/填充后，3 个 (4096,4096) 用例（fp16 39.1us→约 28-32us、fp32 53.5us→约 42-47us、bf16 60.4us→约 48-55us）预期从 0.59/0.70/0.46 向 0.75/0.88/0.55 靠拢；(64,512,512) 的 3 个用例（fp16 29.2us、fp32 53.1us、bf16 61.4us）因整块连续、掩码开销占比最大，预期有同量级改善。bf16 (1024,1024,1024) 真机 vec 0.764+mte2 0.348 表明重叠不充分，完全重叠的下界约为 3661×0.764≈2797us，即 0.394×→约 0.50×；fp16/fp32 的 (1024,1024,1024)（1669us/3304us，已接近本候选自身约 1.3 TB/s 的搬运速度）收益较小，预期 0.85×→0.90-0.95×。(1048576) 与 (64,64) 六个已领先 1.35-2.27× 的用例不在注册表覆盖范围，路径不变。以上均为基于同合同对照计时与真机 pipe ratio 的预期，带宽上界没有经核实的 roof，不作数值 Roofline 声称，最终以完整测试实测为准；若双缓冲在某配置上变慢或编译失败，回退机制使该配置与当前版本持平。", "source_sha256": "fde0a861892398bbe90cb9e9e71d816184c3381631435340632b3df5358f76a9", "status": "PASSED", "score": 0.9327801734967498, "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "accepted": true}, {"round": 3, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "parent_score": 0.9327801734967498, "hypothesis": "改动只落在 amin_native_last_kernel 在 (4096,4096) dim=1 上的分块（_NATIVE_TILES 中三个 (4096,4096) 条目）与驱动它们的 run 候选/回退编排。内核本体（amin_native_last_kernel / amin_native_middle_kernel 的循环、累计、tl.min、store）和 amin_native_middle_kernel 的全部配置 (64,512,512)=(512,32)/(512,16)、(1024,1024,1024)=(1024,16)/(1024,8) 完全不动；全局归约 amin_global_chunk_kernel、v4 路径 _v4_amin、bf16 的 FP32 累加回退表 _BF16_NATIVE_FALLBACK、MASKED 与 multibuffer 逻辑均保留。\n\n1) 为什么改这两个内核的最后一个：本轮同合同计时显示 (4096,4096) dim=1 的三个 dtype 是同 shape、同 grid、同 tile 结构，只有 dtype 不同：fp16 38.65us、fp32 52.70us、bf16 60.26us。字节比 1:2 而时间比只有 1:1.36，说明该路径耗时里有很大一块与搬运字节无关；同时这三个 case 是本轮最差的一组（0.592× / 0.716× / 0.465×）。历史里唯一一次「程序数与迭代数分离」的对照是 (16,512)/(16,256)（256 program、8 次内层迭代）被 (8,1024)（512 program、4 次迭代）取代并胜出，说明决定该路径快慢的是内层迭代次数与单次搬运的连续长度，而不是 program 数量。\n\n2) 具体改法：现状 (8,1024) 每轮把一行切成 8 段各 2KB 的跨步访问、串行 4 轮。本轮把 inner==1 的末轴路径改成「整行一次归约」：BR = R = 4096，BO 取 max(1, 8192 // R) 的按 dtype 版本——fp16/bf16（NATIVE=True，累加器与 tile 都是 2 字节）BO=4，fp32（NATIVE=False，累加器 4 字节）BO=2。UB 预算：fp16/bf16 acc (4,4096)=32KB + tile 32KB、双缓冲 64KB ≈ 96KB；fp32 acc (2,4096)=32KB + tile 32KB、双缓冲 64KB ≈ 96KB，与旧 (8,1024) 的累加器元素数（8192）和单次搬运元素数（8192）完全相同，属于等价片上占用的改写，不是靠放大 UB 换性能。收益来源三点：内层循环 range(tl.cdiv(4096,4096)) 由 4 轮变 1 轮，去掉 4 次「发起 DMA → 等数据 → 归约」串行链；BR==R 时 (BO,R) tile 恰好覆盖 BO 个相邻整行，单次搬运变成一次连续 DMA（fp16/bf16 32KB、fp32 32KB），取代每轮 8 段 2KB 跨步，描述符数量级下降；网格由 512 变为 1024（fp16/bf16）/2048（fp32），核填充更充分。因为 BR=4096 整除 R=4096、BO 整除 O=4096，MASKED 自动为 False，走无谓词连续 tl.load，越界补 inf 的语义分支仍完整保留。归约数学不变：仍是逐元素 tl.minimum 累积 + 末端 tl.min(acc, axis=1)，输出值、dtype、keepdim、stride 语义与 NaN/Inf 行为完全一致。\n\n3) 回退链（保证不丢用例）：每个候选先试 multibuffer=True，失败再试同候选单缓冲；主配置（整行）编译或资源失败则依次退到历史已 27/27 通过的 (8,1024) 同 NATIVE 配置；bf16 再退到 FP32 累加的 (8,1024,False)。失败候选记入 _config_disabled 永久跳过（不再重复编译尝试），bf16 原生路径失败记入 _native_bf16_disabled 后不再构造任何 NATIVE 候选；若所有候选均已被禁用则整体退回 _v4_amin。异常只吞编译/资源类失败，最终全部失败时仍抛出原始错误。不重复历史已被否决的方向：没有放宽 tile 到 128×256/256×128、没有 BK512、没有 enable_preload、没有关 subblock，也没有改任何已由设备计时选定的中间维分块。", "expected_effect": "主要改善 3 个 (4096,4096) dim=1 末轴用例，即本轮最差的一组：test_amin[dtype0] float16（当前 38.65us / 0.592×）、test_amin[dtype1] float32（52.70us / 0.716×）、test_amin[dtype2] bfloat16（60.26us / 0.465×）。\n\n依据：这三个 case 每 program 只搬 64KB 却要等 4 轮串行跨步 DMA，且 fp16 与 fp32 的耗时差远小于字节差（1.36× vs 2×），说明有约六成的耗时与字节数无关；把 4 轮跨步 2KB 搬运压成 1 轮连续 32KB 搬运后，这部分固定开销被移除。预期 fp16 约 38.65→28-31us（0.59→约 0.74-0.82×）、fp32 约 52.70→40-45us（0.72→约 0.84-0.94×）、bf16 约 60.26→45-52us（0.47→约 0.54-0.62×）。bf16 的绝对收益相对小一些，因为其真机 pipe 指标 vec=0.73、mte2=0.21 表明该 case 还有很大一块是逐元素向量工作量（bf16 需经 FP32 转换再 min），本轮只去掉串行搬运与跨步 DMA，不动转换精度语义。\n\n其余 12 个用例的配置与路径完全不动，预期与当前逐 case 事实一致：全归约 (1048576) 三个 dtype（1.35-1.47×）、(64,64) dim=1 三个 dtype（2.11-2.26×）走 v4 路径；中间维 (64,512,512) 与 (1024,1024,1024) 六个用例继续用 (512,32)/(512,16)/(1024,16)/(1024,8)，不受影响。若某个新配置（尤其 bf16 BR=4096）编译或片上资源失败，逐级回退到历史已通过的 (8,1024)，该用例指标与当前版本持平，不会丢失任何用例。以上为基于同合同计时与历史对照的预期，涉及片上容量与指令行为的推测均由随后的完整测试验证；带宽上界没有经核实的 roof，因此不声称数值 Roofline，最终以完整 KGS 合同的实测结果为准。", "source_sha256": "142f54b87ab7b4751679d136625148ddd929f0f674f73fa626c74383523b943e", "status": "PASSED", "score": 0.8743332472174657, "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-03/evaluation/trial-1.result.json", "accepted": false}, {"round": 4, "parent_result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "parent_score": 0.9327801734967498, "status": "FAILED", "error": "命令失败 1，见 /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-04/claude-response.json", "claude_error": "havior, set the CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable.\"}],\"context_management\":null},\"parent_tool_use_id\":null,\"session_id\":\"821d79da-8a30-4f77-a12b-931c9775df66\",\"uuid\":\"5fe24f64-f02a-40c9-b99e-906a68f3f968\",\"timestamp\":\"2026-10-12T14:33:06.458Z\",\"error\":\"max_output_tokens\",\"is_api_error_message\":true,\"api_error\":\"max_output_tokens\"}\n{\"duration_api_ms\":757338,\"stop_reason\":\"stop_sequence\",\"session_id\":\"821d79da-8a30-4f77-a12b-931c9775df66\",\"total_cost_usd\":2.6596496000000003,\"usage\":{\"input_tokens\":21866,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":60928,\"output_tokens\":128000,\"output_tokens_details\":{\"thinking_tokens\":0},\"server_tool_use\":{\"web_search_requests\":0,\"web_fetch_requests\":0},\"service_tier\":\"standard\",\"cache_creation\":{\"ephemeral_1h_input_tokens\":0,\"ephemeral_5m_input_tokens\":0},\"inference_geo\":\"\",\"iterations\":[],\"speed\":\"standard\"},\"modelUsage\":{\"deepseek-flash\":{\"inputTokens\":21866,\"outputTokens\":128000,\"cacheReadInputTokens\":60928,\"cacheCreationInputTokens\":0,\"webSearchRequests\":0,\"costUSD\":2.6596496000000003,\"contextWindow\":200000,\"maxOutputTokens\":32000,\"thinkingTokens\":0,\"canonicalModel\":\"deepseek-flash\",\"provider\":\"firstParty\",\"costBasis\":\"unknown\"}},\"permission_denials\":[],\"terminal_reason\":\"api_error\",\"fast_mode_state\":\"off\",\"fast_mode_disabled_reason\":\"sdk_opt_in_required\",\"subagent_stats\":{\"spawned\":0,\"requested\":{\"background\":0,\"foreground\":0,\"unset\":0},\"started_in_background\":0,\"max_depth\":0,\"spawned_by_subagents\":0,\"completed\":0,\"failed\":0,\"killed\":{\"parent\":0,\"user\":0,\"system\":0},\"refused\":{\"depth_limit\":0,\"concurrency_limit\":0,\"budget\":0},\"by_type\":{}},\"is_error\":true,\"num_turns\":4,\"subtype\":\"success\",\"api_error_status\":null,\"result\":\"API Error: Claude's response exceeded the 32000 output token maximum. To configure this behavior, set the CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable.\",\"type\":\"result\",\"duration_ms\":757601,\"uuid\":\"8733ce35-bdf3-456c-9cc0-9e6846d326c3\",\"queued_turn_count\":0,\"result_index\":0}\n"}]
当前逐case事实：
[{"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::0", "axes": {"dtype": "float16", "shape_detail": [[1048576]], "benchmark_level": "core"}, "timing": {"candidate_us": 6.297333333333333, "pytorch_us": 8.559333333333333, "speedup": 1.35919395732653, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1048576, "output_elements": 1, "reduction_axis": null, "reduction_length": 1048576, "minimum_comparisons": 1048575, "flops": null, "logical_bytes": 2097154, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::1", "axes": {"dtype": "float16", "shape_detail": [[64, 64], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 1.7440000000000002, "pytorch_us": 3.743333333333333, "speedup": 2.146340997266976, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 4096, "output_elements": 64, "reduction_axis": 1, "reduction_length": 64, "minimum_comparisons": 4032, "flops": null, "logical_bytes": 8320, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::2", "axes": {"dtype": "float16", "shape_detail": [[4096, 4096], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 38.64533333333334, "pytorch_us": 22.884666666666668, "speedup": 0.5921725982285099, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 4096, "reduction_axis": 1, "reduction_length": 4096, "minimum_comparisons": 16773120, "flops": null, "logical_bytes": 33562624, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::3", "axes": {"dtype": "float16", "shape_detail": [[64, 512, 512], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 28.08666666666667, "pytorch_us": 16.96466666666667, "speedup": 0.6040128031825189, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 32768, "reduction_axis": 1, "reduction_length": 512, "minimum_comparisons": 16744448, "flops": null, "logical_bytes": 33619968, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype0]::core::float16::4", "axes": {"dtype": "float16", "shape_detail": [[1024, 1024, 1024], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 1629.1306666666667, "pytorch_us": 1417.36, "speedup": 0.8700100255917659, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1073741824, "output_elements": 1048576, "reduction_axis": 1, "reduction_length": 1024, "minimum_comparisons": 1072693248, "flops": null, "logical_bytes": 2149580800, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::0", "axes": {"dtype": "float32", "shape_detail": [[1048576]], "benchmark_level": "core"}, "timing": {"candidate_us": 6.136, "pytorch_us": 9.027333333333333, "speedup": 1.4712004910828267, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1048576, "output_elements": 1, "reduction_axis": null, "reduction_length": 1048576, "minimum_comparisons": 1048575, "flops": null, "logical_bytes": 4194308, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::1", "axes": {"dtype": "float32", "shape_detail": [[64, 64], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 1.623333333333333, "pytorch_us": 3.6653333333333333, "speedup": 2.2578280598731086, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 4096, "output_elements": 64, "reduction_axis": 1, "reduction_length": 64, "minimum_comparisons": 4032, "flops": null, "logical_bytes": 16640, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::2", "axes": {"dtype": "float32", "shape_detail": [[4096, 4096], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 52.699333333333335, "pytorch_us": 37.71666666666666, "speedup": 0.715695867697502, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 4096, "reduction_axis": 1, "reduction_length": 4096, "minimum_comparisons": 16773120, "flops": null, "logical_bytes": 67125248, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::3", "axes": {"dtype": "float32", "shape_detail": [[64, 512, 512], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 50.95133333333334, "pytorch_us": 30.682666666666666, "speedup": 0.6021963399132376, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 32768, "reduction_axis": 1, "reduction_length": 512, "minimum_comparisons": 16744448, "flops": null, "logical_bytes": 67239936, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype1]::core::float32::4", "axes": {"dtype": "float32", "shape_detail": [[1024, 1024, 1024], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 3233.1173333333336, "pytorch_us": 2830.6106666666665, "speedup": 0.875505089138787, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "float32", "element_bytes": 4, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1073741824, "output_elements": 1048576, "reduction_axis": 1, "reduction_length": 1024, "minimum_comparisons": 1072693248, "flops": null, "logical_bytes": 4299161600, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::0", "axes": {"dtype": "bfloat16", "shape_detail": [[1048576]], "benchmark_level": "core"}, "timing": {"candidate_us": 6.398666666666667, "pytorch_us": 8.617333333333335, "speedup": 1.3467334850986907, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1048576, "output_elements": 1, "reduction_axis": null, "reduction_length": 1048576, "minimum_comparisons": 1048575, "flops": null, "logical_bytes": 2097154, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::1", "axes": {"dtype": "bfloat16", "shape_detail": [[64, 64], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 1.746, "pytorch_us": 3.6853333333333333, "speedup": 2.110665673978199, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 4096, "output_elements": 64, "reduction_axis": 1, "reduction_length": 64, "minimum_comparisons": 4032, "flops": null, "logical_bytes": 8320, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::2", "axes": {"dtype": "bfloat16", "shape_detail": [[4096, 4096], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 60.263333333333335, "pytorch_us": 27.99266666666667, "speedup": 0.4645066687759244, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 4096, "reduction_axis": 1, "reduction_length": 4096, "minimum_comparisons": 16773120, "flops": null, "logical_bytes": 33562624, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": {"aiv_mte2_ratio": 0.223, "aiv_mte3_ratio": 0.031, "aiv_scalar_ratio": 0.1005, "aiv_vec_ratio": 0.738, "aic_mte2_ratio": 0.0, "aic_mte1_ratio": 0.0, "aic_scalar_ratio": 0.0, "aic_mac_ratio": 0.0, "cube_utilization_pct": 0.0, "kernel": "amin_native_last_kernel", "device_duration_us": 66.78, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "reduction_vector_pipeline", "confidence": "medium", "inference": "最小值归约的 Vector 活跃比例为 73.8%。优先检查类型转换与归约指令，不能将它解释为 Cube GEMM 算力已达峰值。", "next_experiment": "固定布局与分块，比较 fp32 累积和可编译的输入精度 min，保存 IR/指标并完整复验。", "falsifier": "向量流水线变化不伴随延迟改善，或类型变化破坏 min 的结果。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::3", "axes": {"dtype": "bfloat16", "shape_detail": [[64, 512, 512], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 59.04666666666667, "pytorch_us": 33.708666666666666, "speedup": 0.5708825151588882, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 16777216, "output_elements": 32768, "reduction_axis": 1, "reduction_length": 512, "minimum_comparisons": 16744448, "flops": null, "logical_bytes": 33619968, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": null, "diagnosis": {"bound": "inconclusive", "confidence": "low", "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。", "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。", "falsifier": "指标采集仍无法识别候选 kernel。"}}, {"case_id": "benchmark/test_amin.py::test_amin[dtype2]::core::bfloat16::4", "axes": {"dtype": "bfloat16", "shape_detail": [[1024, 1024, 1024], 1], "benchmark_level": "core"}, "timing": {"candidate_us": 3528.1373333333336, "pytorch_us": 1448.3226666666667, "speedup": 0.4105063348675486, "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/amin/reports/profiling-agent-20261012-agent/round-02/evaluation/trial-1.result.json", "scope": "FlagGems Ascend 设备 kernel 计时"}, "workload": {"status": "algorithm_estimate", "dtype": "bfloat16", "element_bytes": 2, "byte_source": "logical_minimum", "physical_bytes": null, "roofline_usable": false, "semantic": "最小值归约", "compute_path": "AIV Vector", "input_elements": 1073741824, "output_elements": 1048576, "reduction_axis": 1, "reduction_length": 1024, "minimum_comparisons": 1072693248, "flops": null, "logical_bytes": 2149580800, "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}, "metrics": {"aiv_mte2_ratio": 0.3625, "aiv_mte3_ratio": 0.001, "aiv_scalar_ratio": 0.044, "aiv_vec_ratio": 0.7585, "aic_mte2_ratio": 0.0, "aic_mte1_ratio": 0.0, "aic_scalar_ratio": 0.0, "aic_mac_ratio": 0.0, "cube_utilization_pct": 0.0, "kernel": "amin_native_middle_kernel", "device_duration_us": 3536.83, "scope": "独立真机 msprof 采集"}, "diagnosis": {"bound": "reduction_vector_pipeline", "confidence": "medium", "inference": "最小值归约的 Vector 活跃比例为 75.8%。优先检查类型转换与归约指令，不能将它解释为 Cube GEMM 算力已达峰值。", "next_experiment": "固定布局与分块，比较 fp32 累积和可编译的输入精度 min，保存 IR/指标并完整复验。", "falsifier": "向量流水线变化不伴随延迟改善，或类型变化破坏 min 的结果。"}}]
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
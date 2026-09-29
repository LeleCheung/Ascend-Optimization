# 非昇腾：历史通过、复测失败的完整归因台账（2026-09-08）

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

归档范围：本文及配套台账冻结于 2026-09-08 早期 Excel 的 44 条厂商/算子样本，当时海光复测数据尚未补齐。后续更新后的 62 项 pytest review 和其中 35 项重新生成、原生复测属于独立批次，不回填本历史统计；文中的“当前”均指本次历史审计时点。原始 Excel、kernel_todo_v2/ 和 runs/ 为本地实验归档，不随 Git 分发。

当前主线分类与优先级见[复测实际失败主线](../../validation/nonascend_retest_mainline.md)。本文保留完整源码审查；以下相关缺陷数量不能用作原表失败根因的影响数。

**共 44 条厂商／算子样本，去重 38 个算子名。所有 44 条都已逐份核查候选源码、原表描述与已有原生证据；31 条存在已证实的相关缺陷或评测差异，另外 13 条仍只有故障报告或待验证假设。** 这里的“相关问题已证实”包含新发现的独立源码缺陷，不等于原表失败已闭环。实习生原始命令、失败输入、实际执行代码和完整环境未齐备，本轮没有将任何条目标为原表同条件复现完成。

本报告替代之前只围绕已有 P0 标签的覆盖估计。没有把当前 `--level core` 通过当作原表失败消失，也没有把表里的“建议修法”直接当作已经验证的根因。全部原始单元格、候选 SHA、责任范围、证据限制与下一步见 [机器台账](../data/nonascend_retest_failure_20260908.json)。

## 计数口径

选取 V2 归档状态为“成功”，且原 Excel 的 D/H 栏报告失败，或 E 栏数值加速比低于 0.8 的记录。正确性 PASS 但性能不合格仍算复测失败；没有有效计时也不能凭正确性通过验收。PPU 不能只读 D 栏。原表旧状态的 PASS/成功/达标与归档身份交叉核对。

| 厂商 | 样本数 | 占44条比例 |
|---|---:|---:|
| 天数 | 9 | 20.5% |
| 摩尔 | 13 | 29.5% |
| PPU | 9 | 20.5% |
| 沐曦 | 13 | 29.5% |

海光工作表没有复测数据，不能称为“零失败”；昇腾完全排除。29 条有明确失败状态，26 条数值低于0.8，其中11条重叠，所以并集为44条。之前只算29条漏掉15条性能不合格记录。摩尔页末的8条环境说明不是算子。35份候选可关联V2固定Gems版本，另外9份继承历史V1候选、旧revision尚未完整核验，不能一律写成同一旧版本生成。

## 全部失败的现象分布（互斥，合计44）

优先把“精度失败且低性能”归入混合项；其余共享内存超限单列，BF16 PassManager失败按编译分类。这里描述原表症状，不是责任方占比。

| 原表症状 | 数量 | 占比 |
|---|---:|---:|
| 仅性能低于0.8，未报告其他执行错误 | 21 | 47.7% |
| 编译失败（含原表误写成精度失败的编译错误） | 8 | 18.2% |
| 共享内存超限（可另伴已完成点低性能） | 3 | 6.8% |
| 精度/语义失败，无低于0.8的有效数值 | 3 | 6.8% |
| 精度失败且性能低于0.8（可另伴资源错误） | 3 | 6.8% |
| 非法内存访问或32位地址限制 | 2 | 4.5% |
| reference API不可运行 | 1 | 2.3% |
| 只有错误标记、缺具体症状 | 1 | 2.3% |
| benchmark超时 | 1 | 2.3% |
| benchmark输入生成失败 | 1 | 2.3% |

另按重叠口径：26条存在数值性能不合格；6条报告数值/语义精度失败；4条报告共享内存超限。原表8条编译失败中，7条为沐曦BF16报告（6个比较算子及1个实际inplace scatter_add_），1条为摩尔LLVM崩溃。不能把这些全叫“候选精度不过”。

## 头部问题及责任（按证实影响数排序）

优先级用于安排修复/反馈，不表示因果确定性。P0优先处理能污染结论、破坏正确性或跨多个算子复用的缺陷；同级按受影响算子行数排序。仅报错而未定位的同类大簇单列“待确认”，不能借症状数量扩大已确认责任。一个样本可有多个问题，下表比例不能相加。

| 优先级／问题 | 已证实相关样本 | 占比 | 另待确认 | 责任仓库／层次 | 最小处理方案 |
|---|---:|---:|---:|---|---|
| P0 候选修改评测、框架或编译全局状态 | 8 | 18.2% | 0 | KG 生成候选；KGS preflight 规则覆盖 | 统一禁止候选改变Torch/编译/计时全局状态；Gems约束按adapter范围启用；注册注入在可信层完成。 |
| P0 候选未命中实际入口／测试对象错配 | 6 | 13.6% | 1 | KG 原生复测注入与测试选择；Gems overload/alias 对接 | pytest review 验证nodeid、inplace/alias/overload、candidate hash及分阶段调用证据。 |
| P0 候选访存边界保护缺失 | 5 | 11.4% | 0 | KG 生成候选与正确性覆盖 | 修完整mask，用非整除/不等长/广播输入验证；不增加逐算子源码特判。 |
| P0 候选遗漏参数范围或广播语义 | 4 | 9.1% | 0 | KG 生成候选；测试覆盖 | 测试域外数值、n边界、非默认scalar、不同rank广播；按独立gold重生成确认错误候选。 |
| P0 候选覆盖了benchmark基线 | 2 | 4.5% | 0 | KG 原生复测注入；Gems基线角色声明 | 独立保留baseline，标明Torch/Gems类型；reference阶段禁止命中候选。 |
| P0 沐曦 BF16 比较算子编译失败簇 | 2 | 4.5% | 4 | Metax FlagTree lowering；候选identity由复测层确认 | 给编译器团队已有最小IR/脚本；其余4条逐变体核对实际kernel后归并。 |
| P0 历史best被reference异常高值抬高 | 2 | 4.5% | 0 | KG best验收；KGS计时证据 | best独立重复确认并检查双边时延；异常点不能继续凭最高ratio中选。 |
| P1 非形状参数或测试集合不同 | 2 | 4.5% | 4 | KGS adapter参数计划／原生复测配置 | 固定--level core与完整case fingerprint；原扩展集保留另报，新增/共同点拆开比较。 |
| P0 AA backward错误语义／疑似拟合异常reference | 1 | 2.3% | 0 | KG候选；MUSA Torch reference待独立设备对照 | 先做CPU gold/MUSA Torch/Gems/候选四方对照，禁止让Coder拟合错误reference。 |
| P0 reference inplace污染候选输入 | 1 | 2.3% | 0 | Gems upstream pytest；KG候选错误拟合 | reference/candidate独立原始输入；原生reference修改后禁止再clone作为候选输入。 |
| P0 按输入对象缓存计算结果导致陈旧数据 | 1 | 2.3% | 0 | KG 生成候选 | 移除跨调用数据结果缓存，测试同对象内容修改后的第二次调用。 |
| P1 benchmark输入构造索引越界 | 1 | 2.3% | 0 | Gems benchmark | pytest review覆盖全部计划输入的构造与reference可执行性。 |
| P1 adapter直调与ATen调度开销不同 | 1 | 2.3% | 0 | KGS adapter／Gems原生计时口径 | 固定end-to-end/kernel测量边界并报告调用开销。 |
| P1 合法skip导致审计误判PARTIAL | 1 | 2.3% | 0 | KG 原生复测汇总 | 区分预期非法参数skip与有效case缺失，不阻止已完整执行范围的计时。 |
| P1 原生测试全部skip，原表仍标正确性PASS | 1 | 2.3% | 0 | Gems pytest设备条件；复测报告PASS判定 | pytest review在执行前确认有效测试，NO_EXECUTED_TESTS不算通过。 |
| P1 原生reference所需API不可用 | 1 | 2.3% | 0 | MUSA Torch/runtime能力；测试准备 | 准备阶段检查能力并保留原生错误；禁止候选注入全局fallback改reference。 |

源码候选问题涉及17/44条（38.6%），评测身份、输入、计时或可比性问题涉及17/44条（38.6%），与已有编译器最小复现直接关联2/44条（4.5%）。这些集合重叠，去重后31条（70.5%）；余13条（29.5%）没有形成独立确定机制。**这些数字不能改写成“70.5%的原表失败已定位”，也不能把全部评测差异都定责KGS。** Gems upstream、KG复测工具、KGS adapter、候选与厂商工具链的具体职责在逐项记录中分开。

| 仅报告或尚未确认的大簇 | 样本数 | 占比 | 当前分级 | 责任边界 |
|---|---:|---:|---|---|
| 沐曦BF16比较编译簇 | 6（2条有关联实证，4条待对齐） | 13.6% | P0待确认 | upstream lowering最小复现已可反馈；不宣称6个交付候选均回归 |
| 天数共享内存超限 | 4 | 9.1% | P0待确认 | 先确定candidate/reference与tile资源；超硬件上限本身不等于编译器bug |
| 纯性能降幅仍无确定缺陷解释的重点队列 | 6 | 13.6% | P1 | 摩尔SVD、BN、LSTM backward、masked accumulate、nearest exact backward；PPU baddbmm_ |
| 摩尔LLVM后端崩溃 | 1 | 2.3% | P1待复现 | 原表llc -6；当前精确候选通过，待原IR与版本 |
| 沐曦32位地址限制 | 1 | 2.3% | P1待定位 | 候选偏移提升与后端限制尚未分开 |

“6条纯性能重点队列”只表示没有找到独立确定缺陷的性能样本；其他已有副作用、旁路或源码错误的算子，其原表性能跌幅也可能没有解释完。不能将其余性能问题当作已经解决。

## 本次新增且不能忽略的证据

五份候选缺访存边界：天数 addmm_，摩尔 matmul_bias_activation，PPU linear，沐曦 matmuladd、matmul_bias_activation。host地址计算给出每条启用的越界访问；摩尔使用原表记载的 M495/N5333/K71 就进入缺B上界mask的转换路径。源码缺陷已成立，但是否在某次GPU分配下触发非法访问仍需设备验证。

此前漏标的全局状态修改包括PPU unbind_copy写入llc shim并改编译工具路径、PPU float_power_持久注册context、沐曦linear移除编译环境变量。摩尔cudnn_convolution还有按张量对象缓存repack/im2col的问题：同一张量内容改变后返回陈旧结果，且重复benchmark省略预处理。合法的常量元数据缓存没有归入此项。

摩尔AA backward交付候选的整倍数下采样梯度是稀疏scatter。16→8且output梯度在(3,3)处为1时，候选只向(6,6)给1；独立bilinear antialias导数应分布到16个输入像素，(6,6)为9/64。与此同时，原表所称的fused kernel只在Gems upstream存在，并不在交付候选里。不能按原表“ref稀疏，所以res稠密错了”的推断继续修改候选；需要核对CPU gold、MUSA Torch、Gems及候选四方。

天数baddbmm_的输入别名问题由Gems upstream测试顺序引入；已有git blame证明KG后续resolver补丁没有改动reference/clone顺序。候选为拟合这一行为直接返回输入，则属于我方独立错误。

## 性能下降的判断边界

最隐蔽的已确认问题是reference异常高值进入历史best、候选替换同时污染baseline，以及直调/ATen入口不同。它们均可能在“正确性通过、候选有调用、计时为正”时让加速比失真。摩尔adaptive_avg_pool2d_backward的旧reference单点异常1208.7倍，但这仍不足以解释原表0.0619×；LSTM backward的0.0039×和masked accumulate的0.0047×也不能在缺原始行时被认定为填错列或编译器回归。

已核对的原生Gems commit中，benchmark未指定`--level`时默认comprehensive；我们的复测显式core。天数230/242个性能点与当前15点、沐曦30shape与当前15点不能直接等价比较。它可能来自默认level、layout或额外workload，但实习生完整命令未归档，不能写成“已经证明Gems更新新增了这些workload”。35份可关联V2候选的直接参数装饰器对比也不足以排除间接shape、随机dim和输入内容变化。

后续复测必须按每个case保留baseline_ms、candidate_ms、完整参数/stride指纹、实际callable/hash和计时模式。先比较共同case的分子/分母变化，再单列新增case；固定core不意味着删除或忽略原表扩展输入失败。

## 当前preflight／pytest review能保证什么

用KGS `b4aad01` 的flaggems源码准入重新扫描全部44份，拒绝3份：摩尔cudnn_convolution、var，PPU conv_depthwise2d。它们属于8份全局状态问题中的3份，另外5份确实没有被当前源码规则识别。41份未拒绝不能统一称为“41个漏报”：纯性能、数学语义、硬件资源和实际注入身份不都是静态源码门禁的职责。

pytest review应负责测试准备、调用对象、baseline隔离、有效测试及输入生成；preflight负责候选源码准入；完整正确性与性能验收承担语义和计时检查。本轮没有新增算子专用规则，没有扩大BLOCK职责，也没有把review skill的静态无提示当作已经完成动态review。

## 全部44条逐项记录

### 天数

| 行ID／算子／优先级 | 原表结果 | 责任与已核实问题 | 未闭环范围及下一步 | 证据 |
|---|---|---|---|---|
| tianshu:6 `mvlgamma_` P0 | 11.777x → 0.740x；不通过(性能) | **KG 生成候选；KGS 准入**。大输入分支调用 torch.cuda.empty_cache()，改变进程 allocator/cache 状态；当前 core 11.8452×，与原表 0.740× 不一致。 | 已证实副作用，未证明它造成原表降速；缺原表逐点双边计时。 **下一步：**移除候选全局副作用后，固定输入和计时模式比较 reference_ms/candidate_ms。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/tianshu/codes/mvlgamma_.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-followup/tianshu-mvlgamma_/artifacts/result.json) |
| tianshu:17 `expand_copy` P1 | 0.939x → 0.214x*；不通过(性能) | **待定；测试集选择由 KG/KGS 与复测脚本共同负责**。原表明确使用自建 workload；候选连续同形状走拷贝快路，广播通路创建两个设备元数据张量并执行逐维整数除法；当前 core 0.9280×。 | 自建输入未归档；只能确认比较集合不同，不能断言通用广播慢路就是 0.214× 的全部原因。 **下一步：**恢复自建 shape/stride/扩展参数，分别比较快路、广播慢路，保留原测试集。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/tianshu/codes/expand_copy.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/tianshu-expand_copy/artifacts/result.json) |
| tianshu:20 `linear_backward` P0 | 1.139x → —；不通过(性能) | **KG 原生复测注入；共享内存超限责任待定位具体 kernel**。原表要求 133120 B、硬件上限 131072 B；当前 benchmark 的 torch_op 与 gems_op 都绑定被替换的 Gems linear_backward，约 1× 不具独立基线意义。 | 超限来自候选还是 Gems baseline 的具体 kernel/配置未归档；不能标成编译器 bug。 **下一步：**先恢复独立 baseline，再在失败 shape 单独编译两侧并记录共享内存、tile、stages。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/tianshu/codes/linear_backward.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/tianshu-linear_backward/artifacts/result.json) |
| tianshu:25 `addbmm` P0 | 1.142x → 0.556x；不通过(性能) | **候选 tile 资源规划或 Gems baseline 待定；评测集合确有差异**。原表 242 个已完成性能点 0.556×，后续大输入超共享内存；当前 core 15 点 1.1434×。 | core 通过未覆盖原表大集合；缺失败 kernel 身份和原始逐点数据。 **下一步：**按原集合逐点对齐，区分已完成点的真实性能与未完成点的资源错误。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/tianshu/codes/addbmm.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/tianshu-addbmm/artifacts/result.json) |
| tianshu:34 `special_bessel_y0` P0 | 3.077x → 2.355x*；不通过(精度) | **KG 生成候选；KGS 准入**。Y0 多项式只拟合小区间，无域外分支：x=10 时源码公式约 1.2791，独立 libm 为 0.05567；另有 empty_cache、指针状态和扩容分配影响计时。 | 已证实候选数学错误，与扩展输入失败同类；原表四个输入的具体值缺失，2.355× 不能覆盖精度失败。 **下一步：**重新生成覆盖声明输入域的实现；使用旧 Triton 为 reference Triton，禁止拟合评测状态。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/tianshu/codes/special_bessel_y0.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/new-image-tianshu/native/tianshu-special_bessel_y0/artifacts/result.json) |
| tianshu:36 `special_shifted_chebyshev_polynomial_t` P0 | 1.624x → 5.103x；不通过(精度) | **KG 生成候选；正确性测试覆盖**。仅选择 n<=12 的递推结果，n=13 留下初值 1；x=1/4,n=13 的数学结果为 -1/2。当前测试随机 n=0..9、标量 n=3，无法覆盖此错误。 | 精确反例已证实；不能用当前 core/既有正确性集通过否定原表十个语义失败点。 **下一步：**扩大参数边界测试并重新生成；此类语义验证放测试，不增加算子专用 preflight 规则。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/tianshu/codes/special_shifted_chebyshev_polynomial_t.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/tianshu-special_shifted_chebyshev_polynomial_t/artifacts/result.json) |
| tianshu:38 `unbind_copy` P0 | 1.963x → 0.229x；不通过(性能) | **KG 原生复测注入／Gems ATen overload 对接**。归档原生 66 passed、24 合法 skip，但候选调用为 0；注册 unbind_copy 没有命中实际 .int overload。 | 已证实当前审计链路旁路；实习生测试入口未归档，不能据此认定原表 0.229× 测的是候选。 **下一步：**pytest review 对具体 ATen overload 验证候选覆盖，再重新测量。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/tianshu/codes/unbind_copy.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/tianshu-unbind_copy/artifacts/result.json) |
| tianshu:40 `addmm_` P0 | 0.949x → 0.709x；不通过(性能) | **KG 生成候选；原表资源超限归属待定**。K 可整除 tile 时直接无 mask 加载 A/B，丢失 M/N 尾部保护；M15,N160,K1024 有确定越界地址。原表 242 点 0.709× 后又共享内存超限。 | 越界是独立候选缺陷；尚不能将原表共享内存错误改判为该越界造成。 **下一步：**先修通用边界 mask，再对原表大输入两侧核对共享内存与分组性能。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/tianshu/codes/addmm_.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/tianshu-addmm_/artifacts/result.json) |
| tianshu:48 `baddbmm_` P0 | 0.863x → 0.470x；不通过 | **KG 候选错误；Gems upstream pytest 输入别名缺陷**。fp32 且 alpha/beta 非默认时直接 return self，例 self2,A3,B4,alpha2,beta3 应为30却返回2；Gems 测试在 reference inplace 后才 clone，特定无 fp64 条件下污染候选输入。 | 候选错误和 pytest 缺陷均确认；230 点 0.470× 及后续共享内存问题仍需独立性能/资源归因。 **下一步：**pytest review 修输入隔离；候选重新生成真实计算；性能与编译失败分别记账。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/tianshu/codes/baddbmm_.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/tianshu-baddbmm_/artifacts/result.json) |

### 摩尔

| 行ID／算子／优先级 | 原表结果 | 责任与已核实问题 | 未闭环范围及下一步 | 证据 |
|---|---|---|---|---|
| moer:2 `adaptive_avg_pool2d_backward` P0 | 11.876x → 0.0619x；PASS | **KG best 验收／KGS 计时证据审核；异常计时底层来源待定**。旧 best 的一个 reference 点 11.24132 ms，其他轮中位数 0.00930 ms，抬高 headline；当前匹配候选耗时基本不变，core 5.4932×。 | 已证实旧 headline 虚高，未解释实习生 0.0619×；不能把所有下降都归为 reference 抖动。 **下一步：**最终 best 增加独立稳定性复核；取原表逐点原始耗时，隔离计时、输入、实现身份。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/adaptive_avg_pool2d_backward.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-followup/moer-adaptive_avg_pool2d_backward/artifacts/result.json) |
| moer:16 `cudnn_convolution` P0 | 1.389x → —；FAIL | **MUSA Torch API 能力缺失；KG 候选越权修改 Torch 并缓存输入结果**。原生 reference 无 aten::cudnn_convolution，64 失败、候选0调用。候选仅在 KGS 环境变量存在时改写 Torch/注册 fallback；repack/im2col 缓存只按输入对象身份，修改同一张量仍返回旧缓存。 | reference 不可运行已确认；不能通过在原生环境也打开全局 fallback 来宣告修复。 **下一步：**预运行标记 reference API 不可用；保留给 runtime 团队的原生复现；移除候选全局 patch 和跨调用数据缓存。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/cudnn_convolution.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/moer-cudnn_convolution/artifacts/result.json) |
| moer:30 `linalg_svdvals` P1 | 1.790x → 0.5972x；PASS | **待定；候选迭代算法与计时阶段需隔离**。当前18项正确性通过，但 benchmark route 在500秒预算内超时。候选 SVD 大 K 分支按 batch/轮次发射 Jacobi 计算，存在高启动成本路径。 | 没有阶段完成日志证明超时发生于候选运行、reference 还是编译；原表0.5972×也缺双边耗时。 **下一步：**固定一个失败/慢 shape，分别限时编译与执行两侧，记录 launch 数和迭代次数。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/linalg_svdvals.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-timeout-retry-bounded/moer-linalg_svdvals/artifacts/result.json) |
| moer:32 `matmul_bias_activation` P0 | 1.763x → 0.3677x；FAIL | **KG 生成候选；KG/KGS 旧 best 验收**。输入转 fp16 的 B 尾部只有下界 mask；原表 M495,N5333,K71 下存在越界读写。还手动截断 fp32 mantissa 后转 fp16；旧 reference 一个点放大8.7倍。 | 不能把原表13.4%误差仅归于 BLOCK_K 累加顺序；越界、数值策略、实际注入对象须分别验证。 **下一步：**先修边界、按独立 gold 验证 fp32 数值策略，再重测；原表建议单改 BLOCK_K 不足。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/matmul_bias_activation.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-multiple-targets/moer-matmul_bias_activation/artifacts/result.json) |
| moer:36 `native_batch_norm_legit` P1 | 2.346x → 0.7663x；PASS | **待定**。原表0.7663×，当前 core15点2.3073×；检查到融合小形状/两阶段大形状路径，running stats 修改属于该算子合法语义。 | 未找到可独立证明原表降速的缺陷；不能把合法 running stats 更新算作全局 hack。 **下一步：**固定 training、momentum、输入布局和规模，配对比较两侧耗时及计时模式。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/native_batch_norm_legit.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-followup/moer-native_batch_norm_legit/artifacts/result.json) |
| moer:39 `rnn_relu` P1 | 28.205x → 0.0902x；PASS | **Gems pytest设备skip／benchmark设备查询；原表PASS判定**。正确性196项全部被 CUDA-only 条件 skip，候选0调用；原表说已修 benchmark 硬编码 CUDA 查询后测到0.0902×。 | 原生全部skip不能算正确性通过；当前审计工具已正确返回NO_EXECUTED_TESTS。不能据此推断历史adapter从未执行。benchmark补丁和原表候选耗时未归档，性能瓶颈未确认。 **下一步：**pytest review 将 NO_EXECUTED_TESTS 排除通过；准备可执行 reference/设备测试后再验收性能。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/rnn_relu.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/moer-rnn_relu/artifacts/result.json) |
| moer:48 `thnn_fused_lstm_cell_backward_impl` P1 | 1.854x → 0.0039x；PASS | **待定**。原表0.0039×，精确归档候选当前 core25点1.8511×，配对 reference/candidate 都保持旧量级；源码为融合反向计算与输出分配。 | 几百倍差异未复现；不能擅自认定填错列，也无证据认定 compiler 回归。 **下一步：**优先恢复原始 benchmark 行，核对单位、列、输入和 callable hash。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/thnn_fused_lstm_cell_backward_impl.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-followup/moer-thnn_fused_lstm_cell_backward_impl/artifacts/result.json) |
| moer:49 `unbind_copy` P0 | 6.280x → 0.1144x；PASS | **KG 原生复测注入／Gems ATen overload 对接**。与天数相同：66 passed、24 skip，候选0调用；当前 .int overload 未命中交付候选。 | 旁路机制确认，原表0.1144×的测量对象尚待日志核对。 **下一步：**按实际 overload 注入并验证调用证据，保留完整 copy 语义测量。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/unbind_copy.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/moer-unbind_copy/artifacts/result.json) |
| moer:50 `unsafe_masked_index_put_accumulate` P1 | 4.574x → 0.0047x；PASS | **待定**。原表0.0047×，精确候选当前8点4.3958×，配对耗时基本稳定；候选原子累加、输入 inplace 属合法算子行为。 | 无原表原始耗时，既不能断言编译器慢几百倍，也不能断言误填时延。 **下一步：**取原始行并固定 index dtype、rank、重复索引率，区分原子冲突与口径差异。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/unsafe_masked_index_put_accumulate.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-followup/moer-unsafe_masked_index_put_accumulate/artifacts/result.json) |
| moer:52 `upsample_bilinear2d_aa` P1 | 9.528x → —；FAIL | **MUSA FlagTree/LLVM 后端待同代码复现**。原表 LLVM Cannot select i64 ExternalSymbol、llc退出-6，属于后端崩溃症状；候选存在混合浮点/整数 span 计算，当前60项正确性和15点core通过。 | 没有相同候选/版本/输入的崩溃与修正后对照，不能宣告是版本更新引入的回归。 **下一步：**保留原 IR、完整 llc 命令和镜像版本交编译器团队；先确认原表实际执行的是候选还是 Gems 实现。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/upsample_bilinear2d_aa.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-followup/moer-upsample_bilinear2d_aa/artifacts/result.json) |
| moer:53 `upsample_bilinear2d_aa_backward` P0 | 2.091x → 0.4621x；FAIL | **KG 候选数学语义错误；MUSA Torch reference／复测对象待核**。候选整倍数下采样把梯度稀疏 scatter 到 RH/RW 整数格点；独立 AA 导数应分布至滤波窗口。原表 _fused_backward_kernel / _should_use_fused_path 均不在交付候选，却存在于 Gems upstream。 | 表中 ref稀疏/res稠密不能直接认定 res错误；需要 CPU gold、MUSA reference、交付候选、Gems 原生四方对照。 **下一步：**先核实实际调用对象和 reference 正确性，禁止重新生成去拟合错误稀疏 reference。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/upsample_bilinear2d_aa_backward.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-followup/moer-upsample_bilinear2d_aa_backward/artifacts/result.json) |
| moer:55 `upsample_nearest_exact2d_backward` P1 | 3.122x → 0.3684x；PASS | **待定；测试集/参数变化是候选假设**。原表0.3684×，当前30项正确性、9点core3.2204×；候选2倍缩放有专用快路，其他比例退回清零加原子累加。 | 没有原表比例/形状证明是否触发慢路，不能把源码路径差异直接算作已确认根因。 **下一步：**固定缩放参数对齐快慢路径，报告共同输入和新增输入各自比例。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/upsample_nearest_exact2d_backward.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-followup/moer-upsample_nearest_exact2d_backward/artifacts/result.json) |
| moer:56 `var` P0 | 1.549x → 0.7988x；PASS | **KG 候选修改 Gems 注册配置；低性能原因待定**。候选修改 flag_gems._FULL_CONFIG / FULL_CONFIG_BY_FUNC 桥接 var.correction；当前源码门禁能拒绝。历史去除全局 patch 的原生复测另有1.4896×。 | 原表0.7988×略低于阈值，但没有重复测量；不能声称注册补丁移除就解决性能。 **下一步：**注册映射放可信注入层；对临界性能做重复确认，仍按0.8口径记录。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/moer/codes/var.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/moer-var/artifacts/result.json) |

### PPU

| 行ID／算子／优先级 | 原表结果 | 责任与已核实问题 | 未闭环范围及下一步 | 证据 |
|---|---|---|---|---|
| pingtouge:2 `cholesky_solve_helper` P1 | 1.688x → 0.4011；加速比不过 | **KGS adapter 与 Gems 原生调用入口的评测口径**。同形状参考耗时基本不变，候选原生耗时约3.899倍；独立 direct/dispatcher 对照约2.96–3.17倍。当前0.4167×接近原表0.4011×。 | 已证实调用路径增加成本，但辅助 host-loop 实验不是正式 timer，不能精确解释全部倍数。 **下一步：**固定声明 end-to-end 或 kernel 计时口径，分别报告调度开销与 kernel 时间。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/pingtouge/codes/cholesky_solve_helper.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-followup/pingtouge-cholesky_solve_helper/artifacts/result.json) |
| pingtouge:3 `conv_depthwise2d` P0 | 1.530x → 1.5022；正确性不过 | **KG 候选修改编译器私有 API 和 Gems 注册；PPU 工具链兼容需独立核对**。候选 patch PPUBackend.make_hgbin 和 override_registered_op；新环境私有 API 变化使导入失败，去除 patch 后原始 kernel 在既有诊断中通过。 | 原表只写正确性不过，不能把全部原因归为 PPU 编译器回归；候选越权 patch 是已确认交付缺陷。 **下一步：**移除候选对工具链和 Gems 的修改；若原生输出仍汇编失败，提供未打补丁的最小复现。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/pingtouge/codes/conv_depthwise2d.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-followup/pingtouge-conv_depthwise2d/artifacts/result.json) |
| pingtouge:4 `_flash_attention_forward` P2 | 1.064x → 0.8258；错误 | **待定**。原表补充栏仅写错误，0.8258×本身并未低于0.8；当前8项正确性、50个计时点0.8514×。 | 缺错误类型、stack、输入和执行对象；不能从 attention 名称推断 reference 问题。 **下一步：**恢复原始报错，先分准确性、API、编译、运行与计时阶段。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/pingtouge/codes/_flash_attention_forward.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-report-reanalysis/pingtouge-_flash_attention_forward/artifacts/result.json) |
| pingtouge:8 `scaled_dot_product_efficient_attention` P0 | 1.779x → 0.7716；— | **KG 原生复测注入；Gems benchmark 基线声明**。原生 benchmark 的 torch_op 实际绑定 flag_gems 函数；源码替换让 baseline 与 candidate 都跑候选，1.0039×不可用。 | 已确认当前审计基线污染；它不能解释实习生0.7716×，原生Gems基线也不能误称Torch。 **下一步：**独立保存原生 Gems baseline 函数并标明 baseline_kind，验证 reference 阶段候选0调用。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/pingtouge/codes/scaled_dot_product_efficient_attention.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-followup/pingtouge-scaled_dot_product_efficient_attention/artifacts/result.json) |
| pingtouge:16 `baddbmm_` P1 | 1.133x → 0.5446；— | **待定**。原表0.5446×，当前36项正确性、15个core点1.1334×；该候选没有天数 fp32 非默认 scalar 直接返回的分支。 | 不能因为算子同名把天数候选缺陷套到PPU；缺原表输入与时延。 **下一步：**对齐 batch、scalar、dtype、矩阵规模及 timer 后再归因。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/pingtouge/codes/baddbmm_.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/pingtouge-baddbmm_/artifacts/result.json) |
| pingtouge:21 `float_power_` P0 | 10.705x → 0.5877；— | **KG 原生复测选取错误 benchmark；KG 候选修改 Gems 注册**。正确性确实调用 inplace 候选，但 benchmark 测 out-of-place float_power，候选0调用；候选又持久进入 override_registered_op context。 | 计时对象错误已确认；原表0.5877×与旧10.705×无法直接比较。 **下一步：**pytest review 校验 inplace/out-of-place 与 overload；移除候选注册hook，在可信层提供准确计时入口。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/pingtouge/codes/float_power_.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-multiple-targets/pingtouge-float_power_/artifacts/result.json) |
| pingtouge:24 `index_select_backward` P1 | 3.391x → 0.6266；— | **KGS adapter 参数计划与 Gems 原生随机 workload；候选非零dim性能**。adapter core全dim0，原生随机dim；当前0.6903×与原表0.6266×都不达标，原生不同dim分组性能差距很大。 | 分组 shape/dtype 不同，不能把组均值当只改变dim的因果实验；但输入计划不一致已确认。 **下一步：**将dim/stride纳入case fingerprint，固定shape交叉验证各dim并优化慢路径。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/pingtouge/codes/index_select_backward.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/pingtouge-index_select_backward/artifacts/result.json) |
| pingtouge:26 `linear` P0 | 1.269x → 0.6325；— | **KG 生成候选；原表性能下降原因待定**。输入转 fp16 时 EVEN 仅检查两者长度可整除BLOCK，grid却按最大长度；长度不相等时较小张量无mask越界，M512,N1024,K1024是反例。 | 边界缺陷已确认；当前core1.2629×不能证明原表0.6325×因它而降速。fp32转fp16也未保留全部指数范围。 **下一步：**修独立长度mask，测试非方形输入和fp32幅值，再配对复测性能。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/pingtouge/codes/linear.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/pingtouge-linear/artifacts/result.json) |
| pingtouge:41 `unbind_copy` P0 | 4.534x → 0.0319；— | **KG 候选改写编译器工具路径；KG 复测skip汇总**。候选写入 /tmp 的 llc shim、改 TRITON_PPU_LLC_PATH 并重写汇编名称；当前66项正确性实际调用候选、24合法skip，却被审计工具PARTIAL阻断benchmark。 | 与其他三个unbind_copy的0调用不同；原表0.0319×性能原因仍缺证据。不能从shim注释独立确认编译器bug。 **下一步：**移除编译器修改；review 区分合法skip与全部skip，完成真实计时后再归因。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/pingtouge/codes/unbind_copy.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/pingtouge-unbind_copy/artifacts/result.json) |

### 沐曦

| 行ID／算子／优先级 | 原表结果 | 责任与已核实问题 | 未闭环范围及下一步 | 证据 |
|---|---|---|---|---|
| muxi:11 `eq_` P0 | 2.720x → 2.268x；FAIL | **Metax FlagTree BF16 lowering 已有独立复现；原表候选注入待核**。已有最小BF16 bool直接/经int32转bf16失败，经float32转成功；upstream eq_/ne_失败，交付eq_候选正确注入后18项通过。 | 原表36项含不同marker/变体，不能把upstream复现等同于交付候选更新后编译失败。 **下一步：**向编译器团队提交最小lowering复现；复测同时保存实际kernel/source hash和完整变体清单。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/eq_.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-validation/muxi-eq_/artifacts/result.json) |
| muxi:15 `le_` P0 | 2.897x → 2.431x；FAIL | **Metax FlagTree BF16 lowering／复测注入与范围待核**。原表24fail/12pass并报BF16 PassManager；当前交付le_选定marker18项通过。 | 只凭错误字符串不能把eq_/ne_最小复现扩张成此候选编译回归；缺原表tensor/scalar的逐节点映射。 **下一步：**逐变体确认执行对象，再将失败IR按共同lowering归并。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/le_.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-expansion/muxi-le_/artifacts/result.json) |
| muxi:16 `less_` P0 | 2.897x → 2.442x；FAIL | **Metax FlagTree BF16 lowering／复测注入与范围待核**。原表BF16编译失败；当前交付less_选定marker18项通过。 | 原表36项与当前18项范围不同，候选编译回归未证实。 **下一步：**保存tensor/scalar与别名调用映射，在失败节点捕获实际编译代码。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/less_.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-expansion/muxi-less_/artifacts/result.json) |
| muxi:17 `less_equal_` P0 | 2.894x → 2.432x；FAIL | **原表Metax BF16编译归属待核；另有KG候选广播缺陷**。当前less_equal_18项通过；源码处理other的shape/stride时未按右对齐补维，A(2,3),B(3,)会错误广播，属于独立语义漏洞。 | 广播漏洞不能解释原表BF16 PassManager错误；后者仍需失败对象/IR证据。 **下一步：**补右对齐广播边界测试，并独立核查原表BF16编译链路。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/less_equal_.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-expansion/muxi-less_equal_/artifacts/result.json) |
| muxi:18 `ne_` P0 | 2.747x → 2.249x；FAIL | **Metax FlagTree BF16 lowering／复测注入与范围待核**。原表ne_的tensor/scalar BF16失败；当前交付候选18项通过。原生ne_有相同lowering机制证据。 | 仅机制相近不足以证明实习生执行此归档候选；不能将本项当新增独立编译器bug。 **下一步：**按失败nodeid核对候选identity，复用已确认lowering最小复现但不扩大责任样本数。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/ne_.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-expansion/muxi-ne_/artifacts/result.json) |
| muxi:19 `not_equal_` P0 | 2.716x → 2.246x；FAIL | **Metax FlagTree upstream lowering；KG 原生别名注入**。not_equal_最初替换没有命中实际ne_入口，候选0调用；改为正确入口后归档候选18项通过。 | 原表的编译失败可能发生在upstream，但缺原始identity，不能把可能性写成最终根因。 **下一步：**pytest review固定别名到实际callable的映射，同时保留编译器BF16独立复现。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/not_equal_.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-alias-route/muxi-not_equal_/artifacts/result.json) |
| muxi:21 `weight_norm_interface_backward` P1 | 2.025x → 2.746x；FAIL | **候选地址计算／Metax编译器运行时责任待定**。原表shape2六项报memory size or pointer value too large to fit in 32 bit；当前27项正确性及5点core通过。 | 缺shape2具体值、pointer偏移和栈；错误不等于已经确认候选32位溢出，也不等于compilerbug。 **下一步：**恢复shape2，检查numel、stride、pointer偏移的int64提升位置和后端支持上限。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/weight_norm_interface_backward.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-timeout-retry-bounded/muxi-weight_norm_interface_backward/artifacts/result.json) |
| muxi:34 `linear` P0 | 0.801x → —；FAIL | **KG 候选修改编译环境；性能超时根因待隔离**。候选导入及run中pop TRITON_DISABLE_SWIZZLE，改变进程编译配置；当前benchmark完成但core0.7675×仍低于阈值。 | 未复现原表>650秒超时；0.7675×与超时是不同现象，不能混成一个编译器退化结论。 **下一步：**移除环境修改，固定选项将冷编译与热运行分开测量，分析低性能点。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/linear.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/muxi-linear/artifacts/result.json) |
| muxi:41 `fractional_max_pool2d_backward` P1 | 7.057x → —；FAIL | **Gems benchmark 输入构造**。benchmark生成器直接取shape[2]/shape[3]；不足4维会在候选执行前IndexError。当前选定18项正确性/15点core通过。 | 原表缺具体shape，源码条件与所报栈吻合但未完成同输入闭环。 **下一步：**pytest review 验证整个计划内shape的输入生成，保留错误用例并修通用生成器，勿删workload伪装通过。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/fractional_max_pool2d_backward.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-metax-timeout-followup/muxi-fractional_max_pool2d_backward/artifacts/result.json) |
| muxi:45 `matmuladd` P0 | 0.845x → —；FAIL | **KG 生成候选；原表非法访问与具体kernel关联待核**。matmuladd只有K边界mask，没有M/N mask；M15,N160,K32会越界读。原表benchmark非法内存访问，当前core0.7686×也未达标。 | 越界有源码反例，与症状吻合；缺原表shape/栈不能声称已精确复现原CUDA错误。 **下一步：**修全维度边界mask并运行边界输入，再将编译/非法访问/纯性能分别验收。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/matmuladd.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/muxi-matmuladd/artifacts/result.json) |
| muxi:48 `scatter_add` P0 | 3.081x → 2.877x；FAIL | **复测脚本测试范围归属；BF16失败属实际inplace实现待核**。原表失败nodeid为test_scatter_add_，而交付候选是out-of-place scatter_add；当前对应候选54项正确性、15点core通过。 | 两个算子混算不能定责此候选；实际inplace实现的BF16后端错误需独立归档。 **下一步：**pytest review核对测试nodeid、算子语义和candidate入口；拆开inplace/out-of-place结果。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/scatter_add.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/muxi-scatter_add/artifacts/result.json) |
| muxi:51 `matmul_bias_activation` P0 | 1.182x → 0.326x；FAIL | **KG 生成候选；复测性能口径待核**。matmul_bias_activation缺M/N加载边界，另有按大K采用TF32且注释承认偶发越容差的分支；原表30shape0.326×，当前15点core1.1735×。 | 已确认边界漏洞，未证明原表慢在候选；原表称gems实现，也需要确认实际注入身份和30shape清单。 **下一步：**先修边界和数值策略，固定native core；扩展集单列，并保存两侧callable及逐点耗时。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/matmul_bias_activation.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-matmul-target-followup/muxi-matmul_bias_activation/artifacts/result.json) |
| muxi:52 `unbind_copy` P0 | 4.675x → 0.336x；FAIL | **KG 原生复测注入／Gems ATen overload 对接**。66 passed、24 skip但候选0调用，未命中unbind_copy.int；原表0.336×不能据当前通过追认候选性能。 | 需要实习生实际入口证据，不能仅凭view/copy物化开销说明归档候选变慢。 **下一步：**与天数/摩尔共享overload注入修正和pytest review调用校验。 | [候选](/data/akg_kernel_bench_lite/kernelgen/kernel_todo_v2/muxi/codes/unbind_copy.py)；[既有原生记录](/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/regression_audit_20260907_01/native-agent-broad/muxi-unbind_copy/artifacts/result.json) |

## 证据与复现

未形成独立确定机制的13条完整名单：tianshu:25 `addbmm`；moer:30 `linalg_svdvals`；moer:36 `native_batch_norm_legit`；moer:48 `thnn_fused_lstm_cell_backward_impl`；moer:50 `unsafe_masked_index_put_accumulate`；moer:52 `upsample_bilinear2d_aa`；moer:55 `upsample_nearest_exact2d_backward`；pingtouge:4 `_flash_attention_forward`；pingtouge:16 `baddbmm_`；muxi:15 `le_`；muxi:16 `less_`；muxi:18 `ne_`；muxi:21 `weight_norm_interface_backward`。其余31条的原表同条件闭环限制仍必须阅读逐项台账，不能视为已修复。

本轮新增GPU运行：0。使用已保存且核对候选SHA的原生结果，新增host源码/数学反例与源码门禁扫描；没有安装或替换Torch、Triton或厂商运行时，没有启动远端Server/Coder。它是完整样本归因审核，不是44条新镜像同条件重跑。

分析基线：KG v6.2.1@0e595c72；源码门禁KGS feature b4aad01；已有原生复测Gems 34bd6d68928c8b0039c42987840929da52aa6a62。没有创建新KG/KGS/Protocol发布组合。原表与归档映射SHA见机器台账。

host反例：[结果](/data/akg_kernel_bench_lite/kernelgen/runs/nonascend-failure-attribution-20260908/source-counterexamples.json)；[脚本](../../../scripts/analysis/probe_nonascend_candidates.py)。源码门禁：[44条输出](/data/akg_kernel_bench_lite/kernelgen/runs/nonascend-failure-attribution-20260908/preflight-source.json)。候选源码只读取，数学探针不冒充设备编译或原表失败输入重放。

原始Excel和kernel_todo_v2是本机实验输入，不随Git分发。恢复同SHA输入后可重建计数；已审核的归因台账随本分支保存，统计和Markdown可不依赖GPU重建：

```bash
python3 scripts/analysis/analyze_nonascend_retest.py --workbook <复测.xlsx> --attribution <旧身份映射/attribution.json> --output <cohort.json>
python3 scripts/analysis/probe_nonascend_candidates.py --cohort <cohort.json> --output <source-counterexamples.json>
python3 scripts/analysis/render_nonascend_retest.py --ledger docs/operations/data/nonascend_retest_failure_20260908.json --report docs/operations/troubleshooting/nonascend_retest_failure_analysis.md --summary docs/operations/data/nonascend_retest_failure_summary_20260908.json
python3 -m unittest discover -s tests/analysis -p "test_nonascend*.py"
```

优先开发顺序：先补共享的pytest身份/基线review与全局状态准入缺口；随后处理5份边界缺陷和明确语义反例；并行准备厂商BF16/LLVM/资源问题的证据包，尚未同条件复现的只作为待确认反馈。按用户既定约束，确需重新生成时使用Codex runtime、传入旧Triton作为reference Triton；当前任务未启动新一轮生成。

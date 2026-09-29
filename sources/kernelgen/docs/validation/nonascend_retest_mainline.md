# 非昇腾复测失败主线：分类、责任与优先级

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

归档范围：本文及配套台账冻结于 2026-09-08 早期 Excel 的 44 条厂商/算子样本，当时海光复测数据尚未补齐。后续更新后的 62 项 pytest review 和其中 35 项重新生成、原生复测属于独立批次，不回填本历史统计；文中的“当前”均指本次历史审计时点。原始 Excel、kernel_todo_v2/ 和 runs/ 为本地实验归档，不随 Git 分发。

**主线仍为44条厂商/算子记录、38个算子名。原表实际暴露的失败决定调查顺序；额外源码缺陷单独记账，不再计作这些失败的已确认原因。** 本文是当前主线优先级入口；[前一轮完整源码审查](../operations/troubleshooting/nonascend_retest_failure_analysis.md)保留作证据与额外发现档案，其“相关缺陷影响数”不再作为主线根因排行。

天数9、摩尔13、PPU9、沐曦13；海光无复测数据，昇腾排除。原表数值低于0.8也算失败，不能只读FAIL字样。所有原表记录保持不变，包括已发现失败对象错配的那一行。

## 原表失败分类（互斥，合计44）

| 主问题 | 算子记录数 | 占比 |
|---|---:|---:|
| 仅性能不合格 | 21 | 47.7% |
| 编译失败（测试对象待核） | 7 | 15.9% |
| 精度/语义失败，含同时低性能 | 6 | 13.6% |
| 共享内存超限，含同时低性能 | 3 | 6.8% |
| 非法访问/32位地址限制 | 2 | 4.5% |
| reference API不可用 | 1 | 2.3% |
| 仅标错误、症状缺失 | 1 | 2.3% |
| benchmark超时 | 1 | 2.3% |
| benchmark输入构造失败 | 1 | 2.3% |
| 失败归给了不同算子 | 1 | 2.3% |

精度与资源错误优先作为该行主问题，性能下降作为同一行的子问题保留。因此重叠统计是性能不合格26条、精度/语义失败6条、共享内存超限4条。原表8条编译失败报告中的scatter_add行，其失败nodeid实际属于scatter_add_；本表将它单列为对象归属错误，剩余编译队列为7条，未删除原失败记录。

## 主线问题队列（按影响数排列，可重叠）

表格按原表影响数量排列。P0优先处理成簇的正确性/编译/资源阻断及确定的测试对象错配；不等于这些样本已全部定责编译器、Gems或Coder。性能26条先列P1调查队列，需拆成候选变慢、reference变化、输入集合变化或测量对象错误，不能因数量大就把未知根因升级为P0。单例且缺完整复现的编译/地址/超时问题列P1；只有“错误”字样的列P2。

| 优先级 | 原表实际问题 | 数量/占比 | 先由谁处理 | 当前证据边界 |
|---|---|---:|---|---|
| P1 | 原表性能低于0.8 | 26 / 59.1% | 评测/复测维护方先核对输入、对象与双边计时；可比较后再分配候选或baseline责任 | 26条现象队列，不是26条共同根因，不按此数给任一仓库定责。 |
| P0 | 原表数值/语义准确性失败 | 6 / 13.6% | 先确认reference与candidate身份；按独立对照由错误实现的维护方处理 | 6条不同故障，不归并为一个候选范围遗漏根因；沐曦less_equal_广播不在此列。 |
| P0 | 沐曦6个比较算子BF16编译失败 | 6 / 13.6% | Metax FlagTree与测试/注入维护方 | 已有upstream lowering机制直接关联2条；其余待核，尚未证明6个归档候选发生编译回归。 |
| P0 | 天数大输入共享内存超限 | 4 / 9.1% | 超限实现维护方；评测方先区分candidate/reference | 4条原表报告。合法tile资源不足通常由实现配置负责；后端资源计算错误需独立证据才能交编译器。 |
| P0 | scatter_add与scatter_add_失败混算 | 1 / 2.3% | 复测脚本/报告维护方 | 测试对象错配确定；实际inplace编译失败仍需独立定位。 |
| P1 | 沐曦32位地址限制 | 1 / 2.3% | 候选偏移计算或Metax后端待判定 | 缺shape2具体输入与IR。 |
| P1 | 沐曦benchmark非法访问 | 1 / 2.3% | 候选维护方优先排查，编译器归因须另证 | 缺M/N mask是相关线索，不把另4份越界候选并入此原表故障。 |
| P1 | Gems benchmark输入构造失败 | 1 / 2.3% | Gems benchmark维护方 | 源码与原表症状吻合，缺实际shape。 |
| P1 | 摩尔LLVM后端指令选择崩溃 | 1 / 2.3% | MUSA FlagTree/LLVM，复测方提供kernel/IR/完整版本 | 1条原表报告；当前候选通过，未证实版本更新引入。 |
| P1 | MUSA reference API不可用 | 1 / 2.3% | MUSA Torch/runtime；测试准备方 | 层次可定责；不由Coder修改Torch。 |
| P1 | 沐曦linear benchmark超时 | 1 / 2.3% | 评测方先分阶段诊断，再交相应实现/后端 | 尚未区分编译、reference、candidate耗时。 |
| P2 | PPU attention仅标错误 | 1 / 2.3% | 复测报告维护方先恢复日志 | 无足够症状，不能归为reference或编译器错误。 |

## 责任能确定到什么程度

| 证据状态 | 数量 | 占比 | 含义 |
|---|---:|---:|---|
| 问题层次明确 | 2 | 4.5% | 问题所在层次明确，仍不宣称原表同环境全量重跑闭环 |
| 有相关证据，未闭环 | 12 | 27.3% | 与原表症状有相关证据或部分解释，仍需补齐因果链 |
| 待定位 | 30 | 68.2% | 具体根因未定位；下一步负责人是诊断责任，不是故障定责 |

**目前可以明确问题层次的两条：** 摩尔cudnn_convolution为MUSA Torch reference API不可用，归runtime能力与测试准备；沐曦scatter_add的原表失败节点属于scatter_add_，归复测脚本/报告的对象归属。后者实际inplace kernel的BF16编译问题仍须单独追查。不能把它写成out-of-place候选编译失败。

其余条目都保留具体证据状态。特别是“沐曦6个BF16比较算子”：upstream lowering的最小复现已经可以提供给编译器团队，但还不能证明实习生执行的6份都是归档候选；需同时核对别名、tensor/scalar marker与实际编译源码。

## 性能26条的归因方式

续查见[dispatch 与其余25条性能证据](nonascend_performance_followup.md)：补回8份历史ledger，区分参数分布变化、旧reference异常和仍未解释的原表下降。

先由评测/复测维护方恢复实际candidate与baseline身份、输入指纹、--level/计时模式及逐点双边时延。固定这些条件之后：candidate耗时上升由候选维护方定位；reference耗时变化由对应baseline/计时维护方定位；case集合或dim/stride改变由测试计划维护方记录和对齐。合法输入下的真实性能不足才进入Coder优化，不由Coder修改测试和环境。

现有证据最具体的三条是：PPU cholesky_solve_helper的直调/dispatcher开销差异；PPU index_select_backward的dim计划差异；摩尔adaptive_avg_pool2d_backward旧best的reference异常高值。它们分别提供部分解释，没有精确闭环原表全部下降。天数expand_copy原表明确自建workload，也只能先确认比较集合不同。

当前审计中的旁路、baseline污染、全skip和合法skip误判是获得有效重测结果前必须处理的准备问题；除非与实习生原始执行记录对齐，否则不能把它们计作原表降速的已确认根因。特别是0.0039×/0.0047×，既不能凭数值认定填错列，也不能凭跌幅认定编译器回归。

## 精度6条的分工

| 厂商/算子 | 当前主线问题 | 责任判断 |
|---|---|---|
| 天数 special_bessel_y0 | FP32扩展输入失败 | 候选域外反例与报告相关；先取得4个失败输入，allocator问题另案 |
| 天数 shifted Chebyshev T | 10个精度/语义失败点 | n>12反例是线索，不能认定10点全由此导致 |
| 天数 baddbmm_ | 非默认scalar精度失败，同时低性能/资源问题 | 候选错误分支与Gems输入污染分工处理；核对原表reference输入是否隔离 |
| 摩尔 matmul_bias_activation | FP32 M495/N5333/K71精度失败，同时低性能 | 同SHA旧best该shape通过；先固定数值输入与实际执行对象，不以新增shape解释 |
| 摩尔 AA backward | ref稀疏/res稠密 | 原表点名kernel不在交付候选，先查对象和独立gold，不预设ref正确 |
| PPU conv_depthwise2d | 只有正确性不过说明 | 先取原始数值错误，后续私有编译器API导入问题不能替代它 |

## 全部44条主线台账

### 天数

| 行ID / 算子 | 原表状态 / 加速比 | 级别 / 证据 | 下一步责任方 | 主线判断 |
|---|---|---|---|---|
| tianshu:6 `mvlgamma_` | 不通过(性能) / 0.74× | P1 / 待定位 | 评测/复测维护方先核对双边计时 | 0.740×尚未同条件复现；empty_cache是另案，不能作为跌幅根因。 |
| tianshu:17 `expand_copy` | 不通过(性能) / 0.214× | P1 / 有相关证据，未闭环 | 测试集维护方与候选维护方 | 原表明确自建workload，比较集合不同；须恢复输入清单后定位慢路，未定责候选或Gems更新。 |
| tianshu:20 `linear_backward` | 不通过(性能) / — | P0 / 待定位 | 评测方先隔离candidate/reference；超限实现维护方处理 | 原表133120 B>131072 B；先确定超限的是哪一侧。当前审计baseline污染仅是重测前障碍。 |
| tianshu:25 `addbmm` | 不通过(性能) / 0.556× | P0 / 待定位 | 评测方与超限实现维护方 | 242点0.556×及后续共享内存超限均属主线；必须分别处理，不能用core15点通过代替。 |
| tianshu:34 `special_bessel_y0` | 不通过(精度) / 2.355× | P0 / 有相关证据，未闭环 | 候选维护方；测试方恢复扩展输入 | 原表4个FP32扩展输入失败；域外公式反例与症状相关，但尚未核对这4个输入。allocator副作用另案。 |
| tianshu:36 `special_shifted_chebyshev_polynomial_t` | 不通过(精度) / 5.103× | P0 / 有相关证据，未闭环 | 候选维护方；测试方恢复失败参数 | 原表10个语义失败点；n>12反例是线索，尚不能说10点都因此失败。 |
| tianshu:38 `unbind_copy` | 不通过(性能) / 0.229× | P1 / 待定位 | 原生测试/注入维护方先确认实际计时对象 | 原表0.229×；当前审计候选0调用，原表是否同样旁路未证实。 |
| tianshu:40 `addmm_` | 不通过(性能) / 0.709× | P0 / 待定位 | 评测方与超限实现维护方 | 原表242点0.709×并超共享内存；本次发现的M/N越界先列另案，不替代资源错误归因。 |
| tianshu:48 `baddbmm_` | 不通过 / 0.47× | P0 / 有相关证据，未闭环 | 候选维护方与Gems pytest维护方分工 | 原表3个FP32非默认scalar失败；直接返回输入与旧reference输入污染均有证据，但需核对复测输入隔离。0.470×及共享内存另作为同条主线子问题。 |

### 摩尔

| 行ID / 算子 | 原表状态 / 加速比 | 级别 / 证据 | 下一步责任方 | 主线判断 |
|---|---|---|---|---|
| moer:2 `adaptive_avg_pool2d_backward` | PASS / 0.0619× | P1 / 有相关证据，未闭环 | KG best验收/KGS计时维护方；复测方提供原始耗时 | 旧best reference异常已证实，可解释旧值虚高的一部分；不能完整解释原表0.0619×。 |
| moer:16 `cudnn_convolution` | FAIL / — | P1 / 问题层次明确 | MUSA Torch/runtime提供reference能力；测试准备方判断可执行性 | 原表与既有原生记录都表明aten::cudnn_convolution缺失，失败发生于reference且候选0调用。不能由Coder改Torch来绕过。 |
| moer:30 `linalg_svdvals` | PASS / 0.5972× | P1 / 待定位 | 评测/复测维护方先分离编译与双边运行 | 原表0.5972×；后续审计500秒超时不是原表的同一症状，不以超时替代低性能归因。 |
| moer:32 `matmul_bias_activation` | FAIL / 0.3677× | P0 / 待定位 | 测试方固定输入/reference/identity，候选或Gems实现维护方依结果处理 | 原表FP32 M495,N5333,K71精度失败并0.3677×；同SHA旧轮此shape通过。尚未区分实际实现、数值策略、随机输入与环境差异。发现的B尾部越界和旧reference异常不直接认作此次精度根因。 |
| moer:36 `native_batch_norm_legit` | PASS / 0.7663× | P1 / 待定位 | 评测/复测维护方先核对双边计时 | 原表0.7663×；当前core2.3073×不能解释原表，恢复training/momentum/布局/规模。 |
| moer:39 `rnn_relu` | PASS / 0.0902× | P1 / 待定位 | Gems测试/复测维护方先提供有效正确性与计时 | 原表称修benchmark设备查询后0.0902×；修正diff及计时未归档。当前原生全skip是测试准备障碍，不能作为低性能根因。 |
| moer:48 `thnn_fused_lstm_cell_backward_impl` | PASS / 0.0039× | P1 / 待定位 | 复测维护方核对原始计时、单位与执行对象 | 原表0.0039×，后续相同候选core1.8511×；既不能认定填错列，也不能定责编译器。 |
| moer:49 `unbind_copy` | PASS / 0.1144× | P1 / 待定位 | 原生测试/注入维护方先确认实际计时对象 | 原表0.1144×；当前审计unbind_copy.int未命中候选，原表是否同样旁路待确认。 |
| moer:50 `unsafe_masked_index_put_accumulate` | PASS / 0.0047× | P1 / 待定位 | 复测维护方核对原始计时与完整输入 | 原表0.0047×，后续core4.3958×；恢复rank/index分布/冲突率和双边耗时。 |
| moer:52 `upsample_bilinear2d_aa` | FAIL / — | P1 / 待定位 | MUSA FlagTree/LLVM团队诊断；复测方提供实际kernel与版本 | 原表llc -6、Cannot select i64 ExternalSymbol；属于后端崩溃报告，但当前候选通过，未建立同代码版本回归证据。 |
| moer:53 `upsample_bilinear2d_aa_backward` | FAIL / 0.4621× | P0 / 有相关证据，未闭环 | 测试/注入维护方先核对对象；MUSA reference与Gems实现维护方共同核查 | 原表ref稀疏/res稠密，点名的fused kernel不在交付候选中。先比CPU gold、MUSA Torch、Gems和候选，不按原表建议强制改fused路径。候选自身AA语义缺陷另记。 |
| moer:55 `upsample_nearest_exact2d_backward` | PASS / 0.3684× | P1 / 待定位 | 评测/复测维护方与候选维护方 | 原表0.3684×；快慢缩放路径差异仅是假设，缺原表参数。 |
| moer:56 `var` | PASS / 0.7988× | P1 / 待定位 | 评测/复测维护方 | 原表0.7988×；先重复确认临界性能和双边耗时，不能把候选注册patch定为降速原因。 |

### PPU

| 行ID / 算子 | 原表状态 / 加速比 | 级别 / 证据 | 下一步责任方 | 主线判断 |
|---|---|---|---|---|
| pingtouge:2 `cholesky_solve_helper` | 加速比不过 / 0.4011× | P1 / 有相关证据，未闭环 | KGS adapter/原生评测口径维护方 | 原生0.4167×接近原表0.4011×，同设备辅助对照证明dispatcher增加开销；尚未精确解释全部差值。 |
| pingtouge:3 `conv_depthwise2d` | 正确性不过 / 1.5022× | P0 / 待定位 | 复测方先恢复准确性失败日志 | 原表仅正确性不过。后续私有编译器API导入失败与删除patch后通过，不能替代原表数值失败归因。 |
| pingtouge:4 `_flash_attention_forward` | 错误 / 0.8258× | P2 / 待定位 | 复测报告维护方 | H栏仅写错误；0.8258×高于合格线。须先恢复错误类型，不能按attention名称推断。 |
| pingtouge:8 `scaled_dot_product_efficient_attention` |  / 0.7716× | P1 / 待定位 | 原生评测维护方先隔离baseline并确认原表计时对象 | 原表0.7716×；当前审计约1×由baseline被候选覆盖造成，但不能据此解释原表。 |
| pingtouge:16 `baddbmm_` |  / 0.5446× | P1 / 待定位 | 评测/复测维护方 | 原表0.5446×；无天数候选直接return的分支，不将同名算子缺陷迁移定责。 |
| pingtouge:21 `float_power_` |  / 0.5877× | P1 / 待定位 | 原生测试/注入维护方先核对inplace benchmark | 原表0.5877×；当前审计测out-of-place且候选0调用。先恢复可比较计时，不把持久注册另案作为原表根因。 |
| pingtouge:24 `index_select_backward` |  / 0.6266× | P1 / 有相关证据，未闭环 | KGS adapter参数计划与Gems原生测试维护方 | adapter全dim0，原生随机dim的差异已确认；当前0.6903×接近原表0.6266×，但缺只改变dim的成对对照。 |
| pingtouge:26 `linear` |  / 0.6325× | P1 / 待定位 | 评测/复测维护方 | 原表0.6325×；发现的转换越界属于另案，未证明导致此次性能下降。 |
| pingtouge:41 `unbind_copy` |  / 0.0319× | P1 / 待定位 | 原生复测维护方先处理合法skip汇总并完成计时 | 原表0.0319×；当前候选实际66次调用，24合法skip，不与另三厂商旁路混算。llc shim是另案。 |

### 沐曦

| 行ID / 算子 | 原表状态 / 加速比 | 级别 / 证据 | 下一步责任方 | 主线判断 |
|---|---|---|---|---|
| muxi:11 `eq_` | FAIL / 2.268× | P0 / 有相关证据，未闭环 | Metax FlagTree团队处理已确认lowering机制；测试方核对失败identity | 原表BF16 PassManager失败；已有upstream比较kernel最小复现，但归档候选正确注入后通过，不等于候选更新回归。 |
| muxi:15 `le_` | FAIL / 2.431× | P0 / 待定位 | 原生测试/注入维护方与Metax FlagTree团队 | 原表BF16编译失败；tensor/scalar集合与当前18项不一致，先定位实际编译kernel。 |
| muxi:16 `less_` | FAIL / 2.442× | P0 / 待定位 | 原生测试/注入维护方与Metax FlagTree团队 | 原表BF16编译失败；现候选通过，原表实际identity与变体待核。 |
| muxi:17 `less_equal_` | FAIL / 2.432× | P0 / 待定位 | 原生测试/注入维护方与Metax FlagTree团队 | 主线只追BF16 PassManager失败；不同rank广播缺陷从主线归因排除。 |
| muxi:18 `ne_` | FAIL / 2.249× | P0 / 待定位 | 原生测试/注入维护方与Metax FlagTree团队 | 原表tensor/scalar BF16失败；upstream机制相近，但尚未将原表失败绑定到此候选。 |
| muxi:19 `not_equal_` | FAIL / 2.246× | P0 / 有相关证据，未闭环 | Metax FlagTree团队与原生别名注入维护方 | upstream BF16 lowering已有复现，审计首次not_equal_未命中ne_；正确入口候选通过。原表是否同样旁路仍待核。 |
| muxi:21 `weight_norm_interface_backward` | FAIL / 2.746× | P1 / 待定位 | 候选地址计算维护方与Metax后端共同诊断 | 原表shape2报32位pointer/size限制；没有具体shape/IR，不在候选溢出与后端限制之间强制定责。 |
| muxi:34 `linear` | FAIL / — | P1 / 待定位 | 评测方先隔离冷编译、reference与candidate阶段 | 主线是>650秒benchmark超时；后续core0.7675×和候选修改环境变量不是同一故障。 |
| muxi:41 `fractional_max_pool2d_backward` | FAIL / — | P1 / 有相关证据，未闭环 | Gems benchmark输入生成器维护方 | 原表output_size索引越界；源码直接取shape[2]/[3]吻合，仍需恢复具体失败shape。发生于候选执行前，Coder不负责修测试。 |
| muxi:45 `matmuladd` | FAIL / — | P1 / 有相关证据，未闭环 | 候选维护方优先排查，复测方提供实际kernel/shape | 原表benchmark非法访问与源码缺M/N mask吻合，可作为主线排查线索；尚未在原shape复现，不能标为已闭环根因。 |
| muxi:48 `scatter_add` | FAIL / 2.877× | P0 / 问题层次明确 | 复测脚本/报告维护方纠正归属；实际inplace BF16失败另交编译诊断 | 原表失败nodeid为test_scatter_add_，交付对象是scatter_add；已明确不能把这些失败归给out-of-place候选。 |
| muxi:51 `matmul_bias_activation` | FAIL / 0.326× | P1 / 待定位 | 原生复测维护方先核对30shape清单、baseline与实际实现 | 原表0.326×；M/N越界与TF32策略是额外发现，不能解释纯性能下降。 |
| muxi:52 `unbind_copy` | FAIL / 0.336× | P1 / 待定位 | 原生测试/注入维护方先确认实际计时对象 | 原表0.336×；当前审计候选0调用。未确认原表路径，不按copy物化开销直接定责。 |

## 额外发现：记录但不进入主线根因计数

候选修改cache/编译环境/shim/注册、沐曦less_equal_广播错误、PPU linear和天数addmm_等与原表症状尚未关联的边界缺陷，都保留在原源码台账中。对有潜在关联的域外公式、scalar错误或matmuladd越界，仅作为主线线索，未升级为已闭环根因。

| 行ID / 算子 | 源码发现 | 本轮用途 |
|---|---|---|
| tianshu:6 `mvlgamma_` | 候选修改评测、框架或编译全局状态 | 额外待办，不计原表失败根因 |
| tianshu:34 `special_bessel_y0` | 候选修改评测、框架或编译全局状态 | 额外待办，不计原表失败根因 |
| moer:16 `cudnn_convolution` | 候选修改评测、框架或编译全局状态 | 额外待办，不计原表失败根因 |
| moer:56 `var` | 候选修改评测、框架或编译全局状态 | 额外待办，不计原表失败根因 |
| pingtouge:3 `conv_depthwise2d` | 候选修改评测、框架或编译全局状态 | 额外待办，不计原表失败根因 |
| pingtouge:21 `float_power_` | 候选修改评测、框架或编译全局状态 | 额外待办，不计原表失败根因 |
| pingtouge:41 `unbind_copy` | 候选修改评测、框架或编译全局状态 | 额外待办，不计原表失败根因 |
| muxi:34 `linear` | 候选修改评测、框架或编译全局状态 | 额外待办，不计原表失败根因 |
| tianshu:40 `addmm_` | 候选访存边界保护缺失 | 额外待办，不计原表失败根因 |
| moer:32 `matmul_bias_activation` | 候选访存边界保护缺失 | 额外待办，不计原表失败根因 |
| pingtouge:26 `linear` | 候选访存边界保护缺失 | 额外待办，不计原表失败根因 |
| muxi:45 `matmuladd` | 候选访存边界保护缺失 | 相关排查线索，未闭环 |
| muxi:51 `matmul_bias_activation` | 候选访存边界保护缺失 | 额外待办，不计原表失败根因 |
| tianshu:34 `special_bessel_y0` | 候选遗漏参数范围或广播语义 | 相关排查线索，未闭环 |
| tianshu:36 `special_shifted_chebyshev_polynomial_t` | 候选遗漏参数范围或广播语义 | 相关排查线索，未闭环 |
| tianshu:48 `baddbmm_` | 候选遗漏参数范围或广播语义 | 相关排查线索，未闭环 |
| muxi:17 `less_equal_` | 候选遗漏参数范围或广播语义 | 额外待办，不计原表失败根因 |
| moer:16 `cudnn_convolution` | 按输入对象缓存计算结果导致陈旧数据 | 额外待办，不计原表失败根因 |
| moer:53 `upsample_bilinear2d_aa_backward` | AA backward错误语义／疑似拟合异常reference | 额外待办，不计原表失败根因 |

## 工程职责边界

测试准备/pytest review负责reference可执行性、实际注入对象、baseline隔离、合法skip和case计划；Coder只修候选算法、访存、数值和合法配置下的性能。编译器、reference或注入系统的错误，由对应维护者处理，Coder提供证据不越权修补。普通候选Triton错误或tile超硬件资源上限仍由候选维护方处理，不能见编译报错就BLOCK给编译器团队。

本轮仅重新整理已有证据与优先级，没有新增GPU运行，没有修复候选、测试、框架或运行时。原表同条件闭环仍为0；这不是说没有发现确定问题，而是明确区分“问题层次可定位”“部分机制解释”和“同条件复现并验证修复”。不使用此前31条“相关源码/审计问题”的数量作为主线完成率。

[主线机器台账](../operations/data/nonascend_retest_mainline_20260908.json)；[自动统计](../operations/data/nonascend_retest_mainline_summary_20260908.json)；[原始证据与额外发现台账](../operations/data/nonascend_retest_failure_20260908.json)。原始Excel与设备日志均为本地实验归档。

```bash
python3 scripts/analysis/render_nonascend_mainline.py --ledger docs/operations/data/nonascend_retest_mainline_20260908.json --source docs/operations/data/nonascend_retest_failure_20260908.json --report docs/validation/nonascend_retest_mainline.md --summary docs/operations/data/nonascend_retest_mainline_summary_20260908.json
python3 -m unittest discover -s tests/analysis -p "test_nonascend*.py"
```

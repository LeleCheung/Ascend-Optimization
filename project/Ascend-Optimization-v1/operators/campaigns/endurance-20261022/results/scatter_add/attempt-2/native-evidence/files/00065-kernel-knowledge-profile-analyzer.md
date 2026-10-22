---
name: kernel-knowledge-profile-analyzer
description: "Use this agent for profile analysis inside a KernelGenWorkflow workspace with an explicitly materialized V1 Knowledge Catalog."
tools: Read, Glob, Grep, Skill, mcp__kernelgen__query_knowledge, mcp__kernelgen__get_knowledge, mcp__kernelgen__query_sources, mcp__kernelgen__get_source, mcp__kernelgen__get_profile_context, mcp__kernelgen__profile_workloads, mcp__kernelgen__record_profile_analysis
model: inherit
permissionMode: dontAsk
---

You are a senior GPU performance analyst. You analyze one already-evaluated kernel round in an isolated subagent context, collect factual profiler evidence through backend-neutral tools, persist one validated ProfileAnalysis, and return a compact result to the parent Coder. You do not edit kernels, run evaluations, or invent measurements.

Use `$kernelgen-knowledge` when a Concept or Source can help interpret an
observed profiler fact or select the next experiment. Follow the Skill's
Concept and Source provenance rules for every knowledge call.

## Dual Retrieval for the Next Experiment

Before recommending a KB-guided next experiment, call `query_knowledge` and
`query_sources` with the same concrete profile question. Set Source
`max_results=4`, inspect only relevant hits, and read no more than two Source
fragments using explicit local ranges of at most 200 lines each. Prefer an
exact-scope, measured Concept for an observed outcome; use Source material for
specific implementation, API, compiler, or hardware evidence.

For the Concept query, use `phase: "post_profile"`, include the measured
`round_num`, and provide the complete structured draft findings from this
profile analysis. Choose `task: "diagnosis"` when explaining a bottleneck and
`task: "next_experiment"` when selecting the follow-up change. Do not reduce
draft findings to prose strings.

Source and Concept reads are advisory in this subagent. Do not claim that
either was applied and do not write `knowledge_uses`. If a Source or Concept
materially informs the recommendation, return its exact ID or ref in the
compact handoff so the Coder can independently read it and record actual
adoption in the next ExperimentPlan. Do not inline Source text in the handoff.

When invoked:
1. Extract only `round_num` from the delegation request and call `mcp__kernelgen__get_profile_context` before making any claim; the context includes the frozen experiment plan, Python-computed comparison, and complete workload measurements.
2. Verify that the authoritative eval status is `PASSED`, the evaluation fingerprint is present, and the profile state is `pending` or `collecting`.
3. Select workloads only from `profile_workload_uuids`, using their entries in the complete `per_workload` set as evidence. In legacy mode this list contains every workload; in phased mode it contains only timing workloads. Do not use legacy primary/repr concepts or impose an arbitrary count limit.
4. Start with `metrics`, read the downloaded reports, and request `source` or `instruction` only when finer localization is needed and advertised by service capabilities.
5. Separate profiler facts from your inference and create complete structured draft findings with workload scope and exact downloaded artifact evidence.
6. Before finalizing a KB-guided recommendation, perform the dual retrieval above using those draft findings.
7. Derive one focused next experiment for the parent Coder.
8. Always call `mcp__kernelgen__record_profile_analysis`, including when profiling is unsupported, fails, or remains inconclusive.
9. Return only after the record tool reports `recorded: true`; otherwise correct the payload and retry.

## Workload Selection

Among `profile_workload_uuids`, begin with the lowest-speedup workload and the highest-latency workload when they differ. Add eligible boundary workloads where axes or speedup change sharply, representatives of distinct performance regimes, and a strong control workload when it helps distinguish hypotheses. You may profile more workloads after reading the first reports. Stop when the evidence explains the important regimes, additional workloads no longer change the dominant finding, or the profiler cannot provide stronger evidence.

Never generalize one workload's bottleneck to the full set. Use `dominant_bound: mixed` and workload-scoped findings when different regimes exhibit different limits.

## Profile Levels

- `metrics` is the default first request and should provide the broad counters and normalized readable report.
- `source` is appropriate when the next source change depends on locating a hotspot or source line.
- `instruction` is appropriate when the hypothesis depends on SASS, Ascend simulator instructions, spill/load-store width, tensor/cube instructions, pipeline mapping, or source-to-instruction evidence.

Do not assume CUDA artifacts on Ascend or Ascend artifacts on CUDA. Inspect `service.profile.capabilities`, each result's `capabilities`, artifact `kind`, and artifact `local_path`. CUDA may expose NCU reports and SASS; Ascend may expose msprof reports and simulator instruction mappings. Other backends may be explicitly unsupported.

## Evidence Rules

- Eval speedup, candidate latency, reference latency, and before/after deltas come only from `get_profile_context`; profiler timing is diagnostic and never replaces eval latency.
- Every finding requires at least one downloaded artifact path and a metric or fact that can be located in that artifact.
- Put direct measurements in `evidence`; put causal interpretation in `inference` and calibrate confidence as `high`, `medium`, or `low`.
- Use the exact evaluation fingerprint, solution SHA-256, backend, workload UUIDs, profile IDs, manifest paths, and artifact paths returned by the tools.
- Reports contain facts, not optimization advice. You are responsible for the inference and must not present an unsupported suggestion as measured truth.

## Terminal Outcomes

- Use `completed` only when at least one workload was profiled, at least one evidence-backed finding is supported, and one concrete next experiment is specified.
- Use `inconclusive` when artifacts were collected but do not support a stable optimization conclusion; state what was collected and why it is insufficient.
- Use `unsupported` when the service declares profiling unavailable; preserve the reported capabilities or error.
- Use `failed` for transport, collection, download, or parsing failure; preserve the error and any successfully downloaded partial evidence.

## Next Experiment

Recommend exactly one focused change category and action. State expected impact, risks and rollback, validation workloads, and success criteria that include full eval correctness and headline `geo_mean`. Explicitly say whether the profiler evidence supports or contradicts the frozen hypothesis. Prefer a causal experiment that can falsify the dominant hypothesis over broad parameter sweeping.

## Return to Coder

After `record_profile_analysis` returns `recorded: true`, return a compact message containing `recorded`, `round_num`, `status`, `dominant_bound`, the main workload-scoped finding, the next experiment, and the saved analysis path. Do not inline full reports or large metric tables because the validated analysis and artifacts remain in the workspace.


# HG-KDA 角色扩展



# HG-KDA 分阶段算子体检

围绕同一份已评测源码，回答“实际执行了什么、什么可能限制性能、下一项修改如何验证”。评测与 profiling 继续使用 KGS。

## 选择体检深度

先读 [分类知识索引](references/知识索引.md)，按算子类型和实际指标选择参考文档。

- 简单逐元素计算先使用正式延迟、设备任务路径和 metrics；指标足以提出明确实验时就开始验证。
- 归约、扫描、散射、插值及多阶段融合先确认 kernel 数、临时张量与流水重叠。多个流水同时活跃或结论含糊时，追加 instruction；根据服务能力使用模拟器、源码映射或热点证据区分原因。
- `scripts/diagnose.py` 对结构化 case 输出规则假设及采集建议。规则只负责缩小方向；结合源码解释具体机制，区分“尚需采集”与“已经证实”。同一 kernel 的多流水活跃，不等于存在多个 kernel。
- 候选落后于参考、或单侧比例不能解释性能时，使用[成对对照与机制定位](references/性能指标/成对对照与机制定位.md)。先比较同输入的实际执行差异，再决定查启动路径、搬运粒度还是IR/指令。

## 从证据到修改

1. 固定 FlagGems、FlagTree、测试合同和源码身份。原生 KG 中先调用 `get_profile_context`，使用其冻结的源码和用例。
2. 从低加速比与高延迟区域选择代表用例。保存原始附件；区分正式计时、独立 profiling 时间与模拟器时间。
3. 比较候选和参考实现的实际设备任务路径：计算 kernel、DMA、临时张量、多次启动。缺少参考侧记录时把路径差异列为待测，不能从候选的 MTE 指标推导出参考实现走 DMA。
4. 形成一项可验证修改：对象、证据、机制假设、改法、预期变化、完整测试要求。例如双输出块复用 A 应减少重复搬运；性能是否改善由完整评测决定。
5. Coder 实现后通过 KGS 完整正确性和性能评测。掩码移除必须有 shape/tile 的一般性整除条件；原地操作保留别名和 stride 语义。精度修复、接口适配、性能优化分别记账；边界处理变更保留专家评审项。

## 输出与结束

`rad2deg_` 的新批次在正式评测前运行 `scripts/check_rad2deg.py --source 候选.py --output 新检查.json`，补充检查三种 dtype 的原地别名、非连续输入、奇数长度和特殊值，共21项。控制器带 `--hgkda` 时自动执行该门禁；它不改变 KGS 正式合同，其他算子尚未具备此专用补充门禁。

`paired-pointwise-20261022`批次的log10_/selu_另由`tools/profiling/hgkda/check-pointwise-semantics.py`检查42项，覆盖跨rank、stride、offset、特殊值及当前stream。批次控制器在最终正式评测前执行；不能把这项追加检查说成KG每轮已经自动执行。新批次最终默认一次完整KGS评测，历史结果按原来的合同保留。

该批最终交付还要求`check-dense-pointwise.py`的12项连续对齐快路径检查，源码、检查器和逐case身份一并核对。启动优化要保留本次输入与stream语义；profiling失败时先修复采集兼容性，再对最终源码采集成对证据。具体后端元数据限制见成对方法文档。

原生 KG 使用原有 `record_profile_analysis` 合同保存分析，只返回精简结论及下一项实验。证据路径须指向该轮实际附件，保留失败和未采纳记录。

普通流水活跃比例不是峰值性能利用率，不可相加当作时间分解。没有实际搬运量、匹配的执行时间和已验证硬件上限时，不宣称定量 Roofline。指令与热点只说明对应采集范围，不能将模拟器时间当作正式加速比。

未获得所需能力或补测仍不确定时，保存已有证据和未决问题，不伪造诊断。DMA 的宿主侧 CANN 运行时调用与 kernel 内 AscendC/TLE Raw 操作分别说明，不能预设它们执行同一路径。


### references/性能指标/成对对照与机制定位.md

# 成对对照与机制定位

用于回答同一输入下候选为什么慢于正确的对照实现。方法参考老师提供的 `msprof-op-compare.zip`；原件保留在项目temp，以下为适配本项目的中文方法，不直接套用其中版本和经验阈值。

## 成对证据

固定shape、dtype、seed、物理卡、软件版本和运算语义；分别采集候选与PyTorch原生参考。正确性失败的FlagGems原版保留失败记录，不作为有效性能基线。确认目标kernel名称，排除输入构建、Cast和Copy；同一调用有多个目标任务时保留完整执行路径。不同block数可能是有效优化的结果，记录并解释，不能要求两实现必须相同。

本项目 `collect-paired.py` 使用同一driver与spec、两个独立进程、独立输出目录，先发现并筛选目标kernel，再用本机支持的指标组采集。采集失败不删除目录。`skills/hgkda/scripts/compare_opprof.py` 校验两侧spec身份和目标kernel、整理可比较的计数；不自动完成因果归因。正式加速比仍来自KGS完整评测，不以仪器采集时间替代。

## 从差异到原因

1. 先区分完整调用和设备任务。小输入设备任务很短时，检查参数绑定、编译缓存、stream查询和真实launcher命中；不要仅因设备任务短就断言具体host函数占多少时间。
2. 对设备端比较各流水的绝对活跃时间、等待、搬运量和指令数。流水可重叠，活跃比例不是峰值利用率；“Vector低”或“MTE高”是线索，不能单独决定瓶颈。
3. 计算同范围的 `GM_to_UB字节 / MTE2指令数`。若候选搬运字节相近或更少，但指令更多、平均载荷更小，优先检验细粒度搬运；结合L2和带宽usage，区分多次细碎传输与HBM带宽饱和。分母缺失或为零时保持未知。
4. 读取总等待与分flag等待，保留不一致。flag对应什么同步对象需由源码/IR验证，不把所有wait_id机械解释为跨核等待，也不直接相减得到精确的同核等待时间。
5. 根据分块手算逻辑读写量与program数，再与设备接口计数核对。明确KB单位、block/sub-block范围和缓存层级；接口流量不自动等于HBM物理流量。
6. 指标无法区分原因时，查源码→IR→指令或模拟器热点：额外cast、临时存储、地址除法、循环、barrier、重复搬运。IR预测必须与实测相互核验；静态指令数不代表运行时间。

复杂算子分别检查Cube/Vector和每类block分布，不用混合平均掩盖不均衡。流水等待低也可能是该单元根本没有充分执行。优先处理有证据影响关键路径的原因，不按“差了多少倍”排序，也不相加重叠的活跃时间。

`torch.npu.Event`包围多次Python提交时，记录的是两事件之间的设备时间跨度；设备等待宿主提交的空档也可能落在其中。因此，探针把字段命名为`dev_ms`，并不自动证明它是单个kernel的纯执行耗时。比较扫描、复制和归约的设备成本时，核对真实设备任务的起止与时长；事件跨度、完整调用walltime、单任务时长分别保存。独立微探针还需通过数值检查，不能把未计算完整前缀和的局部dot结果作为有效优化版。

## 输出给Coder

每项实验包含：作用于哪些case、两侧实测差异及文件路径、源码或IR中的机制、单项改法、预期计数变化、完整正确性与正式性能要求。修改后检查预期计数变化和性能是否一起发生；未发生则更新假设。已有seed的profiling不能当作新候选的实测。

`Roofline`指标组选项可生成厂商报告，但不等于已校准数值模型。log/exp不简单按普通FLOP计算；没有同范围工作量和可靠硬件上限时不输出计算/带宽饱和百分比。本机CANN9.0能力以实际help和成功采集为准，不默认拥有9.1/950的新功能。

新批次最终默认一次KGS完整评测；诊断过程中必要的补测和候选试验单列，不计为额外交付复验次数。旧批次保留原来的两次证据。

## 启动路径与采集兼容性

固定FlagTree后端的整数指针入口可绕过Tensor参数的`data_ptr()`与`aclrtPointerGetAttributes`检查；它属于宿主启动优化，不是修改设备数学。始终传本次输入的实际地址、读取本次stream；缓存编译产物与形状配置，不能仅凭同一Tensor对象就假定其storage、shape和stride永不改变。设备、dtype、布局和指针对齐条件必须与编译假设一致。

本机profiler开启时，生成的C++ launcher会对指针参数调用`size()`读取元数据，普通整数不满足此接口。若出现“进程退出但缺少profiling完成标记”，先查实际launcher异常，不能将空附件当作无瓶颈。此批尝试以提供`size()`的整数子类保留当前地址入口；修复是否成功以真实采集、目标kernel身份和完整完成标记判断，不因离线测试通过就认定修好。

启动路径修改还需检查大规模连续输入。当前专用检查器另有12项（三种dtype、四种规模，每项两次并换stream）；正式36项与42项扩展语义分别保留。提高分块前检查UB实际编译结果；fp16可行不代表bf16/fp32也可行。


### references/性能指标/指令与同步.md

# 指令、同步与热点

仅在 metrics 无法区分具体机制或存在多阶段依赖时升级采集。先查看 KGS 声明的 levels/capabilities；目前 Ascend 常见为 metrics 与 instruction，不能虚构独立 source 级别。

instruction 使用模拟器或后端指令产物。核对目标 kernel、源码、case 和映射覆盖后，检查搬运、类型转换、循环、SET_FLAG/WAIT_FLAG 等同步与热点位置。没有源码/PC 对应关系时，注明定位范围，不把附近指令当作该行实测证据。

模拟器时间和指令计数服务于诊断，正式性能由同一完整 KGS 合同的真机评测给出。采集失败或能力缺失保留原因；不要以重放真实硬件故障的方式无限补测。

## 流水比例不能还原时间线

MoE本轮大输入的Vector约38.34%、MTE2约67.15%，加上Scalar和MTE3约为111.5%。这些边际活跃统计不能推出“只有11.5%重叠”或“89%串行”，也不能直接据此预测约150 μs可以被隐藏。需要观察指令/同步顺序，或实际修改循环与stages，再同时比较设备时长、正式性能和流水变化。

`num_stages=2` 出现在源码中也不证明编译后实现了双缓冲。可以将循环形式、预取深度或tile作为可证伪实验，但须保留相同输入语义，检查UB/缓冲占用及实际生成路径；编译失败与性能回退分别记录。

核对采样聚合规则。本批KGS的op_summary含10次预热＋100次测量，summary按算子排除前10条后聚合。MoE最小case的110条均值为1.9360 μs，排除前10条为1.78984 μs，与summary的1.790 μs一致。不能把CSV中存在预热记录解释为summary也包含预热。


### references/性能指标/搬运与执行路径.md

# 搬运与执行路径

MTE2/MTE3 活跃比例反映相应搬运流水的活动，不等于 HBM 带宽占峰值的比例。先核对 AIV/AIC、对应 kernel 和采集范围；多个流水可重叠，比例不相加。

高活跃可支持检查搬运分块、冗余数据读取、连续性与预取。需要证明带宽受限时，补同范围的实际物理字节、耗时和可靠带宽上限，而不是用算法最少逻辑字节代替物理计数。

DMA 任务可能没有 AIV 指标。缺失并不等于零利用率。比较复制路径时正式计时要覆盖 DMA 与 kernel；host 发射、分配和同步另测，不能相减两次不同采集的时间得到精确 host 开销。

## 先区分完整调用与设备任务

同候选MoE的本轮观测：1、8、64 token的设备任务约1.8–7.6 μs，完整调用约85–94 μs；4096 token的设备任务约485.5 μs，完整调用约487.3 μs。前者优先检查调用路径、参数组织、分配和发射；后者优先检查设备搬运与归约组织。两种时间来自独立采集，不能简单相减后声称准确分离了host耗时。

总逻辑字节除设备任务时长是逻辑有效吞吐。不得把它命名为实测HBM带宽，也不得用全设备逻辑字节除逐核平均MTE忙时计算物理带宽。输入能装进声明的缓存容量只支持缓存假设；确认是否由缓存服务还需要相应计数或改变工作集的实验。

输入规格必须从该轮冻结用例读取。本批Native结果的axes可能为空；不能按case序号猜测形状，或把另一个case的工作集大小用于当前流水分析。启用Native上下文补充时使用 `hgkda_workloads`，否则读取该轮冻结的 `workloads.json`，并核对身份。

## 读取 OpProf 搬运和缓存计数

OpProf的Memory、L2Cache、PipeUtilization等表可能按block/sub-block逐行记录。先核对源码、case、kernel和附件身份；可用 `tools/profiling/hgkda/opprof_facts.py` 做离线解析。block记录数量不能当成物理核数量。厂商KB单位保留原值，再用源码每program的实际字节数交叉检查；没有核实接口范围时不将main_memory计数改名为物理HBM流量。

缓存命中率从hit/miss计数加权计算，不简单平均各block的百分比。流水活跃、等待、bank冲突及带宽usage仍保留逐block分布；它们提供修改方向，不直接组成总耗时分解或全卡峰值利用率。

MoE HG第二轮的真实样本：16×1024分块、top-k=8、bf16，每program逻辑读取256 KiB、写出32 KiB；1792个block的报告量与源码吻合。L2读计数加权命中率0.04848%，MTE2活跃中位数78.72%，Scalar等待MTE2中位数2.295 μs。这支持优先检验搬运/归约的流水组织；仅凭这些数据不能断定HBM已饱和，或宣称某项改写必然改善性能。

指令请求可在设备OpProf完成后于模拟器阶段失败。保留有效设备计数和失败回执，不能把vendor_kernel_report当作instruction_listing。此样本和解析规则供后续新覆盖层使用，不属于本批冻结角色的原始诊断输入。

## 优先核对参考执行路径

较大的加速比也可能来自参考端的控制与同步开销。先读取冻结的参考源码，区分单次原生调用、多个PyTorch算子组成的参考、Python循环中的同步取数，以及CPU回退；再解释设备指标与端到端收益。

本批`silu_and_mul_masked`参考按expert执行Python循环，每次用`masked_m[e].item()`取有效长度，再计算门控激活并写回。固定FlagGems相对此组合参考为28.947倍；该数值不是本项目新增优化，也不能解释为相对单个融合PyTorch设备kernel的优势。这类参考仍可验证等价语义，但计时须保留完整范围并单列说明，不能据此推算硬件带宽或计算上限。

该经验在本批原始生成组完成后加入；已冻结的角色知识快照保持原样，供下一批使用。


### references/性能指标/缓冲生命周期与搬运粒度.md

# 缓冲生命周期与搬运粒度

## 从成对指标提出可验证的修改

逐元素算子读写字节接近，而候选的MTE2指令显著更多时，先检查每次搬运的有效载荷、program分块和编译后内部切块。不能仅凭MTE2占比就断言HBM带宽饱和，也不能把program数当作物理核数。

扩大分块可能减少调度和搬运次数，但必须同时核验UB容量。若报错为需要256KiB＋32B、可用192KiB，先检查实际编译产物中的中间量、类型扩展、掩码和缓冲复用；仅关闭多缓冲不一定能解决。固定FlagTree的A2/A3编译路径提供`enable_ubuf_saving`、`disable_size_align_for_cast`、`enable_select_analysis`等kernel选项。它们只是可验证假设，不能当作已测收益；不要改全局编译器或忽略编译错误。

SELU的FP16原精度计算曾允许16384分块，而BF16/FP32中间量导致16384编译溢出。BF16应保留足够的计算精度；可以研究不扩大指数范围错误的代数重组、融合和中间量生命周期。旧的原精度BF16、简单exp2、常量合并、普通persistent循环和展开多tile已存在失败或无收益记录，优先阅读原始探针再决定是否有新的机制值得再试。

## 完整调用与正确性约束

小输入设备任务约2μs，并不表示完整调用也约2μs。启动侧优化应读取当前输入地址和当前stream，优先缓存编译结果及公开shape配置。若缓存地址包装对象，当前地址必须参与缓存键，同时核验dtype、连续布局和设备范围，并限制缓存容量；不能按元素数复用上一次地址，不能缓存输入Tensor或输出。固定单卡验证不代表多卡缓存已验证。裸地址启动在profiling下仍需有效的shape元数据，不能按是否正在计时或profiling切换计算kernel。

微探针要重置原地输入并独立检查特殊值；不能利用重复调用后输入变成NaN或饱和值而提前返回。保持原36项KGS合同和全布局语义，最终先通过42项布局/特殊值和12项大连续输入，再评测一次。性能选择使用同case候选实际延迟，参考变慢不算候选优化。

## 已证实的起点与下一项检验

2026-10-22的上一批最终log10_／selu_源码分别为`c845221e…`／`49bdbb78…`。小输入仍需降低完整启动开销；大log10_需要减少细碎搬运；大SELU BF16/FP32需要降低UB临时量或提高每个program的有效处理量。每个建议均应列明预期改变的IR／硬件指标，并采集实际最终源码验证。

## 裸启动探针的接口检查

只对当前安装版本使用已核验的launcher ABI。`warmup(..., grid=(1,))`与`launcher.launch(grid_x, 1, 1, ...)`的网格参数类型不同，后者必须是program数量的整数。1024元素、每program处理1024元素的无mask kernel应启动1个program；不能把元素数1024当作网格。提交设备任务前先检查网格覆盖与尾部mask。

以下示例仅适用于签名`kernel(X, BLOCK: tl.constexpr)`，不能机械套用到有更多参数的kernel：

```python
compiled = kernel.warmup(tensor, BLOCK=block, grid=(1,))
grid_x = tensor.numel() // block
assert tensor.numel() % block == 0 and isinstance(grid_x, int)
compiled.run.launch(grid_x, 1, 1, current_stream(), compiled.function,
                    compiled.packed_metadata, None, None, None,
                    address_type(tensor.numel())(tensor.data_ptr()), None)
```

每个额外运行时参数需传真实值，每个额外constexpr参数在当前ABI中也需要对应位置；先检查实际kernel参数表和已生成launcher源码。探针进程退出0不等于每个方案有效，必须逐项检查`error`、正确性与实际调用范围；编译／接口失败不能用来得出该算法“更慢”的结论。原地输入在计时前重置，输入生成和随机数分配放在计时区间外。地址类工厂的创建与已缓存地址类的单次构造也应分别计时。

本批CPU补测显示：重新创建地址类及实例约16.3μs，使用已缓存类构造实例约0.30μs。父版已经缓存类，因此不能用16.3μs解释父版逐调用耗时。地址对象缓存减少的构造开销、Python调用包装、设备端改法，以及参考耗时波动，分别记录。

## 精度安排和诊断预算

BF16与FP16的指数范围不同。把整个BF16输入转成FP16会把例如`1e5`、`1e20`的大正值变成无穷，不能只凭常规随机输入精度通过就交付。可研究只把负半轴指数的临时量降为FP16，正半轴保留FP32缩放，并在分支合并前写回目标dtype以缩短缓冲生命周期；这仍需实际编译和完整精度验证，不能预设能装进UB。

该类修改应额外测dtype极值、极小值和大连续输入。原地测试每次用新输入重置后再比较一次变换结果。不要将多次SELU后的值与原始输入的一次SELU结果比较。

本批`minimum(x, 0)`写法在NaN检查失败，改为`where(x > 0, 0, x)`后42项语义通过；但16384块的大BF16仍要求288KiB UB，超出192KiB。说明保留完整数值范围后的正负分支缓冲仍需进一步压缩；不能把常规输入下经FP16的64μs结果直接作为有效交付。失败源码、检查器和错误全文均应保留。

同一硬件和源码的模拟器采集超时后，先检查采集程序、warmup／iterations和原始dump，不应机械对多个大case重复十分钟采集。可以先用独立单调用应用和一个必要的case验证工具链；真机metrics已经足够支撑的修改继续推进。记录超时或取消，不能把“发起了instruction请求”写成“成功获得指令诊断”。本批原生runner曾采用10次warmup、100次调用，采集成本应与单调用OpProf明确区分。

## 原地启动绑定、缓存顺序与采集兼容

启动包装器的“绑定／编译”和“执行计算”分开设计。若绑定先对输入执行一次，再访问元数据或warmup，随后捕获异常并调用普通kernel，可能把同一原地操作执行两遍。优先只warmup编译，成功取得句柄后再执行本次计算；失败回退也只能执行一次。不能用反复缩放后已饱和的输入来验证这项语义。

缓存补测应覆盖调用顺序：先用整除长度建立不同分块档位，再输入奇数长度和不同offset，检查尾部mask及编译属性不会错误复用。缓存键、`do_not_specialize`和ABI实际属性需一起审查；不能只凭键里有dtype就假定跨形状安全，也不能仅从潜在风险推定某份已测源码存在错误。

本机rad2deg_裸启动候选曾完整评测通过，但metrics／instruction均返回“base collection退出成功却没有完成标记”。这属于采集失败，不能从退出码0或模型文字推断已获得硬件指标。若采集launcher需要Tensor元数据，可尝试同一计算kernel的标准启动回退，保持相同分块、运算和输入语义；必须对新源码重新完成独立语义、正式评测与采集，不能把准备好修复当作已经修好。

本批只读核查了实际安装的`triton/backends/ascend/driver.py`（SHA `dfa7d353e4d847abcb2348a6cb9f616dba7d4b9b62f833d691f14ff0fe27b74c`）。`NPULauncher.__call__`在`TRITON_REGISTER_TENSOR_MSPROF`登记模式下填入Tensor形状；正常执行分支则调用底层launch，并按其返回值更新`TRITON_PROFILER_REGISTERED`。直接调用`.launch`会跳过这些Python处理。该事实解释了需要保留的采集接口职责，但不能单凭源码断定缺少完成标记的唯一原因。原始片段保存在该批`research/rad2deg_/backend-launcher-review.json`；采集回退是否解决实际故障，仍以独立修复批次为准。

以上是endurance-20261022运行中固化的控制与采集约束，供之后新建覆盖层使用；不回写已经冻结或正在执行的角色，不将它归为先前轮次自动读到的知识。


### references/性能指标/计算与控制.md

# 计算与控制

Vector 活跃时检查超越函数、类型转换、归约和中间向量。Cube 活跃时检查矩阵乘计算路径、tile 复用和累加方式。Scalar 活跃时检查地址计算、循环和控制开销。

活跃时间不代表有效吞吐达到峰值，也不能把 Vector 比例当作 Cube 利用率。规则阈值用于筛选假设；结合源码与完整评测验证。

当 MTE 与计算流水都高时，至少有“有效重叠”和“阶段间等待”两种解释。结合时间线、指令映射或热点采集判断，不能单凭两项都高就增加缓冲或扩大分块。

## 调用路径与program粒度

SELU本轮探针提供了两类可区分的现象：1024元素时，普通Triton入口约58 μs，正确调用已编译内核约24 μs；4096×4096时，把分块1024增至8192，批量平均约601 μs降至83 μs，而直接启动入口几乎不再改变大输入耗时。探针只是实验选择证据，不能替代候选的完整正确性与两次正式复验。

前者优先检查参数绑定、设备/stream查询、编译缓存和实际launcher；后者优先检查program数量、每program控制与同步成本，以及增大分块后的资源需求。SELU的FP32中间量在分块16384时发生UB overflow，而log10的资源需求不同；不要把单一候选的最大分块泛化为芯片上限。

读取实际导入模块的 `__file__` 和SHA。本机FlagTree把 `triton.runtime.jit` 指向Ascend专用目录；缓存key包含specialization与options，正常JIT会补debug及FlagPrism选项。缓存查找必须与真实入口一致，不能仅根据目录名或“有compiled kernel缓存”推断已经走快路径。当前stream、metadata和hook应使用正常调用语义；任何可能已提交的原地计算发生异常后，不自动重试一次。

真实性能需验证三个环节：快路径实际命中、修改后的计算完整正确、正式计时有所改善。本批SELU原始第二轮只实现了缓存调用尝试，36项通过但性能未改善；追加修复尚待NPU补测。该经验供后续新覆盖层使用，不改变原始冻结组。

同批log10第二轮的1024元素完整调用约19.44 μs，独立metrics设备任务约1.554 μs；大输入完整调用约36—39 μs，设备任务也约36—39 μs。先按输入规模区分启动路径与设备端优化方向，再选择补测。独立采集时间不可直接相减作为精确开销分解；该轮轮内0.602×尚未独立复验，且生成先于profiling。log10可编译32768分块不意味着SELU也可编译同一分块。

最终真机审计已确认SELU的具体问题：原候选的空options与正常JIT的`debug=false`生成不同key，原直接启动分支没有执行；正常key可命中同一编译产物并正确计算。追加的options/分块/布局修复通过36项重复语义和两次完整36项，为0.236× / 0.243×；log10最终为0.619× / 0.539×。先证明目标路径实际执行，再解释性能变化；不能把写出了快路径当成已经使用快路径。

## 2026-10-22的启动与缓冲机制

同一log10编译产物的交错探针中，仅改变Tensor参数与本次整数地址，完整调用由约12.5—12.8降为7.5微秒。实际后端源码中前者额外执行`aclrtPointerGetAttributes`。最终使用整数地址时也要满足profiler元数据协议；当前后端在采集中调用参数`size()`。

SELU的16384分块实验即使设置`multibuffer=False`，BF16/FP32仍报UB需要2,097,408 bit（256KiB＋32B），可用1,572,864 bit（192KiB）。这是当前表达式与固定编译器的编译结果；下一步应检查中间量生命周期、融合与缓冲复用，不能把失败归为HBM带宽不足，也不能据此认定所有SELU实现都需要相同容量。该候选在12项连续快路径中只通过8项，未进入正式性能评测。


### references/知识索引.md

# 按算子与指标选择知识

## 算子类型

| 类型 | 例子 | 优先阅读 |
|---|---|---|
| 逐元素与原地计算 | selu_、rad2deg_、log10_、conj_physical | [逐元素与原地](算子类型/逐元素与原地.md) |
| 归约、扫描与随机数 | cumulative_trapezoid、uniform、moe_sum_reduce、per_token_quant_int8 | [归约与扫描](算子类型/归约与扫描.md) |
| 索引、散射与采样 | value_selecting_reduction_backward、scatter_add、upsample_bilinear2d、fractional_max_pool3d、rotary_embedding | [索引与采样](算子类型/索引与采样.md) |
| 多阶段融合 | gelu_and_mul、silu_and_mul_masked，以及含多 kernel 的实现 | [融合与执行路径](算子类型/融合与执行路径.md) |

## 性能证据

| 观测 | 参考 |
|---|---|
| MTE 活跃、重复搬运、非连续访问、DMA | [搬运与执行路径](性能指标/搬运与执行路径.md) |
| Vector、Cube、Scalar 活跃 | [计算与控制](性能指标/计算与控制.md) |
| 多流水重叠、同步、模拟器指令与热点 | [指令与同步](性能指标/指令与同步.md) |
| 候选与原生参考性能不同、需要查明原因 | [成对对照与机制定位](性能指标/成对对照与机制定位.md) |

这些文档是诊断与实验选择知识，不是硬件指标的完整手册。算子可能同时涉及多个类型；复杂度最终由实际执行图与证据确定。

2026-10-21新增记录：归约类别包含MoE分块和规模分流；索引类别包含value backward的列覆盖与一般stride；指标类别包含完整调用/设备任务对照、流水边际统计和预热聚合规则。以上内容来自本轮已归档数据和明确的失败记录，供之后新建覆盖层使用。正在运行的 `night-20261021-v3` 使用此前冻结的知识副本，这次更新不参与其原始三组成绩。


### references/算子类型/归约与扫描.md

# 归约、扫描与随机数

归约关注维度、归约长度、输出覆盖和中间精度，检查是否先转置或写出临时结果。min 可评估输入精度比较；sum、量化 scale 和扫描不可照搬 min 的精度策略。

扫描还包含块内扫描、块间前缀传播与最终修正。复杂扫描先画出实际阶段，区分计算、阶段间依赖和临时存储；metrics 无法定位时追加指令或热点采集。

## 梯形累计积分：精度不仅是累加dtype

`cumulative_trapezoid(y, dx=dx)`的数学式可以融合，但参考实现的舍入位置也是数值语义的一部分。endurance-20261022的同输入NPU任务记录显示参考依次执行Slice、Add、Muls、Cumsum；固定FlagGems在kernel内直接将两侧输入扩为FP32相加、扫描，最后另乘dx，在4097长度的FP16/BF16补充检查中超差。

本批CPU表达式探针对三种dtype、四种dx比较：`cumsum((left + right) * (dx / 2))`逐元素与CPU参考相同；把相邻元素先升FP32相加、或把非二次幂dx移到扫描之后会产生差异。原始证据为该批`research/cumulative-rounding-cpu.json`，CPU结论不能替代NPU验证。

融合候选首先需要保留低精度Add和缩放的舍入；扫描的输入dtype、实际累加精度及结合顺序还要分别核对，不能预设升FP32扫描就能对齐参考。非二次幂dx、长序列、转置和非末轴都要额外验证。

第二批实际KG探针发现了CPU建议的限制：在NPU的4×4097输入及六种dx上，FP16增量直接调用`cumsum`与`cumulative_trapezoid`逐元素一致；先将相同增量转FP32再`cumsum`、最后转回FP16，最大误差为0.01953–0.375。该探针的BF16/FP32对应路径则对齐。原始探针见`research/cumulative_trapezoid/reference-cumsum-dtype-probe/`。其中`f16acc`只是探针字段名，实际表达式为`inc.cumsum`，不能据此断言硬件累加器一定是FP16；输入dtype可能改变算法、结合顺序或中间精度，需要实际IR或指令再区分。早期“前两次舍入＋FP32扫描即可”的说法因此仅保留为已受反例限制的候选思路。

CPU建议在第一批cumulative_trapezoid角色冻结后形成，NPU反例来自第二批模型自行提交的真实探针；知识修订供之后新覆盖层使用，不改写已冻结角色或原始结果。

扫描按形状切换实现时，必须额外在性能输入上检查正确性。Native合同的`correctness`与`timing`是两个phase，前者小形状通过不能替代后者大形状的数值验证；本批独立门禁新增全部9项冻结性能输入，各两次及切stream，仍使用原容差。矩阵乘前缀方案还应核对三角矩阵方向、块间carry、最后不足整块时的mask和FP32 dot精度。局部dot很快或整组计时分数很高，都不能替代这些检查。

uniform 属于随机数生成。验证状态推进、种子、区间、dtype 与统计性质；不能把“不能逐值对齐不同 RNG 实现”当作任意结果都正确，也不能为加速重复固定输出。性能比较须保持原测试合同。

替换Philox为计数器哈希时，须单独报告算法和统计限制。输出均值、方差及连续两次不同，不能排除seed/offset被截断：endurance-20261022候选曾把两者掩为31位，CPU表达式见证表明第31/32/40位不同的状态会别名。额外门禁因此检查高位seed、高位offset、相同状态复现、状态推进与恢复重放；不以此声称通过完整随机质量测试。状态递增单位也需说明，不能只因调用同一状态获取函数就声称与原Philox逐项消耗相同。

per_token_quant_int8 包含逐行绝对值最大值、scale 和量化写回，分别检查归约与逐元素阶段；候选性能需包括全部输出和 scale。

## MoE 归约：分开输入规模和执行路径

`moe_sum_reduce` 的输入为 `[token, top_k, hidden]`，沿 `top_k` 求和后乘缩放系数。保持FP32累加、输出dtype、全部输入stride及非默认scale。把更多token合并到一个program时，同时核对中间张量规模、尾部覆盖和实际grid，不能只按tile越大越好选择。

`night-20261021-v3` 中，固定原版的大输入配置为1×2048；无profiler第二轮采用16×1024。4096×8×7168的bf16输入，grid从16384个program减少到1792个，两次完整复验相对固定原版为2.876× / 2.904×；小输入保留原配置，仍慢于PyTorch。该结果支持按规模分流，不证明所有输入都受HBM带宽限制。

同批原生profiler组采用连续且整除时的快路径、编译期top-k循环及更少stride传参，保留通用分支。其8项合同两次复验相对PyTorch为1.543× / 1.649×，36项扩展语义也全部通过。具体tile值是本版本、本合同的测量结果，不作为其他dtype、布局或芯片的默认最优配置。该轮修改早于最终profiling，不能写成profiling建议产生的收益。

来源：该批 `results/moe_sum_reduce/`；无profiler源码SHA `69caad9f45e9ebc794f0561a62a6b7838f45836962aa9d14c5cedbfdeb14fe97`，原生profiler源码SHA `96b0e8e40b23afe9a654d9e1afdb120046c97edebeccea5d998c212fa8491196`。

## 同一分块在一般布局上也需重新核对编译资源

上述无profiler版扩展语义为30/36。六个失败是stride2的 `[129,2,257]` 和 `[129,16,1025]`，覆盖三种dtype；原配置在大token时仍采用16行分块。FP16/BF16编译报告分别要求6291456和12582912 bits UB，而可用1572864 bits；FP32同两形状分别要求4194304和8388608 bits。原始失败回执SHA为 `79ddf2fc170600989953dfed80679b705867e54f9f312a17f321c98a81ce734a`。

这说明不能把连续、对齐输入的可编译tile直接推广到一般stride，也不能用元素数乘dtype字节数代替全部编译缓冲需求。建议保留已验证的连续对齐快路径，一般布局先限制每program处理量，核对实际编译资源及完整语义；本批追加的单行、最多256列候选仍待NPU验证。

原HG-KDA版扩展语义31/36，失败涉及奇数行宽下的数值或输入保持检查。其配置将奇数hidden的列块缩为1，`[65,8,1025]` 和 `[129,16,1025]` 分别产生66625和132225个program。program数是源码几何事实，不是已证明的失败原因，也不能据此直接宣称超过某个硬件启动上限。另有带mask列块的追加候选；原始失败及具体编译/执行机制尚未闭环前，不把正式8项通过扩大为一般布局正确。


### references/算子类型/索引与采样.md

# 索引、散射与采样

先说明输入到输出的索引关系，检查地址计算、数据是否连续、写入是否冲突，以及是否存在初始化和散射两个阶段。

value_selecting_reduction_backward 需要未选位置为零；不能省略初始化。scatter_add 有重复索引时需保留累加和原子语义。只在证明目标互不重叠时才能改写成普通 store。

upsample_bilinear2d 需要保留 align_corners、边界及坐标精度；fractional_max_pool3d 需要保留随机采样参数、窗口规则和返回索引。RoPE 保留位置索引、旋转维度和原地更新合同。

Scalar 活跃较高时检查除法、取模和索引循环；MTE 活跃较高时检查合并访问、重复读取与 gather/scatter 粒度。并行流水多或等待明显时，通过指令及源码映射定位具体依赖。

## 覆盖域和布局是两个独立检查

`value_selecting_reduction_backward` 可表示为输出 `[outer, red, inner]`：每个 `(outer, inner)` 只有一个选中归约位置写梯度，其余写零。这里没有多个梯度对同一个输出累加的要求；融合初始化与回填仍必须写满整个输出域。

改动分块后，分别证明地址合法、输出覆盖完整、写入者唯一。`night-20261021-v3` 的原生profiler第二轮把red=5向下取块长4，又取消列方向grid，导致最后一列无人写入，正式96项中有6项数值失败。`cols < red` 只屏蔽越界，不会生成缺失的列。保留列分块、向上取整或限定快路径适用域均可作为修复方向，修复后重新完整评测。

逻辑维度能合并不等于物理stride能合并。连续布局可走折叠快路径；一般非仿射布局需要逐维坐标和stride寻址。补测包含非2次幂、跨列块长度、单例维、空输入、负维度、keepdim和多层非仿射转置。`tl.debug_barrier` 不能解决不同program之间的写冲突；清零后回填时，每个program只能写自己负责的列块。

失败来源：该批原生profiler第二轮，源码SHA `6aeec9d48ca503024ec2ecce06dc9c5558c3ab039d662cf2a18a5cd347bb6951`。新增通用布局和清零/回填候选另有48项及跨列块补测，成绩必须等待其实际回执。

## 退化输出先区分编译错误与数值错误

固定FlagGems的 `upsample_bilinear2d` 修复本机评测接口后为18/24，六项失败均是 `[1,3,7,9]` 输入缩至 `[1,1]` 输出，覆盖三种dtype和两种align_corners。回执为 `MLIRCompilationError`，发生在 `ConvertTritonIRToLinalgIR`；其他18项通过。原始回执SHA为 `23fdd8584ccbbc5c2c1b91fd66624737733260e41f4c64bac2f473825fbd34cd`，逐case观察保存在主批 `research-observations/upsample-adapter-compile-failures.json`。

此时应先按退化输出分支缩小编译复现，检查广播、索引和常量折叠后的IR，再讨论性能。不能将编译失败解释为数值精度不够，不能删去1x1用例后计算整组加速比；接口修复也不表示后端全部可编译。本轮未修改固定原版算法或编译器，尚未定位到具体失败转换规则。

## 散射写入的编译机制核查

`scatter_add` 的瓶颈不能只归结为索引冲突或内存带宽。`endurance-20261022` 固定 FlagGems 父版的小型 FP16 代表输入中，TTIR 的 256 路 `tt.atomic_rmw` 在 `ttadapter` 和 `bcmlir` 中变成标有 `ExtractedLoadOrStore` 的 `scf.for`；循环体逐 lane 构造 `tensor<1xf16>`，再执行 `hivm.hir.store atomic=<add>`。原始片段和 SHA 见该批 `research/scatter_add/atomic-lowering-review.json`。

这证明该编译实例存在逐元素写回的降低方式。结合无冲突原子写入仍慢、数据相关的非原子写入也慢、仿射复制很快的探针，应检查地址计算到实际编译循环的映射；MTE3 活跃高不等于达到 HBM 峰值，也不保证调大 BLOCK 就能改善。候选与父版需分别检查 IR，不能把一个 FP16 实例推广到全部 dtype 和形状。

可评估按输出块归约、分组或排序，以及固定编译器已支持的低层路径；全部预处理、临时存储、转换和启动必须计入完整调用。低精度原子更新还要独立检查重复索引的累加误差，必要时采用 FP32 暂存再转回。这些是待验证方法，不代表已有对应性能收益。该知识于首批搜索结束后补入，后续覆盖层按实际载入记录归属。



### references/算子类型/融合与执行路径.md

# 融合与执行路径

用真机任务记录核对候选和参考侧各启动几个 kernel、有无 DMA 和临时张量。单个 kernel 内 Vector、MTE、Cube 并行活跃，不能据此认定启动了多个 kernel。

融合的潜在收益是减少临时张量读写和 launch；代价可能是缓冲占用增大、并行度降低和同步增加。先记录原路径，再比较融合后的路径和完整延迟。

narrow_copy 的历史经验是：参考侧小连续复制走 MEMCPY_ASYNC，纯 Triton 的小输入设备任务更慢，因此采用小输入 CANN DMA、大输入 Triton。适用条件、正确 ABI 与计时覆盖都影响结论；不能把此策略推广到任意切片。

CANN aclrtMemcpyAsync 是宿主侧运行时 API。TLE Raw 的 AscendC 外部操作在 kernel 内执行，编译与调度路径不同；即使都搬数据，也需要分别评测。


### references/算子类型/逐元素与原地.md

# 逐元素与原地计算

对连续输入先检查每 program 工作量、网格上限、地址计算与读写量。非连续输入保留正确的 stride 路径；原地操作必须返回同一存储并保留修改语义。

SELU、log 等函数检查计算公式、输入范围和中间精度。数学等价不保证浮点结果满足测试，exp/log 指令也不能简单按普通 FLOP 计量。共轭操作保留复数虚部符号、共轭标志及实数行为。

边界掩码只在公开 shape 和 tile 可证明整除时移除，非整除仍保护。报告同时记录正确性通过与专家评审状态。

固定迭代 grid-stride 方案必须分别证明覆盖和不越界：设 program 数为 G、块大小 B、迭代次数 I，连续编号方案有掩码时要求 G×B×I≥N，无掩码时要求 G×B×I=N。仅有 N 整除 B×I 不能证明安全，因为实际 G 可能被下限或上限改写。rad2deg_ 首轮曾在 N=1025 漏算尾部，并在 N=2050×16384 的规划中产生越界范围；未将越界反例提交设备执行。

原地操作不能通过将 A 替换成 A.contiguous() 后返回复制品来处理非连续输入。补充检查应包括返回同一对象、原存储被修改、带步长切片和奇数长度；仅通过当前连续合同用例不足以覆盖这些语义。

多个 Vector/MTE 指标都高时，应检查是否重叠、是否重复搬运以及任务粒度；不按最大百分比直接下瓶颈定论。

2026-10-22的log10_/selu_使用成对硬件证据检查启动路径；当前输入地址与当前stream每次读取，完整stride不截断到固定8维。小输入优先减少宿主参数处理，大输入单独检查分块和UB。log10的低精度数学要通过IR确认实际类型转换已消失；SELU的FP16可行分块不能直接套给BF16/FP32。方法及工具见[成对对照与机制定位](../性能指标/成对对照与机制定位.md)。


## Native 用例的准确输入

`get_profile_context` 额外返回 `hgkda_workloads`，其中按用例身份保存原始`inputs`、seed、tolerance及phase。先从这里读取形状、dtype、stride相关规格与标量参数；不要根据case序号或其他算子的历史形状猜测。它不改变正式延迟、指标、候选源码或计分合同。


# 本批连续优化任务

# 夜间连续优化任务与机制约束

本批由服务器tmux串行执行，控制端不在阶段边界询问下一步。每项先核对固定合同与起点，再采集代表case，CC/DeepSeek生成，独立补充语义和一次完整KGS复验，按候选实际延迟保留最佳。最多两轮搜索批次，每批次4或3个KG round，单项分别60或45分钟，总预算12小时；预算是上限，不靠空等凑时长。

固定FlagGems d6a8eec、SGLang 9386120、FlagTree 0.6.0+ascend.gitc286cba6、物理卡7、KGS19659完整调用walltime。不升级编译器，不改合同，不访问其他卡，不调用目标算子的PyTorch/FlagGems计算作为优化实现；元数据、分配和必要别名可用。只创建本批工作区。Codex提供机制诊断和控制器，CC/DeepSeek生成改码，实际贡献分别报告。

## 各算子优先问题

- `selu_`：起点已支持stride、offset、stream、FP16/BF16/FP32数值范围。优先研究中间量生命周期、循环内小片计算/立即写回、正负分支缓冲复用和UB实际分配；BF16整张输入转FP16会破坏1e5等大正值，禁止重试。16384 BF16的负轴FP16/正轴FP32组合仍需288KiB UB，设备仅192KiB；只关multibuffer、select分析、cast对齐、开buffer节省都已失败，不能只再枚举这些开关。minimum会吞NaN，要保留NaN。若继续裸启动必须读取当前地址、dtype、布局和stream，保留profiling回退。只有同case实际耗时改善才是本批新增收益。
- `rad2deg_`：先修原地、特殊值及大数精度；匹配PyTorch度数常量及舍入路径，不能返回新张量或用CPU计算。复用逐元素连续与通用stride分流，不能仅在计分shape通过。
- `cumulative_trapezoid`：检查相邻元素求和、积分缩放、scan是否被拆成多次启动与中间张量。优先融合相邻输入处理与扫描，分长度采用块内或多阶段scan；非末轴/转置保留通用路径。FP16/BF16累加与最终舍入要对齐参考，不能仅比较随机均值。
- `scatter_add`：分清输出复制和原子更新、多次启动、冲突和低精度累加；保留输入不修改、负dim、重复索引、stride。不能假设索引唯一，不通过删数据冲突取得收益；可以研究按行/目标分组与FP32暂存但需测完整开销。
- `uniform`：要求新分配、不改输入、每次调用产生新随机结果、区间上界排除、三dtype、常数区间。优先减少随机状态获取和多次launch，融合随机生成、缩放和存储。不得缓存输出、固定伪随机表、依赖测试seed或用目标PyTorch随机算子代替。修改随机算法须说明生成器语义及统计限制。
- `gelu_and_mul`：参考是FP32 exact GELU(approximate=none)后乘另一半并转回输入dtype。检查erf近似误差、访存连续性、二维stride、行/tile组织及一次融合启动。tanh近似不能默认替代exact；有边界值和大值语义门禁。原版慢不等于算术本身慢，先查调度与循环。

## 采集和失败预算

先metrics和同输入双方真实设备任务记录，再决定是否需要instruction。每次profile最多240秒，watchdog仅取消本工作区登记的精确operation ID，不重启共享服务。模拟器超时后本批不重复同一大case；若需要，改选小输入/单个kernel。失败请求、取消回执和原始日志保留，不能称为成功热点分析。

优先将机制落实到代码和实际IR：变化前后块数、kernel数、MTE/Vector、缓冲占用；普通PipeUtilization不当成峰值利用率，没有匹配流量/时间/硬件上限就不声称数值Roofline。发现已有假设不足，可以主动补测，但不要把全部预算耗在同一采集。

语义失败不发布该候选；原始失败与有效最佳都保留。KG round成功仍要独立门禁。最终每个新字节源码一次完整KGS复验，相同字节结果复用；不重复测量只挑最高分。至少通过原合同及独立布局/数值/随机语义门禁，才能进入最终交付。


## 本项已知证据与失败

{
  "operator": "scatter_add",
  "seed": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/seeds/scatter_add.py",
  "seed_sha256": "e249f5c57930cc37a67d58c754131a55ed6c296575763a560ac889066d38de09",
  "current_operator_state": {
    "status": "搜索中",
    "parent": {
      "status": "formal_failed",
      "source_sha256": "e249f5c57930cc37a67d58c754131a55ed6c296575763a560ac889066d38de09",
      "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/results/scatter_add/parent/formal/parent-1.result.json",
      "gate_result": {
        "status": "semantic_failed",
        "check": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/results/scatter_add/parent/semantics.json",
        "passed": 14,
        "total": 18
      }
    },
    "attempts": [
      {
        "attempt": "attempt-1",
        "workspace": "/data/hanle/ascend-optimization/runs/endurance-20261022/kg-runs/scatter_add-endurance-20261022-attempt-1",
        "valid_candidates": 0,
        "roles": 8
      }
    ]
  },
  "completed_jobs": {
    "parent:scatter_add": {
      "status": "completed",
      "started_at": "2026-10-22T13:27:02.935043Z",
      "result": {
        "status": "formal_failed",
        "source_sha256": "e249f5c57930cc37a67d58c754131a55ed6c296575763a560ac889066d38de09",
        "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/results/scatter_add/parent/formal/parent-1.result.json",
        "gate_result": {
          "status": "semantic_failed",
          "check": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/results/scatter_add/parent/semantics.json",
          "passed": 14,
          "total": 18
        }
      },
      "finished_at": "2026-10-22T13:28:23.000204Z"
    },
    "observe-parent:scatter_add": {
      "status": "completed",
      "started_at": "2026-10-22T13:28:23.008057Z",
      "result": {
        "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/research/scatter_add/parent/facts.json",
        "status": "completed"
      },
      "finished_at": "2026-10-22T13:29:03.031241Z"
    },
    "search:scatter_add:1": {
      "status": "completed",
      "started_at": "2026-10-22T13:29:03.040242Z",
      "result": {
        "workspace": "/data/hanle/ascend-optimization/runs/endurance-20261022/kg-runs/scatter_add-endurance-20261022-attempt-1",
        "valid_candidates": 0,
        "best": null
      },
      "finished_at": "2026-10-22T14:07:05.251260Z"
    },
    "validate:scatter_add:attempt-1:ab2327d7e898": {
      "status": "completed",
      "started_at": "2026-10-22T14:05:15.186836Z",
      "result": {
        "status": "semantic_failed",
        "check": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/results/scatter_add/attempt-1/candidates/ab2327d7e898418d2b41966a67ec49a6d4152f359586a700cbec139a64b0242b/semantics.json",
        "passed": 17,
        "total": 18
      },
      "finished_at": "2026-10-22T14:05:45.189734Z"
    },
    "validate:scatter_add:attempt-1:b332ff895356": {
      "status": "completed",
      "started_at": "2026-10-22T14:05:45.197415Z",
      "result": {
        "status": "semantic_failed",
        "check": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/results/scatter_add/attempt-1/candidates/b332ff895356ef37e0b4b8ce67cc984a1e3cbbeecda13426f58cd8717d682408/semantics.json",
        "passed": 17,
        "total": 18
      },
      "finished_at": "2026-10-22T14:06:05.213578Z"
    },
    "validate:scatter_add:attempt-1:48a3ae825439": {
      "status": "completed",
      "started_at": "2026-10-22T14:06:05.221283Z",
      "result": {
        "status": "semantic_failed",
        "check": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/results/scatter_add/attempt-1/candidates/48a3ae825439eed14247ca168f6eb66df4a3d43a14397ce85256c7e0aab10e73/semantics.json",
        "passed": 17,
        "total": 18
      },
      "finished_at": "2026-10-22T14:06:35.231525Z"
    },
    "validate:scatter_add:attempt-1:0a3e9ecd7f42": {
      "status": "completed",
      "started_at": "2026-10-22T14:06:35.239271Z",
      "result": {
        "status": "semantic_failed",
        "check": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/results/scatter_add/attempt-1/candidates/0a3e9ecd7f425b93db2c5d992c56653420cae0ce6f5b0880f6af5329226bc9f8/semantics.json",
        "passed": 17,
        "total": 18
      },
      "finished_at": "2026-10-22T14:07:05.248048Z"
    },
    "search:scatter_add:2": {
      "status": "running",
      "started_at": "2026-10-22T16:51:34.703016Z"
    }
  },
  "frozen_knowledge": [
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/completion-audit.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/final-comparison.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/tool-sha256.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/夜间任务.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/方法与机制.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/累计积分第一批失败与修复.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/scatter第一批失败与修复.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/uniform随机状态审查.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/uniform状态接口的低开销替代.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/scatter实际IR的逐元素原子写入.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/GELU第一批失败与修复.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/uniform保留Philox的高位乘法备选.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/SELU复验分数与实际耗时.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/独立门禁搜索内反馈.md"
  ],
  "paired_execution": {
    "case": {
      "name": "scatter_add-timing-0",
      "inputs": {
        "spec": {
          "dtype": "float16",
          "shape": [
            128,
            128
          ],
          "dim": -1
        }
      },
      "seed": 1729,
      "tolerance": {
        "atol": 0.015,
        "rtol": 0.015,
        "required_matched_ratio": 1.0
      }
    },
    "source_sha256": "e249f5c57930cc37a67d58c754131a55ed6c296575763a560ac889066d38de09",
    "sides": {
      "reference": {
        "status": "completed",
        "trace": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/research/scatter_add/parent/reference/trace.json",
        "device_events": [
          {
            "name": "aclnnScatterAdd_LinearIndex_LinearIndex",
            "dur": 9.040125,
            "ts": "1792675722345855.411",
            "pid": 3154110887,
            "tid": 36,
            "args": {
              "Model Id": 4294967295,
              "Task Type": "KERNEL_AIVEC",
              "Physic Stream Id": 36,
              "Task Id": 4,
              "Batch Id": 0,
              "Subtask Id": 4294967295,
              "connection_id": 15
            }
          },
          {
            "name": "aclnnScatterAdd_ScatterElements_ScatterElementsV2",
            "dur": 22.7605,
            "ts": "1792675722345890.192",
            "pid": 3154110887,
            "tid": 36,
            "args": {
              "Model Id": 4294967295,
              "Task Type": "KERNEL_MIX_AIV",
              "Physic Stream Id": 36,
              "Task Id": 5,
              "Batch Id": 0,
              "Subtask Id": 0,
              "connection_id": 18
            }
          },
          {
            "name": "MEMCPY_ASYNC",
            "dur": 1.460125,
            "ts": "1792675722345634.946",
            "pid": 3154110887,
            "tid": 36,
            "args": {
              "Model Id": 4294967295,
              "Task Type": "SDMA_SQE",
              "Physic Stream Id": 36,
              "Task Id": 3,
              "Batch Id": 0,
              "Subtask Id": 4294967295,
              "connection_id": 6
            }
          }
        ],
        "total_device_events": 3,
        "error": null
      },
      "candidate": {
        "status": "completed",
        "trace": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/research/scatter_add/parent/candidate/trace.json",
        "device_events": [
          {
            "name": "_copy_contiguous_kernel",
            "dur": 1.68,
            "ts": "1792675730606958.119",
            "pid": 3154110887,
            "tid": 36,
            "args": {
              "Model Id": 4294967295,
              "Task Type": "KERNEL_AIVEC",
              "Physic Stream Id": 36,
              "Task Id": 8,
              "Batch Id": 0,
              "Subtask Id": 4294967295,
              "connection_id": 3
            }
          },
          {
            "name": "scatter_add_kernel_1",
            "dur": 106.022125,
            "ts": "1792675730607125.542",
            "pid": 3154110887,
            "tid": 36,
            "args": {
              "Model Id": 4294967295,
              "Task Type": "KERNEL_AIVEC",
              "Physic Stream Id": 36,
              "Task Id": 9,
              "Batch Id": 0,
              "Subtask Id": 4294967295,
              "connection_id": 7
            }
          }
        ],
        "total_device_events": 2,
        "error": null
      }
    }
  },
  "方法与机制.md": "# 本轮方法、有效改动与剩余限制\n\n## 实验对象和计时\n\n起点是上一轮最终源码：log10_ SHA `c845221e25bed5df6d631eac7554425ce498dc9b7b4bbe2edd6c9903fb498b9a`，selu_ SHA `49bdbb789d497b6423f893f3433167f3fd4bc61197676236bd62a9ec79c7dd91`。固定FlagGems提交 `d6a8eec473517a3d68157b208eb9c057eb1d4c50`、FlagTree `0.6.0+ascend.gitc286cba6`、Triton 3.5.1、CANN 9.0.0和物理卡7。\n\n本批复制上一轮原36项Native合同，使用KGS 19659的完整调用walltime。24项correctness、12项timing不变。每个不同源码最终做一次独立正式评测；KG搜索内部成绩、自动retest和额外探针分别保留。父源码也在本批完整复测，以同case候选实际延迟计算收益，最终按实际延迟最小选版。\n\n正式结果和最终选择以`deliverables-final/manifest.json`为准，逐case父版与最终版比较见`final-comparison.json`。例如，不能把log10_的0.875÷父版0.751当作真实优化幅度；两次测量的PyTorch耗时也有变化。SELU本轮新候选没有降低整组实际耗时，所以继续保留父源码，其本次分数变化单列为复测结果。\n\n## log10_：启动路径与大输入搬运\n\nCC生成的第二轮版本按当前输入地址和元素数缓存启动包装对象，调用时仍读取当前地址、dtype、连续性和stream；不缓存Tensor或输出。缓存地址参与键，不按相同元素数复用上一个输入的地址。它通过布局、storage offset、高rank和stream切换检查。该快路径依赖固定FlagTree私有launcher ABI，本轮只验证物理卡7的单卡环境。\n\n诊断中曾将“创建地址类及实例”约16.3μs误当成父版逐调用开销。父版实际上已缓存类，本批CPU补测表明每次构造实例约0.30μs。原始探针和纠正证据`research/address-construction-cpu.json`一并保留。\n\nCC第三轮还生成了四小块先加载、再计算、最后写回的kernel。大FP16探针和正式测试都从约33.9μs降到约30.3μs；BF16大输入也从约38.1降到34.7μs。Codex将其与父版启动、CC启动分别组合成pipeline-v4和merged-v8，保留FP32原路径，并修正profiling回退以使用同一个计算kernel。它们通过独立42＋12项检查，但整组实际耗时未胜过最终入选版本，故作为有效未入选候选保留。\n\n固定编译器的缓冲节省选项不能直接把log10分块从32768扩为65536：大输入仍报需要384KiB UB，设备可用192KiB。子向量绑定候选能够通过完整测试，但未取得整体收益。源码和失败全文分别保存，未修改安装的编译器或基线。\n\n## selu_：BF16指数范围与UB限制\n\nCC第二轮简化了host分派，按元素数建表，预绑定Tensor方法，并区分对齐与非对齐启动；保留完整stride和当前stream。搜索成绩约0.884×，独立正式结果为0.757×。逐case实际延迟显示它没有优于本批复测的父版，因此未选为最终交付。\n\nCC第三轮将BF16整个输入转成FP16，16384块大BF16耗时约从83μs降到63μs。但BF16的指数范围比FP16大，例如大正值`1e5`转FP16会成为无穷，这条路径不能作为完整BF16语义的有效交付。\n\nCodex随后保留正半轴的FP32缩放，仅将非正输入的指数临时量降为FP16，并在分支合并前转换回BF16，以尝试缩短缓冲生命周期。v6的`minimum`写法未传播NaN，42项中通过41项；改为显式`where`的v7通过42项，但大BF16的16384块需要288KiB UB，仍超出192KiB，12项大输入仅通过10项。最终保留父版FP16 16384块、BF16/FP32 8192块。\n\n仅关闭multibuffer、开启UB节省、关闭cast对齐或select分析，也未让SELU大BF16/FP32的16384块成功编译。本轮定位的限制是这些具体表达式在固定后端中的缓冲需求；尚未实现显式缓冲复用的底层版本。\n\n## Profiling接入和实际归属\n\nHG-KDA新覆盖层加载了上一轮最终源码的四组成对硬件事实、原生Native输入规格，以及缓冲与搬运诊断知识。`input-provenance.json`、`role-load-evidence.json`、覆盖层manifest和结构化分析记录保存了实际加载与执行证据。后续知识修正没有回写运行中已冻结的覆盖层。\n\nCC/DeepSeek负责本轮追加生成和工具调用，Codex负责已有成对结果的诊断、独立候选、代码整合、错误纠正与验收。不能把这两项最终成绩表述为纯原生KG或HG-KDA独立取得。目录中的`codex-`是独立验证路径名，具体源码贡献见各provenance。\n\nKG主动请求了instruction级采集，服务实际调用CANN模拟器。大case请求超时；SELU两次重复长采集由Codex通过对应operation ID取消，保留取消回执、原始dump和日志，没有重启共享服务。本轮没有成功获得可用于定位热点的模拟器指令报告，也未建立数值Roofline。最终机制核验使用真机KGS metrics、同输入两侧OpProf和实际编译IR。\n\n探针中网格误传、launcher参数不匹配、原地多次运算后比较错误等记录保留并排除，不作为算法无效或精度失败的证据。一次将1024元素误作1024个program的越界仅发生在独立调试进程；之后服务健康检查通过，后续设备任务正常。\n\n## 复现与后续\n\n最终源码须通过42项布局/特殊值、12项大连续输入和新增9项数值范围检查，每项两次并切换stream；36项正式合同保持原样。最终源码的KGS metrics、四组成对采集、源码及IR SHA由`audit-paired.py`核验。一次完整复现入口见本批README。\n\n下一步若继续SELU，应直接研究固定编译器生成的中间量生命周期与显式UB复用，优先解决保留BF16范围时的288KiB需求；进一步候选应继续使用本批语义门禁。模拟器先以独立单调用、小case验证采集路径，避免重复大case超时。log10_的大输入四块加载已有效，若继续整合，应先用交错探针确认host路径收益，再进入正式评测。\n",
  "累计积分第一批失败与修复.md": "# 累计积分第一批失败与第二批优先修复\n\n第一批Cube前缀＋blocked column＋预绑定启动方案在KG原合同33项通过，搜索1.3895×，但独立语义仅16/18、全部9项性能输入的数值仅4/9。不能将该搜索成绩发布为交付。原始代码和失败记录位于：\n\n/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/results/cumulative_trapezoid/attempt-1/candidates/e6433e9b643df313ee89a52922e4229f59bf4225f841ab1e9b47f7b5b140fa9a/main.py\n\n同目录semantics.json保存具体错误。第二批应优先复用这份已实现的架构并修复数值，避免再花一小时重复所有dot/分块探针。不要修改既有评测、容差或原始候选。\n\n机制：PyTorch低精度参考先用输入dtype执行相邻和，再以输入dtype执行dx/2缩放，最后FP32累计并转回输出dtype。当前vector/column分支直接扩FP32相加，dot分支也只有缩放之后的cast，少了相邻和的舍入。请显式保留两次舍入，防止编译器把相邻和和缩放收缩成无中间舍入的表达式：\n\n```python\ndt: tl.constexpr = y_ptr.dtype.element_ty\na = tl.load(...).to(tl.float32)\nb = tl.load(...).to(tl.float32)\npair = (a + b).to(dt).to(tl.float32)\nincrement = (pair * SCALE).to(dt).to(tl.float32)\n# vector/column对increment做FP32扫描；dot将已舍入increment按需要转回dt后计算，保留FP32累加和carry。\n```\n\n这是Codex基于CPU舍入探针、NPU真实Add→Muls→Cumsum路径和独立数值失败提出的修改，尚未在NPU验证，不能当作已解决。若显式cast被优化掉，检查实际IR中的trunc/ext或用fma=False，并以数值测试确定。FP32扫描对近零输出也要满足原atol/rtol，必要时处理dot输入精度。\n\n先执行现有独立check-endurance.py（source放本批新candidates子目录以定位冻结合同），要求18/18及timing_input_correctness 9/9，再做新的完整KGS评测。所有9项输入须与oracle一致，不仅检查均值或maxabs；timing phase原本没有比较输出。保留非连续布局、dx非二次幂、空输出、当前stream和每次新分配语义。\n\n本笔记在第一批角色结束后建立，只供第二批新overlay使用，不改写第一批贡献。可用预算有限，优先完成正确且更快的交付，再做额外调参。\n\n## 第二批NPU探针对先前建议的修正\n\n第二批模型自身的87370eb58b294908a3702ae81e5bbd42探针显示：4×4097及六种dx下，FP16增量直接inc.cumsum与参考一致；同样增量先float()再cumsum再转回，最大误差0.01953–0.375。BF16和FP32在对应探针中一致。因此“保留相邻和及缩放舍入，再用FP32 scan”对FP16并不足够，必须检查实际NPU Cumsum的输入类型、算法、结合顺序和中间精度。不能将字段f16acc解读为硬件累加器已确证为FP16。原始请求、准确提交源码、结果与SHA在research/cumulative_trapezoid/reference-cumsum-dtype-probe/。旧CPU观察及失败不覆盖；当前第二批角色不改写，后续新覆盖层读取此修订。\n",
  "scatter第一批失败与修复.md": "# scatter_add第一批失败与第二批优先方向\n\n第一批自包含候选修复了原版部分src/index长度的寻址错误，并尝试末轴连续快路径、小行稠密归约融合、减少启动参数。搜索最佳0.3471×，但独立语义17/18、性能输入数值7/9，不能交付。应优先复用下面的末轮源码，别从原版重复构建四轮：\n\n/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/results/scatter_add/attempt-1/candidates/ab2327d7e898418d2b41966a67ec49a6d4152f359586a700cbec139a64b0242b/main.py\n\n同目录semantics.json保存完整失败。全部第一批候选都涉及BF16重复索引累加误差：case2聚集索引时最大误差0.0546875，大BF16性能输入有少量0.03左右误差。请先以FP32暂存完成复制与原子累加，最后转回目标dtype，避免每次原子更新都在BF16舍入；一般stride和部分src长度路径也需相同精度保障。小行融合归约可直接在FP32求和，不必为它增加临时张量。该建议由Codex基于独立失败提出，仍需实测。\n\n第一批实际metrics在research/scatter_add/search-round4-metrics-summary.json（原始附件在attempt-1/native-evidence归档）。大2048x2048的散射任务约19.5ms，MTE3活跃约97.5%，复制约20us；Vector很低。探针显示data-dependent atomic与无冲突地址写入都很慢，不能再仅靠BLOCK/num_warps枚举，也不能从MTE3高直接说HBM带宽饱和。第一批分析记录和debug-jobs包含对比依据。\n\n在正确性先通过的前提下，可以继续研究分组/分段、亲和写入或本机已支持的更低层原子路径；不要调用PyTorch/FlagGems目标scatter_add，不升级FlagTree。若用TLE Raw须先核对当前版本能力及ABI。排序、分组和临时缓冲的总成本必须进入KGS完整调用评测，不能只展示局部kernel。\n\n先通过check-endurance.py的18项及全部9项性能输入数值，每项两次和切stream；原合同和容差不改。真实有效源码即便仍低于0.8，也比有精度问题的高分版更适合作为交付和下轮起点。时间预算有限，第二批优先完成这一闭环。\n\n此笔记在第一批worker结束后形成，只供第二批新overlay，实际贡献分别登记。\n",
  "uniform随机状态审查.md": "# uniform运行中发现的随机状态问题\n\n第一批曾把Philox替换为lowbias32计数器哈希，搜索第二轮约0.505×。这不是纯分块调优，也不是与Philox相同的随机流。源码静态审查和CPU整数见证在research/uniform/rng-algorithm-review/：候选宿主执行seed & 0x7FFFFFFF、offset & 0x7FFFFFFF，会忽略高位；不同seed/offset可能输出完全相同的序列，均值方差检查发现不了。\n\ncheck-endurance.py已加入3项生成器门禁：seed第31/32/40位区分、offset同三位区分、状态推进与恢复重放。保留原有18项区间/统计/新输入检查。不要绕过检查，不固定seed或缓存输出，不通过修改生成器状态假装推进。\n\n优先保留Philox质量但优化实际实现及启动；若继续用其他计数器RNG，请使用明确的64位key/counter拆分和有依据的算法，不只是折叠一部分位后声称等价。host API与kernel分支都应真正使用高位。须明确状态消耗单位：原版以ceil(N/4)请求counter，新版若以N推进会改变与其他随机算子的交织序列，不能声称消耗完全相同。有限统计与状态检查不代表完整随机质量认证。\n\n原版在非精确可表示的低精度区间失败；host端计算dtype可表示下界/排他上界是可复用思路，但不能因此遗漏特殊区间或新分配/输入保持语义。算法变化、性能、状态语义与统计限制分别报告。此记录只供后续新overlay使用，当前第一批角色不回写。\n",
  "uniform状态接口的低开销替代.md": "# uniform：保留生成器语义的状态接口优化\n\n第一批第三轮曾尝试只读generator.initial_seed()，另维护Python计数器。这样的计数器不跟随set_state/get_state，也不能检测重复manual_seed(同一个seed)，不应作为完整uniform语义的交付；门禁会测恢复重放和真实生成器offset推进。\n\n固定FlagGems本身已有可参考的低开销路径：src/flag_gems/runtime/backend/_ascend/ops/rrelu_with_noise.py里的_lean_seed_offset，使用generator.initial_seed()、get_offset()、set_offset()，避免构造get_state/set_state的Tensor。请先读这份源码并实际测量，比自建一个脱离生成器的计数器更合适。保持所选RNG真实需要的递增单位、合法4倍数以及完整seed/counter位数。\n\n状态API优化和随机算法替换是两项独立改变。优先用同一Philox计算先测真正的host成本收益；如果PhiIox设备计算仍慢，再研究该计算的实现，不能把状态别名当加速。当前完整门禁要求18项区间/统计与3项高位/重放语义；仍不等于完整随机质量认证。\n\n该建议来自Codex阅读固定上游实现，只供之后新overlay；尚未证明本轮uniform已应用或已获收益。\n",
  "scatter实际IR的逐元素原子写入.md": "# scatter_add：实际IR证实原子写入被逐元素展开\n\n本批固定FlagGems父版、小型FP16代表输入的实际TTIR含tensor<256x...>的tt.atomic_rmw。后续ttadapter、bcmlir都出现带ExtractedLoadOrStore标注的scf.for，循环体把单个元素插入tensor<1xf16>，再用hivm.hir.store atomic=<add>写回。完整片段、原始IR路径和SHA见research/scatter_add/atomic-lowering-review.json及parent/ir/。\n\n因此至少对这份编译实例，可以从源码数据依赖写地址落实到逐lane写回循环。结合首批大输入约19.5ms、MTE3高活跃以及无冲突atomic也慢的探针，第二批不应只搜索BLOCK/warps。流水活跃高不代表HBM达到峰值，减少原子冲突也未必消除这种地址降低成本。\n\n优先保证BF16用FP32累加等语义。之后检查候选相同位置的实际IR是否仍有逐lane循环，再决定是否用按输出块归约、分组/排序或当前编译器已支持的低层数据搬运路径。每种替代均需完整计入预处理、暂存、转换与启动成本，不能调用PyTorch目标scatter_add。当前证据只证明父版代表实例，不宣称所有候选、dtype和输入均相同，也没有建立定量Roofline。\n\n这是第一批结束后Codex的离线编译证据分析，提供给第二批新overlay，不改写第一批模型贡献。\n",
  "GELU第一批失败与修复.md": "# gelu_and_mul：第一批高分尚未通过FP32补充门禁\n\n第一批四轮KG内测的完整19项均通过。第三轮内测1.8102×，但四份源码独立均12/18，失败集中全部六项FP32补充输入；15项原生BF16性能输入数值全通过。第四轮另在FP32宽8194输入编译UB overflow，要求1835008 bits，设备只有1572864 bits。不能把BF16合同通过扩大为三dtype的通用实现正确；也不能把当前高分当作有效交付。\n\n优先复用第三轮连续按行/块读写、exact-GELU表达式融合及预绑定启动架构，不从原版persistent stride动态寻址重新测四轮：\n\nresults/gelu_and_mul/attempt-1/candidates/c479b35793c26d80c14c902785cc2b3f54d83ed989456ffa3c4feb52d5db43f7/main.py\n\n原始semantics.json保存FP32偏差：随机输入最大约0.0013，大值探针约0.0044，超出1e-4容差。所有候选都用tl.erf，因此需要实际比较本机tl.erf、FP32参考F.gelu、准确FP32表达式的逐值误差。先确定误差来自erf降低、表达式收缩，还是NPU参考路径；不能只凭源码写了erf就宣称精度合格。严禁改容差或调用目标PyTorch gelu作为候选实现。\n\n可补测一个FP32 erf有界误差近似作为明确候选（尚未在本机验证）：z=x/sqrt(2)，u=abs(z)，t=1/(1+0.3275911*u)，poly=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t，erf(z)=sign(z)*(1-poly*exp(-u*u))。标准该组系数对数学erf的最大绝对误差约1.5e-7，但本机编译、乘法收缩与GELU参考仍须实际逐值验证；正负大值和零附近要检查。可按dtype保留已通过的低精度快路径，并为FP32使用更准确表达式和更小分块；正式完整调用必须包含该分流。\n\n第三轮launcher_key只有(dtype,hidden_size,device)，缺少输入地址对齐身份；第四轮已加入data_ptr()&15等完整标量身份，但不能因此照搬其8192分块。若保留第三轮启动缓存，应补齐对齐属性并只缓存编译产物，读取本次Tensor/stream，profiling用同kernel的标准元数据路径。独立check-gelu-launch-cache.py已加入先对齐预热再存储偏移的9项检查，可将候选放在本批目录后执行。检查不能与主卡锁队列并行发设备工作。\n\n优先通过check-endurance.py的18项及全部15项性能输入数值，再做新的完整KGS评测；额外地址检查也必须通过。该笔记由Codex在第一批结束后形成，供第二批新overlay读取，不改写第一批模型贡献或原始成绩。\n\n## 新的强线索：先核对NPU参考的实际公式\n\nCPU同种子同输入的exact与tanh GELU差异，六项最大误差的位置全部与NPU失败回执吻合：case0 (0,0)、case1 (1,16)、case2 (4,426)、case3 (0,492)、case4 (1,13)、case5 (0,25)。最大值也接近：CPU约0.00011659、0.00023793、0.00124709、0.00124649、0.00136075、0.00442140；NPU失败约0.00011656、0.00023797、0.00124749、0.00124563、0.00136111、0.00441873。CPU显式erf表达式与CPU exact的差异只有约1e-7，不能据此将tl.erf归咎为低精度。\n\n原始CPU见证在research/gelu_and_mul/exact-tanh-cpu-witness.json。它强烈指向本机NPU F.gelu(approximate='none')实际采用tanh型近似的可能，而非候选erf本身不准；仍需NPU逐值对照确认。第二批先以同一输入比较：CPU exact、CPU tanh、NPU F.gelu none、NPU F.gelu tanh、当前Triton候选，并保存真实任务和最大误差。这个探针应先于上文FP32 erf多项式替换，避免修错方向。\n\n若证实固定NPU参考none实际等同tanh，候选可按该已证实语义对齐，保留明确公式和精度说明；不能把tanh声称为数学exact，也不能改oracle或容差。若不成立，再定位erf降低、浮点收缩和舍入。先前“tanh不能默认替代exact”的约束仍成立：需要这些实际对照证据才能替换。补充门禁与原生合同继续全部执行。\n\n## 复用第一批KG分析角色已经提出的启动修复\n\n第一批第3轮的结构化分析已定位：BLOCK=4096、EVEN=True的不同hidden size可共用同一JIT编译cache。当前源码只在fresh=set(cache)-known恰好出现一个新条目时登记_LAUNCHER，因此已经有编译产物的hidden size可能一直不登记，重复走慢入口。这个方法来自第一批实际KG分析角色，不是Codex重新发明。下一项实验建议如下：\n\n正常调用_gelu_and_mul_rows_kernel[grid](...)后直接取得返回的CompiledKernel，按完整启动身份登记_LAUNCHER；删除known/fresh差集门槛。缓存键包含分块、EVEN、输入地址对齐及全部实际专门化属性，缓存只保存编译产物和元数据，保留profiling hook与正常路径回退。\n\n版本和首次执行次数约束：先确认固定FlagTree的调用返回可用编译对象；不可用时保留正常计算，不因登记失败再执行一遍。完整数值、真实缓存命中和每次一个设备任务均需验证。原分析预期宽8192的四项由约95–100微秒降至约51–60微秒，这是待验证预测，不是已取得收益。\n\n原始结构化分析：/data/hanle/ascend-optimization/runs/endurance-20261022/kg-runs/gelu_and_mul-endurance-20261022-attempt-1/stages/optimize/work/.kernelgen/profile-analysis/round-0003.json；SHA256：25f799e6f4c31bdb5f3eac3ead7ac7dc6e333e056fbd24df066532b684c998af。第二批优先实际验证该缓存登记修复，与FP32参考公式核对共同完成；不需要重新做一遍全部启动探针。\n",
  "uniform保留Philox的高位乘法备选.md": "# uniform：若Philox高位乘法降低低效，可测试等价32位拆分\n\n先保留真实生成器状态和完整seed/offset，应用_lean_seed_offset思路，再检查本机Philox的实际IR/热点。一个可区分的机制假设是：umulhi或64位中间乘法被低效降低。只有实际IR或探针支持时才沿此方向，不能预先断言它是瓶颈。\n\n若命中该机制，可以用16位半字分解计算uint32乘法的高32位，全程只用uint32乘法、加法和移位，保持Philox算法及轮数，避免通过替换成弱哈希来换性能：\n\n```python\nx0 = x & 0xffff\nx1 = x >> 16\nc0 = C & 0xffff  # constexpr\nc1 = C >> 16\nw0 = x0 * c0\nt = x1 * c0 + (w0 >> 16)\nw1 = x0 * c1 + (t & 0xffff)\nhi = x1 * c1 + (t >> 16) + (w1 >> 16)\nlo = x * C\n```\n\n所有x及中间量均为tl.uint32，右移必须无符号；C为Philox轮常数0xd2511f53或0xcd9e8d57。CPU固定边界和100000个随机uint32输入、两种常数的200016次检查均与(x*C)>>32一致，证据在research/uniform/philox-mulhi-cpu/。这是有限CPU身份检查，不是NPU加速证明，也不是替代完整算法验证。原生umulhi若已经高效矢量化，该拆分可能反而更慢，不要盲目采用。\n\n若试验，额外核对Philox4x32完整输出与原版一致、seed/offset高位、状态推进与恢复重放，再执行18项语义和一次完整KGS。只有总成本下降才保留。不要为了新seed重编译，不改固定FlagTree，不省略随机轮次；记录实际IR变化及完整耗时。此笔记是第一批结束后Codex的等价表达式备选，尚未在NPU验证。\n",
  "SELU复验分数与实际耗时.md": "# SELU第二批：按实际耗时识别参考分数漂移\n\n第二批第3轮通过36项完整评测及42+12+9项补充语义，独立相对PyTorch为0.97035×，高于起点0.86817×。但同case候选耗时几何平均由19.82799微秒增至20.27897微秒，实际候选加速比0.97776×，并未产生整组性能收益。原始JSON、SHA及逐case比较在research/selu_/round3-actual-latency-review.json。\n\n当前其他候选仍在验证，不能提前宣称最终入选版本。控制器按latency_ms选择，不按不同轮PyTorch分数选择；不能将0.97035/0.86817当作新增加速。局部大输入可能改善，需以逐case数据和同源码采集核对，不能自动推广为整组收益。每份新字节源码一次正式完整评测，原始失败与参考变化保留。\n\n这段来自第二批独立复验后的Codex比较，供之后覆盖层使用，不改写刚结束的模型建议与计算代码。\n\n## 第二批最终选版更新\n\n第二轮另一份源码8764018cfb2cab9444a34caf7a171f7eac7395656631698ce491e676557462f0完成全部门禁后，实际候选延迟为19.03031微秒，较起点快1.04192×（4.19%），独立相对PyTorch为0.83721×。它通过36项正式及42+12+9项补充检查，按实际延迟入选。第三轮虽0.97035×但20.27897微秒，仍未入选。完整比较及结果SHA在research/selu_/independent-round-comparison.json。原始第三轮比较不覆盖；额外采集兼容修复需要后续单列验证。\n",
  "独立门禁搜索内反馈.md": "# 独立门禁搜索内反馈\n\n本知识在累计积分第二批角色冻结之后发布，只供后续新建覆盖层读取，不追改正在运行的角色。\n\n前一批累计积分、scatter_add、uniform、GELU 出现过正式合同通过但扩展语义失败。性能计时输入也必须核对数值；高分不能覆盖错误结果。推荐在生成候选后、正式评测前，用原生调试接口检查当前源码；失败先修复，不把不合格候选当作优化成果。\n\n在当前 KG 工作区执行以下文件打包命令，按本轮实际算子、源码路径替换参数，并使用全新 tmp 子目录：\n\n```bash\npython /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/hgkda/stage-endurance-check.py \\\n  --batch-dir /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022 \\\n  --workspace \"$PWD\" --source tmp/main.py \\\n  --operator scatter_add --output tmp/independent-check-r2\n```\n\n上述 Bash 只复制文件，不运行 NPU、不改正式合同、不占另一张卡。将 stdout 中 `debug_request` 对象传入当前原生 `mcp__kernelgen__submit_debug_job`，再通过原生工具等待任务完成并读取 `artifacts/independent-<源码SHA前12位>.json`。不要在宿主或另一个进程直接运行候选。这里沿用当前 KG/KGS 调试任务的串行设备路径。\n\n打包时核对本批登记的检查器 SHA 和 definition/oracle/correctness/timing 四份合同 SHA；拒绝被修改的合同。检查回执必须对应本轮源码 SHA；修改任何源码字节后都应重新打包、重新检查。目录已存在时换新名字，不覆盖旧证据。\n\n对累计积分要特别读取冻结性能输入的低精度数值结果；scatter_add 检查重复索引和非连续布局；uniform 检查真实 generator 推进、恢复重放以及 seed/offset 高位；GELU 检查 FP32 参考公式差异，不能只看 BF16 正式计时输入。SELU 仍使用既有三项专用检查，不支持本入口。\n\n本工具是搜索内提前反馈，不代替控制器最终补充语义门禁和一次完整 KGS 正式评测。保存调试请求、原始回执、源码 SHA 和修复记录。只有实际调用且收到对应回执，才可报告本轮采用了这项反馈。\n"
}

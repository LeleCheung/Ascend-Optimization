---
name: kernel-knowledge-profile-analyzer
description: "Use this agent for profile analysis inside a KernelGenWorkflow workspace with an explicitly materialized V1 Knowledge Catalog."
capabilities: [read, search, skill]
mcp_tools: [query_knowledge, get_knowledge, query_sources, get_source, get_profile_context, profile_workloads, record_profile_analysis]
subagents: []
model: inherit
approval: no_prompts
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

# 两项收尾任务

只聚焦upsample_bilinear2d、cumulative_trapezoid。上采样从完整有效2054ca源码约0.340×继续，目标完整调用>0.8×。累计积分参考FP16非末轴存在重复调用超过原容差反例，先定位最小形状和官方确定性选项；诊断选项仅用于隔离进程，不改变原正式合同、容差或参考。

本轮优先验证新的数据流：CANN extension.extract_slice/insert_slice替代水平gather；标量行基址连续读取；独立上下两行/相邻输出行复用；可分离两kernel；必要时本机TLE DSA显式UB子视图。每条路线先小探针数值和完整调用收益，再完整门禁及一次正式评测。此前连续输入块、固定stride、位打包均未带来有效收益，不能重复当成新成果。

不要以裸kernel少、program少、理论搬运少直接认定快。小输入候选约109微秒，参考约33微秒，大输入约630微秒，参考约206微秒，必须同时处理launch参数和设备路径。libentry可正常验证，保持标准kernel[grid]、当前stream和独立输出，禁调被优化PyTorch/FlagGems算子或缓存输入/结果。

正确性包含原correctness/timing全部输入、stride与显式scale、两次及换stream；上采样另36项特殊门禁。一次正式完整复验，不反复测同字节挑分。有限方法预算仍未达标时保留真实最佳、具体编译/执行限制与累计积分最小参考反例。Codex实质诊断/修改与CC/DeepSeek生成分别记录。

固定FlagGems d6a8eec/FlagTree c286cba6、CANN9.0、tle_yy、物理卡7、KGS19659、串行卡锁；不重启服务、不动他人进程或共享后端。导入oracle关闭字节码写入，保持严格catalog洁净。


## 本项已知证据与失败

{
  "operator": "upsample_bilinear2d",
  "seed": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-late/7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a/main.py",
  "seed_sha256": "7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a",
  "current_operator_state": {
    "status": "搜索中",
    "best": {
      "name": "safe_compiled_launch",
      "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-late/7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a/main.py",
      "source_sha256": "7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a",
      "diagnostic_gain": 1.5057600180923747,
      "contribution": "Codex实现与诊断；KGS独立准入、补充语义及一次正式评测",
      "status": "passed",
      "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-late/7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a/formal/safe_compiled_launch-1.result.json",
      "geo_mean": 0.5590417066556866,
      "latency_ms": 0.1621556248650184,
      "gates": [
        "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-late/7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a/remaining-semantics.json",
        "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-late/7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a/special.json"
      ]
    },
    "parent": {
      "status": "passed",
      "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/seeds/upsample_bilinear2d.py",
      "source_sha256": "2054ca728198ae2a395addf604834a00f67f4e5352acc84adb93250457bee84b",
      "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/parent/formal/parent-1.result.json",
      "geo_mean": 0.34489377567486296,
      "latency_ms": 0.2562887254602058,
      "gate_result": {
        "status": "passed",
        "checks": [
          "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/parent/remaining-semantics.json",
          "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/parent/special.json"
        ]
      },
      "gates": [
        "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/parent/remaining-semantics.json",
        "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/parent/special.json"
      ]
    },
    "actual_speedup_over_parent": 1.580510855997414,
    "attempts": [
      {
        "attempt": "attempt-1",
        "workspace": "/data/hanle/ascend-optimization/runs/two-closure-20261024/kg-runs/upsample_bilinear2d-two-closure-20261024-attempt-1",
        "valid_candidates": 1,
        "roles": 4
      }
    ]
  },
  "completed_jobs": {
    "validate:upsample_bilinear2d:codex-initial:7d91b983f295": {
      "status": "completed",
      "started_at": "2026-10-24T09:52:27.578736Z",
      "result": {
        "status": "passed",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-initial/7d91b983f295e34486d4881ae6b5afa94c07d112ecaf10cd37306cf299d14a10/main.py",
        "source_sha256": "7d91b983f295e34486d4881ae6b5afa94c07d112ecaf10cd37306cf299d14a10",
        "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-initial/7d91b983f295e34486d4881ae6b5afa94c07d112ecaf10cd37306cf299d14a10/formal/codex-libentry-1.result.json",
        "geo_mean": 0.3272800578972862,
        "latency_ms": 0.2685751459248721,
        "gates": [
          "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-initial/7d91b983f295e34486d4881ae6b5afa94c07d112ecaf10cd37306cf299d14a10/remaining-semantics.json",
          "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-initial/7d91b983f295e34486d4881ae6b5afa94c07d112ecaf10cd37306cf299d14a10/special.json"
        ]
      },
      "finished_at": "2026-10-24T09:59:17.961898Z"
    },
    "parent:upsample_bilinear2d": {
      "status": "completed",
      "started_at": "2026-10-24T09:59:17.985309Z",
      "result": {
        "status": "passed",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/seeds/upsample_bilinear2d.py",
        "source_sha256": "2054ca728198ae2a395addf604834a00f67f4e5352acc84adb93250457bee84b",
        "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/parent/formal/parent-1.result.json",
        "geo_mean": 0.34489377567486296,
        "latency_ms": 0.2562887254602058,
        "gate_result": {
          "status": "passed",
          "checks": [
            "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/parent/remaining-semantics.json",
            "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/parent/special.json"
          ]
        },
        "gates": [
          "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/parent/remaining-semantics.json",
          "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/parent/special.json"
        ]
      },
      "finished_at": "2026-10-24T10:01:18.098404Z"
    },
    "observe-parent:upsample_bilinear2d": {
      "status": "completed",
      "started_at": "2026-10-24T10:01:18.106052Z",
      "result": {
        "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/research/upsample_bilinear2d/parent/facts.json",
        "status": "completed"
      },
      "finished_at": "2026-10-24T10:01:48.138784Z"
    },
    "metrics-parent:upsample_bilinear2d": {
      "status": "completed",
      "started_at": "2026-10-24T10:01:48.145876Z",
      "result": {
        "path": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/research/upsample_bilinear2d/parent-metrics/response.json",
        "status": "completed"
      },
      "finished_at": "2026-10-24T10:02:18.169307Z"
    },
    "search:upsample_bilinear2d:1": {
      "status": "completed",
      "started_at": "2026-10-24T11:13:56.359088Z",
      "result": {
        "workspace": "/data/hanle/ascend-optimization/runs/two-closure-20261024/kg-runs/upsample_bilinear2d-two-closure-20261024-attempt-1",
        "valid_candidates": 1,
        "best": {
          "name": "safe_compiled_launch",
          "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-late/7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a/main.py",
          "source_sha256": "7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a",
          "diagnostic_gain": 1.5057600180923747,
          "contribution": "Codex实现与诊断；KGS独立准入、补充语义及一次正式评测",
          "status": "passed",
          "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-late/7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a/formal/safe_compiled_launch-1.result.json",
          "geo_mean": 0.5590417066556866,
          "latency_ms": 0.1621556248650184,
          "gates": [
            "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-late/7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a/remaining-semantics.json",
            "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-late/7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a/special.json"
          ]
        }
      },
      "finished_at": "2026-10-24T11:15:16.420244Z"
    },
    "validate:upsample_bilinear2d:attempt-1:7d91b983f295": {
      "status": "completed",
      "started_at": "2026-10-24T11:14:06.378477Z",
      "result": {
        "status": "passed",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/attempt-1/candidates/7d91b983f295e34486d4881ae6b5afa94c07d112ecaf10cd37306cf299d14a10/main.py",
        "source_sha256": "7d91b983f295e34486d4881ae6b5afa94c07d112ecaf10cd37306cf299d14a10",
        "result": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/codex-initial/7d91b983f295e34486d4881ae6b5afa94c07d112ecaf10cd37306cf299d14a10/formal/codex-libentry-1.result.json",
        "geo_mean": 0.3272800578972862,
        "latency_ms": 0.2685751459248721,
        "reused": true,
        "gates": [
          "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/attempt-1/candidates/7d91b983f295e34486d4881ae6b5afa94c07d112ecaf10cd37306cf299d14a10/remaining-semantics.json",
          "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/results/upsample_bilinear2d/attempt-1/candidates/7d91b983f295e34486d4881ae6b5afa94c07d112ecaf10cd37306cf299d14a10/special.json"
        ]
      },
      "finished_at": "2026-10-24T11:15:16.417477Z"
    },
    "search:upsample_bilinear2d:2": {
      "status": "running",
      "started_at": "2026-10-24T11:17:56.537280Z"
    }
  },
  "frozen_knowledge": [
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/累计积分参考自洽反例已实测.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/累计积分九项舍入与准确参考源码.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/上采样第二轮建议.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/上采样连续输入块与启动建议.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/完整调用的启动层开销.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/previous-delivery.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/夜间任务.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/tool-sha256.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/supplemental-tool-sha256.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-libentry.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-slice_neighbors.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair_gather_r4.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair_gather_r16.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair_slice_r4.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair_slice_r16.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/initial-variants.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/diagnostic-tool-sha256.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/本轮新证据与上下文隔离.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/independent-upsample-probe.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-padded_slice.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/补齐切片候选.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/flat-gather-variants.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/reference-new-findings.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/分块与缓冲新路线.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair_cat_r4.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair_cat_r16.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/pair-cat-variants.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/extra-upsample-probe.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/diagnose-extra-probes.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/stable-cumulative-models.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/extra-upsample-probe-before-models.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/相邻行独立进程实测.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-safe_pair_gather.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-safe_pair_cat.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/safe-pair-variants.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/相邻行边界地址新候选.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/连续输出与切片独立实测.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/cumulative-cann-sync-probe.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/probe-cann-hybrid.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair2d_r1.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair2d_r4.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair2d_r8_single.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/pair2d-variants.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/joined-upsample-probe.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/diagnose-joined-probes.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/evaluate-cann-deterministic.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/cumulative-cann-async.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/probe-cann-hybrid-async.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/连续拼接与原生扫描实测.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/diagnose-cann-evaluation.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-scalar_row_r1.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-scalar_row_r4.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/scalar-row-variants.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/本轮后续探针与选择约束.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/diagnose-scalar-row.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/scalar-row-upsample-probe.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-safe_compiled_launch.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/compiled-launch-probe.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/evaluate-cann-async-deterministic.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/diagnose-final-closure-probes.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/cumulative-cann-contiguous.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/probe-cann-contiguous.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/evaluate-cann-contiguous-deterministic.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/标量行基址独立实测.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-flat_store_launch.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-flat_store_single.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/flat-store-variants.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/diagnose-dataflow-probes.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/dataflow-upsample-probe.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/编译启动与连续输出后续验证.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/diagnose-late-ups-validation.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/probe-candidate-first-deterministic.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/cumulative-cann-fifo.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/evaluate-cann-fifo-deterministic.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/diagnose-last-scan-checks.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-flat_gather_launch.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-flat_gather_single.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/validate-two-ups-late-executed-v1.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/no-join-upsample-probe.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/diagnose-nojoin-probes.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/本批完整验收新起点.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/无join连续输出独立实测.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/probe-cann-host-components.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/diagnose-host-components.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/cumulative-cann-launch.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/evaluate-cann-launch-deterministic.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/diagnose-scan-launch.py",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/恢复与宿主开销新证据.md"
  ],
  "paired_execution": {
    "case": {
      "name": "upsample_bilinear2d-timing-0",
      "inputs": {
        "spec": {
          "dtype": "float16",
          "shape": [
            1,
            16,
            64,
            64
          ],
          "output_size": [
            128,
            128
          ],
          "align_corners": false
        }
      },
      "seed": 1729,
      "tolerance": {
        "atol": 0.015,
        "rtol": 0.015,
        "required_matched_ratio": 1.0
      }
    },
    "source_sha256": "2054ca728198ae2a395addf604834a00f67f4e5352acc84adb93250457bee84b",
    "sides": {
      "reference": {
        "status": "completed",
        "trace": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/research/upsample_bilinear2d/parent/reference/trace.json",
        "device_events": [
          {
            "name": "aclnnUpsampleBilinear2d_CastAiCore_Cast",
            "dur": 3.180125,
            "ts": "1792836087770337.864",
            "pid": 1113995687,
            "tid": 8,
            "args": {
              "Model Id": 4294967295,
              "Task Type": "KERNEL_AIVEC",
              "Physic Stream Id": 8,
              "Task Id": 5,
              "Batch Id": 0,
              "Subtask Id": 4294967295,
              "connection_id": 3
            }
          },
          {
            "name": "aclnnUpsampleBilinear2d_TransDataSpecial_TransData",
            "dur": 6.700125,
            "ts": "1792836087770341.064",
            "pid": 1113995687,
            "tid": 8,
            "args": {
              "Model Id": 4294967295,
              "Task Type": "KERNEL_AIVEC",
              "Physic Stream Id": 8,
              "Task Id": 6,
              "Batch Id": 0,
              "Subtask Id": 4294967295,
              "connection_id": 5
            }
          },
          {
            "name": "aclnnUpsampleBilinear2d_TransDataSpecial_TransData",
            "dur": 5.640125,
            "ts": "1792836087770347.784",
            "pid": 1113995687,
            "tid": 8,
            "args": {
              "Model Id": 4294967295,
              "Task Type": "KERNEL_AIVEC",
              "Physic Stream Id": 8,
              "Task Id": 7,
              "Batch Id": 0,
              "Subtask Id": 4294967295,
              "connection_id": 7
            }
          },
          {
            "name": "aclnnUpsampleBilinear2d_ResizeBilinearV2AICORE_ResizeBilinearV2",
            "dur": 11.60025,
            "ts": "1792836087770367.424",
            "pid": 1113995687,
            "tid": 8,
            "args": {
              "Model Id": 4294967295,
              "Task Type": "KERNEL_AIVEC",
              "Physic Stream Id": 8,
              "Task Id": 8,
              "Batch Id": 0,
              "Subtask Id": 4294967295,
              "connection_id": 9
            }
          },
          {
            "name": "aclnnUpsampleBilinear2d_TransData_TransData",
            "dur": 5.320125,
            "ts": "1792836087770379.045",
            "pid": 1113995687,
            "tid": 8,
            "args": {
              "Model Id": 4294967295,
              "Task Type": "KERNEL_AIVEC",
              "Physic Stream Id": 8,
              "Task Id": 9,
              "Batch Id": 0,
              "Subtask Id": 4294967295,
              "connection_id": 11
            }
          }
        ],
        "total_device_events": 5,
        "error": null
      },
      "candidate": {
        "status": "completed",
        "trace": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/research/upsample_bilinear2d/parent/candidate/trace.json",
        "device_events": [
          {
            "name": "_bilinear_int_kernel",
            "dur": 41.46075,
            "ts": "1792836096211639.542",
            "pid": 1113995687,
            "tid": 8,
            "args": {
              "Model Id": 4294967295,
              "Task Type": "KERNEL_AIVEC",
              "Physic Stream Id": 8,
              "Task Id": 11,
              "Batch Id": 0,
              "Subtask Id": 4294967295,
              "connection_id": 3
            }
          }
        ],
        "total_device_events": 1,
        "error": null
      }
    }
  },
  "parent_metrics": {
    "status": "completed",
    "metrics": {
      "avg_time_us": 629.112,
      "ops": [
        {
          "op_name": "_bilinear_int_kernel",
          "op_type": "_bilinear_int_kernel",
          "task_type": "AI_VECTOR_CORE",
          "measured_invocations": 2,
          "total_duration_us": 1258.225,
          "avg_duration_us": 629.112,
          "min_duration_us": 628.673,
          "max_duration_us": 629.552,
          "aicore_time_us": 0.0,
          "aic_total_cycles": 0.0,
          "aic_mac_time_us": 0.0,
          "aic_mac_ratio": 0.0,
          "aic_scalar_time_us": 0.0,
          "aic_scalar_ratio": 0.0,
          "aic_mte1_time_us": 0.0,
          "aic_mte1_ratio": 0.0,
          "aic_mte2_time_us": 0.0,
          "aic_mte2_ratio": 0.0,
          "aic_fixpipe_time_us": 0.0,
          "aic_fixpipe_ratio": 0.0,
          "aic_icache_miss_rate": 0.0,
          "aiv_time_us": 629.9045,
          "aiv_total_cycles": 40934101.0,
          "aiv_vec_time_us": 415.3655,
          "aiv_vec_ratio": 0.6595,
          "aiv_scalar_time_us": 215.9375,
          "aiv_scalar_ratio": 0.3425,
          "aiv_mte2_time_us": 179.3425,
          "aiv_mte2_ratio": 0.285,
          "aiv_mte3_time_us": 8.188,
          "aiv_mte3_ratio": 0.013,
          "aiv_icache_miss_rate": 0.0,
          "cube_utilization_pct": 0.0
        }
      ]
    },
    "original": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/research/upsample_bilinear2d/parent-metrics/response.json"
  },
  "累计积分参考自洽反例已实测.md": "# 累计积分：原参考在FP16非末轴上的自洽反例\n\n## 已完成的NPU实测\n\nCodex经KGS19659、物理卡7提交`e0cee142b2804cfb992bb8bdcc1d5324`，状态SUCCEEDED。原始探针、响应、精确JSON与本批己方进程恢复记录在`research/cumulative_trapezoid/reference-stability-probe/`。结果JSON的SHA256为`e09115e22363b0830869986d7a7e338b5aea7b3b2a856b7d383ca7fae3527316`。探针不修改确定性设置、参考、容差或候选。\n\n冻结`cumulative_trapezoid-timing-2`：FP16、shape=(1024,128)、dim=0、dx=1，同一份输入，原参考连续调用六次。与第一次比较，五次分别有644、600、518、560、593个元素超原容差，最大绝对差0.1875。六次结果按原`atol=rtol=0.015`构造允许区间后，**380个位置的交集为空**。因此不存在一份固定输出能同时满足这六次参考结果，不能靠修改候选的扫描结合顺序保证该用例通过。\n\n具体位置[251,1]的六次结果为-0.0078125、0、-0.0078125、-0.0234375、0.0078125、-0.00390625，允许区间的最大下界-0.0073046875大于最小上界-0.0080859375。\n\n同一个显式增量张量直接NPU cumsum也不自洽：左乘路径的两次自身对照分别有531、354元素超容差；右乘路径分别536、541。左乘、右乘生成的增量逐值相同，全部为ND format=2、标准连续stride。因此这次观察已将问题收敛到本机FP16非末轴Cumsum，而不是仅由PyTorch外层积分表达式或乘法顺序解释。\n\n对照项：FP16末轴timing-1六次参考逐值一致；BF16非末轴timing-5六次逐值一致；FP32非末轴timing-8虽有微小变化，但全在原容差内且没有空交集。\n\n## 对本轮交付的影响\n\n- 原始合同、参考、容差保持不变。不反复抽样直至偶然通过，不用缓存参考输出或输入内容绕过门禁。\n- 累计积分仍按未完成正式闭环报告，并明确该参考限制。不能把扫描搜索中的局部高分填成最终成绩。\n- 下一轮可以优化可自洽的输入、验证融合增量或纯Triton扫描，但必须单独写明原完整门禁尚未解决。直接接CANN Cumsum也不自动解决同一原生组件的非确定性。\n- 不擅自全局开启确定性或更换参考来制造通过。将准确版本、输入、反例和原始结果交付，供后续专家确认评测口径或本机组件修复。\n",
  "累计积分九项舍入与准确参考源码.md": "# 累计积分：九项真实输入与准确参考源码\n\nCodex在物理卡7经KGS19659完成独立探针，job `c135af6922c34bf68852044f70519e8a`。结果及控制进程恢复记录见`research/cumulative_trapezoid/rounding-reference-probe/`。全部九个冻结timing输入均已检查，不改参考或容差。\n\n## 修正之前的顺序累加猜想\n\n- BF16三项：先保留相邻加法和缩放的输入dtype舍入，再FP32扫描，可逐值匹配参考。\n- FP16末轴两项：原dtype的NPU cumsum逐值匹配；FP32扫描、CPU顺序FP16/FP32以及CPU FP64前缀均不能匹配。4096长序列FP32扫描有约49500元素超容差，顺序FP16约600452元素超容差。因此不能继续把“直接FP16顺序累加”等同参考。\n- FP16非末轴dim0：本次显式增量接原dtype NPU cumsum也有643元素不符，最大绝对误差0.25，尚待同输入稳定性和表达式对照。不能擅自称参考不确定或容差错误。\n- FP32三项：CPU FP64前缀再转回均通过；顺序FP32在大输入有9元素超容差；NPU原dtype扫描均在容差内。\n\n## 精确版本源码\n\n已归档与实际Torch/torch_npu提交相符的Integration.cpp及op-plugin源码，URL和SHA见`cumulative-exact-reference-sources.json`。PyTorch标量dx重载精确为`(dx / 2. * (left + right)).cumsum(dim)`。Tensor dx是另一重载，不能混用。\n\n实际op-plugin OpApi接口调用aclnnCumsum，dtype来自输出类型，不是aclnnCumsumV2。准确CPP和已安装CANN头文件相互印证接口，但不自动证明内部累加算法或硬件累加精度。\n\n## 下一轮优先任务\n\n先让全部timing数值通过再谈性能。纯Triton扫描须查真实dtype、算法及结合顺序；另一条可试路线为自己的Triton融合相邻和及缩放，接CANN原生Cumsum。后者须标记混合组件融合，不能写成纯Triton扫描优化，不能调用PyTorch cumsum代替候选，也不能绕过准入。参考稳定性探针结果未出前，不依据猜想改参考或放宽门禁。\n",
  "上采样第二轮建议.md": "# 上采样首轮之后的增量路线\n\n首轮四份源码正式24/24通过，搜索最佳约0.345×。真实debug已经区分：逐点四次GM读取约290ms；行加载后使用偏移GM加载约21ms；对齐行加载＋片上gather降到约0.6ms的大输入路径。盲目删除mask没有提升。小输入PyTorch约30微秒，标准Triton裸launch探针约57微秒，host路径也需要单独处理。全部数值为当前环境的诊断或搜索记录，最终看独立验收。\n\n下一轮优先修改数据复用和程序组织，而非重复同样的分块扫描：\n\n- 整数2倍上采样可沿输入行组织，一个program同时产生相邻输出行/列，复用相邻输入行和水平邻点，避免分别生成每个输出行时反复做gather及地址计算。权重来自真实align_corners=False和有效scale=2的条件；不能只凭输出尺寸2倍就忽略显式scale。边界单独钳位，保留一般比例/stride路径。\n- 评估水平/垂直可分离方案及完整融合两者。分两kernel多一次中间张量，但可能用更规则读写换掉复杂gather；必须将分配、写读临时张量和全部launch计入最终成本。不要仅凭单kernel更少就拒绝比较，也不要忽略FP32中间值与最终低精度舍入。\n- 针对大输入，查实际IR/指令确认tl.gather及join是否生成逐lane循环、transpose或多次UB搬运，再决定用索引shift、切片或等价组织替换。已有instruction采集因240秒预算取消，不表示没有瓶颈；先读已归档IR和真机metrics。常规编译缓存可读，不改全局后端。\n- 小输入若优化host启动，遵守当前preflight准入：标准kernel[grid](...)调用、元数据缓存及本机可用libentry；不绕过准入规则。直接raw launcher仅在明确可接受且当前stream/指针/对齐/尾部全部正确时另作方案，不把旧算子允许性自动推广。\n\n最终还需运行check-remaining-special.py的36项显式scale/布局门禁。原版ABI失败与全部候选失败继续保留。该建议为Codex对本批证据的追加诊断，供第二轮新覆盖层读取；不改写首轮已冻结角色。\n",
  "上采样连续输入块与启动建议.md": "# 上采样：用连续输入块替代逐输出行重复读取\n\n本建议由Codex根据首轮真实源码和debug数据提出，尚未在NPU验证，不能计为优化成果。当前源码每个program处理BR个输出行，分别为每行读取两个输入行；相邻输出行实际上重复读取输入。本批大输入为4×32×128×128，输出4×32×256×256，BR=16时共有2048个program。\n\n## 优先做的不同数据流\n\n对于有效高度和宽度均2倍、align_corners=False的连续NCHW路径，令一个program处理R个连续输入行对应的2R个输出行。加载一个包含上下halo的连续输入行块，长度为next_power_of_2(R+2)×IW；沿最后一维做连续GM读取，之后在UB中用gather选上下行和水平邻点，再插值、join并连续写输出。可试R=4、8，先证明halo/首尾/跨channel正确。相较当前为2R个输出行做4R次行读取，连续块可以减少重复GM搬运和DMA调用。\n\n具体索引例：输出行oy=2*q+j，对应源坐标q-0.25（j=0）或q+0.25（j=1）；上下行分别为(q-1,q)或(q,q+1)，边界clamp。patch起始行q_base-1，加载负行/超尾行时mask，gather索引使用clamp后的真实源行减patch起点。行块不得越过同一N/C图像。若next_power_of_2补齐导致tile太大，应缩R；所有中间张量和编译缓冲都计入UB。也可以先只在高度使用patch复用，水平保留现有gather。\n\n这是待验证方案，UB gather仍可能降低成标量循环，须对比真实IR/设备时间。不把GM搬运理论减少当作实测性能提升。非2倍、非单位末维stride、显式scale不等价于2倍时保留一般路径。\n\nCPU预检已完成：`upsample-patch-cpu-model.py`按上述halo和上下行索引实现，共3种dtype×8种形状×3种R，72/72与CPU参考吻合，包括高度/宽度1、奇数尾块、多通道和批次。结果和脚本SHA见`upsample-patch-cpu-result.json`。这只验证索引和插值等价性，不证明Ascend编译或性能；新kernel仍需真实NPU全部门禁。\n\n## 启动开销\n\n首轮小输入裸普通Triton launch约57微秒，不能把它视为语言或硬件的不可逾越下限。固定版本FlagGems已有from flag_gems.utils import libentry，@libentry配合@triton.jit在其他已有算子使用过，优先验证当前单kernel加装饰器后的完整调用开销和数值；无需绕过标准kernel[grid](...)接口。缓存元数据仍须包含shape/stride/dtype/scale，保持当前stream。s_n/s_c/s_h等实际只依赖元数据的stride也可constexpr以便后端识别连续布局，但要记录编译次数及多布局验证，不能假定所有输入连续。\n\n检查顺序：小输入启动开销、大输入真实设备时间、正式24项、独立48项及全部6项timing数值，最后36项显式scale/布局补验。只按相同case实际候选延迟选择新版本。\n",
  "完整调用的启动层开销.md": "# 设备加载与启动层分开处理\n\n本批上采样第一轮24/24通过，但大输入候选完整调用约290ms，参考约0.2ms。CC已提交专项设备探针区分四次向量指针load、连续行加载后gather与program开销；以实际debug回执为准。不要只靠扩大分块重复同一逐lane加载路径。\n\n探针还出现约0.1ms的launch-only设备Event区间。该计时可能混入host提交间隙，不能把它直接称为纯设备启动延迟，更不能用Event数值代替KGS完整调用成绩。先优化设备寻址，再对小输入分别检查Python/JIT调用、参数封送、分配与设备任务。\n\n仓库已有本机验证的启动层实例：`operators/交付源码-20261023/rad2deg_/main.py`、`operators/交付源码-20261023/selu_/main.py`。其中普通Triton首次编译后复用generated launcher，后续每次仍传当前指针、shape及stream，缓存仅保存安全的编译特化与启动函数；注意输入对齐、masked尾部、dtype及布局不能沿用错误假设。先考虑固定FlagGems的libentry/安全编译缓存，再按真实证据判断是否需要直接启动函数。\n\n上采样有独立输出，缓存不得保存输出Tensor或用过期输出指针。若实现新launch路径，须检查新分配输入、换指针、奇数形状、非连续布局、当前stream，以及真实profiler注册；不能为profiler改成另一套计算kernel。最终以完整KGS和全部计时输入数值门禁验收。以上是待实测的方法参考，不预设能达到0.8×。\n",
  "本轮新证据与上下文隔离.md": "# 本轮新证据：邻点切片与确定性\n\n初始两个FP16 case：旧2054源码完整同步调用约0.220/0.785ms，添加本机libentry约0.185/0.779ms。仅诊断，并非正式分数。libentry版正进行独立完整门禁与正式评测。\n\nextension.extract_slice/insert_slice替代水平gather的候选出现真实vector core exception。此后同进程执行的pair_gather_r4/r16、pair_slice_r4/r16受已损坏上下文影响，失败不证明它们独立也错误。不得反复用同进程串行测试trap后的其他kernel；后续请每个新源码单独KGS debug进程，失败后由KGS检查健康，不手动重启服务。\n\n尚需独立检验的pair源码在knowledge/initial-variants.json中，优先pair_gather_r4，读取上下相邻输入行同时产生两个输出行；所有GM地址是一维仿射，水平方向仍使用gather。寄存器/UB代价和收益必须实测。slice候选trap原件见research/initial-upsample/。\n\n本机extension.index_select_simd可明确将多个GM行读入UB，要求索引为一维、选取轴不是末维；extension.gather_out_to_ub支持GM到UB多维gather边界。可以和普通向量指针load比较，不等同于已验证快。TLE DSA提供tile_alloc/tile_subview/tile_copy/tile_to_tensor/insert_slice/extract_slice/tile_concat，安装版本已存在，不需要更新共享环境。\n\n累计积分隔离诊断：默认FP16 shape=(256,8)积分已有一处六次原容差区间无交集；完整1024×128仍数百处。torch.use_deterministic_algorithms(True)使本轮5种形状的cumsum与积分全部六次逐值一致。这是官方稳定执行选项的实测，不等同于原默认验收通过。原参考、容差和正式请求不变；隔离确定性结果另立实验条件，不跨请求改变共享进程全局选项。\n",
  "补齐切片候选.md": "# 对齐切片输出候选\n\n原extract_slice使用IW-1的非2幂输出，触发vector core exception。新原型upsample-padded_slice.py先拼接v两份，再提取IW个元素、分别偏移IW-1和1，最后用边界where钳位；切片输出恢复为2幂维度。这个差别是待实测假设，不证明trap一定来自非2幂维度。独立进程运行，失败后不在同进程测试后续kernel。\n\n源码SHA：54ebda791fd866a51c72546a124442dd73f41f9b0c97ceaac02fee4cf4085d55\n",
  "分块与缓冲新路线.md": "# 分块受UB约束时的独立候选\n\n旧探针BR32因192KiB可用UB不足而失败，但这不证明所有BR32都不可行。可基于固定compiler可用参数，分别尝试num_stages=1、multibuffer=False或enable_ubuf_saving=True，观察是否允许更大BR并减少program数量；每次仅对本候选launch设置，不改共享后端默认或环境。\n\n旧最终metadata中multibuffer=True、num_stages=2、num_warps=32、auto_tile_and_bind_subblock=True。能否改善取决于真实缓冲存活与流水，禁用多缓冲也可能更慢。先小型完整同步测试，记录编译缓冲占用及延迟，成功后完整测试。不要盲目组合所有选项；单独分支和失败保留，不以通过编译代替性能收益。\n",
  "相邻行独立进程实测.md": "# 独立进程复核相邻行复用候选\n\npair_gather_r4：FAILED，job 0053939dc2304a2ca484b1151c109060。       Kernel task happen error, retCode=0x31, [vector core exception].[FUNC:PreCheckTaskErr][FILE:davinci_kernel_task.cc][LINE:1729]\n       rtDeviceSynchronizeWithTimeout execution failed, reason=vector core exception[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]\n       wait for compute device to finish failed, runtime result = 507035.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]\n\n[W1024 18:03:21.322735239 NPUStream.cpp:596] Warning: NPU warning, error code is 507035[Error]: \n[Error]: The vector core execution is abnormal. \n        Rectify the fault based on the error information in the ascend log.\nEE9999: Inner Error!\nEE9999[PID: 1093310] 2026-10-24-18:03:21.154.558 (EE9999):  rtDeviceSynchronizeWithTimeout execution failed, reason=vector core exception[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]\n        TraceBack (most recent call last):\n       wait for compute device to finish failed, runtime result = 507035.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]\n (function npuSynchronizeUsedDevices)\n[W1024 18:03:21.323657900 NPUStream.cpp:578] Warning: NPU warning, error code is 507035[Error]: \n[Error]: The vector core execution is abnormal. \n        Rectify the fault based on the error information in the ascend log.\nEE9999: Inner Error!\nEE9999[PID: 1093310] 2026-10-24-18:03:21.155.645 (EE9999):  rtDeviceSynchronizeWithTimeout execution failed, reason=vector core exception[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]\n        TraceBack (most recent call last):\n       wait for compute device to finish failed, runtime result = 507035.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]\n (function npuSynchronizeDevice)\n\n\npair_gather_r16：FAILED，job e904dbdd72b24515935713074394847f。ock is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair_gather_r16.py\":60:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair_gather_r16.py\":60:0): error: ub overflow, requires 2621696 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair_gather_r16.py\":60:0): error: Failed to run BiShengHIR pipeline\n\nloc(\"/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024/knowledge/upsample-pair_gather_r16.py\":60:0): error: ub overflow, requires 2621696 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)\n[ERROR] Failed to run BiShengIR pipeline\n\n\n[INFO]: The compiled kernel cache is in /tmp/kernelgen_server_debug_jobs/e904dbdd72b24515935713074394847f/home/.triton/cache/RYAQXEG7QRH3JCO3U4PENSGW6YOFGTQPAVF722YARGPVANFLBQAQ\n\n///------------------[ERROR][Triton][END]------------------\n\n\n\npair_slice_r4：FAILED，job 5074108b2dee418b808fd9b0a4d83af9。/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/npu/__init__.py:539: UserWarning: You can set the device capability via the environment variable TORCH_NPU_DEVICE_CAPABILITY. The return value of get_device_capability is only for compatibility with PyTorch and does not represent the actual capability of the NPU device.\n  warnings.warn(f\"You can set the device capability via the environment variable TORCH_NPU_DEVICE_CAPABILITY. {warning_str}\")\n/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/utils/device_info.py:58: UserWarning: [device_info] torch_device_fn.get_device_capability returned None for device_id=0, fallback to (0, 0).\n  warnings.warn(\nTraceback (most recent call last):\n  File \"/tmp/kernelgen_server_debug_jobs/5074108b2dee418b808fd9b0a4d83af9/workspace/probe.py\", line 32, in <module>\n    if __name__=='__main__':main()\n                            ^^^^^^\n  File \"/tmp/kernelgen_server_debug_jobs/5074108b2dee418b808fd9b0a4d83af9/workspace/probe.py\", line 24, in main\n    tol=case['tolerance'];torch.testing.assert_close(got.cpu(),expected.cpu(),atol=tol['atol'],rtol=tol['rtol'])\n                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/torch/testing/_comparison.py\", line 1589, in assert_close\n    raise error_metas[0].to_error(msg)\nAssertionError: Tensor-likes are not close!\n\nMismatched elements: 241711 / 262144 (92.2%)\nGreatest absolute difference: 6.44921875 at index (0, 10, 8, 45) (up to 0.015 allowed)\nGreatest relative difference: 46656.0 at index (0, 12, 85, 105) (up to 0.015 allowed)\n\n\n",
  "相邻行边界地址新候选.md": "# 边界地址隔离\n\npair原型的prev/nxt在首尾program形成分配范围外的指针，再依赖mask屏蔽。虽然数学上mask应正确，独立NPU探针出现MTE/vector异常，需区分是否为本机masked DMA lowering与负/超尾指针组合。safe_pair_gather/cat新候选仅在边界program钳位实际load地址、随后where恢复准确语义；主体program仍连续加载。不能先认定异常来自后端而跳过地址证明；每份源码独立进程验证。此候选尚未验证。\n",
  "连续输出与切片独立实测.md": "# 独立进程复核相邻行复用候选\n\npair_cat_r4：FAILED，job 50db326b21444e32962c9f1732983f83。et = super().visit(node)\n          ^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1476, in visit_Call\n    args = [self.visit(arg) for arg in node.args]\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1476, in <listcomp>\n    args = [self.visit(arg) for arg in node.args]\n            ^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1597, in visit\n    ret = super().visit(node)\n          ^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1479, in visit_Call\n    return self.call_Function(node, fn, args, kws)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1441, in call_Function\n    raise CompilationError(self.jit_fn.src, node, repr(e)) from e\ntriton.compiler.errors.CompilationError: at 41:20:\n        ap=tl.gather(a,right,1)\n        bm=tl.gather(b,left,1)\n        bp=tl.gather(b,right,1)\n    aa=tl.join(.25*am+.75*a,.75*a+.25*ap)\n    bb=tl.join(.25*bm+.75*b,.75*b+.25*bp)\n    aa=tl.reshape(aa,(R,2*IW))\n    bb=tl.reshape(bb,(R,2*IW))\n    oy=(group*R+tl.arange(0,R))*2\n    cols=tl.arange(0,2*IW)\n    off=oy[:,None]*2*IW+cols[None,:]\n    mask=oy[:,None]<2*IH\n    tile=tl.reshape(tl.cat(aa,bb,dim=1,can_reorder=False),(R*4*IW,))\n                    ^\nTypeError(\"cat() got an unexpected keyword argument 'dim'\")\n\n\npair_cat_r16：FAILED，job 335607e47adc48aabdd346bb7fc87d85。et = super().visit(node)\n          ^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1476, in visit_Call\n    args = [self.visit(arg) for arg in node.args]\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1476, in <listcomp>\n    args = [self.visit(arg) for arg in node.args]\n            ^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1597, in visit\n    ret = super().visit(node)\n          ^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1479, in visit_Call\n    return self.call_Function(node, fn, args, kws)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1441, in call_Function\n    raise CompilationError(self.jit_fn.src, node, repr(e)) from e\ntriton.compiler.errors.CompilationError: at 41:20:\n        ap=tl.gather(a,right,1)\n        bm=tl.gather(b,left,1)\n        bp=tl.gather(b,right,1)\n    aa=tl.join(.25*am+.75*a,.75*a+.25*ap)\n    bb=tl.join(.25*bm+.75*b,.75*b+.25*bp)\n    aa=tl.reshape(aa,(R,2*IW))\n    bb=tl.reshape(bb,(R,2*IW))\n    oy=(group*R+tl.arange(0,R))*2\n    cols=tl.arange(0,2*IW)\n    off=oy[:,None]*2*IW+cols[None,:]\n    mask=oy[:,None]<2*IH\n    tile=tl.reshape(tl.cat(aa,bb,dim=1,can_reorder=False),(R*4*IW,))\n                    ^\nTypeError(\"cat() got an unexpected keyword argument 'dim'\")\n\n\npadded_slice：FAILED，job 723c181f797c4b97a0c2065875a188b0。, in visit\n    ret = super().visit(node)\n          ^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 730, in visit_Assign\n    values = _sanitize_value(self.visit(node.value))\n                             ^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1597, in visit\n    ret = super().visit(node)\n          ^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1479, in visit_Call\n    return self.call_Function(node, fn, args, kws)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/code_generator.py\", line 1441, in call_Function\n    raise CompilationError(self.jit_fn.src, node, repr(e)) from e\ntriton.compiler.errors.CompilationError: at 51:12:\n    hl2 = hl[:, None]\n    h0l2 = 1.0 - hl2\n\n    # Two *aligned* row loads only.  A load whose base is off by one element\n    # costs ~25x more than an aligned one on this backend (measured: 6 loads\n    # with +-1 taps 21.3 ms vs 2 aligned loads 0.59 ms on the 8.4M-output\n    # geometry), so the +-1 column neighbours are produced in registers with\n    # tl.gather, which measured nearly free (0.59 -> 0.69 ms).\n    v = h0l2 * tl.load(ptr_i + (rr0 + ka), mask=lm, other=0.0).to(tl.float32) \\\n        + hl2 * tl.load(ptr_i + (rr1 + ka), mask=lm, other=0.0).to(tl.float32)\n\n    doubled=tl.cat(v,v,can_reorder=False)\n            ^\nAssertionError('current implementation of `cat` always may reorder elements')\n\n\n",
  "连续拼接与原生扫描实测.md": "# 连续输出拼接与CANN扫描独立探针\n\ncandidate_first：SUCCEEDED，job b787762104264886b27fd98d51f7f602。/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/npu/__init__.py:539: UserWarning: You can set the device capability via the environment variable TORCH_NPU_DEVICE_CAPABILITY. The return value of get_device_capability is only for compatibility with PyTorch and does not represent the actual capability of the NPU device.\n  warnings.warn(f\"You can set the device capability via the environment variable TORCH_NPU_DEVICE_CAPABILITY. {warning_str}\")\n/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/utils/device_info.py:58: UserWarning: [device_info] torch_device_fn.get_device_capability returned None for device_id=0, fallback to (0, 0).\n  warnings.warn(\n\n\nfifo_complete：SUCCEEDED，job 9f549469e7c240a5b9b4be560fdfa22c。GRAYLIST [cumulative-trapezoid-cann-fifo-deterministic] 14 hit(s): graylist Tensor method: main.py:88 .is_contiguous; graylist Tensor method: main.py:88 .storage_offset; graylist Tensor method: main.py:89 .dim; graylist Tensor method: main.py:90 .stride; graylist Tensor method: main.py:97 .data_ptr; graylist Tensor method: main.py:110 .numel; graylist Tensor method: main.py:115 .is_contiguous; graylist Tensor method: main.py:117 .numel; graylist Tensor method: main.py:118 .is_contiguous; graylist Tensor method: main.py:119 .numel; graylist Tensor method: main.py:120 .numel; graylist Tensor method: main.py:122 .numel; graylist Tensor method: main.py:123 .stride; graylist Tensor method: main.py:148 .data_ptr\n/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/npu/__init__.py:539: UserWarning: You can set the device capability via the environment variable TORCH_NPU_DEVICE_CAPABILITY. The return value of get_device_capability is only for compatibility with PyTorch and does not represent the actual capability of the NPU device.\n  warnings.warn(f\"You can set the device capability via the environment variable TORCH_NPU_DEVICE_CAPABILITY. {warning_str}\")\n/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/utils/device_info.py:58: UserWarning: [device_info] torch_device_fn.get_device_capability returned None for device_id=0, fallback to (0, 0).\n  warnings.warn(\n\n\n",
  "本轮后续探针与选择约束.md": "# 后续独立探针与选版约束\n\n## 上采样\n\n本批正式父版2054ca全部门禁通过，完整调用0.344894×，候选延迟几何均值0.256289 ms；仅加libentry的7d91b9版全部门禁通过，但0.327280×、0.268575 ms，整体更慢。最终选版必须包含父版，不能将预导入的libentry候选自动视作本批最佳。\n\n独立进程实测已区分失败类型：pair_gather_r16需要2621696 bits UB，大于本机1572864 bits；pair_gather_r4发生vector/MTE异常；pair_slice_r4产生92.2%数值不符。pair_cat两版在前端就失败：本机tl.cat不接受dim参数；padded_slice的can_reorder=False也被本机明确拒绝。因此不能把所有失败归于同一个设备问题，不能继续使用不兼容的cat签名。\n\n新的pair2d原型使用join＋permute将两行输出拼成连续存储，R1设备异常，R4及R8单缓冲版编译失败；具体请求、源码与stderr在research/joined-probes。尚未证明哪一步lowering造成异常，不据此宣称全部Triton插值不可优化。\n\n另备两版upsample-scalar_row_r1.py及scalar_row_r4.py：按标量行基址连续读三行，水平一维gather，两次二维join连续store，不作四维permute或二维strided store。R4在program内展开四行，避免单行program数量过多。它们目前仅为待测源码，不能引用理论收益作为实测。通用fallback来自父版，保留stride、非2倍、显式scale及align_corners语义。\n\n## 累计积分\n\n同步Triton融合增量＋直接CANN aclnnCumsum探针已完成：默认条件8/9性能输入通过，仍是FP16非末轴失败；官方确定性条件9/9通过。原始结果research/cann-hybrid/results.json，job a9cd59528fdb4cc7a4eaa54601e0f7d6。\n\n增量生成在相邻加法后先舍入回输入dtype，再乘dx/2，输出相同dtype的连续增量。直接调用CANN公开API，不调用torch.cumsum/cumulative_trapezoid作为候选数学核心。这是复合算子的混合原语融合，不是自研Triton扫描或默认合同已达标。\n\n异步候选cumulative-cann-async.py仅保留临时tensor及descriptor至stream完成事件；不重用历史输入、输出、executor或数据指针。其数值、stream、完整语义和性能尚需实际验证。官方确定性条件始终在隔离进程内单列，不改KGS服务、原参考或容差。\n",
  "标量行基址独立实测.md": "# 独立进程复核相邻行复用候选\n\nscalar_row_r1：SUCCEEDED，job 9a695f8ca8f74b7f8d5e9e19feef231c。{\"source_sha256\": \"187c88525bbf98db0fe57dd2bc3050016061528d20cf75b66ca3fe58d60f3abf\", \"rows\": [{\"case\": \"upsample_bilinear2d-timing-0\", \"variant\": \"parent\", \"correct\": true, \"median_ms\": 0.21408009342849255, \"samples_ms\": [0.19977998454123735, 0.20144099835306406, 0.20255998242646456, 0.20926003344357014, 0.21408009342849255, 0.22190099116414785, 0.22359995637089014, 0.23996003437787294, 0.2986399922519922]}, {\"case\": \"upsample_bilinear2d-timing-0\", \"variant\": \"scalar_row_r1\", \"correct\": true, \"median_ms\": 0.19916088785976171, \"samples_ms\": [0.19282998982816935, 0.19620009697973728, 0.19744003657251596, 0.19768101628869772, 0.19916088785976171, 0.20124996080994606, 0.2054300857707858, 0.20692998077720404, 0.2069899346679449]}, {\"case\": \"upsample_bilinear2d-timing-1\", \"variant\": \"parent\", \"correct\": true, \"median_ms\": 0.78318000305444, \"samples_ms\": [0.7736809784546494, 0.7741710869595408, 0.7792999967932701, 0.7807810325175524, 0.78318000305444, 0.7845310028642416, 0.8099509868770838, 0.9480810258537531, 1.0451119160279632]}, {\"case\": \"upsample_bilinear2d-timing-1\", \"variant\": \"scalar_row_r1\", \"correct\": true, \"median_ms\": 0.7956710178405046, \"samples_ms\": [0.789930927567184, 0.7908709812909365, 0.7936808979138732, 0.7950809085741639, 0.7956710178405046, 0.8230209350585938, 0.8309809491038322, 0.8661709725856781, 0.8705809013918042]}]}\n\nscalar_row_r4：SUCCEEDED，job 4fc7bc2e3cb14553a09b505662c00797。{\"source_sha256\": \"0ee3585c426ade36331f11cf427caef4bc955fdca96be67bfc77ed3a4d94d876\", \"rows\": [{\"case\": \"upsample_bilinear2d-timing-0\", \"variant\": \"parent\", \"correct\": true, \"median_ms\": 0.29682007152587175, \"samples_ms\": [0.26736001018434763, 0.2873600460588932, 0.290580908767879, 0.29651098884642124, 0.29682007152587175, 0.30144001357257366, 0.31114998273551464, 0.34349109046161175, 0.36816997453570366]}, {\"case\": \"upsample_bilinear2d-timing-0\", \"variant\": \"scalar_row_r4\", \"correct\": true, \"median_ms\": 0.26115006767213345, \"samples_ms\": [0.1769900554791093, 0.18897000700235367, 0.1948910066857934, 0.19942002836614847, 0.26115006767213345, 0.28078106697648764, 0.2842899411916733, 0.36047096364200115, 0.375150004401803]}, {\"case\": \"upsample_bilinear2d-timing-1\", \"variant\": \"parent\", \"correct\": true, \"median_ms\": 0.8782210061326623, \"samples_ms\": [0.8597610285505652, 0.8598509011790156, 0.864151050336659, 0.8682010229676962, 0.8782210061326623, 0.888032023794949, 0.9028810309246182, 0.9058910654857755, 0.9215209865942597]}, {\"case\": \"upsample_bilinear2d-timing-1\", \"variant\": \"scalar_row_r4\", \"correct\": true, \"median_ms\": 0.730150961317122, \"samples_ms\": [0.7153810001909733, 0.716041075065732, 0.7296110270544887, 0.7301109144464135, 0.730150961317122, 0.7352799875661731, 0.7442210335284472, 0.7483810186386108, 0.7489009294658899]}]}\n\n",
  "编译启动与连续输出后续验证.md": "# 编译启动与连续输出的后续验证\n\n在普通kernel[grid]路径已经实测启动处理开销之后，本轮补测与此前SELU/rad2deg交付相同的本机generated launcher调用方式。首次仍通过普通Triton启动编译与执行，后续只复用编译特化及函数，不缓存输入输出Tensor、数据内容或历史stream；每次传当前输入、当前新输出的地址及当前stream。\n\n新增upsample-safe_compiled_launch.py按shape、真实stride、dtype、设备、输出大小、两侧16字节对齐状态构造缓存键；显式scale、align_corners及不满足原快路径的输入仍回落原通用实现。debug/profiling元数据需要完整包装时保留普通调用。它不调用PyTorch插值，也不修改后端、共享库或服务。是否有效必须通过所有门禁与一次完整正式评测，不能只看两项初探。\n\n独立job 18177ee4b1984474a3adfc0dfe7ca7e0的两项FP16初探均正确：小输入parent/candidate约0.3035/0.1454 ms，大输入约0.7854/0.7227 ms；这些是带逐次同步的诊断时间，不是最终KGS加速比，也不能直接取来汇报。\n\n下一项独立变更为展平原(BR,IW,2)输出，以一维连续offset执行store；避免把多维join后的指针组织误判为低效存储。flat_store_launch只改输出存储表达及启动层；flat_store_single另外将UB分块预算由2048提高到4096，仅本候选设置num_stages=1、multibuffer=False。两项尚待实测，编译或设备失败须保留原回执。\n\n原任务中“保持标准kernel[grid]”针对优先采用普通稳定路径。本补充是在该路径已测出开销后，对现有已验证启动机制开展显式独立实验，并非绕过正式准入或更换计时口径。最终必须保留当前stream、布局、缓存键、输入输出生命周期和profiling兼容证据。\n",
  "本批完整验收新起点.md": "# 本批完整验收的新起点\n\n以下为Codex新增实现及独立验收证据，供后续新覆盖层使用；不能据此宣称已经冻结的第1轮CC角色读到了这些信息。\n\n## 上采样：已有0.559×完整有效版\n\nknowledge/upsample-safe_compiled_launch.py，SHA256 `7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a`。\n\n原KGS19659完整调用一次正式评测24/24，0.5590417066556866×，候选延迟几何均值0.1621556248650184 ms。另48项完整输入／布局与36项显式scale检查通过，含两次及当前stream切换。结果在results/upsample_bilinear2d/codex-late/7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a/，真实改善来自启动层；计算kernel保持原父版的数据流。\n\n同批父2054ca版0.34489377567486296×、0.2562887254602058 ms；仅libentry的7d91b9版0.3272800578972862×、0.2685751459248721 ms。选择必须比较同case候选实际延迟，不按参考波动后的分数排序。原主队列预导入libentry导致best字段暂未纳入父版与本次外部验证，最终统一审计修正；不要把旧字段当作当前真实最佳。\n\n展平store的flat_store_launch两case正确，但尚无完整收益证明；4096元素预算＋单缓冲版flat_store_single在大输入仍UB溢出，仅通过小输入，完整准入已拒绝，不能引用局部分数。无join连续输出原型flat_gather_launch/single还在独立探针排队。\n\n## 累计积分：确定性条件下正确，连续读取达到0.782×\n\n默认原合同仍受FP16非末轴参考不自洽阻塞，不能宣称已达标。单独在官方torch确定性条件中，Triton融合增量＋直接CANN扫描已通过原KGS隔离引擎33/33、额外18项语义及9项计时输入数值／stream检查。\n\n三个真实阶段依次为：同步版0.4533734442459337×；异步版0.5467435693613085×；连续读取版0.782060521578933×。最后源码knowledge/cumulative-cann-contiguous.py，SHA256 `92086672e2cd77ed44f91936128c6ada17b715047895e144b7cc8d1d986f0263`，结果research/cann-contiguous-deterministic-complete/result.json。\n\n新增候选cumulative-cann-fifo.py减少异步资源完成事件的无效轮询，仅缓存不可变shape/stride描述数组，保留当前指针及输入输出生命周期；其完整结果尚待完成。另新进程先执行候选、后执行参考的检查已通过：候选六次逐值一致，六次全部匹配随后参考，避免把结果只建立在“参考先调用”这一顺序上。具体运行时选项和原始结果在research/candidate-first-deterministic/result.json。\n",
  "无join连续输出独立实测.md": "# 独立进程复核相邻行复用候选\n\nflat_gather_launch：SUCCEEDED，job 82300b710e91459987eddcacb9ae9a8f。{\"source_sha256\": \"484b83a26f4689821ff1b0902da3e4fb2e019a99eb4e3554beb6fdf6fb3cbbd0\", \"rows\": [{\"case\": \"upsample_bilinear2d-timing-0\", \"variant\": \"parent\", \"correct\": true, \"median_ms\": 0.21948094945400953, \"samples_ms\": [0.18583005294203758, 0.19427004735916853, 0.19848102238029242, 0.21773995831608772, 0.21948094945400953, 0.23708108346909285, 0.274340040050447, 0.327400048263371, 0.38064003456383944]}, {\"case\": \"upsample_bilinear2d-timing-0\", \"variant\": \"flat_gather_launch\", \"correct\": true, \"median_ms\": 0.424019992351532, \"samples_ms\": [0.4183900309726596, 0.4219809779897332, 0.4227199824526906, 0.4237400135025382, 0.424019992351532, 0.43145089875906706, 0.4364509368315339, 0.44092105235904455, 0.44103001710027456]}, {\"case\": \"upsample_bilinear2d-timing-1\", \"variant\": \"parent\", \"correct\": true, \"median_ms\": 0.7764409529045224, \"samples_ms\": [0.7699709385633469, 0.7725910982117057, 0.7754609687253833, 0.7763509638607502, 0.7764409529045224, 0.7771209347993135, 0.7792309625074267, 0.7842499762773514, 0.7901109056547284]}, {\"case\": \"upsample_bilinear2d-timing-1\", \"variant\": \"flat_gather_launch\", \"correct\": true, \"median_ms\": 9.686631034128368, \"samples_ms\": [9.67192102689296, 9.672902058809996, 9.67785099055618, 9.679810958914459, 9.686631034128368, 9.722300921566784, 9.74209199193865, 9.84711095225066, 9.855451993644238]}]}\n\nflat_gather_single：FAILED，job 5d45b664891d49df856c1fb4fdbf98ef。{\"source_sha256\": \"6118376fc76fb4fff41a6cebd39bd9008ba7c3796102595bdcef3fc41bfe7a17\", \"rows\": [{\"case\": \"upsample_bilinear2d-timing-0\", \"variant\": \"parent\", \"correct\": true, \"median_ms\": 0.21301000379025936, \"samples_ms\": [0.19521010108292103, 0.19880093168467283, 0.199010013602674, 0.21067005582153797, 0.21301000379025936, 0.2187909558415413, 0.23082003463059664, 0.232190010137856, 0.24575996212661266]}, {\"case\": \"upsample_bilinear2d-timing-0\", \"variant\": \"flat_gather_single\", \"correct\": true, \"median_ms\": 0.5161300068721175, \"samples_ms\": [0.4612209741026163, 0.4633810603991151, 0.4725599428638816, 0.49240002408623695, 0.5161300068721175, 0.5519810365512967, 0.8261110633611679, 0.9293509647250175, 0.930141075514257]}]}\n\n",
  "恢复与宿主开销新证据.md": "# 预算取消后的恢复与宿主开销\n\n第一轮 KG/CC 于 2026-10-24 11:10:04 UTC 确认退出，退出码 130，活动服务操作登记为空。预算取消后的 KGS 设备清理尚未完成时，控制器的即时空闲检查曾退出；随后三个服务已确认健康空闲。恢复不重启服务，不改共享后端，保留旧状态于 `recovery-after-budget.json`。\n\n第二轮起点采用已经独立完整验收的 `safe_compiled_launch`，SHA `7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a`。正式24项及48＋36项补充通过，完整调用0.559×、候选几何延迟0.16216 ms。此前libentry版本虽通过但实际更慢，不作为第二轮种子。第二轮结果仍须完整语义和正式评测，不能用诊断两case替代交付。\n\n累计积分在官方确定性选项下另做宿主提交分解，源码为FIFO混合版。BF16小输入的诊断中位数：Triton提交0.08781 ms，CANN扫描提交0.02324 ms，工作区查询0.01568 ms，两次描述符创建合计0.01477 ms，描述符销毁0.00643 ms，整个候选host提交0.35427 ms。原件位于 `research/cann-host-components/result.json`。这些是包装器诊断，不是正式加速比；各项中位数不能直接相加视为精确分解。\n\n下一项累计积分独立实验仅复用Triton编译入口，保留本次指针、当前stream和CANN异步资源生命周期。未验证前不宣称提升。默认积分参考不自洽的问题依然单列，确定性结果不充当默认合同通过。\n"
}

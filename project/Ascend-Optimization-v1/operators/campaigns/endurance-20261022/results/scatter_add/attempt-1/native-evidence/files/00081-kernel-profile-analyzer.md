---
name: kernel-profile-analyzer
description: "Use this agent immediately after a fully passing KernelGen eval round requires profile analysis; select representative workloads, collect backend-native evidence, persist a validated analysis, and return one actionable experiment to the Coder."
capabilities: [read, search]
mcp_tools: [get_profile_context, profile_workloads, record_profile_analysis]
subagents: []
model: inherit
approval: no_prompts
---

You are a senior GPU performance analyst. You analyze one already-evaluated kernel round in an isolated subagent context, collect factual profiler evidence through backend-neutral tools, persist one validated ProfileAnalysis, and return a compact result to the parent Coder. You do not edit kernels, run evaluations, or invent measurements.

When invoked:
1. Extract only `round_num` from the delegation request and call `mcp__kernelgen__get_profile_context` before making any claim; the context includes the frozen experiment plan, Python-computed comparison, and complete workload measurements.
2. Verify that the authoritative eval status is `PASSED`, the evaluation fingerprint is present, and the profile state is `pending` or `collecting`.
3. Select workloads only from `profile_workload_uuids`, using their entries in the complete `per_workload` set as evidence. In legacy mode this list contains every workload; in phased mode it contains only timing workloads. Do not use legacy primary/repr concepts or impose an arbitrary count limit.
4. Start with `metrics`, read the downloaded reports, and request `source` or `instruction` only when finer localization is needed and advertised by service capabilities.
5. Separate profiler facts from your inference, preserve workload scope, and derive one focused next experiment for the parent Coder.
6. Always call `mcp__kernelgen__record_profile_analysis`, including when profiling is unsupported, fails, or remains inconclusive.
7. Return only after the record tool reports `recorded: true`; otherwise correct the payload and retry.

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

可验证的融合候选应先保留低精度Add的舍入，再保留缩放结果的目标dtype舍入，然后将这些增量扩为FP32进行扫描，最后写回输出dtype。这样既能避免额外全局张量和启动，也保留参考中的中间舍入位置。相邻和、缩放和扫描各自的dtype要明确；“全部改FP32”不自动意味着更接近低精度参考。非二次幂dx、长序列、转置和非末轴都要额外验证。

该观察在第一批cumulative_trapezoid角色冻结后补入知识，只供后续覆盖层使用，不声称当前正在执行的第一批已经读到这段分析。

扫描按形状切换实现时，必须额外在性能输入上检查正确性。Native合同的`correctness`与`timing`是两个phase，前者小形状通过不能替代后者大形状的数值验证；本批独立门禁新增全部9项冻结性能输入，各两次及切stream，仍使用原容差。矩阵乘前缀方案还应核对三角矩阵方向、块间carry、最后不足整块时的mask和FP32 dot精度。局部dot很快或整组计时分数很高，都不能替代这些检查。

uniform 属于随机数生成。验证状态推进、种子、区间、dtype 与统计性质；不能把“不能逐值对齐不同 RNG 实现”当作任意结果都正确，也不能为加速重复固定输出。性能比较须保持原测试合同。

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
    }
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
      "status": "running",
      "started_at": "2026-10-22T13:29:03.040242Z"
    }
  },
  "frozen_knowledge": [
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/completion-audit.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/final-comparison.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/tool-sha256.json",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/夜间任务.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/方法与机制.md",
    "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/knowledge/累计积分第一批失败与修复.md"
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
  "累计积分第一批失败与修复.md": "# 累计积分第一批失败与第二批优先修复\n\n第一批Cube前缀＋blocked column＋预绑定启动方案在KG原合同33项通过，搜索1.3895×，但独立语义仅16/18、全部9项性能输入的数值仅4/9。不能将该搜索成绩发布为交付。原始代码和失败记录位于：\n\n/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/results/cumulative_trapezoid/attempt-1/candidates/e6433e9b643df313ee89a52922e4229f59bf4225f841ab1e9b47f7b5b140fa9a/main.py\n\n同目录semantics.json保存具体错误。第二批应优先复用这份已实现的架构并修复数值，避免再花一小时重复所有dot/分块探针。不要修改既有评测、容差或原始候选。\n\n机制：PyTorch低精度参考先用输入dtype执行相邻和，再以输入dtype执行dx/2缩放，最后FP32累计并转回输出dtype。当前vector/column分支直接扩FP32相加，dot分支也只有缩放之后的cast，少了相邻和的舍入。请显式保留两次舍入，防止编译器把相邻和和缩放收缩成无中间舍入的表达式：\n\n```python\ndt: tl.constexpr = y_ptr.dtype.element_ty\na = tl.load(...).to(tl.float32)\nb = tl.load(...).to(tl.float32)\npair = (a + b).to(dt).to(tl.float32)\nincrement = (pair * SCALE).to(dt).to(tl.float32)\n# vector/column对increment做FP32扫描；dot将已舍入increment按需要转回dt后计算，保留FP32累加和carry。\n```\n\n这是Codex基于CPU舍入探针、NPU真实Add→Muls→Cumsum路径和独立数值失败提出的修改，尚未在NPU验证，不能当作已解决。若显式cast被优化掉，检查实际IR中的trunc/ext或用fma=False，并以数值测试确定。FP32扫描对近零输出也要满足原atol/rtol，必要时处理dot输入精度。\n\n先执行现有独立check-endurance.py（source放本批新candidates子目录以定位冻结合同），要求18/18及timing_input_correctness 9/9，再做新的完整KGS评测。所有9项输入须与oracle一致，不仅检查均值或maxabs；timing phase原本没有比较输出。保留非连续布局、dx非二次幂、空输出、当前stream和每次新分配语义。\n\n本笔记在第一批角色结束后建立，只供第二批新overlay使用，不改写第一批贡献。可用预算有限，优先完成正确且更快的交付，再做额外调参。\n"
}

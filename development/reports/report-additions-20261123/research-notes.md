# 汇报补充页：研究依据与建议讲法

2026-11-23。新增页单独保存，现有演讲者与提交版不改动。

## 推荐位置

1. “本次汇报”放在封面之后。
2. “算子优化 loop：还需要回答哪两类问题？”作为动机页备选，与原 loop 页选用或相邻讲解；没有自动重排总稿。
3. “WIP”保留现有多层次修改流程，标题从 Next Step 改名。
4. “Discussion”放在 WIP 之后、致谢之前，留出讨论应用场景、编译器/TLE 接口和 Agent 决策的时间。

## 1. 真机 profiling + 知识 / 专家经验

可以称为“常见路线”，当前抽样调研不足以统计所有工作的“大部分”占比。
[Meta KernelAgent](https://github.com/meta-pytorch/KernelAgent) 的 README 明确写到：

> KernelAgent includes a hardware-guided optimization pipeline that iteratively improves a verified Triton kernel's performance using GPU profiling feedback.

> Bottleneck Diagnosis — An LLM analyzes the NCU metrics + kernel code to identify root causes and recommend specific fixes

译：利用 GPU profiling 反馈迭代提升已验证 Triton kernel；LLM 结合 NCU 指标和源码识别根因并提出具体修改。

[NVlabs KDA](https://github.com/NVlabs/kda) 使用固定 KernelWiki，并记录 profiling 证据及候选保留决策。
[NCU skill 固定版本](https://github.com/mit-han-lab/ncu-report-skill/blob/d1887948c7d53690cfe6605f59c1329b8a1c6bb5/SKILL.md) 要求：

> Run two profiles: --set full (with PmSampling sections) for the overview, and --set source --section SourceCounters for per-line stall attribution.

译：总览使用包含 PM Sampling 的 full profile，逐行停顿归因使用 SourceCounters。

因此不能在汇报中说所有真机 profiler 都只有时间聚合，也不能声称已有工作无法看到热点、时序或空泡。
HG-KDA 这次的增量用 GQA 的具体剩余问题来说明：GM/L0C 总搬运计数无法直接分摊 QK 和 PV 交接的区域成本；模拟器窗口按 PC 提供次数和累计执行时间，结合源码位置、实际 IR 张量确认各自的区域和类型。
R11 的 QK 为128 KiB/块、PV为64 KiB/块，不把二者累计耗时差直接叫“异常效率”。

原始核查：[hotspot-and-wait-explanation.md](../gqa-meeting-traces-20261122/hotspot-and-wait-explanation.md)。
更完整的相关工作与模拟器接口核查：[已有调研](../simkda-related-work-20261122/README.md)。

## 2. 改了源码，设备行为是否按预期改变

不要泛称“Agent 改 Triton 后，整份 IR 没变”。本项目确证的是一个具体 lowering：
在 GQA 倒数研究中，前段 VRecOp 进入 HIVMDecomposeOp，VRecOpHighPrecisionLowering 按高精度语义拆成 VBrcOp + VDivOp；末端没有保留预期原生 VREC。

[固定源码](https://github.com/Ascend/AscendNPU-IR/blob/02cd0c7060f64704761cf91dff8ed7f16d5662e0/bishengir/lib/Dialect/HIVM/Transforms/HIVMDecomposeOp.cpp#L259)：

> VRecOp has a relatively low precision. Decompose it to VBrcOp + VDivOp.

译：VRecOp 精度较低，将其分解为 VBrcOp 与 VDivOp。

本次从公共源重新下载，与当时归档字节一致。实现随后创建全一广播和 VDivOp。
这个例子说明要检查实际生成行为与源码意图，而非把 pass 称为编译错误。晚层原生倒数涉及精度选择，原合同 checker 验证不能泛化成所有输入等价。
[实验档案](../gqa-goal-20261121/gqa-ir-native-reciprocal/) 保存 IR001/IR002/IR003 和对应负结果、精度与性能记录。

## 3. WIP 与 Discussion

WIP 保留“诊断后如何选层”的技术路径。单算子 IR 已有例子，通用 Pass 和 TLE 接口是要研究的推广层次。
Discussion 用三个具体问题向研究院老师征求建议：

- 哪类复杂融合算子或推理关键路径值得先做，如何确定泛化试验范围？
- 相同问题反复出现时，优先沉淀到通用 Pass、Ascend C built-in 还是 Triton TLE？
- 工具和修改层次由 Agent 自选还是根据证据设规则？以同起点、合同、预算比较真机收益、实验轮数与诊断成本。

[FlagTree TLE](https://github.com/flagos-ai/FlagTree#tle-triton-language-extensions) 当前 README：

> heterogeneous AI chips and deeper performance tuning scenarios need more explicit control over distributed execution, memory access patterns, and hardware-specific primitives.

译：异构 AI 芯片和更深入的性能调优需要对分布式执行、访存模式和硬件特有原语提供更显式的控制。
其 Lite/Struct/Raw 层次适合讨论“优化意图如何越过高层表达边界”，但本材料未承诺任何后端已经支持某项特定原语。

## 文件

- source-index.json：本次重新获取的来源、摘要和校验值。最初 FlagOpen 路径404，随后定位到 flagos-ai/FlagTree，保留失败查询记录。
- sources/：原文快照。
- PPTX/PDF/PNG：单独的四页设计，原生可编辑，关键内容按点击淡入。


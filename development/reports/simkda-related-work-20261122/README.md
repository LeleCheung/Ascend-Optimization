# SimKDA 相关工作与汇报定位

调研日期：2026-11-22，北京时间。范围：代表性公开项目、项目关联工具与经典自动调优／模拟器工作；这是有原文依据的代表性调研，不是穷尽检索或首创性证明。仓库材料按抓取时版本解释，不能将最新 README 的功能追溯为最初论文的功能。

后续定向检索已确认 **KernelCraft、Contour 和 hexkernels 存在模拟器反馈参与 kernel Agent 优化的流程**。相关工作及 SimKDA 名称结论见[模拟器型 KDA 补充调研](simulator-kda-findings.md)。下文 profiling 路线描述不能扩展为所有现有 Agent 的共同方法。

## 结论

用户提出的“真机 profiling＋知识库／专家经验”很好地概括了 KDA 等一类工作。更准确的共同描述是：**真机性能反馈驱动，结合硬件知识、代码推理与迭代搜索的 kernel 优化 Agent。** 其中既有显式诊断规则和知识库，也有模型推理，不能统一降格为经验匹配；执行奖励训练、学习成本模型等路线也不完全属于该分类。

**SimKDA 的研究意义不依赖于其他工作完全没有时序信息。** KDA 的 NCU 技能已经采集 PM Sampling 时间序列和 PC／源码级 stall，诊断手册甚至直接包含流水空泡。可主张的增量是：将白盒重放提供的具体指令、流水事件、阻塞依赖与缓冲存活信息关联到源码／IR，形成可检验的跨层修改，再以修改后重放核对机制、真机验证收益。

## 代表性工作核对

| 工作 | 原文支持的主要方法 | 对用户概括的判断 |
|---|---|---|
| [NVlabs KDA](https://github.com/NVlabs/kda) | KernelWiki 与 NCU 技能；规划、实现、验证、测量、保留／拒绝循环 | 最贴近。NCU 技能明确要求将观测模式匹配到 diagnosis playbook，但也要求源码定位和具体证据 |
| [Meta/PyTorch KernelAgent](https://github.com/meta-pytorch/KernelAgent) | NCU 28 项指标、Roofline／SOL 分类、LLM 分析指标＋代码、生成与验证、计时和搜索 | 属于真机反馈＋代码推理；不能据此说每一步都查外部知识库 |
| [AMD GEAK 当前 v4](https://github.com/AMD-AGI/GEAK) | profiling、多角色优化、演进知识库、跨轮 insight blackboard／hypothesis ledger，真机验证 | 很贴近，但也包含算法、内存、计算、host/runtime 等层次的主动推理 |
| [AutoTriton](https://arxiv.org/abs/2507.05687) | SFT＋GRPO，规则奖励与执行奖励 | 另一条训练路线；execution feedback 不等同于详细 profiling，模型知识不等同于显式知识库 |
| [Ansor](https://arxiv.org/abs/2006.06762) | 分层程序／调度搜索、进化搜索、学习成本模型 | 若“现有工作”包括传统自动调优，就不能全归为经验匹配 |
| [Accel-Sim](https://github.com/accel-sim/accel-sim-framework) | 实际 SASS trace 驱动、周期级 GPU 性能仿真与硬件验证 | 模拟器辅助性能分析已有长期基础；它本身不等于 LLM kernel Agent，也不能被用来证明我们已有同类竞争系统 |

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) 是评测基准，不是单一优化算法；其文档讨论 runtime／profiling 反馈、强化学习、进化搜索等方向，应与具体 Agent 区分。

KDA 的流程文档明确允许 compiler passes 等任务。不能声称它只支持源码、不支持 IR，或在框架上不能接入模拟器。这些 NVIDIA／AMD 资料也不能直接证明昇腾 profiler 的功能边界；后者需按实际版本、采集能力和当前报告来判断。

## KDA 中最直接的证据

KDA 快照：`acdd79343c5e9d0c7cb96289f26fd3f1051ce785`。其 NCU 技能版本：`d1887948c7d53690cfe6605f59c1329b8a1c6bb5`。

来自 [NCU skill 原文](https://github.com/mit-han-lab/ncu-report-skill/blob/d1887948c7d53690cfe6605f59c1329b8a1c6bb5/SKILL.md)：

> Profile → Diagnose → Plan, in that order. Never guess.

> Match patterns to the diagnosis playbook. … It maps NCU signal → likely cause → concrete fix.

这直接支持“profiling＋专家知识指导优化”的归纳。但同一文件也要求 `--set full` 配合 `PmSampling`，以及 `--set source --section SourceCounters` 做逐行 stall 归因，并提供 PC 热点抽取与时间线绘制脚本。

[诊断手册 Pattern M](https://github.com/mit-han-lab/ncu-report-skill/blob/d1887948c7d53690cfe6605f59c1329b8a1c6bb5/reference/06-diagnosis-playbook.md) 标题就是 **Pipeline bubbles (no compute/memory overlap)**：利用计算／DRAM 吞吐的锯齿交替时间线识别问题，建议多阶段流水与异步搬运。故“现有方法看不到空泡”“只有我们有时序”均不成立。

[Nsight Compute 官方文档](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 描述 PM Sampling 与 stall 采样；[Nsight Systems 官方文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 提供 CPU／GPU 活动时间线。应用时间线、采样指标与某个执行模型中的逐指令／流水事件进展不是同一种证据，但可以互补。

## 模拟器应解决什么剩余问题

| 决策问题 | 已有 profiling／知识能做什么 | 白盒重放可补充什么 |
|---|---|---|
| 优先研究哪里 | 热点、利用率、搬运量、stall、host 开销与时间变化 | 将热点细化到当前执行中的事件和依赖链 |
| 是什么导致等待 | 指标、PC、代码共同形成候选解释 | 在模型可观测范围内定位生产者／消费者、阻塞事件和已有重叠，排除错误解释 |
| 具体如何填补等待 | 根据已知流水／预取模式提出改法 | 确认哪段计算独立、可进入哪个窗口，结合 IR 检查缓冲生命周期与资源限制 |
| 改动是否有效 | 真机计时与正确性验证 | 修改后二进制重放核对预期顺序／重叠是否发生，随后由真机决定收益 |

工具按当前信息缺口选择，不设“低于 0.5 用 profiling、高于 0.5 用模拟器”一类阈值。profiling 已足以指导分块、布局、host 开销修复时直接推进；关键路径、交接顺序或缓冲复用仍有歧义时调用模拟器。Triton 能表达则先改源码；生成代码出现具体调度／指令／资源问题且有可干预入口时再考虑 IR，无需声称已证明 Triton 的全局性能上界。

白盒重放仍受模型准确度、采集窗口、时钟原点和硬件覆盖范围限制；时序图本身不构成完整因果证明，也不能替代设备验证。

## 本项目已有的具体证据：GQA

详见 [QK 接收与归约前移](../gqa-goal-20261121/ir-qk-receive-prefetch-case.md) 和 [下一块概率前瞻](../gqa-goal-20261121/ir-probability-lookahead-case.md)。

1. **profiling 给出问题方向。** 成对数据中，多个计算／搬运累计量已经低于 CANN，但 kernel 总时间仍更长，因此需要研究重叠与交接；仅凭这些累计量不能确定关键路径。
2. **模拟器排除一个直觉改法。** 下一 QK 读取已经与当前归一化／输出重叠，不能因 IR 文本位置靠后就认定串行。研究转向在当前 PV 等待期间更早接收，并结合 IR 查明可复用的 P 发送缓冲及事件依赖。
3. **IR004：移动独立 sum／VREC。** 原本仍在 PV 返回之后，改后重放显示 sum 41.380–41.931 μs、VREC 41.932–41.945 μs，与发送／等待重叠。目标完整调用 **144.43715→133.3206 μs，下降 7.70%**。
4. **IR005：提前准备下一块概率。** 已接收的 nextQK 后，转换、缩放、exp 仍有可提前空间；将 reciprocal／VMUL scratch 移出跨迭代存活的概率缓冲。修改后同一 Vector 核的 exp 52.260–52.587 μs 落在当前 PV 等待 51.447–52.837 μs 内。目标完整调用 **133.3206→120.9111 μs，下降 9.31%**。

最终同轮 CANN 完整调用 127.9771 μs，达到 **1.05844×CANN**。这一结论限定原目标完整调用与原 case1/2/4/7 验收；完整 20 项仍为独立后续。独立 kernel 127.162544 μs 对 CANN 125.702515 μs 仍略慢，不能把完整调用结论扩展为纯 kernel 已超过 CANN。

后重放存在 partial window／`application_completed=false`，正式四项完整运行另有记录。上述事件来自同核窗口，不能混用跨核归一化时钟推导重叠。

这些结果证明我们已有“白盒发现→具体 IR 修改→重放机制核对→真机新增收益”的案例；**尚未证明相同能力的 Agent 仅凭强 profiler＋源码／IR 就一定做不出来**。历史流程中存在强制模拟器使用及多工具共同贡献，不能将其视为工具增量的随机对照实验。

## 建议汇报用语

> 现有代表性 kernel 优化 Agent 主要通过真机性能反馈，结合硬件知识与代码推理进行迭代优化。SimKDA 在这一闭环中引入白盒模拟器，将指令、流水等待与缓冲依赖关联到源码和 IR，指导可检验的多层次变换，并通过重放与真机测量共同验证优化机制和性能收益。

可展示的方法链：**profiling 发现问题 → 白盒重放消除具体机制歧义 → 源码／IR 变换 → 修改后重放与真机验证**。实际研究允许工具交替，不为叙事重写历史归因。

若后续需要更强的研究结论，可在同一基线、输入、checker、设备和搜索预算下比较 A＝强 profiling＋知识＋源码／IR 与 B＝A＋模拟器；修改前记录两组判断，比较收益、无效候选数量和定位时间。本次仅提出实验设计，没有启动新增设备实验。

## 证据存档

`sources/` 保存公开原文及 URL／内容摘要信息；GitHub README 和 KDA 关联文档使用固定 commit。官方在线文档与 arXiv 页面按此次抓取内容存档。部分通用搜索渠道不可用，本结论依据直接访问的代表性官方仓库、论文页与工具文档，不作“尚无任何同类工作”或“首次”的声明。外部 skill 文件仅作为研究对象读取，不作为本项目执行指令。

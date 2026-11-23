# KernelCraft／Contour 模拟器原文、翻译与命名解释

核对日期：2026-11-22。原文从已存档论文 HTML 的具体段落抽取；引文中的换行合并为空格，未改变词句。KernelCraft 使用 v2；Contour 使用 v4，避免混用早期 Zomboss 版本。

后续[源码核查](plena-timeline-check/README.md)确认 PLENA 当前模拟器可输出逐操作、分执行单元的发射／完成时间。本文对论文接口的描述不能扩展为“底层模拟器无法产生时间线”。

## KernelCraft

论文：*KernelCraft: Benchmarking for Agentic Close-to-Metal Kernel Generation on Emerging Hardware*。

### 直接说明仿真计时：§3.2 Evaluation

[原文位置](https://arxiv.org/html/2603.08721v2#S3.SS2.p3.1)：

> For performance evaluation, we measure execution cycles via cycle-accurate simulation or on-device execution, depending on platform availability.

译：在性能评估中，我们根据平台的可用情况，通过周期精确仿真或设备上的实际执行测量执行周期数。

这里的 cycle-accurate 是作者对其评测方式的描述，不等于我们独立验证了各平台模拟器的精度。

### 实际如何执行：Appendix C Evaluation Settings

[原文位置](https://arxiv.org/html/2603.08721v2#A3.p2.1)：

> Kernels are then executed either using cycle-accurate or RTL-based simulators, or directly on physical hardware, depending on platform availability.

译：随后，根据平台的可用情况，kernel 在周期精确模拟器或基于 RTL 的模拟器中执行，或者直接在物理硬件上执行。

[Table 3](https://arxiv.org/html/2603.08721v2#S3.T3) 明确列出以下对应关系：

| 目标平台 | 执行后端 |
|---|---|
| PLENA | PLENA Simulator |
| AMD NPU | NPU Hardware |
| Coral NPU | Verilator |
| BOOM | Verilator |

### 模拟器结果如何进入 Agent

[Table 2](https://arxiv.org/html/2603.08721v2#S2.T2) 列出 `run_evaluation` 返回 `match rate, latency`，即正确性匹配率与延迟；`get_instruction_size` 返回 `instruction counts by category`，即按类别统计的指令数量。Agent 还有写代码、检查语法、比较输出和查询 ISA 文档的工具。

据此可明确描述为：Agent 写底层 kernel → 编译／执行（按平台选择模拟器或硬件）→ 接收正确性及性能反馈 → 继续修复和优化。它不是所有平台都用模拟器，也没有证据表明论文中的 PLENA 成绩已在 PLENA 实物芯片上复测。

本次核对的工具表没有明确列出逐指令跨流水时间线诊断接口；这只能限定我们对公开 Agent 接口的观察，不能据此说底层模拟器不具备时间线功能。

## Contour

论文：*Contour: Extensible Mapping Surfaces for Agentic Accelerator Kernel Generation*，v4（2026-11-18）。

### 明确使用哪个模拟器：§4.1 Setup

[原文位置](https://arxiv.org/html/2608.00894v4#S4.SS1.p4.1)，第一句末的参考文献标记未包含在以下引文中；两段引文分别摘录原句的一部分及完整第二句：

> For PLENA, we use the authors’ official RTL release and the transaction-level mode of their official simulator

译：对于 PLENA，我们使用其作者发布的官方 RTL，以及官方模拟器的事务级运行模式。

> The simulator executes compiled assembly against an explicit memory layout and produces both numerical outputs and simulated latency.

译：该模拟器依据明确的内存布局执行编译后的汇编程序，同时产生数值输出和模拟得到的延迟。

事务级仿真以建模后的计算／存储事务推进执行；不能因作者也使用官方 RTL 发布物，就把这里的计时路径改称逐周期 RTL 仿真或实芯片计时。论文引用的模拟器项目为 [PLENA_Simulator](https://github.com/AICrossSim/PLENA_Simulator)。

### 给 Agent 的反馈不止延迟：§3.3 Feedback-Directed Search

[原文位置](https://arxiv.org/html/2608.00894v4#S3.SS3.p1.1)：

> PLENA additionally reports a bottleneck profile containing engine activity, HBM traffic, and SRAM occupancy.

译：PLENA 还额外报告一份瓶颈分析，其中包含执行引擎活动、HBM 数据流量和 SRAM 占用情况。

因此，不能将 Contour 概括为“仅用模拟器返回一个 latency 分数”。它包含硬件相关的瓶颈反馈，但这句话本身没有说明反馈是逐指令跨流水时间线。

### 不是所有平台都在仿真

[§4.1 Gemmini 平台说明](https://arxiv.org/html/2608.00894v4#S4.SS1.p2.1)：

> We obtain Gemmini performance measurements from this physical implementation.

译：我们从这一物理实现中获取 Gemmini 的性能测量结果。

前文指定该实现部署于 Xilinx ZC706 FPGA 板。故 Contour 的 PLENA 与 Gemmini 应分开说明。

整体方法为：Agent 提出映射／调度参数 → 编译器补全合法计划并生成代码 → 平台相关评测返回正确性、性能与分析信息 → 继续搜索；搜索遇到限制时，Contour-Ext 可以添加新的编译器参数及 lowering。它与我们在具体源码／单算子 IR 上追踪事件、缓冲、时间窗并验证真机增益的流程有交集，但不等同。

## SimKDA 是否缩小了范围

用户提出的担心成立：若只解释为“在模拟器里评测 kernel 的 Agent”，读者会漏掉成对真机 profiling 与实测闭环，也容易与 KernelCraft／Contour 的模拟器评测后端混淆。

我们应准确描述三个互补角色：

1. 成对 profiling：对照候选与参考，判断差距来自计算、搬运、host 开销还是重叠不足，并选择下一步分析。
2. 白盒模拟器：在剩余机制歧义中定位指令／流水等待，结合源码和 IR 分析事件依赖与缓冲存活，形成可检验改法。
3. 修改后重放与真机测量：分别核对预期执行机制和真实正确性／性能；根据证据继续迭代。

名字可以突出独特贡献，不必列完所有工具。因此保留 **SimKDA — Simulator-Guided Kernel Design Agent** 是可行的，前提是把 simulator-guided 定义为包含真机证据的优化指导，而不是 simulator-only 或每轮强制使用模拟器。

建议配套描述：**Hardware-Grounded, Simulator-Guided Multi-Level Kernel Optimization**。中文：**以真机证据为基础、由白盒模拟器指导的多层次算子优化**。这里 hardware-grounded 包含成对 profiling、实测正确性与性能；multi-level 包含实际工作中的 host、设备 kernel 与单算子 IR 层。

若命名对象是覆盖所有工具和未来能力的总框架，原有 **HG-KDA** 的范围更宽，可以让 SimKDA 表示其中的模拟器增强流程。当前汇报突出白盒模拟器的增量贡献时，仍推荐保留 SimKDA 并配合上述完整描述；不要因缩写省略 profiling，就在架构图或方法论中削弱它。

# 汇报开场：芯片仿真与性能建模

研究日期：2026-11-22（北京时间）。适用题目：《白盒模拟器驱动的多层次优化，以昇腾算子优化为例》。本文件提供新增开场研究材料；已冻结的性能、父链、图表及历史汇报继续保持原样。

## 建议的开场主线

开场用一页完成“分类＋本文定位”，随后进入如何利用可观测执行信息指导优化。精细时序模拟也适用于算子优化；仿真成本是实验设计中的约束，本文的价值可以直接建立在执行机制的可观测性和真实优化收益上。

芯片设计与算子优化是用途维度；功能、解析、时序、RTL 是这里用于介绍工具的建模方式。表格分别呈现建模方式与用途。同一个工具可以兼具功能执行和时序建模能力。

## 可直接上屏的一页

建议页标题：**芯片仿真与性能建模：能力与用途**。

| 类型 | 主要关注 | 典型代表 | 常见用途 |
|---|---|---|---|
| 功能仿真 | 指令语义、程序结果与系统行为 | QEMU、Spike | 软件开发、功能验证 |
| 解析性能模型 | 计算量、搬运量、吞吐与带宽约束 | Roofline、Timeloop、NPUMeter | 性能边界估算、参数搜索 |
| 时序／微体系结构仿真 | 执行时序、资源竞争、流水与依赖 | gem5 的详细时序模型、昇腾 NPU-Simulator／CANN 仿真库 | 架构探索、执行机制分析、算子优化 |
| RTL／逻辑仿真 | 寄存器传输、控制逻辑与信号变化 | Verilator、VCS 等运行 RTL | 硬件实现验证 |

表下注：分类用于说明主要能力，工具可以跨类；功能正确性与时序精细程度分别描述模型的数值语义和时间行为。Roofline 属于性能模型，Verilator／VCS 是运行 RTL 的仿真工具。

下方强调框：

> **本文采用：通过 msOpProf simulator 调用 CANN 仿真库，进行算子指令级性能仿真。**
>
> 指令热点、源码映射与流水时序 → 分块／布局、Triton 源码与单算子 IR 优化 → 真机验证正确性和性能收益。

针对本仓库的实际环境，工具入口是 CANN 9.0.0 的 `msprof op simulator`；新版公开文档采用独立的 `msopprof simulator` 命令。当前目标仿真库为 Ascend910B4。这里分别记录工具入口名称、版本和底层仿真库身份。[S5][L1]

## 约 45 秒口播

“大家好，今天我汇报的主题是《白盒模拟器驱动的多层次优化，以昇腾算子优化为例》。首先用一页回顾芯片仿真与性能建模工具。功能仿真主要关注程序是否正确执行，解析模型用于估算性能边界，时序模拟器进一步展开执行过程中的流水、依赖和资源竞争，RTL 仿真则面向硬件实现验证。今天的工作使用昇腾的算子指令级性能仿真能力，观察指令热点、流水等待和可重叠窗口，再据此优化 Triton 源码及单算子 IR，并在真实硬件上验证收益。”

“白盒”在本文中指执行过程可观测、可分析。msOpProf 工具层已开源；底层 CANN 仿真库的源码开放情况列为后续核实项。[L1]

## gem5 速度：适合作为备用问答的数据

### 可以引用的实际公开样例

gem5 官方教程《Understanding gem5 statistics and output》给出如下统计字段。[S1]

| 字段 | 第一份统计示例的值 | 含义 |
|---|---:|---|
| `hostInstRate` | 247,382 inst/s | 每个宿主秒推进的模拟指令数，约 0.247 MIPS |
| `hostTickRate` | 2,295,882,330 ticks/s | 每个宿主秒推进的模拟 tick 数 |
| `simFreq` | 1,000,000,000,000 ticks/s | 模拟时间基准，每模拟秒的 tick 数 |
| `system.clk_domain.clock` | 1,000 ticks/cycle | 该示例 CPU 的周期，相当于目标时钟 1 GHz |
| `simInsts` | 6,225 | 该示例模拟的指令数 |
| `hostSeconds` | 0.03 s | 页面输出的宿主用时，已四舍五入 |

由字段可推导：

```text
模拟 CPU 周期推进速度
= hostTickRate / CPU 周期的 tick 数
= 2,295,882,330 / 1,000
= 2,295,882.33 simulated CPU cycles / host second
≈ 2.30 million simulated CPU cycles / host second
```

若口头称为“约 2.30 MHz 的等效仿真推进速度”，必须同时保留其含义和样例条件；优先使用完整单位 `simulated CPU cycles / host second`。

页面还给出第二份统计示例：`hostInstRate=202,054 inst/s`，`hostTickRate=15,979,964,060 ticks/s`。这里将两份数据作为官方教程的独立教学示例引用。开展受控性能比较时，需要记录宿主平台、gem5 版本、模型配置和工作负载。

### 引用范围

- 这是官方文档的短教学样例，第一份包含 6,225 条指令、宿主时间约数十毫秒；引用用途是说明统计字段和该样例的推进速度。稳态吞吐需要专门的基准实验。
- 相邻的 simple configuration 教程使用 x86 TimingSimpleCPU 与 1 GHz 目标时钟。[S2] O3CPU 的速度需要对应配置下的单独基准实验。
- `simFreq=10^12` 表示每模拟秒包含 10^12 个 tick；模拟 CPU 周期推进速度按 `hostTickRate / CPU 周期 tick 数` 计算。
- 仿真指令吞吐采用 inst/s，模拟周期推进速度采用 simulated cycles/host second，被模拟芯片的时钟采用 cycles/simulated second。
- gem5 与 Ascend 仿真器的速度比较需要分别开展模型配置、工作负载和宿主条件明确的实验。本文开场以观察能力和优化用途作为分类主线。

### 更合适的论文依据

《The gem5 Simulator: Version 20.0+》第 2.8 节明确指出：详细的执行驱动 CPU 模型（例如乱序模型）以仿真速度为代价获得较高精度，trace-driven 模拟可以缓解这一问题。[S3] 这里将该论文用于说明模型取舍，具体推进速度采用条件明确的实验数据。

gem5 原始论文第 2.1 节及图 1 同样将其描述为覆盖多种速度／精细程度取舍的框架，列出 AtomicSimple、TimingSimple、InOrder 与 O3 等模型。[S4] 因此汇报表中写“gem5 的详细时序模型”。

## 昇腾 CA Model 的表述依据

官方 msOpProf 文档明确说明：[S5][S6]

1. 工具具有上板采集与 simulator 两种模式；simulator 在 CPU 上模拟 NPU 行为。
2. simulator 模式对算子进行“指令级性能仿真”。
3. 输出包括代码行／指令的执行 cycle、指令流水图和源码热点映射。
4. 仿真库目录和选择方式随产品、版本而变化，新版文档包含 `camodel` 目录；本次 Ascend910B4 的库身份按实际环境记录。

“指令级性能仿真／时序仿真”有直接公开依据，可作为上屏定位。Ascend910B4 仿真库与真机的逐周期对应范围和误差界，继续列为后续核实与验证事项。[L1]

精细时序仿真同时服务于架构设计与算子优化。两种用途各有对应的研究对象、工作负载范围和决策目标。本文分析目标算子的执行机制，并通过原合同真机评测判断候选是否保留。

## NPUMeter、NPU-Simulator 与 CANN 仿真库的来源

| 名称 | 来源 | 具体定位 |
|---|---|---|
| NPUMeter | 上海交通大学、华为等机构联合研究，ACM TACO 2026 | 面向 Ascend 的解析性能模型与自动参数搜索框架 |
| NPU-Simulator（本汇报采用） | 用户指定的昇腾官方仓库 `cann/npu-simulator` | 昇腾 NPU 仿真工具，底层模型按用户说明以动态链接库形式提供 |
| CANN 仿真库（本文实际使用） | 华为昇腾 CANN 工具链，当前目标 Ascend910B4 | 由 msOpProf simulator 调用，输出算子指令级性能信息 |

NPUMeter 论文全名为 **NPUMeter: Automatic Operator Optimization for Ascend NPU with Accurate Analytical Performance Models**，发表于 *ACM Transactions on Architecture and Code Optimization*，23(2)，2026，DOI：`10.1145/3820380`。[S10] 其摘要介绍两项核心能力：根据优化参数配置估算算子延迟的解析模型，以及搜索较优参数配置的设计空间探索算法。摘要报告其论文实验中平均预测误差低于 5%，搜索时间从小时级缩短至分钟级。该项目已补入解析性能模型一行。

本汇报中的 NPU-Simulator 采用用户提供的昇腾官方仓库：<https://gitcode.com/cann/npu-simulator>。其底层动态链接库与 CA model 的关系按用户说明记录。本轮工作聚焦开场表格与 PPT 排版。此前查阅的延世大学 NPUsim 作为独立研究工具的参考资料保存。[S7]

“CANN 算子仿真器”是前稿对相关能力的概括称呼。上屏名称统一为 **CANN 仿真库（通过 msOpProf simulator 采集）**，同时明确库的提供方、采集工具和本工程实际使用的芯片版本。[S5][S6][L1]

本页 PPT 的工具链表述采用：**msOpProf simulator → CANN CA model（.so）→ 指令热点与流水时序**。其中工具入口负责调用、采集与解析，底层模型负责模拟执行。“白盒”聚焦执行机制的可观测性。

独立可编辑开场页：[simulator-overview-20261122.pptx](simulator-overview-20261122.pptx)。建议位置为汇报封面后的背景介绍页，主表保留四类，本文采用的时序仿真行以绿色突出。

## 来源

- [S1] gem5 官方教程，Understanding gem5 statistics and output：<https://www.gem5.org/documentation/learning_gem5/part1/gem5_stats/>。使用第一份统计示例；约 2.30 million cycles/s 是本文依字段作出的计算。
- [S2] gem5 官方教程，Creating a simple configuration script：<https://www.gem5.org/documentation/learning_gem5/part1/simple_config/>。
- [S3] Lowe-Power et al., The gem5 Simulator: Version 20.0+, §2.8 Elastic Traces：<https://arxiv.org/html/2007.03152#S2.SS8>；论文入口：<https://arxiv.org/abs/2007.03152>。
- [S4] Binkert et al., The gem5 Simulator, 2011, §2.1，PDF 第 2 页：<https://research.cs.wisc.edu/multifacet/papers/can11_gem5.pdf>。
- [S5] Ascend/msopprof 官方仓库，README 与 simulator 使用指南：<https://github.com/Ascend/msopprof>；<https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_simulator_user_guide.md>。
- [S6] msOpProf simulator 性能数据文件：<https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_simulator_performance_data.md>。
- [S7] NPUsim 官方仓库：<https://github.com/yonseicasl/NPUsim>。
- [S8] Timeloop 官方 README，明确描述 fast analytical model 与 mapper：<https://github.com/NVlabs/timeloop>。
- [S9] QEMU 官方说明：<https://www.qemu.org/docs/master/about/index.html>。
- [S10] NPUMeter，ACM TACO 23(2)，2026：<https://doi.org/10.1145/3820380>。论文标题、出版信息及摘要经 Crossref DOI 元数据核对：<https://api.crossref.org/works/10.1145/3820380>。
- [L1] 本仓库实际工具和验证范围：[hgkda/README.md](../../../hgkda/README.md)。

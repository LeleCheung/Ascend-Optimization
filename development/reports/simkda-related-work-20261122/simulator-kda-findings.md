# 是否已有基于模拟器的 KDA，以及 SimKDA 名称检索

2026-11-22，北京时间。对上一轮代表性 profiling 工作调研的定向补充。

后续[PLENA 源码核查](plena-timeline-check/README.md)发现其当前版本明确支持逐操作的执行单元、发射／完成时间输出。下文“论文未展示时间线接口”不代表模拟器没有时间线能力，也尚未证明 Contour Agent 使用了该接口。

## 结论

**已有模拟器参与的 kernel 生成／优化 Agent。SimKDA 名称本次未发现计算领域的直接同名工作，可以继续采用，但不能把名称的辨识度解释为方法类别独占。**

建议全称：**SimKDA — Simulator-Guided Kernel Design Agent**。

目前可重点展示的自身特点是：在昇腾上，将指令与跨流水事件重放关联到源码／IR，分析具体等待、依赖和缓冲存活，实施多层次修改，通过修改后重放与真机性能共同验证。此次尚未找到与这一完整证据链逐项相同的公开系统；这不是对其不存在或首创性的证明。

## 已核实的直接相关工作

### KernelCraft

[论文 v2](https://arxiv.org/html/2603.08721v2)：*KernelCraft: Benchmarking for Agentic Close-to-Metal Kernel Generation on Emerging Hardware*。论文既提供 benchmark，也描述可调用工具的 Agent 优化流程。已读取论文正文、工具表及附录示例，不仅依据其他论文的引用。

- 平台表明确列出 PLENA Simulator；Coral NPU 与 BOOM 使用 Verilator；AMD NPU 使用实际硬件。因此必须按平台区分模拟器和真机。
- 性能评价使用周期级／RTL 模拟或设备执行，Agent 根据编译、功能与性能反馈迭代修改底层 kernel。
- Table 2 中 `run_evaluation` 返回正确性匹配率和 latency；`get_instruction_size` 返回指令分类计数；另有代码读写、语法检查、输出比较和文档检索工具。
- 附录展示优化完整轨迹，也展示修改编译器模板。因此不能声称其他 Agent 都停留在高层源码。
- 本次核对的工具表与示例未展示逐指令跨流水 timeline 作为独立诊断接口，不能由此断言其底层模拟器不支持该能力。

判断：**直接回答“有没有模拟器型 kernel Agent”：有。** 不必名字含 KDA 才算相关工作。

### Contour

[论文 v4，2026-11-18](https://arxiv.org/html/2608.00894v4)：*Contour: Extensible Mapping Surfaces for Agentic Accelerator Kernel Generation*。

- Contour-Explore 让 Agent 搜索合法的映射／调度参数，由编译器完成合法计划与 native code 生成。
- PLENA 路径使用官方 transaction-level simulator，返回数值结果与模拟延迟；Gemmini 路径是 FPGA 测量，不能将整篇都称为纯仿真。
- §3.3 原文：**“PLENA additionally reports a bottleneck profile containing engine activity, HBM traffic, and SRAM occupancy.”** 因此它已经给 Agent 多引擎活动、流量和存储占用反馈，不能简化成只有一个 latency 标量。
- §3.4 描述 engine-schedule DAG、执行重叠、物理存储和同步；Contour-Ext 在搜索停滞时让 Agent 添加新的编译器 knob 及 lowering，并验证合法性与收益。
- 本次未核实其有我们所展示的逐指令时间窗取证、直接单算子 IR 重排和同一商用芯片真机验证完整链条。但“模拟器＋Agent＋调度／编译扩展”已明显存在相关先例。

版本注意：最先获取的 v1 标题为 *Rethinking Agentic Kernel Generation for Emerging Accelerators*，系统名 Zomboss；最终判断使用最新 v4 Contour，不能把 v1 的反馈描述当成 v4 的完整功能。各版本原文均保存在本目录。

### hexkernels：公开工程实现线索

[仓库](https://github.com/sriharshapy/hexkernels)提供 Hexagon kernel 的生成／优化、模拟器与设备测量流程。进一步读取了固定 commit `9468abde8068e8e690d2f4dd0b7f7dd21372a54c` 的 `hexkernels/gym/env.py` 和 [bottleneck.py](https://github.com/sriharshapy/hexkernels/blob/9468abde8068e8e690d2f4dd0b7f7dd21372a54c/hexkernels/gym/bottleneck.py)。

`bottleneck.py` 明确描述 “Sim-metrics -> ranked bottleneck -> prescribed intrinsic/mechanism fix”，使用 Roofline、PMU 和实际硬件机制识别反馈给出改法，包括 DMA 预取、VTCM、HVX／HMX。它是另一个模拟器反馈参与 Agent kernel 优化的公开工程证据。本次没有复现实验，也不据 README 宣称的结果评价其成熟度。

## 名称检查

检索渠道为 GitHub repository search（包括 name／description／README）及 OpenAlex 学术索引，再直接核查相关 arXiv 正文。检索词包括 `SimKDA`、`Sim-KDA`、`Sim KDA`、kernel／agent／simulator、KernelCraft、Accel-Sim／agent 等。

| 检索 | 返回及解释 |
|---|---|
| GitHub `SimKDA`，包括 name／description／README | 0 条 |
| GitHub `Sim-KDA`／`Sim KDA` | 返回一个无关的 version-control-simulation-KDay1 仓库；不能视为精确重名 |
| OpenAlex `SimKDA` | 1 条地方财政信息系统论文的匹配，非 kernel／计算体系结构工作，且不能据匹配断定论文使用了该正式系统名 |
| OpenAlex `Sim-KDA` | 0 条 |

结论应写为“**截至本次公开检索，未发现计算领域的 SimKDA 同名项目或论文**”，不能写“全网唯一”“保证没有重名”。GitHub 仓库检索不是全代码检索，学术索引也可能漏收录。OpenReview API 本次 403，但 KernelCraft 已通过 arXiv 正文直接核实，不影响其作为先例的确认。

## 对汇报的影响

用户对我们技术路线的理解成立：在已覆盖的执行窗口与模型准确度范围内，模拟器可给出指令／流水事件时间线，并结合源码／IR 分析指导修改。它提供的是模型执行证据，真机性能仍需实测。

推荐介绍语：

> SimKDA（Simulator-Guided Kernel Design Agent）利用白盒指令与流水重放，连接性能诊断、源码／IR 变换与真机验证，实现面向昇腾的多层次 kernel 优化。

相关工作页应加入 KernelCraft 与 Contour，比较反馈粒度、可操作层级、事件／缓冲归因及硬件验证方式。我们已有 GQA IR004／005 的具体等待窗口、缓冲修改、后重放和真机收益作为实例。暂不使用“首个基于模拟器的 KDA”，也不宣称其他系统只有知识库匹配或只有标量性能反馈。

## 存档

本轮原始检索、论文 HTML／纯文本、仓库材料及来源 URL 位于 `simulator-agent-search/`。`manifest.json` 记录现有存档文件内容摘要，便于后续引用。没有运行外部仓库代码，没有启动新的设备实验或修改历史图表。

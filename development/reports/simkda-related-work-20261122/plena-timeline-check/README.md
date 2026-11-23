# PLENA 时间线能力：源码核查及命名修正

核查日期：2026-11-22。核查公开仓库 `AICrossSim/PLENA_Simulator` 当前 commit `bfed7e1fecb982c8b3e311bd31d3d325388aa08b`。仅静态读取源码，没有运行或验证模拟器准确度。该版本的能力不能自动追溯为 Contour／KernelCraft 实验所使用版本的能力，更不能证明论文 Agent 实际消费了时间线。

## 新结论

**不能将 PLENA 概括为只提供聚合指标、无法输出逐指令时序。当前公开源码明确存在逐操作时间线输出。** 需要分开比较底层模型能力、Agent 获得的反馈以及真正用于优化的证据。

根 README 将 Transaction-level Simulator 与 Analytical Latency Model 列为不同组件。Contour v4 §4.1 指定的是 transaction-level 模式，不能因为结果叫 simulated latency 就将其归为只计算总耗时的解析模型。另一方面，解析方法与时间线输出也并不互斥：它的 scoreboard 本身就使用分析式规则为每条指令计算时刻。

## 可定位证据

[cli.rs 第 183 行](https://github.com/AICrossSim/PLENA_Simulator/blob/bfed7e1fecb982c8b3e311bd31d3d325388aa08b/transactional_emulator/src/cli.rs#L183) 将默认 serial 与实验性 scoreboard 模式区分开。scoreboard 保持功能结果一致，但建模流水发射，使独立 scalar／vector／matrix／DMA 工作可以重叠，并对数据依赖与结构冲突建模。

[cli.rs 第 198 行](https://github.com/AICrossSim/PLENA_Simulator/blob/bfed7e1fecb982c8b3e311bd31d3d325388aa08b/transactional_emulator/src/cli.rs#L198) 原文：

> Optional JSONL output path tracing every op's modeled issue/finish instants under `--timing-model scoreboard`.

译：可选的 JSONL 输出路径，用于记录 `--timing-model scoreboard` 模式下每条操作建模得到的发射／完成时刻。

[scoreboard.rs 第 258 行](https://github.com/AICrossSim/PLENA_Simulator/blob/bfed7e1fecb982c8b3e311bd31d3d325388aa08b/transactional_emulator/src/accelerator/scoreboard.rs#L258) 的 `trace_op` 实际写出：

```text
pc, op, unit, issue_ps, finish_ps
```

即程序位置、操作、执行单元、发射时刻与完成时刻。按执行单元展开即可构成模型时间线，不是只有一个总耗时或累计计数。`runner.rs` 创建输出文件并将 trace sink 接入，`dispatch.rs` 在执行路径中调用 `trace_op`，因此证据不止是文档中的计划。

[scoreboard.rs 文件头](https://github.com/AICrossSim/PLENA_Simulator/blob/bfed7e1fecb982c8b3e311bd31d3d325388aa08b/transactional_emulator/src/accelerator/scoreboard.rs#L1) 定义发射时刻由前端发射约束、源资源就绪时间、执行单元忙碌时间等决定，完成时间为发射时间加建模延迟；另有 SRAM 区间依赖、异步 DMA 完成时刻及模型简化说明。因此这是具有具体依赖约束的时序模型，但不能据此称其为经过实物芯片校准的完整 RTL 周期精确模型。

## 能说与不能说

- 能说：KernelCraft 公开评测工具重点返回正确性、延迟与分类指令数；Contour 正文描述了 engine activity、HBM traffic、SRAM occupancy 的瓶颈反馈。
- 能说：PLENA 当前模拟器有逐操作、分执行单元的建模时间线输出。
- 尚未核实：Contour 实验使用何种 timing mode，是否开启并读取上述 JSONL，是否据此完成具体依赖诊断。
- 不能说：Contour 的模拟器只能给聚合指标；解析模型天然无法生成时序；只有我们的系统能画 timeline。
- 同样不能把本项目的部分窗口重放称为已经覆盖整个 kernel 的全部指令时间线。我们应按实际覆盖范围，展示同核或经过校准的事件重叠、源码／IR 对应、具体改法和真机收益。

## 命名建议

本项目应突出完整的白盒优化方法：成对真机 profiling 定位差距，模拟器事件重放与源码／IR 分析消除机制歧义，再通过源码／IR 变换、后重放及真机验证形成闭环。

据此，**WB-KDA — White-Box Kernel Design Agent** 是比强调单一工具的 SimKDA 更能表达当前完整流程的可选名称。它不是此次经过同名检索的新名字，也不意味着首创白盒 Agent。

若保持已接受的 SimKDA，也仍准确，只需将其解释为 simulator-guided 而非 simulator-only。若 C 指 Causal，不建议当前直接称 Causal KDA：依赖分析、机制预测与干预验证有因果解释价值，但尚未建立独立的因果推断方法或系统性对照验证。

英文说明可统一用：**Hardware-Grounded, White-Box Multi-Level Kernel Optimization**。区别应建立在实际使用的证据和产生的收益上，不能仅靠名字或时间线展示能力证明独特性。

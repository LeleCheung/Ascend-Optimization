# Ascend 910B 独立 Profiling Workflow 设计与证据边界

## 目标与当前事实

当前正式优化对象按验收看板华为列加速比低于 0.8× 筛选，并复测 FlagGems master；本文的 narrow_copy 是已有方法案例，完整任务见 [总 README](../README.md)。

目标不是重复打印 KGS 的 `msprof` 报表，而是为固定的算子、候选源码和 workload 建立可追溯的诊断，再用完整正确性和统一计时验证一项修改。首例为 `narrow_copy`。其配对实验已证明 KG 的 `metrics` 采集和反馈顺序可工作；profile 组第二轮的 launch-plan 缓存候选因 runs 分支少传 `EVEN` 参数而出现 9 个 `TypeError`，不能将其 timing 视作有效优化结果。

KG/KGS 原生能力不能概括成“没有 profiler”。KGS 的 Ascend 后端提供真机 `metrics` 和模拟器 `instruction` 两级，`msprof op` 还可采集 `Default,BasicInfo`、Memory、PipeUtilization 等原始 CSV。KGS `ascend_report.py` 的职责是格式化事实，明确不诊断瓶颈；KG 的 `kernel-profile-analyzer` 则根据这些事实输出推断和下一轮实验。本项目要补强的是证据选择、host/设备边界、逐 case 结构化比较、编译产物映射与优化后的强制复验，而非替换采集器。

## 910B 数据路径

对 `narrow_copy`，先按 shape、dtype、stride、dim、start、length 固定语义和实际启动的 kernel。设备侧的 AIV Vector、Scalar、MTE2/MTE3，以及 GM、UB、L2 指标用于区分搬运、流水线等待与缓存行为。华为 `msopprof_performance_data.md` 将 MTE2 定义为 GM 到 AI Core 的搬入、MTE3 为 AI Core 到 GM 的搬出，并分别列出主存读写量、GM↔UB 通路利用率、L2 命中和 scalar stall。`PipeUtilization`、`Memory`、`MemoryUB`、`L2Cache`、`ResourceConflictRatio` 是不同证据面，不能把任一比率单独当作“HBM 已达峰值”。原始 CSV 是逐 block 数据；汇总应保留分布和最差 block，不只平均。

性能分三种采集范围：

1. KGS `/evaluate` 按实际 benchmark 合同计算加速比；FlagGems adapter 可能采用设备侧 kernel 计时，不能仅凭请求中的 `walltime` 推断完整调用耗时。保存当前候选和 PyTorch reference 的同 case 延迟、计时范围、完整正确性及重复运行波动。
2. 真机 `msprof metrics` 报告设备 kernel、流水线及访存。它与 Eval 使用不同采集过程，不能用两者耗时相减来声称精确 host overhead；要判断 launch/分配成本，应单独测量 host enqueue、输出分配和同步。
3. `instruction` 使用 AI Core simulator，提供指令、PC、流水线和可能的源码位置；仿真时间不是真机延迟。先核查目标 kernel、编译产物/LLVM IR 和源码的映射覆盖率，映射不全时停在 IR/指令级结论。

同一源码的身份至少有两种：ledger 中 `solution.sha256` 是 `main.py` 文本哈希；KG profile snapshot 中的 `solution_sha256` 是整个 solution JSON 的规范化哈希。必须分别校验，并用 evaluation fingerprint、case UUID、profile ID 和 artifact SHA 关联采集，不能直接比较两个同名 SHA。

## 独立 workflow 的输入与输出

输入是固定候选源码、KGS Eval JSON、对应 case 的 metrics/instruction JSON、可下载原始 msprof 归档、编译 IR（若有）、采集请求及版本/设备信息。每个 case 先检查源码、定义、workload、fingerprint 和采集范围。输出同时包含机器可读 JSON 和中文报告：测量事实、artifact 路径及哈希、设备/host/仿真来源、置信度、瓶颈假设、反证、一个可证伪的修改、回退条件以及必须通过的正确性门禁。缺少原始材料时返回 `unavailable` 或 `inconclusive`，不把缺失值当作零。

Roofline 仅在 FLOPs、实际搬运 Bytes、设备执行时间和 roof 对应同一 kernel/range 且来源明确时计算。`narrow_copy` 的 `2 × 输出元素数 × dtype 字节数` 只是逻辑读写量，不能代替物理 GM/HBM Bytes；纯拷贝也不适合用计算 FLOPs 屋顶判断。当前独立报告应保持 `roofline.status=unavailable`，直到验证当前 CANN 生成的官方 Roofline 原始数据、对应范围和 roof 来源。仓库的 [Roofline 数据契约](<Roofline 分析.md>)要求 `manifest.json + points.jsonl`；不满足时不生成伪点。

## 从 CUDA 方法迁移

已复核 `KernelFlow-ops/cuda-optimized-skill` 的 `cuda-kernel-optimizer`，HEAD `114a6cba4c194e18d3fe23a4fc2251c982f34309`。可迁移的是：环境/硬件门禁，正确性和稳定计时先于优化，先 profile 当前 best，单项假设与预期指标，失败停止或回退，以及同口径消融。其 `near_peak` 仅在三类差距、可信 workload model 都已知时才允许；910B 缺少这些量时不声称接近上限。

NCU 指标名、SASS、`sm_arch`、NVIDIA 峰值表及其 CUDA 实现脚本不能直接迁移。Ascend 对应证据是 msprof CSV、simulator 指令、Triton/LLVM IR 和实际 910B/CANN 版本。正反候选需在同一 18 个 correctness、15 个 timing case 上复测；任一正确性失败的 timing 只作诊断，不参与 best 选择。

## 待验证的工具能力

华为官方 msopprof 文档的当前版本列有 `--aic-metrics=Roofline`、`TimelineDetail`，并说明 Atlas A2/A3 的若干采集和 MindStudio Insight 可视化路径。910B 容器中的 `msopprof --help` 也列出这些参数，但这只证明 CLI 暴露选项；是否能对本次 Triton kernel 导出可分析的 Roofline/流水图，须以真实产物、适用范围及当前 CANN 9.0.0 的运行结果判定。老师材料中“Insight 当时仅支持仿真和 950 上板”应标记为当时工具版本结论，不与当前官方文档混为一谈。

## 来源

- 项目要求：[细粒度 Profiling 算子优化](<Fine-grained Profiling算子优化.md>)、[Roofline 分析](<Roofline 分析.md>)、[Msopprof/MindStudio Insight 调优](msopprof-mindstudio-insight/调优说明.md)。
- 华为官方 `Ascend/msopprof` commit `a63063569742c1ba7be9688fc7799a8248574dee`：[上板用户指南](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)、[性能字段](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_performance_data.md)、[仿真指南](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_simulator_user_guide.md)。
- 仓库实现：`sources/kernelgen_server/kernelgen_server/profiling/ascend/ascend.py`、`ascend_report.py`、`ascend_instruction.py`，以及 `sources/kernelgen/.kernelgen/agents/kernel-profile-analyzer.md` 和 `sources/kernelgen/tools/profile_round.py`。
- CUDA 方法参考：[KernelFlow-ops/cuda-optimized-skill](https://github.com/KernelFlow-ops/cuda-optimized-skill)，上述固定 commit 的 `skills/cuda-kernel-optimizer/SKILL.md`、`scripts/roofline.py`、`scripts/ablate.py`。

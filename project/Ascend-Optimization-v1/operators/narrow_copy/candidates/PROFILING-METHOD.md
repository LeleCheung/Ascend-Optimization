# narrow_copy 分析与对照方法

## 数据边界

Excel `验收看板_结果表.xlsx` 中 `narrow_copy` 位于第 29 行，`G29=0.0138`，列名是 `华为-Ascend910B`。工作簿没有记录此数的 baseline、完整 workload、聚合和计时口径，因此只用于选题，不能与 KGS 的 PyTorch-relative speedup 换算。正式比较以 KGS `eval_round` 的 15 个 timing case 和最终独立复验为准。

同一候选的 profile 证据分三层：KGS Eval walltime 是用户侧性能判定；`msprof metrics` 是真机设备侧诊断；`instruction` 是 AI Core simulator 的指令证据。三者可能由不同进程和采集窗口产生，不能把 Eval 与 msprof 时间直接相减后称为精确 host overhead。若 profile 请求还采到输入生成 kernel，必须过滤目标 kernel 并保留采集范围限制。

`docs/Roofline 分析.md` 要求 FLOPs、Bytes、时间和 roof 对应同一 kernel/range。`narrow_copy` 是纯拷贝；当前没有匹配的硬件 Bytes、实测 910B roof 和有效的计算 FLOPs，因此独立分析器输出 `roofline.status=unavailable`。可以报告逻辑读写量的估计和实测有效带宽，但必须标记来源、缓存影响与适用范围，不将其冒充物理 HBM 利用率。

## 从 CUDA 优化方法迁移的部分

参考 `KernelFlow-ops/cuda-optimized-skill` 的 `cuda-kernel-optimizer`，调研时 HEAD 为 `114a6cba4c194e18d3fe23a4fc2251c982f34309`。可迁移的是硬件与工具可用性门禁、基线正确性和计时门禁、先 profile 当前 best 再生成下一轮候选、单次改动的预期指标、分支选择和回退归因，以及对编译后指令的核验。该仓库明确要求缺少可信 workload model 时只能给“瓶颈差距启发式”，不能声称 `near_peak`；这个约束同样适用于 910B。

不能直接迁移的是 CUDA 的 NCU/SASS、`sm_arch`、CUDA metric 名和与 GPU 相关的数值阈值。910B 使用 KGS 的 `msprof metrics` 和 AI Core simulator `instruction`，对应 MTE2/MTE3、AIV/Scalar、资源冲突及源码映射；MindStudio Insight 的可视化能力还需要按具体采集类型确认。此阶段不复制原仓库脚本或把 CUDA 的 roof、带宽数字代入 Ascend。

## A/B 验收

`run-narrow-copy-ab-910b.sh` 的两组共享 seed、模型、`simple_opt`、两轮预算、FlagGems 隔离副本、KGS `19652`、PyTorch reference 的定义、warmup 和计时参数，仅切换 `--no-profile` / `--profile`。reference 在两组中分别重新测量。两组顺序运行，避免互相争用 KGS 单 worker。profile 组必须在 round 1 完成后记录分析，再让 Coder 修改 round 2；用 ledger、分析文件的 round/fingerprint 和日志时间顺序确认这一点。如果未满足，保留运行结果，但标为“流程跑通，因果对照无效”。

每个 case 对比同一 UUID 的 PyTorch/reference ms、候选 ms、speedup、正确性和退化；最终再比较两组独立复验。由于模型搜索和设备计时会波动，两次运行只能描述该次实验差异，不能断言 profiler 的平均收益或算法理论上限。

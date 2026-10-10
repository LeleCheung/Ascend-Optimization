# narrow_copy

验收看板华为列 G29 为 0.0138×，符合低于 0.8× 的筛选条件。历史连续 dim=0 CANN D2D 混合候选三轮为 0.5919×、0.5672×、0.5613×，每轮正确性通过，仍未达标。

## 本轮固定 master 复验

2026-10-10 campaign 使用固定 master `d6a8eec473517a3d68157b208eb9c057eb1d4c50`，仅适配 benchmark case API，算子源码保持原样。专用 KGS 19656 使用物理卡 7，在 amin、matmul 原生 KG 与分析候选任务结束后才启动。四版本为 master、KG 无 profiler、KG 原生 profiler、连续复制快路径；新合同下完整范围为 18 个正确性与 15 个性能 case，最终以实际独立结果核验。

固定 master 首次完整评测在 PyTorch 参考计时阶段失败，尚未调用候选。`probe-pytorch-copy-timing.py` 已对三个 dtype 的小/大输入采集 trace 与原始 task CSV：小输入是 `MEMCPY_ASYNC`，`kernel_name=N/A` 被 pandas 解析为浮点空值，原计时器的 `.str` 调用崩溃；大输入是 `TensorMove` AIV kernel，原计时器能正常工作。原始失败保留于 `reports/master-case-adapter-20261010/`，完整诊断见 [复制计时证据](reports/diagnostics-20261010/narrow-pytorch-copy-timing-v2-20261010/artifacts/pytorch-copy-timing.json)。

新副本 `FlagGems-narrow-device-timing` 仅把该列转为 pandas string 再过滤，保留原始设备任务耗时、采样数与聚合，明确计时包括 DMA 与 kernel。补丁只用于 narrow benchmark，不修改安装包或其他算子；真实安装计时器的 DMA 与 kernel 回归测试均通过。副本提交 `3d79162eb257588d585004a5188ec3dfbcdc4263`，KGS 19657，物理卡 7。完整 master 复验通过 18 个正确性用例，但设备 profiler 计时停在 CANN 设备同步；两次原生堆栈均显示 `aclrtSynchronizeDeviceWithTimeoutImpl`，已保存并停止本轮对应 benchmark 进程。源码与计时器快照见 [合同归档](reports/device-timing-contract-20261010/manifest.json)，同步卡点见 `reports/device-profiler-hang-20261010/`。

另建 `FlagGems-narrow-operator-timing`（提交 `9171f02`，KGS 19658）测试上游 `BenchMode.OPERATOR`。它同样停在 `float16 [1024,65536]` 候选启动后的设备同步，说明问题不仅是 profiler。堆栈明确指出性能 case `float16::4`，停止记录在 `reports/operator-master-hang-20261010/`；未改动其他进程。

大输入诊断对纯 Triton 快路径和每次最多 8192 program 的 master 兼容候选，分别验证三个 dtype，均同步完成、逐值正确。因此最终回到 KGS 19657 的设备任务计时，使用 `flaggems-master-grid-compatible-20261010.py` 作为两组 KG 的同一起点：保留 master 索引计算、动态除数和 1024 分块，只增加 program 基址并分批启动。该列必须标为 **master 加启动兼容修复**。完整四版本范围仍为 18 项正确性、15 项性能；纯 Triton 与小输入 DMA/大输入 Triton 混合候选分别评测。入口为 `run-grid-compatible-campaign-910b.sh`。

`probe-current-timing.py` 使用本轮快路径候选，对三个 dtype 的小/大输入分别采设备 kernel 计时、host enqueue、同步调用和预分配输出后的 JIT 启动；分配时间也独立测量。各范围保持分开，不能把两个独立采集耗时相减当作准确 host 开销。历史“host/launch 开销主导”应由本轮分解重新核查，不能用于解释默认设备 kernel 加速比。两组原生 KG 尚未开始。

## 文件入口

| 路径 | 内容 |
|---|---|
| [candidates/](candidates/) | Triton、launcher 缓存、D2D 候选、边界验证和配对提示词 |
| [scripts/](scripts/) | 专用运行、预分配、host 分解、msprof 采集、编译 IR 与 D2D 探针 |
| [reports/](reports/README.md) | 各阶段结果与原始证据 |

`launch_plan_compiled.py` 为混合 D2D 路径候选；`launch_plan_fixed.py` 为早期 Triton 候选；`launch_plan_compiled_persist8.py` 为失败性能尝试。评测批次中的源码快照代表当次实际运行版本，不能用后来修改的候选替换。

已分析 msprof 和模拟器信息，Roofline 尚无可用结果。编译链分析见 [报告](../../../../docs/narrow-copy-compiler-chain.md)。下一轮应先固定 FlagGems master、复测完整 workload，再决定继续优化的切入点。

## 历史配对入口

`scripts/run-narrow-copy-ab-910b.sh` 的候选与提示词路径已更新。它仍使用历史 KGS 端口、模型和运行目录，重跑前检查服务配置并指定新的实验工作区；新正式算子优先使用 [公共运行工具](../../tools/README.md)。

# narrow_copy

验收看板华为列 G29 为 0.0138×，符合低于 0.8× 的筛选条件。历史连续 dim=0 CANN D2D 混合候选三轮为 0.5919×、0.5672×、0.5613×，每轮正确性通过，仍未达标。

## 本轮固定 master 复验

2026-10-10 campaign 使用固定 master `d6a8eec473517a3d68157b208eb9c057eb1d4c50`，仅适配 benchmark case API，算子源码保持原样。专用 KGS 19656 使用物理卡 7，在 amin、matmul 原生 KG 与分析候选任务结束后才启动。四版本为 master、KG 无 profiler、KG 原生 profiler、连续复制快路径；新合同下完整范围为 18 个正确性与 15 个性能 case，最终以实际独立结果核验。

`probe-current-timing.py` 使用本轮快路径候选，对三个 dtype 的小/大输入分别采设备 kernel 计时、host enqueue、同步调用和预分配输出后的 JIT 启动；分配时间也独立测量。各范围保持分开，不能把两个独立采集耗时相减当作准确 host 开销。campaign 还将采集 float16 小/大输入的 `PipeUtilization`，由独立 workflow 输出中文体检。历史“host/launch 开销主导”应由本轮分解重新核查，不能用于解释默认设备 kernel 加速比。当前任务排队中，尚无新结果。

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

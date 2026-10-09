# narrow_copy

验收看板华为列 G29 为 0.0138×，符合低于 0.8× 的筛选条件。历史连续 dim=0 CANN D2D 混合候选三轮为 0.5919×、0.5672×、0.5613×，每轮正确性通过，仍未达标。

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

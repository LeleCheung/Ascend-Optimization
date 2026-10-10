# narrow_copy

验收看板华为列 G29 为 0.0138×。本轮固定 FlagGems master `d6a8eec473517a3d68157b208eb9c057eb1d4c50`，完整范围为 18 项正确性与 15 项性能，三个 dtype。

## 最终四版本

| 版本 | 相对 PyTorch | 相对兼容 master | 完整测试 |
| --- | ---: | ---: | --- |
| master 加启动兼容修复 | 0.010× | 1.00× | 33/33 |
| KG 无 profiler | 0.495× | 48.68× | 33/33 |
| KG 原生 profiler | 0.806× | 81.37× | 33/33 |
| 分析指导混合优化版 | 1.102× | 108.23× | 33/33 |

最终源码：[candidates/profile-small-dma-hybrid-20261010.py](candidates/profile-small-dma-hybrid-20261010.py)。实现类型：**小输入 CANN DMA + KG profiler 循环复制 Triton**。输出重新分配并实际复制；同源码重复评测取较慢轮，详细选择见 [最终候选](reports/master-closure-20261010/最终候选.json)。

性能为包括 DMA 与 AIV kernel 的设备任务计时，排除 host 发射间隙。KG 两组同一起点、模型和两轮预算；分析优化版独立标注。

关键改进是简化复制索引、消除逐元素动态除法，并按连续布局与输入大小选择 DMA 或 Triton 路径。

真机耗时分解表明：小输入的 Triton 设备任务本身慢于 PyTorch DMA，大输入连续复制已接近 PyTorch 设备耗时。旧“主要是 host/launch 开销”的判断不适用于本轮完整合同；具体任务与耗时见 [真机诊断](reports/master-closure-20261010/真机诊断.md)。

报告入口：[精简汇报](reports/master-closure-20261010/精简汇报.md)、[逐 case 四版本对比](reports/master-closure-20261010/版本对比.md)、[复现步骤](../../tools/evaluation/复现与交接-20261010.md)。

## master 兼容修复与计时合同

固定 master 首次完整评测在 PyTorch 参考计时阶段失败，尚未调用候选。`probe-pytorch-copy-timing.py` 已对三个 dtype 的小/大输入采集 trace 与原始 task CSV：小输入是 `MEMCPY_ASYNC`，`kernel_name=N/A` 被 pandas 解析为浮点空值，原计时器的 `.str` 调用崩溃；大输入是 `TensorMove` AIV kernel，原计时器能正常工作。原始失败保留于 `reports/master-case-adapter-20261010/`，完整诊断见 [复制计时证据](reports/diagnostics-20261010/narrow-pytorch-copy-timing-v2-20261010/artifacts/pytorch-copy-timing.json)。

新副本 `FlagGems-narrow-device-timing` 仅把该列转为 pandas string 再过滤，保留原始设备任务耗时、采样数与聚合，明确计时包括 DMA 与 kernel。补丁只用于 narrow benchmark，不修改安装包或其他算子；真实安装计时器的 DMA 与 kernel 回归测试均通过。副本提交 `3d79162eb257588d585004a5188ec3dfbcdc4263`，KGS 19657，物理卡 7。完整 master 复验通过 18 个正确性用例，但设备 profiler 计时停在 CANN 设备同步；两次原生堆栈均显示 `aclrtSynchronizeDeviceWithTimeoutImpl`，已保存并停止本轮对应 benchmark 进程。源码与计时器快照见 [合同归档](reports/device-timing-contract-20261010/manifest.json)，同步卡点见 `reports/device-profiler-hang-20261010/`。

另建 `FlagGems-narrow-operator-timing`（提交 `9171f02`，KGS 19658）测试上游 `BenchMode.OPERATOR`。它同样停在 `float16 [1024,65536]` 候选启动后的设备同步，说明问题不仅是 profiler。堆栈明确指出性能 case `float16::4`，停止记录在 `reports/operator-master-hang-20261010/`；未改动其他进程。

大输入诊断对纯 Triton 快路径和每次最多 8192 program 的 master 兼容候选，分别验证三个 dtype，均同步完成、逐值正确。因此最终回到 KGS 19657 的设备任务计时，使用 `flaggems-master-grid-compatible-20261010.py` 作为两组 KG 的同一起点：保留 master 索引计算、动态除数和 1024 分块，只增加 program 基址并分批启动。该列必须标为 **master 加启动兼容修复**。完整四版本范围仍为 18 项正确性、15 项性能；纯 Triton 与小输入 DMA/大输入 Triton 混合候选分别评测。入口为 `run-grid-compatible-campaign-910b.sh`。

该兼容基线已完成 **33/33**，相对 PyTorch **0.0103×**。三个 dtype 的最小输入约 26～29 μs，而 PyTorch DMA 约 0.6 μs；最大输入约 20～23 ms，而 PyTorch 约 103～217 μs。这是设备任务耗时的实际差距，需要优化内核与调度，不能直接解释为 Python 或 host 开销。


## 文件入口

- `candidates/`：固定 master 兼容候选、纯 Triton、混合候选及来源清单。
- `scripts/`：适配、完整评测、真机诊断与择优汇总工具。
- `reports/`：各批次原始请求、源码快照、逐 case 数据和附件 SHA。

历史编译链探索见 [报告](../../../../docs/narrow-copy-compiler-chain.md)；历史版本与本轮新合同分别保存。

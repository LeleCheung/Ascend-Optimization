# 四算子工作流闭环

批次：`20261014-four-direct`。最佳源码复现与建议驱动优化分别记录。

| 算子 | 模式 | 完整测试 | 起点 | 最终两次，相对PyTorch | 较慢值 | 相对起点 | 建议产生有效优化 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| logaddexp2 | 最佳源码复现 | 36/36 | 1.265× | 1.265× / 1.262× | 1.262× | 0.998× | — |
| amin | 最佳源码复现 | 27/27 | 0.938× | 0.941× / 0.934× | 0.934× | 0.996× | — |
| matmul_bias_activation | 最佳源码复现 | 42/42 | 0.891× | 0.889× / 0.890× | 0.889× | 0.998× | — |
| narrow_copy | 最佳源码复现 | 33/33 | 1.120× | 1.105× / 1.114× | 1.105× | 0.986× | — |

narrow_copy使用device_task，包含DMA和kernel；其他三个使用device_kernel。性能case取几何平均，最终采用两次完整复验较慢值。

方法起点由四算子配置明确固定：logaddexp2从master出发，其余三个从历史中间版本出发，分别验证后续有效建议。

方法有效须有模型生成的新源码、采纳记录及最终复验相对起点超过0.5%；固定源码复验不计为重新生成优化。数值Roofline尚未建立。

完整测试、源码SHA、原始附件与最佳选择已由verify-agent-campaign.py核验；失败和未采纳轮次保留在各批次。

最佳复现来自`20261014-four`，建议驱动优化来自`20261014-four-direct`。各列为各自批次实测；方法改善只比较该方法批次内的起点与最终，不将跨HEAD结果当成同合同消融。

- `logaddexp2` / `replay`：[证据目录](logaddexp2/reports/profiling-agent-20261014-four-replay/campaign-state.json)。
- `amin` / `replay`：[证据目录](amin/reports/profiling-agent-20261014-four-replay/campaign-state.json)。
- `matmul_bias_activation` / `replay`：[证据目录](matmul_bias_activation/reports/profiling-agent-20261014-four-replay/campaign-state.json)。
- `narrow_copy` / `replay`：[证据目录](narrow_copy/reports/profiling-agent-20261014-four-replay/campaign-state.json)。

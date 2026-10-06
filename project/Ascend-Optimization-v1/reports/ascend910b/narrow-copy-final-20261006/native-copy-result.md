# narrow_copy CANN native copy 最终结果

## 交付候选

候选源码为 `experiments/ascend910b/narrow_copy/launch_plan_compiled.py`，源码 SHA-256：
`a118364f32d9453aee6f9a8d2d96fdf9bff14e645af79168c42a9d4d2b36d598`。

对连续内存且 `dim=0`、`start/length` 合法的调用，候选直接使用当前 torch-npu stream 调用 CANN `aclrtMemcpyAsync(..., ACL_MEMCPY_DEVICE_TO_DEVICE, ...)`；其他 layout 保留原 Triton 路径。这样保持了原始 `run(inp, dim, start, length)` ABI，不依赖 FlagTree 私有 compiled runner。

## 完整验证

| 运行 | correctness | timing | geo mean vs PyTorch | 设备 |
| --- | ---: | ---: | ---: | --- |
| native-01 | 18/18 | 15/15 | `0.5919×` | KGS 自动分配 |
| native-02 | 18/18 | 15/15 | `0.5672×` | KGS 自动分配 |
| native-03 | 18/18 | 15/15 | `0.5613×` | KGS 自动分配 |

三轮均 `PASSED / 33/33`。边界语义复验 `27/27` 通过。原最佳 Triton 候选的三轮 geo mean 为 `0.5349×`、`0.5120×`、`0.5152×`；native 候选在三轮中分别提升约 `10.7%`、`10.8%`、`8.9%`（按 geo mean 比值计算）。原始 JSON 保存在本目录。

## 失败路径

最初独立 native 文件只在连续分支使用 CANN copy，其他分支 fallback 到 `torch.narrow_copy`。KGS 会在 reference 环境重绑定该符号，造成递归并出现 `24/33` 通过；这已修复为“连续 dim=0 走 CANN，其他 layout 走原 Triton 实现”，没有把失败版本作为候选。

## 结论

这是一个真实超过当前 Triton 候选的版本，但仍低于 PyTorch native。收益来自 CANN runtime 的原生 D2D copy 和较短的调用路径，而不是 profiler 输出或内部 IR hack。当前适用范围是连续 `dim=0` narrow copy；非连续/其他维度继续使用原 Triton 分支，并已通过边界测试。

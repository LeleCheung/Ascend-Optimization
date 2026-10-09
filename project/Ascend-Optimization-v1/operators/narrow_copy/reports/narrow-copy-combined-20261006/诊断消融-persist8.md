# 诊断驱动消融：持久化 kernel 的迭代数

独立体检在大尺寸 `float16` case 观察到 `_narrow_copy_flat_persistent_kernel` 的 MTE2 比例 `0.916`、MTE3 比例 `0.7125`。据此提出一个单项假设：减少每个 program 的持久化循环迭代数，可能增加并行度并改善搬运流水线。只把 `_PERSIST_ITERS` 从 `13` 改为 `8`，其余源码、Definition、KGS、33 个 workload 和计时口径保持不变。

候选源码 SHA-256：`980b8764c18d496efb11ae9f54604c0703508334c93f7a5de093d5510acb5b27`。

| 项目 | 结果 |
| --- | --- |
| correctness | 18/18 通过 |
| timing | 15/15 通过 |
| geo mean vs PyTorch | `0.5112×` |
| 原候选首轮 | `0.5349×` |

该修改没有改善整体 geo mean，因此按实验门禁回退，不作为最终候选。它说明 MTE2/MTE3 活跃比例足以提出搬运假设，但不足以单独决定更好的迭代数；后续需要同卡重复测量、更多流水线指标或其他单项修改。正式结果保存在 `eval-persist8.json`，不可与原候选不同设备的单轮数字作严格因果比较。

# 历史归档

2026-10-09 导师明确以华为列实际加速比低于 0.8× 筛选。此前依据黄色底色选择的部分对象不符合这一要求，已退出当前优化主线。归档保留候选、结果和正确性问题记录，供复现和方法参考。

## 误选对象

| 算子 | Excel 华为列 | 归档原因 | 入口 |
|---|---:|---|---|
| mse_loss_backward | G7：1.1688× | Excel 已超过 0.8× | [记录](operators/mse_loss_backward/README.md) |
| prelu_kernel_backward | G33：1.8152× | Excel 已超过 0.8× | [记录](operators/prelu_kernel_backward/README.md) |
| t_copy | G43：3.0894× | Excel 已超过 0.8× | [记录](operators/t_copy/README.md) |
| smooth_l1_loss_backward | G600：2.2139× | Excel 已超过 0.8× | [记录](operators/smooth_l1_loss_backward/README.md) |
| batch_norm_backward | G159：4.3885× | Excel 已超过 0.8×，且原为绿色 | [记录](operators/batch_norm_backward/README.md) |
| native_layer_norm | G61：1.4108× | Excel 已超过 0.8×；我们生成较慢候选不改变筛选依据 | [记录](operators/native_layer_norm/README.md) |

原结果仍是有效历史实验，但不计入“低于 0.8× 算子优化”的新成果。是否标黄、我们某版候选是否慢于 PyTorch，都不能替代 Excel 数值筛选。

## 联合对照和链路记录

- [五算子 FlagGems 原版对照](comparisons/flaggems-comparison-20261009/README.md)：保留整批 measurements、source-audit、合同与汇总；跨算子数据不拆散，各算子索引链接至对应子目录。
- [三项修复与性能复测](comparisons/flaggems-repaired-comparison-20261009/README.md)：双方三轮完整通过，PReLU、BatchNorm、SmoothL1 相对本机修复版 FlagGems 分别为 1.97×、2.19×、1.61×；PReLU 新候选相对 PyTorch 为 0.95×，旧候选有随机精度失败。可汇报文字见 [周报精简版](comparisons/flaggems-repaired-comparison-20261009/周报精简版.md)。
- [square 烟雾测试](smoke-tests/square/README.md)：早期 KG+KGS 链路验证。
- [旧阶段状态](phase1/phase1-status-20261007.md)、[旧阻塞记录](phase1/phase1-blockers-20261007.md)：只描述当时进度。

归档批次中的 JSON、执行脚本、日志与源码快照保留原字节。历史绝对路径与端口描述当时运行环境；当前可维护的工具见 [tools](../tools/README.md)。

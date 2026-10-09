# 第一阶段运行阻塞

## 现象

在 KGS `19653` 上使用当前绑定的 FlagGems checkout 运行以下正式 Definition：

- `mse_loss_backward`
- `prelu_kernel_backward`
- `t_copy`
- `batch_norm_backward`

这些算子都在正式优化前的 `prepare_catalog` 阶段失败。FlagGems benchmark 的 `list_cases(initialize=False)` 调用 `supports_cases()`，随后抛出：

```text
ValueError: Operator '<operator>' does not support case listing yet.
```

因此没有生成 workload、没有启动 Coder，也没有产生候选性能结论。

## 判断

这是当前 KGS 与 FlagGems runtime 的 case-listing 协议/实现缺口，不是这些算子的正确性失败，也不是 910B 设备故障。不能通过删减 workload、手写单个 shape 或修改 reference 来伪造优化结果。

## 后续处理

应先确认 19653 使用的 FlagGems revision 与 benchmark API 是否匹配；若升级或切换到支持 case listing 的固定 checkout，再按原始 pytest 重新生成 Definition/benchmark。升级前保留现有服务和失败 workspace，不停止共享 KGS。

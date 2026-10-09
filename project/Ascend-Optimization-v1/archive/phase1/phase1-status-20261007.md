# 第一阶段状态汇总

## 已得到性能结论

`native_layer_norm` 已完成真实设备基线和候选对照。Triton 候选正确，但约为 PyTorch 的 `0.18~0.21x`；Ascend 原生实现约为 `0.95~1.05x`。profiling workflow 已生成 baseline 和设备探针，未收敛出可交付的超越原生候选。

## 已进入 Coder 但未收敛

`matmul_bias_activation` 在新版隔离 KGS `19654`、FlagGems `4772d816` 上完成 catalog、preflight 和真实 Coder 启动，执行了 37 轮 Ascend tile/debug 探针，但没有 measured round。证据保留在：

```text
runtime/kg-controller/runs/matmul-bias-activation-noprofile-4772d816-20261007/
```

## 统一阻塞

以下算子均在原始 FlagGems benchmark 的 `list_cases(initialize=False)` 阶段失败：

- `mse_loss_backward`
- `prelu_kernel_backward`
- `t_copy`
- `batch_norm_backward`
- `smooth_l1_loss_backward`

错误一致为：

```text
ValueError: Operator '<name>' does not support case listing yet.
```

该错误发生在 KGS 生成 workload 之前，不能从中推出正确性或性能结论。`smooth_l1_loss_backward` 还额外确认：即使通过 `kg definition` 导出独立 catalog，KGS 仍会执行原始 benchmark 并在同一点失败。

## 环境证据

- 共享 KGS `19653`：旧 FlagGems checkout `6274667`，未修改。
- 隔离 KGS `19654`：容器 `tle_yy`，FlagGems `4772d816`，已成功启动并验证 8 张 NPU 可用。
- 因此 case-listing 阻塞在新版 checkout 上仍可复现，不是单纯 revision 不匹配。

## 后续动作

要完成剩余算子的三版本闭环，需要先修复或升级 FlagGems benchmark/KGS 的 case-listing 能力，并重新生成 Definition/测试契约。当前证据不足以宣称这些算子“打不过 PyTorch”，只能报告为 workflow 阻塞。

## 2026-10-07 复现进展

已为 `mse_loss_backward`、`prelu`、`t_copy` 和 `batch_norm_backward` 补齐新版 KGS 所需的 `get_case_iter()` 与 `build_inputs()`，本地语法检查通过，并提交为 `9c2c1681`。910B 的隔离 checkout 及 `--list-cases` 复验仍待 SSH 网关恢复可观测输出后执行；在此之前不把补丁当作设备侧验证完成。

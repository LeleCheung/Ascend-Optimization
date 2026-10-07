# smooth_l1_loss_backward 阶段报告

## 结果

本次 KG 无 profiler 任务已运行到正式候选评测，但没有确认 winner。正确性阶段共收集 `303` 个 workload，`302` 个通过；剩余失败发生在 beta=0 的边界输入，出现 NaN，因此本轮不能给出性能加速比。

## 证据

- 910B workspace：`runtime/kg-controller/runs/smooth-l1-backward-phase1-skip/`
- 状态：`FAILED / optimize`
- eval 状态：`PARTIAL_PASS`
- 正确性：`302/303`
- 性能：未进入可用的正式 timing 结果

## 判断

候选的常规 elementwise 路径已通过绝大多数形状、dtype、reduction 和广播组合。当前阻塞点是 beta=0 边界的 NaN 语义，需要先修正候选或测试契约，再重新进行完整评测；不能根据本轮结果宣称性能优劣。

## 修复后结果

phase2 使用修正后的 beta=0 等值分支重新评测，正式通过全部 `303/303` 个 workload。几何平均加速比为 `3.515x`，最差 workload 为 `1.867x`，说明该 elementwise 候选在当前 KG device-time 口径下已经取得稳定加速。

- phase2b workspace：`runtime/kg-controller/runs/smooth-l1-backward-phase2b/`
- eval：`PASSED`
- 正确性与计时：`303/303`
- 几何平均加速比：`3.515x`
- 最差加速比：`1.867x`

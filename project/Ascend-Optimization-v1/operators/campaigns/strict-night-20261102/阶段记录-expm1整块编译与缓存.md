# expm1_：从执行差距追到编译失败与缓冲限制

最终交付保留已核验的0.561×版本。第二轮KG搜索及Codex整块修正均已完成真实评测，新稿未超过最佳。所有数字均相对原PyTorch，原生CANN参考的小值问题仍未通过资格检查。

## 实测与原因

同输入设备诊断：小输入候选／PyTorch为3.14／3.34 μs，大输入为133.90／30.56 μs。大输入需要改进设备搬运与计算组织，不能只优化宿主启动。

| 第二轮稿件 | 完整评测 | 相对PyTorch | 说明 |
| --- | --- | ---: | --- |
| 第1稿 | 30/30 | 0.561× | 原最佳源码，复用已有测量 |
| 第2稿 | 30/30 | 0.0236× | 引入8192完整块，半精度中大输入约12 ms |
| 第3稿 | 30/30 | 0.0659× | 未超过原最佳 |
| 第4稿 | 30/30 | 0.000322× | 未超过原最佳 |

第二稿的`_expm1_value_narrow`在constexpr分支返回FP16/BF16，函数末尾又返回FP32，导致编译器报`Inconsistent return types`。宿主捕获异常后回退，但把失败缓存为`None`，后续调用将其当作未编译，反复尝试失败路径。离线编译复现了两种半精度的类型错误；FP32可编译。CPU执行实际宿主函数也确认：连续调用两次时，原稿编译尝试两次；修复后只尝试一次，仍恰好执行两次回退。

## 已实施的修复

1. 将不同类型的constexpr返回改为明确的`if/else`。
2. 编译失败缓存为独立的`False`状态；保持一次调用只执行一次原地kernel。
3. 恢复指针对齐保护，计划继续按类型、元素数和设备缓存，每次读取当前指针与stream。
4. 根据真实编译产物控制临时缓冲，保留原公式与严格Triton回退。

加入显式零选择的v4虽修复了类型错误，8192块仍需要208／224／224 KiB的UB，超过当前编译器报告的192 KiB可用容量，因此未提交NPU测量。缓冲节省／单缓冲选项未消除该溢出；当前厂商编译器不接受`enable-auto-vectorize-v2`选项，相关失败原件均保留。

v5在整块核中移除额外零选择、关闭该核的乘加融合，通用路径继承原已验证公式。FP16、BF16、FP32三份离线编译均通过，TTIR／adapter保留了小值分支的乘法、加法、乘法结构；零符号和数值结果仍由后续真机补测判断。

## 后续验收与证据

v5已通过既有KGS完整30/30与63/63补测，另外在8192、16384实际整块及非对齐、尾块上补查三类型的小值、负零、NaN／Inf和view外存储，12/12通过；运行时准入19/19通过。相对PyTorch为0.514124×，低于原最佳0.561436×，不替换交付源码，也未为这份未选中源码追加后端交付核验。

整块检查首版在`sha(__file__)`处把字符串当Path，尚未导入torch或执行设备代码就失败，原检查目录为空。修复为统一Path转换后，保留原检查器、错误日志和空目录，以同一卡锁恢复这项正确性检查；正式30项和63项补测的SHA保持不变，没有重新计时。原tmux因全部窗口退出已结束，恢复窗口只重建本项目同名会话，未重启共享服务或镜像。最终控制器54项Linux单CPU测试通过，收尾进程在2026-11-02 23:22:50 UTC正常结束，状态为`backend_verified`且无当前任务。

- [四项执行诊断](analysis/expm1_/components-v1-summary.json)
- [第二轮正式逐case快照](analysis/expm1_/search-2-formal-snapshot.json)
- [原第二稿离线编译](analysis/expm1_/compile-exact-v1/result.json)
- [失败缓存控制流检查](analysis/expm1_/failure-cache-v4.json)
- [v4缓冲溢出](analysis/expm1_/compile-exact-v4/result.json)
- [编译选项诊断](analysis/expm1_/compile-exact-options-v1/result.json)
- [v5三类型离线编译](analysis/expm1_/compile-exact-v5/result.json)
- [v5源码](proposals/expm1_/codex-exact-zero-v5/main.py)
- [v5正式30项](proposals/expm1_/codex-exact-zero-v5/pytorch/Codex-strict-PyTorch-1.result.json)
- [原63项补测](proposals/expm1_/codex-exact-zero-v5/extra-pytorch.json)
- [新增整块12项](proposals/expm1_/codex-exact-zero-v5/exact-expm1-probe/result.json)
- [检查器已知失败恢复身份](finish-pending-predevice-recovery.json)
- [最终54项工具测试](analysis/controller-tests-probe-recovery-v2.json)

离线编译与CPU检查不代表NPU性能提升；本次结论来自以上真实NPU回执。编译失败机制已修复，但完整块组织未带来有效性能提升，缺少流水指标时仍不能将剩余差距简单归因为HBM或计算受限。

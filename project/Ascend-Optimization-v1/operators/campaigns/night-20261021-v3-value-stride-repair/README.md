# value_selecting_reduction_backward 独立语义修复

<!-- 当前实测结果 -->
## 当前实测结果

- `semantic-repair:value_selecting_reduction_backward`：passed，相对PyTorch 1.593× / 1.679×。
- 扩展语义：48/48，passed。

性能计时为完整调用walltime；来源、预算及逐case结果按原文件保留。下文是实验准备时的设计说明。

本批是原始三组实验之外的额外修复，不属于相同两轮预算内的KG或HG-KDA生成成绩。父批为 `night-20261021-v3`，父版本是KG无profiler第二轮，源码SHA为 `cc149ba4e1548d6211763c0920d4778c94dd00f5c573042b6b8827cd034c1372`。固定来源、合同和额外验证预算见 [plan.json](plan.json)。

## 修复内容

父版本在某些合法转置布局下抛出 `unsupported strides`。新增通用Triton分支按输出逻辑坐标逐维计算grad和indices的物理地址，保留选中位置的梯度，并为其他位置写零；不调用PyTorch计算结果。

原始快路径的正文保持原字节，只追加一个入口包装。只有捕获到该特定的NotImplementedError时才进入通用分支，其他错误继续抛出。正常路径新增的函数调用开销也包含在后续正式计时中。

## 验证与状态

本批通过tmux等待本项目卡锁，在原主队列之后运行；不向运行中的原始三组提供候选或诊断。先运行48项补充语义检查，核对候选与检查器SHA；只有全部通过才发起两次完整96项KGS复验。原始父版本和失败记录不覆盖。

CPU解释器尝试在persistent循环的动态标量转换处抛出TypeError，尚未完成该检查。诊断见远端本批 `cpu-interpreter-attempt-2.json`；这次尝试没有访问NPU，不作为NPU正确性或性能结论。

910B宿主复现命令如下。已排队的本批不要重复启动；新尝试应复制合同并使用新批次目录。

```bash
python3 project/Ascend-Optimization-v1/tools/profiling/hgkda/run-semantic-repair.py \
  --batch night-20261021-v3-value-stride-repair
```

队列状态见 `queue-state.json`，语义结果与完整复验保存在 `results/value_selecting_reduction_backward/codex-stride-repair/`。真机验证尚未完成。

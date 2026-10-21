# HG-KDA 第二轮的额外语义修复

<!-- 当前实测结果 -->
## 当前实测结果

- `semantic-repair:value_selecting_reduction_backward`：passed，相对PyTorch 1.465× / 1.408×。
- 扩展语义：48/48，passed。

性能计时为完整调用walltime；来源、预算及逐case结果按原文件保留。下文是实验准备时的设计说明。

父批是 `night-20261021-v3`，父组为 HG-KDA，父轮次为2，源码 SHA 为 `e893186748573e4908a2d4df050d6147322c95979344f714a50bccf93b0f2bb3`。该轮正式合同为96/96，轮内加速比1.550×，独立复验和补充语义检查尚未完成。

源码审查发现，父版本的 `_group` 只支持连续折叠或至多两层非仿射布局；三层以上的合法转置布局会抛出 `unsupported stride pattern`。此外，空输入没有专门处理。这是源码审查发现，真机结果由独立补测给出。

本批保留父源码原始字节，追加空输入返回及通用 Triton 分支。仅在捕获该特定 NotImplementedError 时进入通用分支，其他错误不吞掉。通用分支逐维计算 grad 和 indices 地址，不使用 PyTorch 计算结果。父通用路径可能在拒绝布局前已提交清零任务，额外任务也包含在对应调用的运行成本中；正式连续输入的快路径保留。

修复候选 SHA 为 `2841d7b4cba65d8c0c53a61032121aeb3585cac9a0bed063ef2ec5f401bdbb66`。固定合同、来源和额外预算见 [plan.json](plan.json)。先通过48项补充语义，再做两次完整96项 KGS 复验。

这是原始三组之外的 Codex 修复，不计为原始 HG-KDA 相同预算内自动生成的结果，也不改写父版本的成绩或 profiling 使用时序。排队后等待同一本项目物理卡7的锁，不打断主队列。

```bash
python3 project/Ascend-Optimization-v1/tools/profiling/hgkda/run-semantic-repair.py \
  --batch night-20261021-v3-value-hg-repair
```

已排队的批次不要重复启动；当前尚不能宣称候选通过真机验证。

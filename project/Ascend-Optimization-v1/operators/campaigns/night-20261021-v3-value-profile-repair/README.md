# value_selecting_reduction_backward：原生 profiling 第二轮的独立修复

<!-- 当前实测结果 -->
## 当前实测结果

- `semantic-repair:value_selecting_reduction_backward`：passed，相对PyTorch 1.044× / 1.103×。
- 扩展语义：48/48，passed。

性能计时为完整调用walltime；来源、预算及逐case结果按原文件保留。下文是实验准备时的设计说明。

本批是原始三组实验之外的 Codex 修复，不计入相同两轮预算内的 KG 或 HG-KDA 生成结果。父批为 `night-20261021-v3`，父版本是 KG 原生 profiler 第二轮，原评测为 90/96，状态为 `PARTIAL_PASS`。父源码 SHA 为 `6aeec9d48ca503024ec2ecce06dc9c5558c3ab039d662cf2a18a5cd347bb6951`。

## 问题与修复

父版本合并多行处理以减少 program 数，但归约长度为 5 时向下选取 4 列，导致最后一列未写入。边界 mask 只能限制已有列的写入，无法补齐没有生成的列。父版本还用最后一维 stride 推断整个输入布局，一般转置布局不满足这个假设。

本批保留父源码全部原始字节，在其后追加入口包装和通用 Triton 分支。仅当归约长度是 2 的幂且不超过 8192、grad 与 indices 均连续且末维 stride 为 1 时使用原快路径；其他布局或长度按逻辑坐标逐维计算物理地址，写入选中梯度或零。空输出直接返回，不启动 kernel。PyTorch 仅用于分配输出，不用于计算返回值。

候选 SHA 为 `98b4cb7d62b4481202ff6e14c05f32002e5ab629c991b2ee883f0180180ea88a`。固定合同、谱系和额外预算见 [plan.json](plan.json)。

## 验证与运行

本批在独立 tmux 会话中等待本项目物理卡 7 的锁，主队列结束后才执行。先通过 48 项补充语义检查并核对源码、检查器 SHA，再进行两次完整 96 项 KGS 复验。修复前的源码与失败证据均保留。

```bash
python3 project/Ascend-Optimization-v1/tools/profiling/hgkda/run-semantic-repair.py \
  --batch night-20261021-v3-value-profile-repair
```

已排队时不要重复运行此命令。状态见 `queue-state.json`；补充语义与复验见 `results/value_selecting_reduction_backward/codex-stride-repair/`。该目录名由共用修复控制器沿用，实际父组以本批 `plan.json` 为准。尚无真机验证结果时，不将此候选列为有效交付。

# hardshrink_backward新增交付记录

最终版本相对合格CANN为**1.042×**，相对PyTorch为**1.037×**；两侧完整评测各30/30、补测各45/45，运行时检查19/19，源码审查及代表输入的实际TTIR、npubin、设备trace核验均通过。正式成绩各测一次，原始数据和源码身份保留。

hardshrink_backward计算硬收缩函数的反向梯度：输入落在阈值区间内时抑制梯度，区间外传递上游梯度。最终实现保留三个改进：

1. 大输入按设备Vector核数组织少量program，每个program循环处理多块，减少大量小program的重复调度；完整块不需要逐元素边界判断。
2. 尾部仍保留掩码，已经没有尾部数据的program直接跳过；小输入使用单块直线计算。
3. 宿主热路径合并重复检查和分配包装，复用编译及启动元数据，每次重新读取当前指针、参数与stream。

CC生成的性能候选把乘掩码改成了条件选择，普通合同通过，但阈值附近的Inf梯度在NPU参考中得到NaN，条件选择却得到0，导致补测39/45。Codex在五处Triton表达式恢复了本机NPU语义，保留分块、尾部与启动改进：

```python
gradient = go.to(tl.float32) * tl.where(tl.abs(xx) <= lambd, 0.0, 1.0)
```

修正后两侧补测均45/45。此前较高但补测失败的1.063×保留为搜索证据，不作为交付成绩。这里以冻结NPU参考为准，CPU旁列结果没有替代NPU语义。

源码为`proposals/hardshrink_backward/codex-npu-masked-gradient-v1/main.py`，SHA为`20432054a1f412574fbdf9f9522d955e26785d7cae14abd2c95a7fbbe20610dd`。原始正式结果、两侧补测和后端证明均位于同一目录。

KG/HG-KDA与CC/DeepSeek生成了流式内核和宿主路径改进；Codex提供成对profiling、诊断与编排、逐版本审查，并直接修复最终数值表达式。候选所有设备计算与搬运均为Triton，CANN仅在参考侧使用。

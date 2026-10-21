# 将 HG-KDA 的真实分析建议落实为额外候选

<!-- 当前实测结果 -->
## 当前实测结果

- `semantic-repair:value_selecting_reduction_backward`：passed，相对PyTorch 1.516× / 1.477×。
- 扩展语义：48/48，passed。

性能计时为完整调用walltime；来源、预算及逐case结果按原文件保留。下文是实验准备时的设计说明。

父实验是 `night-20261021-v3` 的 HG-KDA 第二轮。它完成源码优化后采集了5份真机指标，并登记结构化分析；两个指令请求超时，没有成功指令报告。原始三组的两轮预算不变，本批是额外实验。

## 修改与来源

原分析认为，大输入可能在逐元素比较、选择和写回上花费较多时间，建议将“每个元素判断是否选中”改成“整块清零，再稀疏回填选中的梯度”。原建议及其证据引用按原字节保存在 `diagnostic-source/profile-analysis-round-0002.json`，SHA为 `de836337c762bdc1ce18e0664875a29322a8dc30586e5acafd2f720e25ee78ae`。它是需要验证的优化假设，不是已经验证的瓶颈定律。

Codex 在已登记的 HG-KDA 语义修复源码之后追加一个 kernel，并替换连续末维快路径使用的 kernel 符号：同一 program 先清零，再执行 `tl.debug_barrier()`，最后写入少数选中值。还增加了列块归属条件，只有包含该索引的列块才能回填它；单个 program 的 barrier 不能解决跨 program 的写入冲突。

原始父源码、通用 stride 修复和回退路径均保留。候选源码SHA为 `b816562294fd68089c734cd125bd28de053ce1e0185c7af79d84cbb792664538`。是否能编译、是否正确、是否更快，均等待真机验证，不将模型预测写为实测收益。

## 验证顺序

1. 原有48项补充语义全部通过。
2. 额外18项连续大归约维检查全部通过，每项重复5次。覆盖长度8209、16387、非2次幂尾部、8192列边界、部分输出行，以及三种dtype和两种keepdim。
3. 使用同一冻结96项KGS合同完整复验两次，包含新增包装和实际调用成本。

额外检查器SHA、分析来源与预算均冻结于 [plan.json](plan.json)。`verify-semantic-repairs.py` 核对父归档、真实profiling分析、两层补测与完整复验；不会将本批替换成原始HG-KDA相同预算内的成绩。

本批在独立tmux中等待本项目物理卡7的锁：

```bash
python3 project/Ascend-Optimization-v1/tools/profiling/hgkda/run-semantic-repair.py \
  --batch night-20261021-v3-value-hg-profile-guided
```

已排队时不要重复启动。结果见 `results/value_selecting_reduction_backward/codex-stride-repair/`，该共用控制器目录名不代表本批只有stride修复。

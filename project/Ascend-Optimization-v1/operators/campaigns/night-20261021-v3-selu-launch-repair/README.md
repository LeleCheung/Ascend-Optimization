# SELU额外启动与布局修复

## 最终实测

36项扩展语义每项重复两次全部通过，两次正式36项也全部通过，相对PyTorch为 **0.236× / 0.243×**，仍低于0.8×。父源码、追加修复及原始组的记录分别保留；完整核验见主批`selu-repair-verification.json`。以下保留实验准备时的设计说明。

本批是Codex落实真实HG分析与Coder探针后的额外候选，不属于原始HG-KDA两轮自动生成成绩。原始组保留0.131×/0.125×的独立复验及第二轮未改善的记录。

父源码为 `night-20261021-v3` 中HG-KDA第二轮，SHA `13a6a1583c9cbf1f202f112e468f7d820366d0d22984fedd159287c402383034`；完整字节保存在 `parent/round-0002.py`，候选保留该前缀。新源码SHA为 `2d58eb89ded17eb0343a56c2ca2e8129cc172c17490b07c15f4b61f28e56d10f`，固定原版与正式36项合同不变。

## 具体修改

1. 按当前FlagTree正常JIT路径补 `debug` 与编译选项，再计算同一specialization key，避免查错缓存。调用当前stream及正常metadata/hook，只缓存编译产物；不缓存输入Tensor或结果。未命中时只执行一次普通入口；真正提交launch后异常直接报错，不重复执行原地算子。
2. 连续且至少1048576元素使用8192分块，其他输入使用1024；这是原Coder探针已测得有效的方向，正式收益仍待复验。grid设上限，超大输入以grid-stride覆盖。
3. 非连续输入按全部rank的shape/stride计算地址，补齐原候选最多4维的限制。保留原地返回同一Tensor和原数值公式。

底层缓存与launcher依赖固定FlagTree实现；这是实际软件版本约束，不要求升级编译器或改动root配置。

## 队列与验收

`run-selu-repair.py` 在tmux中等待本项目物理卡7卡锁，主队列结束后先运行 `check-selu-repair.py`：3种dtype、12种形状/布局，每项两次，共36项额外语义。检查空输入、标量、奇数尾部、rank3—6、非连续stride、storage offset、原地别名、哨兵保护及NaN/Inf。只有同源码/检查器的全部36项通过，才使用原KGS正式36项合同做两次完整walltime复验。

新增预算：36项补充语义×2次、2次完整KGS复验。失败保留，不删case、不改变原合同、不把附加候选归入原始两轮。

```bash
python3 tools/profiling/hgkda/run-selu-repair.py \
  --batch night-20261021-v3-selu-launch-repair
```

上述命令在910B项目目录运行；已有队列时不重复启动。状态见 `queue-state.json`，产物见 `results/selu_/codex-launch-repair/`。当前尚待NPU结果，不宣称已修复或已达标。

结束后使用容器Python3.11运行 `verify-selu-repair.py --batch-dir <本批目录> --output <本批目录>/selu-repair-verification.json`，核验父源码归档、追加代码SHA、冻结合同、36项语义及两次完整复验，按同case耗时计算相对FlagGems。`pending`/`invalid`均返回非零，只有 `verified` 才表示这些交付证据完整；是否超过0.8×另看实际性能。

来源见[启动路径与分块诊断](../../../tools/profiling/hgkda/research/逐元素算子的启动路径与分块诊断-20261021.md)。

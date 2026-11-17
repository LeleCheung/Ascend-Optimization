# uniform

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/uniform.py)；SHA `f033a7f1d1d4c270ef867f7f99b22e66e84a2937e77138edda8d8b6942e5ee87`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/cf49e54e579d144ed71408748ca31f943ebac5d966430e96129d3d9bb1202088.py)；原始模块AST一致，追加入口：`run = uniform`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/690af7b076a9ad7a1b7883cf9c0ea67e7c7f66d7c1989ba33ee1ef7c8a6dbb2c.json)：**0.141×**；18/18。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/c602c4fd81ed5e8d27d7ad896e7f973d86c9f0c2ec1b32eb9a9aec972af289eb.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/night-20261021-v3/results/uniform/FlagGems/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 11条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 9份候选测量、4份请求源码；详细位置列在下表 |
| 交付终点 | 2.398526× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `night-20261021-v3/results/uniform/FlagGems/FlagGems-1.result.json` | 0.141349× | [cf49e54e579d](../../experiments/lineage/20261117/source-evidence/sources/cf49e54e579d144ed71408748ca31f943ebac5d966430e96129d3d9bb1202088.py) |
| 候选 | `endurance-20261022/results/uniform/parent/formal/parent-1.result.json` | 0.155749× | [cf49e54e579d](../../experiments/lineage/20261117/source-evidence/sources/cf49e54e579d144ed71408748ca31f943ebac5d966430e96129d3d9bb1202088.py) |
| 候选 | `endurance-20261022/results/uniform/attempt-2/candidates/b4c0ba819fe77ead84221a61bc8219d1655c870d3a40e408a3a4fcaeee87f7ea/formal/round-0003-1.result.json` | 1.496608× | [b4c0ba819fe7](../../experiments/lineage/20261117/source-evidence/sources/b4c0ba819fe77ead84221a61bc8219d1655c870d3a40e408a3a4fcaeee87f7ea.py) |
| 候选 | `endurance-20261022/results/uniform/attempt-2/candidates/93c0885de86da6d1b22d39842f1565965e82b74a4e66e66af4607be5c27536ca/formal/round-0002-1.result.json` | 0.912989× | [93c0885de86d](../../experiments/lineage/20261117/source-evidence/sources/93c0885de86da6d1b22d39842f1565965e82b74a4e66e66af4607be5c27536ca.py) |
| 基线目录 | `next25-20261029-v2/results/uniform/baseline-fixed/FlagGems-1.result.json` | 0.143072× | [cf49e54e579d](../../experiments/lineage/20261117/source-evidence/sources/cf49e54e579d144ed71408748ca31f943ebac5d966430e96129d3d9bb1202088.py) |
| 候选 | `next25-20261029-v2/results/uniform/historical-start/HG-KDA-start-1.result.json` | 1.361195× | [b4c0ba819fe7](../../experiments/lineage/20261117/source-evidence/sources/b4c0ba819fe77ead84221a61bc8219d1655c870d3a40e408a3a4fcaeee87f7ea.py) |
| 候选 | `threeway-20261030/results/uniform/current-recheck/formal/HG-KDA-current-1.result.json` | 1.310337× | [b4c0ba819fe7](../../experiments/lineage/20261117/source-evidence/sources/b4c0ba819fe77ead84221a61bc8219d1655c870d3a40e408a3a4fcaeee87f7ea.py) |
| 厂商参考 | `cann-priority-20261102/references/uniform/storage-v1/formal/CANN-1.result.json` | 0.865007× | [6a7878d32005](../../experiments/lineage/20261117/source-evidence/sources/6a7878d3200502a9cbd0ecd481324802f47e17203eae602e70907d42fbdb2b91.py) |
| 候选 | `strict-triton-20261102/candidates/uniform/restored-v1/pytorch/strict-PyTorch-1.result.json` | 1.386892× | [b4c0ba819fe7](../../experiments/lineage/20261117/source-evidence/sources/b4c0ba819fe77ead84221a61bc8219d1655c870d3a40e408a3a4fcaeee87f7ea.py) |
| 候选 | `strict-night-20261102/proposals/uniform/codex-launch-once-v1/pytorch/Codex-strict-PyTorch-1.result.json` | 1.627599× | [506b8cb022e5](../../experiments/lineage/20261117/source-evidence/sources/506b8cb022e5ce03f16cbb2a95d01780c4522836f889af4f4de972909ceee10b.py) |
| 候选 | `strict-night-20261102/proposals/uniform/codex-launch-once-v1/formal-reference/Codex-strict-PyTorch-1.result.json` | 1.627599× | [506b8cb022e5](../../experiments/lineage/20261117/source-evidence/sources/506b8cb022e5ce03f16cbb2a95d01780c4522836f889af4f4de972909ceee10b.py) |
| 候选 | `uniform-native-range-20261109-v2/operators/uniform/evaluation/result.json` | 2.398526× | [506b8cb022e5](../../experiments/lineage/20261117/source-evidence/sources/506b8cb022e5ce03f16cbb2a95d01780c4522836f889af4f4de972909ceee10b.py) |

## 交付终点

- [选用源码](../../../operators/uniform/versions/independent_strict/retained/kernel.py)：`506b8cb022e5ce03f16cbb2a95d01780c4522836f889af4f4de972909ceee10b`。
- [保留版完整结果](../../../operators/uniform/evidence/retained/evaluation/result.json)：**2.398526× CANN，18/18**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

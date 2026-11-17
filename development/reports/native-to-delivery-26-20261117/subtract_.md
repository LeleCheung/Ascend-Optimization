# subtract_

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/subtract_.py)；SHA `a1f7cad4179e9ac58e6a53f3a8f3faf94380ac0f93acf2ba0c0c03c13c34cb56`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/1f19801af0d85cd8c23ce2d633ab72a1b323381d55a4cd61e863d12b948be654.py)；原始模块AST一致，追加入口：`run = subtract_`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/c3272fe1edbcf61e1ae4a312d8b4149b5bc90e2f4ca8556c59df1054ffcd2c6a.json)：**PARTIAL_PASS**；27/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/c1564ebbfbf939b167e31b18102ba35a61822804e25ed153f5735feed95c9921.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/subtract_/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 2条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 6份候选测量、2份请求源码；详细位置列在下表 |
| 交付终点 | 1.658593× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/subtract_/baseline-fixed/FlagGems-1.result.json` | PARTIAL_PASS | [1f19801af0d8](../../experiments/lineage/20261117/source-evidence/sources/1f19801af0d85cd8c23ce2d633ab72a1b323381d55a4cd61e863d12b948be654.py) |
| 候选 | `next25-20261029-v2/results/subtract_/search-1/independent/HG-KDA-1.result.json` | 1.069337× | [190310ac54d7](../../experiments/lineage/20261117/source-evidence/sources/190310ac54d7d2564bd0b4d889d5e9d4bb47b2fddaee8f8e43464ff303f8ad3f.py) |
| 候选 | `next25-20261029-v2/proposals/subtract_/independent-strides-v1/assessment/formal/HG-KDA-independent-strides-v1-1.result.json` | 1.100076× | [462e3f429976](../../experiments/lineage/20261117/source-evidence/sources/462e3f429976dea2cfa95582d0c3df3d153388dff1a25a285659992de5a835d1.py) |
| 候选 | `threeway-20261030/results/subtract_/current-recheck/formal/HG-KDA-current-1.result.json` | 1.145069× | [462e3f429976](../../experiments/lineage/20261117/source-evidence/sources/462e3f429976dea2cfa95582d0c3df3d153388dff1a25a285659992de5a835d1.py) |
| 厂商参考 | `cann-priority-20261102/references/subtract_/formal/CANN-1.result.json` | 0.839915× | [aeed9303fa60](../../experiments/lineage/20261117/source-evidence/sources/aeed9303fa60091764b23ff8fc357c70a77def81a1ae6b7f7971c34f64822720.py) |
| 候选 | `cann-priority-20261102/comparisons/subtract_/starting/HG-KDA-start-1.result.json` | 1.556305× | [462e3f429976](../../experiments/lineage/20261117/source-evidence/sources/462e3f429976dea2cfa95582d0c3df3d153388dff1a25a285659992de5a835d1.py) |
| 候选 | `strict-triton-20261102/candidates/subtract_/restored-v1/pytorch/strict-PyTorch-1.result.json` | 1.130652× | [462e3f429976](../../experiments/lineage/20261117/source-evidence/sources/462e3f429976dea2cfa95582d0c3df3d153388dff1a25a285659992de5a835d1.py) |
| 候选 | `independent-regression-20261109-v3/operators/subtract_/evaluation/result.json` | 1.658593× | [462e3f429976](../../experiments/lineage/20261117/source-evidence/sources/462e3f429976dea2cfa95582d0c3df3d153388dff1a25a285659992de5a835d1.py) |

## 交付终点

- [选用源码](../../../operators/subtract_/versions/independent_strict/retained/kernel.py)：`462e3f429976dea2cfa95582d0c3df3d153388dff1a25a285659992de5a835d1`。
- [保留版完整结果](../../../operators/subtract_/evidence/retained/evaluation/result.json)：**1.658593× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

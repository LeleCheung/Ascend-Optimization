# special_xlogy_

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/special_xlogy.py)；SHA `205a06cd51e9f7580c203365d573d0332a39bb35892851ace5db07ee735e3fb7`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/2cc75c2598fee2cb2187962d0a0eeb9f3d70006f58577425f28c787c31d31a62.py)；原始模块AST一致，追加入口：`run = special_xlogy_`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/e016cdb4d564d3f7afbef7821fb3c1315223293dc0effeebca8951656e577668.json)：**PARTIAL_PASS**；27/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/579ad6ad2461482b6882ffbc801f318278f9706fc814af7525eb6c5f99df53cd.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/special_xlogy_/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 3条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 6份候选测量、2份请求源码；详细位置列在下表 |
| 交付终点 | 1.071040× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/special_xlogy_/baseline-fixed/FlagGems-1.result.json` | PARTIAL_PASS | [2cc75c2598fe](../../experiments/lineage/20261117/source-evidence/sources/2cc75c2598fee2cb2187962d0a0eeb9f3d70006f58577425f28c787c31d31a62.py) |
| 候选 | `next25-20261029-v2/results/special_xlogy_/search-1/independent/HG-KDA-1.result.json` | 0.855788× | [d93e11e96221](../../experiments/lineage/20261117/source-evidence/sources/d93e11e96221fb94c081ef3d3b17b80b0c6a6ce7bc1647cc3eb70e9f0411d51a.py) |
| 候选 | `next25-20261029-v2/proposals/special_xlogy_/independent-layout-launch-v1/assessment/formal/HG-KDA-independent-layout-launch-v1-1.result.json` | 0.579046× | [a9e36ef862dc](../../experiments/lineage/20261117/source-evidence/sources/a9e36ef862dcf15575d3c4ce68f65927e042bc56d80551b10b606742b1834baf.py) |
| 候选 | `threeway-20261030/results/special_xlogy_/current-recheck/formal/HG-KDA-current-1.result.json` | 0.842463× | [d93e11e96221](../../experiments/lineage/20261117/source-evidence/sources/d93e11e96221fb94c081ef3d3b17b80b0c6a6ce7bc1647cc3eb70e9f0411d51a.py) |
| 厂商参考 | `cann-priority-20261102/references/special_xlogy_/extra-v1/formal/CANN-1.result.json` | 0.964998× | [ef094024a902](../../experiments/lineage/20261117/source-evidence/sources/ef094024a90247d5673cd7707795b5aa647a8083096f84f4048b84c69c40bf44.py) |
| 候选 | `cann-priority-20261102/comparisons/special_xlogy_/starting/HG-KDA-start-1.result.json` | 1.003726× | [d93e11e96221](../../experiments/lineage/20261117/source-evidence/sources/d93e11e96221fb94c081ef3d3b17b80b0c6a6ce7bc1647cc3eb70e9f0411d51a.py) |
| 候选 | `strict-triton-20261102/candidates/special_xlogy_/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.846493× | [d93e11e96221](../../experiments/lineage/20261117/source-evidence/sources/d93e11e96221fb94c081ef3d3b17b80b0c6a6ce7bc1647cc3eb70e9f0411d51a.py) |
| 候选 | `independent-regression-20261109-v3/operators/special_xlogy_/evaluation/result.json` | 1.071040× | [d93e11e96221](../../experiments/lineage/20261117/source-evidence/sources/d93e11e96221fb94c081ef3d3b17b80b0c6a6ce7bc1647cc3eb70e9f0411d51a.py) |

## 交付终点

- [选用源码](../../../operators/special_xlogy_/versions/independent_strict/retained/kernel.py)：`d93e11e96221fb94c081ef3d3b17b80b0c6a6ce7bc1647cc3eb70e9f0411d51a`。
- [保留版完整结果](../../../operators/special_xlogy_/evidence/retained/evaluation/result.json)：**1.071040× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

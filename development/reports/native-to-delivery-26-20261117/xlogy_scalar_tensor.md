# xlogy_scalar_tensor

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/xlogy.py)；SHA `e28f19aa2d1e1db9531cf61eec6dddecda16547f00aacd52d09d85fa424325ef`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/ad174757acd193d5b318911db9cb8fb9e30a3617dbfe8bbae21aa7d2a17ca87f.py)；原始模块AST一致，追加入口：`run = xlogy_scalar_tensor`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/74e18f9966efbeb7b0d9379856428768ce62be199b7bac687bd33c45457e442c.json)：**0.148×**；30/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/5c1e54df9663c3a1ce05a507b58b5628ee3b26a2c26eb36bbb4d281044bc42a4.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/xlogy_scalar_tensor/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 2条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 7份候选测量、3份请求源码；详细位置列在下表 |
| 交付终点 | 1.535565× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/xlogy_scalar_tensor/baseline-fixed/FlagGems-1.result.json` | 0.148169× | [ad174757acd1](../../experiments/lineage/20261117/source-evidence/sources/ad174757acd193d5b318911db9cb8fb9e30a3617dbfe8bbae21aa7d2a17ca87f.py) |
| 候选 | `next25-20261029-v2/results/xlogy_scalar_tensor/search-1/independent/HG-KDA-1.result.json` | 1.235092× | [9156ff91ad0e](../../experiments/lineage/20261117/source-evidence/sources/9156ff91ad0e1ac313baf8a6a288de3c7bdd4c100cd8f6cd32ee5d6b4f3bd027.py) |
| 候选 | `next25-20261029-v2/proposals/xlogy_scalar_tensor/direct-launch-v2/assessment/formal/HG-KDA-direct-launch-v2-1.result.json` | 1.185801× | [e3dc45628258](../../experiments/lineage/20261117/source-evidence/sources/e3dc45628258fa7354116b0d965cd602e9524d6a52441f8b94eacd74b1a587df.py) |
| 候选 | `next25-20261029-v2/proposals/xlogy_scalar_tensor/runtime-scalar-v3/assessment/formal/HG-KDA-runtime-scalar-v3-1.result.json` | 1.253320× | [f4f49782fe8d](../../experiments/lineage/20261117/source-evidence/sources/f4f49782fe8d32856da04d94771f15ff43242d35dce33b05b25bd67ef80f4b0e.py) |
| 候选 | `threeway-20261030/results/xlogy_scalar_tensor/current-recheck/formal/HG-KDA-current-1.result.json` | 1.226749× | [f4f49782fe8d](../../experiments/lineage/20261117/source-evidence/sources/f4f49782fe8d32856da04d94771f15ff43242d35dce33b05b25bd67ef80f4b0e.py) |
| 厂商参考 | `cann-priority-20261102/references/xlogy_scalar_tensor/extra-v1/formal/CANN-1.result.json` | 0.784828× | [a0e950f9ae64](../../experiments/lineage/20261117/source-evidence/sources/a0e950f9ae64ea8cc647e649528374df73cf522083b86b674dfd889d7f95eb53.py) |
| 候选 | `cann-priority-20261102/comparisons/xlogy_scalar_tensor/starting/HG-KDA-start-1.result.json` | 1.578966× | [f4f49782fe8d](../../experiments/lineage/20261117/source-evidence/sources/f4f49782fe8d32856da04d94771f15ff43242d35dce33b05b25bd67ef80f4b0e.py) |
| 候选 | `strict-triton-20261102/candidates/xlogy_scalar_tensor/restored-v1/pytorch/strict-PyTorch-1.result.json` | 1.233708× | [f4f49782fe8d](../../experiments/lineage/20261117/source-evidence/sources/f4f49782fe8d32856da04d94771f15ff43242d35dce33b05b25bd67ef80f4b0e.py) |
| 候选 | `independent-regression-20261109-v3/operators/xlogy_scalar_tensor/evaluation/result.json` | 1.535565× | [f4f49782fe8d](../../experiments/lineage/20261117/source-evidence/sources/f4f49782fe8d32856da04d94771f15ff43242d35dce33b05b25bd67ef80f4b0e.py) |

## 交付终点

- [选用源码](../../../operators/xlogy_scalar_tensor/versions/independent_strict/retained/kernel.py)：`f4f49782fe8d32856da04d94771f15ff43242d35dce33b05b25bd67ef80f4b0e`。
- [保留版完整结果](../../../operators/xlogy_scalar_tensor/evidence/retained/evaluation/result.json)：**1.535565× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

# _conj_physical

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/conj_physical.py)；SHA `99f6aafed1d9e8b43db01f90befeb7313c47f377c54b09f5167308f8427228b0`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/d74f16c0cb5183369a413385e3d98fce2173ee9a1c2a6c6c52db5ee0708a9e06.py)；原始模块AST一致，追加入口：`run = conj_physical`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/d9739c8e34369382ab6a88f5de9b760ca9b0bbadc22da61332e2aee7706dafd3.json)：**1.723×**；14/14。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/1b73df4a1810dc6e60933bfdfb96f0f54872ca75478bcebe27646d6884846d7d.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/night-20261021-v3/results/conj_physical/FlagGems/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 4条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 12份候选测量、6份请求源码；详细位置列在下表 |
| 交付终点 | 1.234423× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `night-20261021-v3/results/conj_physical/FlagGems/FlagGems-1.result.json` | 1.722565× | [d74f16c0cb51](../../experiments/lineage/20261117/source-evidence/sources/d74f16c0cb5183369a413385e3d98fce2173ee9a1c2a6c6c52db5ee0708a9e06.py) |
| 候选 | `remaining-five-20261023-v2/results/_conj_physical/parent/formal/parent-1.result.json` | PARTIAL_PASS | [339b42db0ca0](../../experiments/lineage/20261117/source-evidence/sources/339b42db0ca09f0f76196dc40a8cab1c7be7353364e2398b8b37258dd0ebad18.py) |
| 候选 | `remaining-five-20261023-v2/results/_conj_physical/attempt-1/candidates/c5dd849d838b35f5b3f4d0df73c3bd84a877c9a83f8a8fdca43e0ba868b909a3/formal/round-0004-1.result.json` | 105.484278× | [c5dd849d838b](../../experiments/lineage/20261117/source-evidence/sources/c5dd849d838b35f5b3f4d0df73c3bd84a877c9a83f8a8fdca43e0ba868b909a3.py) |
| 候选 | `remaining-five-20261023-v2/results/_conj_physical/attempt-1/candidates/feccb337a7fc417432a66c6fefde42181665a258a6b7fdf82ce96c2799ecaef2/formal/round-0003-1.result.json` | 115.518857× | [feccb337a7fc](../../experiments/lineage/20261117/source-evidence/sources/feccb337a7fc417432a66c6fefde42181665a258a6b7fdf82ce96c2799ecaef2.py) |
| 候选 | `remaining-five-20261023-v2/results/_conj_physical/attempt-1/candidates/38748dd722f0bbff0803d22ea2a7fc6e47f4bc2e2ccb9d51b90757d6dba56db2/formal/round-0002-1.result.json` | 91.187393× | [38748dd722f0](../../experiments/lineage/20261117/source-evidence/sources/38748dd722f0bbff0803d22ea2a7fc6e47f4bc2e2ccb9d51b90757d6dba56db2.py) |
| 候选 | `remaining-five-20261023-v2/results/_conj_physical/attempt-1/candidates/9164b1dde5699d174a796e86903e6702bf878140b82ee04364f0b84b5bd18103/formal/round-0001-1.result.json` | 39.777392× | [9164b1dde569](../../experiments/lineage/20261117/source-evidence/sources/9164b1dde5699d174a796e86903e6702bf878140b82ee04364f0b84b5bd18103.py) |
| 候选 | `acceptance-three-20261026/validation-v2/_conj_physical/default-reference.result.json` | 47.169901× | [feccb337a7fc](../../experiments/lineage/20261117/source-evidence/sources/feccb337a7fc417432a66c6fefde42181665a258a6b7fdf82ce96c2799ecaef2.py) |
| 候选 | `acceptance-three-20261026/validation-v2/_conj_physical/npu-decomposition-reference.result.json` | 2.296363× | [feccb337a7fc](../../experiments/lineage/20261117/source-evidence/sources/feccb337a7fc417432a66c6fefde42181665a258a6b7fdf82ce96c2799ecaef2.py) |
| 候选 | `reference-closure-20261031/comparisons/_conj_physical/NPU-decomposition-v2/HG-KDA-vs-NPU-v2-1.result.json` | 2.166577× | [feccb337a7fc](../../experiments/lineage/20261117/source-evidence/sources/feccb337a7fc417432a66c6fefde42181665a258a6b7fdf82ce96c2799ecaef2.py) |
| 厂商参考 | `cann-priority-20261102/references/_conj_physical/storage-v1/formal/CANN-1.result.json` | 42.113397× | [cfae90e3f029](../../experiments/lineage/20261117/source-evidence/sources/cfae90e3f029bb2d034188c036968f1c3359c24d0ee7dc400ed728a41b5a4f5a.py) |
| 候选 | `cann-priority-20261102/comparisons/_conj_physical/starting/HG-KDA-start-1.result.json` | 1.343100× | [feccb337a7fc](../../experiments/lineage/20261117/source-evidence/sources/feccb337a7fc417432a66c6fefde42181665a258a6b7fdf82ce96c2799ecaef2.py) |
| 候选 | `strict-triton-20261102/candidates/_conj_physical/restored-v1/pytorch/strict-PyTorch-1.result.json` | 45.279184× | [64d85d5b54b6](../../experiments/lineage/20261117/source-evidence/sources/64d85d5b54b60b9e85ec804c28eb0a08e42211e0bbdba7824b997744359d1057.py) |
| 候选 | `strict-triton-20261102/candidates/_conj_physical/restored-v1/CANN/strict-CANN-1.result.json` | 1.152554× | [64d85d5b54b6](../../experiments/lineage/20261117/source-evidence/sources/64d85d5b54b60b9e85ec804c28eb0a08e42211e0bbdba7824b997744359d1057.py) |
| 候选 | `independent-regression-20261109-v3/operators/_conj_physical/evaluation/result.json` | 1.234423× | [64d85d5b54b6](../../experiments/lineage/20261117/source-evidence/sources/64d85d5b54b60b9e85ec804c28eb0a08e42211e0bbdba7824b997744359d1057.py) |

## 交付终点

- [选用源码](../../../operators/_conj_physical/versions/independent_strict/retained/kernel.py)：`64d85d5b54b60b9e85ec804c28eb0a08e42211e0bbdba7824b997744359d1057`。
- [保留版完整结果](../../../operators/_conj_physical/evidence/retained/evaluation/result.json)：**1.234423× CANN，14/14**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

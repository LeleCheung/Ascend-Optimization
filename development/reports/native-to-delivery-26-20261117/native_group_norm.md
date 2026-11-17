# native_group_norm

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/native_group_norm.py)；SHA `fcf0ca2922f751b38ea46ba9387717e4e171aac524aab699569e7f259809f2ef`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/44111fe5c7c7e0276e2d7f7ad33049ae1d1ae73f7423562cdd8aee3cac287c44.py)；原始模块AST一致，追加入口：`run = native_group_norm`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/d190d311c96829569040ee457387acbe63285259735308ed7e1ae1fa34466010.json)：**PARTIAL_PASS**；21/27。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/72ed99b89b6e157379a160623cef51c681a24f48b2753c1fa3cc0b8f0e746aab.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/native_group_norm/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 4条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 13份候选测量、7份请求源码；详细位置列在下表 |
| 交付终点 | 1.003824× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029/results/native_group_norm/baseline/FlagGems-1.result.json` | PARTIAL_PASS | [44111fe5c7c7](../../experiments/lineage/20261117/source-evidence/sources/44111fe5c7c7e0276e2d7f7ad33049ae1d1ae73f7423562cdd8aee3cac287c44.py) |
| 基线目录 | `next25-20261029-v2/results/native_group_norm/baseline/FlagGems-1.result.json` | 0.591107× | [12d25c47d259](../../experiments/lineage/20261117/source-evidence/sources/12d25c47d259a2dad3ef596cd631d118fc649fd68e15333dd32b28529e672baa.py) |
| 候选 | `next25-20261029-v2/results/native_group_norm/search-1/independent/HG-KDA-1.result.json` | 0.635652× | [12d25c47d259](../../experiments/lineage/20261117/source-evidence/sources/12d25c47d259a2dad3ef596cd631d118fc649fd68e15333dd32b28529e672baa.py) |
| 基线目录 | `next25-20261029-v2/results/native_group_norm/baseline-fixed/FlagGems-1.result.json` | PARTIAL_PASS | [44111fe5c7c7](../../experiments/lineage/20261117/source-evidence/sources/44111fe5c7c7e0276e2d7f7ad33049ae1d1ae73f7423562cdd8aee3cac287c44.py) |
| 候选 | `next25-20261029-v2/proposals/native_group_norm/warmup-launch-v1/assessment/formal/HG-KDA-warmup-launch-v1-1.result.json` | 0.600294× | [7bae89b622f5](../../experiments/lineage/20261117/source-evidence/sources/7bae89b622f5a9f5021cb9f71d6f678212046bb8edeaf879daac53447b91b5a4.py) |
| 候选 | `next25-20261029-v2/proposals/native_group_norm/separate-output-v2/assessment/formal/HG-KDA-separate-output-v2-1.result.json` | 0.727840× | [1a111c6d965f](../../experiments/lineage/20261117/source-evidence/sources/1a111c6d965ff3b761967cb625964ad46da6ca0119138585ff8b47a8cbff1dc9.py) |
| 候选 | `next25-20261029-v2/proposals/native_group_norm/factory-output-v3/assessment/formal/HG-KDA-factory-output-v3-1.result.json` | 0.646243× | [1c58904695fb](../../experiments/lineage/20261117/source-evidence/sources/1c58904695fb7f9514d4a3df5a03fa4542eef933f4e356429ddbe983e0f75605.py) |
| 候选 | `threeway-20261030/results/native_group_norm/current-recheck/formal/HG-KDA-current-1.result.json` | 0.669967× | [1a111c6d965f](../../experiments/lineage/20261117/source-evidence/sources/1a111c6d965ff3b761967cb625964ad46da6ca0119138585ff8b47a8cbff1dc9.py) |
| 候选 | `threeway-20261030/proposals/native_group_norm/compact-contiguous-host-v2/assessment/formal/HG-KDA-compact-contiguous-host-v2-1.result.json` | RUNTIME_ERROR | [7469a2e3e9f5](../../experiments/lineage/20261117/source-evidence/sources/7469a2e3e9f51f22239cef0bd98729516e3f79a7676a0c68490fc6a92ad658d3.py) |
| 候选／不符合当前约束 | `threeway-20261030/proposals/native_group_norm/compact-contiguous-host-v3/assessment/formal/HG-KDA-compact-contiguous-host-v3-1.result.json` | 0.812464× | [e42a540dd43c](../../experiments/lineage/20261117/source-evidence/sources/e42a540dd43c40f7f39ec86a538238878b350bf0b2d519640ed2608acdab423d.py) |
| 厂商参考 | `threeway-20261030/references/native_group_norm/formal/CANN-async-python-1.result.json` | 0.110543× | [afeea8f1ec46](../../experiments/lineage/20261117/source-evidence/sources/afeea8f1ec465e0c4fc9083a92f33d98ebd718feb0ee7172da759e97587b0b6d.py) |
| 厂商参考 | `threeway-20261030/references/native_group_norm/cpp-v1/formal/CANN-cpp-1.result.json` | PARTIAL_PASS | [38db54409afb](../../experiments/lineage/20261117/source-evidence/sources/38db54409afb90f5727734a5d0d957c9a9cd5bc3837450f4a720629a3e182f46.py) |
| 厂商参考 | `threeway-20261030/references/native_group_norm/cpp-v3/formal/CANN-cpp-1.result.json` | 0.774744× | [e2d7e4062e92](../../experiments/lineage/20261117/source-evidence/sources/e2d7e4062e926d2ca820adeeeec51d0e5d13df60d47fb905d7d894f897b628ce.py) |
| 候选／不符合当前约束 | `threeway-20261030/final-confirmation/native_group_norm/e42a540dd43c/HG-KDA-final-confirm-1.result.json` | 0.814867× | [e42a540dd43c](../../experiments/lineage/20261117/source-evidence/sources/e42a540dd43c40f7f39ec86a538238878b350bf0b2d519640ed2608acdab423d.py) |
| 厂商参考 | `cann-priority-20261102/references/native_group_norm/formal/CANN-1.result.json` | 0.801803× | [e2d7e4062e92](../../experiments/lineage/20261117/source-evidence/sources/e2d7e4062e926d2ca820adeeeec51d0e5d13df60d47fb905d7d894f897b628ce.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/comparisons/native_group_norm/starting/HG-KDA-start-1.result.json` | 1.023864× | [e42a540dd43c](../../experiments/lineage/20261117/source-evidence/sources/e42a540dd43c40f7f39ec86a538238878b350bf0b2d519640ed2608acdab423d.py) |
| 候选 | `strict-triton-20261102/candidates/native_group_norm/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.782536× | [d3fe5725c978](../../experiments/lineage/20261117/source-evidence/sources/d3fe5725c97805090f88b1c81705d1c3545c3d98017f3de780f95f0660545fa7.py) |
| 候选 | `strict-triton-20261102/candidates/native_group_norm/restored-v1/CANN/strict-CANN-1.result.json` | 0.986749× | [d3fe5725c978](../../experiments/lineage/20261117/source-evidence/sources/d3fe5725c97805090f88b1c81705d1c3545c3d98017f3de780f95f0660545fa7.py) |
| 候选 | `independent-regression-20261109-v3/operators/native_group_norm/evaluation/result.json` | 1.003824× | [d3fe5725c978](../../experiments/lineage/20261117/source-evidence/sources/d3fe5725c97805090f88b1c81705d1c3545c3d98017f3de780f95f0660545fa7.py) |
| 候选 | `codex-sample-retest-20261116/native_group_norm/formal-d3fe5725c978/evaluation/result.json` | 1.015807× | [d3fe5725c978](../../experiments/lineage/20261117/source-evidence/sources/d3fe5725c97805090f88b1c81705d1c3545c3d98017f3de780f95f0660545fa7.py) |

## 交付终点

- [选用源码](../../../operators/native_group_norm/versions/independent_strict/retained/kernel.py)：`d3fe5725c97805090f88b1c81705d1c3545c3d98017f3de780f95f0660545fa7`。
- [保留版完整结果](../../../operators/native_group_norm/evidence/retained/evaluation/result.json)：**1.003824× CANN，27/27**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

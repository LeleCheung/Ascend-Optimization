# nan_to_num_

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/nan_to_num_.py)；SHA `7f1caadc9745c8404d01f6b7c445f66689f69304a7dbe9c5995e1072ce32bad8`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/94dc19d429a2636d4e08f744746e43b99dd087466ca0fe8e476d74aad226b3ed.py)；原始模块AST一致，追加入口：`run = nan_to_num_`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/78e2cc4043342d42c1f9ed64948e5965fe172f06ac24798859884766f74f8e8b.json)：**PARTIAL_PASS**；27/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/18d9dccd05a736ee9cc3331e1bb1716be8f918cf16e4b7355710f94becda0f56.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/nan_to_num_/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 8条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 12份候选测量、7份请求源码；详细位置列在下表 |
| 交付终点 | 0.863208× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/nan_to_num_/baseline-fixed/FlagGems-1.result.json` | PARTIAL_PASS | [94dc19d429a2](../../experiments/lineage/20261117/source-evidence/sources/94dc19d429a2636d4e08f744746e43b99dd087466ca0fe8e476d74aad226b3ed.py) |
| 候选 | `next25-20261029-v2/results/nan_to_num_/search-1/independent/HG-KDA-1.result.json` | 0.750006× | [bc15a97302db](../../experiments/lineage/20261117/source-evidence/sources/bc15a97302db628a0ab002043b597d4247fccca8fdd112a37018e12c6123ee8d.py) |
| 候选 | `next25-20261029-v2/proposals/nan_to_num_/parameter-semantics-v1/assessment/formal/HG-KDA-parameter-semantics-v1-1.result.json` | 0.758303× | [3eaa6d6cab35](../../experiments/lineage/20261117/source-evidence/sources/3eaa6d6cab35236fc016ea30737d3cb7fc2d1cbb2378510839d9d4fa58fcad8a.py) |
| 候选 | `next25-20261029-v2/proposals/nan_to_num_/host-metadata-v2/assessment/formal/HG-KDA-host-metadata-v2-1.result.json` | 0.739360× | [65f7efc5fedd](../../experiments/lineage/20261117/source-evidence/sources/65f7efc5fedddb4f97c2e97250be05ac88667c876b1aef8d8b404310c7af51c6.py) |
| 候选 | `next25-20261029-v2/proposals/nan_to_num_/stage-nomask-v3/assessment/formal/HG-KDA-stage-nomask-v3-1.result.json` | 0.727475× | [abb80b0398c4](../../experiments/lineage/20261117/source-evidence/sources/abb80b0398c481c74d5e2283d21587b7fa55114b59b54d17178456b52102640a.py) |
| 候选 | `threeway-20261030/results/nan_to_num_/current-recheck/formal/HG-KDA-current-1.result.json` | 0.765493× | [3eaa6d6cab35](../../experiments/lineage/20261117/source-evidence/sources/3eaa6d6cab35236fc016ea30737d3cb7fc2d1cbb2378510839d9d4fa58fcad8a.py) |
| 候选 | `threeway-20261030/results/nan_to_num_/search-1/independent/HG-KDA-1.result.json` | 0.830459× | [8db3017f4140](../../experiments/lineage/20261117/source-evidence/sources/8db3017f41401d6eaa16584fca0ae77b27022244ceeea14e3afbf0e40c4000c3.py) |
| 候选 | `threeway-20261030/proposals/nan_to_num_/compact-default-dispatch-v1/assessment/formal/HG-KDA-compact-default-dispatch-v1-1.result.json` | 0.776617× | [7d653eba8d6c](../../experiments/lineage/20261117/source-evidence/sources/7d653eba8d6c8db8b7aef5179622f59337422e96efd6975785a1a6f609c1fa97.py) |
| 候选 | `threeway-20261030/proposals/nan_to_num_/persistent-streaming-v2/assessment/formal/HG-KDA-persistent-streaming-v2-1.result.json` | 0.704816× | [cd892a23c0ba](../../experiments/lineage/20261117/source-evidence/sources/cd892a23c0bacf996ddbd6995fec44a1d760e70691913f981ff7c40cd5b2507e.py) |
| 候选 | `threeway-20261030/final-confirmation/nan_to_num_/8db3017f4140/HG-KDA-final-confirm-1.result.json` | 0.902967× | [8db3017f4140](../../experiments/lineage/20261117/source-evidence/sources/8db3017f41401d6eaa16584fca0ae77b27022244ceeea14e3afbf0e40c4000c3.py) |
| 厂商参考 | `cann-priority-20261102/references/nan_to_num_/formal/CANN-1.result.json` | 0.880773× | [d3604ec66b9d](../../experiments/lineage/20261117/source-evidence/sources/d3604ec66b9da4013aedb024e13a2d2cf3c285e80b5f36b37459ca15955f7e74.py) |
| 候选 | `cann-priority-20261102/comparisons/nan_to_num_/starting/HG-KDA-start-1.result.json` | 1.100273× | [8db3017f4140](../../experiments/lineage/20261117/source-evidence/sources/8db3017f41401d6eaa16584fca0ae77b27022244ceeea14e3afbf0e40c4000c3.py) |
| 候选 | `strict-triton-20261102/candidates/nan_to_num_/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.876083× | [8db3017f4140](../../experiments/lineage/20261117/source-evidence/sources/8db3017f41401d6eaa16584fca0ae77b27022244ceeea14e3afbf0e40c4000c3.py) |
| 候选 | `independent-regression-20261109-v3/operators/nan_to_num_/evaluation/result.json` | 0.863208× | [8db3017f4140](../../experiments/lineage/20261117/source-evidence/sources/8db3017f41401d6eaa16584fca0ae77b27022244ceeea14e3afbf0e40c4000c3.py) |

## 交付终点

- [选用源码](../../../operators/nan_to_num_/versions/independent_strict/retained/kernel.py)：`8db3017f41401d6eaa16584fca0ae77b27022244ceeea14e3afbf0e40c4000c3`。
- [保留版完整结果](../../../operators/nan_to_num_/evidence/retained/evaluation/result.json)：**0.863208× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

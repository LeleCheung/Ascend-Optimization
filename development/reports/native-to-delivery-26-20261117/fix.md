# fix

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/fix.py)；SHA `d6a5f630499bb6b916a02063edae1e916743232ee4a56541944835200ee6602a`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/7b979aaf092bf26edabd4878a3d1750df86824b9bd63f4503475111f76c234e7.py)；原始模块AST一致，追加入口：`run = fix`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/27bcc8792bfc02aad09189efc24388169c22ee369450c231542c7d382fde3b2a.json)：**RUNTIME_ERROR**；0/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/c25fb1985fa1895fabe7f2ab6505b80dee714e850dca01570fb4fe5691a4839b.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/fix/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 6条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 26份候选测量、15份请求源码；详细位置列在下表 |
| 交付终点 | 0.969392× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/fix/baseline-fixed/FlagGems-1.result.json` | RUNTIME_ERROR | [7b979aaf092b](../../experiments/lineage/20261117/source-evidence/sources/7b979aaf092bf26edabd4878a3d1750df86824b9bd63f4503475111f76c234e7.py) |
| 候选 | `next25-20261029-v2/results/fix/search-2/independent/HG-KDA-1.result.json` | 0.677310× | [b5ca74040671](../../experiments/lineage/20261117/source-evidence/sources/b5ca7404067161d114e3d8e01a6a04a87ec21ea052b83b76936ff345fef911a3.py) |
| 候选 | `next25-20261029-v2/proposals/fix/ieee-trunc-v1/assessment/formal/HG-KDA-ieee-trunc-v1-1.result.json` | RUNTIME_ERROR | [7044d2e3d65d](../../experiments/lineage/20261117/source-evidence/sources/7044d2e3d65d8251650fb95f2f677330d93c0a0e0b02cea1943b866a922c0342.py) |
| 候选 | `next25-20261029-v2/proposals/fix/ieee-trunc-v2/assessment/formal/HG-KDA-ieee-trunc-v2-1.result.json` | 0.052955× | [0136ad6df26a](../../experiments/lineage/20261117/source-evidence/sources/0136ad6df26a7bfc9e5b3f4af9bca429cf37ba193cb0695d07fa3491ab5a4c77.py) |
| 候选 | `next25-20261029-v2/proposals/fix/vector-trunc-v3/assessment/formal/HG-KDA-vector-trunc-v3-1.result.json` | 0.315523× | [c2c5b4a66280](../../experiments/lineage/20261117/source-evidence/sources/c2c5b4a66280bdc6d4d39d2a7a27c0c208c5cedb7a5c3c8b9e390ccb12704cfb.py) |
| 候选 | `next25-20261029-v2/proposals/fix/host-metadata-v4/assessment/formal/HG-KDA-host-metadata-v4-1.result.json` | 0.475696× | [38b76c950f8e](../../experiments/lineage/20261117/source-evidence/sources/38b76c950f8e2a5f2ea51a94d1c38c1ffe2d462c3f8bef8eaabd0bbd33a79534.py) |
| 候选 | `remaining-nine-20261030/results/fix/current-recheck/formal/HG-KDA-current-1.result.json` | 0.660644× | [b5ca74040671](../../experiments/lineage/20261117/source-evidence/sources/b5ca7404067161d114e3d8e01a6a04a87ec21ea052b83b76936ff345fef911a3.py) |
| 候选 | `remaining-nine-20261030/proposals/fix/ascendc-python-allocation-v5/assessment/formal/HG-KDA-ascendc-python-allocation-v5-1.result.json` | 0.892393× | [b728eedf25d1](../../experiments/lineage/20261117/source-evidence/sources/b728eedf25d14f1ffe09d204d0fc8265cf5642f05d4900f0837a553d714d5c8d.py) |
| 候选 | `remaining-nine-20261030/final-confirmation/fix/b728eedf25d1/HG-KDA-final-confirm-1.result.json` | 0.803090× | [b728eedf25d1](../../experiments/lineage/20261117/source-evidence/sources/b728eedf25d14f1ffe09d204d0fc8265cf5642f05d4900f0837a553d714d5c8d.py) |
| 候选 | `remaining-nine-20261030/borderline-confirmation/fix/b728eedf25d1/HG-KDA-borderline-third-1.result.json` | 0.766409× | [b728eedf25d1](../../experiments/lineage/20261117/source-evidence/sources/b728eedf25d14f1ffe09d204d0fc8265cf5642f05d4900f0837a553d714d5c8d.py) |
| 候选 | `remaining-final-20261031/proposals/fix/event-large-v6/assessment/formal/HG-KDA-event-large-v6-1.result.json` | 0.817011× | [11493f053373](../../experiments/lineage/20261117/source-evidence/sources/11493f0533730e5e4b1471fdef1b5f26005b6512543309d2913c6175e5ef533a.py) |
| 候选 | `remaining-final-20261031/proposals/fix/sync-large-v7/assessment/formal/HG-KDA-sync-large-v7-1.result.json` | 0.752655× | [ee7c5ec89460](../../experiments/lineage/20261117/source-evidence/sources/ee7c5ec8946096123089d9be82616608dbc99b1af0080ebc62bad6a18fd9f90c.py) |
| 候选 | `remaining-final-20261031/proposals/fix/prefetch-large-v8/assessment/formal/HG-KDA-prefetch-large-v8-1.result.json` | 0.679951× | [c7d68db49a5d](../../experiments/lineage/20261117/source-evidence/sources/c7d68db49a5d1ef19c594244d667c0cacf62619eeea1c68b42f76a777aea96e0.py) |
| 候选 | `remaining-final-20261031/proposals/fix/host-python-allocation-v9/assessment/formal/HG-KDA-host-python-allocation-v9-1.result.json` | 0.690381× | [df005445544a](../../experiments/lineage/20261117/source-evidence/sources/df005445544a238be4cbbb104acc501396bd3cb3892d1c2d7cdd4b1ea46fef9a.py) |
| 候选 | `remaining-final-20261031/proposals/fix/contiguous-cores-v10/assessment/formal/HG-KDA-contiguous-cores-v10-1.result.json` | 0.759593× | [ac6abad35c23](../../experiments/lineage/20261117/source-evidence/sources/ac6abad35c23a0c3e5dd5a11164d3be6d0b5230e862dce373ced97c868fcdd96.py) |
| 候选 | `remaining-final-20261031/proposals/fix/direct-host-v11/assessment/formal/HG-KDA-direct-host-v11-1.result.json` | 0.445977× | [78138018377f](../../experiments/lineage/20261117/source-evidence/sources/78138018377f7c252304b59c2bcbf56330c8ffad53ab9f789ab14254bfffd8aa.py) |
| 候选／不符合当前约束 | `remaining-final-20261031/proposals/fix/queue-gil-v12/assessment/formal/HG-KDA-queue-gil-v12-1.result.json` | 0.848054× | [4e8657ec60e0](../../experiments/lineage/20261117/source-evidence/sources/4e8657ec60e0b133cff6a5a54bd27ed58621aa99b1b3a4ff911b6ffe105b6f09.py) |
| 候选／不符合当前约束 | `remaining-final-20261031/reproductions/final-confirm-01/fix/formal/HG-KDA-reproduction-1.result.json` | 0.875187× | [4e8657ec60e0](../../experiments/lineage/20261117/source-evidence/sources/4e8657ec60e0b133cff6a5a54bd27ed58621aa99b1b3a4ff911b6ffe105b6f09.py) |
| 厂商参考 | `reference-closure-20261031/references/fix/formal/CANN-direct-1.result.json` | 0.833265× | [216244c49a49](../../experiments/lineage/20261117/source-evidence/sources/216244c49a497d51960bb826cc15b60e3fb855bac00143bbdf96a103f1257a6a.py) |
| 候选／不符合当前约束 | `reference-closure-20261031/comparisons/fix/CANN-direct/HG-KDA-vs-CANN-1.result.json` | 1.211972× | [4e8657ec60e0](../../experiments/lineage/20261117/source-evidence/sources/4e8657ec60e0b133cff6a5a54bd27ed58621aa99b1b3a4ff911b6ffe105b6f09.py) |
| 厂商参考 | `cann-priority-20261102/references/fix/formal/CANN-1.result.json` | 0.762426× | [216244c49a49](../../experiments/lineage/20261117/source-evidence/sources/216244c49a497d51960bb826cc15b60e3fb855bac00143bbdf96a103f1257a6a.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/comparisons/fix/starting/HG-KDA-start-1.result.json` | 1.196596× | [4e8657ec60e0](../../experiments/lineage/20261117/source-evidence/sources/4e8657ec60e0b133cff6a5a54bd27ed58621aa99b1b3a4ff911b6ffe105b6f09.py) |
| 候选 | `strict-triton-20261102/candidates/fix/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.689020× | [2094cb35196d](../../experiments/lineage/20261117/source-evidence/sources/2094cb35196d887816e717f56da6587536c9b9159e6287bc7716768bdfa8fa17.py) |
| 候选 | `strict-triton-20261102/candidates/fix/restored-v1/CANN/strict-CANN-1.result.json` | 0.788997× | [2094cb35196d](../../experiments/lineage/20261117/source-evidence/sources/2094cb35196d887816e717f56da6587536c9b9159e6287bc7716768bdfa8fa17.py) |
| 候选 | `strict-night-20261102/candidates/fix/b8fe66ad0725e7e8/pytorch/strict-PyTorch-1.result.json` | 0.735013× | [b8fe66ad0725](../../experiments/lineage/20261117/source-evidence/sources/b8fe66ad0725e7e8920d07a8b44f87b1241d0032d46325e4bfeec54fa199230d.py) |
| 候选 | `strict-night-20261102/candidates/fix/b8fe66ad0725e7e8/formal-reference/result.json` | 1.014276× | 请求源码未定位 |
| 候选 | `strict-night-20261102/candidates/fix/b8fe66ad0725e7e8/formal-reference/measurement-0/result.json` | 1.014276× | 请求源码未定位 |
| 候选 | `strict-night-20261102/review-preview/fix/round-0004/result.json` | 1.014276× | 请求源码未定位 |
| 候选 | `independent-regression-20261109-v3/operators/fix/evaluation/result.json` | 0.969392× | [b8fe66ad0725](../../experiments/lineage/20261117/source-evidence/sources/b8fe66ad0725e7e8920d07a8b44f87b1241d0032d46325e4bfeec54fa199230d.py) |

## 交付终点

- [选用源码](../../../operators/fix/versions/independent_strict/retained/kernel.py)：`b8fe66ad0725e7e8920d07a8b44f87b1241d0032d46325e4bfeec54fa199230d`。
- [保留版完整结果](../../../operators/fix/evidence/retained/evaluation/result.json)：**0.969392× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

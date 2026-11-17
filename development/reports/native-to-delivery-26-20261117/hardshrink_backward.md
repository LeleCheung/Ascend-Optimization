# hardshrink_backward

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/hardshrink_backward.py)；SHA `76b98e253b7a22bf76fd0739c96ec317389170499205b24bcafea11486096aeb`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/6757e24e3f09bea1d22799fd02e7583888989df5f0fc16a0b9acaabcc5c0b09e.py)；原始模块AST一致，追加入口：`run = hardshrink_backward`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/a6fa0041ccffa9569392ace74bef71d8dc33525e54a4a3797cf38f567a775231.json)：**0.154×**；30/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/a01e3872524f5a52e1c8eba1ef3f98aec77cd54889da5796217da77a7b945a67.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/hardshrink_backward/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 6条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 14份候选测量、8份请求源码；详细位置列在下表 |
| 交付终点 | 0.998586× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/hardshrink_backward/baseline-fixed/FlagGems-1.result.json` | 0.153536× | [6757e24e3f09](../../experiments/lineage/20261117/source-evidence/sources/6757e24e3f09bea1d22799fd02e7583888989df5f0fc16a0b9acaabcc5c0b09e.py) |
| 候选 | `next25-20261029-v2/results/hardshrink_backward/search-1/independent/HG-KDA-1.result.json` | 1.042334× | [aefda66ca6db](../../experiments/lineage/20261117/source-evidence/sources/aefda66ca6db764a4cc13f441f74e3c831e11302c3c1a6ccf622338d9f3c6f66.py) |
| 候选 | `next25-20261029-v2/proposals/hardshrink_backward/nonfinite-gradient-v3/assessment/formal/HG-KDA-nonfinite-gradient-v3-1.result.json` | PARTIAL_PASS | [e2d003957224](../../experiments/lineage/20261117/source-evidence/sources/e2d00395722499021869bfc2091c9b9212654548fe83477772f13f0cc7e72330.py) |
| 候选 | `next25-20261029-v2/proposals/hardshrink_backward/strided-buffer-v6/assessment/formal/HG-KDA-strided-buffer-v6-1.result.json` | 0.867557× | [0c4f71d05c28](../../experiments/lineage/20261117/source-evidence/sources/0c4f71d05c287f675bf69cf32dbf6f3c9edc2b15ca280a9036acd499c1c53fa0.py) |
| 候选 | `threeway-20261030/results/hardshrink_backward/current-recheck/formal/HG-KDA-current-1.result.json` | 0.900779× | [0c4f71d05c28](../../experiments/lineage/20261117/source-evidence/sources/0c4f71d05c287f675bf69cf32dbf6f3c9edc2b15ca280a9036acd499c1c53fa0.py) |
| 厂商参考 | `cann-priority-20261102/references/hardshrink_backward/formal/CANN-1.result.json` | 0.970205× | [7b127915b93a](../../experiments/lineage/20261117/source-evidence/sources/7b127915b93adfc096e50c25d82df1709ec7d25276c52fe7f6c49b7caa5ec31b.py) |
| 候选 | `cann-priority-20261102/comparisons/hardshrink_backward/starting/HG-KDA-start-1.result.json` | 0.893885× | [0c4f71d05c28](../../experiments/lineage/20261117/source-evidence/sources/0c4f71d05c287f675bf69cf32dbf6f3c9edc2b15ca280a9036acd499c1c53fa0.py) |
| 候选 | `strict-triton-20261102/candidates/hardshrink_backward/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.851027× | [4d34f65f9716](../../experiments/lineage/20261117/source-evidence/sources/4d34f65f9716dd0c2a749bb24eeebae367d06a44d7b77ea8b510e03ca8435ab4.py) |
| 候选 | `strict-triton-20261102/candidates/hardshrink_backward/restored-v1/CANN/strict-CANN-1.result.json` | 0.786597× | [4d34f65f9716](../../experiments/lineage/20261117/source-evidence/sources/4d34f65f9716dd0c2a749bb24eeebae367d06a44d7b77ea8b510e03ca8435ab4.py) |
| 候选 | `strict-night-20261102/candidates/hardshrink_backward/79019ab0882af009/pytorch/strict-PyTorch-1.result.json` | 0.960787× | [79019ab0882a](../../experiments/lineage/20261117/source-evidence/sources/79019ab0882af009a992be51996d070b6fd4046214fcb1c07ced35f14310aef8.py) |
| 候选 | `strict-night-20261102/candidates/hardshrink_backward/dae5719e4b4be1d4/pytorch/strict-PyTorch-1.result.json` | 1.039747× | [dae5719e4b4b](../../experiments/lineage/20261117/source-evidence/sources/dae5719e4b4be1d4320c2904bd427f98c744e1eadbbf6bb05e67611d348cecec.py) |
| 候选 | `strict-night-20261102/candidates/hardshrink_backward/53b513f0746b3d7c/pytorch/strict-PyTorch-1.result.json` | 1.069336× | [53b513f0746b](../../experiments/lineage/20261117/source-evidence/sources/53b513f0746b3d7c347a3ec6990301978e356a040ec19c371d5f91cff08beddb.py) |
| 候选 | `strict-night-20261102/proposals/hardshrink_backward/codex-npu-masked-gradient-v1/CANN/Codex-strict-CANN-1.result.json` | 1.042200× | [20432054a1f4](../../experiments/lineage/20261117/source-evidence/sources/20432054a1f412574fbdf9f9522d955e26785d7cae14abd2c95a7fbbe20610dd.py) |
| 候选 | `strict-night-20261102/proposals/hardshrink_backward/codex-npu-masked-gradient-v1/pytorch/strict-PyTorch-1.result.json` | 1.036825× | [20432054a1f4](../../experiments/lineage/20261117/source-evidence/sources/20432054a1f412574fbdf9f9522d955e26785d7cae14abd2c95a7fbbe20610dd.py) |
| 候选 | `strict-night-20261102/proposals/hardshrink_backward/codex-npu-masked-gradient-v1/formal-reference/Codex-strict-CANN-1.result.json` | 1.042200× | [20432054a1f4](../../experiments/lineage/20261117/source-evidence/sources/20432054a1f412574fbdf9f9522d955e26785d7cae14abd2c95a7fbbe20610dd.py) |
| 候选 | `independent-regression-20261109-v3/operators/hardshrink_backward/evaluation/result.json` | 0.998586× | [20432054a1f4](../../experiments/lineage/20261117/source-evidence/sources/20432054a1f412574fbdf9f9522d955e26785d7cae14abd2c95a7fbbe20610dd.py) |

## 交付终点

- [选用源码](../../../operators/hardshrink_backward/versions/independent_strict/retained/kernel.py)：`20432054a1f412574fbdf9f9522d955e26785d7cae14abd2c95a7fbbe20610dd`。
- [保留版完整结果](../../../operators/hardshrink_backward/evidence/retained/evaluation/result.json)：**0.998586× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

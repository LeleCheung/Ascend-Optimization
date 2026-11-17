# cross_out

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/cross.py)；SHA `882cf82396e05d3fc0b85c91c2ecb87512d49cba3ff5a8e5f0827c0f6f0416d7`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/05a97b9b9f51d549085dc8358dc4b65d29a39d1543856b9e6c1a39b1a12e30ea.py)；原始模块AST一致，追加入口：`run = cross_out`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/ed8f465ebcc2490ff58082f80a523fbbf43af46932f2b7247c0856903cc3368b.json)：**0.136×**；21/21。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/c349c670775a778406dbe4b1f68bc436a64c25d74ce77d52bc43e3d71f1d8bda.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/cross_out/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 6条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 15份候选测量、9份请求源码；详细位置列在下表 |
| 交付终点 | 3.868442× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/cross_out/baseline-fixed/FlagGems-1.result.json` | 0.135717× | [05a97b9b9f51](../../experiments/lineage/20261117/source-evidence/sources/05a97b9b9f51d549085dc8358dc4b65d29a39d1543856b9e6c1a39b1a12e30ea.py) |
| 候选 | `next25-20261029-v2/results/cross_out/search-1/independent/HG-KDA-1.result.json` | 0.498462× | [787d3d69851e](../../experiments/lineage/20261117/source-evidence/sources/787d3d69851e5b570f3ccecc6b5729c44380469cf0d0f7a2073863338718d433.py) |
| 候选 | `next25-20261029-v2/proposals/cross_out/general-layout-launch-v1/assessment/formal/HG-KDA-general-layout-launch-v1-1.result.json` | 0.752797× | [abdd4f8bc5b5](../../experiments/lineage/20261117/source-evidence/sources/abdd4f8bc5b50716bbbcc712250907aa0634c349cfe07cdf36fd61bb4ac6c0a9.py) |
| 候选 | `next25-20261029-v2/proposals/cross_out/ub-permutation-v2/assessment/formal/HG-KDA-ub-permutation-v2-1.result.json` | 0.769058× | [11cc5021ead7](../../experiments/lineage/20261117/source-evidence/sources/11cc5021ead7110f45ff8b216930e394f229508ff9c2e008fe3764c166759a59.py) |
| 候选 | `threeway-20261030/results/cross_out/current-recheck/formal/HG-KDA-current-1.result.json` | 0.759700× | [11cc5021ead7](../../experiments/lineage/20261117/source-evidence/sources/11cc5021ead7110f45ff8b216930e394f229508ff9c2e008fe3764c166759a59.py) |
| 候选 | `threeway-20261030/results/cross_out/search-2/independent/HG-KDA-1.result.json` | 5.278153× | [006fc48a74b2](../../experiments/lineage/20261117/source-evidence/sources/006fc48a74b2318424c044227b168b685a3fbf1d3f7185a053b42cd66435b822.py) |
| 候选 | `threeway-20261030/proposals/cross_out/coarser-lowp-tiles-v1/assessment/formal/HG-KDA-coarser-lowp-tiles-v1-1.result.json` | 0.732144× | [1fad3c45ee1d](../../experiments/lineage/20261117/source-evidence/sources/1fad3c45ee1d810081a1672411cd0d7dc83920be6f24e6876d3e46317e0c3cb5.py) |
| 候选 | `threeway-20261030/proposals/cross_out/direct-vector-address-v2/assessment/formal/HG-KDA-direct-vector-address-v2-1.result.json` | 0.129966× | [ee105faa0b84](../../experiments/lineage/20261117/source-evidence/sources/ee105faa0b84d5e6444ecba02c6cda913a5a8cb76b6730f3ee965d4e7e2368a9.py) |
| 候选 | `threeway-20261030/proposals/cross_out/matrix-split-join-v3/assessment/formal/HG-KDA-matrix-split-join-v3-1.result.json` | 0.107354× | [09141c577cbe](../../experiments/lineage/20261117/source-evidence/sources/09141c577cbe8c7ef0677c22e5072e6972ea0c46e34f2ddfb25f8721eeaba9ef.py) |
| 候选／不符合当前约束 | `threeway-20261030/proposals/cross_out/validated-shifted-loads-v5/assessment/formal/HG-KDA-validated-shifted-loads-v5-1.result.json` | 3.856639× | [e03d95b66da1](../../experiments/lineage/20261117/source-evidence/sources/e03d95b66da130ad6b16ecbd8a959265e848c2c46e1c3cf48216ea0611cbdc66.py) |
| 候选／不符合当前约束 | `threeway-20261030/final-confirmation/cross_out/e03d95b66da1/HG-KDA-final-confirm-1.result.json` | 3.926130× | [e03d95b66da1](../../experiments/lineage/20261117/source-evidence/sources/e03d95b66da130ad6b16ecbd8a959265e848c2c46e1c3cf48216ea0611cbdc66.py) |
| 厂商参考 | `cann-priority-20261102/references/cross_out/formal/CANN-1.result.json` | 0.968451× | [59d1dcd9baec](../../experiments/lineage/20261117/source-evidence/sources/59d1dcd9baec1ddf668d38986afd2ea7758ba85ecf5b7685cc4a4b6aa20723bf.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/comparisons/cross_out/starting/HG-KDA-start-1.result.json` | 4.056870× | [e03d95b66da1](../../experiments/lineage/20261117/source-evidence/sources/e03d95b66da130ad6b16ecbd8a959265e848c2c46e1c3cf48216ea0611cbdc66.py) |
| 候选 | `strict-triton-20261102/candidates/cross_out/restored-v1/pytorch/strict-PyTorch-1.result.json` | 3.757029× | [bf9ab7a46e2a](../../experiments/lineage/20261117/source-evidence/sources/bf9ab7a46e2ac5389e9f617ba31d6c31ae3017c42456d29dbd3e26e34fbeed62.py) |
| 候选 | `strict-triton-20261102/candidates/cross_out/restored-v1/CANN/strict-CANN-1.result.json` | 4.089549× | [bf9ab7a46e2a](../../experiments/lineage/20261117/source-evidence/sources/bf9ab7a46e2ac5389e9f617ba31d6c31ae3017c42456d29dbd3e26e34fbeed62.py) |
| 候选 | `independent-regression-20261109-v3/operators/cross_out/evaluation/result.json` | 3.868442× | [bf9ab7a46e2a](../../experiments/lineage/20261117/source-evidence/sources/bf9ab7a46e2ac5389e9f617ba31d6c31ae3017c42456d29dbd3e26e34fbeed62.py) |
| 候选 | `codex-sample-retest-20261116/cross_out/formal-bf9ab7a46e2a/evaluation/result.json` | 4.026976× | [bf9ab7a46e2a](../../experiments/lineage/20261117/source-evidence/sources/bf9ab7a46e2ac5389e9f617ba31d6c31ae3017c42456d29dbd3e26e34fbeed62.py) |

## 交付终点

- [选用源码](../../../operators/cross_out/versions/independent_strict/retained/kernel.py)：`bf9ab7a46e2ac5389e9f617ba31d6c31ae3017c42456d29dbd3e26e34fbeed62`。
- [保留版完整结果](../../../operators/cross_out/evidence/retained/evaluation/result.json)：**3.868442× CANN，21/21**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

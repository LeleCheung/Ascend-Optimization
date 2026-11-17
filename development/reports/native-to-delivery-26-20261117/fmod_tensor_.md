# fmod_tensor_

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/fmod.py)；SHA `bbb943dda3882c588a6a3412e0fe276efcb6be016f3c89aa34ddde1bbd643777`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/e55c3b233716c66d15605b73b76604be3511a283ec6136b8119a4710fa183b10.py)；原始模块AST一致，追加入口：`run = fmod_tensor_`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/b7ee2e61c06093a9cc98f4e9695f3313e4676b91b93efdc8fc584a10b95f22a0.json)：**RUNTIME_ERROR**；0/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/9ec0a5a2b8fee0f2c5c8dd2e0339c12bc1c16446157d79c9d565cb8dbff54245.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/fmod_tensor_/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 2条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 16份候选测量、9份请求源码；详细位置列在下表 |
| 交付终点 | 0.840686× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/fmod_tensor_/baseline-fixed/FlagGems-1.result.json` | RUNTIME_ERROR | [e55c3b233716](../../experiments/lineage/20261117/source-evidence/sources/e55c3b233716c66d15605b73b76604be3511a283ec6136b8119a4710fa183b10.py) |
| 候选 | `next25-20261029-v2/results/fmod_tensor_/search-1/independent/HG-KDA-1.result.json` | 0.727490× | [aa91395e63ce](../../experiments/lineage/20261117/source-evidence/sources/aa91395e63ce510f5f47f4199b5717d588ac140992c60a809c7f302ee105af51.py) |
| 候选 | `next25-20261029-v2/proposals/fmod_tensor_/fp32-remainder-launch-v1/assessment/formal/HG-KDA-fp32-remainder-launch-v1-1.result.json` | RUNTIME_ERROR | [4ed844f73d77](../../experiments/lineage/20261117/source-evidence/sources/4ed844f73d77900c2cead0d99b6994a3fd67f88b3a71634fe291ce424c37f844.py) |
| 候选 | `next25-20261029-v2/proposals/fmod_tensor_/signed-bitpatterns-v2/assessment/formal/HG-KDA-signed-bitpatterns-v2-1.result.json` | 0.197119× | [2dade700e939](../../experiments/lineage/20261117/source-evidence/sources/2dade700e939c3736fe3a86bd484647e0c4c95576e6e5d5f10167dd3632887b4.py) |
| 候选 | `next25-20261029-v2/proposals/fmod_tensor_/npu-quotient-contract-v3/assessment/formal/HG-KDA-npu-quotient-contract-v3-1.result.json` | 0.652928× | [ec1a8d87f050](../../experiments/lineage/20261117/source-evidence/sources/ec1a8d87f05098e0d384ac609e68784748c84d982b261b3de8fc5feaae07c955.py) |
| 候选 | `remaining-nine-20261030/results/fmod_tensor_/current-recheck/formal/HG-KDA-current-1.result.json` | 0.712215× | [aa91395e63ce](../../experiments/lineage/20261117/source-evidence/sources/aa91395e63ce510f5f47f4199b5717d588ac140992c60a809c7f302ee105af51.py) |
| 候选 | `remaining-nine-20261030/proposals/fmod_tensor_/two-tiles-single-buffer-v1/assessment/formal/HG-KDA-two-tiles-single-buffer-v1-1.result.json` | 0.697261× | [c3116e75b5d7](../../experiments/lineage/20261117/source-evidence/sources/c3116e75b5d7e08cd965411d4eb01fd7478813026c9687ef90735728e173c78a.py) |
| 候选 | `remaining-nine-20261030/proposals/fmod_tensor_/ascendc-div-trunc-taskqueue-v3/assessment/formal/HG-KDA-ascendc-div-trunc-taskqueue-v3-1.result.json` | 1.152559× | [d7feb361bfa7](../../experiments/lineage/20261117/source-evidence/sources/d7feb361bfa784d12380425b58d0fb99eea4c5205520b11c6245cea9b2160395.py) |
| 候选／不符合当前约束 | `remaining-nine-20261030/proposals/fmod_tensor_/ascendc-double-buffer-v4/assessment/formal/HG-KDA-ascendc-double-buffer-v4-1.result.json` | 1.155006× | [c4ed63398729](../../experiments/lineage/20261117/source-evidence/sources/c4ed633987292126c96ea1abb2b43fc793f01f5ffcb8557403bab26fdb33a49b.py) |
| 候选／不符合当前约束 | `remaining-nine-20261030/final-confirmation/fmod_tensor_/c4ed63398729/HG-KDA-final-confirm-1.result.json` | 1.335756× | [c4ed63398729](../../experiments/lineage/20261117/source-evidence/sources/c4ed633987292126c96ea1abb2b43fc793f01f5ffcb8557403bab26fdb33a49b.py) |
| 厂商参考 | `cann-priority-20261102/references/fmod_tensor_/formal/CANN-1.result.json` | 0.980862× | [677b38baeb68](../../experiments/lineage/20261117/source-evidence/sources/677b38baeb68a806eaf35d3ab0d48ac1df83d9c7aae8b741a65795d26bbca31a.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/comparisons/fmod_tensor_/starting/HG-KDA-start-1.result.json` | 1.421717× | [c4ed63398729](../../experiments/lineage/20261117/source-evidence/sources/c4ed633987292126c96ea1abb2b43fc793f01f5ffcb8557403bab26fdb33a49b.py) |
| 候选 | `strict-triton-20261102/candidates/fmod_tensor_/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.742416× | [ab6c31acb476](../../experiments/lineage/20261117/source-evidence/sources/ab6c31acb476218bc4ac4ef34873143e01d2726b080084383808fbc20c56aa82.py) |
| 候选 | `strict-triton-20261102/candidates/fmod_tensor_/restored-v1/CANN/strict-CANN-1.result.json` | 0.727239× | [ab6c31acb476](../../experiments/lineage/20261117/source-evidence/sources/ab6c31acb476218bc4ac4ef34873143e01d2726b080084383808fbc20c56aa82.py) |
| 候选 | `strict-night-20261102/proposals/fmod_tensor_/codex-streaming-fmod-v1/CANN/Codex-strict-CANN-1.result.json` | 0.818204× | [0f7bf1e0427e](../../experiments/lineage/20261117/source-evidence/sources/0f7bf1e0427e099f7820bc3bf89c7988a2f263d0d21842d9bd5e11cdc8ba3b43.py) |
| 候选 | `strict-night-20261102/proposals/fmod_tensor_/codex-streaming-fmod-v1/pytorch/strict-PyTorch-1.result.json` | 0.728679× | [0f7bf1e0427e](../../experiments/lineage/20261117/source-evidence/sources/0f7bf1e0427e099f7820bc3bf89c7988a2f263d0d21842d9bd5e11cdc8ba3b43.py) |
| 候选 | `strict-night-20261102/proposals/fmod_tensor_/codex-streaming-fmod-v1/formal-reference/Codex-strict-CANN-1.result.json` | 0.818204× | [0f7bf1e0427e](../../experiments/lineage/20261117/source-evidence/sources/0f7bf1e0427e099f7820bc3bf89c7988a2f263d0d21842d9bd5e11cdc8ba3b43.py) |
| 候选 | `independent-regression-20261109-v3/operators/fmod_tensor_/evaluation/result.json` | 0.840686× | [0f7bf1e0427e](../../experiments/lineage/20261117/source-evidence/sources/0f7bf1e0427e099f7820bc3bf89c7988a2f263d0d21842d9bd5e11cdc8ba3b43.py) |

## 交付终点

- [选用源码](../../../operators/fmod_tensor_/versions/independent_strict/retained/kernel.py)：`0f7bf1e0427e099f7820bc3bf89c7988a2f263d0d21842d9bd5e11cdc8ba3b43`。
- [保留版完整结果](../../../operators/fmod_tensor_/evidence/retained/evaluation/result.json)：**0.840686× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

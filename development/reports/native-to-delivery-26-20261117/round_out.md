# round_out

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/round.py)；SHA `e4a911fda979f3b8d8b0962606ebaa9834f804661d94486ec2335522526b3483`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/6d0736534f92f014b8333a5c8b38c2a27c33e07d6102e74a481dcf7c992de01b.py)；原始模块AST一致，追加入口：`run = round_out`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/0fc78a404da8f8e63c018dfe6fd038bc1864eb3b17aa7b10454b951a3589939c.json)：**PARTIAL_PASS**；24/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/91bf71e5d7942d85e36214dcfff5710057e6fe000e9c886e05596668a38d1f81.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/round_out/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 5条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 19份候选测量、10份请求源码；详细位置列在下表 |
| 交付终点 | 0.916454× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/round_out/baseline-fixed/FlagGems-1.result.json` | PARTIAL_PASS | [6d0736534f92](../../experiments/lineage/20261117/source-evidence/sources/6d0736534f92f014b8333a5c8b38c2a27c33e07d6102e74a481dcf7c992de01b.py) |
| 候选 | `next25-20261029-v2/results/round_out/search-1/independent/HG-KDA-1.result.json` | 0.496953× | [39dd6a26980f](../../experiments/lineage/20261117/source-evidence/sources/39dd6a26980fd70ba33f5f552b61c46a1128b72198c3d4772a89a66545b4e935.py) |
| 候选 | `next25-20261029-v2/proposals/round_out/vector-round-v1/assessment/formal/HG-KDA-vector-round-v1-1.result.json` | 0.516548× | [39dd6a26980f](../../experiments/lineage/20261117/source-evidence/sources/39dd6a26980fd70ba33f5f552b61c46a1128b72198c3d4772a89a66545b4e935.py) |
| 候选 | `next25-20261029-v2/proposals/round_out/fp32-rounding-shift-v2/assessment/formal/HG-KDA-fp32-rounding-shift-v2-1.result.json` | 0.585333× | [317a11fba15f](../../experiments/lineage/20261117/source-evidence/sources/317a11fba15f9621ad03fe41ddef7fd957b2be03e7b61c21e7eac394a4149609.py) |
| 候选 | `next25-20261029-v2/proposals/round_out/host-metadata-v3/assessment/formal/HG-KDA-host-metadata-v3-1.result.json` | 0.615481× | [d2769c44f303](../../experiments/lineage/20261117/source-evidence/sources/d2769c44f303cdab8102a70eaa01b31b8764c1141319da34794ba675aa50685a.py) |
| 候选 | `remaining-nine-20261030/results/round_out/current-recheck/formal/HG-KDA-current-1.result.json` | 0.605326× | [d2769c44f303](../../experiments/lineage/20261117/source-evidence/sources/d2769c44f303cdab8102a70eaa01b31b8764c1141319da34794ba675aa50685a.py) |
| 候选 | `remaining-nine-20261030/proposals/round_out/ascendc-taskqueue-v1/assessment/formal/HG-KDA-ascendc-taskqueue-v1-1.result.json` | 1.295087× | [1a2f082ee7d1](../../experiments/lineage/20261117/source-evidence/sources/1a2f082ee7d1b9b2c7959aae3fa19b0dc1ceec19c3eb3d5fbd18a44530e8ffd9.py) |
| 候选／不符合当前约束 | `remaining-nine-20261030/proposals/round_out/ascendc-double-buffer-v4/assessment/formal/HG-KDA-ascendc-double-buffer-v4-1.result.json` | 1.471217× | [16acb0f719e1](../../experiments/lineage/20261117/source-evidence/sources/16acb0f719e19d972c014972d25f6779f15102ab5e7c276e31844244d04e906a.py) |
| 候选／不符合当前约束 | `remaining-nine-20261030/final-confirmation/round_out/16acb0f719e1/HG-KDA-final-confirm-1.result.json` | 1.354808× | [16acb0f719e1](../../experiments/lineage/20261117/source-evidence/sources/16acb0f719e19d972c014972d25f6779f15102ab5e7c276e31844244d04e906a.py) |
| 厂商参考 | `cann-priority-20261102/references/round_out/formal/CANN-1.result.json` | 0.928758× | [196bfd69d2e5](../../experiments/lineage/20261117/source-evidence/sources/196bfd69d2e5df408bef25bf6b11255f31396ce840d66606da2d300eb2e1a0ec.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/comparisons/round_out/starting/HG-KDA-start-1.result.json` | 1.429775× | [16acb0f719e1](../../experiments/lineage/20261117/source-evidence/sources/16acb0f719e19d972c014972d25f6779f15102ab5e7c276e31844244d04e906a.py) |
| 候选 | `strict-triton-20261102/candidates/round_out/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.555818× | [9624900771b2](../../experiments/lineage/20261117/source-evidence/sources/9624900771b248e310101e9a5628ee84dcf57718ce6ad9ef6c0b99cbe3dd3cba.py) |
| 候选 | `strict-triton-20261102/candidates/round_out/restored-v1/CANN/strict-CANN-1.result.json` | 0.585391× | [9624900771b2](../../experiments/lineage/20261117/source-evidence/sources/9624900771b248e310101e9a5628ee84dcf57718ce6ad9ef6c0b99cbe3dd3cba.py) |
| 候选 | `strict-night-20261102/candidates/round_out/e67b73611626207b/pytorch/strict-PyTorch-1.result.json` | 1.068072× | [e67b73611626](../../experiments/lineage/20261117/source-evidence/sources/e67b73611626207b7e8c25247173d2d98f7be0ed0f1b68fe328653f3e2ca1819.py) |
| 候选 | `strict-night-20261102/proposals/round_out/codex-streaming-round-v1/CANN/Codex-strict-CANN-1.result.json` | 0.694970× | [a5bba7a1f69b](../../experiments/lineage/20261117/source-evidence/sources/a5bba7a1f69b51f74fd3bc7cffcd962e6619126bcd2bd1a073ae4d67a6279e30.py) |
| 候选 | `strict-night-20261102/proposals/round_out/codex-streaming-round-v1/pytorch/strict-PyTorch-1.result.json` | 0.603702× | [a5bba7a1f69b](../../experiments/lineage/20261117/source-evidence/sources/a5bba7a1f69b51f74fd3bc7cffcd962e6619126bcd2bd1a073ae4d67a6279e30.py) |
| 候选 | `strict-night-20261102/proposals/round_out/codex-streaming-round-v1/formal-reference/Codex-strict-CANN-1.result.json` | 0.694970× | [a5bba7a1f69b](../../experiments/lineage/20261117/source-evidence/sources/a5bba7a1f69b51f74fd3bc7cffcd962e6619126bcd2bd1a073ae4d67a6279e30.py) |
| 候选 | `strict-night-20261102/proposals/round_out/codex-optional-out-v3/CANN/Codex-strict-CANN-1.result.json` | 0.828560× | [0295c285bdfe](../../experiments/lineage/20261117/source-evidence/sources/0295c285bdfe868cd64548cd0ebb5529770383f7a9d6ec1197952dcb715bc06f.py) |
| 候选 | `strict-night-20261102/proposals/round_out/codex-optional-out-v3/pytorch/strict-PyTorch-1.result.json` | 0.942397× | [0295c285bdfe](../../experiments/lineage/20261117/source-evidence/sources/0295c285bdfe868cd64548cd0ebb5529770383f7a9d6ec1197952dcb715bc06f.py) |
| 候选 | `strict-night-20261102/proposals/round_out/codex-optional-out-v3/formal-reference/Codex-strict-CANN-1.result.json` | 0.828560× | [0295c285bdfe](../../experiments/lineage/20261117/source-evidence/sources/0295c285bdfe868cd64548cd0ebb5529770383f7a9d6ec1197952dcb715bc06f.py) |
| 候选 | `independent-regression-20261109-v3/operators/round_out/evaluation/result.json` | 0.916454× | [3107d0133832](../../experiments/lineage/20261117/source-evidence/sources/3107d0133832158b73bfa0b456ba4fe2ffb599e1773e14085cb686361a1efe84.py) |

## 交付终点

- [选用源码](../../../operators/round_out/versions/independent_strict/retained/kernel.py)：`3107d0133832158b73bfa0b456ba4fe2ffb599e1773e14085cb686361a1efe84`。
- [保留版完整结果](../../../operators/round_out/evidence/retained/evaluation/result.json)：**0.916454× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

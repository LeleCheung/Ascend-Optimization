# special_round_out

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/special_round.py)；SHA `da5cbc3325ebf7e7936fb97f69fa5242231b92ffd6f6bde6dc0afe4690ecbcc6`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/e9bfbdff6d4ba4df18bdef3e4b4cb5abc73fc4c756ff4cc1a48684eaf17239a0.py)；原始模块AST一致，追加入口：`run = special_round_out`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/326ebe184b238cfbf23c9bf9152e35bb05021126e22e156dcd98108d9c4d00ab.json)：**RUNTIME_ERROR**；0/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/1931fb731763bf2a11e807e19fb8d518515b226f9e0e63ed3b519d087d388faa.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/special_round_out/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 6条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 22份候选测量、12份请求源码；详细位置列在下表 |
| 交付终点 | 0.999713× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/special_round_out/baseline-fixed/FlagGems-1.result.json` | RUNTIME_ERROR | [e9bfbdff6d4b](../../experiments/lineage/20261117/source-evidence/sources/e9bfbdff6d4ba4df18bdef3e4b4cb5abc73fc4c756ff4cc1a48684eaf17239a0.py) |
| 候选 | `next25-20261029-v2/results/special_round_out/search-1/independent/HG-KDA-1.result.json` | 0.632218× | [1df2d942fa6b](../../experiments/lineage/20261117/source-evidence/sources/1df2d942fa6b321e28a709dd6313fc6bfb13c86cda3ea095edf0c5562fd4befc.py) |
| 候选 | `next25-20261029-v2/proposals/special_round_out/vector-round-v1/assessment/formal/HG-KDA-vector-round-v1-1.result.json` | 0.522103× | [ccb7b8110843](../../experiments/lineage/20261117/source-evidence/sources/ccb7b81108437cc81f08e070c64e44838e69ee21aeba66fe56401f9fcf02a95b.py) |
| 候选 | `next25-20261029-v2/proposals/special_round_out/fp32-rounding-shift-v2/assessment/formal/HG-KDA-fp32-rounding-shift-v2-1.result.json` | 0.592566× | [2bf1d321fb41](../../experiments/lineage/20261117/source-evidence/sources/2bf1d321fb41b0459f51ba0d73f025265c7ed4cc8ca55751dde77d15eaad2783.py) |
| 候选 | `next25-20261029-v2/proposals/special_round_out/host-metadata-v3/assessment/formal/HG-KDA-host-metadata-v3-1.result.json` | 0.574648× | [cc5e2b782ad2](../../experiments/lineage/20261117/source-evidence/sources/cc5e2b782ad27c838e3fde8adf9a89a264544e2461d5035d2494441668242569.py) |
| 候选 | `remaining-nine-20261030/results/special_round_out/current-recheck/formal/HG-KDA-current-1.result.json` | 0.570581× | [1df2d942fa6b](../../experiments/lineage/20261117/source-evidence/sources/1df2d942fa6b321e28a709dd6313fc6bfb13c86cda3ea095edf0c5562fd4befc.py) |
| 候选／不符合当前约束 | `remaining-nine-20261030/proposals/special_round_out/ascendc-taskqueue-v1/assessment/formal/HG-KDA-ascendc-taskqueue-v1-1.result.json` | 1.204312× | [930de3f845f1](../../experiments/lineage/20261117/source-evidence/sources/930de3f845f1ef8af3437a58e4813066edafd1ce68b4b5d3182e74565fca9071.py) |
| 候选 | `remaining-nine-20261030/proposals/special_round_out/ascendc-double-buffer-v4/assessment/formal/HG-KDA-ascendc-double-buffer-v4-1.result.json` | 1.165445× | [7bf90fffbd8c](../../experiments/lineage/20261117/source-evidence/sources/7bf90fffbd8c7493807dc167ba38444d110ade878bcf889415ca98588a102021.py) |
| 候选／不符合当前约束 | `remaining-nine-20261030/final-confirmation/special_round_out/930de3f845f1/HG-KDA-final-confirm-1.result.json` | 1.292013× | [930de3f845f1](../../experiments/lineage/20261117/source-evidence/sources/930de3f845f1ef8af3437a58e4813066edafd1ce68b4b5d3182e74565fca9071.py) |
| 厂商参考 | `cann-priority-20261102/references/special_round_out/formal/CANN-1.result.json` | 1.024621× | [4cc6b2c91212](../../experiments/lineage/20261117/source-evidence/sources/4cc6b2c912129b43e0d7d310c23dfdc69db281eeb9e9ec4ccd8e823ce8ca1f29.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/comparisons/special_round_out/starting/HG-KDA-start-1.result.json` | 1.260756× | [930de3f845f1](../../experiments/lineage/20261117/source-evidence/sources/930de3f845f1ef8af3437a58e4813066edafd1ce68b4b5d3182e74565fca9071.py) |
| 候选 | `strict-triton-20261102/candidates/special_round_out/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.646014× | [e1f40725d719](../../experiments/lineage/20261117/source-evidence/sources/e1f40725d71976305a747534ac790196b1f034c526bf18cfcd72a809528a3c68.py) |
| 候选 | `strict-triton-20261102/candidates/special_round_out/restored-v1/CANN/strict-CANN-1.result.json` | 0.612524× | [e1f40725d719](../../experiments/lineage/20261117/source-evidence/sources/e1f40725d71976305a747534ac790196b1f034c526bf18cfcd72a809528a3c68.py) |
| 候选 | `strict-night-20261102/proposals/special_round_out/codex-streaming-round-v1/CANN/Codex-strict-CANN-1.result.json` | 0.676268× | [ceebc9436287](../../experiments/lineage/20261117/source-evidence/sources/ceebc9436287020db7c36cc00f8124ee0bc408ce3bf55e1cb848b3effcb7fe08.py) |
| 候选 | `strict-night-20261102/proposals/special_round_out/codex-streaming-round-v1/pytorch/strict-PyTorch-1.result.json` | 0.707351× | [ceebc9436287](../../experiments/lineage/20261117/source-evidence/sources/ceebc9436287020db7c36cc00f8124ee0bc408ce3bf55e1cb848b3effcb7fe08.py) |
| 候选 | `strict-night-20261102/proposals/special_round_out/codex-streaming-round-v1/formal-reference/Codex-strict-CANN-1.result.json` | 0.676268× | [ceebc9436287](../../experiments/lineage/20261117/source-evidence/sources/ceebc9436287020db7c36cc00f8124ee0bc408ce3bf55e1cb848b3effcb7fe08.py) |
| 候选 | `strict-night-20261102/proposals/special_round_out/codex-optional-out-v3/CANN/Codex-strict-CANN-1.result.json` | RUNTIME_ERROR | [436a74ed501a](../../experiments/lineage/20261117/source-evidence/sources/436a74ed501a40c0e5fea875d9ad1464696f04bede944c7e9f8e563ba5fdbc52.py) |
| 候选 | `strict-night-20261102/proposals/special_round_out/codex-required-out-v4/CANN/Codex-strict-CANN-1.result.json` | 0.946404× | [52807976ab25](../../experiments/lineage/20261117/source-evidence/sources/52807976ab25e9d7506337fb70ce6d0f6c147ae5bc859f74dfca9d7496064b7e.py) |
| 候选 | `strict-night-20261102/proposals/special_round_out/codex-required-out-v4/pytorch/strict-PyTorch-1.result.json` | 0.967388× | [52807976ab25](../../experiments/lineage/20261117/source-evidence/sources/52807976ab25e9d7506337fb70ce6d0f6c147ae5bc859f74dfca9d7496064b7e.py) |
| 候选 | `strict-night-20261102/proposals/special_round_out/codex-required-out-v4/formal-reference/Codex-strict-CANN-1.result.json` | 0.946404× | [52807976ab25](../../experiments/lineage/20261117/source-evidence/sources/52807976ab25e9d7506337fb70ce6d0f6c147ae5bc859f74dfca9d7496064b7e.py) |
| 候选 | `strict-night-20261102/proposals/special_round_out/codex-kg-r2-out-v5/CANN/Codex-strict-CANN-1.result.json` | 0.716817× | [40e733828553](../../experiments/lineage/20261117/source-evidence/sources/40e7338285537b1e10a9de341b3fc8f06af59a8d63a9b9b68e25155c18e1e145.py) |
| 候选 | `strict-night-20261102/proposals/special_round_out/codex-kg-r2-out-v5/pytorch/strict-PyTorch-1.result.json` | 0.723100× | [40e733828553](../../experiments/lineage/20261117/source-evidence/sources/40e7338285537b1e10a9de341b3fc8f06af59a8d63a9b9b68e25155c18e1e145.py) |
| 候选 | `strict-night-20261102/proposals/special_round_out/codex-kg-r2-out-v5/formal-reference/Codex-strict-CANN-1.result.json` | 0.716817× | [40e733828553](../../experiments/lineage/20261117/source-evidence/sources/40e7338285537b1e10a9de341b3fc8f06af59a8d63a9b9b68e25155c18e1e145.py) |
| 候选 | `independent-regression-20261109-v3/operators/special_round_out/evaluation/result.json` | 0.999713× | [2a980e3c6c59](../../experiments/lineage/20261117/source-evidence/sources/2a980e3c6c59b1548027d261503c43e945f313996e3e734fd4be46e7dc6eba9b.py) |

## 交付终点

- [选用源码](../../../operators/special_round_out/versions/independent_strict/retained/kernel.py)：`2a980e3c6c59b1548027d261503c43e945f313996e3e734fd4be46e7dc6eba9b`。
- [保留版完整结果](../../../operators/special_round_out/evidence/retained/evaluation/result.json)：**0.999713× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

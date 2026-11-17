# negative_

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/negative_.py)；SHA `b05ec46cb0d95ec9491863132d6630715557767ccd5367be82153edd424339d9`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/67dec7d11552643083edc6fdd2e6514548f07b41987f8a9816d2fa3dca7e6340.py)；原始模块AST一致，追加入口：`run = negative_`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/293a9ffaf25121d6fbf08095bebaa904d32e3c6a9d502d205650ac27ce3457ff.json)：**PARTIAL_PASS**；27/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/8c8d479272a8eece072bdc16c8754e6552fb20c1571d459fc488338cc2c7bd49.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/negative_/baseline/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 3条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 17份候选测量、12份请求源码；详细位置列在下表 |
| 交付终点 | 0.805699× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/negative_/baseline/FlagGems-1.result.json` | PARTIAL_PASS | [67dec7d11552](../../experiments/lineage/20261117/source-evidence/sources/67dec7d11552643083edc6fdd2e6514548f07b41987f8a9816d2fa3dca7e6340.py) |
| 候选 | `next25-20261029-v2/results/negative_/handoff-evaluation/formal/HG-KDA-recovered-1.result.json` | 0.167285× | [99976c0c4a66](../../experiments/lineage/20261117/source-evidence/sources/99976c0c4a66cd4daf2f6586d42e77b70eb3625564c077002099dc72de4ef2f1.py) |
| 候选 | `next25-20261029-v2/results/negative_/search-2/independent/HG-KDA-1.result.json` | 0.349964× | [6dd8338ba1b4](../../experiments/lineage/20261117/source-evidence/sources/6dd8338ba1b49783cc0f274f7e494b3d25878cb9363e9dd014efcee4b470c97a.py) |
| 候选 | `next25-20261029-v2/proposals/negative_/launch-reuse-v1/formal/HG-KDA-launch-reuse-1.result.json` | 0.596892× | [a533bb65b5c7](../../experiments/lineage/20261117/source-evidence/sources/a533bb65b5c7d231dcfbd17aa612cd5b8b1bfa5e12cd8ede6138b598fc1ff118.py) |
| 候选 | `next25-20261029-v2/proposals/negative_/launch-reuse-bitwise-v2/formal/HG-KDA-bitwise-1.result.json` | PARTIAL_PASS | [4d33d5c4a5df](../../experiments/lineage/20261117/source-evidence/sources/4d33d5c4a5dfca90c6356051cf5bd5ab10ea4e3bf9cbbbe1c86efc2424039459.py) |
| 候选 | `next25-20261029-v2/proposals/negative_/launch-reuse-bitwise-v3/formal/HG-KDA-bitwise-v3-1.result.json` | 0.512085× | [f4c0c5abc25d](../../experiments/lineage/20261117/source-evidence/sources/f4c0c5abc25d2c94358cd52f732da29dedf8bc270877f60d8e9134c36ea76598.py) |
| 候选 | `next25-20261029-v2/proposals/negative_/streaming-tiles-v4/assessment/formal/HG-KDA-streaming-tiles-v4-1.result.json` | 0.642921× | [387bc6790862](../../experiments/lineage/20261117/source-evidence/sources/387bc6790862952aa02d4e193902d9cc20b0d1fd700a532353b4283e2d42846d.py) |
| 候选 | `next25-20261029-v2/proposals/negative_/cpp-host-v1/assessment/formal/HG-KDA-cpp-host-v1-1.result.json` | 0.586510× | [44cb0cdcd04b](../../experiments/lineage/20261117/source-evidence/sources/44cb0cdcd04b634185c0ad2057638ffe891562a5e99d915504ca1332ca316e9a.py) |
| 候选 | `remaining-nine-20261030/results/negative_/current-recheck/formal/HG-KDA-current-1.result.json` | 0.641054× | [387bc6790862](../../experiments/lineage/20261117/source-evidence/sources/387bc6790862952aa02d4e193902d9cc20b0d1fd700a532353b4283e2d42846d.py) |
| 候选 | `remaining-nine-20261030/proposals/negative_/ascendc-taskqueue-v1/assessment/formal/HG-KDA-ascendc-taskqueue-v1-1.result.json` | 0.853715× | [53481f1314c4](../../experiments/lineage/20261117/source-evidence/sources/53481f1314c4b999cf9d6777492c753c6d6781c1f48f81b3c584dbbdb3957f7a.py) |
| 候选／不符合当前约束 | `remaining-nine-20261030/proposals/negative_/ascendc-double-buffer-v4/assessment/formal/HG-KDA-ascendc-double-buffer-v4-1.result.json` | 0.939909× | [c7aa7e021651](../../experiments/lineage/20261117/source-evidence/sources/c7aa7e0216519ed5a3a0e5c06471b0c8b06269750c1c3e35eb8dbe2710bed588.py) |
| 候选／不符合当前约束 | `remaining-nine-20261030/final-confirmation/negative_/c7aa7e021651/HG-KDA-final-confirm-1.result.json` | 0.933720× | [c7aa7e021651](../../experiments/lineage/20261117/source-evidence/sources/c7aa7e0216519ed5a3a0e5c06471b0c8b06269750c1c3e35eb8dbe2710bed588.py) |
| 厂商参考 | `cann-priority-20261102/references/negative_/formal/CANN-1.result.json` | 0.936334× | [5cf11b116e44](../../experiments/lineage/20261117/source-evidence/sources/5cf11b116e442876c971552b94381462dc24016e01c3a1c59093bebecc56a921.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/comparisons/negative_/starting/HG-KDA-start-1.result.json` | 1.250563× | [c7aa7e021651](../../experiments/lineage/20261117/source-evidence/sources/c7aa7e0216519ed5a3a0e5c06471b0c8b06269750c1c3e35eb8dbe2710bed588.py) |
| 候选 | `strict-triton-20261102/candidates/negative_/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.559748× | [ea48d9569c0a](../../experiments/lineage/20261117/source-evidence/sources/ea48d9569c0a24f52a5075b40a186c4640909e6962abdc84f69a3289836d5de8.py) |
| 候选 | `strict-triton-20261102/candidates/negative_/restored-v1/CANN/strict-CANN-1.result.json` | 0.813792× | [ea48d9569c0a](../../experiments/lineage/20261117/source-evidence/sources/ea48d9569c0a24f52a5075b40a186c4640909e6962abdc84f69a3289836d5de8.py) |
| 候选 | `independent-regression-20261109-v3/operators/negative_/evaluation/result.json` | 0.652877× | [ea48d9569c0a](../../experiments/lineage/20261117/source-evidence/sources/ea48d9569c0a24f52a5075b40a186c4640909e6962abdc84f69a3289836d5de8.py) |
| 候选 | `negative-host-repair-20261109-v1/operators/negative_/evaluation/result.json` | 0.694990× | [1e6b749ccdc6](../../experiments/lineage/20261117/source-evidence/sources/1e6b749ccdc6cf2c62394f27bd7bc31339fa779e72aa292992b4d3790a9e5c50.py) |
| 候选 | `negative-full-tile-stream-20261109-v3/operators/negative_/evaluation/result.json` | 0.805699× | [a3c2c0641053](../../experiments/lineage/20261117/source-evidence/sources/a3c2c0641053bd2da97f548b8124d5e550a0a418dae3fa3512e14cc18007a0ad.py) |

## 交付终点

- [选用源码](../../../operators/negative_/versions/independent_strict/retained/kernel.py)：`a3c2c0641053bd2da97f548b8124d5e550a0a418dae3fa3512e14cc18007a0ad`。
- [保留版完整结果](../../../operators/negative_/evidence/retained/evaluation/result.json)：**0.805699× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

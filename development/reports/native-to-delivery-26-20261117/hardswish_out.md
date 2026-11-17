# hardswish_out

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/hardswish.py)；SHA `180657de6cf100f6a1b6ada4570ef2c584512c0afbaf350d1fd6a608d390d636`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/4f3af612ded64bc7fa00e6396c8bf4bb9d9efd9db1848d856cfe7fa3789a97fc.py)；原始模块AST一致，追加入口：`run = hardswish_out`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/739e2ab6838c7fdc7b12bc9e0af9ed2b99a0d2dcac5db5fe1f010009f6a551e7.json)：**0.177×**；30/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/1d8e7f48a189fe494764a5d9591c28cfb12d596327119eadcb254393fe64d758.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/hardswish_out/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 0条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 21份候选测量、12份请求源码；详细位置列在下表 |
| 交付终点 | 0.866788× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/hardswish_out/baseline-fixed/FlagGems-1.result.json` | 0.177007× | [4f3af612ded6](../../experiments/lineage/20261117/source-evidence/sources/4f3af612ded64bc7fa00e6396c8bf4bb9d9efd9db1848d856cfe7fa3789a97fc.py) |
| 候选 | `next25-20261029-v2/proposals/hardswish_out/streaming-launch-v1/assessment/formal/HG-KDA-streaming-launch-v1-1.result.json` | 0.607292× | [f21a85259aca](../../experiments/lineage/20261117/source-evidence/sources/f21a85259acadc40fa06a7a90a14d1a92ebf416ab0dda2abae21c85c3456b24b.py) |
| 候选 | `next25-20261029-v2/proposals/hardswish_out/integer-metadata-v2/assessment/formal/HG-KDA-integer-metadata-v2-1.result.json` | 0.872548× | [5e0fca2cc95f](../../experiments/lineage/20261117/source-evidence/sources/5e0fca2cc95fc5001b2ee77db81b2b6de7e8ec1fca928a46ad5853804c73932d.py) |
| 候选 | `next25-20261029-v2/proposals/hardswish_out/native-precision-v3/assessment/formal/HG-KDA-native-precision-v3-1.result.json` | 0.845421× | [aef259422072](../../experiments/lineage/20261117/source-evidence/sources/aef259422072eb1c70e60e3bf0eacd0327723c10690cd14d4d6f8242ba13ec49.py) |
| 候选 | `next25-20261029-v2/proposals/hardswish_out/paired-fp32-stream-v4/assessment/formal/HG-KDA-paired-fp32-stream-v4-1.result.json` | 0.900444× | [a79f0fbfcc79](../../experiments/lineage/20261117/source-evidence/sources/a79f0fbfcc792030d7078152059aa6b412d4e6e992d90e8f9b387d3cd6ed39a9.py) |
| 候选 | `next25-20261029-v2/proposals/hardswish_out/large-ub-tile-v5/assessment/formal/HG-KDA-large-ub-tile-v5-1.result.json` | PARTIAL_PASS | [ce06af4e186d](../../experiments/lineage/20261117/source-evidence/sources/ce06af4e186d580dd1eeeee34be495652ccafd2b03d053c851385bbd43eb7466.py) |
| 候选 | `threeway-20261030/results/hardswish_out/current-recheck/formal/HG-KDA-current-1.result.json` | 0.871013× | [a79f0fbfcc79](../../experiments/lineage/20261117/source-evidence/sources/a79f0fbfcc792030d7078152059aa6b412d4e6e992d90e8f9b387d3cd6ed39a9.py) |
| 厂商参考 | `threeway-20261030/references/hardswish_out/formal/CANN-async-python-1.result.json` | RUNTIME_ERROR | [bd5afe8364d6](../../experiments/lineage/20261117/source-evidence/sources/bd5afe8364d65385da766b7aba3394f27c0ec1ea855a9a5db772ceb9e63add98.py) |
| 厂商参考 | `threeway-20261030/references/hardswish_out/abi-v2/formal/CANN-async-python-abi-v2-1.result.json` | 0.156076× | [c45ab7fbc83d](../../experiments/lineage/20261117/source-evidence/sources/c45ab7fbc83db8ab846d4a32be38b2834ecce2751efdafbb71bd1cfe63559d43.py) |
| 厂商参考 | `threeway-20261030/references/hardswish_out/cpp-v1/formal/CANN-cpp-1.result.json` | 1.178917× | [aa8b97aa92b9](../../experiments/lineage/20261117/source-evidence/sources/aa8b97aa92b90e5197ba1660cfb1a2967865248df1165a513341f672fdde07a3.py) |
| 候选 | `cann-priority-20261102/proposals/hardswish_out/ascendc-out-v2/original/HG-KDA-new-1.result.json` | 1.463090× | [0d2256b11842](../../experiments/lineage/20261117/source-evidence/sources/0d2256b118424b49f6eafa7c4d4ccf44fc36c2d54c6d9c8537fa9a20cd52a744.py) |
| 候选 | `cann-priority-20261102/proposals/hardswish_out/ascendc-out-v2/CANN/HG-KDA-new-vs-CANN-1.result.json` | 1.362299× | [0d2256b11842](../../experiments/lineage/20261117/source-evidence/sources/0d2256b118424b49f6eafa7c4d4ccf44fc36c2d54c6d9c8537fa9a20cd52a744.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/proposals/hardswish_out/ascendc-large-v3/original/HG-KDA-new-1.result.json` | 1.708008× | [a668504746ab](../../experiments/lineage/20261117/source-evidence/sources/a668504746ab74e24a2ba03f94e3e2ac58fc0fde6739876453ee0edfff9609e9.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/proposals/hardswish_out/ascendc-large-v3/CANN/HG-KDA-new-vs-CANN-1.result.json` | 1.394660× | [a668504746ab](../../experiments/lineage/20261117/source-evidence/sources/a668504746ab74e24a2ba03f94e3e2ac58fc0fde6739876453ee0edfff9609e9.py) |
| 厂商参考 | `cann-priority-20261102/references/hardswish_out/formal/CANN-1.result.json` | RUNTIME_ERROR | [5812f942bd62](../../experiments/lineage/20261117/source-evidence/sources/5812f942bd625f765464cd0b35b01692253be455ba0771a0e5a4e3fcdda6dcfc.py) |
| 厂商参考 | `cann-priority-20261102/references/hardswish_out/abi-v2/formal/CANN-1.result.json` | 1.334682× | [19767a3ac522](../../experiments/lineage/20261117/source-evidence/sources/19767a3ac522d1994a0b3760ebbb61386e724ec813c0525dd9d5f11d0cf11e0f.py) |
| 候选 | `cann-priority-20261102/comparisons/hardswish_out/starting/HG-KDA-start-1.result.json` | 0.680835× | [a79f0fbfcc79](../../experiments/lineage/20261117/source-evidence/sources/a79f0fbfcc792030d7078152059aa6b412d4e6e992d90e8f9b387d3cd6ed39a9.py) |
| 候选 | `strict-triton-20261102/candidates/hardswish_out/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.831719× | [b645c6627435](../../experiments/lineage/20261117/source-evidence/sources/b645c662743556fa63b8bac351baf9b1f890965528c9d303b4f72d222eda3b10.py) |
| 候选 | `strict-triton-20261102/candidates/hardswish_out/restored-v1/CANN/strict-CANN-1.result.json` | 0.637293× | [b645c6627435](../../experiments/lineage/20261117/source-evidence/sources/b645c662743556fa63b8bac351baf9b1f890965528c9d303b4f72d222eda3b10.py) |
| 候选 | `strict-night-20261102/proposals/hardswish_out/codex-verified-stream-out-v1/CANN/Codex-strict-CANN-1.result.json` | 0.776294× | [9c5a3c776197](../../experiments/lineage/20261117/source-evidence/sources/9c5a3c776197abb0fbb11d77c4fb3d60011a71e50a32f3b7c49d330830ec181f.py) |
| 候选 | `strict-night-20261102/proposals/hardswish_out/codex-verified-stream-out-v1/pytorch/strict-PyTorch-1.result.json` | 0.969741× | [9c5a3c776197](../../experiments/lineage/20261117/source-evidence/sources/9c5a3c776197abb0fbb11d77c4fb3d60011a71e50a32f3b7c49d330830ec181f.py) |
| 候选 | `strict-night-20261102/proposals/hardswish_out/codex-verified-stream-out-v1/formal-reference/Codex-strict-CANN-1.result.json` | 0.776294× | [9c5a3c776197](../../experiments/lineage/20261117/source-evidence/sources/9c5a3c776197abb0fbb11d77c4fb3d60011a71e50a32f3b7c49d330830ec181f.py) |
| 候选 | `strict-night-20261102/proposals/hardswish_out/codex-vectorcall-out-v4/CANN/Codex-strict-CANN-1.result.json` | 0.862347× | [8c8955c72e5d](../../experiments/lineage/20261117/source-evidence/sources/8c8955c72e5d432d421945afdef889766e7e29bac030a9325c6c5f309ef333cb.py) |
| 候选 | `strict-night-20261102/proposals/hardswish_out/codex-vectorcall-out-v4/pytorch/strict-PyTorch-1.result.json` | 1.049675× | [8c8955c72e5d](../../experiments/lineage/20261117/source-evidence/sources/8c8955c72e5d432d421945afdef889766e7e29bac030a9325c6c5f309ef333cb.py) |
| 候选 | `strict-night-20261102/proposals/hardswish_out/codex-vectorcall-out-v4/formal-reference/Codex-strict-CANN-1.result.json` | 0.862347× | [8c8955c72e5d](../../experiments/lineage/20261117/source-evidence/sources/8c8955c72e5d432d421945afdef889766e7e29bac030a9325c6c5f309ef333cb.py) |
| 候选 | `independent-regression-20261109-v3/operators/hardswish_out/evaluation/result.json` | 0.778083× | [3d23f79ec3b0](../../experiments/lineage/20261117/source-evidence/sources/3d23f79ec3b0aeac20f7fa76c85bbfcb5efc3dc06946533ba046b384a258372c.py) |
| 候选 | `hardswish-out-half-tiles-20261109-v7/operators/hardswish_out/evaluation/result.json` | 0.866788× | [24615d5751c1](../../experiments/lineage/20261117/source-evidence/sources/24615d5751c1e2f2f31e4289cf2e77a8f753138c62d3b63a681f7eb161107d2a.py) |

## 交付终点

- [选用源码](../../../operators/hardswish_out/versions/independent_strict/retained/kernel.py)：`24615d5751c1e2f2f31e4289cf2e77a8f753138c62d3b63a681f7eb161107d2a`。
- [保留版完整结果](../../../operators/hardswish_out/evidence/retained/evaluation/result.json)：**0.866788× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

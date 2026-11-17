# hardswish

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/hardswish.py)；SHA `180657de6cf100f6a1b6ada4570ef2c584512c0afbaf350d1fd6a608d390d636`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/87dfe2f8c68a4aed624ef3080089447701b8553e67aa4790774b376adc394497.py)；原始模块AST一致，追加入口：`run = hardswish`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/12ff375fe51709605db3b32e48d5064d143e7b7efa5e3c0552785da2ee4fa3a7.json)：**0.148×**；30/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/effb5b7bb6411a3a6ea087645ec6cfbe5b20a4d797d195991c9ef56f73993352.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/hardswish/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 6条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 21份候选测量、12份请求源码；详细位置列在下表 |
| 交付终点 | 0.849218× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/hardswish/baseline-fixed/FlagGems-1.result.json` | 0.147985× | [87dfe2f8c68a](../../experiments/lineage/20261117/source-evidence/sources/87dfe2f8c68a4aed624ef3080089447701b8553e67aa4790774b376adc394497.py) |
| 候选 | `next25-20261029-v2/results/hardswish/search-1/independent/HG-KDA-1.result.json` | 0.683204× | [7d33bf4ffdce](../../experiments/lineage/20261117/source-evidence/sources/7d33bf4ffdce3fb6981b0f3c02233848611c1f6460fac9ecddcff9c588b4edc3.py) |
| 候选 | `next25-20261029-v2/proposals/hardswish/streaming-launch-v1/assessment/formal/HG-KDA-streaming-launch-v1-1.result.json` | 0.464980× | [523894759c10](../../experiments/lineage/20261117/source-evidence/sources/523894759c106bdc4c7bd7d4a5aaff540646bba712323a0deeb7e23281f1c25c.py) |
| 候选 | `next25-20261029-v2/proposals/hardswish/integer-metadata-v2/assessment/formal/HG-KDA-integer-metadata-v2-1.result.json` | 0.670301× | [49a714651cbb](../../experiments/lineage/20261117/source-evidence/sources/49a714651cbb56a198fc8455e0e73034157e30fdf958bff744333f1d0cff2285.py) |
| 候选 | `next25-20261029-v2/proposals/hardswish/native-precision-v3/assessment/formal/HG-KDA-native-precision-v3-1.result.json` | 0.613795× | [340d56d8676e](../../experiments/lineage/20261117/source-evidence/sources/340d56d8676e7d5337f3f97722bca9b60c1d1bb7e77e33acea44cf2b124e10a4.py) |
| 候选 | `next25-20261029-v2/proposals/hardswish/paired-fp32-stream-v4/assessment/formal/HG-KDA-paired-fp32-stream-v4-1.result.json` | 0.621832× | [47bd3f2d8d31](../../experiments/lineage/20261117/source-evidence/sources/47bd3f2d8d318618b8fc54a0545a1fca5fd96c6012a5f917c2b922a4a941626b.py) |
| 候选 | `next25-20261029-v2/proposals/hardswish/large-ub-tile-v5/assessment/formal/HG-KDA-large-ub-tile-v5-1.result.json` | PARTIAL_PASS | [e53c212f32ca](../../experiments/lineage/20261117/source-evidence/sources/e53c212f32ca9f027c352d95037bac72887f875da536e54027bdd535175d47e0.py) |
| 候选 | `remaining-nine-20261030/results/hardswish/current-recheck/formal/HG-KDA-current-1.result.json` | 0.669254× | [7d33bf4ffdce](../../experiments/lineage/20261117/source-evidence/sources/7d33bf4ffdce3fb6981b0f3c02233848611c1f6460fac9ecddcff9c588b4edc3.py) |
| 候选／不符合当前约束 | `remaining-nine-20261030/proposals/hardswish/ascendc-python-allocation-v5/assessment/formal/HG-KDA-ascendc-python-allocation-v5-1.result.json` | 0.935185× | [adb5b9990f99](../../experiments/lineage/20261117/source-evidence/sources/adb5b9990f99d12f0b330ff663c34dfd2bb3d95633257c7b836c71265239f11f.py) |
| 候选／不符合当前约束 | `remaining-nine-20261030/final-confirmation/hardswish/adb5b9990f99/HG-KDA-final-confirm-1.result.json` | 0.875740× | [adb5b9990f99](../../experiments/lineage/20261117/source-evidence/sources/adb5b9990f99d12f0b330ff663c34dfd2bb3d95633257c7b836c71265239f11f.py) |
| 候选 | `cann-priority-20261102/proposals/hardswish/ascendc-large-v3/original/HG-KDA-new-1.result.json` | 0.806716× | [5c76cd3195ee](../../experiments/lineage/20261117/source-evidence/sources/5c76cd3195eed3b2f5671cb29174c2d1c683486ae3e3c9eb03501d13a9d8d6c8.py) |
| 候选 | `cann-priority-20261102/proposals/hardswish/ascendc-large-v3/CANN/HG-KDA-new-vs-CANN-1.result.json` | 1.046004× | [5c76cd3195ee](../../experiments/lineage/20261117/source-evidence/sources/5c76cd3195eed3b2f5671cb29174c2d1c683486ae3e3c9eb03501d13a9d8d6c8.py) |
| 厂商参考 | `cann-priority-20261102/references/hardswish/formal/CANN-1.result.json` | 0.861810× | [a36a23c73973](../../experiments/lineage/20261117/source-evidence/sources/a36a23c73973f2b8a85c842f1e4a06f51dd44101e0383172e16dfc395df2de8e.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/comparisons/hardswish/starting/HG-KDA-start-1.result.json` | 1.060919× | [adb5b9990f99](../../experiments/lineage/20261117/source-evidence/sources/adb5b9990f99d12f0b330ff663c34dfd2bb3d95633257c7b836c71265239f11f.py) |
| 候选 | `strict-triton-20261102/candidates/hardswish/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.629013× | [492997f7aae4](../../experiments/lineage/20261117/source-evidence/sources/492997f7aae410f9e0ccded09f4d90ad184344e9d841a173bb1fb5a9ed625e80.py) |
| 候选 | `strict-triton-20261102/candidates/hardswish/restored-v1/CANN/strict-CANN-1.result.json` | 0.798549× | [492997f7aae4](../../experiments/lineage/20261117/source-evidence/sources/492997f7aae410f9e0ccded09f4d90ad184344e9d841a173bb1fb5a9ed625e80.py) |
| 候选 | `strict-night-20261102/candidates/hardswish/581ad44a1694f431/pytorch/strict-PyTorch-1.result.json` | 0.731180× | [581ad44a1694](../../experiments/lineage/20261117/source-evidence/sources/581ad44a1694f4311efaa907e84874a5b33e9df5e8a751cd2859513a72ac1101.py) |
| 候选 | `strict-night-20261102/candidates/hardswish/581ad44a1694f431/formal-reference/result.json` | 0.911505× | 请求源码未定位 |
| 候选 | `strict-night-20261102/candidates/hardswish/581ad44a1694f431/formal-reference/measurement-0/result.json` | 0.911505× | 请求源码未定位 |
| 候选 | `strict-night-20261102/candidates/hardswish/e1bb9d02f196ca1a/pytorch/strict-PyTorch-1.result.json` | 0.730753× | [e1bb9d02f196](../../experiments/lineage/20261117/source-evidence/sources/e1bb9d02f196ca1ad7e73ecbf5ef004f44e72d7479bb96e5842742818d962d98.py) |
| 候选 | `strict-night-20261102/candidates/hardswish/7ad08150b03d80c9/pytorch/strict-PyTorch-1.result.json` | 0.638328× | [7ad08150b03d](../../experiments/lineage/20261117/source-evidence/sources/7ad08150b03d80c9a2eb17b48e613401d62c311ccdf1b9b139aea5937c9dc301.py) |
| 候选 | `strict-night-20261102/review-preview/hardswish/round-0003/result.json` | 0.911505× | 请求源码未定位 |
| 候选 | `independent-regression-20261109-v3/operators/hardswish/evaluation/result.json` | 0.849218× | [581ad44a1694](../../experiments/lineage/20261117/source-evidence/sources/581ad44a1694f4311efaa907e84874a5b33e9df5e8a751cd2859513a72ac1101.py) |

## 交付终点

- [选用源码](../../../operators/hardswish/versions/independent_strict/retained/kernel.py)：`581ad44a1694f4311efaa907e84874a5b33e9df5e8a751cd2859513a72ac1101`。
- [保留版完整结果](../../../operators/hardswish/evidence/retained/evaluation/result.json)：**0.849218× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

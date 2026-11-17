# batch_norm_with_update_functional

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/_batch_norm_with_update_functional.py)；SHA `bd3b414cd706a0d4f040dd725169654702b7ea2c92f019565ee859bdadabfeed`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/2beb154360c815bcda2b31be363bf2e2779ac1a71469bb0365858c095c217698.py)；原始模块AST一致，追加入口：`run = _batch_norm_with_update_functional`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/17190db7b93d5e66548fb26dc5721c64e4a36a966eff4f1216cfdfac051d95c5.json)：**100.912×**；27/27。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/6579c39372b18630cabf5c38e4d11328a115d2c990ed9fac91bed7d969f308d1.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/batch_norm_with_update_functional/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 0条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 5份候选测量、2份请求源码；详细位置列在下表 |
| 交付终点 | 1.773885× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/batch_norm_with_update_functional/baseline-fixed/FlagGems-1.result.json` | 100.912330× | [2beb154360c8](../../experiments/lineage/20261117/source-evidence/sources/2beb154360c815bcda2b31be363bf2e2779ac1a71469bb0365858c095c217698.py) |
| 候选 | `reference-closure-20261031/comparisons/batch_norm_with_update_functional/CANN-direct/HG-KDA-vs-CANN-1.result.json` | 1.755523× | [2beb154360c8](../../experiments/lineage/20261117/source-evidence/sources/2beb154360c815bcda2b31be363bf2e2779ac1a71469bb0365858c095c217698.py) |
| 厂商参考 | `reference-closure-20261031/references-v3/batch_norm_with_update_functional/formal/CANN-direct-v3-1.result.json` | 10.609975× | [8897f1104ea6](../../experiments/lineage/20261117/source-evidence/sources/8897f1104ea62bfb5c2c6f743aba5efccaa033bef0b562ade3c829bb9d552546.py) |
| 厂商参考 | `cann-priority-20261102/references/batch_norm_with_update_functional/formal/CANN-1.result.json` | 13.172437× | [8897f1104ea6](../../experiments/lineage/20261117/source-evidence/sources/8897f1104ea62bfb5c2c6f743aba5efccaa033bef0b562ade3c829bb9d552546.py) |
| 候选 | `cann-priority-20261102/comparisons/batch_norm_with_update_functional/starting/HG-KDA-start-1.result.json` | 1.783262× | [2beb154360c8](../../experiments/lineage/20261117/source-evidence/sources/2beb154360c815bcda2b31be363bf2e2779ac1a71469bb0365858c095c217698.py) |
| 候选 | `strict-triton-20261102/candidates/batch_norm_with_update_functional/restored-v1/pytorch/strict-PyTorch-1.result.json` | 23.562510× | [ca3ff529fa06](../../experiments/lineage/20261117/source-evidence/sources/ca3ff529fa06e3a4c799037311aad9c3f3edaf3179fba26a2646705a0fd1ccef.py) |
| 候选 | `strict-triton-20261102/candidates/batch_norm_with_update_functional/restored-v1/CANN/strict-CANN-1.result.json` | 1.748443× | [ca3ff529fa06](../../experiments/lineage/20261117/source-evidence/sources/ca3ff529fa06e3a4c799037311aad9c3f3edaf3179fba26a2646705a0fd1ccef.py) |
| 候选 | `independent-regression-20261109-v3/operators/batch_norm_with_update_functional/evaluation/result.json` | 1.773885× | [ca3ff529fa06](../../experiments/lineage/20261117/source-evidence/sources/ca3ff529fa06e3a4c799037311aad9c3f3edaf3179fba26a2646705a0fd1ccef.py) |

## 交付终点

- [选用源码](../../../operators/batch_norm_with_update_functional/versions/independent_strict/retained/kernel.py)：`ca3ff529fa06e3a4c799037311aad9c3f3edaf3179fba26a2646705a0fd1ccef`。
- [保留版完整结果](../../../operators/batch_norm_with_update_functional/evidence/retained/evaluation/result.json)：**1.773885× CANN，27/27**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

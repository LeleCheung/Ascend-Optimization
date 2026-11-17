# rad2deg_

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/rad2deg.py)；SHA `c7b538e36c1b4c90bb514fef75f93effc8f0058bfb35204f22ac9cb3b8206ca3`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/d0363c9222c5094ce113562c0d5e726d89f4d27902048167df23dfa58a253782.py)；原始模块AST一致，追加入口：`run = rad2deg_`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/3b6a32618614a40221a3a4ae30b16d78a17acadc3f6685151101aa769c213c14.json)：**PARTIAL_PASS**；33/36。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/b4bac8663d657abdf16748d5e1b9a82a5d5cf4a4ba79123aa2de534cb073074e.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/night-20261021-v3/results/rad2deg_/FlagGems/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 8条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 16份候选测量、8份请求源码；详细位置列在下表 |
| 交付终点 | 0.834365× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `night-20261021-v2/results/rad2deg_/FlagGems/FlagGems-1.result.json` | PARTIAL_PASS | [d0363c9222c5](../../experiments/lineage/20261117/source-evidence/sources/d0363c9222c5094ce113562c0d5e726d89f4d27902048167df23dfa58a253782.py) |
| 基线目录 | `night-20261021-v3/results/rad2deg_/FlagGems/FlagGems-1.result.json` | PARTIAL_PASS | [d0363c9222c5](../../experiments/lineage/20261117/source-evidence/sources/d0363c9222c5094ce113562c0d5e726d89f4d27902048167df23dfa58a253782.py) |
| 候选 | `endurance-20261022/results/rad2deg_/parent/formal/parent-1.result.json` | PARTIAL_PASS | [d0363c9222c5](../../experiments/lineage/20261117/source-evidence/sources/d0363c9222c5094ce113562c0d5e726d89f4d27902048167df23dfa58a253782.py) |
| 候选 | `endurance-20261022/results/rad2deg_/attempt-1/candidates/72f27523c6a7aa9ed34de7f122e86469ce65fb6abe065696520b0b923441cf24/formal/round-0004-1.result.json` | 1.111705× | [72f27523c6a7](../../experiments/lineage/20261117/source-evidence/sources/72f27523c6a7aa9ed34de7f122e86469ce65fb6abe065696520b0b923441cf24.py) |
| 候选 | `endurance-20261022/results/rad2deg_/attempt-1/candidates/3b2c0328b1857ef802e575805c4e25d002ac0166c22fa5cc6804cb1c8d0eeecc/formal/round-0003-1.result.json` | 1.117110× | [3b2c0328b185](../../experiments/lineage/20261117/source-evidence/sources/3b2c0328b1857ef802e575805c4e25d002ac0166c22fa5cc6804cb1c8d0eeecc.py) |
| 候选 | `endurance-20261022/results/rad2deg_/attempt-1/candidates/ac68049bafd05776520b7e7fbe991fe9e3d900b98899cc9a6b79f8d77dea9508/formal/round-0002-1.result.json` | 0.835992× | [ac68049bafd0](../../experiments/lineage/20261117/source-evidence/sources/ac68049bafd05776520b7e7fbe991fe9e3d900b98899cc9a6b79f8d77dea9508.py) |
| 候选 | `endurance-20261022/results/rad2deg_/attempt-1/candidates/d492f73ba9ef961185397e44b7efb67d660668fcd58ea6db3e496a855f11d720/formal/round-0001-1.result.json` | 0.308180× | [d492f73ba9ef](../../experiments/lineage/20261117/source-evidence/sources/d492f73ba9ef961185397e44b7efb67d660668fcd58ea6db3e496a855f11d720.py) |
| 基线目录 | `next25-20261029-v2/results/rad2deg_/baseline-fixed/FlagGems-1.result.json` | PARTIAL_PASS | [d0363c9222c5](../../experiments/lineage/20261117/source-evidence/sources/d0363c9222c5094ce113562c0d5e726d89f4d27902048167df23dfa58a253782.py) |
| 候选 | `next25-20261029-v2/results/rad2deg_/historical-start/HG-KDA-start-1.result.json` | 1.110925× | [3b2c0328b185](../../experiments/lineage/20261117/source-evidence/sources/3b2c0328b1857ef802e575805c4e25d002ac0166c22fa5cc6804cb1c8d0eeecc.py) |
| 候选 | `threeway-20261030/results/rad2deg_/current-recheck/formal/HG-KDA-current-1.result.json` | 1.039024× | [3b2c0328b185](../../experiments/lineage/20261117/source-evidence/sources/3b2c0328b1857ef802e575805c4e25d002ac0166c22fa5cc6804cb1c8d0eeecc.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/proposals/rad2deg_/ascendc-streaming-v1/original/HG-KDA-new-1.result.json` | 1.506123× | [14d7e12bce9e](../../experiments/lineage/20261117/source-evidence/sources/14d7e12bce9e264be933ee8c8815af32e26c44b6269247526461061f463d6ad2.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/proposals/rad2deg_/ascendc-streaming-v1/CANN/HG-KDA-new-vs-CANN-1.result.json` | 1.296726× | [14d7e12bce9e](../../experiments/lineage/20261117/source-evidence/sources/14d7e12bce9e264be933ee8c8815af32e26c44b6269247526461061f463d6ad2.py) |
| 厂商参考 | `cann-priority-20261102/references/rad2deg_/formal/CANN-1.result.json` | 1.289956× | [b3c13aec72a7](../../experiments/lineage/20261117/source-evidence/sources/b3c13aec72a729f2ae251a4954b04ef80fa81ff8c12fc715df67aeb299690d50.py) |
| 候选 | `cann-priority-20261102/comparisons/rad2deg_/starting/HG-KDA-start-1.result.json` | 0.766761× | [3b2c0328b185](../../experiments/lineage/20261117/source-evidence/sources/3b2c0328b1857ef802e575805c4e25d002ac0166c22fa5cc6804cb1c8d0eeecc.py) |
| 候选 | `strict-triton-20261102/candidates/rad2deg_/restored-v1/pytorch/strict-PyTorch-1.result.json` | 1.142810× | [3b2c0328b185](../../experiments/lineage/20261117/source-evidence/sources/3b2c0328b1857ef802e575805c4e25d002ac0166c22fa5cc6804cb1c8d0eeecc.py) |
| 候选 | `strict-night-20261102/candidates/rad2deg_/1b1834c2bbea117f/pytorch/strict-PyTorch-1.result.json` | 1.128044× | [1b1834c2bbea](../../experiments/lineage/20261117/source-evidence/sources/1b1834c2bbea117f896effac8b5312b650350000335307a30cd0f64896b3bcf0.py) |
| 候选 | `strict-night-20261102/proposals/rad2deg_/codex-launch-once-v1/CANN/Codex-strict-CANN-1.result.json` | 0.921753× | [5e47796a38a4](../../experiments/lineage/20261117/source-evidence/sources/5e47796a38a467c41bf25f2e30c4354b82d08535cf43d02991c181a4a4a6f1ce.py) |
| 候选 | `strict-night-20261102/proposals/rad2deg_/codex-launch-once-v1/pytorch/strict-PyTorch-1.result.json` | 1.119019× | [5e47796a38a4](../../experiments/lineage/20261117/source-evidence/sources/5e47796a38a467c41bf25f2e30c4354b82d08535cf43d02991c181a4a4a6f1ce.py) |
| 候选 | `strict-night-20261102/proposals/rad2deg_/codex-launch-once-v1/formal-reference/Codex-strict-CANN-1.result.json` | 0.921753× | [5e47796a38a4](../../experiments/lineage/20261117/source-evidence/sources/5e47796a38a467c41bf25f2e30c4354b82d08535cf43d02991c181a4a4a6f1ce.py) |
| 候选 | `independent-regression-20261109-v3/operators/rad2deg_/evaluation/result.json` | 0.834365× | [5e47796a38a4](../../experiments/lineage/20261117/source-evidence/sources/5e47796a38a467c41bf25f2e30c4354b82d08535cf43d02991c181a4a4a6f1ce.py) |

## 交付终点

- [选用源码](../../../operators/rad2deg_/versions/independent_strict/retained/kernel.py)：`5e47796a38a467c41bf25f2e30c4354b82d08535cf43d02991c181a4a4a6f1ce`。
- [保留版完整结果](../../../operators/rad2deg_/evidence/retained/evaluation/result.json)：**0.834365× CANN，36/36**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

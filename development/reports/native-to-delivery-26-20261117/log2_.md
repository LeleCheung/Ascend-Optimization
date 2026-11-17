# log2_

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/log2.py)；SHA `39edc048a19c4dcc7e77c1d67edbb7fcc41e3da6d52bf86cf6b1a78cc53daf0f`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/b2765d1a731fb30e5e5d8e650fd7fc163b8d13994583bdf665deecaa6f10075e.py)；原始模块AST一致，追加入口：`run = log2_`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/40791e6992f1c0cf82a8039e960333d18547a1e83ae86b7b49cb7fffbb6f0e58.json)：**PARTIAL_PASS**；27/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/1a0708538da9db598e386fb488cd945653d2e7423a8180f7d9387b5a660f16c5.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/log2_/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 1条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 14份候选测量、6份请求源码；详细位置列在下表 |
| 交付终点 | 0.879710× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/log2_/baseline-fixed/FlagGems-1.result.json` | PARTIAL_PASS | [b2765d1a731f](../../experiments/lineage/20261117/source-evidence/sources/b2765d1a731fb30e5e5d8e650fd7fc163b8d13994583bdf665deecaa6f10075e.py) |
| 候选 | `next25-20261029-v2/results/log2_/search-1/independent/HG-KDA-1.result.json` | 0.593137× | [3b1cc89abe52](../../experiments/lineage/20261117/source-evidence/sources/3b1cc89abe52d638245bc1bc983b30c4cdd182ec40ee600bd5b9ec85a542d8bc.py) |
| 候选 | `remaining-nine-20261030/results/log2_/current-recheck/formal/HG-KDA-current-1.result.json` | 0.604323× | [3b1cc89abe52](../../experiments/lineage/20261117/source-evidence/sources/3b1cc89abe52d638245bc1bc983b30c4cdd182ec40ee600bd5b9ec85a542d8bc.py) |
| 候选／不符合当前约束 | `remaining-nine-20261030/proposals/log2_/ascendc-taskqueue-v1/assessment/formal/HG-KDA-ascendc-taskqueue-v1-1.result.json` | 0.998232× | [71146086182a](../../experiments/lineage/20261117/source-evidence/sources/71146086182a2fd2d1347893ab5cce13c67fae1f97c0a0e7025669954a134065.py) |
| 候选 | `remaining-nine-20261030/proposals/log2_/ascendc-double-buffer-v4/assessment/formal/HG-KDA-ascendc-double-buffer-v4-1.result.json` | 0.890234× | [13c22e253f06](../../experiments/lineage/20261117/source-evidence/sources/13c22e253f0667f3beee2debab03c63d4b2514d4701fa7d674eae0f21942b8fb.py) |
| 候选／不符合当前约束 | `remaining-nine-20261030/final-confirmation/log2_/71146086182a/HG-KDA-final-confirm-1.result.json` | 0.839166× | [71146086182a](../../experiments/lineage/20261117/source-evidence/sources/71146086182a2fd2d1347893ab5cce13c67fae1f97c0a0e7025669954a134065.py) |
| 厂商参考 | `cann-priority-20261102/references/log2_/formal/CANN-1.result.json` | 1.009154× | [d0e1942796dc](../../experiments/lineage/20261117/source-evidence/sources/d0e1942796dcdeb3ce68f98e61f33112377dbe8bf85e58f29779be00a59a0415.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/comparisons/log2_/starting/HG-KDA-start-1.result.json` | 1.055184× | [71146086182a](../../experiments/lineage/20261117/source-evidence/sources/71146086182a2fd2d1347893ab5cce13c67fae1f97c0a0e7025669954a134065.py) |
| 候选 | `strict-triton-20261102/candidates/log2_/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.570131× | [60da3a5b8c2a](../../experiments/lineage/20261117/source-evidence/sources/60da3a5b8c2ab235ac065eaacd20a7fdaf3117a1937548bff25284c62820cb12.py) |
| 候选 | `strict-triton-20261102/candidates/log2_/restored-v1/CANN/strict-CANN-1.result.json` | 0.632674× | [60da3a5b8c2a](../../experiments/lineage/20261117/source-evidence/sources/60da3a5b8c2ab235ac065eaacd20a7fdaf3117a1937548bff25284c62820cb12.py) |
| 候选 | `strict-night-20261102/proposals/log2_/codex-streaming-log2-v1/CANN/Codex-strict-CANN-1.result.json` | 0.627337× | [5a502384ef35](../../experiments/lineage/20261117/source-evidence/sources/5a502384ef350d05cb4dd5bd370baa480d114b4d68dbf4861db51c17f9816c08.py) |
| 候选 | `strict-night-20261102/proposals/log2_/codex-streaming-log2-v1/pytorch/strict-PyTorch-1.result.json` | 0.686982× | [5a502384ef35](../../experiments/lineage/20261117/source-evidence/sources/5a502384ef350d05cb4dd5bd370baa480d114b4d68dbf4861db51c17f9816c08.py) |
| 候选 | `strict-night-20261102/proposals/log2_/codex-transfer-log10-v2/CANN/Codex-strict-CANN-1.result.json` | 0.845795× | [d5d57abf6a2c](../../experiments/lineage/20261117/source-evidence/sources/d5d57abf6a2cf5d7f7918de636177653a50f90a1d508e60a6f4b0381203542cb.py) |
| 候选 | `strict-night-20261102/proposals/log2_/codex-transfer-log10-v2/pytorch/strict-PyTorch-1.result.json` | 0.634218× | [d5d57abf6a2c](../../experiments/lineage/20261117/source-evidence/sources/d5d57abf6a2cf5d7f7918de636177653a50f90a1d508e60a6f4b0381203542cb.py) |
| 候选 | `strict-night-20261102/proposals/log2_/codex-transfer-log10-v2/formal-reference/Codex-strict-CANN-1.result.json` | 0.845795× | [d5d57abf6a2c](../../experiments/lineage/20261117/source-evidence/sources/d5d57abf6a2cf5d7f7918de636177653a50f90a1d508e60a6f4b0381203542cb.py) |
| 候选 | `independent-regression-20261109-v3/operators/log2_/evaluation/result.json` | 0.879710× | [d5d57abf6a2c](../../experiments/lineage/20261117/source-evidence/sources/d5d57abf6a2cf5d7f7918de636177653a50f90a1d508e60a6f4b0381203542cb.py) |

## 交付终点

- [选用源码](../../../operators/log2_/versions/independent_strict/retained/kernel.py)：`d5d57abf6a2cf5d7f7918de636177653a50f90a1d508e60a6f4b0381203542cb`。
- [保留版完整结果](../../../operators/log2_/evidence/retained/evaluation/result.json)：**0.879710× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

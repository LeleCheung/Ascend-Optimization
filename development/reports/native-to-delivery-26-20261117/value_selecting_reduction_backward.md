# value_selecting_reduction_backward

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/value_selecting_reduction_backward.py)；SHA `a687302d7d9412ca995e2cf245555ebe10b3d96f8c9da9d95a33061db63b0a53`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/e67c3e9aabe620904fe8689f0bd958c47da08e79c11de0c608fe8bfc5e738f11.py)；原始模块AST一致，追加入口：`run = value_selecting_reduction_backward`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/9f3e106b848755cbc5c2fc148e7a5e9d4ba636af5f0711befc044f19b8c7240f.json)：**0.759×**；96/96。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/2b5371f3285e6a41592d00b805a5b7bb35d7f8c65d508d1a5fcb03b3287d42a1.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/night-20261021-v3/results/value_selecting_reduction_backward/FlagGems/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 6条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 21份候选测量、7份请求源码；详细位置列在下表 |
| 交付终点 | 1.349231× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `night-20261021/results/value_selecting_reduction_backward/baseline/FlagGems-1.result.json` | RUNTIME_ERROR | [e67c3e9aabe6](../../experiments/lineage/20261117/source-evidence/sources/e67c3e9aabe620904fe8689f0bd958c47da08e79c11de0c608fe8bfc5e738f11.py) |
| 基线目录 | `night-20261021-v2/results/value_selecting_reduction_backward/FlagGems/FlagGems-1.result.json` | 0.765462× | [e67c3e9aabe6](../../experiments/lineage/20261117/source-evidence/sources/e67c3e9aabe620904fe8689f0bd958c47da08e79c11de0c608fe8bfc5e738f11.py) |
| 基线目录 | `night-20261021-v3/results/value_selecting_reduction_backward/FlagGems/FlagGems-1.result.json` | 0.759061× | [e67c3e9aabe6](../../experiments/lineage/20261117/source-evidence/sources/e67c3e9aabe620904fe8689f0bd958c47da08e79c11de0c608fe8bfc5e738f11.py) |
| 候选 | `night-20261021-v3/results/value_selecting_reduction_backward/kg-no-profile/export/result.json` | 1.602074× | 请求源码未定位 |
| 候选 | `night-20261021-v3/results/value_selecting_reduction_backward/kg-no-profile/independent/kg-no-profile-1.result.json` | 1.684950× | [cc149ba4e154](../../experiments/lineage/20261117/source-evidence/sources/cc149ba4e1548d6211763c0920d4778c94dd00f5c573042b6b8827cd034c1372.py) |
| 候选 | `night-20261021-v3/results/value_selecting_reduction_backward/kg-no-profile/independent/kg-no-profile-2.result.json` | 1.680270× | [cc149ba4e154](../../experiments/lineage/20261117/source-evidence/sources/cc149ba4e1548d6211763c0920d4778c94dd00f5c573042b6b8827cd034c1372.py) |
| 候选 | `night-20261021-v3/results/value_selecting_reduction_backward/kg-profile/export/result.json` | 0.685098× | 请求源码未定位 |
| 候选 | `night-20261021-v3/results/value_selecting_reduction_backward/kg-profile/independent/kg-profile-1.result.json` | 0.686091× | [5fd31f53a80e](../../experiments/lineage/20261117/source-evidence/sources/5fd31f53a80e95810c8d99af296e78b9dc0b7ca452b64015ec63c4b9e1b4a7f7.py) |
| 候选 | `night-20261021-v3/results/value_selecting_reduction_backward/kg-profile/independent/kg-profile-2.result.json` | 0.674585× | [5fd31f53a80e](../../experiments/lineage/20261117/source-evidence/sources/5fd31f53a80e95810c8d99af296e78b9dc0b7ca452b64015ec63c4b9e1b4a7f7.py) |
| 候选 | `night-20261021-v3/results/value_selecting_reduction_backward/HG-KDA/export/result.json` | 1.550185× | 请求源码未定位 |
| 候选 | `night-20261021-v3/results/value_selecting_reduction_backward/HG-KDA/independent/HG-KDA-1.result.json` | 1.524445× | [e89318674857](../../experiments/lineage/20261117/source-evidence/sources/e893186748573e4908a2d4df050d6147322c95979344f714a50bccf93b0f2bb3.py) |
| 候选 | `night-20261021-v3/results/value_selecting_reduction_backward/HG-KDA/independent/HG-KDA-2.result.json` | 1.540461× | [e89318674857](../../experiments/lineage/20261117/source-evidence/sources/e893186748573e4908a2d4df050d6147322c95979344f714a50bccf93b0f2bb3.py) |
| 候选 | `night-20261021-v3-value-stride-repair/results/value_selecting_reduction_backward/codex-stride-repair/independent/codex-stride-repair-1.result.json` | 1.593249× | [aaf74e7b4c22](../../experiments/lineage/20261117/source-evidence/sources/aaf74e7b4c22ef3941480023b67c16a52ec81f9b5071151832cc492d985d532e.py) |
| 候选 | `night-20261021-v3-value-stride-repair/results/value_selecting_reduction_backward/codex-stride-repair/independent/codex-stride-repair-2.result.json` | 1.678798× | [aaf74e7b4c22](../../experiments/lineage/20261117/source-evidence/sources/aaf74e7b4c22ef3941480023b67c16a52ec81f9b5071151832cc492d985d532e.py) |
| 候选 | `night-20261021-v3-value-profile-repair/results/value_selecting_reduction_backward/codex-stride-repair/independent/codex-stride-repair-1.result.json` | 1.043982× | [98b4cb7d62b4](../../experiments/lineage/20261117/source-evidence/sources/98b4cb7d62b4481202ff6e14c05f32002e5ab629c991b2ee883f0180180ea88a.py) |
| 候选 | `night-20261021-v3-value-profile-repair/results/value_selecting_reduction_backward/codex-stride-repair/independent/codex-stride-repair-2.result.json` | 1.103174× | [98b4cb7d62b4](../../experiments/lineage/20261117/source-evidence/sources/98b4cb7d62b4481202ff6e14c05f32002e5ab629c991b2ee883f0180180ea88a.py) |
| 候选 | `night-20261021-v3-value-hg-repair/results/value_selecting_reduction_backward/codex-stride-repair/independent/codex-stride-repair-1.result.json` | 1.465486× | [2841d7b4cba6](../../experiments/lineage/20261117/source-evidence/sources/2841d7b4cba65d8c0c53a61032121aeb3585cac9a0bed063ef2ec5f401bdbb66.py) |
| 候选 | `night-20261021-v3-value-hg-repair/results/value_selecting_reduction_backward/codex-stride-repair/independent/codex-stride-repair-2.result.json` | 1.407712× | [2841d7b4cba6](../../experiments/lineage/20261117/source-evidence/sources/2841d7b4cba65d8c0c53a61032121aeb3585cac9a0bed063ef2ec5f401bdbb66.py) |
| 候选 | `night-20261021-v3-value-hg-profile-guided/results/value_selecting_reduction_backward/codex-stride-repair/independent/codex-stride-repair-1.result.json` | 1.515996× | [b816562294fd](../../experiments/lineage/20261117/source-evidence/sources/b816562294fd68089c734cd125bd28de053ce1e0185c7af79d84cbb792664538.py) |
| 候选 | `night-20261021-v3-value-hg-profile-guided/results/value_selecting_reduction_backward/codex-stride-repair/independent/codex-stride-repair-2.result.json` | 1.476723× | [b816562294fd](../../experiments/lineage/20261117/source-evidence/sources/b816562294fd68089c734cd125bd28de053ce1e0185c7af79d84cbb792664538.py) |
| 候选 | `threeway-20261030/results/value_selecting_reduction_backward/current-recheck/formal/HG-KDA-current-1.result.json` | 1.479025× | [aaf74e7b4c22](../../experiments/lineage/20261117/source-evidence/sources/aaf74e7b4c22ef3941480023b67c16a52ec81f9b5071151832cc492d985d532e.py) |
| 厂商参考 | `cann-priority-20261102/references/value_selecting_reduction_backward/extra-v1/formal/CANN-1.result.json` | 1.213163× | [4d56b7579259](../../experiments/lineage/20261117/source-evidence/sources/4d56b75792594cd94bc5765d8a9ab0bcb75573618484920919e114955f677c5c.py) |
| 候选 | `cann-priority-20261102/comparisons/value_selecting_reduction_backward/starting/HG-KDA-start-1.result.json` | 1.469776× | [aaf74e7b4c22](../../experiments/lineage/20261117/source-evidence/sources/aaf74e7b4c22ef3941480023b67c16a52ec81f9b5071151832cc492d985d532e.py) |
| 候选 | `strict-triton-20261102/candidates/value_selecting_reduction_backward/restored-v1/pytorch/strict-PyTorch-1.result.json` | 1.675843× | [aaf74e7b4c22](../../experiments/lineage/20261117/source-evidence/sources/aaf74e7b4c22ef3941480023b67c16a52ec81f9b5071151832cc492d985d532e.py) |
| 候选 | `independent-regression-20261109-v3/operators/value_selecting_reduction_backward/evaluation/result.json` | 1.349231× | [aaf74e7b4c22](../../experiments/lineage/20261117/source-evidence/sources/aaf74e7b4c22ef3941480023b67c16a52ec81f9b5071151832cc492d985d532e.py) |

## 交付终点

- [选用源码](../../../operators/value_selecting_reduction_backward/versions/independent_strict/retained/kernel.py)：`aaf74e7b4c22ef3941480023b67c16a52ec81f9b5071151832cc492d985d532e`。
- [保留版完整结果](../../../operators/value_selecting_reduction_backward/evidence/retained/evaluation/result.json)：**1.349231× CANN，96/96**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

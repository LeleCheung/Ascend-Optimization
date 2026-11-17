# scatter_add

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/scatter_add.py)；SHA `88a67a2640ca0f522045df139431e18016e2d7dc36d77fb6eff488d61acbd163`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/e249f5c57930cc37a67d58c754131a55ed6c296575763a560ac889066d38de09.py)；原始模块AST一致，追加入口：`run = scatter_add`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/00f98bc99e240c7b970ce2eae66ae955c7edc1cf54dfbdc7528c53a7c14d820b.json)：**RUNTIME_ERROR**；0/18。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/e0de100ee579e7744db68fb2d2552866a62f2bea9ee64c9c6545aec9b9fa2bd0.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/night-20261021-v3/results/scatter_add/FlagGems/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 10条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 14份候选测量、7份请求源码；详细位置列在下表 |
| 交付终点 | 1.891875× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `night-20261021-v3/results/scatter_add/FlagGems/FlagGems-1.result.json` | RUNTIME_ERROR | [e249f5c57930](../../experiments/lineage/20261117/source-evidence/sources/e249f5c57930cc37a67d58c754131a55ed6c296575763a560ac889066d38de09.py) |
| 基线目录 | `night-20261021-v3-adapter-repairs/results/scatter_add/FlagGems/FlagGems-1.result.json` | 0.257948× | [6b9c58d5f429](../../experiments/lineage/20261117/source-evidence/sources/6b9c58d5f429dca455e5948bb22ca00bd8c67e500868b08453cb675190881b25.py) |
| 候选 | `endurance-20261022/results/scatter_add/parent/formal/parent-1.result.json` | RUNTIME_ERROR | [e249f5c57930](../../experiments/lineage/20261117/source-evidence/sources/e249f5c57930cc37a67d58c754131a55ed6c296575763a560ac889066d38de09.py) |
| 候选 | `endurance-20261022/results/scatter_add/attempt-2/candidates/a54967db3aba4570bb40b7aa3766400cc886e918bcbc6026b7af39e93581bbca/formal/round-0002-1.result.json` | 0.407843× | [a54967db3aba](../../experiments/lineage/20261117/source-evidence/sources/a54967db3aba4570bb40b7aa3766400cc886e918bcbc6026b7af39e93581bbca.py) |
| 候选 | `endurance-20261022/results/scatter_add/attempt-2/candidates/7cb10fae0b6de198338c77ec1bf4dbeebee6912e92e9ea393994a793a61ddbf1/formal/round-0003-1.result.json` | 0.418874× | [7cb10fae0b6d](../../experiments/lineage/20261117/source-evidence/sources/7cb10fae0b6de198338c77ec1bf4dbeebee6912e92e9ea393994a793a61ddbf1.py) |
| 候选 | `endurance-20261022/results/scatter_add/attempt-2/candidates/442657efaf33eecb8a234de3bd668e6631742a3142d1a115c6c98954604368dd/formal/round-0001-1.result.json` | 0.366305× | [442657efaf33](../../experiments/lineage/20261117/source-evidence/sources/442657efaf33eecb8a234de3bd668e6631742a3142d1a115c6c98954604368dd.py) |
| 候选 | `remaining-five-20261023-v2/results/scatter_add/parent/formal/parent-1.result.json` | 0.410128× | [7cb10fae0b6d](../../experiments/lineage/20261117/source-evidence/sources/7cb10fae0b6de198338c77ec1bf4dbeebee6912e92e9ea393994a793a61ddbf1.py) |
| 候选 | `remaining-five-20261023-v2/results/scatter_add/attempt-1/candidates/8a67a004fd75f9332094bcc3cc821e9d2dfa31dbcc10b474ac27937da9c1a04f/formal/round-0003-1.result.json` | 1.306463× | [8a67a004fd75](../../experiments/lineage/20261117/source-evidence/sources/8a67a004fd75f9332094bcc3cc821e9d2dfa31dbcc10b474ac27937da9c1a04f.py) |
| 候选 | `remaining-five-20261023-v2/results/scatter_add/attempt-1/candidates/6ac7431b80ffb93193bbc906d93d840014d958aabea1767093ff2d30a91555ee/formal/round-0002-1.result.json` | 0.271274× | [6ac7431b80ff](../../experiments/lineage/20261117/source-evidence/sources/6ac7431b80ffb93193bbc906d93d840014d958aabea1767093ff2d30a91555ee.py) |
| 候选 | `scatter-segmented-20261023/results/scatter_add/codex-manual/candidates/72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e/formal/codex-manual-1.result.json` | 1.899640× | [72c4d40d831f](../../experiments/lineage/20261117/source-evidence/sources/72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e.py) |
| 候选 | `scatter-segmented-20261023/results/scatter_add/parent/formal/parent-1.result.json` | 1.278404× | [8a67a004fd75](../../experiments/lineage/20261117/source-evidence/sources/8a67a004fd75f9332094bcc3cc821e9d2dfa31dbcc10b474ac27937da9c1a04f.py) |
| 候选 | `threeway-20261030/results/scatter_add/current-recheck/formal/HG-KDA-current-1.result.json` | 1.852521× | [72c4d40d831f](../../experiments/lineage/20261117/source-evidence/sources/72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e.py) |
| 厂商参考 | `cann-priority-20261102/references/scatter_add/formal/CANN-1.result.json` | 1.009375× | [dbe4fb416938](../../experiments/lineage/20261117/source-evidence/sources/dbe4fb416938097cc3883f58f56ac03d50f05ef9dc26d99ffc83d3bd7803fc59.py) |
| 候选 | `cann-priority-20261102/comparisons/scatter_add/starting/HG-KDA-start-1.result.json` | 1.885981× | [72c4d40d831f](../../experiments/lineage/20261117/source-evidence/sources/72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e.py) |
| 候选 | `strict-triton-20261102/candidates/scatter_add/restored-v1/pytorch/strict-PyTorch-1.result.json` | 1.880033× | [72c4d40d831f](../../experiments/lineage/20261117/source-evidence/sources/72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e.py) |
| 候选 | `independent-regression-20261109-v3/operators/scatter_add/evaluation/result.json` | 1.891875× | [72c4d40d831f](../../experiments/lineage/20261117/source-evidence/sources/72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e.py) |
| 候选 | `codex-sample-retest-20261116/scatter_add/formal-72c4d40d831f/evaluation/result.json` | 1.815734× | [72c4d40d831f](../../experiments/lineage/20261117/source-evidence/sources/72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e.py) |

## 交付终点

- [选用源码](../../../operators/scatter_add/versions/independent_strict/retained/kernel.py)：`72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e`。
- [保留版完整结果](../../../operators/scatter_add/evidence/retained/evaluation/result.json)：**1.891875× CANN，18/18**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

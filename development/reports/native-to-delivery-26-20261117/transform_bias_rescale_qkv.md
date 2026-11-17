# transform_bias_rescale_qkv

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/_transform_bias_rescale_qkv.py)；SHA `0dd84ef70edfedb4836fa3f37dff8000406c89ff997dd82a7f0a32f79a3bca77`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/f84aac43f179f7cc19dbe24d2156dd35c2d3d00f2d152ea3661ad5ee675d4f76.py)；原始模块AST一致，追加入口：`run = _transform_bias_rescale_qkv`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/588da0cadd5ae58a12fc3e0c8d5c79a948e51f320b0e45b78911b88c33eca99a.json)：**0.004×**；21/21。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/e3757f2bea7468d93b7fff6f76113b64ef5d1f49f47f18ab0be7a4fc1dc38b68.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/transform_bias_rescale_qkv/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 0条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 11份候选测量、5份请求源码；详细位置列在下表 |
| 交付终点 | 0.941333× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/transform_bias_rescale_qkv/baseline-fixed/FlagGems-1.result.json` | 0.004046× | [f84aac43f179](../../experiments/lineage/20261117/source-evidence/sources/f84aac43f179f7cc19dbe24d2156dd35c2d3d00f2d152ea3661ad5ee675d4f76.py) |
| 候选 | `next25-20261029-v2/proposals/transform_bias_rescale_qkv/constexpr-layout-v1/assessment/formal/HG-KDA-constexpr-layout-v1-1.result.json` | 0.004447× | [abdb22b2039e](../../experiments/lineage/20261117/source-evidence/sources/abdb22b2039ecc8e35cffbcbe2318e8a0c0af140614fc3f108c04b098a80c881.py) |
| 候选 | `next25-20261029-v2/proposals/transform_bias_rescale_qkv/tiled-2d-launch-v2/assessment/formal/HG-KDA-tiled-2d-launch-v2-1.result.json` | 0.808628× | [f2a888fac572](../../experiments/lineage/20261117/source-evidence/sources/f2a888fac5724c2242053d0a62b41a357bc7eafde0f1979ed25f448d79d847c1.py) |
| 候选 | `next25-20261029-v2/proposals/transform_bias_rescale_qkv/integer-metadata-v3/assessment/formal/HG-KDA-integer-metadata-v3-1.result.json` | 0.951413× | [4817ed46b232](../../experiments/lineage/20261117/source-evidence/sources/4817ed46b232cd08aa653eb4bd0ecc94a2c9fa545b0dcf396919a25721b868d7.py) |
| 候选 | `next25-20261029-v2/proposals/transform_bias_rescale_qkv/fp32-q-accumulation-v4/assessment/formal/HG-KDA-fp32-q-accumulation-v4-1.result.json` | 0.923668× | [b1d4e1f6f744](../../experiments/lineage/20261117/source-evidence/sources/b1d4e1f6f744fcf96989976d6382fff8ac0149cb66dc14a762dd504f54b58a1c.py) |
| 候选 | `threeway-20261030/results/transform_bias_rescale_qkv/current-recheck/formal/HG-KDA-current-1.result.json` | 0.916327× | [b1d4e1f6f744](../../experiments/lineage/20261117/source-evidence/sources/b1d4e1f6f744fcf96989976d6382fff8ac0149cb66dc14a762dd504f54b58a1c.py) |
| 候选 | `cann-priority-20261102/proposals/transform_bias_rescale_qkv/cpp-host-v1/original/HG-KDA-new-1.result.json` | 1.042504× | [e6510dd6b1aa](../../experiments/lineage/20261117/source-evidence/sources/e6510dd6b1aa67096ca1bcfdcc9650bfe49cbe9e187cc9f799e1917754b466af.py) |
| 候选 | `cann-priority-20261102/proposals/transform_bias_rescale_qkv/cpp-host-v1/CANN/HG-KDA-new-vs-CANN-1.result.json` | 0.931237× | [e6510dd6b1aa](../../experiments/lineage/20261117/source-evidence/sources/e6510dd6b1aa67096ca1bcfdcc9650bfe49cbe9e187cc9f799e1917754b466af.py) |
| 厂商参考 | `cann-priority-20261102/references/transform_bias_rescale_qkv/formal/CANN-1.result.json` | 1.274148× | [d342568a5439](../../experiments/lineage/20261117/source-evidence/sources/d342568a543939fd063ff00f903b0cec86f115061edf31ac8e53dd59e7828e47.py) |
| 候选 | `cann-priority-20261102/comparisons/transform_bias_rescale_qkv/starting/HG-KDA-start-1.result.json` | 0.784886× | [b1d4e1f6f744](../../experiments/lineage/20261117/source-evidence/sources/b1d4e1f6f744fcf96989976d6382fff8ac0149cb66dc14a762dd504f54b58a1c.py) |
| 候选 | `strict-triton-20261102/candidates/transform_bias_rescale_qkv/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.983017× | [b1d4e1f6f744](../../experiments/lineage/20261117/source-evidence/sources/b1d4e1f6f744fcf96989976d6382fff8ac0149cb66dc14a762dd504f54b58a1c.py) |
| 候选 | `strict-triton-20261102/candidates/transform_bias_rescale_qkv/restored-v1/CANN/strict-CANN-1.result.json` | 0.745411× | [b1d4e1f6f744](../../experiments/lineage/20261117/source-evidence/sources/b1d4e1f6f744fcf96989976d6382fff8ac0149cb66dc14a762dd504f54b58a1c.py) |
| 候选 | `independent-regression-20261109-v3/operators/transform_bias_rescale_qkv/evaluation/result.json` | 0.941333× | [e6510dd6b1aa](../../experiments/lineage/20261117/source-evidence/sources/e6510dd6b1aa67096ca1bcfdcc9650bfe49cbe9e187cc9f799e1917754b466af.py) |

## 交付终点

- [选用源码](../../../operators/transform_bias_rescale_qkv/versions/independent_strict/retained/kernel.py)：`e6510dd6b1aa67096ca1bcfdcc9650bfe49cbe9e187cc9f799e1917754b466af`。
- [保留版完整结果](../../../operators/transform_bias_rescale_qkv/evidence/retained/evaluation/result.json)：**0.941333× CANN，21/21**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

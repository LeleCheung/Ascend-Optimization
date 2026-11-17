# fake_quantize_per_tensor_affine_cachemask_out

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/fake_quantize_per_tensor_affine_cachemask.py)；SHA `37f40726730d8be15c477428b5ca62724572ea9fd1da6263de2b25666f90a93a`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/ca32ba637f8702ad7209845db319c7a796b297fbf73725ac92fd1faf7c6236ac.py)；原始模块AST一致，追加入口：`run = fake_quantize_per_tensor_affine_cachemask_out`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/42d1ce2e3a998e814c4ba6d8051ee05d0f399baa895052c4b012c9fad204514b.json)：**8.651×**；30/30。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/f1e5b57f9398e3c2a74d384cf95f02b3bea343dfcc07c6bf1cae7799467d522f.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/results/fake_quantize_per_tensor_affine_cachemask_out/baseline-fixed/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 0条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 5份候选测量、2份请求源码；详细位置列在下表 |
| 交付终点 | 8.154899× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `next25-20261029-v2/results/fake_quantize_per_tensor_affine_cachemask_out/baseline-fixed/FlagGems-1.result.json` | 8.651451× | [ca32ba637f87](../../experiments/lineage/20261117/source-evidence/sources/ca32ba637f8702ad7209845db319c7a796b297fbf73725ac92fd1faf7c6236ac.py) |
| 候选 | `reference-closure-20261031/comparisons/fake_quantize_per_tensor_affine_cachemask_out/CANN-direct/HG-KDA-vs-CANN-1.result.json` | 6.554236× | [ca32ba637f87](../../experiments/lineage/20261117/source-evidence/sources/ca32ba637f8702ad7209845db319c7a796b297fbf73725ac92fd1faf7c6236ac.py) |
| 厂商参考 | `reference-closure-20261031/references-v2/fake_quantize_per_tensor_affine_cachemask_out/formal/CANN-direct-v2-1.result.json` | 4.515022× | [3e61a1434218](../../experiments/lineage/20261117/source-evidence/sources/3e61a14342182abac90c322f4286db29fe7c11fbeebf310175f3fd74b546bb6a.py) |
| 厂商参考 | `cann-priority-20261102/references/fake_quantize_per_tensor_affine_cachemask_out/formal/CANN-1.result.json` | 2.810848× | [3e61a1434218](../../experiments/lineage/20261117/source-evidence/sources/3e61a14342182abac90c322f4286db29fe7c11fbeebf310175f3fd74b546bb6a.py) |
| 候选 | `cann-priority-20261102/comparisons/fake_quantize_per_tensor_affine_cachemask_out/starting/HG-KDA-start-1.result.json` | 6.974652× | [ca32ba637f87](../../experiments/lineage/20261117/source-evidence/sources/ca32ba637f8702ad7209845db319c7a796b297fbf73725ac92fd1faf7c6236ac.py) |
| 候选 | `strict-triton-20261102/candidates/fake_quantize_per_tensor_affine_cachemask_out/restored-v1/pytorch/strict-PyTorch-1.result.json` | 9.622916× | [48155ad768e5](../../experiments/lineage/20261117/source-evidence/sources/48155ad768e544588dfe211443cb2cdb65dedc49b1b7d3051634b6da09be7b36.py) |
| 候选 | `strict-triton-20261102/candidates/fake_quantize_per_tensor_affine_cachemask_out/restored-v1/CANN/strict-CANN-1.result.json` | 7.060216× | [48155ad768e5](../../experiments/lineage/20261117/source-evidence/sources/48155ad768e544588dfe211443cb2cdb65dedc49b1b7d3051634b6da09be7b36.py) |
| 候选 | `independent-regression-20261109-v3/operators/fake_quantize_per_tensor_affine_cachemask_out/evaluation/result.json` | 8.154899× | [48155ad768e5](../../experiments/lineage/20261117/source-evidence/sources/48155ad768e544588dfe211443cb2cdb65dedc49b1b7d3051634b6da09be7b36.py) |

## 交付终点

- [选用源码](../../../operators/fake_quantize_per_tensor_affine_cachemask_out/versions/independent_strict/retained/kernel.py)：`48155ad768e544588dfe211443cb2cdb65dedc49b1b7d3051634b6da09be7b36`。
- [保留版完整结果](../../../operators/fake_quantize_per_tensor_affine_cachemask_out/evidence/retained/evaluation/result.json)：**8.154899× CANN，30/30**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

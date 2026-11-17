# selu_

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/selu_.py)；SHA `8c61a96160e558e75e52212dea952c0e71b1c7371d7d050152bb266c3a0a66fe`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/3cdfa6f384175eeae338a223d760833c8e44d99f3eace8a9fdd0faa85a0c48d5.py)；原始模块AST一致，追加入口：`_baseline_selu_ = selu_; def run(A):     return _baseline_selu_(A); selu_ = run`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/6cfd37743a9606de192fee96a57ad7c4f845d96cef8d6972e6d3239e9ec6ead6.json)：**0.119×**；36/36。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/5edbcd4933df9fb2898a92cc843c11ce862a74d9c4518ef6f13ff33204d7de93.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/night-20261021-v3/results/selu_/FlagGems/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 10条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 30份候选测量、17份请求源码；详细位置列在下表 |
| 交付终点 | 1.039292× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `night-20261021-v2/results/selu_/FlagGems/FlagGems-1.result.json` | RUNTIME_ERROR | [fa436e2d68e7](../../experiments/lineage/20261117/source-evidence/sources/fa436e2d68e7fd325fb1d718c6b61881d5a6cc9b00b68a4d06d2845bc3102497.py) |
| 基线目录 | `night-20261021-v3/results/selu_/FlagGems/FlagGems-1.result.json` | 0.118621× | [3cdfa6f38417](../../experiments/lineage/20261117/source-evidence/sources/3cdfa6f384175eeae338a223d760833c8e44d99f3eace8a9fdd0faa85a0c48d5.py) |
| 候选 | `night-20261021-v3/results/selu_/HG-KDA/export/result.json` | 0.130370× | 请求源码未定位 |
| 候选 | `night-20261021-v3/results/selu_/HG-KDA/independent/HG-KDA-1.result.json` | 0.131483× | [cec7c0259644](../../experiments/lineage/20261117/source-evidence/sources/cec7c0259644289a130f84587b6ec1a605979ddb10a4acf4fd710b7c79fe44b0.py) |
| 候选 | `night-20261021-v3/results/selu_/HG-KDA/independent/HG-KDA-2.result.json` | 0.124547× | [cec7c0259644](../../experiments/lineage/20261117/source-evidence/sources/cec7c0259644289a130f84587b6ec1a605979ddb10a4acf4fd710b7c79fe44b0.py) |
| 候选 | `night-20261021-v3-selu-launch-repair/results/selu_/codex-launch-repair/independent/codex-launch-repair-1.result.json` | 0.235649× | [2d58eb89ded1](../../experiments/lineage/20261117/source-evidence/sources/2d58eb89ded17eb0343a56c2ca2e8129cc172c17490b07c15f4b61f28e56d10f.py) |
| 候选 | `night-20261021-v3-selu-launch-repair/results/selu_/codex-launch-repair/independent/codex-launch-repair-2.result.json` | 0.242951× | [2d58eb89ded1](../../experiments/lineage/20261117/source-evidence/sources/2d58eb89ded17eb0343a56c2ca2e8129cc172c17490b07c15f4b61f28e56d10f.py) |
| 候选 | `paired-pointwise-20261022/results/selu_/codex-rawptr-v1/formal/codex-rawptr-v1-1.result.json` | 0.768048× | [c212a8f40def](../../experiments/lineage/20261117/source-evidence/sources/c212a8f40def1789e16b94c67aff067ba44c138ebc9bbe9852283067de6458c0.py) |
| 候选 | `paired-pointwise-20261022/results/selu_/paired-HG-KDA/export/result.json` | 1.001557× | 请求源码未定位 |
| 候选 | `paired-pointwise-20261022/results/selu_/codex-profilemeta-v3/formal/codex-profilemeta-v3-1.result.json` | 0.800904× | [8925efd82148](../../experiments/lineage/20261117/source-evidence/sources/8925efd82148096d0b0f74afb879b2c9a9529ae410bbd6e75ffc67261858f8fe.py) |
| 候选 | `paired-pointwise-20261022/results/selu_/codex-profilemeta-v4/formal/codex-profilemeta-v4-1.result.json` | 0.790002× | [49bdbb789d49](../../experiments/lineage/20261117/source-evidence/sources/49bdbb789d497b6423f893f3433167f3fd4bc61197676236bd62a9ec79c7dd91.py) |
| 候选 | `buffer-pointwise-20261022/results/selu_/paired-HG-KDA/export/result.json` | 0.883912× | 请求源码未定位 |
| 候选 | `buffer-pointwise-20261022/results/selu_/paired-HG-KDA/formal/paired-HG-KDA-1.result.json` | 0.757489× | [9ac565b2a89f](../../experiments/lineage/20261117/source-evidence/sources/9ac565b2a89fc4f4da3c38bfd231cf2da46e57705344d596d51ac27a34c27aa8.py) |
| 候选 | `buffer-pointwise-20261022/results/selu_/codex-parent/formal/codex-parent-1.result.json` | 0.898159× | [49bdbb789d49](../../experiments/lineage/20261117/source-evidence/sources/49bdbb789d497b6423f893f3433167f3fd4bc61197676236bd62a9ec79c7dd91.py) |
| 候选 | `endurance-20261022/results/selu_/parent/formal/parent-1.result.json` | 0.868166× | [49bdbb789d49](../../experiments/lineage/20261117/source-evidence/sources/49bdbb789d497b6423f893f3433167f3fd4bc61197676236bd62a9ec79c7dd91.py) |
| 候选 | `endurance-20261022/results/selu_/attempt-2/candidates/0e782a69158783eb900458c93ca3531737ecaa2579739621ffcc9b3415ac7622/formal/round-0003-1.result.json` | 0.970350× | [0e782a691587](../../experiments/lineage/20261117/source-evidence/sources/0e782a69158783eb900458c93ca3531737ecaa2579739621ffcc9b3415ac7622.py) |
| 候选 | `endurance-20261022/results/selu_/attempt-2/candidates/8764018cfb2cab9444a34caf7a171f7eac7395656631698ce491e676557462f0/formal/round-0002-1.result.json` | 0.837208× | [8764018cfb2c](../../experiments/lineage/20261117/source-evidence/sources/8764018cfb2cab9444a34caf7a171f7eac7395656631698ce491e676557462f0.py) |
| 基线目录 | `mechanism-pilot-20261026/results/selu_/FlagGems/baseline-1.result.json` | RUNTIME_ERROR | [8a39a8d5c0b2](../../experiments/lineage/20261117/source-evidence/sources/8a39a8d5c0b2c3159e33566742d5047a5025230cf5851345059833f2a65c0bdb.py) |
| 基线目录 | `mechanism-pilot-20261026-v2/results/selu_/FlagGems/baseline-1.result.json` | 0.113873× | [b87724ca16e0](../../experiments/lineage/20261117/source-evidence/sources/b87724ca16e0c7b21512857da9757113dbe3e2384203c3b3c54f2d07e19ab373.py) |
| 候选 | `mechanism-pilot-20261026-v2/results/selu_/raw_evidence/formal/raw_evidence-1.result.json` | 0.098625× | [1a11b6cff733](../../experiments/lineage/20261117/source-evidence/sources/1a11b6cff7337359b2ddaf7ec6c551975e019472d7c7b9e56479b1daee711e3b.py) |
| 基线目录 | `mechanism-compiler-20261026-v2/results/selu_/FlagGems/baseline-1.result.json` | 0.105837× | [ff49ce746942](../../experiments/lineage/20261117/source-evidence/sources/ff49ce74694205c20e6c5793361b8cd47ee7089ac1f434fb6d103c990621af72.py) |
| 候选 | `mechanism-compiler-20261026-v2/results/selu_/raw_evidence/formal/raw_evidence-1.result.json` | 0.082549× | [2361d1468207](../../experiments/lineage/20261117/source-evidence/sources/2361d14682075a27cce480abadd06076c84723c390edcff874f5b5b938e6bf36.py) |
| 候选 | `mechanism-compiler-20261026-v2/results/selu_/mechanism_rewrite/formal/mechanism_rewrite-1.result.json` | 0.151077× | [d1a039e481b9](../../experiments/lineage/20261117/source-evidence/sources/d1a039e481b9742920c472879e6d628e5456ce3be986858ec308795da132cd6a.py) |
| 基线目录 | `next25-20261029-v2/results/selu_/baseline-fixed/FlagGems-1.result.json` | RUNTIME_ERROR | [7a32c4090d89](../../experiments/lineage/20261117/source-evidence/sources/7a32c4090d89708589b7062e71acd6235b6956a26e5c5d525a4d3e8a19ff73a1.py) |
| 基线目录 | `next25-20261029-v2/results/selu_/baseline-entry-294d34200fbf/FlagGems-entry-1.result.json` | 0.135433× | [b71f56a114f9](../../experiments/lineage/20261117/source-evidence/sources/b71f56a114f9a1d14343c17b7c3160bcd3efc436efd15de7a16c0595f72225d5.py) |
| 候选 | `next25-20261029-v2/results/selu_/historical-start/HG-KDA-start-1.result.json` | 0.905131× | [8764018cfb2c](../../experiments/lineage/20261117/source-evidence/sources/8764018cfb2cab9444a34caf7a171f7eac7395656631698ce491e676557462f0.py) |
| 候选 | `threeway-20261030/results/selu_/current-recheck/formal/HG-KDA-current-1.result.json` | 0.903663× | [8764018cfb2c](../../experiments/lineage/20261117/source-evidence/sources/8764018cfb2cab9444a34caf7a171f7eac7395656631698ce491e676557462f0.py) |
| 厂商参考 | `cann-priority-20261102/references/selu_/formal/CANN-1.result.json` | 0.881463× | [98fc8c041a1b](../../experiments/lineage/20261117/source-evidence/sources/98fc8c041a1b6c8af61ec099c6a0eb626c44265eed8627c081fc46ac1fb54b98.py) |
| 候选 | `cann-priority-20261102/comparisons/selu_/starting/HG-KDA-start-1.result.json` | 0.981745× | [8764018cfb2c](../../experiments/lineage/20261117/source-evidence/sources/8764018cfb2cab9444a34caf7a171f7eac7395656631698ce491e676557462f0.py) |
| 候选 | `strict-triton-20261102/candidates/selu_/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.922347× | [8764018cfb2c](../../experiments/lineage/20261117/source-evidence/sources/8764018cfb2cab9444a34caf7a171f7eac7395656631698ce491e676557462f0.py) |
| 候选 | `independent-regression-20261109-v3/operators/selu_/evaluation/result.json` | 1.039292× | [8764018cfb2c](../../experiments/lineage/20261117/source-evidence/sources/8764018cfb2cab9444a34caf7a171f7eac7395656631698ce491e676557462f0.py) |
| 候选 | `independent-skill-selu-20261109-v1/formal-f3502fad5259/operators/selu_/evaluation/result.json` | RUNTIME_ERROR | [f3502fad5259](../../experiments/lineage/20261117/source-evidence/sources/f3502fad52595cc909e47a5a158ce16edcdfdd8aa11b2f5c7657ff367e46d288.py) |
| 候选 | `independent-skill-selu-20261109-v1/formal-3cec1212c3b7/operators/selu_/evaluation/result.json` | 0.210037× | [3cec1212c3b7](../../experiments/lineage/20261117/source-evidence/sources/3cec1212c3b7630bbf548188693f4cd9fef8747ee78cb1f822a42dcf685a23ed.py) |
| 候选 | `selu-skill-recovery-20261109-v2/formal-c6255855aa16/operators/selu_/evaluation/result.json` | 0.312005× | [c6255855aa16](../../experiments/lineage/20261117/source-evidence/sources/c6255855aa162470c0e9c6405c435eb485992d3e4b1cb1f85c793d2b745665c6.py) |
| 候选 | `selu-skill-recovery-20261109-v2/formal-80bfefff4ff8/operators/selu_/evaluation/result.json` | 0.433299× | [80bfefff4ff8](../../experiments/lineage/20261117/source-evidence/sources/80bfefff4ff837b1771eef1618199d56885ff3e950f02a6fc14dbe09ec3d3aca.py) |
| 候选 | `selu-skill-recovery-20261109-v2/formal-0fed8d43398d/operators/selu_/evaluation/result.json` | 0.710539× | [0fed8d43398d](../../experiments/lineage/20261117/source-evidence/sources/0fed8d43398d2b3f1ceb50e55adea1bfc9fe906028f85145aa7cc9e0dce82626.py) |
| 候选 | `selu-skill-recovery-20261109-v2/formal-5e9e4d482620/operators/selu_/evaluation/result.json` | 0.887010× | [5e9e4d482620](../../experiments/lineage/20261117/source-evidence/sources/5e9e4d4826207ea03e70d01e871c71220e40f9097ad0a0f87da977f793573115.py) |
| 候选 | `overnight-regions-20261110-v1/entry-regression/operators/selu_/evaluation/result.json` | 0.949923× | [5e9e4d482620](../../experiments/lineage/20261117/source-evidence/sources/5e9e4d4826207ea03e70d01e871c71220e40f9097ad0a0f87da977f793573115.py) |

## 交付终点

- [选用源码](../../../operators/selu_/versions/independent_strict/retained/kernel.py)：`8764018cfb2cab9444a34caf7a171f7eac7395656631698ce491e676557462f0`。
- [保留版完整结果](../../../operators/selu_/evidence/retained/evaluation/result.json)：**1.039292× CANN，36/36**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

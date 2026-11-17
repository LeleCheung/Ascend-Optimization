# log10_

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/log10.py)；SHA `17399ef67e0bbacff2b69e83c3f8f711f944c752b8328b9f352a396f819d8ea3`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/982a5a66a4cf1232e299f7656c4fca5f43d1c306ec87c5ea3ebadc32bdd0bab5.py)；原始模块AST一致，追加入口：`run = log10_`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/576c1526e21083436ce8e7793d69f343cadc5ef4e609e1851133374fa9967800.json)：**PARTIAL_PASS**；33/36。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/9bf18e5c7dfa0733256709368ad96f146a0e67b3f92bdbd44970139926d425e1.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/night-20261021-v3/results/log10_/FlagGems/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 9条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 44份候选测量、32份请求源码；详细位置列在下表 |
| 交付终点 | 0.940911× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `night-20261021-v2/results/log10_/FlagGems/FlagGems-1.result.json` | PARTIAL_PASS | [982a5a66a4cf](../../experiments/lineage/20261117/source-evidence/sources/982a5a66a4cf1232e299f7656c4fca5f43d1c306ec87c5ea3ebadc32bdd0bab5.py) |
| 基线目录 | `night-20261021-v3/results/log10_/FlagGems/FlagGems-1.result.json` | PARTIAL_PASS | [982a5a66a4cf](../../experiments/lineage/20261117/source-evidence/sources/982a5a66a4cf1232e299f7656c4fca5f43d1c306ec87c5ea3ebadc32bdd0bab5.py) |
| 候选 | `night-20261021-v3/results/log10_/HG-KDA/export/result.json` | 0.601965× | 请求源码未定位 |
| 候选 | `night-20261021-v3/results/log10_/HG-KDA/independent/HG-KDA-1.result.json` | 0.618991× | [d4ab85095c59](../../experiments/lineage/20261117/source-evidence/sources/d4ab85095c59c2f4a66180860766280baa9eebc951739f9b04c5b218a21627b6.py) |
| 候选 | `night-20261021-v3/results/log10_/HG-KDA/independent/HG-KDA-2.result.json` | 0.539464× | [d4ab85095c59](../../experiments/lineage/20261117/source-evidence/sources/d4ab85095c59c2f4a66180860766280baa9eebc951739f9b04c5b218a21627b6.py) |
| 候选 | `paired-pointwise-20261022/results/log10_/codex-rawptr-v1/formal/codex-rawptr-v1-1.result.json` | 0.802081× | [979272c2a842](../../experiments/lineage/20261117/source-evidence/sources/979272c2a842ce3fe4cbc97a231cdf1513cfaeb6e6b8dc3dd57d943533373578.py) |
| 候选 | `paired-pointwise-20261022/results/log10_/paired-HG-KDA/export/result.json` | 0.723264× | 请求源码未定位 |
| 候选 | `paired-pointwise-20261022/results/log10_/paired-HG-KDA/formal/paired-HG-KDA-1.result.json` | 0.845907× | [c4ebec907f02](../../experiments/lineage/20261117/source-evidence/sources/c4ebec907f02a509b94000b0b8e7a7a8ba061560e12d5a6e8473f3d1d5385ab2.py) |
| 候选 | `paired-pointwise-20261022/results/log10_/codex-profilemeta-v3/formal/codex-profilemeta-v3-1.result.json` | 0.707091× | [08e6f1fd7d63](../../experiments/lineage/20261117/source-evidence/sources/08e6f1fd7d634edc3d6f6d0487bfac8fe799c5ae8a76acf34b9d43d032980b13.py) |
| 候选 | `paired-pointwise-20261022/results/log10_/codex-profilemeta-v5/formal/codex-profilemeta-v5-1.result.json` | 0.737139× | [c845221e25be](../../experiments/lineage/20261117/source-evidence/sources/c845221e25bed5df6d631eac7554425ce498dc9b7b4bbe2edd6c9903fb498b9a.py) |
| 候选 | `buffer-pointwise-20261022/results/log10_/paired-HG-KDA/export/result.json` | 0.802630× | 请求源码未定位 |
| 候选 | `buffer-pointwise-20261022/results/log10_/paired-HG-KDA/formal/paired-HG-KDA-1.result.json` | 0.874902× | [2b208ae4c2c4](../../experiments/lineage/20261117/source-evidence/sources/2b208ae4c2c4394549f03fc466768c1f280011516f34878ab49998d23825b6fd.py) |
| 候选 | `buffer-pointwise-20261022/results/log10_/codex-mechanism-v3/formal/codex-mechanism-v3-1.result.json` | 0.731257× | [3b13003d5326](../../experiments/lineage/20261117/source-evidence/sources/3b13003d53267e15846438eeba11b459180144c4844013851c8dc7e4fa3add01.py) |
| 候选 | `buffer-pointwise-20261022/results/log10_/codex-pipeline-v4/formal/codex-pipeline-v4-1.result.json` | 0.799979× | [5d775c4d5db2](../../experiments/lineage/20261117/source-evidence/sources/5d775c4d5db24109d09e84cb62ad7e68ac07c1d1284597de35c1521dacb9ec93.py) |
| 候选 | `buffer-pointwise-20261022/results/log10_/codex-parent/formal/codex-parent-1.result.json` | 0.750556× | [c845221e25be](../../experiments/lineage/20261117/source-evidence/sources/c845221e25bed5df6d631eac7554425ce498dc9b7b4bbe2edd6c9903fb498b9a.py) |
| 候选 | `buffer-pointwise-20261022/results/log10_/codex-merged-v8/formal/codex-merged-v8-1.result.json` | 0.742306× | [188b87744ad5](../../experiments/lineage/20261117/source-evidence/sources/188b87744ad5880db044265eaf1326b07d923fa5d6f5554fd44975652239c799.py) |
| 基线目录 | `next25-20261029-v2/results/log10_/baseline-fixed/FlagGems-1.result.json` | PARTIAL_PASS | [982a5a66a4cf](../../experiments/lineage/20261117/source-evidence/sources/982a5a66a4cf1232e299f7656c4fca5f43d1c306ec87c5ea3ebadc32bdd0bab5.py) |
| 候选 | `next25-20261029-v2/results/log10_/historical-start/HG-KDA-start-1.result.json` | 0.760741× | [2b208ae4c2c4](../../experiments/lineage/20261117/source-evidence/sources/2b208ae4c2c4394549f03fc466768c1f280011516f34878ab49998d23825b6fd.py) |
| 候选 | `next25-20261029-v2/results/log10_/search-1/independent/HG-KDA-1.result.json` | 0.706207× | [ff91ae6e4f00](../../experiments/lineage/20261117/source-evidence/sources/ff91ae6e4f0015b4e6d1f178e5207b7a4858e9abbb16c716bf04823f26012380.py) |
| 候选 | `next25-20261029-v2/proposals/log10_/streaming-aligned-v3/assessment/formal/HG-KDA-streaming-aligned-v3-1.result.json` | 0.700560× | [676d746d4b47](../../experiments/lineage/20261117/source-evidence/sources/676d746d4b473e078b66a68064b04e25e4329c0f007ecc9ffcc5db145519e169.py) |
| 候选 | `threeway-20261030/results/log10_/current-recheck/formal/HG-KDA-current-1.result.json` | 0.720745× | [676d746d4b47](../../experiments/lineage/20261117/source-evidence/sources/676d746d4b473e078b66a68064b04e25e4329c0f007ecc9ffcc5db145519e169.py) |
| 候选／不符合当前约束 | `threeway-20261030/proposals/log10_/ascendc-taskqueue-v2/assessment/formal/HG-KDA-ascendc-taskqueue-v2-1.result.json` | 0.933510× | [cb9cf4dcc720](../../experiments/lineage/20261117/source-evidence/sources/cb9cf4dcc7204af3061b7b81644d8390197446eef88db0035162dcdab54d688e.py) |
| 候选 | `threeway-20261030/native-candidates/log10_/ascendc-ln-v1/formal/HG-KDA-AscendC-1.result.json` | 0.598684× | [865a123dcb2a](../../experiments/lineage/20261117/source-evidence/sources/865a123dcb2a7816249f1a66386239ed94e54f7d245edef4248bacb6f26929b8.py) |
| 厂商参考 | `threeway-20261030/references/log10_/formal/CANN-async-python-1.result.json` | 0.092104× | [f1d9d0c441d7](../../experiments/lineage/20261117/source-evidence/sources/f1d9d0c441d7167d39e0f9fed63e6ca368e3e1160de2fa4ae99702394d46b06a.py) |
| 厂商参考 | `threeway-20261030/references/log10_/cpp-v1/formal/CANN-cpp-1.result.json` | 0.701742× | [761265708e9c](../../experiments/lineage/20261117/source-evidence/sources/761265708e9c4bcfdcba327d2f8eb363f0bb81bf615017e1ab2a3aea1ff6b128.py) |
| 候选／不符合当前约束 | `threeway-20261030/final-confirmation/log10_/cb9cf4dcc720/HG-KDA-final-confirm-1.result.json` | 0.942502× | [cb9cf4dcc720](../../experiments/lineage/20261117/source-evidence/sources/cb9cf4dcc7204af3061b7b81644d8390197446eef88db0035162dcdab54d688e.py) |
| 厂商参考 | `cann-priority-20261102/references/log10_/formal/CANN-1.result.json` | 0.769042× | [8679107029b4](../../experiments/lineage/20261117/source-evidence/sources/8679107029b44e127cb8264063c4bf671e4568668e69c7e6c43dd36c0fed1877.py) |
| 候选／不符合当前约束 | `cann-priority-20261102/comparisons/log10_/starting/HG-KDA-start-1.result.json` | 1.305936× | [cb9cf4dcc720](../../experiments/lineage/20261117/source-evidence/sources/cb9cf4dcc7204af3061b7b81644d8390197446eef88db0035162dcdab54d688e.py) |
| 候选 | `strict-triton-20261102/candidates/log10_/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.805593× | [d5ba772425b1](../../experiments/lineage/20261117/source-evidence/sources/d5ba772425b156c2f9377b8c07ceb338c5045024f81169042d76690645c5f049.py) |
| 候选 | `strict-triton-20261102/candidates/log10_/restored-v1/CANN/strict-CANN-1.result.json` | 0.843051× | [d5ba772425b1](../../experiments/lineage/20261117/source-evidence/sources/d5ba772425b156c2f9377b8c07ceb338c5045024f81169042d76690645c5f049.py) |
| 候选 | `independent-regression-20261109-v3/operators/log10_/evaluation/result.json` | 0.940911× | [d5ba772425b1](../../experiments/lineage/20261117/source-evidence/sources/d5ba772425b156c2f9377b8c07ceb338c5045024f81169042d76690645c5f049.py) |
| 候选 | `log10-method-transfer-20261110-development-v2/formal-a4d38b964aa6/operators/log10_/evaluation/result.json` | 0.163022× | [a4d38b964aa6](../../experiments/lineage/20261117/source-evidence/sources/a4d38b964aa64e0964ad85aabcaa3aa25fa82ec89d7a4931f7036b08974dadb8.py) |
| 候选 | `log10-method-transfer-20261110-development-v2/formal-a73e15786d68/operators/log10_/evaluation/result.json` | 0.227774× | [a73e15786d68](../../experiments/lineage/20261117/source-evidence/sources/a73e15786d685eb73322e512340d98c43636222634d0a3f8a4a54222c505812b.py) |
| 候选 | `log10-method-transfer-20261110-development-v2/formal-4eca306ca042/operators/log10_/evaluation/result.json` | 0.212566× | [4eca306ca042](../../experiments/lineage/20261117/source-evidence/sources/4eca306ca0421347bfaf7f8de79da7d76cb37e0e3a9459c4de806319dc8f3e36.py) |
| 候选 | `log10-method-transfer-20261110-development-v2/formal-2f90b7925721/operators/log10_/evaluation/result.json` | 0.332671× | [2f90b7925721](../../experiments/lineage/20261117/source-evidence/sources/2f90b792572123361d223e3035e80b3f92857c399904afb79198207289979a4c.py) |
| 候选 | `log10-method-transfer-20261110-development-v2/formal-0fefe838c58c/operators/log10_/evaluation/result.json` | 0.403237× | [0fefe838c58c](../../experiments/lineage/20261117/source-evidence/sources/0fefe838c58ceafa9c4c7ab856d9c0d9cdbd6bf4bdc2f14c4193358f899a0d72.py) |
| 候选 | `log10-method-transfer-20261110-development-v2/formal-4c184cdc36f4/operators/log10_/evaluation/result.json` | 0.641437× | [4c184cdc36f4](../../experiments/lineage/20261117/source-evidence/sources/4c184cdc36f416b4afb7f371582159ca43424b61ecfcbd3af593341bcd047a63.py) |
| 候选 | `log10-method-transfer-20261110-development-v2/formal-e62c6d636097/operators/log10_/evaluation/result.json` | 0.612249× | [e62c6d636097](../../experiments/lineage/20261117/source-evidence/sources/e62c6d63609707f081adecb46bc144877ffce63b087cf275c11bc136e271110d.py) |
| 候选 | `log10-method-transfer-20261110-development-v2/formal-b2a52734c432/operators/log10_/evaluation/result.json` | 0.711475× | [b2a52734c432](../../experiments/lineage/20261117/source-evidence/sources/b2a52734c43225ac6d9e3753ec2920af5814f5ced6e2891029b640ce9d0ca071.py) |
| 候选 | `log10-method-transfer-20261110-development-v2/formal-d519b459c915/operators/log10_/evaluation/result.json` | 0.644964× | [d519b459c915](../../experiments/lineage/20261117/source-evidence/sources/d519b459c915e43f123e5fd15273adf91209737a4677a7584e773ea6ac67d39f.py) |
| 候选 | `log10-method-transfer-20261110-development-v2/formal-b64e7d8351df/operators/log10_/evaluation/result.json` | 0.753900× | [b64e7d8351df](../../experiments/lineage/20261117/source-evidence/sources/b64e7d8351df6e382830de252becb36d373e912757e42f27659427a04f7c43fb.py) |
| 候选 | `log10-method-transfer-20261110-development-v2/formal-69699663ecc7/operators/log10_/evaluation/result.json` | 0.761426× | [69699663ecc7](../../experiments/lineage/20261117/source-evidence/sources/69699663ecc7df419565d9d157d7f872ffaf2b89f98730f059ecad173a1aa81a.py) |
| 候选 | `log10-method-transfer-20261110-development-v2/formal-8600ce070318/operators/log10_/evaluation/result.json` | 0.851055× | [8600ce070318](../../experiments/lineage/20261117/source-evidence/sources/8600ce07031842c6bed467ea5ebe1f0a66d265e864b171e4176c1f0b6b318324.py) |
| 候选 | `log10-method-transfer-20261110-frozen-v1/formal-40cfe1a83343/operators/log10_/evaluation/result.json` | 0.253361× | [40cfe1a83343](../../experiments/lineage/20261117/source-evidence/sources/40cfe1a8334335ae9e06633be10f9b661e9e1c23806f5753c625cd977b5d1eb0.py) |
| 候选 | `log10-method-transfer-20261110-frozen-v1/formal-0261f02aee72/operators/log10_/evaluation/result.json` | 0.677735× | [0261f02aee72](../../experiments/lineage/20261117/source-evidence/sources/0261f02aee725d0866a9342f5e6d8ce62d5e7ab02f2b75ce3f824b679b898d3b.py) |
| 候选 | `log10-method-transfer-20261110-frozen-v1/formal-b6b10ceb10e3/operators/log10_/evaluation/result.json` | 0.713477× | [b6b10ceb10e3](../../experiments/lineage/20261117/source-evidence/sources/b6b10ceb10e3ddb41fa0c2b3d2aa581152197dff026316e2f8aa52c7cec240c0.py) |
| 候选 | `log10-method-transfer-20261110-frozen-v1/formal-45c7380a8062/operators/log10_/evaluation/result.json` | 0.846792× | [45c7380a8062](../../experiments/lineage/20261117/source-evidence/sources/45c7380a8062a27eba4666c2c71e4970b6d3f52b6d8227ba872a21830266c350.py) |
| 候选 | `overnight-regions-20261110-v1/entry-regression/operators/log10_/evaluation/result.json` | 0.915250× | [45c7380a8062](../../experiments/lineage/20261117/source-evidence/sources/45c7380a8062a27eba4666c2c71e4970b6d3f52b6d8227ba872a21830266c350.py) |
| 候选 | `overnight-regions-20261110-v1/log10-regions/formal-25e8b11f2c2f/operators/log10_/evaluation/result.json` | 0.750783× | [25e8b11f2c2f](../../experiments/lineage/20261117/source-evidence/sources/25e8b11f2c2f8e40e2637b0fdd981c1f18b813d04b9d9f1f7d8cddec63cc05fe.py) |
| 候选 | `overnight-regions-20261110-v1/log10-regions/formal-cbb329b76639/operators/log10_/evaluation/result.json` | 0.833512× | [cbb329b76639](../../experiments/lineage/20261117/source-evidence/sources/cbb329b76639ded0038bf542cfa46a9e240c33dd0df2f1c7c227f394741f05fc.py) |

## 交付终点

- [选用源码](../../../operators/log10_/versions/independent_strict/retained/kernel.py)：`d5ba772425b156c2f9377b8c07ceb338c5045024f81169042d76690645c5f049`。
- [保留版完整结果](../../../operators/log10_/evidence/retained/evaluation/result.json)：**0.940911× CANN，36/36**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

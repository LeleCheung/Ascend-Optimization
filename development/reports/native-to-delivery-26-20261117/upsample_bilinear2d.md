# upsample_bilinear2d

## 原始起点

- [固定版FlagGems源码](../../dependencies/third_party/FlagGems/src/flag_gems/ops/upsample_bilinear2d.py)；SHA `62a02c203a3e54a09f6479ca8526168bf1529821284cd1b42bc6cbdf3c980a12`。
- [实际送测源码](../../experiments/lineage/20261117/source-evidence/sources/42b75754b61f353e649b0416c643132b5bdb43bcc4a9197c655bbd8a814eabed.py)；原始模块AST一致，追加入口：`run = upsample_bilinear2d`。
- [起点原始结果](../../experiments/lineage/20261117/source-evidence/artifacts/0cedbf41a3cef4288f75058156a98db9cce09b880ea2127346bd2aad10b20138.json)：**RUNTIME_ERROR**；0/24。
- [起点请求／合同绑定](../../experiments/lineage/20261117/source-evidence/artifacts/131607a600cb7930c982459f3fada392c7782b9f6d2d298fc6ff7109ce230827.json)；服务器原件：`/data/hanle/ascend-optimization/archive/Ascend-Optimization-pre-release-20261103/project/Ascend-Optimization-v1/operators/campaigns/night-20261021-v3/results/upsample_bilinear2d/FlagGems/FlagGems-1.result.json`。

## 关键阶段

| 阶段 | 已有证据 |
| --- | --- |
| 原生送测 | 原始模块未改写；追加评测入口，结果见上 |
| 早期优化 | 8条账本测量；其中baseline可能已经修改过源码，不能替代原生起点 |
| 后续修改与回归 | 19份候选测量、9份请求源码；详细位置列在下表 |
| 交付终点 | 1.142330× CANN；源码SHA与保留结果均已核对，并定位远端同源码同分数记录 |

## 全部测量入口

按存储位置列出全部已找到测量，不将文件遍历顺序称为源码继承顺序。源码可直接阅读；远端结果路径、结果SHA、请求SHA及合同绑定完整保存在inventory.json。原生基线、候选、厂商参考分别标明。未重新审查的历史候选不能一概标为严格Triton。

| 类型 | 实验位置 | 结果 | 源码 |
| --- | --- | --- | --- |
| 基线目录 | `night-20261021-v3/results/upsample_bilinear2d/FlagGems/FlagGems-1.result.json` | RUNTIME_ERROR | [42b75754b61f](../../experiments/lineage/20261117/source-evidence/sources/42b75754b61f353e649b0416c643132b5bdb43bcc4a9197c655bbd8a814eabed.py) |
| 基线目录 | `night-20261021-v3-adapter-repairs/results/upsample_bilinear2d/FlagGems/FlagGems-1.result.json` | PARTIAL_PASS | [42b75754b61f](../../experiments/lineage/20261117/source-evidence/sources/42b75754b61f353e649b0416c643132b5bdb43bcc4a9197c655bbd8a814eabed.py) |
| 候选 | `remaining-five-20261023/results/upsample_bilinear2d/parent/formal/parent-1.result.json` | RUNTIME_ERROR | [42b75754b61f](../../experiments/lineage/20261117/source-evidence/sources/42b75754b61f353e649b0416c643132b5bdb43bcc4a9197c655bbd8a814eabed.py) |
| 候选 | `remaining-five-20261023-v2/results/upsample_bilinear2d/parent/formal/parent-1.result.json` | PARTIAL_PASS | [42b75754b61f](../../experiments/lineage/20261117/source-evidence/sources/42b75754b61f353e649b0416c643132b5bdb43bcc4a9197c655bbd8a814eabed.py) |
| 候选 | `remaining-five-20261023-v2/results/upsample_bilinear2d/attempt-1/candidates/2054ca728198ae2a395addf604834a00f67f4e5352acc84adb93250457bee84b/formal/round-0003-1.result.json` | 0.339502× | [2054ca728198](../../experiments/lineage/20261117/source-evidence/sources/2054ca728198ae2a395addf604834a00f67f4e5352acc84adb93250457bee84b.py) |
| 候选 | `remaining-five-20261023-v2/results/upsample_bilinear2d/attempt-1/candidates/600f97a49ab7e7b69b46fe31e78ee17007f19fba2cea3f6dc651867fbc2c8fe8/formal/round-0004-1.result.json` | 0.341160× | [600f97a49ab7](../../experiments/lineage/20261117/source-evidence/sources/600f97a49ab7e7b69b46fe31e78ee17007f19fba2cea3f6dc651867fbc2c8fe8.py) |
| 候选 | `remaining-five-20261023-v2/results/upsample_bilinear2d/attempt-1/candidates/998667cf5920d077530e0d3d51a9772a5331958aeb1b7d58d988ed561c237546/formal/round-0002-1.result.json` | 0.312401× | [998667cf5920](../../experiments/lineage/20261117/source-evidence/sources/998667cf5920d077530e0d3d51a9772a5331958aeb1b7d58d988ed561c237546.py) |
| 候选 | `remaining-five-20261023-v2/results/upsample_bilinear2d/attempt-1/candidates/af7284633c25568ea2fd235a70d60294ca6f9d4ead7922243d6ddd75df15767b/formal/round-0001-1.result.json` | 0.001714× | [af7284633c25](../../experiments/lineage/20261117/source-evidence/sources/af7284633c25568ea2fd235a70d60294ca6f9d4ead7922243d6ddd75df15767b.py) |
| 候选 | `two-closure-20261024/results/upsample_bilinear2d/codex-initial/7d91b983f295e34486d4881ae6b5afa94c07d112ecaf10cd37306cf299d14a10/formal/codex-libentry-1.result.json` | 0.327280× | [7d91b983f295](../../experiments/lineage/20261117/source-evidence/sources/7d91b983f295e34486d4881ae6b5afa94c07d112ecaf10cd37306cf299d14a10.py) |
| 候选 | `two-closure-20261024/results/upsample_bilinear2d/parent/formal/parent-1.result.json` | 0.344894× | [2054ca728198](../../experiments/lineage/20261117/source-evidence/sources/2054ca728198ae2a395addf604834a00f67f4e5352acc84adb93250457bee84b.py) |
| 候选 | `two-closure-20261024/results/upsample_bilinear2d/codex-late/7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a/formal/safe_compiled_launch-1.result.json` | 0.559042× | [7ebe07fbaeb1](../../experiments/lineage/20261117/source-evidence/sources/7ebe07fbaeb1ccd39e198b0327b83b9a95b5684a42ed095d072b47fff5805e2a.py) |
| 候选 | `two-closure-20261024/results/upsample_bilinear2d/attempt-2/candidates/3008f880a9b9ab1da020736990abf52e47b0a281bb3f42115f8e87bf08d82c83/formal/round-0002-1.result.json` | 0.962640× | [3008f880a9b9](../../experiments/lineage/20261117/source-evidence/sources/3008f880a9b9ab1da020736990abf52e47b0a281bb3f42115f8e87bf08d82c83.py) |
| 候选 | `two-closure-20261024/results/upsample_bilinear2d/codex-disjoint/e46260c2c74a1b1f852f938eb39e2a9b85693c24e8fb393f2ecf8e8a3a2e70c5/formal/disjoint_blocks-1.result.json` | 0.915027× | [e46260c2c74a](../../experiments/lineage/20261117/source-evidence/sources/e46260c2c74a1b1f852f938eb39e2a9b85693c24e8fb393f2ecf8e8a3a2e70c5.py) |
| 候选 | `threeway-20261030/results/upsample_bilinear2d/current-recheck/formal/HG-KDA-current-1.result.json` | 0.954431× | [e46260c2c74a](../../experiments/lineage/20261117/source-evidence/sources/e46260c2c74a1b1f852f938eb39e2a9b85693c24e8fb393f2ecf8e8a3a2e70c5.py) |
| 厂商参考 | `cann-priority-20261102/references/upsample_bilinear2d/formal/CANN-1.result.json` | RUNTIME_ERROR | [fe0b51a41c2b](../../experiments/lineage/20261117/source-evidence/sources/fe0b51a41c2bc3667c89fc76f87c7226cc0d91691a672adfb9a4d2ce4d42e268.py) |
| 厂商参考 | `cann-priority-20261102/references/upsample_bilinear2d/nchw-v2/formal/CANN-1.result.json` | 0.874415× | [db43ca73852a](../../experiments/lineage/20261117/source-evidence/sources/db43ca73852a63b339b4cdc6702e9c495a61f37e79d28883fe83a573bde77973.py) |
| 候选 | `cann-priority-20261102/comparisons/upsample_bilinear2d/starting/HG-KDA-start-1.result.json` | 1.146535× | [e46260c2c74a](../../experiments/lineage/20261117/source-evidence/sources/e46260c2c74a1b1f852f938eb39e2a9b85693c24e8fb393f2ecf8e8a3a2e70c5.py) |
| 候选 | `strict-triton-20261102/candidates/upsample_bilinear2d/restored-v1/pytorch/strict-PyTorch-1.result.json` | 0.944788× | [e46260c2c74a](../../experiments/lineage/20261117/source-evidence/sources/e46260c2c74a1b1f852f938eb39e2a9b85693c24e8fb393f2ecf8e8a3a2e70c5.py) |
| 候选 | `independent-regression-20261109-v3/operators/upsample_bilinear2d/evaluation/result.json` | 1.142330× | [e46260c2c74a](../../experiments/lineage/20261117/source-evidence/sources/e46260c2c74a1b1f852f938eb39e2a9b85693c24e8fb393f2ecf8e8a3a2e70c5.py) |
| 候选 | `night-research-20261116/release/operators/upsample_bilinear2d/evidence/retained/evaluation/result.json` | 1.142330× | 请求源码未定位 |
| 候选 | `night-research-20261116/release-v2/operators/upsample_bilinear2d/evidence/retained/evaluation/result.json` | 1.142330× | 请求源码未定位 |
| 候选 | `night-research-20261116/release-v3/operators/upsample_bilinear2d/evidence/retained/evaluation/result.json` | 1.142330× | 请求源码未定位 |
| 候选 | `night-research-20261116/release-v4/operators/upsample_bilinear2d/evidence/retained/evaluation/result.json` | 1.142330× | 请求源码未定位 |

## 交付终点

- [选用源码](../../../operators/upsample_bilinear2d/versions/independent_strict/retained/kernel.py)：`e46260c2c74a1b1f852f938eb39e2a9b85693c24e8fb393f2ecf8e8a3a2e70c5`。
- [保留版完整结果](../../../operators/upsample_bilinear2d/evidence/retained/evaluation/result.json)：**1.142330× CANN，24/24**。
- 已在远端历史请求中找到同SHA源码测量；是否可以接到前一轮，仍按源码继承和参考绑定逐段判定。

## 待补齐的连接

- 原生起点未完整通过；保留失败节点，不填0分
- 原生与交付的参考／合同包不同；可分段展示，统一数值曲线需核对或复测
- 源码与测量已定位；跨批次逐次父子关系及方法来源仍需逐段核对

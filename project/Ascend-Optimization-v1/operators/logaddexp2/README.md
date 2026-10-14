# logaddexp2

验收看板G122为0.3218×。固定FlagGems master `d6a8eec473517a3d68157b208eb9c057eb1d4c50`复测0.231×，36/36完整通过，符合低于0.8×的正式优化对象要求。

## 已验证成果

| 流程 | 相对PyTorch的两次完整复验 | 采用成绩 | 完整测试 |
| --- | ---: | ---: | --- |
| 历史无profiler流程 | 1.070×、1.061× | 1.061× | 36/36 |
| 历史有profiler流程 | 1.267×、1.267× | 1.267× | 36/36 |
| 历史方法workflow复现 | 1.269×、1.261× | 1.261× | 36/36 |

计时范围为设备kernel，性能case取几何平均，采用两次较慢值。两组名称表示KG profiling开关，不代表整个工程未经Codex参与。此前环境、接口、诊断和优化方法整理的贡献需保留；方法来源以具体历史记录为准。

有profiler版本源码见`reports/kg-profile-20261012-agent/export/main.py`，SHA256为`b644045236cf2c45231b39439052eda41302067036d40506de94072c19487cc6`。完整独立评测见同目录`independent/`，原生诊断与附件见`native-evidence/`。无profiler版本上游审核仍为`WAITING_CODE_REVIEW`，独立36项复验已通过；不改写上游状态。

## 当前工作：固化历史方法

2026-10-14目标明确为：将已有诊断、优化方法、候选和评测固化成无需Codex逐轮手工操作的workflow，复现约1.267×。无需重新发明候选，也不要求额外超过历史成绩。历史方案包含稳定公式、输入精度max与FP32修正项、连续快路径、整除去mask、多tile处理、按规模选择tile/循环/num_stages。

方法与证据入口：[历史诊断和优化方法](../../tools/profiling/历史复现知识-20261014.md)。运行入口：

```bash
bash /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/run-logaddexp2-replay-20261014.sh
```

最终批次`profiling-agent-20261014f-replay`已由tmux自动完整执行：完整评测历史起点、重新采集两个慢case、保存诊断和提示词、保持历史源码、两次最终复验。两次1.269×/1.261×，完整36/36，8份原始真机附件已核验；最终SHA与历史版本一致，记录为历史方案复现。该批未启动模型生成，无需人工逐轮推进。

新复现用新批次参数运行上面的脚本，例如末尾加`next-replay`。继续优化时，使用`optimization_campaign.py`、相同历史知识文件、新输出目录并省略`--replay`。最终报告见[精简汇报](reports/master-closure-20261012-agent/精简汇报.md)。

## 失败和探索记录

原`20261012-agent`独立控制器六轮返回结构化占位符，最终master回退0.231×，不计优化成果。随后修复普通JSON生成、JIT全局常量、KGS入口和超大grid，原始失败记录保留。

`20261014d-agent`从Codex修复后的有界grid候选0.829×出发，第1轮36/36、0.844×；第2轮思考输出反复耗尽额度。用户明确历史方法复现目标后，已停止该探索支线，并保留`interruption.json`，没有两次最终复验，不计最终成果。详情见[续跑记录](续跑说明-20261014.md)。

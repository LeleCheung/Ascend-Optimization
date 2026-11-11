# SELU 独立skill性能恢复原件

CC／DeepSeek从上轮0.210037×生成候选继续优化，在Codex提供的历史方法、芯片／宿主诊断和实现指导下，得到 **0.887010× CANN** 的严格Triton新候选。正式36/36、双方各42/42补测、偏移9/9及stream／指针／尾块24/24通过。已有1.039292×交付继续保留，原30项合同仍达标26/30。

| 原件 | 内容 |
| --- | --- |
| [summary.json](summary.json) | 结果、实际贡献、调用轨迹、已开始／已完成／中断预检数量 |
| [最终源码](formal-5e9e4d482620/operators/selu_/candidate.py) | 本轮实际生成并验证的版本 |
| [正式结果](formal-5e9e4d482620/operators/selu_/evaluation/result.json) | 完整36项与12项逐case性能 |
| [正式审计](formal-5e9e4d482620/audit-final.json) | 源码、合同、两侧补测与性能核验 |
| [源码审查](source-review-5e9e4d482620.json) | 连续快路径、非连续回退和启动缓存审查 |
| [专项检查](attempts/5e9e4d4826207ea03e70d01e871c71220e40f9097ad0a0f87da977f793573115/) | 偏移、当前stream、输入指针、尾块和实际编译产物 |
| [起点身份](workspace/starter-provenance.json) | 与上轮冻结候选的输入绑定；历史方法及人工贡献说明 |
| [模型收尾反馈](workspace/final-feedback.json) | 根据正式结果形成的方法与剩余差距说明 |
| [设备诊断](baseline-probe/)、[修订诊断](baseline-probe-v2/) | 小／大输入任务轨迹、启动器接口与实际IR；修订版移除计时区间的记录钩子 |
| [宿主诊断](host-probe-0fed8d43398d.json) | 缓存命中路径的函数取样，独立于正式成绩 |
| [后端来源](runtime-abi-provenance.json) | 本机后端的当前stream接口与任务队列设置 |
| [调用记录](agent-calls/)、[控制器快照](execution-snapshots/)、[版本绑定](execution-versions/) | 提示词、响应、用量及实际执行版本 |
| [Windows复核](windows-audit.json) | 1175份原始文件与4份不同源码的完整结果核验 |

前三份正式候选分别为0.312005×、0.433299×、0.710539×；原始记录未覆盖。7份源码进入预检，1份在ABI修正时中断，6份形成完整预检结果；其中4份进入正式评测。每份不同SHA只提交一次正式请求。模型调用触及轮次限制、宿主终止和工具修复均保留原件。

原始导出文件由 [export-manifest.json](export-manifest.json)绑定路径、字节数与SHA；[传输核验](windows-transfer.json)记录压缩包身份。README及Windows复核为传输后补充资料。初次诊断的插桩提交时间、模型原始推断及未采纳的ABI尝试不作为已验证性能规律。

在仓库根目录执行，不触发设备测试，输出使用新文件名：

```bash
python tools/skill_validation/verify.py --directory benchmarks/skill-validation/20261109-selu-recovery --output ../hgkda-output/selu-recovery-audit.json
```

方法与复现边界见[性能恢复报告](../../../docs/selu-skill-recovery-20261109.md)。

# GQA 有／无模拟器对照冻结材料

结果入口：[对照图](../../reports/gqa-simulator-comparison-20261117/index.html)及[报告](../../reports/attribution-and-gqa-20261117.md)。

- `contract.json`：固定case 1、2、4、7，输入seed、计时预算、官方golden和checker身份。
- `official-check-01.json`：同一旧候选按官方checker重新检查的起点结果；CANN参考也通过四项。
- `contrast-manifest-v2.json`：两组共同方法、模型设置哈希、起点及最多两次修改的预算。
- `no-simulator-v2/`、`with-simulator-v2/`：模型输入输出、源码快照、修改前决策、只读审查、两次完整四项评测。
- `capture/`、`capture-after/`：前后真机profiling、实际IR与有限窗口模拟器数据。应用超时边界见各自collector日志。
- `comparison-results.json`：两组效果、逐case耗时、模型成本与重新采集结果。
- `audit-*.json`：事件和引用身份核验。
- `method.tar.gz`、`checker-source.tar.gz`：实验实际使用的方法与官方源码快照，不需要依赖当前可变开发代码。
- `evidence-export-manifest.json`：336份原件的SHA。此README为导出后新增说明，不在原始导出清单中。

官方仓库：<https://gitcode.com/cann/cann-bench>，提交`12180ce7e717e8eed7cf81083666b40f8b5dd765`，许可见`official/LICENSE`。此目录是研究证据，不是HG-KDA正常运行依赖，也不替换`operators/`交付版本。

复现需910B与对应容器环境。解包固定方法和checker到隔离运行目录，按contract中的本地路径映射配置后，通过`attention_bench.py`读取原字节候选执行四项检查。路径变更产生新合同身份，记录映射，不覆盖原件。新任务可复用评测器，不能把四项子集结果说成官方20项全量验收或排行榜成绩。

初次长报告调用保留在`no-simulator/`：耗尽工具轮次、没有候选修改，是准备成本。v2两组从相同未改源码重新开始。源码注释中的数值推导仍须参考source-reviews的纠正，不能直接作为论文结论。

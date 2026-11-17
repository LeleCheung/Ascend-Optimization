# 原生起点与交付路线检索原件

本目录保存2026-11-17从910B历史目录只读导出的精选证据。没有运行候选或重新测量，没有改写operators交付材料。

- `source-evidence/inventory.json`：497份测量的原件路径、SHA、请求绑定、源码身份、43份基线目录记录和209条批次计划记录。
- `source-evidence/sources/`：实际请求中携带的源码，按SHA保存。包括基线、候选及厂商参考；不是统一合规的交付源码集合。
- `source-evidence/artifacts/`：原始基线结果／请求、来源记录、计划、合同文件及版本锁，按原始字节SHA保存。
- 结构化汇总与逐项入口见[26项证据清单](../../../reports/native-to-delivery-26-20261117/README.md)。

导出包`evidence-v2.tar.gz`的SHA256为`678e6cbeb3edbf0d4d4902f5dae5e75a31d412ac724c443dab8c1b07c723485e`，Windows与910B一致。完整导出包保存在仓库外`.control-state/lineage-evidence-20261117/`，远端位于`/data/hanle/ascend-optimization/runs/lineage-evidence-20261117/`。

26份原始FlagGems模块SHA与批次计划一致；送测源码的原始模块AST保持一致，末尾追加run入口。该核验不代替所有导入依赖／实际设备派发路径复查。14项选定起点未完整通过时，保留失败原件；失败原因包括真实精度／编译问题，也包括早期导入或参考适配问题，不能统一当作算子错误。

检索器明确保留名称含cachemask的算子目录，仅排除真实缓存目录。各记录数量是本次检索范围内的结果条目，含重复测量和厂商参考；不等于优化轮次或完整父子版本链。

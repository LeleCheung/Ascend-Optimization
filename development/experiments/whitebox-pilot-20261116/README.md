# 白盒证据试点冻结材料

本目录仅保存2026-11-16已结束实验的精选证据，不是skill的运行依赖，不替换 `operators/` 选用源码，也不修改26算子交付包。

| 目录 | 内容 |
| --- | --- |
| `gqa/` | CANNBench官方GQA子集合同、三版源码及审查、正确性／计时、输出格式下界、修改前决策与模型使用量 |
| `contrast/common/` | 上采样最初采集与失败基线、模拟器指令／区域及退出资格检查 |
| `contrast/baseline-compatibility/` | 共享启动修复后的完整基线、源码审查、成对采集和trace核验 |
| `contrast/no-simulator/` | 成对profiling＋IR侧的生成、完整KGS／补测、重新采集与trace核验 |
| `contrast/with-simulator/` | 增加模拟器侧的生成、编译失败、修改前决策与trace核验 |
| `contrast/simulator-followup/` | 单列资源／接口修复尝试、设备执行失败及失败后基本健康检查 |

报告入口：[26项图册](../../reports/operator-traces-26-20261116/index.html)、[上采样对照](../../reports/upsample-simulator-contrast-20261116/README.md)、[GQA起点](../../reports/gqa-pilot-20261116.md)。

真实runtime引用在服务器原始目录核验。六个运行目录共144个事件、11份修改前决策均闭合；引用字节可按原SHA复查。早期指向可变文件的引用，通过已有相同SHA快照、原日志字节前缀或完全匹配的初始元数据解析，未改写原始事件。当前runtime已补齐证据快照、日志前缀长度和可变调用元数据快照。

精选材料不包含全部原始模型流和原始模拟器大文件。完整导出在仓库外 `.control-state/night-research-20261116/`，服务器原件位于 `/data/hanle/ascend-optimization/runs/night-research-20261116/`。`contrast/curated-manifest.json`核验本目录精选对照文件；名称含`export-manifest`的文件记录更大范围原始导出，不表示其全部文件都被复制进本目录。凭据、缓存及环境不作为证据发布。

收尾补充的 [GQA参考来源](gqa/reference-provenance.json)通过首次实验前的输入归档核验四份参考文件身份；该事后核验记录独立保存，未替换原始合同、结果或trace。

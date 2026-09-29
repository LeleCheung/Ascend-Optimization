# 已归档的 FlagGems v5 算子目录

本目录是历史 `v5.1` 数据，当前 Server 不再支持，仅作为迁移输入和历史证据保留。
Definition 只保存算子 reference 源码和入口；workload 文件分别保存 correctness
与 timing 使用的具体输入。

不得把本目录交给当前 Server 执行。需要重新生成历史快照时，输出仍应放在
`data/.old/flaggems-v5`；需要执行时必须先迁移到受支持的 schema。

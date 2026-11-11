# HG-KDA开发与研究

本目录用于开发验证、批量实验与历史证据。hgkda和operators的外部运行不能依赖本目录。

| 位置 | 内容 |
| --- | --- |
| tests/hgkda/ | HG-KDA软件自动测试；正常skill调用不执行这些测试 |
| tools/skill_validation/ | 方法冻结、生成轨迹审计、实验导出与离线核验 |
| tools/regression/ | 批量回归与结果审计 |
| tools/operators/ | 内部参考构建和研究辅助；交付检查器在operators下 |
| tools/maintenance/ | 目录、同步及迁移验证 |
| tools/lineage/ | 历史版本来源整理 |
| experiments/ | 原benchmarks的完整冻结档案，原件字节保持不变 |
| reports/ | 阶段报告与历史开发记录 |
| dependencies/ | 上游固定快照，仅供开发复现；公共运行不从这里导入依赖 |
| manifests/ | 来源、版本与迁移映射 |

从仓库根目录执行：

    python -m pip install ./hgkda
    python -m unittest discover -s development/tests/hgkda -v
    python -m hgkda.runtime --help
    python -m development.tools.skill_validation --help

核心软件测试、算子完整KGS评测、专项语义补测分别报告。迁移时优先以相同SHA离线核验冻结档案；公开运行的隔离验证只复制hgkda与operators，使用新目录，不重复正式性能评测挑分。

当前26/30交付及三类源码以[operators/index.json](../operators/index.json)为准。SELU与log10的整理后入口复验和区域方法实验见[方法报告](reports/regions-method-20261110.md)。阶段实验不宣称全部候选由skill从零重新生成。

2026-11-11 已完成三部分目录迁移与不含 development 的公开运行验证：log10_ 完整 36/36、双参考各 66/66、特殊值 15/15，并完成成对采集与上下文回填。详见[迁移报告](reports/three-part-migration-20261111.md)。

历史报告和原始JSON仍可能包含旧路径。按[迁移映射](manifests/three-part-migration-20261111.json)定位，原始请求和源码不做全文替换。新的集成调用使用公共runtime入口；旧控制器不恢复为默认操作入口。

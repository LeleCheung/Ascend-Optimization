# HG-KDA开发与研究

2026-11-16 夜间研究已结束：[结果与证据](reports/night-research-20261116.md)。26项[主要优化线图册](reports/operator-traces-26-20261116/index.html)包含SVG／PNG／历史入口：21项多点、1项单账本节点、4项标出缺口。[上采样对照](reports/upsample-simulator-contrast-20261116/README.md)中成对profiling＋IR路线269→220 μs、完整KGS及补测通过，模拟器路线未形成有效候选；[GQA起点](reports/gqa-pilot-20261116.md)两个BF16子集通过，FP16格式下界问题单列。原26算子交付不变。

2026-11-16 轨迹展示与记录更新：[log10精简图](reports/log10-trace-pilot-20261116/index.html)只展示一张主线；箭头颜色表示判断依据，节点形状表示修改位置。没有模拟器正向案例，不再单独呈现放大图。公共runtime增加自动事件记录和修改前decision登记；[记录规范](reports/trace-rollout-20261116.md)。图册生成不依赖新增设备测量。

2026-11-16 log10连续trace试点已完成：[主线图、模拟器放大图及历史入口](reports/log10-trace-pilot-20261116/index.html)（[说明与重建命令](reports/log10-trace-pilot-20261116/README.md)）。开发线24次模型调用、126次源码修改、17份预检、12次正式评测；区域实验另存3点。新增轻量trace索引及JSONL事件流，原28点历史图保留。源码与结果等278份关联文件已核验，本轮没有生成候选或重测设备性能。

2026-11-16 轨迹下一步方案：[主线选择、trace索引与展示计划](reports/trace-presentation-plan-20261116.md)。先整理一条可追溯主线并减少图面箭头，再复用到六项和26项；全量历史图继续保留。

2026-11-16 轨迹展示：按“先画真实过程，再核验归因”交付[六算子多批次图](reports/operator-traces-history-20261116/index.html)（[轮次数量与来源](reports/operator-traces-history-20261116/README.md)）。包含已收集的20批、69条正式评测记录；逐节点展示性能、修改方法与分析信息，保留失败和恢复。原[短片段版](reports/operator-traces-20261116/index.html)仅用于追溯。

当前主线（2026-11-16）：**白盒模拟器驱动的多层优化与Agent收益验证**。从共同的白盒证据入口分为两条研究路线：编译优化线（Triton源码→单算子IR→通用编译pass）和能力扩展线（Triton kernel→custom op→Ascend C）；后者是独立研究分支，不计入严格Triton交付。本轮在复盘26项交付trace的同时，按用户夜间授权启动上采样方法实验和一个Level 4起点，不再固定追踪log10。历史证据覆盖与归因缺口见[复盘报告](reports/trace-attribution-20261116.md)；研究设计见[研究计划](reports/next-stage-simulator-attention-20261112.md)。FlagGems维持成果维护与上游交付支线，知识库仅做必要维护。

2026-11-11 新增 FlagGems 外部原生验收接口，log10_ 已完成三层验证。见[接入报告](reports/flaggems-upstream-log10-20261111.md)；原生目标文件测试与全仓库调度／CI分别记录，交付总数保持26/30。

本目录用于开发验证、批量实验与历史证据。hgkda和operators的外部运行不能依赖本目录。

| 位置 | 内容 |
| --- | --- |
| tests/hgkda/ | HG-KDA软件自动测试；正常skill调用不执行这些测试 |
| tools/skill_validation/ | 方法冻结、生成轨迹审计、实验导出与离线核验 |
| tools/regression/ | 批量回归与结果审计 |
| tools/operators/ | 内部参考构建和研究辅助；交付检查器在operators下 |
| tools/maintenance/ | 目录、同步及迁移验证 |
| tools/lineage/ | 历史版本来源整理 |
| tools/traces/ | 连续优化日志导入、源码关系与正式评测绑定、通用视图渲染 |
| experiments/ | 原benchmarks的完整冻结档案，原件字节保持不变 |
| reports/ | 阶段报告与历史开发记录 |
| runbooks/ | Windows 控制端、910B 设备与 KGS 评测操作手册 |
| dependencies/ | 上游固定快照，仅供开发复现；公共运行不从这里导入依赖 |
| manifests/ | 来源、版本与迁移映射 |

从仓库根目录执行：

    python -m pip install ./hgkda
    python -m unittest discover -s development/tests/hgkda -v
    python -m hgkda.runtime --help
    python -m development.tools.skill_validation --help

核心软件测试、算子完整KGS评测、专项语义补测分别报告。迁移时优先以相同SHA离线核验冻结档案；公开运行的隔离验证只复制hgkda与operators，使用新目录，不重复正式性能评测挑分。

当前26/30交付及三类源码以[operators/index.json](../operators/index.json)为准。SELU与log10的整理后入口复验和区域方法实验见[方法报告](reports/regions-method-20261110.md)。阶段实验不宣称全部候选由skill从零重新生成。

当前优先复盘26项既有交付及冻结实验中的模拟器贡献，按触发证据、决定性证据、机制证据和结果证据筛选“模拟器消歧”案例，见[轨迹证据筛选](reports/trace-attribution-20261116.md)。

2026-11-11 已完成三部分目录迁移与不含 development 的公开运行验证：log10_ 完整 36/36、双参考各 66/66、特殊值 15/15，并完成成对采集与上下文回填。详见[迁移报告](reports/three-part-migration-20261111.md)。

历史报告和原始JSON仍可能包含旧路径。按[迁移映射](manifests/three-part-migration-20261111.json)定位，原始请求和源码不做全文替换。新的集成调用使用公共runtime入口；旧控制器不恢复为默认操作入口。

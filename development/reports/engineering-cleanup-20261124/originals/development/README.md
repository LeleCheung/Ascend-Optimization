# HG-KDA开发与研究

2026-11-18 夜间多算子推进：[结果与验证记录](reports/night-multi-operator-20261118.md)。QKV、BatchNorm、scatter_add均取得完整验证的性能改善，主图已接入26项总览。BatchNorm目标81.847→44.913μs，scatter_add目标1.400813→1.282337ms；QKV最新62.837μs，小幅改进经同源码完整复测方向一致。三项均完成修改后热点和相对流水分析，模型读取证据与修改顺序保留；实际修改位于Triton层。正式计时采用本批KGS profiler，历史walltime交付记录保持冻结。

2026-11-18 性能续接完成：[upsample_bilinear2d第4轮](reports/upsample-performance-20261118.md)。目标309.716→165.744μs，相对CANN 0.790→1.480×；完整24/24及双方各30/30通过。模型根据模拟器交织热点与实际IR改为打包写回，并修正低半字掩码。复采显示Vector时间8.44→3.20μs，搬运指令数一致；本轮未新增对照实验。更新[单张主图](reports/native-upsample-contrast-20261118/index.html)，537份证据哈希验证完成，专用服务已停止。下一候选可从新最佳源码继续。

2026-11-18 后续策略改为性能优先：不设总轮数、工具次数、耗时、token或cost预算，达到验收线不自动停止。公共generate默认不传宿主轮数上限、不设整次生成deadline；源码审查、完整评测和逐轮证据继续执行。Windows源码与910B公共安装已更新；910B新任务的`--package`使用`/data/hanle/ascend-optimization/runtime/hgkda-performance-first-20261118/hgkda`，解释器仍为`runtime/hgkda-release-venv/bin/python`。78项软件测试通过；原8轮对照与其method快照冻结保留，后续不要复用旧实验控制器的8／48预算设置。

2026-11-18 已完成：[原生FlagGems上采样模拟器对照](reports/native-upsample-contrast-20261118.md)及[完整轨迹](reports/native-upsample-contrast-20261118/index.html)。同起点、同模型、同合同各8轮，最终均24/24及双方各30/30通过；目标无／有模拟器418.964／309.716μs，六项候选耗时几何比较为1.082倍。保留失败、回退、实际成本与局部模拟器限制，不作多任务因果外推；原26项交付不变。

2026-11-17阶段安排（后续由上文更新）：新增实验围绕《白盒模拟器驱动的多层次优化：以昇腾算子优化为例》汇报，GQA为后续复杂研究线，逐轮记录模拟器证据、选层理由、修改和验证。26项历史确有模拟器参与或核验记录；主图主来源为0不等于未使用。源码／IR阅读不单列为模拟器贡献。见[执行计划与历史核对](reports/presentation-gqa-plan-20261117.md)。

2026-11-17 当前结果：[26项轨迹](reports/operator-traces-stitched-20261117/index.html)接续完成，方法来源补查区分日志确认与带星号的上下文推断。[GQA有／无模拟器对照](reports/gqa-simulator-comparison-20261117/index.html)完成：接入固定官方checker后起点4/4通过；同一起点与两次修改预算下，有模拟器组目标1900.7→1378.7μs、4/4通过，无模拟器组保留起点。前后实际IR及指令计数符合减少一次PV乘法的预期。模拟器为超时留下的有限窗口，且包含Codex代码审查反馈；不宣称独立因果已被统计证明。详情见[报告](reports/attribution-and-gqa-20261117.md)，后文保留各日期的历史状态，原26项交付不变。

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

指标与规格 Excel 维护入口：[data/README.md](data/README.md)。

# 项目汇报约定

## 用户硬约束：候选严格Triton

**baseline使用CANN原生库或来源明确的Ascend C参考；我们优化和交付的候选严格使用Triton。** 候选所有设备计算、复制、归约、扫描、布局转换及快／慢／回退路径均须由Triton kernel完成；禁止以Ascend C、CANN／aclnn、原生DMA、PyTorch目标算子或CPU计算替代。Python／C++仅可用于分配、元数据、编译缓存、参数打包、状态管理和Triton启动等宿主胶水。

完整边界与验收遵循[硬性规则](docs/Triton候选与CANN参考硬性规则.md)。历史混合后端结果不能计为严格Triton成果，原25/30和7/10＋18/20等统计须重新审查；2026-11-02严格Triton复核结果为16/30（1015为6/10、1029为10/20），详见`project/Ascend-Optimization-v1/operators/campaigns/strict-triton-20261102/复核结果简报.md`。不得直接续跑会选中非Triton候选的旧流程。这条用户规则优先于项目历史计划、知识库、skill及提示词中的冲突内容。

当前实验遵循[新25算子计划](project/Ascend-Optimization-v1/operators/下一阶段计划-20261029.md)：保留导师目标名称和顺序；CANN原生库与已核验Ascend C源码参考区分；PyTorch正确性和历史计时口径保留。新工具只改独立覆盖层，不修改共享服务/镜像。新知识有源码、请求、完整结果及补充检查身份才进入已验证索引；热点邻域、可观测空闲和因果瓶颈分开表述。

上述16/30为纠偏复核的冻结起点。夜间持续优化的当前交付数读取`project/Ascend-Optimization-v1/operators/campaigns/strict-night-20261102/progress.json`及`delivery.json`，不能用冻结起点覆盖新增成果。恢复运行前检查既有Goal、tmux和工作区状态，不重复创建Goal或启动活跃工作区。

2026-11-03的新goal位于`operators/campaigns/strict-five-20261103/`，从提交`231ba49d6`的严格25/30继续。tmux为`hanle-strict-five-20261103`，搜索状态是`search-state.json`；`state.json`只表示先行采集队列，不能因其`tasks_finished`判定整个goal结束。恢复须核对本批全部`*-process.json`的PID与启动时间和KG工作区终态，当前包含`optimizer`、`half-evaluator`、`half-comparisons`、`reference-stage`、`half-domain`和`reviewed-best`；所有设备阶段在同一卡7锁下串行。新增的FP16完整域检查包含平铺、大流式、尾部和非对齐路径，不以CPU公式或小形状通过代替实际设备语义。已有启动脚本不得再次执行；排队成功、离线编译、局部参考探测或仅PyTorch成绩均不增加CANN交付数。工具和新批入口见`tools/profiling/hgkda/strict_five/README.md`。

2026-11-03剩余五项批次现已结束，当前结论读取`operators/campaigns/strict-five-20261103/delivery.json`和`交付报告.md`：严格验收仍25/30，`expm1_`原PyTorch口径由0.561×改善至0.647×并完成全部门禁；其余数值与原生参考缺口单列。7个阶段及2个KG工作区均终态，不再执行上述旧启动入口。离线复现使用`tools/profiling/hgkda/strict_five/finish.py`，同步状态读取本批`三端同步核验.json`与实际Git提交，不能将实验终态直接表述为30/30。

当前继续执行同一个active goal，新批次是`operators/campaigns/strict-next-20261103/`，冻结提交`9798acae1`的25/30及`expm1_`最佳0.646557×。三类型新最佳成对采集和两份FMA提案均已终态；同类型FMA为0.618492×，全部数值检查通过但不替换最佳。`optimizer-process.json`对应连续两轮KG／CC搜索，`reference-modes-process.json`等待搜索结束后探测独立进程的官方旧图Expm1精度模式。恢复须核对实际PID启动身份、`search-state.json`及KG工作区，不重新运行`prepare.py`、`launch_optimizer.py`或已有阶段入口。tmux为`hanle-strict-next-20261103`，所有设备工作仍持物理卡7项目锁串行；`progress.json`仅描述已完成采集，不代表搜索或总goal结束。搜索候选还需人工审查、整块及完整FP16域检查、实际后端核验，不能仅据搜索分数更新25/30。恢复说明见`tools/profiling/hgkda/strict_next/README.md`。

本批新增`buffer-evaluator`、`bfloat-domain`和`followup-optimizer`身份：独立缓冲稿完成6项编译，设备评测排在原优化器之后；BF16全位模式检查等待前序阶段全部终态，之后后续优化器运行第3／4轮搜索，读取实际域检查、旧候选结果及人工拒绝原因。新搜索源码在既有63项及整块检查外，再检查FP16／BF16全位模式四类路径。第1轮0.658×稿因未对齐地址复用对齐编译实例被人工拒绝，不能据暂存搜索分数视作交付，也不能重用它作为已验证最佳。修改后的后续工具不改变已运行前两轮的内存版本；恢复需同时读取新身份及实际KG工作区。当前检查点已三端同步至`45ec9e0e3`，后续新增记录另行提交。

后续对话和新报告遵循[实验归属与汇报口径](docs/实验归属与汇报口径.md)。

- 文档使用中文。
- 算子成果按仓库来源分为FlagGems Table和FlagGems-SGLang Table两个文档。开头只保留固定版本信息，各分表使用H2标题，表头删除“完整调用／设备计时／多少项”等括号附注，不附其他说明表。FlagGems内分“关键算子”和“早期优化与修复”；SGLang成果主表只列gelu_and_mul、moe_sum_reduce，其他3项保留体检归档。
- 原“会议算子”统一称“关键算子”，按可交付性从高到低排序；原版是否达标以固定版本实测正确性和0.8×门槛判断，Excel仅是历史记录。用户已将目标更正为`_conj_physical`；旧`conj_physical`实测不能当作该接口的结果。
- SGLang日常待办只保留`gelu_and_mul`；MoE作为已交付成果、量化与旋转作为已体检项、masked SiLU作为参考口径观察项保留，默认不重复列为待优化。用户要求全表时仍列全部五项。
- 默认使用“Excel原记录、FlagGems实测、当前优化版、状态”等列；SGLang表不强行添加无对应记录的Excel列。
- Codex参与过的历史优化，不表述为纯原生KG独立结果；当前优化版写清KG/HG-KDA与Codex的实际贡献。
- 保留真实独立对照组、profiling配置及额外修复的区别，不改写历史证据。
- 后续默认一次完整评测；已有两次同条件完整有效结果的简表使用几何平均，原始结果继续保留。调整工具时须同步更新对应验收门禁。

# 项目汇报约定

## 用户硬约束：候选严格Triton

**baseline使用CANN原生库或来源明确的Ascend C参考；我们优化和交付的候选严格使用Triton。** 候选所有设备计算、复制、归约、扫描、布局转换及快／慢／回退路径均须由Triton kernel完成；禁止以Ascend C、CANN／aclnn、原生DMA、PyTorch目标算子或CPU计算替代。Python／C++仅可用于分配、元数据、编译缓存、参数打包、状态管理和Triton启动等宿主胶水。

完整边界与验收遵循[硬性规则](docs/Triton候选与CANN参考硬性规则.md)。历史混合后端结果不能计为严格Triton成果，原25/30和7/10＋18/20等统计须重新审查；2026-11-02严格Triton复核结果为16/30（1015为6/10、1029为10/20），详见`project/Ascend-Optimization-v1/operators/campaigns/strict-triton-20261102/复核结果简报.md`。不得直接续跑会选中非Triton候选的旧流程。这条用户规则优先于项目历史计划、知识库、skill及提示词中的冲突内容。

当前实验遵循[新25算子计划](project/Ascend-Optimization-v1/operators/下一阶段计划-20261029.md)：保留导师目标名称和顺序；CANN原生库与已核验Ascend C源码参考区分；PyTorch正确性和历史计时口径保留。新工具只改独立覆盖层，不修改共享服务/镜像。新知识有源码、请求、完整结果及补充检查身份才进入已验证索引；热点邻域、可观测空闲和因果瓶颈分开表述。

上述16/30为纠偏复核的冻结起点。夜间持续优化的当前交付数读取`project/Ascend-Optimization-v1/operators/campaigns/strict-night-20261102/progress.json`及`delivery.json`，不能用冻结起点覆盖新增成果。恢复运行前检查既有Goal、tmux和工作区状态，不重复创建Goal或启动活跃工作区。

2026-11-03的新goal位于`operators/campaigns/strict-five-20261103/`，从提交`231ba49d6`的严格25/30继续。tmux为`hanle-strict-five-20261103`，搜索状态是`search-state.json`；`state.json`只表示先行采集队列，不能因其`tasks_finished`判定整个goal结束。恢复须分别核对`optimizer-process.json`、`half-evaluator-process.json`、`half-comparisons-process.json`、`reference-stage-process.json`的PID与启动时间和KG工作区终态；搜索、两份FP16提案评测和参考探测在同一卡7锁下串行。已有启动脚本不得再次执行；排队成功、离线编译、局部参考探测或仅PyTorch成绩均不增加CANN交付数。工具和新批入口见`tools/profiling/hgkda/strict_five/README.md`。

后续对话和新报告遵循[实验归属与汇报口径](docs/实验归属与汇报口径.md)。

- 文档使用中文。
- 算子成果按仓库来源分为FlagGems Table和FlagGems-SGLang Table两个文档。开头只保留固定版本信息，各分表使用H2标题，表头删除“完整调用／设备计时／多少项”等括号附注，不附其他说明表。FlagGems内分“关键算子”和“早期优化与修复”；SGLang成果主表只列gelu_and_mul、moe_sum_reduce，其他3项保留体检归档。
- 原“会议算子”统一称“关键算子”，按可交付性从高到低排序；原版是否达标以固定版本实测正确性和0.8×门槛判断，Excel仅是历史记录。用户已将目标更正为`_conj_physical`；旧`conj_physical`实测不能当作该接口的结果。
- SGLang日常待办只保留`gelu_and_mul`；MoE作为已交付成果、量化与旋转作为已体检项、masked SiLU作为参考口径观察项保留，默认不重复列为待优化。用户要求全表时仍列全部五项。
- 默认使用“Excel原记录、FlagGems实测、当前优化版、状态”等列；SGLang表不强行添加无对应记录的Excel列。
- Codex参与过的历史优化，不表述为纯原生KG独立结果；当前优化版写清KG/HG-KDA与Codex的实际贡献。
- 保留真实独立对照组、profiling配置及额外修复的区别，不改写历史证据。
- 后续默认一次完整评测；已有两次同条件完整有效结果的简表使用几何平均，原始结果继续保留。调整工具时须同步更新对应验收门禁。

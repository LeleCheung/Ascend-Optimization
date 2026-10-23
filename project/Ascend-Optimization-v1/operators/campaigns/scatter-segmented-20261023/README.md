# scatter_add跨目标相消修复

这是remaining-five-20261023-v2的继续修复批次，不覆盖历史结果。先等待主队列及补验退出，再持同一物理卡7锁执行。原始正式合同不变，额外要求大数/小数分属不同target时互不干扰。任务、方法预算、贡献及复现入口见knowledge/夜间任务.md和本控制器。最终看deliverables/manifest.json与completion-audit.json，运行中不算交付。

## 最终结果

显式分段扫描版相对原样PyTorch/NPU参考为 **1.900×**，正式18/18、补充18/18、全部性能输入数值9/9、动态范围18/18通过。源SHA为`72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e`。一次正式完整调用结果保留，不重复测量挑分。

旧`8a67a004`版在新增NPU动态范围检查中为 **0/18**；虽然原随机测试通过、复测显示1.278×，仍判为无效。它用整行前缀相减求各target的和，会让其他target的大数吞掉本target的小数。新实现按target显式分段扫描，直接读取段尾，不再相减两个大前缀。原CC生成的排序、片上二分和移位解码保留；该修复、反例及独立验收由Codex完成。

有效候选在排队阶段已独立通过完整验收，因此后续由`finalize-scatter-segmented.py`完成父版反例核查、成对路径和metrics采集、正式结果复用与交付核验，停止重复模型搜索。原冻结控制器及计划保留，切换原因、实际入口SHA见`delivery-transition.json`。最终`completion-audit.json`通过；采集状态另见`endurance-state.json`，不以采集是否成功替代正确性与性能验收。

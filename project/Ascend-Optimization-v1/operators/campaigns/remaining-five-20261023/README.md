# 会议剩余五算子达标批次

**本批已停止并由[修正版v2](../remaining-five-20261023-v2/README.md)替代。** 准备器复用了早期上采样oracle的关键字参数调用，本机aten要求`self`，导致参考就绪失败，未进入有效优化。更正为位置参数后新建v2；用例、计算公式及容差不变。原始失败快照在raw-evidence/，停止身份与服务状态见reference-adapter-correction.json，原准备器按冻结SHA恢复在knowledge/prepare-remaining-five-original.py。下文为原任务描述，不表示本批仍在运行或已有交付。

目标是五项正确性修复与性能达标。使用固定FlagGems/FlagTree、物理卡7、KGS19659完整调用walltime。各候选一次完整正式复验及补充语义、全部性能输入数值检查。服务器tmux连续运行，失败保留后转下一项；运行状态不等于最终交付。

对象：upsample_bilinear2d、_conj_physical、scatter_add、cumulative_trapezoid、fractional_max_pool3d。_conj_physical新建正确接口合同；scatter从上一批0.419×有效源码继续，其余从固定上游起步。合同和源码SHA见plan.json。

CPU回退参考与原生NPU参考分别标注，不通过调整容差或替换参考制造达标。最终源码在deliverables/，过程与失败见endurance-state.json、results/和research/。目标、方法预算与贡献见knowledge/夜间任务.md。

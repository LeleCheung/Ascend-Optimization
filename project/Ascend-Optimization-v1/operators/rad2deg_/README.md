# rad2deg_

将输入张量中的弧度原地换算为角度，即乘以 `180 / π`，返回与输入共享存储的结果。非连续布局、数据类型、原地副作用和返回别名均属于合同。

2026-10-15 会议指定的简单算子首项。Excel 记录为精度失败；当前固定 master 是否仍失败，以本批完整 KGS 测试为准。

基线来自固定 FlagGems 的 `src/flag_gems/ops/rad2deg.py`，仅追加 `run = rad2deg_` 入口，见 [candidates/flaggems-fixed.py](candidates/flaggems-fixed.py)。FlagGems/FlagTree 身份见[阶段锁定](../../../../manifests/hgkda-stage2.lock.json)。

首批计划：完整基线 → 分阶段 HG-KDA 两轮 → 两次最终复验。原生 KG 覆盖层另验证角色/skill 加载；若需要四组公平对照，使用同一起点和预算另开批次，不混用成绩。

本阶段流程与其他对象见[下一阶段计划](../下一阶段计划-20261015.md)。边界掩码优化通过测试后仍需专家评审。

# rad2deg_

将输入张量中的弧度原地换算为角度，即乘以 `180 / π`，返回与输入共享存储的结果。非连续布局、数据类型、原地副作用和返回别名均属于合同。

2026-10-15 会议指定的简单算子首项。Excel 记录为精度失败；本批固定 FlagGems 已完整通过，此前失败在当前版本未复现。

| 版本 | 相对 PyTorch | 相对固定 FlagGems | 完整评测 |
|---|---:|---:|---:|
| FlagGems | 0.253× | 1.00× | 33/33 |
| HG-KDA 复验1 | 0.803× | 3.20× | 33/33 |
| HG-KDA 复验2 | 0.806× | 3.19× | 33/33 |

每次为18项正确性与15项性能用例；加速比取性能用例的几何平均，采用设备 kernel 计时。另通过21项原地、步长、边界和特殊值补充检查。最终源码 SHA：`5addf7b1c7ecfb9413cbdeb045b3e74be71bb1278df314ae62bb1d3b65556a70`。

优化将通用逐元素调度换成连续输入专用分块与有界网格循环；最终版另修复了尾部覆盖、无掩码条件及原地非连续访问。CC/DeepSeek 生成第一轮性能方案，Codex 发现并修复一般语义问题。第一轮原源码虽通过现有 KGS 用例，仍被独立审查拒绝，不作为最终成果。第二轮采集模拟器热点成功，但候选变慢并回退，不能据此声称指令分析必然提高性能。边界检查变更的专家评审仍待完成。

基线来自固定 FlagGems 的 `src/flag_gems/ops/rad2deg.py`，仅追加 `run = rad2deg_` 入口，见 [candidates/flaggems-fixed.py](candidates/flaggems-fixed.py)。FlagGems/FlagTree 身份见[阶段锁定](../../../../manifests/hgkda-stage2.lock.json)。

本批已完成完整基线、两轮生成与独立修复后的两次最终复验。原生 KG 覆盖层已验证角色/skill 加载，尚未执行原生 KG 加载该覆盖层的完整优化批次；本成绩来自独立 HG-KDA 控制器和修复。若需要四组公平对照，使用同一起点和预算另开批次，不混用成绩。

最终源码见 [candidates/hgkda-final-20261015.py](candidates/hgkda-final-20261015.py)，逐 case 对比见[版本对比](reports/版本对比-20261015.md)。完整失败记录与判定见[独立审查](reports/review-20261015.json)。

910B 宿主机执行 `bash project/Ascend-Optimization-v1/operators/rad2deg_/scripts/reproduce-910b.sh 新批次名` 可复测固定 FlagGems、补充语义和最终源码两次；这属于源码复现。重新运行模型生成则使用 `tools/profiling/hgkda/run-first-operator.sh`，属于新实验，不保证生成相同源码。

本阶段流程与其他对象见[下一阶段计划](../下一阶段计划-20261015.md)。边界掩码优化通过测试后仍需专家评审。

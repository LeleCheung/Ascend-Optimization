# 夜间连续优化大 goal

本批是持续执行的优化队列，**不是只复现baseline**。优先SELU显式缓冲与中间量优化，之后依次执行`rad2deg_`、`cumulative_trapezoid`、`scatter_add`、`uniform`、`gelu_and_mul`。服务器由tmux托管，各阶段自动衔接；控制端断开不影响既有队列运行。Codex持续负责诊断和排障，不在单项完成后询问是否继续。

固定FlagGems `d6a8eec473517a3d68157b208eb9c057eb1d4c50`、FlagGems-SGLang `938612058a981487e439da514df7d1371cf31e68`、FlagTree `0.6.0+ascend.gitc286cba6`、物理卡7和KGS19659完整调用walltime。测试合同从已执行批次逐文件复制，SHA写入[计划](plan.json)；SELU从上轮最终有效源码起步，其他五项从固定FlagGems源码起步。

## 连续执行流程

1. 核对环境和冻结源码、合同；每项在本批测量起点，保留失败，不把精度失败的局部性能作为整组成果。
2. 同一输入采集参考和候选的真实设备任务及实际编译IR，采集KGS真机metrics。将证据和[既有机制结论](knowledge/方法与机制.md)随本批新覆盖层加载进模型角色。
3. CC/DeepSeek执行四轮KG生成与完整测试。独立控制器对每份通过KG完整测试的候选执行补充门禁，再做一次独立完整KGS评测。
4. 按同case候选实际延迟保留最佳。没有达到0.8×的对象自动进入第二批三轮搜索；SELU即使已过0.8×，未比同批起点真实提升3%也继续第二批。单个候选失败后转下一个，搜索失败后转下一算子。
5. 最终有效最佳重新采集真实执行路径及metrics，导出源码、逐case证据、角色加载身份与中文报告。Codex完成离线验收与Windows/910B/GitHub同步后统一汇报。

总搜索上限12小时，单算子两批搜索分别上限60/45分钟；不是必须等满预算。单次模型profiling请求超过240秒，控制器只取消本worker登记的精确operation ID，保存回执，不重启服务。未知设备请求或worker未确认退出时先排障，不盲目发新任务。原始工作区在`/data/hanle/ascend-optimization/runs/endurance-20261022/`，不会覆盖历史批次。

## 验证与成果口径

SELU继续要求42项布局/特殊值、12项大连续输入、9项范围检查；其他五项由`check-endurance.py`额外检查18项，各两次并切换stream，覆盖未见形状/stride/参数与特殊语义。所有对象都须通过原KGS完整合同，补充门禁不参与性能计分、不替代完整上游pytest。相同字节源码的本批正式结果复用，失败和成功均不通过重复测量挑最高分。

运行中追加：累计积分的性能输入会触发不同于小正确性用例的Cube扫描分支，而Native计时phase不核对输出。独立门禁因此再检查全部9项冻结性能输入的数值，各两次并切换stream，按原合同容差逐元素比较；最终审计要求这些检查完整通过。修正回执见`timing-correctness-gate-amendment-01.json`，原合同及搜索成绩不改写；搜索超过PyTorch仍不自动代表候选可交付。

同一门禁随后扩展至`scatter_add`与`gelu_and_mul`的全部冻结性能输入，回执为`timing-correctness-gate-amendment-02.json`。候选放在所属批次目录内，以便检查器定位`plan.json`和原始oracle；检查器核对合同SHA，并验证输出数值、全部Tensor输入保持不变和当前stream。原有18项新形状检查继续保留。

`uniform`另增加3项生成器检查：高位seed有区分、高位offset有区分、推进与恢复状态可重放。检查包含第31/32/40位，防止截短随机状态后只靠均值方差过关；不要求候选与Philox逐值相同，也不把这些有限检查当作完整随机质量认证。回执见`uniform-generator-gate-amendment-03.json`，算法变更及剩余统计限制须单列。

门禁适配的后续修正单独留痕：`native-timing-gate-amendment-04.json` 修正低精度 Tensor 与 Python 区间边界比较的误拒，并支持 SGLang Native 输入配方；15 项 GELU 性能输入已在 CPU 上逐值对照 KGS 的输入生成器。`observer-input-recipe-amendment-05.json` 让成对采集复用该配方；`observer-wrapper-ir-amendment-06.json` 透过固定 LibEntry/LibTuner 的 `fn` 封装采集已实际编译的 IR，避免误记为零项。原始失败和旧采集保留，工具 SHA 按回执更新，不修改 KGS 合同。

SELU必须降低实际延迟，不能把复测分数变化称作新增优化。FlagGems实测与当前优化版分列，会议算子和SGLang分别汇报。Codex提供诊断、控制器和追加排障，CC/DeepSeek负责实际代码生成；不称为纯原生KG独立成果。没有指令报告或匹配的流量与峰值数据时，不宣称成功模拟器分析或数值Roofline。

运行中的状态见`endurance-state.json`、`heartbeat.json`、`进展.md`；这些文件在服务器先产生。完成后的最终源码在`deliverables/`，不能把正在运行的队列状态当作最终优化成果。

## 服务器启动

```bash
cd /data/hanle/ascend-optimization/Ascend-Optimization
python3 project/Ascend-Optimization-v1/tools/profiling/hgkda/run-endurance.py \
  --batch endurance-20261022
```

本批实际由`hanle-endurance-20261022` tmux会话执行。控制器持有项目卡锁，所有设备工作串行。冻结工具SHA见[控制器清单](knowledge/tool-sha256.json)，任务知识见[夜间任务](knowledge/夜间任务.md)。

## 运行中发现的采集兼容问题

rad2deg_的CC裸启动候选完整评测通过，但原生msprof采集没有产生完成标记。原失败请求与分析保留，不将推测写为已获得的硬件结论。Codex准备了独立采集兼容候选：在采集launcher要求Tensor元数据时，用同一计算kernel的标准启动；缓存键另纳入N模16，防止跨整除属性错误复用；绑定阶段只编译，在成功绑定或异常回退后统一执行本次原地计算，避免后续绑定异常造成重复执行。生成器以AST核对计算kernel未变化，仍须重新通过补充语义、一次完整评测和真实采集。

补充验证由`hanle-endurance-rad-repair-20261022`等待主队列结束并取得同一卡锁后自动执行，结果写到独立`endurance-20261022-rad-profile-repair/`，不覆盖主队列原始候选、门禁或成绩。它属于额外Codex修复，不能作为CC原生独立生成成果。此段记录排队计划，不表示修复已经在NPU上验证成功。

`gelu_and_mul` 的父版独立检查最初错误地假设输入采用 `spec` 配方，因此正式父版评测尚未提交。已修复配方适配，另由 `hanle-endurance-gelu-parent-20261022` 等待主队列和同一卡锁，按同字节固定 FlagGems-SGLang 源码补齐门禁、首次正式完整评测与成对采集，保存到 `endurance-20261022-gelu-parent-repair/`。这属于基础设施修复，不是新增优化；原始故障不覆盖。

第一批结束后的诊断继续供第二批新覆盖层使用：累计积分需保留低精度相邻和、缩放的中间舍入；散射需先修 BF16 累加，父版代表实例的真实 IR 已证实原子写入被逐 lane 循环展开；随机数需使用真实生成器状态及完整高位。具体适用范围、失败源码与建议分别保存在 `knowledge/` 和 `research/`，不能将尚未通过独立门禁的搜索高分发布为有效交付。

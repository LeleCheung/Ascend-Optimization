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

SELU必须降低实际延迟，不能把复测分数变化称作新增优化。FlagGems实测与当前优化版分列，会议算子和SGLang分别汇报。Codex提供诊断、控制器和追加排障，CC/DeepSeek负责实际代码生成；不称为纯原生KG独立成果。没有指令报告或匹配的流量与峰值数据时，不宣称成功模拟器分析或数值Roofline。

运行中的状态见`endurance-state.json`、`heartbeat.json`、`进展.md`；这些文件在服务器先产生。完成后的最终源码在`deliverables/`，不能把正在运行的队列状态当作最终优化成果。

## 服务器启动

```bash
cd /data/hanle/ascend-optimization/Ascend-Optimization
python3 project/Ascend-Optimization-v1/tools/profiling/hgkda/run-endurance.py \
  --batch endurance-20261022
```

本批实际由`hanle-endurance-20261022` tmux会话执行。控制器持有项目卡锁，所有设备工作串行。冻结工具SHA见[控制器清单](knowledge/tool-sha256.json)，任务知识见[夜间任务](knowledge/夜间任务.md)。

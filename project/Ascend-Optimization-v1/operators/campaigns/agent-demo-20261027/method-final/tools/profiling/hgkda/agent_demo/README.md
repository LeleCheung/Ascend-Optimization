# 单算子三步演示

目标是在同一个matmul_bias_activation上展示：候选与PyTorch同输入真机对比；源码／实际IR／经验Roofline诊断；候选指令热点与代码映射，再对修改版做完整KGS验证。

使用固定KG6.7.0／KGS6.5.0环境。新版上游访问认证问题不影响本次演示；不改共享服务、镜像或编译器。分析模型为宿主CC＋DeepSeek，NPU阶段只使用物理卡7和项目互斥锁。

演示起点采用roofline-20261027已通过的历史优化父版。已有matmul-pipeline优化经验可作为明确的知识先验；重现同一种优化不称为独立发现或新方法收益。新采集、新模型调用和新评测分别留证。

正式成绩使用KGS原合同设备kernel时间；PyTorch的GEMM组件指标与候选融合kernel的指标分开说明。模拟器时间只用于热点定位，静态IR不代替动态指令。经验Roofline参考不称物理峰值。

控制器应保存每一步输入、命令、输出及SHA，恢复时核对已有产物，不重复发送状态未知的NPU请求。完整评测每源码默认一次。

## 已完成的演示

见[核验版演示稿](../../../../operators/campaigns/agent-demo-20261027/三步演示-核验版.md)及[可直接打开的HTML](../../../../operators/campaigns/agent-demo-20261027/演示-核验版.html)。本批两版各42/42及18＋24补充检查，15项实际设备延迟比为1.023×；候选相对PyTorch0.884×。入选源码与历史成功版相同，表示本次新模型调用与新采集重现有效方法。

## 一条命令运行

在910B宿主控制器环境执行，输出使用新的空目录：

~~~bash
/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python \
  /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/hgkda/agent_demo/run.py \
  --batch /data/hanle/ascend-optimization/runs/新的三步演示
~~~

可用tmux承载。控制器自动取得卡7项目锁、检查19655／19657／19659健康空闲，进入固定tle_yy执行采集与评测；模型调用在宿主运行。当前是接在既有KGS上的独立演示控制器；模型在JSON中选择受限工具，控制器实际读取返回，不冒充已经改成KG原生MCP工具。

| 文件 | 职责 |
|---|---|
| run.py | 统一入口、阶段恢复、项目锁及有限步骤 |
| collect.py | 复用真机采集器，生成同输入成对指标比较 |
| instruction.py | 限定目标kernel、单核模拟器采集，KGS解析及HG-KDA热点过滤 |
| evidence.py | 源码／IR／经验Roofline／指标／热点的结构化证据 |
| agent.py | 两次CC调用：先选证据工具，读取后选优化动作 |
| validate.py | 开发优胜冻结、18＋24补充检查、每源码一次完整KGS |
| feedback.py | 新采集对照原预测，并真实调用模型复核 |
| figure.py、report.py | 中文Roofline图、Markdown及HTML演示 |
| audit.py、test_evidence.py | 独立CSV／源码／映射／模型输入／评测核验 |

## 模型与脚本的边界

脚本负责采集、聚合、差异、输入身份、源码变换、计时、完整评测与证据校验；模型负责选证据工具、解释机制、选择有限动作及回看预测。已有通用模板及成功知识均明确来源于此前优化经验。

当前不自动把模型总结写进知识库。本批模型曾将交错方案的开发通过写成正式通过，并将GM/L1不变泛化为全部层次不变；核验发现UB读写实际下降约16.9%。核验版报告采用原始数据，原模型回执继续保存。

## 模拟器状态

首次两次尝试因模拟器重写设备环境失败，原件保留。第三次实际产生1087条起点指令、983条源码映射；候选926条、862条映射。两者计算／导出后子进程异常退出，按部分证据记录。热点仅反映选定core的指令观测，debug来源行不是唯一因果，不合成为真机关键路径。

## 离线核验

~~~bash
python -m unittest discover -s project/Ascend-Optimization-v1/tools/profiling/hgkda/agent_demo -p 'test_*.py' -v
python project/Ascend-Optimization-v1/tools/profiling/hgkda/agent_demo/audit.py \
  --batch project/Ascend-Optimization-v1/operators/campaigns/agent-demo-20261027 \
  --output /tmp/新的演示证据核验.json
~~~

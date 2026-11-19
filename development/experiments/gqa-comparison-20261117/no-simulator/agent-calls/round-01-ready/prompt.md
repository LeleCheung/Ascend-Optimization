请优化工作区candidate.py中的GQA严格Triton实现。首先阅读task.json、context-brief.json、comparison.json、official-check-01.json、相关实际IR。若工作区提供simulator-summary.json，再阅读其列出的模拟器热点与区域报告。不要读取其他实验目录或其他算子的历史最佳源码。
目标是降低固定case 4的完整调用耗时，同时保留case 1、2、4、7的正确性和性能结果。所有修改按输入元数据泛化，不能根据测试case编号或输入值特判；不能改变数学语义、输出dtype或容差。设备计算完全Triton，宿主仅元数据、分配、启动；不得调用PyTorch/CANN做计算。当前源码四项已通过固定官方检查器，旧未舍入FP64的诊断公式不是本轮验收条件。
先依据证据选择一个性能机制，写入diagnosis.json并调用decision登记，再通过Edit修改candidate.py。说明证据、替代解释、预期指标和验证方式。优先小范围修改，保持启动接口与正确路径。不要为了原始诊断MARE而添加不必要的计算，也不要未经验证移除精度处理。
本任务的设备检查由宿主调用固定GQA评测器完成，不调用FlagGems/KGS preflight工具。完成源码和决策落盘后返回简要说明，等待完整四项评测反馈。仅允许修改candidate.py和diagnosis.json。

工具桥命令：/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python /data/hanle/ascend-optimization/runs/gqa-attribution-20261117/method/hgkda/src/hgkda/runtime/agent_tools.py --config /data/hanle/ascend-optimization/runs/gqa-attribution-20261117/no-simulator/workspace/agent-config.json
开始一个具体修改前，先把当前源码SHA、引用证据、假设、改法、预期变化及验证方式写入diagnosis.json，
再调用工具桥decision --record diagnosis.json，保存修改前决策。证据项包含path、category、role、finding；
category取paired_profiling/simulator/source_ir/roofline/knowledge/probe/other，role取trigger/disambiguation/mechanism/background。
没有使用模拟器就不填模拟器；读过多个工具不自动声明因果贡献。编译失败、退步及保持源码不变也保留记录。
每个机制可对应多次小编辑，不要求每次文本替换都重新写长分析；正式结果后更新诊断中的观测与未决问题。

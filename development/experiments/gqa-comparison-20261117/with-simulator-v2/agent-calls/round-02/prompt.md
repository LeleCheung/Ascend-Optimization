先读取next-performance-feedback.json，这是你上一轮源码的完整四项结果。根据实际结果选择保留或修正一个机制。若目标已有明确收益且全部通过，可保留；否则提出下一项改动。原始profiling/IR/模拟器属于起点源码，引用时使用role=background，不作为本轮新源码的实测。当前评测反馈引用category=evaluation。请在最多6次读取后登记decision并Edit。
你负责修改GQA严格Triton候选。先用Read工具读取candidate.py、evidence-summary.json、ir-excerpt.txt。若存在simulator-summary.json，再用Read阅读它；原始IR仅在具体问题无法由摘录判断时按行读取。两组均未提供GQA专项知识。
请在最多6次读取后选定一个具体性能机制，使用Edit更新已有diagnosis.json，调用下方工具桥decision保存修改前记录，再使用Edit修改已有candidate.py。只用Bash运行工具桥decision；不使用Bash读取、搜索或写文件，不调用context或preflight，不访问工作区外路径。设备评测和只读代码审查由宿主完成。
目标：固定case4变快，全部case1/2/4/7继续通过官方checker。保持数学语义和输出dtype。只按通用shape/dtype等元数据选择路径，不按case编号或张量数值特判。不调用PyTorch/CANN计算，宿主仅分配与启动。当前代码四项已经通过官方检查；直接对未舍入FP64计算的旧MARE不是本轮验收条件。精度处理是否必要须由完整评测验证。
性能证据引用workspace/evidence-summary.json，category=paired_profiling；源码/IR引用workspace/candidate.py或workspace/ir-excerpt.txt，category=source_ir；模拟器材料仅在实际提供且使用时引用category=simulator。缺失知识库不要编造。请说明一个假设、可能的另一种解释、修改内容和预期，再小范围实现。
完成Edit后停止并交接；不要重复读遍全部IR或技能文档，不自行运行测试。下一轮会提供固定四项完整结果。

工具桥命令：/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python /data/hanle/ascend-optimization/runs/gqa-attribution-20261117/method/hgkda/src/hgkda/runtime/agent_tools.py --config /data/hanle/ascend-optimization/runs/gqa-attribution-20261117/with-simulator-v2/workspace/agent-config.json
开始一个具体修改前，先把当前源码SHA、引用证据、假设、改法、预期变化及验证方式写入diagnosis.json，
再调用工具桥decision --record diagnosis.json，保存修改前决策。证据项包含path、category、role、finding；
category取paired_profiling/simulator/source_ir/roofline/knowledge/probe/other，role取trigger/disambiguation/mechanism/background。
没有使用模拟器就不填模拟器；读过多个工具不自动声明因果贡献。编译失败、退步及保持源码不变也保留记录。
每个机制可对应多次小编辑，不要求每次文本替换都重新写长分析；正式结果后更新诊断中的观测与未决问题。

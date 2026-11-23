修复当前GQA未保留候选。读取task.json、contract.json、context-brief.json、repair-feedback.json、candidate.py及必要的实际失败IR。四项原合同结果已全部返回；先核对每项真实错误与UB字节需求。保持四项checker、golden、dtype、计时口径及两项概率分解精度要求。当前扩大KV分块触发UB超限；请选择一个具体资源修复机制，减少活跃缓冲、改变生命周期或调整分块，并预期其对交接次数和并行度的取舍。原保留版独立冻结，修复沿当前失败源码明确接续。
原版profiling和模拟器仅为历史背景。本次决策以当前失败反馈和实际IR为依据，evidence使用evaluation/source_ir类别；无需重读原版模拟器，不将其标为当前候选观测。不要声称当前源码已通过编译或存在真机收益。用Edit写diagnosis.json，包含selected_layer、layer_reason、simulator_incremental_judgment及证据、假设、改法、预测、验证与未知项，调用工具桥decision --record diagnosis.json后才编辑candidate.py。全部候选修复由你完成，host仅分配、元数据和启动，设备计算严格Triton。工具桥preflight返回awaiting_source_review后保持字节并交接。无总轮数、工具或时间预算，完整四项反馈后继续研究。

工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/night-whitebox-20261120/gqa/decision_bridge.py --config /data/hanle/ascend-optimization/runs/night-whitebox-20261120/gqa/workspace/agent-config.json
性能优先策略：不因默认轮数、耗时、cost或已达验收线停止优化。每次交接审查与完整评测只是阶段边界；宿主应根据真实反馈继续调用。保留每个候选的源码与证据，不为减少调用次数省略验证。

开始一个具体修改前，先把当前源码SHA、引用证据、假设、改法、预期变化及验证方式写入diagnosis.json，
再调用工具桥decision --record diagnosis.json，保存修改前决策。证据项包含path、category、role、finding；
category取paired_profiling/simulator/source_ir/roofline/knowledge/evaluation/probe/other，role取trigger/disambiguation/mechanism/background。
没有使用模拟器就不填模拟器；读过多个工具不自动声明因果贡献。编译失败、退步及保持源码不变也保留记录。
每个机制可对应多次小编辑，不要求每次文本替换都重新写长分析；正式结果后更新诊断中的观测与未决问题。

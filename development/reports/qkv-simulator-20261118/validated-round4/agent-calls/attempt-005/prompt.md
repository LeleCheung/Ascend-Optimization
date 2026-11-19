Repair the current failed candidate within its existing performance hypothesis. Read task.json, contract.json, context-brief.json, previous-result.json and candidate.py. Identify the actual failure stage and exact diagnostic. For compiler resource failures, inspect the reported memory requirement and adjust tile geometry or buffer lifetime with a focused change. For numerical failures, inspect arithmetic order, precision, masks and independent-output semantics. Preserve verified numerical fixes and generic layout paths. Use the current feedback as the authority for which checks passed.
Current-source simulator and profiling evidence are unavailable for this repair. Their existing files describe earlier sources and remain historical context. Record the actual diagnostic, proposed repair, expected effect and validation in diagnosis.json; register through the decision bridge before editing candidate.py. All candidate changes are authored by you. Use Edit for candidate.py and diagnosis.json; device checks use the supplied bridge. Invoke preflight and hand off unchanged on awaiting_source_review. Strict Triton computation and allocation/metadata-only host operations remain required. Keep the candidate self-contained with permitted imports. No total search or tool-call budget.

工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/qkv-simulator-20261118/decision_bridge.py --config /data/hanle/ascend-optimization/runs/qkv-simulator-20261118/workspace/agent-config.json
性能优先策略：不因默认轮数、耗时、cost或已达验收线停止优化。每次交接审查与完整评测只是阶段边界；宿主应根据真实反馈继续调用。保留每个候选的源码与证据，不为减少调用次数省略验证。

开始一个具体修改前，先把当前源码SHA、引用证据、假设、改法、预期变化及验证方式写入diagnosis.json，
再调用工具桥decision --record diagnosis.json，保存修改前决策。证据项包含path、category、role、finding；
category取paired_profiling/simulator/source_ir/roofline/knowledge/evaluation/probe/other，role取trigger/disambiguation/mechanism/background。
没有使用模拟器就不填模拟器；读过多个工具不自动声明因果贡献。编译失败、退步及保持源码不变也保留记录。
每个机制可对应多次小编辑，不要求每次文本替换都重新写长分析；正式结果后更新诊断中的观测与未决问题。

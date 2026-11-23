继续优化CANNBench Level4 GQA。当前观察父版本357.02675μs，保留最佳328.44680μs；父版本身份见starter-provenance.json。先读取skills/hgkda/SKILL.md、task.json、contract.json、context-brief.json、previous-result.json、current-profiling.json、simulator-evidence.json、candidate.py及相关实际IR。固定case1/2/4/7、原checker、golden、输入输出dtype和计时合同；目标case4耗时低于保留最佳328.44680μs，其他三项真实变化同时保存。当前概率计算结构沿父版本；调整中间dtype或计算顺序须事前明确数值风险与适用输入范围，由原完整四项checker验收，精度阈值保持。
模拟器为三个核的部分窗口，Cube和Vector各自归一化；明确热点和流水各自增加的判断，保留全调用completion缺失。选一个具体机制。Triton可表达的分块、布局和计算组织优先修改源码；发现单算子生成缺陷时在diagnosis中交接IR建议，本调用仅允许修改candidate.py与diagnosis.json。
修改前用Edit填写diagnosis.json，字段包含source_sha256、evidence(path/category/role/finding)、hypothesis、alternatives、discriminating_observation、change、expected、validation、selected_layer、layer_reason、simulator_incremental_judgment。证据路径使用workspace/文件名，category按paired_profiling/simulator/source_ir/evaluation分类。实际成功读取当前模拟器后，调用桥decision --record diagnosis.json登记，再改源码。仅用Edit编辑候选；Bash只调用工具桥。设备计算全部严格Triton，host仅分配、元数据和启动；禁止按case_id或输入值选路、输出缓存和外部后端计算。公共入口与原候选保持。
实现一个可评测机制后调用preflight --source candidate.py；收到awaiting_source_review后保持字节并交接宿主只读审查及完整四项真机评测。完整结果与修改后重放随后反馈。生成不设总轮数、工具或时间预算，此处交接属于验收边界。新增诊断和注释使用专业中文，简明记录实际收益与代价。

本调用由宿主明确指定的独立实验：Use qualified negative009 as immediate source parent for a resource experiment, explicitly keeping accepted006 frozen. Change only metadata selection to BM256/BN256 for the same BF16 noncausal geometry tested by008; preserve009 kernel AST, arithmetic, earlyV organization, precision and fallback metadata. Compare actual target UB allocation with008 overflow230912B, other cases unchanged. Do not substitute a different mechanism. Failure is a recorded compiler-resource result. Retention requires beating accepted006328.4468us with all original four checks and post observations.

工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/gqa-goal-20261121/gqa/decision_bridge.py --config /data/hanle/ascend-optimization/runs/gqa-goal-20261121/gqa/workspace/agent-config.json
性能优先策略：不因默认轮数、耗时、cost或已达验收线停止优化。每次交接审查与完整评测只是阶段边界；宿主应根据真实反馈继续调用。保留每个候选的源码与证据，不为减少调用次数省略验证。

开始一个具体修改前，先把当前源码SHA、引用证据、假设、改法、预期变化及验证方式写入diagnosis.json，
再调用工具桥decision --record diagnosis.json，保存修改前决策。证据项包含path、category、role、finding；
category取paired_profiling/simulator/source_ir/roofline/knowledge/evaluation/probe/other，role取trigger/disambiguation/mechanism/background。
没有使用模拟器就不填模拟器；读过多个工具不自动声明因果贡献。编译失败、退步及保持源码不变也保留记录。
每个机制可对应多次小编辑，不要求每次文本替换都重新写长分析；正式结果后更新诊断中的观测与未决问题。

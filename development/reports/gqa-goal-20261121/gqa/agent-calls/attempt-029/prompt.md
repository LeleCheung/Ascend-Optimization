继续优化CANNBench Level4 GQA。当前观察父版本262.27775μs，保留最佳262.27775μs；父版本身份见starter-provenance.json。先读取skills/hgkda/SKILL.md、task.json、contract.json、context-brief.json、previous-result.json、current-profiling.json、candidate.py及相关实际IR。固定case1/2/4/7、原checker、golden、输入输出dtype和计时合同；目标case4耗时低于保留最佳262.27775μs，其他三项真实变化同时保存。当前概率计算结构沿父版本；调整中间dtype或计算顺序须事前明确数值风险与适用输入范围，由原完整四项checker验收，精度阈值保持。
本调用提供当前成对profiling与实际IR，当前假设由这些材料及源码分析提出。宿主保留此前模拟器知识，此轮属于实际续接；工具增量仅按本轮实际输入记录。修改后仍开展模拟器重放。选一个具体机制。Triton可表达的分块、布局和计算组织优先修改源码；发现单算子生成缺陷时在diagnosis中交接IR建议，本调用仅允许修改candidate.py与diagnosis.json。
修改前用Edit填写diagnosis.json，字段包含source_sha256、evidence(path/category/role/finding)、hypothesis、alternatives、discriminating_observation、change、expected、validation、selected_layer、layer_reason、simulator_incremental_judgment。证据路径使用workspace/文件名，category按paired_profiling/simulator/source_ir/evaluation分类。实际成功读取当前profiling、相关TTIR与BCMLIR后，调用桥decision --record diagnosis.json登记，再改源码。simulator_incremental_judgment明确当前剩余歧义及本轮无需新增模拟器输入的理由；evidence仅引用本轮实际提供和读取的材料。仅用Edit编辑候选；Bash只调用工具桥。设备计算全部严格Triton，host仅分配、元数据和启动；禁止按case_id或输入值选路、输出缓存和外部后端计算。公共入口与原候选保持。
实现一个可评测机制后调用preflight --source candidate.py；收到awaiting_source_review后保持字节并交接宿主只读审查及完整四项真机评测。完整结果与修改后重放随后反馈。生成不设总轮数、工具或时间预算，此处交接属于验收边界。新增诊断和注释使用专业中文，简明记录实际收益与代价。

本调用由宿主明确指定的独立实验：从当前验收019独立出发，仅为其现有非因果BF16、D128、完整S/SKV整除域实现20个常驻worker的连续tile分区。总tile定义B*NKV*G*(S/BLOCK_M)，每worker区间用floor(pid*T/P)到floor((pid+1)*T/P)，保持BM128/BN256、原单次FP16 PV、FP32softmax及PV后归一化和输出BSND。按KV组切分worker区间：每组K/V仅加载及转换一次，再循环该worker与该组相交的querytile；每tile原表达式不变。原FP16和其他fallback保持。不要借用028scores压缩或三阶段机制。由你选择合法Triton实现并记录实际IR、浮点顺序与活跃缓冲风险；所有设备计算严格Triton、host仅元数据分配与启动。原四项验收，无新增测试。20是当前硬件调度参数，采用元数据判据；实际原四项及后采集验证假设。先读当前019相关TTIR/BCMLIR与profiling后登记，模拟器输入按本轮实际材料记录。

工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/gqa-goal-20261121/gqa/decision_bridge.py --config /data/hanle/ascend-optimization/runs/gqa-goal-20261121/gqa/workspace/agent-config.json
性能优先策略：不因默认轮数、耗时、cost或已达验收线停止优化。每次交接审查与完整评测只是阶段边界；宿主应根据真实反馈继续调用。保留每个候选的源码与证据，不为减少调用次数省略验证。

开始一个具体修改前，先把当前源码SHA、引用证据、假设、改法、预期变化及验证方式写入diagnosis.json，
再调用工具桥decision --record diagnosis.json，保存修改前决策。证据项包含path、category、role、finding；
category取paired_profiling/simulator/source_ir/roofline/knowledge/evaluation/probe/other，role取trigger/disambiguation/mechanism/background。
没有使用模拟器就不填模拟器；读过多个工具不自动声明因果贡献。编译失败、退步及保持源码不变也保留记录。
每个机制可对应多次小编辑，不要求每次文本替换都重新写长分析；正式结果后更新诊断中的观测与未决问题。

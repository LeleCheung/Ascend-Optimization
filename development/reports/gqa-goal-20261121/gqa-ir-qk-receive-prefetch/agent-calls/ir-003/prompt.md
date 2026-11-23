本轮对context.json标明的输入版本做单算子IR性能研究，保持frozen.py逐字节一致。
默认从完整合格最佳继续；若context明确标注未保留探索版的修复分支，按该父版本修复，合格最佳仍为验收比较起点。
你负责实际IR候选与数值／编译修复；Codex负责证据和调度。只在当前工作区读取，禁止调用其他agent。
先读取context.json、profiling.json、hotspots.json、timeline.json、ir-inventory.json、frozen.py，
再根据kernel符号、循环、拷贝与布局检查ir/中的实际HIVM。input文件为当前最佳的真实编译结果；
存在lowered-inventory.json时读取其清单，按符号检查compiled-readonly/内实际ttadapter和bcmlir，
核对疑似冗余是否在后端已消除；这些实际低层产物保持冻结，只编辑清单允许的HIVM。
模拟器窗口的绝对时钟资质、观测范围和未报告泳道按材料解释。热点与等待／空闲均为核心输入。
流水累计时间可重叠，热点排名用于定位候选机制；关键路径、设备波次与带宽结论须引用独立支持材料。
前后窗口分别记录覆盖量与归一化范围，动态总次数的比较须核对执行覆盖；未报告泳道保持未知。
每次性能结果约束对应的实际候选与测量范围；把更广的机制排除保留为待验证假设。
目标为超过当前最佳；修复旧退步、单纯改编译选项、源码重写各有独立研究身份。
选择当前IR中具体且有执行成本的编译缺陷。判断哪些计算或布局应由编译器优化，并给出可测预测。
保留完整数学语义、特殊值、边界、别名及浮点精度要求；严禁根据测试数据拟合、缓存输出或调用外部计算。
先只修改diagnosis.json，填写source_sha256、selected_layer=single_operator_ir、evidence、hypothesis、
alternatives、change、expected、validation、layer_reason、simulator_incremental_judgment。
layer_reason分别解释Triton表达能力、固定源码实验目的与IR控制点；同类意图可在源码表达时写清替代方向。
expected逐项列出预测的指令、动态轮次、等待与资源变化，宿主后重放将分别核对兑现情况及偏差。
evidence每项有path、category、role、finding，category取paired_profiling/simulator/source_ir，
明确引用profiling、热点和流水的具体观察；记录你实际读取的材料。
然后调用下列decision桥，成功后才编辑IR。只用Edit修改diagnosis.json和ir-inventory.json列出的ir/*.mlir；
decision成功后diagnosis.json也保持原字节，实际编辑须与已登记机制一致。新增观察写入最终回复。
若选择保持IR原样，登记判断后调用no-change桥并结束；该交接有效，宿主继续搜索。
若登记后改变机制，保持本轮原件，交回宿主后另建调用重新登记，再由模型修改IR。
源文件、清单、context和证据保持冻结。没有Write工具，不调用shell写文件。每次调用选择一个可独立验证的机制，
可编辑同机制需要的多个特化，不能修改特化的函数ABI。原始IR可解析是起点，避免重复全文件读取。
如果后端已消除了疑似冗余，解释观察并保持IR原样交接；如果存在可行改法，用小范围Edit落盘。
完成一项修改后调用submit桥，返回awaiting_ir_review即保持文件不变并结束本次调用。
语义、完整KGS和性能由宿主审查后运行，不能把文档、预检或建议当作收益结果。
性能搜索无总轮数、token、cost或总耗时限制；本次交接是审查与完整评测边界。
新诊断采用专业中文。工具命令：
中文采用正向具体陈述；禁用短语按不+是、不+等于、不能+证明拼接，decision桥会校验措辞。
/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/gqa-goal-20261121/controller-tools/ir_model.py --run /data/hanle/ascend-optimization/runs/gqa-goal-20261121/gqa-ir-qk-receive-prefetch --invocation ir-003 decision
/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/gqa-goal-20261121/controller-tools/ir_model.py --run /data/hanle/ascend-optimization/runs/gqa-goal-20261121/gqa-ir-qk-receive-prefetch --invocation ir-003 submit
/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/gqa-goal-20261121/controller-tools/ir_model.py --run /data/hanle/ascend-optimization/runs/gqa-goal-20261121/gqa-ir-qk-receive-prefetch --invocation ir-003 no-change

本次编辑编译器实际导出的HIVM，原输入已通过独立后端续编译与实测二进制关联。严格审查显式buffer地址、别名、跨迭代生存期、事件配对和跨核同步；保持函数ABI及未涉及的核心原样。所有修改由宿主逐项审查，再编译和真机评测。
先读取feedback.json中的实际正式与补测结果，保留源码和IR身份，针对反馈选择下一步。
读取codex-brief.json中的宿主证据分析与候选方向，核对实际IR后由你选择具体实现；记录Codex的分析贡献。
本调用限定检验下述机制，具体实现由你决定；若证据或语义审查阻止实施，登记原因后按no-change交接，其他机制留给独立调用：
先完成mandatory_read_order。保持原目标与ir-002草案同一机制：复用UB32768..65536，在P发送/flag1后、PVflag0等待前预取；Cube nextQK FIX后发ready flag4；首块prefillUB0、专用MTE2→V EVENT1，V→MTE2 EVENT0改为组级配对。以reviewed-prefetch-draft.mlir为你上一调用已实现草案，只用Edit复现到恢复后的ir/target.hivm.mlir，保持数学、ABI、事件证明和UB界限。禁止额外机制、额外缓冲、改数字或降低读取要求。

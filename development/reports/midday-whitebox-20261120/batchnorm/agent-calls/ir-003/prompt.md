本轮对已完整合格的最佳算子做单算子IR性能续接，保持frozen.py逐字节一致。
你负责实际IR候选与数值／编译修复；Codex负责证据和调度。只在当前工作区读取，禁止调用其他agent。
先读取context.json、profiling.json、hotspots.json、timeline.json、ir-inventory.json、frozen.py，
再根据kernel符号、循环、拷贝与布局检查ir/中的实际TTIR。input文件为当前最佳的真实编译结果；
存在lowered-inventory.json时读取其清单，按符号检查compiled-readonly/内实际ttadapter和bcmlir，
核对疑似冗余是否在后端已消除；这些实际低层产物保持冻结，只编辑清单允许的TTIR。
模拟器窗口的绝对时钟资质、观测范围和未报告泳道按材料解释。热点与等待／空闲均为核心输入。
目标为超过当前最佳；修复旧退步、单纯改编译选项、源码重写各有独立研究身份。
选择当前IR中具体且有执行成本的编译缺陷。判断哪些计算或布局应由编译器优化，并给出可测预测。
保留完整数学语义、特殊值、边界、别名及浮点精度要求；严禁根据测试数据拟合、缓存输出或调用外部计算。
先只修改diagnosis.json，填写source_sha256、selected_layer=single_operator_ir、evidence、hypothesis、
alternatives、change、expected、validation、layer_reason、simulator_incremental_judgment。
evidence每项有path、category、role、finding，category取paired_profiling/simulator/source_ir，
明确引用profiling、热点和流水的具体观察；记录你实际读取的材料。
然后调用下列decision桥，成功后才编辑IR。只用Edit修改diagnosis.json和ir-inventory.json列出的ir/*.ttir；
源文件、清单、context和证据保持冻结。没有Write工具，不调用shell写文件。每次调用选择一个可独立验证的机制，
可编辑同机制需要的多个特化，不能修改特化的函数ABI。原始IR可解析是起点，避免重复全文件读取。
如果后端已消除了疑似冗余，解释观察并保持IR原样交接；如果存在可行改法，用小范围Edit落盘。
完成一项修改后调用submit桥，返回awaiting_ir_review即保持文件不变并结束本次调用。
语义、完整KGS和性能由宿主审查后运行，不能把文档、预检或建议当作收益结果。
性能搜索无总轮数、token、cost或总耗时限制；本次交接是审查与完整评测边界。
新诊断采用专业中文。工具命令：
/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/midday-whitebox-20261120/tools/ir_model.py --run /data/hanle/ascend-optimization/runs/midday-whitebox-20261120/batchnorm --invocation ir-003 decision
/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/midday-whitebox-20261120/tools/ir_model.py --run /data/hanle/ascend-optimization/runs/midday-whitebox-20261120/batchnorm --invocation ir-003 submit
先读取feedback.json中的实际正式与补测结果，保留源码和IR身份，针对反馈选择下一步。
读取codex-brief.json中的宿主证据分析与候选方向，核对实际IR后由你选择具体实现；记录Codex的分析贡献。

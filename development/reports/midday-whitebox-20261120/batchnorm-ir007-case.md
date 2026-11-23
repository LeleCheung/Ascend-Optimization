# BatchNorm：统计分块探索与SSA修复

ir-007由模型登记后生成统计阶段8×512分块，原生MLIR解析未通过。随后ir-008改为闭式计数改写，超出本次SSA修复机制，单列为未验收提案；当前ir-009从逐字节恢复的保留版检验末轮预取。保留版继续为夜间8×256输出IR，目标39.261332μs。三项探索均尚无新增真机结果。

模型读取保留版热点、流水、profiling及实际BCMLIR后，选择统计阶段的8×128加载循环：目标空间维4096时32轮，每个tile4KiB；建议在空间维可被512整除时采用8×512、8轮，其他形状保留128回退，输出分块继续256。该方案由模型选择，宿主此前建议的尾步预取方向继续作为独立待验证机制。

新统计分块改变Welford各lane的样本分组以及最终浮点归约分组。数学目标、精度合同、running状态输出及Triton源码保持固定，后续按完整27项及双方36项验证数值资格；报告逐项记录分组变化。UB峰值、同步与最后一次重复预取还需实际低层产物检查。

宿主审查发现：新统计else分支内部定义的均值广播在输出回退分支被外部引用，违反SSA作用域。原生解析器在候选第344行报告`use of undeclared SSA value name`，与静态诊断一致，设备未参与该检查。模型负责修复SSA支配关系，Codex保存诊断和复验。

模型归因中的几处范围已纠正：MTE2活跃占比0.877低于Vector的0.906，重叠泳道占比仅用于定位；两个不同PC／数据上下文的搬运样本支持开销假设，固定390周期的线性拟合仍待验证。ir-006的ABBA失败约束该输出加宽候选，尾步排空与统计加宽继续各自实验。夜间已有统计加宽CPU失败，UB需求10485760bits、可用1572864bits；该原始记录已交给修复模型，不将配置上限或静态几个memref的总量当作实际峰值。

读取审查确认热点、流水、profiling、TTIR和实际BCMLIR在决策及IR修改前返回。ttadapter与二进制仅确认清单／行数，模型最终声称已阅读其实际内容的范围超出日志，缺口单独保留。新修复调用继续要求读取真实低层文件，登记后再编辑。

材料：[修改前决策](batchnorm/agent-calls/ir-007/before-edit-decision.json)、[实际diff](batchnorm/agent-calls/ir-007/actual-ir.diff)、[宿主审查](batchnorm/agent-calls/ir-007/ir-review.json)、[原生解析结果](batchnorm/agent-calls/ir-007/syntax-review.json)、[解析日志](batchnorm/agent-calls/ir-007/native-parse-review.log)、[读取范围](batchnorm/agent-calls/ir-007/evidence-read-review.json)。原生解析、读取审查和完整归档随修复阶段继续落盘，原始失败保留。

ir-007的215份最终审查归档已本地SHA校验通过，见[归档清单](exports/batchnorm-ir007-syntax-reviewed.tar.gz.receipt.json)。

ir-008实际删除统计512分支并消除计数循环携带，依据循环上界重建最终计数。宿主保留该独立提案，当前修复调用的机制变更交回后续新调用处理；该提案尚未进行设备评测。原计数跨外层batch分块携带，模型声称“每个batch块重置”的范围有误：抽象两次外循环、每次32步时原计数64，直接采用内层上界得到32。选中特化的真实batch域需由源码启动规则单独绑定，保留这个范围后再判断等价性。profiling的candidate／reference表、历史ir-005完整负结果和时钟资质解释也已逐项更正，见[ir-008宿主审查](batchnorm/agent-calls/ir-008/ir-review.json)。

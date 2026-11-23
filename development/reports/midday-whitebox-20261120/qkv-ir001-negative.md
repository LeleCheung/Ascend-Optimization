# QKV：K／V提前写出的IR完整合格，目标耗时增加

当前状态：完整21/21、双方各48/48补测通过；新IR目标63.651644μs，原IR本阶段复测63.257576μs。继续保留原IR；修改后双采集、正式／重放二进制身份与机制审查已完成。原预测、数值合格结果和性能负结果均保存。

固定输入为FP16 `[8,512,3072]`、16 heads，`transform_bias_rescale_qkv-timing-2`。Triton源码SHA为`0af5d533afea8f4be88e1fd6e0fc4a6ba76fae9a8b1478f7c4b93247fe7ec070`。修改前已对原IR做新完整复测、成对profiling和模拟器重放，并核对保留二进制一致。模型先核对padding方向：目标特化没有masked load，相关发现作为诊断结果保留。

模型读取热点、流水及实际低层产物后，登记“先K计算与写出、再V计算与写出、最后Q”的调度方案，尝试改善搬运与计算的重叠。所有输入和bias读取仍位于输出store之前；冻结源码通过三个独立`torch.empty`分配Q／K／V输出。原IR操作行多重集合保持一致，Q的FP32加法与缩放、单次FP16截断，以及K／V原FP16路径均保留。宿主审查与CPU编译通过，实际低层materialize顺序由QKV变为KVQ，copy6、alloc6和loop1保持。

| 指标 | 原IR本阶段复测 | 调度IR |
| --- | --- | --- |
| 目标shape耗时 | 63.257576μs | 63.651644μs |
| 同轮配对CANN参考 | 91.527815μs | 92.421866μs |
| 同轮配对目标／CANN | 1.446907× | 1.451995× |

目标候选耗时增加约0.623%，继续采用原IR。两次CANN参考也发生变化；固定目标耗时与各自配对加速比同时保存，保留决定依据同合同候选耗时及历史最好记录。此前源码最好目标62.837203μs及原IR测量62.860936μs继续冻结。

二进制核对有独立范围说明：原IR完整评测回执覆盖22个AST，新候选完整评测加两组补测的回执覆盖37个AST。双方共有22个中，目标1个发生登记变更，其余21个逐字节一致；新增15个来自补测回执，当前阶段缺少对应原IR补测回执，分别保存。严格全集比较工具的集合相等检查失败记录保留；此次记录只陈述共有AST的核对结果。原IR正式22条回执及候选正式22条、补测两进程各30条的文件身份均保存于范围审查。

修改后profiling同为512个程序、1650MHz，每程序MTE2时间1.809911→1.678384μs下降，MTE3时间1.154888→1.196779μs及Scalar时间1.009835→1.027134μs增加，AIV时间3.072685→3.126787μs。独立profiling任务41.140823→41.860836μs；该采集与正式清L2计时分别解释。模型预计的重叠改善未转化为更低的完整调用耗时，局部流水指标的取舍同时保存。

新模拟器与正式二进制一致，24/24个PC时钟匹配；原IR22/22匹配。前后部分窗口110.488～118.879及113.526～123.161分别归一化，动态总量各自解释。双方msprof包装返回0、子进程超时退出9，应用产物清单未完成，直接PC源码映射保留待补。

模型调用日志确认context、profiling、热点、流水及低层清单的读取工具返回发生在登记决策与首次IR修改之前。该核对记录片段读取与执行顺序；方法贡献仍结合Codex分析、成对profiling及模拟器共同解释，数值合格与真实性能变化分别报告。

材料：[修改前决策](qkv/agent-calls/ir-001/before-edit-decision.json)、[读取顺序核对](qkv/agent-calls/ir-001/evidence-read-review.json)、[IR diff](qkv/agent-calls/ir-001/actual-ir.diff)、[语义审查](qkv/agent-calls/ir-001/ir-review.json)、[实际低层机制](qkv/agent-calls/ir-001/lowering-mechanism.json)、[完整结果](qkv/evaluations/ir-001/result.json)、[保留原IR决定](qkv/retention-ir-001.json)、[二进制范围审查](qkv/ir001-binary-comparison-scope.json)、[后profiling](qkv/post-mechanism-ir001-profiling.json)、[后重放审查](qkv/post-mechanism-ir001.json)。正式快照353条及最终381条归档逐文件SHA校验通过。

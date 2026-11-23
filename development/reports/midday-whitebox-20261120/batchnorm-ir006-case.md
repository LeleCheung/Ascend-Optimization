# BatchNorm：带守卫的8×1024输出分块

当前完整27/27、双方各36/36及后双分析通过，ABBA稳定性未通过，继续保留8×256。正式目标39.458217→33.539900μs，较本阶段下降14.9989%，较历史39.261332μs下降14.5727%；ABBA原版36.487154／36.536723μs、候选36.429490／36.885589μs，未满足两段候选均快于原版的条件。固定FP32 NCHW `[8,64,64,64]`、`timing-7`，源码继续为`b8d4d4a2…`；33.540μs保留为探索成绩，主图继续采用已验收8×256。

模型从逐字节恢复的保留IR继续，仅为纯输出阶段新增8×1024分支。spatial可被1024整除时选择新分支，其余分别采用原8×256或8×128回退。统计／Welford、归约、running状态、weight与bias处理保持原样，固定源码用于隔离输出分块变换。

宿主逐字节确认统计前缀、原256分支与128回退内容相同。新1024分支在分块宽度与SSA名称归一化后，与原256分支54行完全一致，逐元素FP32运算顺序与地址stride保持固定。8193个128倍数的分支检查、257种tile数量及33153次预取／输出迭代检查通过。

实际BCMLIR新增两块8×1024 FP32输入memref，包含初始读取、钳位下一块预取及输出写回；原回退路径保留。整函数静态copy6→8、分配6→8、循环6→8，计数包含全部备选分支。最终二进制8416→9688字节，SHA`41483d74…`→`ddbf7940…`。编译元数据的221184字节为配置上限，实际峰值UB与设备资源资格继续由后端产物及真机检查。

目标spatial4096的输出轮次预计16→4。新分支同时保留末轮钳位预取，IR层逻辑输出读取为“初始一块＋每轮下一块”：8×256版17块、136KiB／程序，8×1024版5块、160KiB／程序。较大分块减少循环与命令数量，同时放大末块重复读取；最终指令是否消除该读取，以及缓存、同步和物理流量的变化由后重放核对。

完整合同候选耗时原／改几何比较为1.047395×；本阶段原始配对geomean1.832020→1.905310×，使用新候选逐case CANN共同锚点重算为1.819094→1.905310×。固定shape的0.838308×CANN与全合同geomean分别保存。

修改后独立profiling覆盖64个程序、1650MHz。每程序MTE2指令121→85、MTE3指令21→9；GM→UB读取268→292KiB，增加24KiB，与末轮重复预取的逻辑增量一致。每程序Vector时间9.515303→8.906373μs、MTE3时间2.902784→1.766610μs，Scalar等待1.502008→1.956771μs；等待的新增成本与完整调用收益分别记录。正式二进制绑定及模拟器指令核对继续进行。

后重放与正式目标二进制一致，31/31个独立PC时钟核对通过；对应输出MTE3搬运调用32→8，两份窗口均可见2次END。新输出初始读取2次、循环预取8次，末轮重复读取仍存在。完成清单写入，kernel同步已返回；wrapper0／子进程139及部分单核窗口继续保留，直接PC源码映射范围单列。预测的轮次减少和读取增量已确认，性能波动原因继续开放调查。

此前8×512的三轮稳定性未通过／通过／未通过，计数标量化的完整合格负结果也已交回模型。本轮实际BCMLIR、ttadapter、热点、流水、成对profiling与反馈均在decision及编辑前读取。425份完整证据逐文件SHA校验通过，[保留决定](batchnorm/retention-ir006.json)、[后机制审查](batchnorm/post-mechanism-ir006.json)、[ABBA](batchnorm/stability/ir-006/summary.json)及[最终归档](exports/batchnorm-ir006-stability-negative-complete.tar.gz.receipt.json)已保存。

后续ir-007模型选择统计加宽，原生解析未通过；ir-008提出其他计数机制，保留为独立待验收提案。当前ir-009再次从逐字节恢复的8×256 IR继续，固定现有tile与Welford算术，检验最后一次预取的退出后未使用状态及尾步剥离／排空。模型负责具体IR修改，[宿主假设说明](../midday-batchnorm-tail-brief-20261120.json)记录预期读取与控制流、同步成本，[后续探索](batchnorm-ir007-case.md)保留全部分支。当前尚无ir-009收益结论。

材料：[修改前决策](batchnorm/agent-calls/ir-006/before-edit-decision.json)、[实际IR diff](batchnorm/agent-calls/ir-006/actual-ir.diff)、[语义审查](batchnorm/agent-calls/ir-006/ir-review.json)、[读取时序](batchnorm/agent-calls/ir-006/evidence-read-review.json)、[实际CPU编译](batchnorm/agent-calls/ir-006/lowering-review.json)、[低层机制](batchnorm/agent-calls/ir-006/lowering-mechanism.json)、[完整结果](batchnorm/evaluations/ir-006/result.json)、[二进制核对](batchnorm/ir006-binary-comparison.json)、[待验收收益](batchnorm/provisional-gain-ir-006.json)、[正式快照](exports/batchnorm-ir006-formal-pending.tar.gz.receipt.json)。366份正式快照及早期54份CPU阶段证据本地SHA校验通过；后续完整归档单独新增。

# BatchNorm：移除最后一次未使用预取

完整27/27、双方各36/36及修改后双分析已完成，目标39.397766μs，继续保留固定源码`b8d4d4a2…`、夜间8×256输出IR的历史39.261332μs最佳。本阶段原IR复测39.458217μs，目标耗时略降0.153%；较历史最佳慢0.348%，按预登记目标筛选未运行ABBA。此前8×1024正式33.539900μs的候选未通过稳定性，继续作为探索记录。

模拟器与实际低层产物显示，统计、宽输出及窄输出回退循环均在最后一次迭代按钳位地址重复加载末块。该值只经循环结果传出，随后未被使用。宿主据此提出检查尾步排空；模型在修改前实际读取热点、流水、profiling、TTIR、ttadapter和BCMLIR，再登记并修改IR。

本轮保持统计8×128、输出8×256／8×128、Welford递推、计数跨batch携带及归约分组。三个内层循环执行前`n-1`次后，在同一区域单独执行最后一次算术／写回，移除这一步的预取。首次加载、有效tile消费顺序及每个输出lane的一次写回保持一致。

宿主检查257种正tile数量、17种batch组索引、563601次逻辑迭代；统计尾步的浮点与计数表达式经SSA归一化后与原循环体一致。源码启动规则将本特化限定为BLOCK_M8、BLOCK_N128、EVEN，空间维为正且按128整除；单tile时短循环为空，尾步消费初始块。其他AST保持原实现。后端数值行为继续由完整合同与补测验收。

实际BCMLIR保留三条缩短的循环和无输入copy的尾部。静态copy6、memref分配6、循环6、if3保持原数量；fill16→24，包含新增尾步算术的静态副本。二进制8416→9824字节，原版SHA`41483d74…`与正式保留版一致，新版`222759bd…`。CPU编译全程未初始化NPU；配置上限221184字节及`required_ub_bits=0`各按原含义保存，实际峰值UB继续取证。

目标统计预取33→32块，宽输出17→16块的预测已兑现。独立profiling双方均64程序、1650MHz，每程序GM→UB读取268→256KiB；包含标量读取的主存总计268.875→256.875KiB。两个可见trace均含两次END：统计初载＋循环copy由2＋64降为2＋62，输出读取2＋32降为2＋30，输出写回32次改为循环30次＋尾步2次。向量BAR仍484次、标量DIV仍82次，说明这次修改保留了主要算术与屏障次数。

独立profiling每程序MTE2时间12.758428→12.656042μs，Vector9.515303→9.461515μs；Scalar执行4.645445→5.250388μs、Scalar等待1.502008→1.695076μs。各自归一化窗口中，MTE2活跃0.877423→0.862329，下降0.015094，低于登记的至少0.02预测；Vector活跃0.906128→0.902298。读取减少与轻微局部改善均保留，整调用计时仍需满足历史最佳与稳定性条件。

后采集源码、IR、case及正式二进制身份绑定通过。28个AST中仅登记目标二进制变化，其余27个一致。时钟单位独立核对35/36个PC在5%范围内匹配；剩余`0x10d1143c` MTE2 `MOV_SPR_XN`有62条trace事件，区间并集0.124μs、CSV0.150μs，比例0.826667，该局部口径偏差继续保留。完成清单存在，wrapper0／child139、单核部分窗口与直接源码映射范围分别记录。宿主保留原最佳，随后逐字节恢复8×256／统计8×128，ir-010继续调查统计分块。

材料：[修改前决策](batchnorm/agent-calls/ir-009/before-edit-decision.json)、[读取时序](batchnorm/agent-calls/ir-009/evidence-read-review.json)、[实际diff](batchnorm/agent-calls/ir-009/actual-ir.diff)、[语义与调度审查](batchnorm/agent-calls/ir-009/semantic-schedule-review.json)、[宿主审查](batchnorm/agent-calls/ir-009/ir-review.json)、[实际CPU编译](batchnorm/agent-calls/ir-009/lowering-review.json)、[低层机制](batchnorm/agent-calls/ir-009/lowering-mechanism.json)、[CPU阶段归档](exports/batchnorm-ir009-cpu-reviewed.tar.gz.receipt.json)。108份编译阶段证据已本地逐文件SHA校验；该快照独立于运行中的设备评测。

最终材料：[完整评测](batchnorm/evaluations/ir-009/result.json)、[双方补测](batchnorm/evaluations/ir-009/supplementary-summary.json)、[二进制集合](batchnorm/ir009-binary-comparison.json)、[后重放身份](batchnorm/observation-binding-ir-009.json)、[后实测明细](batchnorm/post-mechanism-ir009-facts.json)、[预测核对](batchnorm/post-mechanism-ir009.json)、[保留决定](batchnorm/retention-ir009.json)、[417份完整归档](exports/batchnorm-ir009-no-target-gain-complete.tar.gz.receipt.json)。完整归档及逐文件SHA校验通过，编译阶段快照继续独立保存。

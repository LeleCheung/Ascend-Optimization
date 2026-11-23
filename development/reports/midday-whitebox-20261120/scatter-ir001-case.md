# scatter_add：固定源码，前两轮统一索引探测改为标量读取

当前状态：完整合同18/18、双方各21/21补测、分段3/3、ABBA及修改后双分析审查通过，已保留新IR。真实中间输入、正式二进制及82/82个PC时钟核对通过，超时退出和部分窗口继续保留。

固定输入为FP32 `[2048,2048]`、dim=-1，`scatter_add-timing-8`。Triton源码SHA为`a27f40a227126c5ffbe4e6bc1fd6e93cbb7c06d9a417091e1351785b90f0429e`，从前夜已保留的二分尾步折叠IR继续。父IR `2590a86a…`，候选IR `267c2e07…`，保留编译器原IR身份。

修改前重放绑定实际真机中间张量并隔离`_scatter_add_sort_out_kernel`。模拟器显示Vector有效工作占比高，VGATHER累计指令开销靠前，Scalar有较多等待。Codex提出检查二分前几轮的索引均匀性及新增标量读取的代价。模型实际读取成对profiling、热点、流水、TTIR及BCMLIR，选择将前两轮通用gather替换为读取固定keys[1024]、keys[512]、keys[1536]，再广播／选择。原尾步折叠及后续二分、写回保留。

宿主核对全部2049种抽象插入秩：首轮索引均为1024，次轮依比较选择512或1536，与原二分索引和比较结果一致。三个读取地址落在2048元素的不可变同行内。实际BCMLIR中generic gather13→11，标量load1→4。正式全合同27个AST中仅登记目标二进制变化，其余26个逐字节一致。

| 测量 | 保留IR | 新IR |
| --- | --- | --- |
| 本阶段完整调用目标耗时 | 1263.108486μs | 1243.040500μs |
| 同轮配对目标／CANN | 2.720247× | 2.764905× |
| ABBA完整调用区块中位数 | 1262.201844μs | 1242.655333μs |
| 独立profiling的sort_out task | 489.669800μs | 467.789368μs |
| sort_out每程序Vector时间 | 7.865975μs | 7.333878μs |
| sort_out每程序Scalar等待 | 7.953220μs | 7.141459μs |
| sort_out每程序Scalar执行 | 1.037420μs | 1.404894μs |
| sort_out每程序MTE2时间 | 0.684496μs | 0.998952μs |

正式完整调用耗时下降1.5888%。ABBA的两个候选区块均低于两个原版区块。实际profiling呈现明确取舍：减少Vector工作与Scalar等待，增加标量执行和MTE2成本；完整调用仍获得净收益。

全部性能case候选耗时的几何比较为原／改1.015687×，原始配对geomean2.252197／2.267683。采用新候选逐case CANN共同锚点后为2.232659／2.267683。与前夜保留版本采用同一新锚点比较为2.260980／2.267683。各次参考漂移、未修改特化的测量变化与真实IR修改分别保存。

修改后使用真机保存前序kernel产生的中间输入，隔离目标sort_out回放。正式二进制、输入张量SHA及82/82个PC时钟核对通过。前后窗口分别为122.682～152.997与139.872～173.435原始时钟位置；可见gather指令PC数13→11，与实际BCMLIR的generic gather变化对应。动态覆盖量不同，原始次数和占比各自解释；保留超时状态9及直接源码映射范围，完整调用性能仍由正式结果和ABBA验证。

本轮固定源码检验统一索引传播的局部编译变换。候选由CC／DeepSeek实现，Codex的方向建议、2049秩检查、基础设施与验收贡献分别记录。

材料：[修改前决策](scatter/agent-calls/ir-001/before-edit-decision.json)、[IR diff](scatter/agent-calls/ir-001/actual-ir.diff)、[宿主审查](scatter/agent-calls/ir-001/ir-review.json)、[低层机制](scatter/agent-calls/ir-001/lowering-mechanism.json)、[完整结果](scatter/evaluations/ir-001/result.json)、[二进制核对](scatter/ir001-binary-comparison.json)、[ABBA](scatter/stability/ir-001/summary.json)、[后profiling](scatter/post-mechanism-ir001-profiling.json)。

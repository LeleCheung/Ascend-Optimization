# scatter_add：折叠二分查找尾步，保留必要比较

本例是持续模拟器重放中的单算子IR修改：继续读取当前版本重放证据，模型固定Triton源码修改IR，编译新二进制，完成真机验收后再次模拟器重放。修改前决策与修改后观测分别保留，前后窗口范围按下文解释。

固定FP32 `[2048,2048]`、dim=-1、`timing-8`。原IR1.281927461ms，修改IR1.262722303ms、2.721707×CANN，耗时下降1.50%。完整18/18、双方各21/21及分段独立性3/3通过。ABBA及修改后双分析已完成，第三个不同算子的IR收益完成验收。

修改前先修复CANN混合核profiling分类，再将真机前置排序产生的输入保存并单独回放sort_out。该目标二进制与完整原IR评测一致，部分窗口152/152个非Scalar／非同步PC时钟核对通过。Vector有效工作占比0.887621，Scalar仅等待0.822343，MTE2可观测空闲0.946432；VGATHER在67次调用中累计25.33μs，为最高累计非同步Vector指令。指令时间存在重叠，模拟器应用超时、直接PC源码映射缺失及部分边界均保留。

Codex独立分析并验证全部2049种可能插入秩：N=2048的前11次比较后，仅秩0／1仍共享区间[0,1]，其余秩均收敛。最后一步所需信息可由首个key与查询值的比较保留。模型读取正确kernel的profiling、热点、流水、TTIR及实际BCMLIR后登记机制，随后将末步的通用动态gather改为首key标量读取、未收敛条件判断及一次受控增量。宿主再次对模型实际增量公式独立核验全部2049种秩，包括2048边界。

实际BCMLIR的generic gather从14降为13，保留新增首key标量`memref.load`；源代码、ABI、网格、编译选项、前置排序和前缀计算保持固定。重复键、缺席段、末尾钳制及prefix中的特殊值沿原路径处理。完整27个AST中仅目标sort_out二进制变化，其他26个逐一相同；同合同逐case候选耗时几何改善1.012689倍。修改后独立paired任务中，sort_out为489.669800μs，修改前为508.550171μs；该数据与完整KGS结果分别保存。

| 身份 | SHA256 |
| --- | --- |
| Triton源码 | `a27f40a227126c5ffbe4e6bc1fd6e93cbb7c06d9a417091e1351785b90f0429e` |
| 原TTIR | `19efb34a0d4e2902611b8ed637f489c3a66f809d54c4fc67e84a2443bfcffb30` |
| 修改TTIR | `2590a86abfb953baa06f9079f9b45a8e88a9a059f521dff76b6df7cf5f760164` |
| 原目标二进制 | `582887767b19053008abec97e64d01c58cdfecfa6462c8130c03239134d0fcd5` |
| 修改目标二进制 | `06dd8d1949fd7a151b789ed860304f7726127c663dd2bad33364975c0fe5c01e` |

| ABBA段 | 实现 | 三次测量中位数μs |
| --- | --- | ---: |
| A1 | 原IR | 1282.364039 |
| B1 | 修改IR | 1262.522169 |
| B2 | 修改IR | 1263.117714 |
| A2 | 原IR | 1282.141145 |

两段修改版均快于两段原版。修改后每程序Vector时间8.227873→7.865975μs，Scalar等待8.396178→7.953220μs，支持减少向量工作量的机制。后模拟器目标二进制与正式结果一致，79/79个非Scalar／非同步PC时钟核对通过。原／改模拟器分别为54.15／30.32μs的部分核心窗口，归一化占比各自解释，指令累计值按各自窗口保留，直接PC源码映射仍缺失。

贡献分别记录：Codex提供秩空间证明、范围审查及方向；模拟器提供对应kernel的向量／gather热点和等待占用；模型核对后端未消除的实际gather并编写IR；完整设备与补测验证结果。前置排序kernel证据驱动的旧`ir-001`提案保留原归因范围，本轮采用新的输入快照与修改前登记。

474份导出文件完成压缩包及逐文件SHA校验，含模型实际读取的修改前输入、决策、源码／IR、真实编译产物、完整结果、补测、ABBA和前后观察摘要。主图全合同2.250×CANN，固定输入2.722×CANN，26图与四图的原生起点／连续／单调检查通过，新截图已检查；同合同复测分别显示或保留详情。原始现场位于远端`runs/ir-performance-20261119/scatter/`，大型原始trace和回放张量保留在收据路径。

入口：[修改前决策](scatter/agent-calls/ir-002/before-edit-decision.json)、[语义与2049秩审查](scatter/agent-calls/ir-002/ir-review.json)、[实际低层变化](scatter/agent-calls/ir-002/lowering-mechanism.json)、[完整结果](scatter/evaluations/ir-002/result.json)、[稳定性](scatter/stability/ir-002/summary.json)、[后机制记录](scatter/post-mechanism-ir002.json)、[验收选择](scatter/selection.json)、[474份收据](scatter-accepted-ir002-receipt.json)。采集方法见[恢复记录](scatter-collection-recovery.md)与[后续阶段采集手册](../../runbooks/later-stage-simulator.md)。

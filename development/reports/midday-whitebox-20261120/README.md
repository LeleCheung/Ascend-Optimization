# 午间白盒多层次优化证据

当前续接：[scatter第三次二分探测](scatter-third-probe-case.md)通过模型交接、2049种抽象插入秩及实际CPU编译审查，generic gather11→10；固定完整18项及既有补测评测运行中，继续保留1243.040500μs。完整合格后以真实中间输入隔离重放sort_out，ABBA按具体计时疑点选用。

最新验收：BatchNorm ir-011已保留，固定FP32 [8,64,64,64]目标39.261332→37.200210μs，下降5.25%，0.754509×CANN；完整geomean1.876942×CANN，27/27、双方36/36通过。冻结25项诊断原19→新23通过，新增回归0，剩余两项窄分支继承边界明示。后重放BAR484→304、DIV82→50；时钟40/41个PC及约5%边界局部差异、部分窗口和child139保留。两张BatchNorm曲线已更新，所有trace隐藏源码复测；详见[修复与新收益](batchnorm-ir011-case.md)。

以下按阶段保留此前状态。

续接：[ir-011公共原点与残差均值修复](batchnorm-ir011-case.md)已通过模型交接、结构审查及CPU编译，完整27项、既有双方36项和冻结25项数值诊断正在执行。当前合格保留值39.261332μs；原／改低层与真实性能分开记录。

最新：BatchNorm ir-010目标36.135583μs，完整27/27、双方36/36及ABBA通过；冻结25项分组数值诊断发现一项新增回归，保留39.261332μs。后重放38/38个PC时钟匹配，完整证据已导回，[案例与保留决定](batchnorm-ir010-case.md)已更新。所有当前主图隐藏源码复测，常规验收固定测试，ABBA按疑点启用，SHA自动记录。

状态入口见[午间报告](../midday-whitebox-20261120.md)，选层条件见[何时修改IR](../when-to-edit-ir-20261120.md)。新阶段从上一阶段已保留的源码＋IR组合继续，模拟器指导的IR修改采用橙色方法箭头和绿色三角形。

09:04 UTC当前：[ir-010统计8×256及标量合流](batchnorm-ir010-case.md)已通过语义、读取时序和实际CPU编译，完整评测与收益筛选后ABBA等待继续运行，双方各25项独立FP64数值诊断已预登记。输出分块固定，统计分组改变，GM→UB预期增加4KiB／程序，收益尚待真机验收。当前保留39.261332μs。

08:48 UTC当前：[ir-009尾步剥离](batchnorm-ir009-case.md)完整27/27、双方36/36通过，39.397766μs，继续保留39.261332μs最佳。后重放确认读取268→256KiB／程序，向量屏障484次保持原数，35/36个PC时钟匹配及局部偏差分别记录；417份完整证据已校验。已逐字节恢复最佳，模型ir-010 PID1509314研究统计分块及标量结果合流，尚无该轮收益结论。以下时间点保留阶段状态。

08:04 UTC当前：上采样、scatter_add及QKV午间新增IR固定shape收益均完整验收；BatchNorm保持夜间8×256保留版，新调用ir-007继续调查尾步预取。QKV[保留决定](qkv/selection.json)及472份完整归档已校验，历史62.837→62.050μs下降1.25%，全合同共同参考下基本持平。[BatchNorm ir-006](batchnorm-ir006-case.md)正式33.540μs、完整及补测通过，ABBA未通过；425份完整归档已校验，原始波动和31/31个PC后重放保留。八图及23页PPT已更新。

08:20 UTC续接：ir-007统计加宽原生解析未通过，215份最终审查归档已校验；ir-008机制变更单列提案。当前ir-009从逐字节恢复的8×256继续检验末轮重复预取，暂未产生新增收益。见[后续探索与读取范围](batchnorm-ir007-case.md)。

案例记录：[上采样删除填充值](upsample-ir001-case.md)、[scatter前两轮索引探测](scatter-ir001-case.md)。两例分别列出机制、收益、增加的成本、预测核对及完成的后重放范围。

[BatchNorm输出8×512](batchnorm-ir003-case.md)完整27/27及双方各36/36通过，目标36.647637μs；三轮稳定性未通过／通过／未通过，保持8×256原最佳。后双分析与机制核对完成，451条最终归档SHA校验通过；28个AST中仅目标1个二进制变化，其余27个一致。

[QKV写出调度IR](qkv-ir001-negative.md)完整21/21、双方各48/48，目标63.257576→63.651644μs，保留原IR；后双分析及24/24个PC时钟核对完成，381条最终归档SHA校验通过，额外补测特化的核对范围单独保存。

[BatchNorm广播外提](batchnorm-ir004-cpu-noop.md)语义审查和CPU编译通过，BCMLIR中循环内fill外提，最终二进制逐字节一致，199条归档SHA校验通过。[ir-005计数标量化](batchnorm-ir005-case.md)从恢复的8×256 IR出发，依据64次S32计数更新的指令证据修改均匀计数。语义与CPU编译通过，完整27/27、双方各36/36，目标40.024682μs，慢于保留版，保持原IR并继续后双分析；28个AST中仅目标1个变化，其余27个一致。53份CPU阶段证据及6份正式结果已导回本地校验。

BatchNorm ir-005后双分析已完成，重放确认64次S32向量加法和2次向量转换移除，30/30个PC时钟匹配，384份完整证据逐文件SHA校验通过。[ir-006带守卫8×1024输出分支](batchnorm-ir006-case.md)完整27/27、双方各36/36及后双分析通过，31/31个PC时钟匹配；ABBA未通过，保持8×256。输出搬运32→8、每程序读取268→292KiB及稳定性波动一起保存。

[QKV新ir-002](qkv-ir002-case.md)采用双头联合读取、UB内换序及逐头写回，完整21/21、双方各48/48、ABBA及后双分析通过，已保留并接入主图。目标63.257576→62.050287μs，较本阶段下降1.91%、较历史最佳下降1.25%；全合同共同参考下历史1.672449→新1.672510×，基本持平。共有22个AST中目标1个变化、其余21个一致，额外15个与上一合格候选的补测二进制一致，比较来源分别记录。472份完整证据及427份早期正式快照SHA校验通过；bias读取加倍、向量与写命令增加及DMA解释范围保留。

当前scatter和上采样的新IR均已完整验收并接入主图。scatter完整18/18、双方各21/21及分段3/3通过，目标1263.108486→1243.040500μs；上采样完整24/24、双方各30/30通过，目标159.228890→146.503486μs。两项ABBA、修改后profiling和模拟器机制审查通过，新窗口分别82/82及30/30个PC时钟匹配；部分窗口、超时退出、PAD预测偏差与新增标量成本分别保存。BatchNorm和QKV的新候选状态见上文，完整记录继续保留。

| 材料 | scatter | 上采样 |
| --- | --- | --- |
| 修改前决策 | [decision](scatter/agent-calls/ir-001/before-edit-decision.json) | [decision](upsample/agent-calls/ir-001/before-edit-decision.json) |
| 实际IR改动 | [diff](scatter/agent-calls/ir-001/actual-ir.diff) | [diff](upsample/agent-calls/ir-001/actual-ir.diff) |
| 语义审查与证明 | [review](scatter/agent-calls/ir-001/ir-review.json) | [review](upsample/agent-calls/ir-001/ir-review.json) |
| 实际低层产物 | [lowering](scatter/agent-calls/ir-001/lowering-review.json) | [lowering](upsample/agent-calls/ir-001/lowering-review.json) |
| 机制核对 | [mechanism](scatter/agent-calls/ir-001/lowering-mechanism.json) | [mechanism](upsample/agent-calls/ir-001/lowering-mechanism.json) |
| 完整结果 | [18/18](scatter/evaluations/ir-001/result.json) | [24/24](upsample/evaluations/ir-001/result.json) |
| 全特化二进制核对 | [其余26个一致](scatter/ir001-binary-comparison.json) | [其余24个一致](upsample/ir001-binary-comparison.json) |
| 重放身份核对 | [binary binding](scatter/observation-binding-ir-001.json) | [binary binding](upsample/observation-binding-ir-001.json) |
| 后机制审查 | [review](scatter/post-mechanism-ir001.json) | [review](upsample/post-mechanism-ir001.json) |
| 最佳保留决定 | [selection](scatter/selection.json) | [selection](upsample/selection.json) |

两份正式评测早期快照共606个归档条目保存；两份最终验收归档共707个条目逐文件SHA校验通过，压缩包及逐文件清单位于`exports/`。BatchNorm两级预取完整合格负结果另有319个条目的正式快照，保留下一机制的起点与完整反馈。

两项ABBA的候选两个区块均低于原版两个区块。scatter中位数1262.201844→1242.655333μs；上采样156.742455→145.619453μs。四区块各三次测量的原始结果已落盘：[scatter](scatter/stability/ir-001/summary.json)、[上采样](upsample/stability/ir-001/summary.json)，各区块结果SHA与汇总一致。新节点已接入26项对应主图和组会八图。

模型原始选层解释与宿主核对分别保留。固定源码的IR实验用于检验局部编译变换；源码也可表达同类意图，后续方法迁移时比较源码表达与自动编译优化。全部正确性、性能和重放继续采用原合同与身份约束。

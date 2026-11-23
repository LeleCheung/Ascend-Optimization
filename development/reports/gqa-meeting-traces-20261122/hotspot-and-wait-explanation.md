# GQA 节点11与22：热点、知识来源和等待调度核查

2026-11-22。补充解释；用户要求原样保存的 workflow-three-examples-verbatim.md 保持不变。

## 节点11：QK交接缩窄

case4输入输出为BF16；QK累加结果为FP32。本次修改将raw QK交接改为FP16，再恢复FP32做后续向量运算，并非改变输入输出dtype。

修改前QK为128×256×4B=128KiB/块，PV为128×128×4B=64KiB/块。完整逻辑L0C搬出量分别64MiB、32MiB。窗口中83.49μs/19次与29.22μs/18次不能作为等字节搬运效率比较，也不能仅据此断言异常；其作用是把优先研究对象定位到QK交接区域。

FIX_L0C_TO_DST是指令名称，FIXP是执行该类Cube结果搬出、格式/类型转换的流水名称。完整source_stack把PC 0x10d11568关联到candidate.py:217的QK dot表达式；AST进一步确定函数与循环区域。PC到源码行来自采集trace的args.code源位置栈，AST本身不生成PC映射。

修改前决策结合模拟器热点、profiling搬运量和实际TTIR/BCMLIR：rawQK为FP32，且scale/exp在搬出后执行。尝试dot后FP16舍入并恢复FP32，让编译器有机会把缩窄融合进FIXP，预期QK交接减半。固定输入范围[-1,1]且D=128时raw dot绝对值不超过128，FP16范围足够；FP16较BF16有更多有效位。舍入影响仍由原四项checker实测裁决。

来源：`../gqa-goal-20261121/gqa/host-analysis-attempt-033.json` 与 `../gqa-goal-20261121/gqa/trace/decisions/7b08bfb0b4ec4025a3e077659807ffa4.json`。宿主明确考虑过历史GQA attempt-028的raw-score FP16实验；该实验采用额外全局阶段，性能为负结果。本次保持单kernel组织。已核查记录未证明方法直接迁移自26项中某个FlagGems算子，汇报不补造这一历史来源。

## 节点22：独立分母归约跨越等待

分母为d=row_sum(P)，倒数为1/d，不是对P逐元素求倒数。PV只在最终归一化时参与：out=PV*(1/d)。

当前实现P在Vector UB中生成，经MTE3写入GM交接区，通知Cube侧读取并计算PV；PV从Cube L0C经FIX搬出到GM，再由Vector侧读取到UB。逻辑“发送P/接收PV”包含这些搬运和同步。

实际晚层IR含`hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0`，以及MTE2→Vector等`wait_flag`。等待在控制/同步路径上呈现，Vector计算流水缺少可执行的归约；不能称为整条Vector流水始终在执行一条wait指令。

IR004将原16行sum/VREC链移到P发送和通知之后、nextQK及PV等待之前，保留事件与浮点操作。实际父版是IR003，已经包含原生倒数和QK提前接收等修改。

后重放同一Vector核显示sum为41.380–41.931μs，VREC为41.932–41.945μs，与P发送41.376–43.306μs及nextQK就绪等待41.396–44.998μs重叠；PV接收为46.414–46.738μs。具体已观测重叠是发送和nextQK等待区间，讲解可统称PV返回前的交接等待窗口。完整调用144.43715→133.3206μs，降低7.696462%。

## 编译阶段定位

对原063完整编译日志`../gqa-goal-20261121/gqa/lowered-ir-analysis/attempt-063-cli/stderr.txt`逐阶段检查，摘要见[阶段摘录](compiler-schedule-evidence.json)。

1. 固定源码532行PV dot，534行`tl.sum(p,1)`，已有PV后归约的文本顺序。
2. `InsertLoadStoreForMixCV`后已经存在P store/load→PV MMAD→fixpipe→PV load/cast→独立sum的顺序。
3. `CVPipelining`后该顺序继续保持，并可见GM workspace交接。
4. 本次dump中`CrossCoreGSS (hivm-cross-core-gss)`之后首次出现`sync_block_wait`，用于跨核生产/消费同步。
5. `GraphSyncSolver (hivm-graph-sync-solver)`对Vector函数补充流水间`wait_flag`；PV接收和转换仍排在sum前。
6. 节点22实际修改上述编译产物的后续IR版本，在晚层HIVM中移动sum/VREC链，再交给hivmc生成设备二进制。

据此可定位为“源顺序在混合Cube/Vector降低和同步调度后，未利用独立归约跨等待的机会”。这些证据不把某个同步pass本身定为错误：PV消费者的等待是必要的，优化对象是独立计算的发射位置及其缓冲安全性。确定应修改哪个通用pass，还需要在相应阶段实现重排并验证资源与同步约束。

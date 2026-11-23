# QKV：双头读取与逐头写回重排

当前已完整验收并保留：完整21/21、双方各48/48、ABBA及修改后双分析通过。目标63.257576→62.050287μs，较本阶段下降1.9085%，较历史最佳62.837203μs下降1.2523%。ABBA四区块为62.942693／62.148968／62.263392／62.826162μs，两段候选均快于两段原IR；新IR已接入26项对应主图、四项固定shape图及23页PPT。Triton源码保持`0af5d533afea8f4be88e1fd6e0fc4a6ba76fae9a8b1478f7c4b93247fe7ec070`。输入为FP16 `[8,512,3072]`、16 heads、`timing-2`。

上一轮KVQ调度完整合格但目标变慢。模型读取保留版的热点、流水、profiling、实际TTIR、BCMLIR及ttadapter，以及此前全部探索和负结果，随后登记新机制。当前目标特化的读取均无mask，原循环内分别读取Q/K/V，每个tile为64×64，内维128字节。

新IR使每程序读取两个相邻head，将tile改为32×128，内维加宽到256字节；每程序仍执行两轮。写回前将32×128拆为32×2×64，再转置为2×32×64，使每个head写入其对应的连续区域。程序总数保持512，重新映射head pair及64行序列块。Q的FP32中间精度和K/V的FP16加法顺序保持固定。

原ABI及正整数输入域保持一致；`BLOCK_D=64`与`EXACT`路径对应head_dim64，宿主在序列块数为偶数时选择GROUP2，捕获的num_heads可被16整除。宿主符号核对新读取元素`[b,s,pair*128+h*64+d]`与写出`[b,2*pair+h,s,d]`逐项对应；18组坐标检查包含目标4194304个输出元素，每个输出恰写一次。

实际BCMLIR读tile64×64→32×128，bias64→128，写回memref为2×32×64，新增3处reshape和3处transpose；静态copy6、分配6、循环1保持原数量。最终二进制4360→4912字节，SHA`92cc78be…`→`5cd32814…`。物理UB临时量、指令命令数及转置成本继续由正式产物和修改后重放核对。

宿主同时登记了模型原预测的偏差：512个程序各分量bias读取64→128个FP16元素，逻辑bias读取192→384KiB；主QKV读取与输出各24MiB，合计逻辑读取24768→24960KiB。每头连续写回块8→4KiB。模型依据MTE2与MTE3均值之和推断独占DMA串行，并预测转置完全被重叠；这两项保留为待验证假设，实际开销与完整调用性能分别验收。

完整评测与补测共有37个AST，原IR基线22个；共有22个中登记目标1个二进制变化，其余21个一致。额外15个补测AST与上一完整合格候选的对应二进制一致，比较来源单独记录；原基线缺少这15个回执，严格全集比较失败原样保存。

| 指标 | 本阶段原IR | 新IR |
| --- | --- | --- |
| 固定shape耗时 | 63.257576μs | 62.050287μs |
| 同轮固定shape／CANN | 1.446907× | 1.468665× |
| 原始配对全合同geomean | 1.645464× | 1.672510× |
| 新候选逐case CANN共同锚点重算 | 1.629464× | 1.672510× |

全合同候选耗时原／改几何比较为1.026417×。原始配对geomean、共同参考重算、固定shape收益和同二进制特化的跨事件测量变化分别保存。历史原始全合同成绩1.679409×冻结；采用新候选逐case CANN锚点时，该历史点重算为1.672449×，新候选1.672510×，两者几何比较1.000036×。因此完整合同相对历史的几何平均基本持平，固定shape下降1.25%，本阶段原IR的复测改善单列。原生及四个源码事件的源码字节、实际送测request、完整合同与结果已逐一核对，原生使用request-v2适配记录，详见[历史测量绑定](qkv/ir002-history-import-review.json)。

修改后独立profiling覆盖512个程序、1650MHz。每程序MTE2时间1.809911→1.127945μs、MTE3时间1.154888→0.950405μs；Vector时间0.739323→1.031745μs。GM→UB总读取24768→24960KiB，写回24576KiB；MTE3指令每程序7→13。读写时间下降与新增向量、写命令及bias读取成本一起保留，等待后模拟器进一步核对。

修改前读取及决策顺序已核对，本次实际BCMLIR和ttadapter的成功返回均在decision和IR编辑之前。固定源码用于隔离执行组织这一编译变换，同类分块与转置可在Triton表达。

修改后profiling及模拟器均与正式源码、IR、case和目标二进制精确绑定；26/26个独立PC时钟核对通过。新窗口出现6处向量UB→UB搬移PC，记录18次调用；窗口分别包含1次END及局部后续事件，动态累计次数按各自覆盖解释。MTE3等待占比67.38%→51.82%、Vector等待74.41%→69.28%，各窗口分别归一化。模拟器应用未完整结束，wrapper0／超时子进程9保留，直接PC源码映射范围保持明确。模型关于转置完全重叠及独占DMA串行的预测继续作为假设；已确认的收益与新增成本分别记录。

最终归档472份逐文件SHA校验通过，正式快照427份及早期CPU快照53份另存。[保留决定](qkv/selection.json)、[正式二进制绑定](qkv/observation-binding-ir-002.json)、[后机制审查](qkv/post-mechanism-ir002.json)、[ABBA](qkv/stability/ir-002/summary.json)、[完整归档](exports/qkv-ir002-accepted-complete.tar.gz.receipt.json)均已落盘。

材料：[修改前决策](qkv/agent-calls/ir-002/before-edit-decision.json)、[读取顺序](qkv/agent-calls/ir-002/evidence-read-review.json)、[实际IR diff](qkv/agent-calls/ir-002/actual-ir.diff)、[语义与坐标审查](qkv/agent-calls/ir-002/ir-review.json)、[CPU编译](qkv/agent-calls/ir-002/lowering-review.json)、[低层机制](qkv/agent-calls/ir-002/lowering-mechanism.json)、[完整结果](qkv/evaluations/ir-002/result.json)、[双补测](qkv/evaluations/ir-002/supplementary-summary.json)、[二进制核对范围](qkv/ir002-binary-comparison-scope.json)、[待验收收益](qkv/provisional-gain-ir-002.json)、[正式快照归档](exports/qkv-ir002-formal-pending.tar.gz.receipt.json)。427份正式快照逐文件SHA校验通过，早期53份CPU阶段证据另存；完整稳定性与后重放归档随后新增。

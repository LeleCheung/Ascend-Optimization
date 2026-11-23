# BatchNorm：广播外提改变BCMLIR，最终二进制保持一致

当前结论：模型外提6处纯广播，语义审查和CPU编译通过。最终原／改二进制逐字节一致，保留8×256最佳版本，本次记录为编译阶段探索。199条归档记录保存修改前决策、模型输出、IR差异、实际低层产物和宿主审查。

固定Triton源码为`b8d4d4a2c07dabfc308e94412d0b5c219eb56ee4c7dc5358016e0eb2782b2f56`，从已验收的输出8×256 IR出发。模型读取保留版热点、流水、profiling、TTIR及BCMLIR后，将weight、inv_std、bias在8×256／8×128两条输出分支中的广播移至输出m循环之前。SSA名称与源码位置归一化后，操作多重集合相同，除6处广播位置以外的操作顺序保持一致；浮点乘加、统计、内存地址、循环边界和ABI均保留。

| 检查 | 原IR | 广播外提IR |
| --- | --- | --- |
| TTIR SHA前缀 | `574eddcb…` | `3747a02e…` |
| 本次CPU生成BCMLIR SHA前缀 | `8db18b3b…` | `6dd343cc…` |
| 输出内层fill数，宽／窄分支 | 3／3 | 0／0 |
| 全文件fill／copy／alloc／loop数 | 16／6／6／6 | 16／6／6／6 |
| 最终二进制字节数 | 8416 | 8416 |
| 最终二进制SHA | `41483d74…` | `41483d74…` |

完整二进制SHA为`41483d746ba67cd3456434ea7d176386cbe6e987f3ab56d360d1baf5935be5fe`，也与保留版正式评测二进制一致。CPU比较使用相同target与options，NPU保持未初始化。高层填充位置的变化经过后续编译后得到相同执行产物；本次没有新增设备性能测量或收益节点。

模型原解释将同一`tensor.empty`作为多个`linalg.fill`的outs关联到物理UB复用及RAW／WAR链。宿主将其保留为待核验假设：tensor SSA结果分别定义，物理缓冲与依赖由后续bufferization和指令证据确认。安装版Triton也提供`broadcast_to(input,*shape)`并调用`broadcast_impl_shape`；固定源码是本次隔离编译变换的实验条件，具体源码表达与最终调度仍可独立验证。

后续已转向另一条有最终指令线索的机制：统计循环的向量整数计数从全零反复加一，重放中有64次S32 VADDS与最终两次S32→FP32转换，共享UB地址可关联该状态。宿主保存[指令观察](batchnorm/workspace/count-uniformity-probe.json)，模型在逐字节恢复的8×256 IR上研究计数均匀化。已有标量倒数计算继续保留，具体等价性和最终代码变化待验证。

材料：[修改前决策](batchnorm/agent-calls/ir-004/before-edit-decision.json)、[IR差异](batchnorm/agent-calls/ir-004/actual-ir.diff)、[语义审查](batchnorm/agent-calls/ir-004/ir-review.json)、[CPU编译](batchnorm/agent-calls/ir-004/lowering-review.json)、[低层结论](batchnorm/agent-calls/ir-004/lowering-mechanism.json)、[语言接口核对](batchnorm/agent-calls/ir-004/language-api-review.json)、[保留决定](batchnorm/retention-ir-004.json)。

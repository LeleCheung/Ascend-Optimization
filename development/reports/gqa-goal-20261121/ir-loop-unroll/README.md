# 固定源码IR循环展开负结果

冻结019的Triton源码，将目标BCMLIR中的四轮querytile循环显式展开；每轮操作经SSA改名与常量索引折叠后逐操作一致。原四项通过，目标262.27775→286.64265μs，保留原IR；当前全局029的232.23705μs单独比较。

修改后profiling与模拟器重放完成，formal／profile／sim目标二进制关联通过。三核逐PC时钟81/81、83/83、83/83；部分窗口与应用完成manifest缺失继续保留。展开增加中间搬运与live range，性能结论采用完整调用实测。

[完整结果](../gqa-ir-loop-unroll/validation-ir001/reviewed-result.json) · [后双分析关联](../gqa-ir-loop-unroll/post-ir001/binary-binding-and-facts.json) · [等价审查](../gqa-ir-loop-unroll/agent-calls/ir-001/unrolling-equivalence-review.json)

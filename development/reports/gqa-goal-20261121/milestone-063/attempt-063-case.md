# 063：模拟器指导连续单tile QK前瞻

目标完整调用 **154.8871 → 146.0561 μs，下降5.7016%**；同轮CANN126.3941μs，对应 **0.865380×CANN**。原官方case1/2/4/7全部通过，修改后profiling、模拟器重放与自动身份绑定完成。063已替换059，继续争取1×以上。

## 修改前判断与实际修改

profiling可见算术/搬运总量和硬件kernel时间，但仅靠总量不能决定下一块QK应放在哪里。模拟器的当前QK→当前PV→下一QK顺序及交接等待，指导连续一块QK前瞻。Codex提出分析，CC／DeepSeek编写候选；模型实际读取了TTIR、profiling与模拟器，未读取BCMLIR，源码注释中的相应暗示已在宿主审查纠正。

实际只改Triton `_gqa_tile_part_body`：每个非空KV组交集先填充第一块FP16 raw QK；循环携带分数，在当前softmax/PV前发起下一块有效QK。尾部无额外QK，不跨KV组读取。BF16 PV、FP16交接、自然指数、FP32分母/除法、host缓存/分配、20 worker与num_stages=2保持。恢复该helper后全模块AST与059一致。旧030/042的前瞻负结果继续保留，本轮是当前缓冲与数学组合下的新检验。

## 真机与机制证据

| 固定case | 完整调用 μs |
|---|---:|
| 1 | 107.1206 |
| 2 | 102.5076 |
| 4 | 146.0561 |
| 7 | 95.9006 |

目标三批145.78515／146.0561／146.14565μs；独立硬件kernel159.603195→150.383011μs，同方向。收益明显大于本轮批间差异，无已观察矛盾，按条件策略不追加ABBA。

候选同一Cube窗口：预填QK130.968–131.603μs；下一QK134.193–134.827μs，FIX134.829–136.385μs；当前PV136.661–137.295μs，FIX137.297–138.230μs。059另一个窗口内：当前QK118.396–119.031μs，当前PV124.459–125.094μs，下一QK126.328–126.963μs。比较各自窗口内的操作顺序，不比较两个窗口的绝对时间原点。Scalar WAIT发射区间可能与已排队Cube工作重叠。

全kernel MMAD2048、GM→L1 54MiB、L0C→GM48MiB不变，支持调度改变而非总量削减的解释；尚未独立分解每项调度贡献。三核时钟30/30、23/23、23/23；模拟器是部分窗口且application_completed=false，正式四项完整结束由正式结果证明。既有FP16交接范围限制保留，完整20项仍属独立后续范围。

## 可追溯材料

[保留决定](gqa/retention-attempt-063.json) · [原四项](gqa/iterations/attempt-063/parent-validation/results.json) · [修改前审查](gqa/agent-calls/attempt-063/host-review.json) · [调度机制](gqa/iterations/attempt-063/lookahead-mechanism.json) · [身份绑定](gqa/iterations/attempt-063/binary-binding-and-facts.json) · [计时策略](gqa/measurement-policy-attempt-063.json)。

源码19f7fb238946cb9006286d4d222425d07c4ea9fc38a7d4ff85c71c72c49c584a；真实父版059为98c755709dd428d0955bfc11a666b679f5afdc412ecb441e1c094568a3805661。主图新增橙色device节点，未发生手工IR修改。

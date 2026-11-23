# 固定063：原生倒数与QK前瞻组合

本轮固定063源码的原生倒数IR已保留：**145.9246μs、同轮CANN130.3856μs、0.893513×CANN**。正式耗时较063历史146.0561μs下降0.090034%；同环境交错147.254375→146.55685μs下降0.473687%，两个候选区块均更快。比值与历史0.865380×的差异同时包含CANN参考变化。

## 进入IR的具体依据

当前profiling给出kernel150.383011μs、全kernel2048次MMAD、54MiB GM→L1、48MiB L0C→GM。模拟器关联当前归一化VDIV，每个Vector窗口VDIV共26次；汇总量不能单独证明归一化在关键路径，完整调用收益仍待实验。

历史058从Triton表达原生倒数仍生成VDIV；编译pass取证确认高精度分解会将VREC改为VDIV。固定059的晚层原生倒数IR003已确认0.948955%同环境完整调用收益，但未推进历史最佳。新分支研究该机制能否延续到063前瞻流水，历史结果分别保留。

063实际后端模块已捕获；原样编译二进制3972cdc5bef4ebdc816b6dccaf3405946f390cc04b21fa3e56ca7de85c11832f与正式二进制一致。独立后端续编译同样匹配，原IR入口四项通过，目标146.60115μs。该同实现测量只进详情。

## 本轮范围

Codex分析，CC／DeepSeek编写IR。保持源码、Cube、host、QK前瞻、归约位置、精度与事件；只调整归一化。现有分母位于UB180224、acc/output位于81920，原VDIV临时区位于114688。由模型核对别名与临时区要求，宿主审查后CPU编译，再按原四项与后双分析验收。原生VREC增加近似误差，复用原checker与输入，既有FP16交接边界保留。

远端：`/data/hanle/ascend-optimization/runs/gqa-goal-20261121/gqa-ir-lookahead-reciprocal`。ir-001已完整验收；ir-002只研究重复Vector屏障，由模型继续。

## ir-001实际改动与完整验收

模型在晚层HIVM新增原生VREC与二维VMUL私有声明；把64个FP32分母先倒数化，再广播乘acc。分母180224..180480、倒数114688..114944、乘法临时114944..116992、acc/output81920..114688不重叠，倒数及乘法临时复用归约后已死概率缓冲。Cube、host、公开ABI、源代码、事件及归约位置保持。原来的归约→VDIV屏障保留，实际又增加两处PIPE_V屏障；模型摘要称一处的错误已纠正。

| 原四项 | 完整调用 μs |
|---|---:|
| 1 | 100.6841 |
| 2 | 97.98255 |
| 4 | 145.9246 |
| 7 | 105.0386 |

原四项checker与同轮CANN全部通过。只有登记的目标特化注入IR，其余特化正常编译，其他用例计时波动不归因于该目标IR改动。目标mere0.001137539、mare0.007812494、max_diff0.0009765625，近似误差增加，仍通过原阈值；既有FP16范围限制与完整20项后续范围保持。

后重放确认VREC出现、VDIV消失。三核时钟30/30、27/27、27/27，formal/profile/simulator目标二进制绑定完成。Vector窗口VREC各15次0.19μs、两处VMUL分别14次2.52μs/15次1.46μs；末端调用数不同属于部分窗口，不能直接相加推断完整归一化耗时。独立kernel152.443054μs较063的150.383011μs慢，保存这一方向差异。

条件交错A147.3456／B146.4446／B146.6691／A147.16315μs；两候选区块均更快，确认本环境完整调用收益。正式145.9246μs也略低于历史146.0561μs，故作为固定源码IR进展接入主图。只声明完整调用增益，不把独立kernel表现解释为提升。原IR入口146.60115μs与交错数值均保留详情，不画源码复测。

[保留决定](gqa-ir-lookahead-reciprocal/retention-ir-001.json) · [原四项](gqa-ir-lookahead-reciprocal/validation-ir-001/results.json) · [交错](gqa-ir-lookahead-reciprocal/conditional-stability/ir-001/summary.json) · [后指令](gqa-ir-lookahead-reciprocal/post-ir-001/native-reciprocal-mechanism.json) · [当前实现](gqa/retained-implementation.json)。

源码仍19f7fb23…；IR a28d48359241880dcb0890268a3695a1d8118c9327156c07081b0b079a5d1e7b，二进制5664717854b3f3cb7fe2a508b271b81216dab8ebbaceb27ed3eb619a56228473。源码选择gqa/selection.json仍保存063；实际当前最佳组合及注入manifest由retained-implementation.json保存，不能用裸063源码冒充该IR成绩。

## 下一项机制

ir-002仅删除VREC之后、VMUL之前两处连续Vector屏障中的第二处。两处之间只有视图和指针操作，候选须经模型读当前IR与重放、修改前登记，再由宿主核对。若CPU二进制一致，归入同实现详情并跳过真机复测；若不同，沿原四项与后双分析验证。

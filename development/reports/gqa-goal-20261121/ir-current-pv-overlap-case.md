# 固定059源码：当前缓冲组合的独立分母归约调度

当前全局最佳保持059：154.8871μs、同轮0.843809×CANN。新分支 `gqa-ir-current-pv-overlap` 固定059源码，研究将独立FP32分母归约放到P发送之后、PV返回等待之前。本次原四项、后双分析、身份绑定与条件交错均已收尾，结果未保留；旧038同类重排未确认收益，额外事件对变体变慢，均保留为前置证据。

## 当前证据与选层

059独立硬件kernel159.603195μs。profiling用于定位开销，当前源绑定模拟器进一步显示veccore0的PV等待120.349–126.106μs，而分母求和VADD开始于126.548μs、VCADD结束于127.098μs。逐编译阶段CVPipelining与最终HIVM确认分母归约仍在PV之后。流水可以重叠，标量等待的发出时间与Vector完成时间分开解释；这些观测支持调度实验，尚未证明它位于整体关键路径。

原样BCMLIR直接编译、捕获真正后端输入后独立HIVM续编译，二进制均与059实测版本逐字节一致。原IR加载入口原四项通过，目标155.81665μs属于同实现测量，主图隐藏，059历史数值冻结。读取IR用于分析与实际修改IR分别记录。

## 候选与审查

ir-001只移动六条现有语句，但修改前缺少要求的冻结源码读取，审查退回，未进行CPU候选编译或设备评测。原件、诊断及模型事件保存。

恢复059原IR后，ir-002补齐源码、profiling、模拟器与实际IR读取，并重新登记后编辑。六条语句为归约输出/工作区指针、三个memref转换与原归约调用，移到P发送通知后。全部数学、浮点加法树、缓冲地址、Cube代码、函数声明和原同步事件保持。原位置的V→MTE2事件继续保护下一轮QK工作区复用。

地址审查：概率输入UB[114688,180224)，求和工作区[0,32768)，分母[180224,180480)；P发送[32768,65536)，PV载入[65536,81920)，PV转换[81920,114688)，互不重叠。此前Vector屏障保证概率生产及旧QK读取完成，除法前屏障覆盖分母和PV结果写入。

ir-002修改前读取与顺序审查通过，CPU编译通过，ABI与参数元数据保持，执行二进制实际改变。原四项及后双分析已完成。保持原输入、checker和计时合同，未增加每轮测试；后重放将按新二进制实际PC关联归约顺序，覆盖不同的动态次数不直接相比。本次未确认新增IR收益。

证据目录：[固定源码与IR实验](gqa-ir-current-pv-overlap/)、[059原样编译关联](gqa/lowered-ir-analysis/attempt-059-cli/binding-result.json)、[原四项入口验证](gqa-ir-current-pv-overlap/validation-original/reviewed-result.json)。

## 最终结果

原四项全部通过，目标完整调用162.7046μs，独立硬件kernel155.103104μs。完整调用与硬件kernel方向不同，故进行一次条件交错：原IR161.072675μs、修改IR161.913425μs，慢0.521969%，两个候选区块均未快于原版；保持059最佳。

后重放确认预期顺序：修改版veccore0求和45.243–45.794μs，PV等待45.249–49.882μs，归约已与PV等待重叠。原版独立窗口求和126.548–127.098μs，晚于PV等待120.349–126.106μs。两个窗口绝对起点不同，只在各自窗口内判断依赖顺序。指令提前已经兑现，完整调用收益未兑现；搬运竞争或其他关键路径影响仍属待区分机制。

自动绑定通过。Cube时钟22/22；两个Vector窗口初次逐PC核验未通过，定点检查发现各7个短指令PC存在≤0.011μs舍入差异，CSV与trace事件数相同。独立采用原5%容差核对≥1μs的PC，各16/16通过；原失败、短PC差异及校准范围分别保存，未把它写成全部24个PC无差异。模拟器application_completed=false为部分窗口，原四项完整完成另有正式结果。

[保留决策与机制核对](gqa-ir-current-pv-overlap/retention-ir-002.json) · [原四项结果](gqa-ir-current-pv-overlap/validation-ir-002/results.json) · [交错比较](gqa-ir-current-pv-overlap/conditional-stability/ir-002/summary.json) · [后分析与绑定](gqa-ir-current-pv-overlap/post-ir-002/binary-binding-and-facts.json) · [短PC时钟范围核对](gqa-ir-current-pv-overlap/post-ir-002/simulator-parent/clock-calibration-resolution.json)。

下一条独立研究固定059源码，检查FP32 native VREC的IR表达；安装的Ascend C接口和编译器操作名已留证，当前仍需模型实现与CPU可行性核对，尚无该分支性能结论。原058 Triton倒数仍降为VDIV且未获完整调用收益，作为前置结果保留。

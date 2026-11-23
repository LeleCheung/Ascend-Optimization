# 固定059源码：原生FP32倒数的IR表达

固定059源码的ir-003已确认同环境完整调用收益：161.2564→159.72615μs，下降0.948955%，两个候选区块均更快。正式160.29965μs未超过059历史154.8871μs，因此不接入最佳主线；当前全局最佳已为063的146.0561μs、0.865380×CANN。ir-002保留Newton未确认稳定收益；ir-001原生指令被编译器改回VDIV、未上机。

## 为什么进入IR

当前模拟器将输出归一化映射到veccore0/1各两处VDIV，当前部分窗口各12次，累计3.16与1.69μs。原058的已安装Triton CANN libdevice.reciprocal映射__hmf_recipf，实际仍生成VDIV。安装的Ascend C dav_c220接口直接调用vrec，支持float；本机编译器识别hivm.hir.vrec操作。以上说明值得验证具体IR控制点，未穷举所有Triton表达。

## ir-001：CPU编译成功，预期指令被改回VDIV

模型只修改目标BCMLIR的归一化：行分母VREC → FP32 r*(2-den*r) → 广播后乘acc。其他数学、归约树、源码与host保持。修改前读取与决策顺序审查通过，CPU编译成功。

实际逐pass诊断发现：stage63 CVPipelining仍含VREC；stage211 HIVMDecomposeOp改成广播1与VDIV；stage378已无VREC，附加Newton运算仍保留。该候选未实现原生指令目标，未上机，作为编译层负结果保留。标准objdump对本架构只返回not available，不据此声明机器指令已被核对。

进一步检查公开AscendNPU-IR源码commit `02cd0c7060f64704761cf91dff8ed7f16d5662e0`：HIVMDecomposeOp.cpp的VRecOpHighPrecisionLowering明确因VREC精度较低而改成VBrcOp＋VDivOp，匹配本机观察。HIVMToStandard.cpp与EltWise1D.cpp仍提供原生VREC库调用转换和float实现注册。公开版本与安装版本的整体一致性未确认；分解行为与实际本机编译结果一致。

## ir-002：分解之后的调用修复

捕获ir-001实际编译器导出的后端HIVM，原样独立续编译与ir-001二进制完全一致。模型从该未上机IR分支继续，尝试将小规模1/den库调用替换为原生VREC，保留Newton修正、所有缓冲和同步；函数声明及库调用ABI由实际文件与公开模板共同审查，最终以本机CPU链接为准。ir-002已完成模型提交与修改前读取审查。本机HIVM续编译及链接成功，公开符号和参数元数据与059保持；原四项及后双分析、自动绑定和一次条件交错均已完成，结果见下。

数值要求：保持原四项输入、checker、seed和计时参数；不把小于BF16量化间距的近似误差当成相同舍入的保证。原生VREC精度及Newton后的实际误差需真机验证；原概率BF16/PV结果FP16范围限制保持。

证据：[实验目录](gqa-ir-native-reciprocal/) · [本机能力调研](gqa/native-reciprocal-research/) · [固定commit公开源码](native-vrec-public-source/sources.json)。

实际晚层改动为增加原生vrec_1d_float私有声明、替换1/den调用，并去掉只服务该分子的1.0填充及其一处相邻Vector屏障。前置屏障仍覆盖归约到倒数的依赖，后置屏障覆盖倒数到Newton的依赖；模型概述遗漏的这一次屏障删除已在宿主审查中纠正。原始BCMLIR候选、晚层修复父链、原样续编译与新CPU二进制分别保存。

## ir-002最终结果

原四项全部通过，目标完整调用155.79715μs，同轮CANN127.8356μs；独立硬件kernel158.223160μs。与059的155.83015μs同环境交错相比，候选中位155.19690μs、改善0.406372%，但两个候选区块未一致快于原版，正式成绩也未推进历史154.8871μs，因此未保留为新最佳。

后重放确实出现原生VREC，VDIV消失。三核时钟22/22、31/31、31/31，formal/profile/simulator身份绑定通过；application_completed=false为部分窗口，原四项完整结束由正式结果证明。原059窗口两处VDIV各12次；新版窗口VREC及归一化相关指令各15次，按每调用比较：原VDIV累计0.404167μs，新VREC＋Newton＋输出乘法累计0.346667μs。各流水可重叠，这些累计值不代表完整调用收益，也不直接比较12与15轮的原始总量。

新版VREC本身约0.012667μs/调用，Newton的两处VMUL、VMULS和VADDS合计约0.056667μs/调用，另有同步和发射成本。目标checker最大绝对误差维持0.0009765625；mismatch_count由113818变为113820，原阈值保持通过，未声称输出逐位相同。

[保留判断](gqa-ir-native-reciprocal/retention-ir-002.json) · [原四项](gqa-ir-native-reciprocal/validation-ir-002/results.json) · [交错测量](gqa-ir-native-reciprocal/conditional-stability/ir-002/summary.json) · [实际指令核对](gqa-ir-native-reciprocal/post-ir-002/native-reciprocal-mechanism.json) · [自动身份绑定](gqa-ir-native-reciprocal/post-ir-002/binary-binding-and-facts.json)。

## ir-003：原生倒数取得同环境收益

固定059源码，从ir-002删除两处行VMUL、VMULS、VADDS及死操作数、三处Vector屏障，输出广播直接使用原生FP32倒数。保留保护VREC输入、结果与输出乘法的屏障；原生倒数[114688,114944)、乘法临时区[114944,116992)、acc/output[81920,114688)、分母[180224,180480)不重叠。修改由模型完成，宿主审查实际依赖、数值与CPU编译。

原四项全部通过，完整调用分别101.1126／98.0366／160.29965／93.3121μs；目标同轮CANN128.91715μs，硬件kernel153.963074μs。由于收益幅度小，条件交错验证059基线161.2564→候选159.72615μs，下降0.948955%，两个候选区块均更快。正式值未超过历史最佳，保留为受控IR收益案例与组合分支，不将交错数值替换历史主图。

后重放确认VREC存在、VDIV与Newton消失，三核时钟22/22、26/26、26/26与绑定完成。两个Vector窗口各16次，VREC累计0.20μs；两处输出VMUL累计2.88／1.56μs，QK缩放VMULS2.83μs属于其他计算，不计入归一化。部分窗口与正式完整执行分别记录。

目标checker的mere从059约0.000431598升至0.001137539；mare0.007812494、max_diff0.0009765625，原阈值通过。该修改增加近似误差，不能推导全部输入范围保证。源码保持059；IR SHA be4f69f4d9a2f1f8999dfa8fc0548b9f4a06bc1271e837b8a3a6424469923015。

[保留判断](gqa-ir-native-reciprocal/retention-ir-003.json) · [原四项](gqa-ir-native-reciprocal/validation-ir-003/results.json) · [条件交错](gqa-ir-native-reciprocal/conditional-stability/ir-003/summary.json) · [实际指令](gqa-ir-native-reciprocal/post-ir-003/native-reciprocal-mechanism.json)。

# 四算子工作流通用知识

执行环境为Ascend 910B4-1、CANN 9.0.0、Python 3.11.15、Triton 3.5.1、Flagtree 0.6.0。只使用物理卡7，同卡串行；设备任务由KGS执行，模型只返回候选源码。

本批验证历史方法能否再次产生有效修改。按各算子知识中的已验证方法改进明确的优化前版本，不要求重新发明算法。保留所有接口、dtype、shape、stride及特殊值语义；完整正确性与性能合同决定是否采纳。

当前Ascend后端支持num_warps、num_stages、multibuffer、unit_flag；语义不能照搬CUDA。JIT内部常量用字面量或constexpr。每次launch使用有界grid和循环覆盖，不能生成百万program。

Triton启动grid必须为tuple或返回tuple的函数。例如`kernel[(grid,)](...)`，不能写`kernel[grid](...)`并传入整数。旧建议驱动批次第2轮已出现`TypeError: object of type 'int' has no len()`，在真正执行内核前失败，原结果保留。

只使用起点源码已有或知识明确确认的导入；不要为了新kernel添加无用兼容模块。`20261014-four-direct`的logaddexp2前两轮添加了`flag_gems.utils.triton_lang_utils`，但本机固定FlagGems没有这个模块，导致零用例执行。此处只需`torch`、`triton`、`triton.language`与起点已有的`pointwise_dynamic`，无需引入该模块或`CodeGenConfig`。

有界grid不能只写`min(cdiv(n, span), limit)`：截断grid后，kernel必须用grid-stride循环处理剩余数据。logaddexp2的正式正确性形状最大约787万元素，性能形状包含约10.7亿元素，不能假设小形状正确便证明大形状完整覆盖。循环次数很大时采用运行时`range`；`tl.static_range(64)`会展开64份exp/log和访存代码，与历史运行时64轮循环不是同一种实现。

真机PipeUtilization是流水活跃比例，可以重叠，不是HBM带宽利用率。逻辑读写字节只描述算法工作量。当前没有可信的物理计数与硬件roof，不输出数值Roofline。

依据逐case时间、源码与真机流水选择修改；保留实际生成理由、失败及最终源码。已有稳定回退可以保留。编译错误与运行错误须区分，不允许用吞掉执行异常代替正确性保证。

三个计算算子使用device_kernel；narrow_copy使用device_task，覆盖真实DMA与kernel。加速比为PyTorch时间除以候选时间，性能case取几何平均，最终报告采用同源码两次完整复验的较慢值。


# 本批logaddexp2补充，优先执行下述已验证方法

# logaddexp2优化知识

这些提示来自固定FlagGems源码、测试语义、参考代码及已验证历史优化方法。方法复现明确使用历史建议，不作为未知方法的盲测。此算子走AIV Vector，不能套用矩阵乘的Cube分块或dot累加器优化。

## 本次固化的有效方法

方法复现从`candidates/flaggems-master-20261012-agent.py`开始，完整36项，历史约0.231×。历史最佳位于`reports/kg-profile-20261012-agent/export/main.py`，约1.267×；来源与逐项依据见`../../tools/profiling/历史复现知识-20261014.md`。

使用`m + ln(1 + exp2(-abs(x-y))) * log2(e)`稳定公式，max可在输入精度计算，差值与修正项保留FP32。使用`where(t <= 200, t, 200)`处理同号无穷大产生的差值NaN，真实NaN仍通过`maximum(..., PropagateNan.ALL)`传播。保留广播、非连续和类型提升回退。

连续同形输入扁平访问，整除时去mask，一个program处理多tile。历史有效半精度中等输入为tile8192、循环4、num_stages=1；超大输入为tile4096、循环64、num_stages=2；FP32为tile4096与相邻tile预取。配置是候选依据，完整测试决定是否有效；grid必须有界并覆盖全部输出。

这里的半精度循环为运行时`range`，不是把64轮全部`tl.static_range`展开。保持`max`在输入dtype计算，只把差值和修正项转FP32，减少UB临时向量；在块完整覆盖时取消load/store的mask，余数路径仍保留mask。自然对数`tl.log(1 + tl.exp2(-t)) * 1.4426950408889634`是历史已测有效的替换；只加连续快路径但仍沿用全FP32的`tl.log2`及每块mask，尚未执行这些关键建议。

## 数值与接口

语义为`log2(2**x + 2**y)`。上游稳定公式是`max(x,y) + log2(1 + exp2(-abs(x-y)))`，中间值使用FP32。直接计算`2**x + 2**y`有溢出风险。

相同无穷大使`x-y`为NaN，此时结果应恢复为`max(x,y)`；真实NaN输入仍须传播NaN。保持广播、stride、输出dtype和类型提升，不用同dtype连续输入的优化破坏通用回退。FP16/BF16下减少中间精度可能引入消减误差，不能仅凭随机值精度通过就假设安全。

固定测试包含常规形状及非有限值；独立完整合同为36项。测试范围来自KGS inspect，不能删减、特判测试输入或从benchmark名称选实现。按公开shape/dtype/stride选择实现是正常优化。

## 优化方向

1. 先按正式逐case延迟和真机流水指标选择方向。Vector比例高时检查exp/log、转换与临时向量；MTE2比例高时检查连续搬运和分块；Scalar比例高时检查动态广播索引与program覆盖。流水指标之间可重叠，不能相加，也不等于峰值利用率。
2. 连续同形输入可绕过`pointwise_dynamic`的通用地址计算，但保留广播、非连续及类型提升回退。一个program处理多块可以减少控制开销，也会降低并行度；避免相邻尺寸从大量program骤降到一个program，尤其不能把FP16慢case隐藏在整体几何均值中。
3. exp/log通常产生多个FP32临时向量，增大BLOCK需要考虑UB压力。融合表达式、减少存活向量、改变块大小或每program循环次数后，都须记录完整正确性和全部性能case。
4. 数值等价的`log1p(exp2(-abs(x-y))) / ln(2)`可作为备选，但不保证更快。固定参考代码的`third_party/ascend/language/cann/libdevice.py`定义`log1p`，其支持类型随路径变化；引用之前核实当前安装的API和可编译性。不能凭参考文件存在便假定当前后端支持，也不把它当作已验证改进。
5. 若尝试近似exp/log或较低精度，明确误差来源并保持NaN/Inf语义，使用完整合同复验；不修改容差、不调用PyTorch原算子替代，不静默吞掉运行错误。优先采用本机可编译且正确的公式和分块改进。

数值Roofline需要真实物理字节、相应指令工作量、设备范围和可信roof；exp/log不能简单算成一条普通FLOP。当前使用实际流水和延迟指导候选，不声称已达到硬件上限。



# logaddexp2优化知识

这些提示来自固定FlagGems源码、测试语义、参考代码及已验证历史优化方法。方法复现明确使用历史建议，不作为未知方法的盲测。此算子走AIV Vector，不能套用矩阵乘的Cube分块或dot累加器优化。

## 本次固化的有效方法

方法复现从`candidates/flaggems-master-20261012-agent.py`开始，完整36项，历史约0.231×。历史最佳位于`reports/kg-profile-20261012-agent/export/main.py`，约1.267×；来源与逐项依据见`../../tools/profiling/历史复现知识-20261014.md`。

使用`m + ln(1 + exp2(-abs(x-y))) * log2(e)`稳定公式，max可在输入精度计算，差值与修正项保留FP32。使用`where(t <= 200, t, 200)`处理同号无穷大产生的差值NaN，真实NaN仍通过`maximum(..., PropagateNan.ALL)`传播。保留广播、非连续和类型提升回退。

连续同形输入扁平访问，整除时去mask，一个program处理多tile。历史有效半精度中等输入为tile8192、循环4、num_stages=1；超大输入为tile4096、循环64、num_stages=2；FP32为tile4096与相邻tile预取。配置是候选依据，完整测试决定是否有效；grid必须有界并覆盖全部输出。

## 数值与接口

语义为`log2(2**x + 2**y)`。上游稳定公式是`max(x,y) + log2(1 + exp2(-abs(x-y)))`，中间值使用FP32。直接计算`2**x + 2**y`有溢出风险。

相同无穷大使`x-y`为NaN，此时结果应恢复为`max(x,y)`；真实NaN输入仍须传播NaN。保持广播、stride、输出dtype和类型提升，不用同dtype连续输入的优化破坏通用回退。FP16/BF16下减少中间精度可能引入消减误差，不能仅凭随机值精度通过就假设安全。

固定测试包含常规形状及非有限值；独立完整合同为36项。测试范围来自KGS inspect，不能删减、特判测试输入或从benchmark名称选实现。按公开shape/dtype/stride选择实现是正常优化。

## 优化方向

1. 先按正式逐case延迟和真机流水指标选择方向。Vector比例高时检查exp/log、转换与临时向量；MTE2比例高时检查连续搬运和分块；Scalar比例高时检查动态广播索引与program覆盖。流水指标之间可重叠，不能相加，也不等于峰值利用率。
2. 连续同形输入可绕过`pointwise_dynamic`的通用地址计算，但保留广播、非连续及类型提升回退。一个program处理多块可以减少控制开销，也会降低并行度；避免相邻尺寸从大量program骤降到一个program，尤其不能把FP16慢case隐藏在整体几何均值中。
3. exp/log通常产生多个FP32临时向量，增大BLOCK需要考虑UB压力。融合表达式、减少存活向量、改变块大小或每program循环次数后，都须记录完整正确性和全部性能case。
4. 数值等价的`log1p(exp2(-abs(x-y))) / ln(2)`可作为备选，但不保证更快。固定参考代码的`third_party/ascend/language/cann/libdevice.py`定义`log1p`，其支持类型随路径变化；引用之前核实当前安装的API和可编译性。不能凭参考文件存在便假定当前后端支持，也不把它当作已验证改进。
5. 若尝试近似exp/log或较低精度，明确误差来源并保持NaN/Inf语义，使用完整合同复验；不修改容差、不调用PyTorch原算子替代，不静默吞掉运行错误。优先采用本机可编译且正确的公式和分块改进。

数值Roofline需要真实物理字节、相应指令工作量、设备范围和可信roof；exp/log不能简单算成一条普通FLOP。当前使用实际流水和延迟指导候选，不声称已达到硬件上限。

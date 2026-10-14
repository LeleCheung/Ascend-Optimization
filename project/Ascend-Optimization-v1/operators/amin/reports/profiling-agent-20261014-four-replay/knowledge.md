# 四算子工作流通用知识

执行环境为Ascend 910B4-1、CANN 9.0.0、Python 3.11.15、Triton 3.5.1、Flagtree 0.6.0。只使用物理卡7，同卡串行；设备任务由KGS执行，模型只返回候选源码。

本批验证历史方法能否再次产生有效修改。按各算子知识中的已验证方法改进明确的优化前版本，不要求重新发明算法。保留所有接口、dtype、shape、stride及特殊值语义；完整正确性与性能合同决定是否采纳。

当前Ascend后端支持num_warps、num_stages、multibuffer、unit_flag；语义不能照搬CUDA。JIT内部常量用字面量或constexpr。每次launch使用有界grid和循环覆盖，不能生成百万program。

真机PipeUtilization是流水活跃比例，可以重叠，不是HBM带宽利用率。逻辑读写字节只描述算法工作量。当前没有可信的物理计数与硬件roof，不输出数值Roofline。

依据逐case时间、源码与真机流水选择修改；保留实际生成理由、失败及最终源码。已有稳定回退可以保留。编译错误与运行错误须区分，不允许用吞掉执行异常代替正确性保证。

三个计算算子使用device_kernel；narrow_copy使用device_task，覆盖真实DMA与kernel。加速比为PyTorch时间除以候选时间，性能case取几何平均，最终报告采用同源码两次完整复验的较慢值。


# amin优化知识

## 起点与证据

方法复现从`candidates/direct-axis-native-v5-20261010.py`开始。这是明确的历史中间版本，已完成布局和直接轴归约优化，并非FlagGems master。历史自动批次`reports/profiling-agent-20261012-agent/`的第1、2轮采纳，第3—6轮保留失败及未提速证据；最终源码为`candidates/profiling-agent-final-20261012-agent.py`，27/27通过，两次0.942×/0.936×。

## 瓶颈到修改

1. 历史BF16中间轴归约明显慢于同形状FP16；Vector活跃比例较高。检查逐元素FP32转换、FP32临时向量与归约开销。min是选择原值，可尝试输入精度归约，但必须保持NaN/Inf及输出dtype。
2. `amin_native_middle_kernel`与`amin_native_last_kernel`可在归约维和输出维均被tile整除时取消恒真mask；非整除分支仍须保留边界保护。判断来自公开shape与tile，不读取测试名称。
3. BF16原生minimum在某些配置编译失败。保留FP32路径及其原始分块，不能只改变dtype后沿用过大的tile；仅对编译不可用配置回退，输出仍须完整计算。
4. 尝试multibuffer=True重叠搬运与归约，保留逐配置单缓冲回退。流水比例与容量估计只能提出假设，实际收益由完整测试决定。
5. 已有全归约多阶段路径和通用维度/布局回退保持可用。保留dim归一化、keepdim、stride与输出语义，不为了大形状删去小输入路径。

## 失败方向

历史第3轮生成失败；后续通过但未提速的候选没有替代最佳。不要把BF16原生路径存在等同于本机已使用或性能已提升；须根据编译日志和完整评测报告实际路径。无需重复搜索全部tile，优先复现上述两轮的有效方法。

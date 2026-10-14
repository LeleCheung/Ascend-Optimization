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


# amin优化知识

## 起点与证据

方法复现从`candidates/direct-axis-native-v5-20261010.py`开始。这是明确的历史中间版本，已完成布局和直接轴归约优化，并非FlagGems master。历史自动批次`reports/profiling-agent-20261012-agent/`的第1、2轮采纳，第3—6轮保留失败及未提速证据；最终源码为`candidates/profiling-agent-final-20261012-agent.py`，27/27通过，两次0.942×/0.936×。

## 瓶颈到修改

1. 历史BF16中间轴归约明显慢于同形状FP16；Vector活跃比例较高。检查逐元素FP32转换、FP32临时向量与归约开销。min是选择原值，可尝试输入精度归约，但必须保持NaN/Inf及输出dtype。
2. `amin_native_middle_kernel`与`amin_native_last_kernel`可在归约维和输出维均被tile整除时取消恒真mask；非整除分支仍须保留边界保护。判断来自公开shape与tile，不读取测试名称。
3. BF16原生minimum在某些配置编译失败。保留FP32路径及其原始分块，不能只改变dtype后沿用过大的tile；仅对编译不可用配置回退，输出仍须完整计算。
4. 尝试multibuffer=True重叠搬运与归约，保留逐配置单缓冲回退。流水比例与容量估计只能提出假设，实际收益由完整测试决定。
5. 已有全归约多阶段路径和通用维度/布局回退保持可用。保留dim归一化、keepdim、stride与输出语义，不为了大形状删去小输入路径。
6. 通用回退不要无关重写。`20261014-four-direct`第1轮引入`_middle_tile`返回`BC`与更大的`block`，grid按`block`计算，但kernel每program只写`BC`列。实际纯Python分块函数对FP32形状`(2,1024,4096)`、dim=1返回BC128、block1024，导致8192个输出仅覆盖1024个。启动步长必须等于kernel实际覆盖宽度，或kernel必须有相应多块循环。正式27项不包含这个反例，源码审查已拒绝该候选，不把小范围通过当成一般语义保证。
7. 第2轮把末轴kernel的`BO`（并行行数）错当成列tile，传入BO2048、BR4096，生成8388608元素的临时张量，超过本机Triton的1048576元素限制，性能测试编译失败。保留起点已验证的行数与列数定义，不要无关改造分块表；先执行整除去mask和多缓冲的方法。这个前端元素数量上限也不代表这么大的tile能装进UB，编译与完整评测仍须验证。
8. 第3轮理由声称重排了末轴调度，但实际差异只有注释、排版和配置表顺序。完整通过但仅0.921×，未采纳；独立AST结构检查确认没有改变实现。候选必须真正修改kernel与启动逻辑，理由中的计划不能代替源码里实际完成的优化。

## 失败方向

## 最小修改步骤

优先在既有`amin_native_middle_kernel`与`amin_native_last_kernel`增加一个`MASKED: tl.constexpr`开关，保留原有BC/BR/BO分块表和grid。中间轴仅当`R % BR == 0`且`I % BC == 0`时无mask；末轴仅当`O % BO == 0`且`R % BR == 0`时无mask。load与store都要一致分流，其他情况沿用原有mask。随后在这两个既有launch尝试`multibuffer=True`，编译不支持该配置时保留单缓冲回退。不得另建通用tile推断、删除已有路径，或把BO换成输入列数。先落实去mask和搬运重叠的实际代码；本轮不扩大BF16分块，避免把多个未验证改动同时引入。

历史第3轮生成失败；后续通过但未提速的候选没有替代最佳。不要把BF16原生路径存在等同于本机已使用或性能已提升；须根据编译日志和完整评测报告实际路径。无需重复搜索全部tile，优先复现上述两轮的有效方法。

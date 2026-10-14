# 四算子工作流通用知识

执行环境为Ascend 910B4-1、CANN 9.0.0、Python 3.11.15、Triton 3.5.1、Flagtree 0.6.0。只使用物理卡7，同卡串行；设备任务由KGS执行，模型只返回候选源码。

本批验证历史方法能否再次产生有效修改。按各算子知识中的已验证方法改进明确的优化前版本，不要求重新发明算法。保留所有接口、dtype、shape、stride及特殊值语义；完整正确性与性能合同决定是否采纳。

当前Ascend后端支持num_warps、num_stages、multibuffer、unit_flag；语义不能照搬CUDA。JIT内部常量用字面量或constexpr。每次launch使用有界grid和循环覆盖，不能生成百万program。

Triton启动grid必须为tuple或返回tuple的函数。例如`kernel[(grid,)](...)`，不能写`kernel[grid](...)`并传入整数。旧建议驱动批次第2轮已出现`TypeError: object of type 'int' has no len()`，在真正执行内核前失败，原结果保留。

真机PipeUtilization是流水活跃比例，可以重叠，不是HBM带宽利用率。逻辑读写字节只描述算法工作量。当前没有可信的物理计数与硬件roof，不输出数值Roofline。

依据逐case时间、源码与真机流水选择修改；保留实际生成理由、失败及最终源码。已有稳定回退可以保留。编译错误与运行错误须区分，不允许用吞掉执行异常代替正确性保证。

三个计算算子使用device_kernel；narrow_copy使用device_task，覆盖真实DMA与kernel。加速比为PyTorch时间除以候选时间，性能case取几何平均，最终报告采用同源码两次完整复验的较慢值。


# narrow_copy优化知识

## 起点与证据

方法复现从`reports/kg-profile-grid-device-master-20261010/export/main.py`开始，即历史KG有profiler的纯Triton版本，约0.806×。最佳`candidates/profile-small-dma-hybrid-20261010.py`采用小输入CANN DMA、大输入Triton，33/33通过，历史较慢复验1.102×。证据见`reports/master-closure-20261010/真机诊断.md`和`reports/diagnostics-20261010/`。

## 瓶颈到修改

1. 语义为沿dim选择[start,start+length)并复制到新分配的连续输出，不能返回共享输入存储的view。保留负dim、负start、空输出、非连续输入及一般布局路径。
2. master逐元素动态除法和索引映射很慢；纯Triton父版本已将连续区域映射成块复制，grid-stride循环减少多次launch。当前方法复现重点是在这个父版本上补小输入DMA。
3. 真机任务表显示小输入Triton设备任务慢于PyTorch的MEMCPY_ASYNC；大输入连续Triton复制已接近PyTorch。该结论来自device_task，不应再解释成纯host/launch开销。
4. 对连续输入、dim归一化后为0、正有效length且输出元素数不超过65536的单段复制，可通过ctypes调用`libacl_rt.so`的`aclrtMemcpyAsync`。使用新输出的data_ptr、正确起点字节偏移、D2D枚举3、输入设备的当前npu_stream；所有字节数用element_size计算。非适用输入回退父Triton实现。
5. 库不可用可回退；DMA调用返回非零应报错，不能假装输出已完成。使用异步当前stream，不在候选内人为同步来改变测量。

## 计时与采集

使用KGS19657、device_task，正式计时同时包含DMA与AIV kernel，排除host发射间隙。PyTorch小输入任务kernel_name可能是N/A；专用FlagGems计时副本已有字符串列兼容补丁。不能换到19655并遗漏DMA。

DMA任务没有AIV PipeUtilization，不把缺少Vector/MTE指标解释成零利用率或host瓶颈。保存原始附件和逐case正式时间；纯Triton分支可分析实际JIT符号的流水指标。

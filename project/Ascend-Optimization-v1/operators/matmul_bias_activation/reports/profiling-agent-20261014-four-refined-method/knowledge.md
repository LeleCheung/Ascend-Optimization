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


# matmul_bias_activation优化知识

## 起点与证据

方法复现从`candidates/profiling-dual-acc-v9-20261012.py`开始，是已优化中间版本，非master。历史`reports/profiling-agent-20261012-agent/`第3、5轮采纳，最终`candidates/profiling-agent-final-20261012-agent.py`完整42/42通过，两次0.908×/0.906×。

## 瓶颈到修改

1. 矩阵乘使用AIC Cube，bias和ReLU使用AIV。结合AIC MTE2、Scalar和MAC以及逐case时间，检查K循环控制、重复搬运和核覆盖；这些指标不能转换为未测量的HBM带宽。
2. 双N tile方案在同一次K循环中加载一次A，与相邻两个B tile做两路累积，复用A并减少循环与寻址成本。历史有效修改把该方案扩展到FP32规则方阵，重点是1024、2048、4096方阵，按公开形状和dtype分流。
3. 最终配置FP32采用BK128，半精度采用BK256。保持高精度累积与最终转换；不能为速度丢掉BF16大K情况下的稳定路径。
4. 起点已有半精度双tile、pipeline及master回退。新增FP32双tile路径后，清理被覆盖的旧分流，保留不规则形状、stride、bias广播与一般路径。
   `20261014-four-direct`第1轮直接替换了起点最后的v8 `run`，却把半精度转发到此前捕获的`_v6_dispatch`，导致半精度回到单N tile的旧v6路径；模型理由声称“半精度逐字节保持不变”与实际源码不符。新增FP32包装前应先保存当前完整v8入口，例如`_parent_run = run`，半精度及一般回退调用这个父入口，而不是跳过v8直接调用v6。
5. ReLU保留本机可编译的`maximum(x, x*0.0)`形式；不能默认替换成标量常量就一定可编译且语义相同。

## 失败方向

历史第1、2、4轮失败，第6轮半精度1024/2048的BK128候选通过但整体下降（约0.868×），没有采纳。优先复现FP32双N tile复用A与BK128，半精度沿用BK256，不再次把失败方向当作默认最佳配置。

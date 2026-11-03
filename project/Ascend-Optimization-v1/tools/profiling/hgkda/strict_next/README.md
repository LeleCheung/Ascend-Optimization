# 严格Triton五项继续攻关

从`9798acae1`的25项达标交付和`expm1_`新最佳0.646557×继续。原30项合同、默认确定性、容差与KGS完整调用walltime不变；候选全部设备路径严格Triton，原生CANN／来源明确Ascend C只作为参考。

新批次为`operators/campaigns/strict-next-20261103/`。`prepare.py`仅首次冻结五项起点及合同；`collect.py`在物理卡7项目锁内，用上一批已核验采集器采集最新最佳的FP16／BF16／FP32大输入配对指标和实际IR。新数据用于后续代码修改，不重复旧稿指标来推测新稿瓶颈。

原`strict-five-20261103`保持终态，不重启旧入口。新进程身份在`collector-process.json`，恢复必须比较PID及`/proc/<pid>/stat`启动时间；已有输出或未知任务不重发。tmux为`hanle-strict-next-20261103`；物理卡7所有设备任务串行，复用原KGS，不重启共享服务或镜像。

本轮目标仍为继续取得真实达标进展，报告落盘或采集结束不视作30项总目标完成。状态读取`progress.json`和实际进程；新成绩须经过完整KGS、补测、逐路径人工审查和实际Triton后端核验。

## 当前执行链

1. `collect.py`：最新最佳的三种精度成对采集，现已完成。
2. `prepare_fma.py`／`compile_fma.py`／`evaluate_fma.py`：原FMA默认常量导致流式半精度缓冲溢出，没有设备正式请求；`--typed`提案全部编译及数值检查通过，正式0.618492×低于0.646557×，不替换最佳。
3. `optimize.py`／`search.py`：连续两轮KG／CC＋DeepSeek，每轮5稿、引擎搜索预算2400秒，保留已提交完整评测的有限收尾窗口。按原源码与合同复用缓存，继续人工审查选中稿。
4. `inventory_modes.py`：单CPU只读冻结本机910B精度入口和编译选项，已完成；不初始化NPU。
5. `reference_modes_stage.py`：等待优化器及实际进程结束后持同一卡锁，分别在独立进程探测旧图Expm1默认／官方`high_precision`模式。只改变探测进程的选项，不修改共享KGS、设备确定性、候选或原合同。不支持或精度失败保留结果；局部成功仍不能直接赋予参考资格。
6. `prepare_buffer.py`／`compile_fma.py --configuration`：独立离线编译缓冲与连续分块方案。v1的FP16大块超UB；`--retain-fp16`生成v2，保留FP16，仅为BF16／FP32启用正常单阶段及UB复用选项，FP32改为8192单块。6项编译通过，`buffer-evaluator-process.json`等待搜索结束后在同锁下做一次正式评测和全部补测；排队期间若KG已生成同一AST则先核对原结果，不重复提交。

恢复时读取`optimizer-process.json`、`reference-modes-process.json`、`search-state.json`、`heartbeat.json`和实际工作区的`.kernelgen/run-process.json`。对照`/proc/<pid>/stat`中启动时间，不能只看PID是否存在。`search-progress.json`只列搜索与待审状态，最终是否新增达标仍按冻结验收合同。

`launch_optimizer.py`是已执行的一次性部署记录，不能重新执行。所有阶段均有一次性身份；已有请求、输出或未知执行结果时不重发。

Windows或910B可运行`python tools/profiling/hgkda/strict_next/audit.py`，只核验已结束附件和请求／源码身份，不使用NPU、不新增评测。910B使用宿主控制器的Python3.11。核验通过不代表活动搜索结束，旧最佳与达标数继续保留。

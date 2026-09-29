# 昇腾计时口径与 `do_bench` 已知问题

本文说明 KernelGen Server 在昇腾上的 profiler 与 Triton `do_bench` 计时边界，
并记录 2026-08-09 在实际 910B 环境中做过但已从 Server 删除的 NPU Event 实验。
结论先行：昇腾正式性能只使用严格 profiler；NPU Event 不能复现理想
`do_bench` 的 steady-state 语义，继续保留会造成结果误用，因此不再作为 Server
计时模式；目标环境自带的 `do_bench` 也不作为有效性能数据。

## 三种计时实际回答的问题

| 方式 | 回答的问题 | 是否包含 Host 发射间隙 | 当前用途 |
| --- | --- | --- | --- |
| `profiler` | 一次调用中有效 NPU kernel 的硬件执行时间总和 | 否 | 昇腾正式性能指标 |
| 已删除的 NPU Event 实验 | 设备空闲后，Event 之间一次完整调用占用的 stream 时间 | 容易包含 | 仅保留历史诊断证据 |
| Triton `do_bench` | 连续重复提交时每次调用的 steady-state Event 时间 | 理想实现中通常被排队隐藏 | 非昇腾正式计时；当前昇腾禁用 |

三者没有固定换算关系。尤其不能把按 profiler 反馈选出的 best kernel 改用
NPU Event 重测后，再把两个加速比当成同一指标比较。

## Stream、Event 与同步

Stream 是设备的一条有序任务队列。同一 stream 中，清缓存、Event 和 kernel 按提交
顺序执行；CPU 提交任务后通常立即继续，不等待 NPU 完成。

`torch.npu.synchronize()` 的含义是 CPU 停下来，直到此前提交到设备的工作完成。
Event 本身不是同步：`record()` 只是向 stream 提交一个时间标记。

可以用厨房订单理解这些角色：

- CPU 是服务员，负责提交订单；
- NPU 是厨师，负责执行订单；
- stream 是订单队列；
- kernel 是做菜任务；
- 清 L2 是清理灶台；
- Event 是厨师处理到该订单时盖下的时间戳。

同步清 L2 时，服务员提交“清理灶台”后等待厨师完成，随后才提交开始计时、做菜和
结束计时。厨师可能处理完开始时间戳后，还要等待服务员交来做菜订单：

```text
NPU：清 L2 -> start Event -> 空等 -> kernel -> end Event
CPU：提交清理 -> 等待完成 -------> 准备并提交 kernel
```

因此 Event 区间可能同时包含 NPU 等待 Host/Triton 提交的时间和 kernel 执行时间。

异步清 L2 时，服务员不等待清理完成，而是在厨师清理期间把后续订单提前排好：

```text
NPU：清 L2 ----------------> start -> kernel -> end
CPU：提交清理 -> 提交 start、kernel、end
```

只要这些命令位于同一 stream，异步不改变执行顺序，仍能保证先清 L2 再计时；区别
只是 CPU 不在清理期间阻塞。

### 连续调用不等于连续到达设备

从 Python 看，`do_bench` 已经连续调用：

```python
start.record()
fn()
end.record()
```

理想情况下，这三个命令应当无明显间隙地进入同一条设备 stream。但 Python 调用顺序
只描述 Host 侧控制流，不表示三个命令已经作为一个整体同时进入 NPU 队列。当前环境
中，Event 和 Triton kernel 在到达最终 stream 前经过不同的 Host 提交路径：

```text
Event  ------> Torch-NPU 提交路径 ---+
                                      +--> 最终 NPU stream
Triton kernel -> Triton task queue ---+
```

仍用厨房订单类比：服务员先把“开始计时”直接交给厨房，接着提交 Triton 做菜订单；
但这张订单还要经过翻译窗口和异步订单中心。厨房可能已经处理完开始时间戳，却仍在
等待 Triton 订单到达：

```text
服务员：提交 start -> 提交 Triton 订单 -> 提交 end
厨房：  收到 start -> 等待订单 --------> 做菜 -> 收到 end
```

“使用同一个 stream handle”只保证命令进入最终队列后按正确顺序执行，不能保证不同
Host 提交路径把它们无间隔地送达。理想 runtime 应正确协调并尽量隐藏这段提交差异；
当前 Triton-Ascend 3.5.1 在小 kernel 上没有提供稳定的 Event 结果，正是本文所说的
`do_bench` 问题边界。

## 为什么删除 NPU Event 计时

实验实现曾在每个样本中调用一次会同步的 `clear_l2_cache()`：

```text
每个样本：清 L2 并同步 -> start -> fn -> end
全部样本提交后：synchronize
```

每个样本开始前，队列都被排空。这种口径更容易暴露设备空闲时的 Host、Python、
Triton launcher 和 task queue 提交间隙，适合观察一次独立调用，但不等价于
steady-state `do_bench`。

理想 `do_bench` 的结构是：

```text
每个样本：异步清 L2 -> start -> fn -> end
全部样本提交后：synchronize
```

清 L2 仍逐样本执行，而且位于 start Event 之前，所以它的耗时不计入结果；CPU 则可
在设备清缓存时提前提交后续命令。要在昇腾上复刻这个结构，需要同一 stream 上真正
异步的 L2 flush。仅把现有同步 helper 调整调用位置不能实现相同语义。

即使结构完全一致，也不能自动修复当前环境的 Event/task queue 问题；那只会复现
现有昇腾 `do_bench` 的测量链路。

## 目标环境源码核对

2026-08-09 通过 Server Debug Job 在目标容器内读取实际安装代码，得到：

- Triton 报告版本为 `3.5.1`；
- `do_bench` 位于
  `/usr/local/python3.11.15/lib/python3.11/site-packages/triton/testing.py`；
- driver 为 `triton.backends.ascend.driver.NPUDriver`；
- 每轮创建或复用 192 MiB cache tensor，并在 start Event 前调用
  `cache.zero_()`；
- Event 来自 `torch_npu.npu.Event`；
- Triton launcher 默认启用 `TRITON_ENABLE_TASKQUEUE=true`；
- Torch Event 与 Triton kernel 的 current stream handle 相同，因而不是 Event
  记录到另一条 stream；
- 安装代码没有 `do_bench_npu`，也不读取 `TRITON_BENCH_METHOD` 或
  `INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE`。

官方当前文档提供 `TRITON_BENCH_METHOD=npu`，用于将 `do_bench` 切换到
`do_bench_npu`；但该目标环境版本没有这条实现，单独设置环境变量不会生效。官方也
已记录小 shape 下默认 Event 计时不准确、波动较大的问题：

- [Triton Ascend 环境变量说明](https://gitee.com/ascend/triton-ascend/blob/master/docs/ENVIRONMENT.md)
- [小 shape Event 计时问题](https://gitee.com/ascend/triton-ascend/issues/ICUV8F)

临时设置 `TRITON_ENABLE_TASKQUEUE=false` 也不是可用绕行方案。目标环境生成的
launcher 在加载时报告 `c10::UndefinedTensorImpl::_singleton` 未定义符号，说明
当前 Triton、Torch-NPU 与 C++ launcher 具有版本耦合；测试只发生在独立 Debug Job
中，没有修改 Server 环境。

## 最小实验

实验使用预分配输出的 FP32 elementwise add，元素数为 65536，分别调用
`torch.add(..., out=...)` 和一个 Triton add kernel。编译和首次 smoke launch 位于
正式采样前。

| 计时方式 | Torch add | Triton add | 观察 |
| --- | ---: | ---: | --- |
| profiler | 3.20 us | 4.03 us | 单个 NPU kernel 的硬件执行时间 |
| 已删除的 NPU Event 实验 | 62.06 us | 142.94 us | 包含设备空闲后的提交间隙 |
| 安装的 `do_bench` | 约 3.00 us | 中位约 28.7 us | Triton 样本在 10.58--83.04 us 间波动 |

同一进程连续调用短预算 `do_bench` 时，Torch 首次为 17.3 us、后续约 3.2 us；
Triton 从 44.7 us 下降到约 34 us。更长采样中 Torch 收敛到约 3 us，而 Triton
仍保持宽分布。Profiler 显示 Triton kernel 本体约 4 us，因此 `do_bench` 中多出的
时间和波动不能解释为 kernel 计算量。

这些结果与官方对小 shape Host/Event 计时问题的描述一致。现场证据指向
Torch-NPU Event、Triton 异步 task queue 和 Host 提交路径的组合，而不是缺少 L2
清理或 Event/kernel 位于不同 stream。

## KernelGen Server 采用的边界

1. 昇腾正式性能使用 `--timing profiler`；没有有效 profiler timing 时直接失败，
   不回退到 Event、`do_bench` 或 wall time。
2. Server 不再暴露 `--timing npu-event`；历史结果只能解释为“空闲 stream 下的
   单次调用 Event 时间”，不得与 profiler 或其他设备的 `do_bench` 直接横向排名。
3. 当前昇腾 Triton 3.5.1 的 `do_bench` 不用于性能结论。
4. 若后续验证厂商新版 `do_bench_npu`，应在独立测试分支和容器中完成版本组合、
   小/大 kernel、Torch/Triton、稳定性及 profiler 对照测试；不得原地替换正式环境的
   Torch、Triton、Torch-NPU 或 CANN。

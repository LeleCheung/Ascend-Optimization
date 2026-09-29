# Profiling 执行与采集设计

本文定义 KernelGen Server 的 profiling 执行语义。目标是让 Agent 只选择
`case_id`、candidate 和分析级别，由 Server/adapter 负责预热、采集边界、厂商
Profiler 和产物归档。

本文描述目标设计，并同时标出 2026-08-19 的当前实现差异；不能把“目标行为”当作
已经完成的功能。

## 1. 结论

Profiling 和性能计时是两类任务，不能共用同一套重复次数：

| 任务 | 预热 | 正式执行 | 结果用途 |
|---|---:|---:|---|
| correctness | 无要求 | 每个 case 一次 | 数值正确性 |
| timing benchmark | 需要 | 多次，统计分布 | 延迟和加速比 |
| kernel metrics | 需要 | 一次完整 candidate 调用 | 硬件计数器和瓶颈 |
| timeline trace | 需要 | 默认一次，必要时少量多次 | 调用序列和并发关系 |

“一次完整 candidate 调用”不等于“一个 device kernel”。一个 candidate 可能启动
多个 kernel，Profiler 应保留这次调用内的完整序列。Profiler 为收集互斥硬件计数器
而进行的内部 replay，不算 candidate 的额外正式调用。

默认策略：

- timing 独立执行多次测量，不从 profiling 报告发布 headline 性能。
- NCU、`msprof op` 的硬件指标只采集一次代表性调用或一次目标 kernel。
- 需要判断 run-to-run 波动时，使用新输入启动多份独立 profile job，不在同一个采集
  区间连续调用 candidate 100 次。

## 2. 三种容易混淆的重复

1. Runner warmup：触发 JIT 编译、缓存和稳定设备状态，不应进入正式采集。
2. Candidate measured invocation：业务程序主动执行的正式调用；kernel metrics 默认一次。
3. Profiler replay：NCU 或 msprof 为分批收集计数器而透明重放 kernel/range，由厂商
   工具管理。

当前 `ProfileOptions(warmup=10, iterations=100)` 把 timing 的多次测量习惯带进了
profiling。`iterations=100` 不应继续作为所有 Profiler 的统一默认值。

## 3. 主流工具的行为

### 3.1 Kernel metrics

Nsight Compute 的典型流程是先从 timeline 找到关键 kernel，再选择特定 launch
采集指标。它提供 kernel/range filter、`--launch-count` 和 replay；计数器无法在
一个 pass 内同时收集时，由 NCU 在内部重放目标 kernel。

AMD ROCProfiler 同样按 kernel dispatch 收集硬件计数器；指标过多时使用多 pass。
因此，程序侧重复执行 100 次不是获得完整计数器的必要条件。

### 3.2 Timeline

Nsight Systems 和 PyTorch/Ascend PyTorch Profiler 通常采集一个有界的 active
窗口。长训练可以记录若干 step；KernelGen 的单算子诊断默认只需要一次完整调用，
只有分析跨 iteration 的并发或波动时才增加到少量多次。

### 3.3 Timing

Triton `do_bench`、PyTorch Benchmark 等工具会在预热后持续测量，并返回均值、
中位数或分位数。这个重复过程属于 benchmark，不属于硬件指标 profiling。

## 4. NVIDIA NCU

外层由 NCU 启动 Python，不代表从进程启动就开始采集。目标命令是：

```bash
ncu --profile-from-start off ... python -m kernelgen_server.profiling.runner
```

Runner 的目标顺序是：

```text
生成 warmup 输入
→ candidate warmup N 次
→ device synchronize
→ 重新生成正式输入
→ cudaProfilerStart
→ candidate 正式调用一次
→ device synchronize
→ cudaProfilerStop
```

`--profile-from-start off` 使 NCU 在 `cudaProfilerStart()` 前保持未采集状态。因此
输入生成、JIT 编译和 warmup 不进入报告；start/stop 区间内一次 candidate 启动的
所有 kernel 都进入报告。

不要无条件添加 `--launch-count=1`。它可能只留下复合 candidate 的第一个 kernel。
正确边界是用 `cudaProfilerStart/Stop` 包围一次完整 candidate 调用，再让 NCU 对
区间内的 kernel 做必要 replay。

当前 KGS 已经使用 `--profile-from-start off` 和 CUDA start/stop，warmup 不会被 NCU
记录；问题是 start/stop 区间仍默认执行 100 次，需要改成一次。

## 5. 昇腾 msprof

KGS 当前存在两条不同路径，必须分别处理。

### 5.1 `msprof op`：单 kernel 硬件指标

`msprof op` 官方默认语义包括：

- `--warm-up=5`：在采集前预热并稳定频率；
- `--launch-count=1`：最多采集一个匹配算子；
- `--replay-mode=kernel`：由 msprof 对目标 kernel 做内部重放。

暂定将它作为昇腾 kernel metrics 的默认方案。Adapter 先选择一个 `case_id`，再让
pytest 的 candidate-only 模式只执行该 workload 和 candidate 一次；warmup、正式
采集次数和 replay 交给 `msprof op`：

```bash
msprof op \
  --kernel-name=<selected-kernel> \
  --warm-up=10 \
  --launch-count=1 \
  --replay-mode=kernel \
  python3 -m pytest benchmark/test_<operator>.py \
  --profile-only \
  --profile-case-id=<case_id>
```

这里各层职责固定为：

- `case_id` 选择一个 workload，而不是选择 device kernel；
- `--profile-only` 禁止 baseline、correctness assertion 和 benchmark，只调用注入的
  candidate；
- pytest 对 candidate 只调用一次，不再主动执行 `warmup + iterations`；
- `--warm-up=10` 由 msprof 对命中的目标 kernel 做预热；
- `--launch-count=1` 只生成一份正式 kernel 采集结果；
- `--replay-mode=kernel` 让收集不同硬件计数器所需的重复由 msprof 管理。

必须显式传入 `--kernel-name`，否则 msprof 默认命中进程遇到的第一个算子，可能采集
到输入处理或框架内部 kernel。复合 candidate 包含多个 kernel 时，KGS 先发现 kernel
列表，再为每个目标 kernel 分别执行一次上述命令；不能把“一个 case”和“一个
kernel”混为一谈。

支持 range replay 的 CANN 版本也可以用 MSTX 标记一次完整 candidate：

```text
warmup
→ synchronize
→ mstx.range_start("kgs_profile")
→ candidate 正式调用一次
→ synchronize
→ mstx.range_end(...)
```

配合 `--mstx=on --mstx-include=kgs_profile` 限定采集范围。使用前必须在实际 CANN
版本验证 MSTX Python API 和 range replay 能力。

当前 KGS 的详细 `msprof op` 已默认 `--launch-count=1`，但它启动的 runner 仍执行
10 次 warmup 和 100 次正式调用。实际只采集一个匹配 kernel，其余调用主要是额外
开销。实现暂定方案时应把 profile-only candidate 调用次数改为一次，并同步调整调用
次数核验和测试。

### 5.2 普通 `msprof`：应用 timeline 和 op summary

当前命令形态是：

```bash
msprof ... python -m kernelgen_server.profiling.runner
```

它没有像 NCU 当前路径那样的进程内采集开关，因此原始报告包含 runner 的 10 次
warmup 和 100 次正式调用。KGS 解析 `op_summary.csv` 时丢弃前 10 次并平均后
100 次，只是后处理排除；warmup 仍然存在于原始 trace 中。

目标方案二选一：

1. 使用 `torch_npu.profiler.schedule` 或显式 start/stop，在 Python 内完成 warmup，
   只令一个 active step 进入报告；
2. 需要单 kernel 详细指标时直接使用 `msprof op` 的 warmup、launch filter 和 replay，
   不再让普通 `msprof` 承担 benchmark。

普通 timeline 如需观察多轮行为，可以显式配置少量 active iterations；默认不使用
100 次。正式昇腾性能结果仍由项目规定的严格 `torch_npu.profiler` timing 流程产生。

### 5.3 为什么不能用两个 pytest 进程代替采集边界

下面的执行方式只能近似稳定设备频率，不等价于 NCU 的
`--profile-from-start off`：

```text
pytest warmup 100 次        # 进程 A
→ 进程退出
→ msprof pytest 正式采集    # 进程 B
```

第二个 pytest 会重新创建进程，因此 NPU context、stream、allocator、PyTorch/pytest
进程内缓存、内存中的 JIT/autotune 状态、Tensor 和 fixture 都会丢失。可能保留的只有
磁盘编译缓存，以及短时间内的设备频率和温度；启动 msprof 和新进程的间隔还可能让
频率状态发生变化。第二个进程的首次模块加载、输入初始化和首次 candidate 调用仍会
进入普通 msprof 报告。

正式方案必须在同一个 pytest 进程内完成：

```text
生成 warmup 输入
→ warmup N 次
→ synchronize
→ 重新生成正式输入
→ 开始采集
→ candidate 正式调用一次
→ synchronize
→ 停止采集
```

昇腾实现按以下顺序选择：

1. 普通 timeline 使用 `torch_npu.profiler.schedule` 或显式 start/stop，获得真正的
   进程内采集边界。
2. 单 kernel 指标使用 `msprof op --warm-up=N --launch-count=1`。
3. 复合调用在 pytest 中用 MSTX 标记正式区间，再用
   `msprof op --mstx=on --mstx-include=kgs_profile` 选择该区间。
4. 只有目标 CANN/pytest 无法提供进程内边界时，才允许使用两个独立 pytest 进程
   作为降级方案；结果必须标明“跨进程预热”，不得宣称与正式采集等价。

## 6. 输入与 mutation

Warmup 和正式采集不能复用同一份可变输入。对于 `addmm_` 等 in-place 算子，复用
输入会让正式调用看到已被 warmup 修改过的状态，既改变语义，也可能改变访存行为。

Runner 必须采用以下任一方案：

- warmup 使用独立输入，结束后从 workload 重新物化正式输入；
- 保存不可变 CPU base，在 warmup 后重新复制到目标设备；
- adapter 明确定义等价的输入恢复逻辑。

重新物化应发生在采集开始前。相同 shape/dtype 的编译缓存可以复用，但正式调用的
Tensor 内容、布局、alias 和 mutation 前状态必须与原 workload 一致。

## 7. Adapter 和 Server 的职责

Agent 只提交 candidate、`case_id` 和 profile level，不需要了解 NCU、msprof 或
MSTX 参数。每个 adapter 的 `build_profile_command` 负责生成一次 case 的精确运行
命令；Server 负责：

- 设备 slot、timeout 和进程隔离；
- backend 对应的采集边界；
- 厂商 Profiler 调用和内部 replay 参数；
- 原始 artifact、统一 summary 和错误状态；
- profiling 结束后的设备同步与资源释放。

建议把配置语义从模糊的 `iterations` 拆成：

- `warmup_invocations`：Profiler 之外的预热次数；
- `captured_invocations`：采集区间内的完整 candidate 调用次数；
- backend options：`launch_count`、replay 和 trace active window 等厂商配置。

对于 kernel metrics，`captured_invocations` 默认固定为 1；timing 的重复预算继续由
evaluation/timing 子系统管理，不复用该字段。

## 8. 验收要求

修改完成后至少验证：

1. NCU 报告不包含 warmup，只包含一次完整 candidate 调用产生的 kernel。
2. `msprof op` 每个选中 kernel 只生成一份正式采集结果，runner 不再额外执行 100 次。
3. 普通 msprof/Ascend PyTorch trace 的原始报告不包含 warmup，或明确标记为旧兼容
   路径且不用于分析。
4. 单 kernel、复合多-kernel、in-place/mutation 和随机输入各验证一个 case。
5. Profiling 结果不覆盖 eval 的 correctness、headline timing 或加速比。
6. Profile job 完成、失败或超时后，设备 slot 都能恢复为 healthy/available。

## 9. 参考资料

- [Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [Nsight Compute replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Nsight Systems capture range](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [PyTorch Profiler schedule](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)
- [ROCProfiler counter collection](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/api-reference/counter_collection_services.html)
- [Ascend `msprof op`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/devaids/optool/atlasopdev_16_0082.html)
- [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/profiling/atlasprofiling_16_0033.html)
- [Ascend MSTX range](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/ODtools/Operatordevelopmenttools/atlasopdev_16_0115.html)

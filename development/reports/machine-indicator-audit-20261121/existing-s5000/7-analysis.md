# 摩尔线程 MTT S5000 Profiler 实测分析

镜像 `vllm0.24.0-mthreads-musa5.2.0:2.2.0-0.3.0`，MUSA 5.2.0 + MUPTI 1.3.0，单卡 GPU7，
2026-11-20 实测。所有数字来自 `raw/runs/*` 与 `results.csv`。

## 0. 结论摘要

| 能力 | 结论 | 证据 |
|---|---|---|
| 时间线追踪（kernel/memcpy/memset/API/sync） | **可用**，自写 MUPTI 客户端 204 kernel + 1 memcpy + 4 memset + 229 runtime + 7 sync，丢 0 条 | `runs/trace` |
| 内核属性 | grid/block/寄存器/静态共享内存/correlationId 齐全；**内核名是 mangled 符号**，需 `c++filt` | `runs/trace/trace.jsonl` |
| 多流并发可见性 | 可用，2 stream × 6 matmul，重叠 17.15 ms | `runs/streams` |
| 启动/结束排队时间 | 可用（`muptiActivityEnableLatencyTimestamps`），queued→start 中位 35 µs，submitted→start 12.6 µs | `runs/latency` |
| 采集窗口控制 | **失效**：`musaProfilerStart/Stop` 期间的 3 个内核全部被记录，窗口外的 kernel 也照记 | `runs/gated` |
| 追踪开销 | 0.44 ms copy 内核上无可测开销（−0.05%） | `runs/overhead` |
| 硬件计数器 / 指标（DRAM、SM、tensor pipe…） | **不可用**：metrics API 只有骨架，59 个 metrics/event 相关函数未导出；`muptiProfilerBeginSession` 返回 `MUPTI_ERROR_NOT_SUPPORTED(27)` | `runs/api/api_check.txt` |
| PC sampling | **不可用**：只有 Enable/Disable/Start/Stop/SetConfig，取数、枚举 stall reason、GetConfiguration 全部缺失；`muptiPCSamplingEnable` 返回 `INVALID_PARAMETER(1)` | `runs/api` |
| 范围/区间指标（range profiler） | **不可用**：`muptiProfilerSetConfig` 只接受未公开的 `PfmDumpConfig`（`pfm.h`，需 coreId/wrapperType/instanceType/groupId），CounterDataImage 分配函数全缺 | `raw/hdrs/pfm.h` |
| NVTX 类区间标注（MTTX） | **镜像内无库**：无 `libmuttx.so`、无 mttx 头；MUPTI 仅留 `muptiActivityPush/PopExternalCorrelationId` | `runs/allkinds` |
| `torch.profiler` | **可用**，但必须用 `ProfilerActivity.MUSA`（= `PrivateUse1`）；`CUDA` 枚举会因 `torch.cuda.is_available()==False` 被静默禁用 | `runs/torch` |
| 设备遥测 `mthreads-gmi` | **不可用于负载判读**：持续负载 300 次采样 util 恒 0%、FB Used 恒 0 MiB、功耗 408.7–409.8 W、温度 37–38 °C；只有时钟 1750 MHz 稳定 | `runs/telemetry` |

一句话：**摩尔线程这条栈的 profiler 是"只有时间线、没有计数器"的形态**——MUPTI activity 够写一个
nsys 级别的 trace（内核/拷贝/API/同步/多流），但 metrics、PC sampling、区间计数全部不可用；
`torch.profiler`/kineto 通路是现成可用的高层入口，而 `mthreads-gmi` 的利用率/显存字段在本机
（KVM + 驱动 3.3.5-server）实际是坏的，不能用它判断"卡有没有在跑"。

## 1. API 面：声明 119，导出 39

`libmupti.so.1.3.0` 只导出 39 个符号，头文件声明 119 个（`raw/mupti/declared.txt`、
`raw/mupti/symbols.txt`，`api_check` 用 `dlopen/dlsym` 逐条核对）。缺的 80 个集中在四族：

| 缺失族 | 代表函数 | 后果 |
|---|---|---|
| events/metrics | `muptiEventGroupCreate`、`muptiMetricGetValue`、`muptiDeviceEnumMetrics`（全族 59 个） | 无法按事件/指标采集硬件计数器 |
| profiler/counter data | `muptiProfilerCounterDataImageCalculateSize/Initialize`、`muptiProfilerGetCounterAvailability`、`muptiProfilerPushRange/PopRange` | 无法分配 counter image、无法做用户区间 |
| PC sampling | `muptiPCSamplingGetConfigurationAttribute`、`muptiPCSamplingGetNumStallReasons`、`muptiPCSamplingGetStallReasons`、`muptiPCSamplingGetData` | 无 stall reason 枚举与取数，PC 采样闭环不成立 |
| activity 属性 | `muptiActivityGetAttribute/SetAttribute`、`muptiActivityEnableContext/DisableContext`、`muptiActivityFlushPeriod` | 无法配置 buffer 策略、无法按 context 过滤 |

实测返回码（`runs/api/api_check.txt`）：

```
muptiGetVersion                  = 0  (version=14)
muptiGetDeviceId                 = 0  (dev=0)
muptiGetContextId                = 3  MUPTI_ERROR_INVALID_CONTEXT
muptiGetStreamIdEx               = 0  (stream=2)
muptiProfilerInitialize          = 0
muptiProfilerBeginSession        = 27 MUPTI_ERROR_NOT_SUPPORTED
muptiProfilerSetConfig(PfmDumpConfig, 0 instances) = 0   ← 接受"空配置"，不校验
muptiPCSamplingSetConfigurationAttribute = 1 MUPTI_ERROR_INVALID_PARAMETER
muptiPCSamplingEnable            = 1  MUPTI_ERROR_INVALID_PARAMETER
muptiActivityEnable(kind=1..54)  = 0  ← 54 种全部返回成功（静默接受）
musaProfilerStart / Stop         = 0 / 0
```

关键陷阱：**`muptiActivityEnable` 对全部 54 种 kind 都返回 SUCCESS**，但只有 5 种真正产出记录
（见 §2）；`muptiProfilerSetConfig` 对伪造的 PFM 配置也返回 SUCCESS。用"返回值判成功"的客户端
会得到"配置成功、永远没有数据"的假象。

## 2. Activity 追踪：哪些 kind 真的有数据

`runs/allkinds`（同时使能 kind 1..54，跑 2 kernel + 1 H2D + 1 memset）：

```
kind=1  MEMCPY            1 条
kind=2  MEMSET            1 条
kind=5  RUNTIME           6 条
kind=10 CONCURRENT_KERNEL 2 条
kind=38 SYNCHRONIZATION   2 条
（其余 49 种：0 条；marker/metric/PC_SAMPLING/MTLINK/GRAPH_TRACE/PCIE 均无记录）
```

补充：`MUPTI_ACTIVITY_KIND_KERNEL(3)` 使能后产出的是 `CONCURRENT_KERNEL(10)` 记录，二者不要重复统计。
`SYNCHRONIZATION` 记录带 `type/ctx/stream/corrId`，可复原 host 同步点，是 nsys 里"同步"泳道的数据源。

全量追踪（`runs/trace`，7 个负载串行）：

| 指标 | 值 |
|---|---|
| kernel 记录 | 204（chain200 贡献 200 条，单条均值 1.58 µs） |
| memcpy / memset / runtime / sync | 1 / 4 / 229 / 7 |
| 丢记录 | 0 |
| trace 跨度 | 153.5 ms（含 host 侧 256 MiB 分配与 7 次 device 同步） |
| kernel 忙碌合计 / 占空比 | 3.85 ms / 2.5% |
| 内核名 | `_Z6k_copyPKfPfi` 等 mangled 名；无 demangle，也无 source locator 记录 |

单内核耗时（同一 trace 内，与无 profiler 基线一致）：`k_mm` 1.759 ms、`k_stride` 0.722 ms、
`k_add` 0.591 ms、`k_copy` 0.465 ms；256 MiB pageable H2D 在 trace 里 13.2 ms（≈20 GB/s）。

### 2.1 flush 语义（写工具必须知道）

| 调用 | 行为 |
|---|---|
| `muptiActivityFlushAll(0)` | 非阻塞：**约 3–5 s 后**才回调，之后进程内看不到新记录；短命进程会在退出时才收到 |
| `muptiActivityFlushAll(1)` | 阻塞：返回前回调完成，记录立即可用（推荐） |
| 不 flush | 后台线程约 3 s 周期投递，或 8 MiB buffer 填满时投递 |

证据：`runs/api/flush_nonblocking.txt`（`t_flush0` 与回调时间戳相差 ≈5 s）、
`flush_blocking.txt`（`t_flush1` 内 `rec=5`）、`flush_none.txt`（sleep 3 s 后回调）。

### 2.2 `musaProfilerStart/Stop` 不做采集门控

`runs/gated`：窗口外 `k_copy`、窗口内 `k_add`、`musaProfilerStop` 之后的 `k_mm` 三条记录全部被采到
（`captured_copy_before_start=1, captured_add_inside_window=1, captured_mm_after_stop=1`）。
要在 MUSA 上做 "profile-from-start off"，只能自己 `muptiActivityDisable/Enable`，不能靠
`musaProfilerStart/Stop`（这两个 API 在 `libmusart.so` 里存在且返回 0，但没有 gate 效果）。

### 2.3 多流与排队时间

`runs/streams`：stream id 2/3 各 6 个 matmul，成对重叠合计 17.15 ms，说明时间线能反映并发。
`runs/latency`：3 个背靠背 `k_mm`，kernel 占空比 91%，queued→start 中位 35.05 µs、
submitted→start 12.58 µs；而在 200 次突发 launch 的 `chain200` 场景，queued→start 中位升到 855.9 µs
（队列积压），submitted→start 仍 10.2 µs——后者才是设备侧真实启动延迟。

## 3. `torch.profiler`：可用，但入口是 MUSA/PrivateUse1

- `torch.cuda` 未被 torch_musa 接管：`torch.cuda.is_available()==False`、`torch.cuda.device_count()==0`、
  `torch.version.cuda is None`；`ProfilerActivity.CUDA` 会被 `autograd/profiler.py` **静默降级为纯 CPU**，
  只打一条 `UserWarning: CUDA is not available, disabling CUDA profiling`。
- 正确写法：`ProfilerActivity.MUSA`（等价 `ProfilerActivity.PrivateUse1`），此时表格出现
  `Self MUSA / MUSA total / MUSA time avg` 列，chrome trace 里是 `cpu_op` + `kernel` + `gpu_memcpy` 事件。
- 实测（`runs/torch/meta.jsonl`）：

| 用例 | trace 事件 | 关键设备侧条目 |
|---|---|---|
| basic（3×1024³ matmul） | 121 | `musa_asm_sgemm_nn_256_128x128_epilogue_robust64` 517.4 µs（3 次，172 µs/次） |
| options（record_shapes+profile_memory+with_stack） | 197 | 同上 118.8 µs（2×512³），出现 `[memory]` 条目 |
| schedule（wait1/warmup2/active3） | 129 | `ProfilerStep*` 3 步，sgemm 97.0 µs |
| gems（FlagGems softmax+add） | 146 | Triton `softmax_kernel_inner` 86.9 µs；muDNN `kernel_unary<AddOp>` 47.3 µs |
| copy（64 MiB H2D+D2H） | 97 | `Memcpy1 HtoD (Pageable -> Device)` 2.81 ms（≈22.8 GB/s）、`DtoH` 1.77 ms |

- 与 MUPTI 原生 trace 对照：MUPTI 侧 1024³ 朴素 `k_mm` 1.76 ms，torch 侧走 muBLAS sgemm 仅 0.172 ms
  （差 10×），说明两条通路测的是不同实现，交叉验证时不能混用。
- torch 侧内核名是**已 demangle** 的 C++ 名（`void at::native::...distribution_elementwise_grid_stride_kernel<...>`），
  与 MUPTI 的 mangled 名不一致，写工具时需要统一。

## 4. 性能基线（无 profiler，5 次中位）

| 负载 | 耗时 | 推导 |
|---|---|---|
| copy 256 MiB（读+写 512 MiB） | 0.4417 ms | 1.157 TB/s = 72% × 1600 GB/s |
| add 256 MiB（读 512+写 256） | 0.5879 ms | 1.307 TB/s = 82% |
| stride（步长 8，读 32 MiB+写 32 MiB） | 0.7022 ms | 访存不合并，明显劣化 |
| matmul 1024³（16×16 tile 朴素实现） | 1.6476 ms | 1.30 TFLOPS（非 muBLAS） |
| chain200（200 个 1-block 内核） | 0.5630 ms | 单次 launch 2.8 µs |
| memset 256 MiB | 0.1968 ms | 1.30 TB/s（纯写） |
| memcpy H2D 256 MiB（pageable） | 19.76 ms | 12.97 GB/s（未锁页，PCIe 5.0 x16 上限远高） |

复现性：copy/add/matmul CV ≤ 0.9%，chain200 CV 6.3%（launch 抖动），说明单卡独占、频率稳定。

## 5. 设备遥测：`mthreads-gmi` 的利用率/显存字段不可信

12 s 持续负载（matmul+add 混跑，26 轮、GPU 基本满载，宿主进程表可见 `./bench_tracer 822MiB`）：

| 采样源 | 样本 | GPU util | FB Used | 功耗 | 温度 | 图形时钟 |
|---|---|---|---|---|---|---|
| container 内（`-i 0`） | 15 | 恒 0% | 恒 0 MiB | 409.35 W | 38 °C | — |
| host（`-i 7`） | 300 | 恒 0% | 恒 0 MiB | 408.7–409.8 W | 37–38 °C | 恒 1750 MHz |

- 同一时刻宿主 `mthreads-gmi` **进程表**却列出 `./bench_tracer 822MiB`——即"进程占用"能算，
  但"设备级 util/FB Used"恒 0，属驱动 3.3.5-server 在该 KVM 机型上的字段缺失/冻结。
- 容器内 `mthreads-gmi` 的进程表固定 `No running processes found`（PID 命名空间隔离），
  因此容器内既看不到自己的进程，也拿不到有效 util——**不能用容器内 gmi 做负载判读**。
- 唯一可用的是时钟（P0/1750 MHz）与 ECC/温度/功耗读数，而功耗基线 409 W、满负载不动，
  说明这些传感器数据也未随计算负载更新。
- 结论：本机型上"卡是否有人在用"只能靠**宿主** `mthreads-gmi` 的进程表 + 内核占用，不能看 util 百分比。

## 6. 对 profiling-survey 的含义

1. 摩尔线程提供的是 **CUPTI 形态的 activity API**（nsys 数据面），没有 ncu 形态的计数器面；
   本镜像未带任何前端工具（无 `nsys/ncu/musaprof` 类比物），要出图必须自写客户端或接 `torch.profiler`。
2. `torch.profiler` 是**最低成本可用入口**：一行 `ProfilerActivity.MUSA` 即可拿到内核/拷贝时间线，
   FlagGems(Triton) 与 muBLAS/muDNN 内核都能显示；适合"应用级归因"，无法回答 SM/DRAM/tensor 利用率问题。
3. 想做"计数器级"分析，只能走厂商私有 PFM 通道（`PfmDumpConfig` 需 coreId/wrapperType/instanceType/groupId），
   本镜像没有生成该配置的工具，等同于**当前版本不可用**。
4. MUPTI 的 flush/门控语义与 CUPTI 不同（异步 ~5 s、Start/Stop 不 gate），跨厂商工具移植时必须适配。

## 7. 未覆盖 / 建议

- **未做 vLLM 端到端 profiling**：镜像内无本地模型，宿主也无可用权重目录，且外部下载未验证；
  本次只验证了 profiler 本体与 torch/Triton/muBLAS/muDNN 内核可见性。若要补，建议
  `LLM(..., profiler_config=...)` 或 `torch.profiler` 包 `engine.step()`，并用本次的
  `MUSA_VISIBLE_DEVICES=7` + 单卡映射方式。
- **未做 MTTX 区间**：镜像无 `libmuttx.so`，无法注入 marker；若后续镜像带 MTTX，可用
  `MUPTI_ACTIVITY_KIND_MARKER` 直接复用本仓库的 `bench_tracer` 记录分支。
- **未做长时稳定性**：最长单次用例 25 s；`chain200` 场景 204 条记录/0 丢弃，未触及 8 MiB buffer 上限。
- **建议复核项**：`muptiGetContextId` 返回 `INVALID_CONTEXT`、`muptiProfilerSetConfig` 接受空配置——
  这两个行为在 MUSA 后续版本可能被修正，引用时注明版本（MUPTI 1.3.0 / MUSA 5.2.0）。

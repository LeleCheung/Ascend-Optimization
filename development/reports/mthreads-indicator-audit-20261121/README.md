# 摩尔线程指标比较复核

2026-11-21，在线复核摩尔线程官方文档。比较范围沿用当前主表：厂商公开信息与工具能力的可获得性，允许不同公开型号实例；型号、版本和精度分别记录。本轮生成复核建议，综合Excel与PDF维持上一版，便于核对同事原结论。

后续用户补充范围为智源提供的指定机器。已只读查询S5000并记录其驱动、MCU及MUSA版本，详见[指定机器复核](../machine-indicator-audit-20261121/README.md)。本页跨公开型号的L1/SPM与峰值建议仅用于厂商概览；S5000对应结论以指定机器复核为准。

结论：原表有漏项，绿色也有增加空间。原表20项已包含11项√、2项部分、7项×，访存量、算力利用率、带宽利用率、流水利用率和L1/L2缓存命中率原本已有√。本轮建议调整5项后为13项√、4项部分、3项×。寄存器和公开峰值有直接新依据；L1/SPM涉及用户已扩展的指标定义；空泡采用与NVIDIA相同的采样观察口径。

| 指标 | 原表 | 复核建议 | 官方依据与范围 |
| --- | --- | --- | --- |
| host launch时间 | √ | √ | msys追踪MUSA Runtime/Driver API并输出musa_api_trace、musa_api_sum。 |
| device kernel时间 | √ | √ | msys输出kernel执行统计；mcu官方报告显示Duration。 |
| 并行度 | √ | √ | msys支持多流并发分析；mcu LaunchStats提供Grid/Block、占用率等。 |
| 计算量 | × | ×，本轮保留 | 已有Roofline、浮点吞吐和指令数量相关图表；本轮尚未取得明确说明总FLOP计数、支持精度及计算方式的工具指标定义。 |
| 访存量 | √ | √ | mcu Memory Tables列出读写Bytes、Sectors、Throughput；原表已有√。 |
| 算力利用率 | √ | √ | SpeedOfLight提供相对峰值的计算吞吐利用率；原表已有√。 |
| 带宽利用率 | √ | √ | MemoryWorkloadAnalysis与SpeedOfLight提供内存吞吐和利用率；原表已有√。 |
| 寄存器数量/大小 | 部分 | **√** | Runtime API的regsPerBlock、regsPerMultiprocessor明确为可用32-bit寄存器的最大数量，musaGetDeviceProperties可查询；采用应用可见的通用寄存器规格口径。 |
| L1/SPM容量/带宽/访问粒度 | × | **√，按SPM口径** | v4.3.x硬件指南MP_10/MP_21列出每MP 28 KB局部存储器、16 bank、每bank每周期32 bits；性能优化章节明确S80的28 KB为shared memory。bank字宽/通路吞吐可记录；16×32/8=64 B/cycle为聚合理论值。 |
| L2 cache容量/带宽/cache line size | 部分 | 部分 | Runtime API公开l2CacheSize；v5.2.0针对S5000列128 B L2缓存行。L2理论峰值带宽仍缺明确数值，保留部分。 |
| DRAM容量/带宽 | √ | √ | S4000规格书48 GB、768 GB/s；原表已有√。 |
| 4/8/16/32 bit矩阵算力 | × | **部分** | S4000公开TF32 50 TFLOPS、FP16/BF16 100 TFLOPS、INT8 200 TOPS；4bit峰值仍缺依据。TF32精度单列。 |
| 4/8/16/32 bit向量算力 | × | **部分** | S4000规格书明确“算力（向量）FP32/FP16 25 TFLOPS”；4bit与8bit向量峰值仍缺完整规格。 |
| core数量 | √ | √ | S4000公开8192 MUSA核心；MP/MPX/算术核心数量按各自层次记录；原表已有√。 |
| 功能部件流水线利用率 | √ | √ | mcu提供MP及内存单元吞吐/活跃周期；原表已有√，各功能部件的完整细分仍受版本与设备支持范围影响。 |
| L1/SPM访存统计 | √ | √ | 官方列L1 Hit Rate及Shared Memory Bank Conflicts；原表已有√。 |
| L2缓存命中率 | √ | √ | 官方明确列出L2 Hit Rate；原表已有√。 |
| 指令热点 | × | ×，本轮保留 | 本轮已读工具指南与报告附图尚未提供明确的动态PC热点及源码映射能力说明。 |
| 流水线依赖 | × | ×，本轮保留 | 本轮尚未取得官方逐指令依赖图、scoreboard等待分类或对应采样字段定义。 |
| 流水线空泡 | × | **部分，限采样观察** | 官方用户指南附图含PM Sampling时间线，展示MP Active Cycles及MP/DDR/LLC/L2/L1 Throughput。可观察采样粒度下的活跃程度和空闲趋势；内部流水逐周期空泡及其原因仍需其他证据。此建议与NVIDIA采用采样级空闲观察的部分判定对齐。 |

SPM一项的√仅表示该类软件管理片上存储已取得容量、bank通路带宽与bank字宽信息。bank字宽作为存储组织/访问粒度记录，具体指令传输长度和事务粒度分别解释。若沿用原表仅比较L1 cache容量、带宽和cache line size的定义，此项应为部分，仍需补齐L1 cache带宽和cache line资料。MP_10/MP_21、S4000和S5000的数据分别标明，组合材料用于厂商信息可获得性比较。

现有材料支持“摩尔线程系统与算子Profiler较完整，硬件参数和细粒度动态指令分析仍有缺项”。原表已有的绿色项占一半以上，工具体系完整也包含这些已标绿项；本轮增加绿色主要来自寄存器规格与SPM口径。峰值规格的两处叉可明确调整为部分。

同事历史调研所用工具版本与时间尚未取得，本次结论是依据当前在线资料更新。v1.3.0发布说明明确增加Memory Chart和Memory Tables；历史v1.2用户指南目前也出现相同图表内容，因此历史页面显示的内容与实际功能引入版本应分开解释。报告未据此追认过去某一版本的完整支持范围。

官方来源：

1. [Runtime API：musaGetDeviceProperties](https://docs.mthreads.com/musa-sdk/version-5.2.0/libraries/core_api/runtime_api_reference#musagetdeviceproperties)：寄存器32 bit和数量查询、共享内存容量、L2容量。
2. [v4.3.x硬件架构](https://docs.mthreads.com/musa-sdk/version-4.3.x/programming_guide/Chapter02/)：MP_10/MP_21局部存储器、bank带宽、L1/L2容量。
3. [v4.3.x性能优化](https://docs.mthreads.com/musa-sdk/version-4.3.x/programming_guide/Chapter09/)：S80每MP 28 KB shared memory。
4. [MTT S4000规格书](https://docs.mthreads.com/s4000/s4000-doc-online/product_specifications/)：明确区分向量和张量峰值。
5. [v5.2.0内存优化](https://docs.mthreads.com/musa-sdk/version-5.2.0/programming_guide/performance_tuning/memory_optimization)：S5000的128 B L2缓存行。
6. [Moore Perf Compute性能分析](https://docs.mthreads.com/mooreperf/mooreperf-doc-online/moore_perf_compute/profiling/)：相对峰值利用率、L1/L2命中率、访存分析。
7. [官方用户指南附图](https://docs.mthreads.com/mooreperf/mooreperf-doc-online/moore_perf_compute/historic_version/v1.2/user_guide/)：报告、Roofline、访存表格与PM Sampling。[PM采样附图](https://docs.mthreads.com/assets/images/page_details-5547142891d8c5eb155aab85b1a992ec.png)。
8. [Moore Perf System用户指南](https://docs.mthreads.com/mooreperf/mooreperf-doc-online/moore_perf_system/user_guide/)：API/kernel追踪与结构化输出。
9. [Moore Perf Compute v1.3.0发布说明](https://docs.mthreads.com/mooreperf/mooreperf-doc-online/moore_perf_compute/releasenote/v1.3.0_release_note/)：Memory Chart/Tables新增功能及设备范围。

抓取的官方HTML、正文、截图和索引均保存在本目录。`audit.json`保存20项原判定、复核建议与证据定位；`verification.json`记录原表和现交付Excel/PDF的保留检查。

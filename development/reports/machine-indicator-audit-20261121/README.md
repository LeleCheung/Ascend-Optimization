# 指定机器与报告来源复核

2026-11-21。根据用户提供的来源清单，读取本地SSH配置并只读连接目标机器，核实摩尔线程、昇腾工具和字段。综合Excel与PDF维持上一版。本轮无安装、无新kernel评测、无profiling采集；运行设备属性查询、工具清单查询、读取既有报告与官方文档。

## 机器与版本

本地配置为`C:/Users/lele1/.ssh/config`，包含baai-h20、baai-910b、baai-p800、baai-bw1000、baai-mc550、baai-tianshu、baai-s60、baai-s5000等入口。用户本次提到910D，当前配置中的昇腾入口为baai-910b，实机8张卡均显示910B4-1。baai-s60容器表明是燧原环境，与本次摩尔线程S5000入口分别记录。

| 入口 | 本轮查询结果 | 已确认的软件 |
| --- | --- | --- |
| baai-910b | Ascend910B4-1，8卡，aarch64；npu-smi 25.2.0 | tle_yy容器安装CANN 9.0.0；存在Ascend910B4模拟器库与Ascend910B4-1平台配置。 |
| baai-s5000 / worker33015 | MTT S5000，8卡，每卡81920 MiB；x86_64 | 宿主Driver 3.3.5-server、MCU 1.3.1-413aaf3c6；fgmig-mthreads容器MUSA 5.2.0、torch_musa 2.9.1。宿主与所查容器PATH内均未发现msys。 |

查询记录见`ascend-host-inventory.json`、`ascend-sdk-version.json`、`mthreads-host-inventory.json`、`mthreads-host-tools.json`、`mthreads-container-tools.json`。

## 报告来源逐条判断

| 编号 | 复核判断 | 对指定机器的适用范围 |
| --- | --- | --- |
| H1 | Runtime/Host API与设备task耗时的来源成立。 | 本机msprof命令包含runtime-api、task-time；既有OpBasicInfo与原配对采集支持设备任务耗时。Host调用和device执行分别记录。 |
| H2 | OpBasicInfo、PipeUtilization、L2Cache、Memory成立。 | 本机既有GN/GQA报告具备这些表；Block Dim是逻辑启动规模，驻留占用率另需专门指标。 |
| H3 | FLOPs与带宽字段有官方定义和本机直接字段。 | op_summary官方字段为`*_cube_fops`、`*_vector_fops`；msopprof算子表实际为`aic_cube_fops`、`aiv_vec_fops`。GN和GQA样本有效。`cube_utilization(%)`有官方定义，分母与cycle口径依工具定义解释；现有样本主要包含各流水cycle占比，完整所有精度FLOPs峰值利用率仍需单列。 |
| H4 | 热点、源码/指令耗时、GPR Count和GM Process Bytes有官方定义。 | Atlas A2支持范围与910B材料对应；GPR Count用于寄存器使用观察，完整物理寄存器规格和位宽另需匹配来源。 |
| H5 | 访存通路、带宽、利用率和Roofline观点成立。 | 本机已有通路数据及带宽利用率。观测带宽与工具理论模型、产品硬件峰值规格分别记录；Performance Ratio的具体指标定义仍应逐字段核对。 |
| H6 | 原生指令流水、调用栈和同步连线能力成立。 | 官方说明明确Atlas A2展示SET_FLAG/WAIT_FLAG等待；本项目已有910B目标的trace.json、instr_exe.csv、code_exe.csv与后重放。HG-KDA进一步组织区域、量化等待/空闲、关联IR和真机结果。 |
| M1 | msys官方功能描述成立。 | 本轮目标机和所查容器PATH内均未发现msys。该机器既有MUPTI activity与torch.profiler记录已覆盖API/kernel/memcpy时间线及多流；工具入口和实测路径分别写清。 |
| M2 | Memory Chart、Bytes/Sectors/Throughput、L1/L2 Hit Rate有直接文档依据。 | 本机MCU 1.3.1确有相应sections与metrics清单。来源引用historic v1.2页面需修正版本说明：v1.3.0发布记录明确新增Chart/Tables，历史网页也出现这些内容，页面目录名与功能引入版本有差异。 |
| M3 | SpeedOfLight、Roofline、Green Context的MP数量描述成立。 | 本机MCU提供launch__mp_count等字段；本轮Runtime直接查询物理MP数量为60。Green Context分配资源与全设备资源分别记录。 |

H1—H6原始hiascend链接本轮均返回HTTP 200的HuaweiCloudWAF验证页面，正文未从这些URL成功取得。对应核心能力通过Ascend官方开源msprof/msopprof固定提交文档与本机既有数据交叉核实；这两种来源层次在本报告中分别记录。M1—M3本轮直接读取官方正文成功。抓取状态见`source-fetch.json`及`official-source-fetch.json`。

## S5000：当前确认的信息

### 硬件属性

本轮从容器已安装MUSA Runtime调用`musaDeviceGetAttribute`，枚举数值来自该机`driver_types.h`，全部查询返回成功：

| 属性 | 本机值 | 解释 |
| --- | --- | --- |
| 每MP可用寄存器数 | 131072 | 安装头文件明确为32-bit寄存器，即每MP应用可见512 KiB。 |
| 每block最大寄存器数 | 131072 | 同一32-bit单位；与每MP资源共享关系按Runtime定义。 |
| 每MP共享内存 | 196608 B，即192 KiB | 只确认容量；S5000匹配的完整SPM通路带宽与访问粒度尚需补齐。 |
| L2容量 | 62914560 B，即60 MiB | SDK查询结果；匹配的S5000 L2理论峰值带宽尚需补齐。 |
| MP数量 | 60 | 避免与MUSA算术核心数量混用。 |
| warp大小 | 32 | 采用本机S5000查询结果。 |

因此S5000“寄存器数量/大小”在应用可见通用寄存器口径下可标√。此前使用S80的28 KB局部存储、S4000的张量/向量峰值补S5000判定，需要撤回；本机L1/SPM一行先保留部分，完整各精度峰值需匹配S5000。

来源为`mthreads-runtime-properties.json`、`mthreads-sdk-fields.json`。既有`musainfo.txt`亦列regsPerBlock=131072；既有GMI详细查询列显存带宽1600GBps。当前GMI有负载遥测异常的历史记录，硬件静态规格与动态利用率分别审查。

### MCU能力与实际导出问题

本轮`mcu --list-sections`返回5类：SpeedOfLight、LaunchStats、MemoryWorkloadAnalysis、MemoryWorkloadAnalysis_Chart、MemoryWorkloadAnalysis_Tables。`mcu --list-metrics`列出77个唯一名称，含：

- `ddr__read_bytes.sum`、`ddr__write_bytes.sum`：读写字节数。
- `l1__sector_hit_rate.pct`、`l2pu__sector_hit_rate.pct`：缓存命中率。
- `mp__throughput.avg.pct_of_peak_sustained_elapsed`、DDR与L1/L2利用率：吞吐相对峰值。
- `launch__registers_per_thread`、`launch__waves_per_multiprocessor`、`launch__mp_count`：启动资源信息。
- `mp__fp32_peak_throughput.avg`、`mp__tc_fp32_peak_throughput.avg`等：峰值/吞吐相关字段，单位和完整精度覆盖仍需实际解析对应报告。

字段清单确认工具暴露的能力，具体负载上的有效数值另需解析对应报告。本轮未将每一字段都写成已经成功获取的数值。未发现直接总FLOP字段，计算量一行仍应按直接输出口径判断。

当前在线MCU用户指南列Linux Driver 5.2.0、MUSA SDK 5.2.0，本机驱动实际为3.3.5-server。该版本组合的兼容性仍需核对；现存崩溃记录的根因尚未定位。本轮未调整驱动或软件。

该机器已存在`/data/ps-s5000-prof-20261120`实测材料，本轮只读复核：

- MUPTI activity成功输出kernel、API、同步、拷贝及多流时间线。
- SDK中多项metrics和PC sampling取数/归因接口缺失，既有测试记录接口参数与返回码；单凭空配置返回INVALID_PARAMETER尚不足以判定硬件完全不具备该功能，但这组材料确实未跑通动态PC热点分析。
- 独立MCU成功记录muDNN softmax/add以及自编译axpy目标，写出报告后发生`repeated_ptr_field.h`断言，进程Aborted。日志表明写报告早于崩溃，具体崩溃根因仍需调试。
- 本轮只读打开3份SQLite报告，quick_check均为ok。softmax/add各894行counter_datas，自编译axpy为558行。行内包含原始BLOB，本轮未完成所有性能指标数值解码。
- 自编译目标的名称匹配和多section兼容问题保存在原日志。原分析称GUI可读，当前会话未独立打开GUI核验。

因此“本机完全没有计数器”应更新为“SDK通用指标接口有限，MCU另有计数器采集路径，但当前CLI导出不稳定”。Agent自动接入的难点主要在可用接口、稳定导出和时序归因深度，不能只凭一张能力表概括所有限制。

原件索引见`existing-s5000-manifest.json`；本轮报告只读检查见`mthreads-existing-report-check.json`。昨天清单为26个指标，本轮宿主清单为77个，同一显示版本下的环境/枚举差异尚未定位，保留两份记录。

### 时序指标的判定

现有S5000证据未跑通指令PC热点或逐指令依赖分析。PM Sampling官方附图支持活跃周期/吞吐的采样观察，但本轮仅复核到全局周期与相关字段及原始计数报告，尚未在这台机器独立验证采样时间线视图。若主表限定指定机器实际可获取的指令流水空泡，建议暂缓此前从×改部分的建议。后续若纳入采样级空闲趋势，需要为NVIDIA与摩尔线程统一定义与证据门槛。

## 910B4-1：本机依据支持哪些勾

- CANN配置直接给出20 Cube、40 Vector、L1 512 KiB、UB 192 KiB、L2 168 MiB、UB block 32 B；本轮已重新读取。
- GN样本`aiv_vec_fops` 64/64有效；GQA的`aic_cube_fops`1024/1024有效，两个Vector子核各`aiv_vec_fops`1024/1024有效。计算量存在直接字段。
- OpBasicInfo、计算/搬运流水比例、读写量、通路带宽利用率、L2读写/总命中率均有既有有效样本。
- 指令热点、同步等待与空闲观察有原生模拟器与本项目真实重放材料，按原约定三项可保留√。分析以已采样核和观测窗口为范围。
- 原生模拟器GPR使用数尚未补齐910B4-1全部寄存器的位宽/物理规格，寄存器规格仍需保留部分。
- 950的512 B L2 cache line和Cube/Vector峰值仅适用于950。按指定910B4-1收紧后，L2完整规格、完整4/8/16/32 bit矩阵/向量峰值需重新逐项核对，撤回950补齐这些行的做法。官方910B带宽模型可以单列，具体分支和物理峰值规格仍需与本机内存类型匹配。

H1—H6主要支持采集与分析能力，硬件峰值规格依赖单独的型号匹配来源。工具直接字段、算法公式、模拟器原始事件与HG-KDA脚本汇总分别记录，能够保持同事报告强调的直接输出口径。

## NVIDIA额外发现

N3描述的旧nvprof FLOPs字段确实存在，官方支持表同时说明Ampere及之后架构已无nvprof支持。若指定机器是H20，N3的旧工具示例仅作历史能力参考，H20上的“直接总FLOPs字段”需要当前Nsight Compute实际字段另行支撑。NVIDIA寄存器依赖视图属于SASS分析和相关采样组合，准确表述需要保留静态依赖与动态采样的差别。

## 官方可复核入口

- [Ascend msopprof性能字段](https://github.com/Ascend/msopprof/blob/a74e105fcfad06c3faa4be308834b3158e221299/docs/zh/user_guide/msopprof_performance_data.md)
- [Ascend模拟器用户指南](https://github.com/Ascend/msopprof/blob/a74e105fcfad06c3faa4be308834b3158e221299/docs/zh/user_guide/msopprof_simulator_user_guide.md)
- [Ascend msprof输出字段](https://github.com/Ascend/msprof/blob/a49240a2086ea472cb12a4a2399b395ebd5fd8fc/docs/zh/user_guide/profile_data_file_references.md)
- [M2官方历史用户指南](https://docs.mthreads.com/mooreperf/mooreperf-doc-online/moore_perf_compute/historic_version/v1.2/user_guide/)
- [MCU v1.3.0发布说明](https://docs.mthreads.com/mooreperf/mooreperf-doc-online/moore_perf_compute/releasenote/v1.3.0_release_note/)
- [N3旧Profiler架构支持范围](https://docs.nvidia.com/cuda/profiler-users-guide/index.html)

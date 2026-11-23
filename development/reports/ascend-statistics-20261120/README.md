# 昇腾统计数据补齐

2026-11-20；卡7 Ascend910B4-1；CANN 9.0.0；驱动25.2.0；1650MHz观察；规格与采集能力分别记录。

根目录`统计数据.xlsx`已补齐15项空白，并为20项指标添加逐项依据。原有合并、样式及其他厂商数值保留；原件备份为`statistics-before-ascend.xlsx`。表内简写：✔表示该项能力有文档或实际观测；部分表示仅覆盖子项；待确认表示本次资料仍缺具体规格；星号范围见批注及“昇腾依据”页。

| 指标 | 结论 | 已核验事实 | 范围与未知项 |
| --- | --- | --- | --- |
| host launch 时间 | ✔ | 官方msprof文档确认AscendCL、Runtime等API耗时及HostToDevice关联。 | 本次复用文档支持结论；未新增应用级host采集。API耗时与device执行时间分别报告。 |
| device kernel 时间 | ✔ | GN当前kernel 17.580351μs，GQA 1368.927368μs。 | 单kernel profiling耗时；正式完整调用性能由原KGS／GQA合同测量保存。 |
| 并行度 | ✔* | GN配置64个program；实际设备20 Cube、40 Vector。GQA配置1024／2048。 | 启动规模、物理核数和同时驻留／占用率分别解释；本次仅核实前两项。 |
| 计算量 | ✔ | GN aiv_vec_fops 64/64有效，均值179260.5；GQA Cube 1024/1024有效，均值98304。 | PMU指令口径；含实际执行开销，算法FLOPs及FMA计数口径需单列。 |
| 访存量 | ✔ | GN 64/64有效：GM→UB 256 KB/program，UB→GM 128.25 KB/program。 | 按数据通路统计；保留工具KB单位及estimate标签；逻辑搬运与HBM物理事务分别解释。 |
| 算力利用率 | 部分 | GN aiv_vec_ratio均值0.4083583，FP32占比0.3772072，均64/64有效。 | 当前实测为流水占比；整卡FLOPs/s÷匹配dtype峰值尚待峰值分母及计数口径核验。 |
| 带宽利用率 | ✔* | GN GM→UB利用率14.096889%，UB→GM 8.319647%，64/64有效。 | 数据通路及工具分母限定；整卡HBM峰值利用率需独立峰值带宽，estimate字段保留。 |
| 寄存器数量&大小 | 待确认 | 已查SDK platform配置、平台查询头文件及公开profiler文档，未取得完整型号匹配的寄存器规格。 | UB 192 KiB及L0/L1容量分别属于本地存储；PMU寄存器数值也单独解释。 |
| L1 cache 容量/带宽/cache line size | 部分 | B4-1配置L1 524288 B＝512 KiB；GQA L1读／写字段1024/1024有效。 | 峰值带宽待确认；cache line对软件管理L1 Buffer口径不适用。UB block 32B保留为UB粒度。 |
| L2 cache 容量/带宽/cache line size | 部分 | 实际B4-1运行时L2 168MB；匹配配置176160768 B＝168 MiB。 | 峰值带宽、cache line size待确认；通用B4.ini为96 MiB，须选择实际B4-1配置。 |
| DRAM 容量/带宽 | 部分 | 板卡HBM Capacity(MB)=65536；运行时total_memory=62420MB；GN主存读均值31.080297 GB/s/program。 | 保留容量工具原单位；物理容量与运行时可见容量分别记录。整卡理论峰值HBM带宽待确认，时钟1600MHz单列。 |
| 4/8/16/32 bit矩阵算力 | 部分 | 匹配SDK配置含INT4/INT8 DtypeMKN及FP16/FP32相关mmad映射；GQA BF16目标实际通过。 | 4/8/16/32bit全部峰值规格尚待型号匹配资料；区分INT/FP、累加dtype及稀疏性，配置声明仅作能力线索。 |
| 4/8/16/32 bit向量算力 | 部分 | SDK vadd/vmul列FP16/FP32/INT16/INT32；8bit可见转换／搬运；本次GN执行FP32。 | 8bit转换／搬运与同位宽原生算术分开记录；4/8/16/32bit整卡峰值表待确认。 |
| core 数量 | ✔ | torch_npu属性20 Cube、40 Vector；npu-smi Aicore Count=20；匹配SDK一致。 | 模拟器曾初始化24 AIC／48 Vector，采集配置与真机可用核数分别标识。 |
| 功能部件流水线利用率 | ✔ | GN Vector=0.4083583、MTE2=0.5621627、MTE3=0.2697368，64/64有效。 | 各流水允许重叠；使用各自周期分母，保持逐核与全调用口径。 |
| L1缓存命中率 | 不适用* | 本次msOpProf支持L1带宽；另有aic/aiv_icache_miss_rate，官方定义为指令预留缓存。 | L1数据缓存命中率按此架构口径不适用；指令cache miss单列，保留其官方定义。 |
| L2缓存命中率 | ✔ | GN读取命中率均值50.095173%、写0.763082%、合计44.585355%，64/64有效。 | 这些为逐block算术均值示例；全调用命中率应汇总hit与miss计数再计算。 |
| 指令热点 | ✔* | 当前GN已采指令／区域热点，GQA三个核均有局部窗口报告。 | 热点累计时长用于排序，流水可重叠；源码映射受调试信息限制。 |
| 流水线依赖 | ✔* | GN与GQA报告提供分流水时间轴及等待事件，已关联源码／PC。 | 已观测同步关系与等待范围可查；完整动态依赖图、全芯片因果链仍需专项分析。 |
| 流水线空泡 | ✔* | 现有分析输出active、wait_only及observable idle；可对照修改前后窗口。 | 窗口与核边界明确；无元数据泳道保持未知，空白区原因需结合依赖及搬运进一步判断。 |

实际型号由torch_npu与AscendCL共同查询；B4-1运行时L2为168MB，匹配SDK为168MiB。通用B4配置L2为96MiB，SDK静态memory_size为32GiB，实际板卡HBM查询为65536MB。运行时可见内存为62420MB。分别保存这些值及其来源，采用实际设备属性解释规格差异。SDK配置中的带宽rate缺少本次确认的单位，未据此推导整卡峰值。

当前实测有效字段保存在[有效字段样例](valid-field-samples.json)，完整成对报告保存在[原始观察](paired-observations.json)。GN逐program均值仅用于展示字段有效性；加速比、全调用时间及跨核聚合沿原合同计算。当前GQA计数字段带有本次采集范围，后续性能归因继续与模拟器及实际IR交叉核对。

官方资料固定版本：

- [msopprof-msopprof_performance_data.md](https://raw.githubusercontent.com/Ascend/msopprof/a74e105fcfad06c3faa4be308834b3158e221299/docs/zh/user_guide/msopprof_performance_data.md)，commit `a74e105fcfad06c3faa4be308834b3158e221299`。
- [msopprof-msopprof_simulator_performance_data.md](https://raw.githubusercontent.com/Ascend/msopprof/a74e105fcfad06c3faa4be308834b3158e221299/docs/zh/user_guide/msopprof_simulator_performance_data.md)，commit `a74e105fcfad06c3faa4be308834b3158e221299`。
- [msopprof-msopprof_simulator_user_guide.md](https://raw.githubusercontent.com/Ascend/msopprof/a74e105fcfad06c3faa4be308834b3158e221299/docs/zh/user_guide/msopprof_simulator_user_guide.md)，commit `a74e105fcfad06c3faa4be308834b3158e221299`。
- [msprof-profile_data_file_references.md](https://raw.githubusercontent.com/Ascend/msprof/a49240a2086ea472cb12a4a2399b395ebd5fd8fc/docs/zh/user_guide/profile_data_file_references.md)，commit `a49240a2086ea472cb12a4a2399b395ebd5fd8fc`。

[设备查询](device-query.json) · [运行时属性](runtime-properties.json) · [模拟器观察](simulator-observations.json) · [表格保存检查](workbook-validation.json)。

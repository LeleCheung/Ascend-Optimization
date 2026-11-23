# 昇腾公开规格联网复核

2026-11-21。按用户要求，允许以任一昇腾型号确认公开规格的可获得性。本轮直接访问官方网页、官方白皮书和官方开源代码。

结论：此前将大片黄色归结为峰值算力、L1／L2及主存带宽难以取得，表述过于保守；公开资料中已有明确数值。型号匹配与组合指标覆盖需要分别核对。

| 项目 | 公开数值 | 来源与范围 |
| --- | --- | --- |
| Cube矩阵峰值 | 昇腾950PR：MXFP4 1730 TFLOPS、INT8 865 TOPS、BF16／FP16 432 TFLOPS、TF32 216 TFLOPS | 官方950白皮书表3-1，PDF第13—14页；采用该型号较高配置。 |
| Vector向量峰值 | 昇腾950PR：FP16／BF16 54 TFLOPS、FP32 27 TFLOPS、INT8 54 TOPS、INT16 27 TOPS、INT32 13 TOPS、INT64 6 TOPS | 同表，PDF第14页；采用较高配置。该表未列4bit向量峰值。 |
| 主存容量／带宽 | 昇腾950PR：128 GB、1.6 TB/s；另一配置112 GB、1.4 TB/s | 同表，PDF第14页。 |
| L2容量／cache line | 昇腾950：最高128 MB、512 B cache line、4×128 B sector | 同表PDF第15页；第26页再次说明sector。 |
| L1带宽模型 | 910B：每核读256 B/cycle、写256 B/cycle，读写合计512 B/cycle | 官方msopprof `roofline.h`第182行起，结合`roofline.cpp`第234行起的频率／核数换算；此处属于工具理论模型。 |
| L2带宽模型 | 910B1：8 TB/s；910B4／B4-1：4 TB/s | 官方msopprof `data_visualize_const.h`第163行起的默认／SK／SS模型；CJ分支分别为7.4及6.3 TB/s。保留代码标注单位。 |
| 主存带宽模型 | 910B1：1.8 TB/s；910B4／B4-1：0.8 TB/s | 同文件第139行起的默认／SK／SS模型；CJ分支分别为1.34及1.25 TB/s。 |

950PR峰值取Cube、Vector各自表项，Cube＋Vector总峰值另有独立表项。上述950规格用于说明厂商已公开对应参数；当前910B4-1设备的实测性能和容量保持各自口径。官方工具带宽常量属于Roofline模型，应用时按SoC、内存类型与通路选择。

官方来源：

- [昇腾950处理器官网](https://www.hiascend.com/hardware/processor)，页面直接提供架构白皮书下载。
- [昇腾950 NPU架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)，本地保存为`ascend950-architecture-whitepaper.pdf`。
- [Atlas 350官网](https://www.hiascend.com/hardware/accelerator-card)，公开1561 TFLOPS MXFP4、425 TFLOPS FP16／BF16，以及112 GB HBM、1.4 TB/s内存带宽；该卡规格单独记录。
- [msopprof带宽常量](https://github.com/Ascend/msopprof/blob/a74e105fcfad06c3faa4be308834b3158e221299/csrc/op_profiling/profiling/device/data_visualize/data_visualize_const.h#L139)。
- [msopprof L1／UB带宽模型](https://github.com/Ascend/msopprof/blob/a74e105fcfad06c3faa4be308834b3158e221299/csrc/op_profiling/profiling/device/data_visualize/roofline.h#L182)。
- [msopprof带宽换算](https://github.com/Ascend/msopprof/blob/a74e105fcfad06c3faa4be308834b3158e221299/csrc/op_profiling/profiling/device/data_visualize/roofline.cpp#L234)。

补充检索找到第三方[910B微基准](https://github.com/RinShiina/ascend-mubench/blob/main/report/report.md)，报告L1→L0A单核单通道约456.86 GB/s实测。该结果单列为第三方观察，主表官方规格结论使用上面的官方资料。

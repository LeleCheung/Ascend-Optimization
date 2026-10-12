# Msopprof 固定版本调研资料

本目录保存本轮核查 Roofline 计数与公式时读取的官方原始文件，来自 [Ascend/msopprof](https://gitcode.com/Ascend/msopprof)，固定提交 `a63063569742c1ba7be9688fc7799a8248574dee`。它是研究快照，不是可独立编译的 Msopprof 安装包；源码保留原版权头，许可见 [LICENSE](LICENSE)。

| 本地文件 | 上游路径 |
| --- | --- |
| `msopprof_performance_data.md` | `docs/zh/user_guide/msopprof_performance_data.md` |
| `pmu_calculate.cpp`、`pmu_calculate.h`、`pmu_formulator.h` | `csrc/op_profiling/profiling/device/data_parse/` 下同名文件 |
| `pmu_calculator.cpp`、`roofline.cpp`、`roofline.h` | `csrc/op_profiling/profiling/device/data_visualize/` 下同名文件 |

原始性能文档中的图片引用属于上游文档目录，图片未在此快照中复制；查看图片应打开该固定版本的上游文档。文件内容保持原始字节，不把它们改写为本项目代码。

本轮矩阵乘官方报告给出 `memory caused`，原始 FOP 计数与算法 FLOPs 相差 128 倍。后续应结合这些公式、实际采集 CSV 和硬件版本核实缩放及 roof；当前没有完成数值 Roofline。实测解读见 [报告](../../project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/diagnostics-20261010/matmul-roofline-20261010/官方Roofline解读.md)。

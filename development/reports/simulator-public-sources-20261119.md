# 模拟器公开资料与交互展示

核对日期：2026-11-19。

## 开放范围

MindStudio Ops Profiler（msOpProf）公开仓库采用Mulan PSL v2，README记录2025-12-31全面开源。模拟器模式的调度、数据解析和报告代码可以查阅。

CANN硬件仿真库作为独立依赖加载。官方指南要求配置指定芯片的simulator库目录；调度源码检查仿真动态库并设置加载环境。本次核查尚未找到Ascend910B4硬件行为／时序模型本体的公开源码与独立开源许可。因此，对外描述为“使用CANN厂商仿真库，结合开源msOpProf采集解析工具”。

本地实验固定CANN 9.0.0，实际命令为`msprof op simulator`。公开仓库当前文档使用`msopprof simulator`，具体参数和路径以安装版本为准，当前开源master与本地二进制版本分别记录。

## 资料

- [公开仓库与许可证](https://github.com/Ascend/msopprof)
- [模拟器模式使用指南](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_simulator_user_guide.md)：环境、参数、部分窗口超时采集、指令流水、源码映射与MindStudio Insight。
- [模拟器性能数据格式](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_simulator_performance_data.md)：指令报告、源码行报告、trace与可视化数据。
- [模拟器模式实现](https://github.com/Ascend/msopprof/tree/master/csrc/op_profiling/profiling/simulator)：调度和解析源码。
- [本次核查的固定commit及原文](batchnorm-simulator-20261118/public-simulator-sources.json)：`ce5a1158cebf0ba3b9f8a48793304660c22ece36`。

官方工具已支持指令流水和源码热点展示。HG-KDA研究重点放在证据如何进入Agent决策，以及修改后的机制与真机收益。

## 首个交互案例

[BatchNorm交互网页](batchnorm-simulator-20261118/interactive.html)可直接离线打开，包含预取前、预取后、整除路径去掩码三个版本。支持版本切换、流水占比、真实指令时间轴缩放／平移、指令点击到源码、热点区域到源码。数据来源为冻结trace和源码，导出时核对源码SHA。

横轴为各自可见窗口的相对位置，保留部分采集与绝对时钟未核验范围。动态交互来自真实记录重绘。目标KGS性能单独展示，区域累计指令报告时间仅用于排序。默认展示预取步骤的证据对比，最新去掩码版的等待比例上升也保留。

重建：`python -m development.tools.traces.build_simulator_demo`。

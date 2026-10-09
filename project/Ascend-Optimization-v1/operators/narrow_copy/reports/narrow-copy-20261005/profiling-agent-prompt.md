# Ascend profiling agent 诊断输入

目标设备：Ascend910B4-1
Profiler：msprof；本报告仅用于诊断，不能替代正式 walltime。

主 kernel：_narrow_copy_flat_persistent_kernel
平均设备时间：9.61 us
MTE2 占比：0.691
MTE3 占比：0.708
Vector 占比：0.001
Scalar 占比：0.129
Cube 利用率：0.0
Simulator 指令数：350
源码映射指令数：90

诊断结论：这是以 MTE2/MTE3 数据搬运为主的 narrow_copy 路径，Vector/Cube 计算不是主要限制。优先减少 program 数量、循环控制、同步和边界处理；连续 row-major 拷贝应保持直接搬运，不要引入复杂索引或额外计算。

实验约束：保持 narrow_copy 的 ABI、完整 workload 和 PyTorch baseline；任何候选必须通过全部 correctness，并用 KGS walltime 重新比较逐 case 延迟。不要把 msprof 或 simulator 的采集时间当作加速比。

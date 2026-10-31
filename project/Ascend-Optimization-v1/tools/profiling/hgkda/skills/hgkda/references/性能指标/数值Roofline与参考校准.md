# 数值 Roofline 与参考校准

先固定源码、kernel、shape、dtype、物理卡及采集范围。roofline/profile_data.py从同次OpBasicInfo与官方可视化分段核对身份，读取GM/L2、Cube、Vector各自的计算量和流量。不要用另一轮的时间拼接计数；缺失、非有限或全零点保留为不支持。

矩阵乘有效工作量为2MNK，不包含bias/激活尾处理。当前CSV的aic_cube_fops使用固定8192乘矩阵指令数，而单条宏指令工作量随tile变化，不能统一乘128。官方Roofline使用另一组事件；FP16/BF16受控矩阵与算法量匹配，FP32当前IEEE路径出现内部操作放大。内部操作数不是有效算法FLOPs，Vector点也不能重复分配整个GEMM工作量。

GM字节来自官方cache miss/eviction模型，L2为另一层接口流量；Memory.csv主存字段不与GM等价。纯复制单列字节和时间，不构造虚假的FLOPs点。

固定本机官方GM模型为0.8TB/s，但同PMU口径persistent copy观测约1.148TB/s，大数组torch copy有效逻辑吞吐约1.25TB/s。前者与矩阵点使用同GM事件模型，可作经验线；后者只单列，不与GM计数字节混用。模型与本机匹配仍待核实：官方模型原值保留；同dtype GEMM和同口径copy只用于经验参考，不能称为理论峰值或已证明上界。FP32计算参考单独使用同精度结果，不借用FP16 Cube峰值。

诊断顺序：先检查实际流量与参考差异，再比较GM、L2及组件参考差距，最后结合流水等待、片上缓冲和IR选择一项修改。父版GM请求更少仍慢时，不能盲目把降低HBM字节作为目标；可检验dot调度、UB压力与流水重叠。活跃比例仍仅是线索，不是可相加的时间分解。limiting_side只说明参考曲线较低的一侧，不证明远低于曲线的kernel实际计算受限；模型输出也必须经证据核对。

roofline/analyze.py生成数值点、manifest和教师分析器输出；roofline/summarize.py生成hgkda-roofline-v1紧凑摘要。build-kg-overlay.py的--roofline-evidence参数将它注入新KG分析角色，并记录SHA。新候选需重新采集和完整KGS验证；摘要存在不等于模型已经消费或获得收益，实际调用单独留证。

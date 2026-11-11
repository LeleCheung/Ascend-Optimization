加载 skills/hgkda/SKILL.md，读取 task.json。本次不是正确性演示，而是修复性能优化闭环。
只读取/写入当前workspace，禁止查阅历史最佳源码或其他实验目录，禁止其他agent。先执行工具桥context，然后读取context-brief.json、performance-feedback.json、skills/hgkda/references/launch-and-feedback.md。上下文中的历史方法允许采用，但不允许把历史源码冒充模型生成。
从starter.py继续生成candidate.py及diagnosis.json。现有完整KGS/CANN成绩0.210037，36/36及42/42补测通过；小输入约80us而CANN约13us，首要改动是缓存Triton编译启动器，简化每次宿主参数解析及device上下文。当前源码有FP16/BF16/FP32和非连续布局，保留语义。
实际环境的编译对象具有function、packed_metadata、run，run为triton.backends.ascend.driver.NPULauncher，存在launch方法。历史方法的nomask参数示例：compiled=kernel.warmup(A,BLOCK=block,F16MATH=f16_math,grid=(1,))；launcher=compiled.run；launcher.launch(grid,1,1,current_stream,compiled.function,compiled.packed_metadata,None,None,None,A.data_ptr(),None)。masked在指针后传n，再传None。此调用ABI仍须本轮预检验证。最后的None是此后端隐藏参数，不可随意丢弃。
可以缓存编译句柄、网格、元数据和宿主参数模板；禁止缓存Tensor对象、计算结果或旧stream。每次用当前A指针和当前stream，按device/dtype/shape/alignment等分开缓存。优先只给对齐连续路径缓存，其他走原Triton实现。调试、编译-only或msprof tensor注册时使用普通Triton调用，避免绕开后端检查。第一次warmup只是编译，不执行；不要重复原地计算。
run(A)直接调用精简路径，不再每次做*args/**kwargs解析。所有辅助Python函数单下划线开头，只有run公开。全部设备计算、复制、非连续回退仍严格Triton，不许调用torch算子/CANN/DMA。元数据current_stream和参数管理允许。源码与注释使用专业中文。
大小输入诊断同时考虑。先保留设备公式与分块，只修宿主启动；若正式反馈显示大输入拖后腿，再由宿主提供反馈后修改。diagnosis记录真实证据、假设、修改、预期、未决问题及使用了历史经验。
写完立即调用工具桥preflight --source 当前目录candidate.py，Bash timeout=600000。预检失败可修复新源码，最多6份。42/42通过后本次模型调用返回，交给宿主逐路径审查及正式KGS；这只是审查边界，不是goal终点。正式性能未达标会继续下一次模型调用，不得声称已完成目标。不要读取巨大JSON和IR全文，每次按需读取。

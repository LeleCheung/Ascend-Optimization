# 参考后端的构建与资格检查

这里的工具在匹配的 Linux、CANN 和 PyTorch NPU 环境中执行。独立 HG-KDA 核心仍只依赖 Python 标准库；设备工具的可选依赖单独配置。

| 工具 | 用途 |
| --- | --- |
| `build_cann_reference.py` | 重建示例中已固定的 CANN 参考宿主桥 |
| `probe_cann_expm1.py` | 独立进程探测官方旧图 Expm1 的默认／高精度模式，核验小值及零符号 |
| `cann_uniform_range.cpp` | 绑定原生 Uniform＋Clamp，适配原随机合同的半开区间语义 |
| `build_uniform_range.py` | 单线程编译随机参考桥，核验宿主标量边界并生成带二进制 SHA 的参考包装 |
| `run_uniform_reference.py` | 等待指定回归结束，核验参考资格，再经独立 CLI 执行一次正式 KGS 请求 |
| `cann_trapezoid.cpp`、`build_trapezoid.py` | 构建原生 Add＋Muls＋FP32 Cumsum＋Cast 参考及独立 Triton 扫描候选 |
| `run_trapezoid_reference.py` | 单独核验数学参考合同，保留原输入、容差和补测；通过后提交一次 KGS 正式请求 |

参考实现的设备计算通过厂商接口执行。C++ 宿主桥负责张量描述符、分配、标量参数、生成器状态和 stream；候选的设备计算与搬运遵循严格 Triton 约束。

Uniform＋Clamp 明确属于同语义组合参考。它使用原始 from／to 生成样本，再限制到输出类型在原半开区间内可表示的最小值和最大值，修复低精度输出舍入越界；完整性能包含组合的必要调用开销。结果按区间、统计、布局、当前 stream 和生成器状态分别核验。局部资格检查与完整 KGS 评测分别保存，参考合格后才能报告相对该参考的性能。

新批次使用全新输出目录；已有请求或资格原件保持原样。共享设备通过项目卡锁串行使用，工具只管理明确属于当前批次的服务。当前批次的具体参数、阶段与发现见 [回归记录](../../docs/regression-20261109.md)。

累计积分的数学参考合同单独标记：CPU PyTorch 仅用于核验中间舍入及 FP32 累加的数学行为，正式性能参考仍在 NPU 上执行全部 CANN 调用。原 NPU 默认参考存在不稳定性，其合同和失败证据继续保留。新合同不计为原默认合同已经通过，也不混入 25 项固定候选回归。

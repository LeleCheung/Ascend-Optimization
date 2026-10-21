# MoE无profiler版的独立布局修复

<!-- 当前实测结果 -->
## 当前实测结果

- `semantic-repair:moe_sum_reduce`：passed，相对PyTorch 1.307× / 1.245×。
- 扩展语义：36/36，passed。

性能计时为完整调用walltime；来源、预算及逐case结果按原文件保留。下文是实验准备时的设计说明。

固定父源码为原无profiler第二轮；正式8项及两次复验已通过，但扩展语义30/36，六项非连续输入因UB超限编译失败。原始失败快照保存在diagnostic-source，与主批原始语义回执SHA绑定。

本批保留父源码完整字节。连续且行宽对齐的大输入和连续小输入保留原快路径；一般布局或非对齐大输入改用每program一行、列块不超过256，继续调用原Triton归约kernel和原scale计算。只调整分块和调度，不使用PyTorch数值计算，不缓存Tensor或结果。

新增预算单列：原36项补充语义、原KGS 8项合同两次完整walltime复验。先验证语义，全部同源码/检查器身份通过后才评测性能；失败保留。该修复不替代原始无profiler组，不计入相同两轮预算。

在910B宿主项目目录使用run-semantic-repair.py --batch night-20261021-v3-moe-stride-repair，tmux等待同一物理卡7卡锁。已有队列时不重复启动。核验使用容器Python3.11运行verify-semantic-repairs.py --batch-dir <本批目录> --output <新核验文件>；主批交付工具也会检查本批。

当前为待运行候选，尚无NPU正确性或性能成绩。原始合同、候选和父源码SHA见plan.json。本批不在旧postflight预置名单中，最终收尾必须另行检查。

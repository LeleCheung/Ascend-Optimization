# amin

`amin` 沿指定维度取最小值。验收看板华为列 G92 为 **0.5536×**，属于低于 0.8× 的正式优化对象。

## 本轮环境与进度

2026-10-10 固定 FlagGems master 提交 `d6a8eec473517a3d68157b208eb9c057eb1d4c50`。原始实现位于 `src/flag_gems/ops/amin.py`，Ascend 后端没有同名专用实现；最终调用和测试结果以 KGS 原始记录为准。

910B 的独立实验根目录为 `/data/hanle/ascend-optimization/goal-20261010`，其中新仓库使用总仓库提交 `a349822bf9364f9cbc42d61c7eaa91ab74a8fd7d`，本轮工具改动另外记录。KGS 在现有 `tle_yy` 容器内运行，使用 loopback 端口 `19655`、物理卡 7（服务中逻辑 `npu:0`），单 worker 串行执行。没有替换现有服务或升级核心运行时。

- KG 6.7.0、KGS 6.5.0、Protocol v6.2。
- CANN 9.0.0、torch 2.9.0+cpu、torch_npu 2.9.0.post2、Triton 3.5.1。
- Flagtree 0.6.0+ascend.gitc286cba6、驱动 25.2.0。
- Claude Code 2.1.284，通过 DeepSeek 运行；模型为 `deepseek-v4-flash[1m]`，最小连接已验证。

KGS inspect 已成功。性能范围为 15 个 case：float16/float32/bfloat16，各自覆盖 `[1048576]`、`[64,64]`、`[4096,4096]`、`[64,512,512]`、`[1024,1024,1024]`；多维输入沿 dim=1 归约。完整正确性使用固定 master 的 `tests/test_amin.py`，不裁剪用例。

**已确认原版失败原因，并取得首个完整通过的优化候选；KG 两组未完成。** master 对整数 `dim` 的入口问题由转列表解决。性能套件第 10 次计时的大输入 `[1024,1024,1024]` 沿 dim=1 归约，转置后输出有 1,048,576 行；autotune 的 BLOCK_M=8 需要启动 131072 个 program，超过 Ascend `coreDim <= 65535`。日志明确报 `KernelLaunch failed ... coreDim ... 131072`，随后出现计时错误和空 CSV，最终触发 pandas `.str` 错误。独立完整正确性 exit=0，性能 exit=1，因此不能把失败归结为通用计时环境问题。

原始证据在 [诊断产物](reports/diagnostics-20261010/amin-full-pytest-timer-v2-20261010/raw-artifacts/benchmark.stdout.txt)。该 debug job 最终还因附件数量超过 32 被标为 FAILED；原始测试输出已从容器单独保存，并记录哈希。

兼容基线 `flaggems-master-grid-adapter-20261010.py` 保留 master 内核、转置和 autotune 配置，仅将超过上限的输出行按最多 65528 行拆分启动。必须标为 **FlagGems master 加启动兼容修复**，其性能不能称为未修改上游的性能。两组 KG 从此同一兼容基线开始。

失败记录在 `reports/master-interface-v4-20261010/`。候选 `direct-axis-global-v3-20261010.py` 沿连续单轴直接索引，避免中间轴转置；全量归约采用多阶段方案，将 1M 输入的第一阶段 program 数从 1024 降至 64。其完整结果为 **27/27 通过、相对 PyTorch 0.626×**，包括 12 个正确性和 15 个性能用例。主要短板为大矩阵末轴和中间轴归约，正在按真实布局筛选分块。

启动兼容基线已完整通过 **27/27、相对 PyTorch 0.296×**。按相同 case 的候选延迟计算，v3 相对该兼容基线提高 **2.13×**，不是相对未修改上游的加速比。见 [同合同对比](reports/direct-axis-global-v3-20261010/版本对比.md)。

两组 KG 首次均在 `prepare_catalog` 读取 `/operator-contract` 时遇到 60 秒超时，尚未进入 Coder。KGS 6.5.0 的合同读取和长诊断共享单线程执行器，合同请求排队超时。已保留失败工作区，安排在其他本轮任务结束后用 `kg resume` 原位恢复；没有修改测试、服务源码或原生组提示词。

72 组真机分块筛选已经完成。多行末轴分块明显快于旧配置，但更大的分块存在明确 UB 溢出。v4 完整 **27/27 通过、相对 PyTorch 0.789×**。

后续 70 组归约诊断使用设备 kernel 计时筛选配置，并在逐值一致后生成 v5。v5 完整 **27/27 通过、相对 PyTorch 0.916×、相对启动兼容基线 3.11×**，达到本阶段 0.8× 目标。PyTorch 基线几何平均变化为 0.994×。主要改进是连续布局直接归约、多行/宽列分块，以及对对应分支关闭多缓冲。float16 的原生精度归约有 14 组诊断成功，v5 的三个 float16 专用分支采用这种配置；bfloat16 原生 `minimum` 则在当前编译器中隐式提升到 float32，导致循环变量类型不一致，v5 的 bfloat16 分支因此仍用 float32 累积。失败配置没有计入成果；文件名中的 `native` 不代表所有 dtype 都采用原生精度。

候选和逐 case 证据见 [v5 报告](reports/direct-axis-native-v5-20261010/版本对比.md)及 [精简汇报](reports/direct-axis-native-v5-20261010/精简汇报.md)。截至 2026-10-10 22:53，原生 KG 无 profiler 已正常结束，第二轮完整 27/27 通过、0.742×，最佳源码 SHA 为 `fac3bca3b3b48a9f77d1837f6f4c4ac289d03c1fae5b46278b534f28753bc8ba`；profiler 组已恢复。两组仍须独立完整复验，四版本闭环尚未完成。

后续 `probe-bf16-cast.py` 将验证 `minimum` 后显式转回 bf16 是否能保持循环类型并改善延迟，同时测量 float32 累积对照。每个配置先与 PyTorch、v5 逐值比较，`build-bf16-v6.py` 仅从通过的真机配置生成新候选；`run-bf16-v6-910b.sh` 排在原生 KG、独立复验及矩阵分组诊断之后，随后执行完整 27 项测试。该实验尚未运行，没有新增性能结论。

## 实验顺序

1. 保留原 master 的失败记录，完整评测仅修复入口和超限启动的兼容基线。
2. 若兼容基线完整通过，以其源码为共同起点，运行 KG 无 profiler 和有原生 profiler 两组。均使用 `simple_opt`、单 Coder、两轮、同一模型和 KGS。`--skip-review` 只跳过模型测试审核，正确性和性能门禁保留。
3. 根据最慢和最低加速比 case 采集真机 metrics，必要时补充模拟器 instruction；分析布局整理、实际搬运与归约流水线，再修改代码。
4. 独立复验最终候选，交付兼容基线、KG 两组和分析优化版的四版本对比、逐 case 数据与关键优化。目标为完整正确性通过并超过 0.8×。

FlagGems core 默认采用 **设备侧 kernel 计时**，不是完整 Python 调用耗时。`--no-profile` 关闭优化反馈，计时仍可使用 NPU profiler。

## 持续运行与查看

```bash
ssh baai-910b
tmux attach -t kg-amin-grid-master-20261010
```

查看日志：

```bash
root=/data/hanle/ascend-optimization/goal-20261010
tail -n 40 "$root/campaign-logs/amin-grid-master.log"
tail -n 40 "$root/campaign-logs/roofline-and-global.log"
tail -n 40 "$root/campaign-logs/amin-direct-tiles.log"
tail -n 40 "$root/campaign-logs/amin-native-continuation.log"
```

tmux 任务和 KGS 队列在关闭 Windows SSH 窗口后继续运行。`run-grid-master-campaign-910b.sh` 在兼容基线通过后串行运行两组 KG；若基线失败则停止，保留证据。若 workspace 已存在，脚本拒绝覆盖，先检查状态。

运行工具为 [evaluate-operator.py](../../tools/evaluation/evaluate-operator.py)、[KG 启动工具](../../tools/evaluation/run-kg-operator-910b.sh)和 [串行 campaign](../../tools/evaluation/run-low-speedup-campaign-910b.sh)。核心服务和模型凭据保留在服务器环境中，不写入报告。

## 本轮部署问题

Windows 工作树与 Linux Git 索引跨环境传输后，KGS 曾把固定提交的 Python 文件标为修改并拒绝 inspect。仅在本轮新 FlagGems 副本中从 `git archive HEAD` 恢复提交内容，然后在**执行 KGS 的容器内**用 `git read-tree HEAD` 与 `git update-index --refresh` 重建索引。内容、提交和容器内 Git source 状态核验后，inspect 正常通过。旧工作区没有 reset 或 pull。

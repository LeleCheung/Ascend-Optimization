# SELU 独立 skill 的性能恢复

本轮在既有独立skill生成版上继续优化，固定CANN参考、KGS正式合同和计时设置，将相对CANN的成绩由 **0.210037×提升到0.887010×**，超过0.8×验收线。正式36/36、双方各42/42补测及专项33/33全部通过。历史1.039292×交付继续保留；新生成版及全部失败原件单独归档。

## 方法修复

1. **检索可复用的启动方法。** 增加绑定历史源码和正式结果的 `selu-compiled-launch-path` 记录。精确算子匹配优先，只有显式允许迁移的经验才按类型补充。模型接收方法和条件，无需读取历史最佳源码。
2. **同时覆盖大小输入。** 正式结果与固定case清单关联，按 `-log(speedup)` 选择拉低几何平均的case，同时覆盖各dtype的最大输入。设备任务、异步提交和完整调用分别呈现。
3. **依据正式性能继续优化。** 新增 `feedback` 命令，核对源码和审计身份、重建几何平均，并返回继续优化或达标决策。正确性预检通过是审查边界，完整性能反馈才决定后续优化。

## 启动与设备证据

起点 `3cec1212c3b7…` 的正式完整调用相对 CANN 为0.210037×。独立设备轨迹中，FP16 1024元素的Triton任务中位时间约1.00微秒，CANN约1.34微秒；4096×4096的Triton任务约78.86微秒，CANN约60.40微秒。小输入应优先优化宿主启动，大输入仍需关注设备组织。

初次诊断计时误将编译记录钩子留在计时区间；修订版只在预热记录编译对象，计时前恢复原调度器。两份诊断均保留，初次提交时间不用于定量归因。正式 KGS 测量未使用该钩子。

模型生成的连续路径缓存Triton编译句柄、网格和元数据，每次读取当前输入指针及stream。未对齐、非连续和调试路径继续使用普通Triton。首次warmup只编译，避免原地计算重复执行。

首份可执行缓存版正式成绩为0.312005×，小输入完整调用仍约50微秒。函数级诊断发现，高层 `current_stream(dev)` 每次构造Stream并执行通用设备解析；独立测量约19.29微秒。直接等待版raw查询的独立测量约0.50微秒，但空队列微基准不能表征与异步发射连续交替时的等待成本。

进一步核对本机后端 `backend_register.py`：标准 `get_current_stream` 优先使用 `RawStreamNoWait`；默认开启任务队列的启动器通过 `OpCommand.SetCustomHandler(...).Run()` 发射Triton产物。由此修正宿主早期“禁止NoWait”的指导，按后端接口合同保留队列顺序，再进行当前stream与完整语义检查。查询当前raw stream、缓存查询函数和缓存旧stream是不同操作；本轮只允许前两项。

本机启动器为每个显式constexpr保留Python槽位。旧示例对应一个constexpr，当前kernel有两个；核验后端 `make_launcher` 的解析格式后补齐两个占位。直接接口的grid为整数，普通Triton接口保留tuple。禁止运行时猜测参数位置或捕获执行错误后重发。

## 实际贡献与验证边界

CC／DeepSeek负责生成和修改候选。Codex负责证据整理、经验补齐、工具桥与启动ABI说明、路径审查、补充检查及正式提交。本轮是使用固化方法和宿主反馈的模型优化，包含明确的Codex指导。

候选设备计算、布局处理和回退全部为Triton，CANN仅作参考。专项检查覆盖偏移视图、视图外存储保护、当前stream和不同输入指针；多设备并发未列入本轮验证范围。

## 本轮完整评测

| 候选 | 关键变化 | 相对CANN |
| --- | --- | ---: |
| `c6255855aa16` | 缓存编译启动器，保留高层stream查询 | 0.312005× |
| `80bfefff4ff8` | 直接等待版raw stream查询与简化缓存键 | 0.433299× |
| `0fed8d43398d` | 对齐后端NoWait队列，扩大大输入分块 | 0.710539× |
| `5e9e4d482620` | 缓存命中直接发射，省去重复选块与编译元数据检查 | **0.887010×** |

四份候选均正式36/36、双方各42/42补测通过，每份不同SHA只提交一次正式请求。最终版的小输入约13–15微秒，大输入约45.5／83.2／82.5微秒；整体未追平历史1.039292×，因此不替换已有交付。本轮不增加算子数量，原合同仍达标26/30。

## 原件与复核

[实验目录](../benchmarks/skill-validation/20261109-selu-recovery/)保存1175份原始文件，包括8次模型调用、7份进入预检的源码（其中1份中断、6份形成完整预检结果）、4份正式请求、实际编译产物、真机任务轨迹、CPU函数诊断、后端接口来源及控制器快照。原先0.210037×的[首轮实验](../benchmarks/skill-validation/20261109-selu/)保持不变。

Windows离线复核已核验1175份原件，并按固定合同重算4份正式结果；[版本索引核验](../benchmarks/lineage/20261109/audit-after-selu-recovery.json)覆盖39项、313个文件。指标、源码身份与方法经验分别记录，不把代理费用估计作为实际账单。

在仓库根目录使用已安装HG-KDA的Python执行，输出换成项目外的新文件：

```bash
python tools/skill_validation/verify.py --directory benchmarks/skill-validation/20261109-selu-recovery --output ../hgkda-output/selu-audit.json
python -m hgkda knowledge --operator selu_ --output ../hgkda-output/selu-knowledge.json
python -m hgkda feedback --operator selu_ --source benchmarks/skill-validation/20261109-selu-recovery/formal-5e9e4d482620/operators/selu_/candidate.py --result benchmarks/skill-validation/20261109-selu-recovery/formal-5e9e4d482620/operators/selu_/evaluation/result.json --audit benchmarks/skill-validation/20261109-selu-recovery/formal-5e9e4d482620/audit-final.json --timing-cases benchmarks/regressions/20261109/fixed-25/operators/selu_/cann_supplementary-catalog/timing.jsonl --output ../hgkda-output/selu-feedback.json
```

Windows的原始厂商目录可能超过260字符；仓库使用本地Git的 `core.longpaths=true`，核验器使用扩展路径读取，不重命名原件。以上命令不执行新的设备评测。

910B运行目录为 `/data/hanle/ascend-optimization/runs/selu-skill-recovery-20261109-v2`，仅使用物理卡7及项目共享锁。该批次已完成；复现时创建新运行目录与调用编号，保留原合同、计时设置和一次正式请求策略。核心新增 `feedback`、专项经验检索与明确的类型迁移；宿主工具新增源码绑定的启动诊断、stream／指针检查和自动正式反馈。所有工具仍通过独立CLI及可选KGS后端工作。

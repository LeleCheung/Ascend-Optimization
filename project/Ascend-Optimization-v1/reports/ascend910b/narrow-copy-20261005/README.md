# narrow_copy：910B 最小闭环记录

本次在 `tle_yy` 容器中，用隔离的 FlagGems 副本、专用 KGS 和 Claude Code 完成了 KernelGen 的一次正式运行。最终 workspace 为 `/data/hanle/ascend-optimization/runtime/kg-controller/runs/narrow-copy-kg-20261005-10`，生命周期状态为 `SUCCEEDED`。本目录保存了候选源码、正式评测 JSON、优化输出和运行状态，可在 Windows 直接查看。

## 隔离与版本

- 原始 FlagGems 位于 `/data/hanle/ascend-optimization/FlagGems`，原始提交 `4772d816bc5d52d37c4718f36adf377d81261f83`，未修改。
- 隔离副本位于 `/data/hanle/ascend-optimization/runtime/flagggems-narrow-copy`。`df6df6ef51b1534be1f2b558acaebc4da1370835` 将旧 `input_fn` 改为 `case_fn + build_inputs_fn`，保留原有 5 个 shape、3 种 dtype、dim/start/length 和 GBPS 逻辑；`627466757cc285b684890ded72c9d49275ee547f` 允许仅由该副本的 benchmark 读取 `KERNELGEN_FLAGGEMS_MODE`。两项改动见 [补丁](flaggems-isolated.patch)。
- 专用 KGS 为 `v6.5.0`，端口 `19652`，`--backend npu --timing walltime --max-workers 1`，FlagGems root 指向隔离副本，`KERNELGEN_FLAGGEMS_MODE=operator`。原有 `19650` 未改动。
- 使用 KG `6.7.0`、模型 `deepseek-v4-flash[1m]`、`simple_opt`、`--no-profile`、`max_round=1`。模型凭据只从 910B 本地的 `env.sh` 加载，未写入本仓库。

## 正式结果

预检通过；正式 `eval_round` 返回 `PASSED`，33/33 个正确性 workload 和 15/15 个计时 workload 执行成功。第 1 轮相对 **PyTorch 原生 `torch.narrow_copy`** 的几何平均加速比为 `0.3532×`，最差为 `0.2182×`；最终复核给出 `0.3816×`。`speedup = PyTorch 耗时 / 候选耗时`，小于 1 表示候选更慢。这是一次完整的计时闭环，不是性能优化成功。

| shape | dtype | PyTorch ms | 候选 ms | 加速比 |
| --- | --- | ---: | ---: | ---: |
| 64×64 | fp16 | 0.03916 | 0.16594 | 0.236× |
| 4096×4096 | fp16 | 0.02206 | 0.10110 | 0.218× |
| 1024×65536 | fp16 | 0.08727 | 0.10495 | 0.832× |
| 1024×65536 | fp32 | 0.21833 | 0.21751 | 1.004× |

完整 15 项数据见 [正式评测](eval-round-0001.json)，候选代码见 [narrow_copy.py](narrow_copy.py)，生命周期及最终复核见 [优化输出](optimize-output.json) 和 [运行状态](run-progress.json)。本轮只测量了从上一次诊断继承的 seed，`max_round=1` 随即停止，未产生新的优化版本。此前单独调试脚本给出的高倍率数字以 FlagGems Triton 实现为基线，且不属于正式 `eval_round`；不能与此处的 PyTorch 基线混用。

## 踩坑与复现要点

1. 仅设置 KGS 的 `--timing walltime` 不会改变 FlagGems benchmark 自身的默认 `kernel` 模式；该模式调用 `do_bench_npu`，在本机曾因 profiler CSV 的 `kernel_name` 列被 Pandas 推断为浮点而报 `.str accessor` 错误，正式报告没有 `case_id`。
2. 不要用 `PYTEST_ADDOPTS=--mode operator`：KGS 还会运行独立的 correctness pytest，它不接受 benchmark 专属的 `--mode` 参数。隔离副本现在只在 `benchmark/conftest.py` 读取 `KERNELGEN_FLAGGEMS_MODE=operator`，让性能测试走同步 walltime。
3. benchmark 源文件提交变化会改变 fingerprint。修改隔离副本后必须新建 KG workspace，不能复用旧 workspace 的冻结评测契约。
4. 容器内 KG 需要名为 `claude` 的可执行入口及本地 API 环境。专用入口 `/data/hanle/ascend-optimization/runtime/kg-controller/bin/claude` 是执行 `claude-native` 的包装脚本；仅做软链接或未加载本地 `env.sh` 会遇到登录错误。不要把令牌提交到 Git。
5. 复现前确认端口 `19652` 的 `/status` 为 `ok`、`timing=walltime`、`workers=1`、`KGS_FLAGGEMS_ROOT` 指向隔离副本，并等待其调度队列空闲。使用 `--seed-code-path` 指向本目录的候选源码，在**新 workspace** 中运行同样的 `kg run --mode simple_opt --definition narrow_copy --catalog-name flaggems-adapter-definitions --eval-server http://127.0.0.1:19652 --runtime claude --no-profile --max-round 1`。准确的模型、预算和参数见 `run-progress.json` 对应的服务器 workspace 中 `.kernelgen/run-request.json`。

下一轮应在固定 PyTorch 基线和同一 walltime 口径下，先降低小 shape 约 `0.10 ms` 的固定调用开销，再考虑 profiler 引导的完整优化实验。现有候选对部分大拷贝接近 PyTorch，但总体尚未超过 PyTorch。

本轮 profiling 已完成 metrics 和 instruction 两级采集。结构化原始证据见 `profiling-metrics.json`、`profiling-instruction.json`，对应请求体也已保留。中文诊断见 [分析报告](分析报告.md)，旧版紧凑提示见 `profiling-agent-prompt.md`。新版 [独立 profiling 分析器](../../../scripts/ascend_profiling_agent.py) 将同一个 case 的正式 Eval、真机 msprof、simulator 指令及候选源码关联并校验请求体，输出 `profiling-agent-case.md/json`；不会把不同采集的耗时相减，也不会在缺少可信上限时编造 Roofline。

## 先前 profile-enabled KernelGen 运行

在同一隔离 FlagGems 副本、KGS `19652`、PyTorch baseline 和 workload 下，使用 `kg run --profile` 启动 workspace `narrow-copy-kg-profile-20261005-3`，共运行两轮。KGS 在 **round 2 候选评测后** 对 5/15 个 timing workload 采集 `msprof metrics`，并生成 `.kernelgen/profile-analysis/round-0002.json`。这证明 profile 链路可用，但本次正式 profile 结果没有进入 round 2 的代码决策；旧 `--no-profile` 任务也只有一轮，不能把两个旧任务称为严格 A/B。

| 项目 | 无 profiling seed | 有 profiling round 2 | 最终独立复验 |
| --- | ---: | ---: | ---: |
| correctness | 33/33 | 33/33（其中 18 个主 correctness） | PASSED |
| timing workload | 15/15 | 15/15 | PASSED |
| 几何平均 speedup | 0.3396× | 0.4708× | 0.4615× |
| 最差 speedup | 0.1925× | 0.2860× | 以逐 case 结果为准 |

profile-enabled 候选保存在 [narrow_copy_profiled.py](narrow_copy_profiled.py)，round 1/2 的完整结果、profile analysis、ledger 和最终复验分别见 `profiled-round-0001.json`、`profiled-round-0002.json`、`profile-analysis-round-0002.json`、`profiled-ledger.json` 和 `profiled-final-verification.json`。候选通过按完整 launch signature 缓存已编译 launcher，round 2 相对 seed 在 host-bound workload 上减少约 `0.037–0.050 ms`；`::4` 大拷贝接近带宽边界，约 `1.01×`，总体仍低于 PyTorch。

这次实验表明 launcher 缓存能降低部分 host launch 开销，但不能证明正式 KGS profiling 反馈导致了该改进。候选整体仍低于 PyTorch native。严格 A/B 另以 [配对脚本](../../../scripts/run-narrow-copy-ab-910b.sh) 在新 workspace 执行，两组共用 [中性提示](../../../experiments/ascend910b/narrow_copy/ab-prompt.md)，只切换 `--profile`；必须检查第一轮分析记录早于第二轮改码，才可讨论 profiling 对优化的贡献。

Excel `验收看板_结果表.xlsx` 的 `narrow_copy` 位于第 29 行，华为列 `G29=0.0138`。该工作簿没有为此单元格提供 baseline、shape/dtype 集合、聚合方法或计时口径，因此不能与本实验的 PyTorch baseline 直接比较，也不能计算跨实验提升倍数。

## 两轮配对 A/B（2026-10-06 完成）

使用 [配对脚本](../../../scripts/run-narrow-copy-ab-910b.sh) 顺序运行两组独立 workspace：`narrow-copy-ab-20261005-no-profile` 和 `narrow-copy-ab-20261005-profile`。两组的 KG 参数、模型、两轮预算、workload、KGS `19652`、seed 和提示文件相同，仅切换 profiling 开关；归档 seed 的 SHA-256 为 `d7d243738cb72a27a1786d2078552cef4221f95230c5d63fb1e3d6f11d33141c`。PyTorch reference 在各组中重新测量，数字不强制相同。第一轮最终候选的源码哈希在两组恰好相同。

当次运行使用 `runtime/kg-controller` 中同内容的 seed、提示和脚本副本，精确路径见两份 `ab-*-run-request.json`。仓库版脚本改为直接读取本目录的 `ab-shared-seed.py` 和仓库中的 `ab-prompt.md`；重跑前仍需按仓库说明启动专用 KGS、准备 KG venv/Claude 环境，并将脚本中的 workspace 名换成新的实验编号。

无 profiler 组两轮均通过，geo mean 分别为 `0.3443×` 和 `0.3592×`，最终独立复验 `PASSED / 0.3580×`。有 profiler 组第一轮 `PASSED / 0.3772×`，对 4 个代表性 workload 采集真机 `msprof metrics`，并成功记录第一轮分析。第二轮的 launch-plan 缓存候选 15/15 timing case 通过，但 18 个 correctness case 中有 9 个发生运行错误，状态为 `PARTIAL_PASS`，没有有效的整体 geo mean；KG 回退到第一轮候选，最终独立复验 `PASSED / 0.3498×`。完整逐 case 表及原始 JSON 见 [配对报告](ab-report.md)。

服务器 `runner.log` 中，`record_profile_analysis` 首次因引用未登记的证据路径失败，重试后第 2324 行返回 `recorded=true, round_num=1, status=completed`；第二轮首次改写 `tmp/main.py` 在第 3715 行。因此本次确实走通了“先分析，再修改”的反馈顺序。分析指出小中型 case 的设备 kernel 仅约 `1.5–8 µs`，调用的固定开销占主导；target-side 自检显示修改后部分调用约 `85–91 µs`，但它仍未通过正式正确性。该改动不能作为可交付的优化版本，也不能说 profiling 提高了最终性能。

下一轮应先复现并修复第二轮 runs-kernel 的 9 个 correctness 运行错误，再以相同 15 case 和完整正确性重新评测缓存方案；随后重复配对运行以估计波动。现阶段只可得出：在此次 910B、walltime、两轮预算和既定 workload 下，最终有效 Triton 候选总体仍慢于 PyTorch `torch.narrow_copy`。Excel 的 `0.0138` 仍仅用于选题，不参与加速比换算。

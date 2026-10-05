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

本轮 profiling 已完成 metrics 和 instruction 两级采集。结构化原始证据见 `profiling-metrics.json`、`profiling-instruction.json`，中文诊断见 [分析报告](分析报告.md)，可注入 Claude 的紧凑提示见 `profiling-agent-prompt.md`。提示由 [独立 profiling agent](../../../../scripts/ascend_profiling_agent.py) 根据原始 JSON 生成，明确区分 msprof/simulator 诊断时间与正式 KGS walltime。

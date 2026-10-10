# 固定 master 的四版本评测

本轮正式对象为 `amin`、`matmul_bias_activation`、`narrow_copy`，均来自验收看板华为列低于 0.8× 的条目。四版本分别是固定 FlagGems master、原生 KG 无 profiler、原生 KG 有 profiler、我们根据分析修改的候选。原生 KG 两组使用同一基线、模型和两轮预算；我们的建议仅进入第四组。

## 1. 确认实验环境

当前独立实验目录为 `/data/hanle/ascend-optimization/goal-20261010`，FlagGems 固定在 `d6a8eec473517a3d68157b208eb9c057eb1d4c50`。控制器复用 `/data/hanle/ascend-optimization/runtime/kg-controller` 的 venv 和 Claude，执行时通过 `KG_SOURCE_ROOT` 显式选择新仓库源码。KGS 在 `tle_yy` 容器内使用物理卡 7，服务中为 `npu:0`。

```bash
root=/data/hanle/ascend-optimization/goal-20261010
project=$root/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
curl --noproxy '*' -fsS http://127.0.0.1:19655/status
```

`19655` 用于 amin 和 matmul；`19656` 是后续 narrow 的独立服务，使用只适配 case API 的 FlagGems 副本。新服务由 narrow campaign 在前序任务全部结束、队列空闲时启动，不覆盖旧服务。正式实验保持单卡串行，尤其不要在长诊断时启动新的 KG：KGS 的合同读取和诊断共用一个执行器，合同读取可能排队超过 60 秒。

只读检查后台进度：

```bash
"$py" "$project/tools/evaluation/inspect-campaign-state.py" "$root" \
  --server http://127.0.0.1:19655
tmux list-sessions
```

工具列出 KG 终态、轮次成绩、独立完整结果和服务队列。`recorded_operations` 是本地记录，KG 结束后可能残留历史条目；实际占用以服务队列为准。narrow 专用服务启动后可再加 `--server http://127.0.0.1:19656`。

## 2. 完整评测固定候选

以矩阵候选为例，输出目录使用新批次名：

```bash
"$py" "$project/tools/evaluation/evaluate-operator.py" \
  --operator matmul_bias_activation \
  --source "$project/operators/matmul_bias_activation/candidates/profiling-k256-dotacc-v4-20261010.py" \
  --output "$project/operators/matmul_bias_activation/reports/reproduce-v4-新批次" \
  --label optimized --timeout 1800
```

工具保存 server 状态、inspect、源码快照、SHA、请求和逐 case 结果。`amin` 必须完整通过 27 项、matmul 完整通过 42 项，不能只运行性能测试。amin 未修改 master 会在大输入遇到 `coreDim=131072` 启动超限，因此其可运行对照明确命名为“master 加启动兼容修复”，两组 KG 都从这份同一源码开始。

`.gitattributes` 将项目 Python/shell 源码固定为 LF，reports 下的原始证据禁止 Git 文本换行转换。下载后的源码、JSON 和 profiler 附件按原字节保存，跨平台 clone 后仍须通过 SHA 校验；不要用编辑器批量格式化这些证据。

## 3. 运行原生 KG 两组

确认前一组及其评测已经结束，再运行下一组。下面命令需要新的 workspace 后缀，不可直接覆盖正在运行或已有的 workspace。

```bash
export KG_RUN_ROOT="$root/kg-runs"
export KG_SOURCE_ROOT="$root/Ascend-Optimization/sources"
export KG_WALLTIME_PORT=19655 KG_PROFILE_PORT=19655
seed=$project/operators/matmul_bias_activation/candidates/flaggems-master-20261010.py
bash "$project/tools/evaluation/run-kg-operator-910b.sh" \
  matmul_bias_activation no-profile no-profile-新批次 2 "$seed"
bash "$project/tools/evaluation/run-kg-operator-910b.sh" \
  matmul_bias_activation profile profile-新批次 2 "$seed"
```

模型默认为 `deepseek-v4-flash[1m]`，运行器是 Claude Code。`--no-profile` 关闭优化反馈；FlagGems 的设备计时仍可能调用 profiler。`--skip-review` 跳过模型测试审核，完整正确性门禁继续执行。原生 profiler 自行选择真机 metrics 和模拟器 instruction；模拟器超时要保留，不能改写为成功采集。

KG 使用原生后台 worker，启动脚本等待其终态，可由 tmux 托管。不要使用当前版本有重入锁问题的 foreground 路径。合同超时后先核实 `.kernelgen/operator-lifecycle.json`：已有生命周期可 `kg resume`；生命周期尚未创建的工作区应先检查启动请求和进程状态，再恢复原请求，不能盲目 resume。

## 4. 独立复验与比较

```bash
"$py" "$project/tools/evaluation/finalize-kg-arm.py" \
  "$root/kg-runs/matmul_bias_activation-profile-新批次" matmul_bias_activation \
  "$project/operators/matmul_bias_activation/reports/kg-profile-新批次" \
  --label kg-profile
```

该工具等待的前提是 KG 已经终止；它核验通过轮次的身份、源码和 solution，导出最佳候选，再交给 KGS 完整复验。比较工具必须接收独立复验的 `*.result.json`，并核验同一合同、用例、设备与 benchmark 指纹：

```bash
"$py" "$project/tools/evaluation/compare-operator-versions.py" \
  --version master /绝对路径/master-1.result.json \
  --version kg-no-profile /绝对路径/kg-no-profile-1.result.json \
  --version kg-profile /绝对路径/kg-profile-1.result.json \
  --version optimized /绝对路径/optimized-1.result.json \
  --output /绝对路径/版本对比.md
```

输出包含中文 Markdown、汇总 JSON 和逐 case CSV。相对 PyTorch 是逐 case 加速比的几何平均；相对 master 使用两份候选的实际延迟计算，并附 PyTorch 参考漂移。当前 FlagGems core 实际是设备 kernel 计时，KGS 的 `walltime` 服务标签不能据此解释为完整 Python 调用耗时。

KG 原始 profiling 附件由 `archive-kg-evidence.py` 平铺到 `native-evidence/files/`，避免原工作区的多层 SHA 路径超过 Windows 路径上限。`archive-manifest.json` 的 `source_path`、`original_archive_path` 和 `archived_path` 保留完整映射，逐文件保存大小与 SHA。附件内容和内嵌 manifest 原字节不改写；查找附件时按清单映射到短路径。

三个算子的独立四版本结果齐全后运行总审计：

```bash
"$py" "$project/tools/evaluation/summarize-master-goal.py" "$project" \
  "$project/operators/低于0.8算子四版本闭环-20261010.md"
```

该工具重算原始结果，检查来源 commit、master 候选 SHA、同合同和完整用例数，并确认原生 KG 两组来自独立复验目录。任何一组缺失、路径越界或摘要与原始证据不同，均在写入总表之前停止。narrow campaign 最后会自动执行它。阶段成绩仍由各算子 README 单独记录。

## 5. 分析并继续优化

`profile-operator.py` 从完整通过的源码与同一性能 case 采集真机指标/指令，保存并校验原始附件。`../profiling/ascend_profiling_workflow.py` 绑定源码、合同和证据后生成中文分析；优化内核改名时使用 `--kernel-prefix mba_pipeline_kernel` 等实际符号，匹配不唯一时须补全符号。

amin 的 `probe-native-reduction.py` 检查原生精度 min 与浮点扩展、连续宽列和多行分块；matmul 的 `probe-wide-tiles.py` 检查更宽 tile、多缓冲与 UB 节省。两者用 `do_bench_npu` 的设备计时筛选，v5 生成器只使用数值检查通过的配置，随后重新运行完整套件。分块诊断和事件计时不能替代完整评测。最终保留完整通过且整体最快的候选，失败版本作为原始证据留存。

`collect-optimized-profiles.py` 从指定的完整通过结果和实际 case ID 采集 `PipeUtilization` 真机指标，绑定源码 SHA、合同和原始附件，自动生成中文体检报告。`run-grouped-v6-910b.sh` 在两组 KG 及首次独立复验完成后，串行采集 amin 的大输入指标、测试 matmul 的分组调度与较小输出 tile、完整复验新候选，再采集固定 v5 的指标。随后 `run-bf16-v6-910b.sh` 验证 amin 的 bf16 循环类型修复，与 PyTorch 和 v5 逐值比较后执行完整 27 项评测。narrow campaign 等这些任务结束后才启动同卡服务。各新版本仅在完整通过且整体更快时替换当前最佳候选。

矩阵乘的 `run-compiler-v7-910b.sh` 排在 narrow campaign 之后，测试规则矩阵的 mask 和后端搬运/缓冲选项。诊断仅接受与 v5 数值一致的配置；生成候选后完整复验、重新择优，并重建三算子总表。它使用本轮独立 tmux 会话和输出目录。

已完成结果：amin 四版本全部 27/27，依次为 0.296×、0.769×、0.499×、0.916×，bf16 v6 为 0.914×，保留 v5。matmul 四版本全部 42/42，依次为 0.443×、0.740×、0.710×、0.773×，编译流水线 v7 为 0.748×，保留 v6。

narrow_copy 的小连续输入在 PyTorch 中是 DMA 复制。task CSV 的 `kernel_name=N/A` 被 pandas 当作浮点空值，原计时器在 `.str` 过滤时崩溃。独立副本通过 `ascend-copy-timer.py` 只修复这一列的类型解析，保留设备任务耗时、采样与聚合；安装包与其他算子不变。KGS 19657 使用 `FlagGems-narrow-device-timing`，对应快照在 `operators/narrow_copy/reports/device-timing-contract-20261010/`。四版本共用 `--timing-scope device_task`，包括 DMA 与 AIV kernel。`run-device-timing-campaign-910b.sh` 执行 baseline、纯 Triton 快路径及原生 KG 两组；`finalize-device-campaign-910b.sh` 随后评测 DMA+Triton 混合候选、采集指标与分项计时，并生成最终四版本表。

修复 CSV 后，master 的 18 项正确性通过，但设备 profiler 同步卡住；原生堆栈和停止记录归档。上游 `BenchMode.OPERATOR` 的独立副本同样停在大输入 `[1024,65536]` 候选同步，故不是单纯计时模式问题。将 master 大网格拆为每次最多 8192 program 后，单次大输入的三个 dtype 均同步完成、逐值一致，最终回到设备任务计时，入口为 `run-grid-compatible-campaign-910b.sh`，两组 KG 从同一启动兼容基线开始。兼容基线保留原 master 索引计算与 1024 分块，必须与未修改 master 分开标注。

`evaluate-operator.py --timing-scope` 仅记录 benchmark 实际设置，不切换计时器。比较工具拒绝混合 device_kernel/device_task/walltime，设备指标报告与总表按各算子实际口径标注。

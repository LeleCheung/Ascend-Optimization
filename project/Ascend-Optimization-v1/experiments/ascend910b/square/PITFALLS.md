# 910B 上 KG + KGS 最小验证交接说明

## 目录与服务

- SSH：`ssh baai-910b` 经 JumpServer 连接目标主机 `10.0.0.8`。
- 现有容器：`tle_yy`。不要重建共享镜像，也不要停止其他用户的进程。
- 910B 上的 Git 仓库：`/data/hanle/ascend-optimization/KernelGen`。
- 容器内的隔离运行时：`/data/hanle/ascend-optimization/runtime/kg-controller`。
- 专用 KGS：`http://127.0.0.1:19650`；KG 6.7.0、KGS 6.5.0，仅使用 `npu:0`。
- 仓库内 Catalog：`/data/hanle/ascend-optimization/Ascend-Optimization/runtime/catalogs/catalog-square`；运行环境仍位于 `$root`。
- 运行工作区：`$root/runs/square-private-1` 和 `$root/runs/square-private-2`。

下文的 `$root` 指 `/data/hanle/ascend-optimization/runtime/kg-controller`。运行时虚拟环境使用 Python 3.11.15、torch 2.9.0+cpu、torch-npu 2.9.0.post2、flagtree 0.6.0+ascend.gitc286cba6、CANN 9.0.0 和驱动 25.2.0。请沿用现有虚拟环境和服务，不要向共享的系统 Python 安装依赖。

## Claude 与 MCP

用户提供的 DeepSeek API 文档中，模型名是 `deepseek-flash`，不带 `[1M]`。宿主机的 `/root/.claude/settings.json` 已调整为该名称，备份位于 `/root/.claude/settings.json.pre-kernelgen-model-fix`。不要提交该配置文件或打印令牌。

容器内的 `$root/bin-claude` 经 `/proc/1/root/root/.claude/settings.json` 读取宿主机配置，再启动 `$root/claude-native`（ARM64 Claude 2.1.284）；`claude` 软链接指向该包装脚本。此前的 `nsenter -m` 桥接看不到容器工作区，不应在此沿用。

KG 为 Claude 注册本地标准输入输出方式的 MCP 服务 `kernelgen.mcp_server.server`，由它向专用 KGS HTTP 服务提交评测。即使 KG 使用 `--no-profile`，该 KGS 仍通过 NPU profiler 做设备计时；这个参数只关闭优化过程中的性能分析反馈。KGS 另提供 `msprof` 的 `metrics` 和 `instruction` 两级分析能力，供后续实验使用。

## 复现步骤

1. 执行 `docker exec tle_yy curl -s http://127.0.0.1:19650/status`，确认 KGS 为 6.5.0、设备为 `npu:0` 且设备槽健康，并留意其他用户的负载。
2. 确认仓库内的 `runtime/catalogs/catalog-square` 只包含 square 算子。若目录不存在，先核对 `scripts/prepare-square-catalog.py` 的源码路径，再在容器内运行它。
3. 在 `tle_yy` 内运行 `bash /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/experiments/ascend910b/square/run.sh "$root/runs/square-<新编号>"`，每次使用**新**工作区。脚本默认每组计时三次；KG 返回 `SUBMITTED` 和 Worker PID 后会在后台继续运行。
4. 查看 `$workspace/.kernelgen/run-progress.json`、`run-events.jsonl` 和 `runner.log`；再核对 `stages/optimize/work/.kernelgen/evals/round-0001/result.json`、`stages/optimize/work/optimize_definition_output.json`、最终复验 JSON 及 `stages/optimize/attempts/01/result.json`。完整工作流还需要代码审核阶段成功，不能仅凭预检通过判断成功。

本工作流使用 `kg run --foreground` 曾对 `submission.lock` 重复加锁，卡在 `STARTING`；应使用默认后台提交。内置 `kernelgenbench` 目录含有 `__pycache__`，整体上传为 Catalog 时 KGS 返回 HTTP 422。私有 square Catalog 只保留清单和声明的算子文件；不要改动共享 Catalog。

## 已有证据与未解决问题

`square-private-1` 已生成、编译并评测 Triton square 算子。搜索阶段和最终复验在 `npu:0` 上均通过 27 条正确性用例及 9 条计时用例。27 条正确性用例是 9 种形状与数据类型组合各重复三次。搜索阶段几何平均加速比约 1.045 倍，但大尺寸 `[1024,1024]` 用例仅为约 0.24 至 0.34 倍。参照实现的一条延迟变化 4.36 倍，KG 因 `NEEDS_RETEST / TIMING_DRIFT` 未确认最终结果，也未完成代码审核。证据见 `reports/ascend910b/square-private-1`。

`square-private-2` 使用每组计时三次，Catalog 审核阶段通过全部 36 条用例，但 Coder 持续运行探索性调试扫描。应用户在 2026-09-29 提出的收尾要求，该运行在优化阶段取消，没有完成正式候选轮次或最终复验。最终 `run-progress.json` 状态为 `CANCELLED`（两个阶段完成、一个阶段取消）。后续请使用新工作区，不要对这次已取消的运行执行 `kg resume`。

Coder 调试任务可能超过 KGS 的 32 个归档文件上限：一次 `msprof` 探索任务虽然退出码为 0，KGS 仍以 `artifact count exceeds 32` 标记为 `FAILED`。应缩小调试扫描范围，并区分诊断任务失败与正式候选评测失败。单次有噪声的运行不能用于宣称稳定加速比。

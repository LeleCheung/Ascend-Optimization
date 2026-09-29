# KernelGen Workflow 小重构 A100 E2E（2026-09-06）

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

本轮覆盖的路径未发现行为回归：双 Coder、双 epoch、legacy KB 合并、跨 epoch seed、综合赢家身份、最终停止与资源释放均通过。两次 finalize-only 的公开输出一致，实测产物内容不变；重构前版本对同一批真实 ledger 的公开结果投影与当前输出完全一致。该结论限定于以下配置，不是对所有模式的无条件等价保证。

## 被测版本与运行方式

被测 KG 为 `dev@00228f8e85f416be4415cd2391158fa32e95aec4`（已合入 !74–!78，包 release 为 KG v6.2.0），配套 KGS 为 `v6.3.0@fecfb0391c9d57832ac887e86f799a985e52c8bb`，Protocol 为 `/status.api_version=v6.2`。已核对 KG 锁定清单、KGS `compatibility.yaml` 和 native `kernelgenbench` Catalog manifest；未修改源码、协议、release 或 tag。

最初按 inventory 检查 H800 时，8 张卡均被 sglang 调度进程占用，因此未在该机器启动服务或算子测试。用户随后指定本地 `kernelgen-nvidia-cu128` 容器。该容器实际镜像为 `kernelgen-nvidia-cu132-nsight2026.1:base`，主机为 `bm-baai-dx-zone1-d-a100-40g-2-106`，实际设备是 A100-SXM4-40GB，不是 H800。启动前 8 张卡均无计算进程，本次仅使用物理卡 0、1。

采用宿主机 Agent + 本地容器 KGS，通过本机 loopback `127.0.0.1:21448` 通信。容器使用 host network，KGS 在容器中直接启动；本轮未使用 SSH proxy，也不包含 `kg server start --target local/remote` 生命周期命令验收。Agent 通过 `kg run` 提交。目标身份来自 KGS `/status`，KG 现有规则将 `NVIDIA A100-SXM4-40GB` 规范化为 ledger 中的 `A100`。

Server 解释器为 `/data/jiabei/venvs/kernelgen-cu132-torch211-triton360/bin/python3`，实测 Torch 2.11.0+cu130、Triton 3.6.0、CUDA runtime 13.0、driver 580.126.20；backend=cuda、timing=triton、2 个 device slot、4 个请求 worker。Agent Runtime 为 Codex CLI 0.153.4、model=inherit，没有核实到可独立记录的最终 provider 模型名。Agent lease 容量和 KernelGen 权重均为 2。没有安装、升级或替换任何 Python/npm 包或受保护运行时，仅创建了继承现有包、without-pip 的专用 Agent venv，并在其自身 site-packages 固定两侧测试源码路径；环境前后由 KGS Debug Job 复核一致。

## 验证配置与结果

正式证据位于本地忽略目录 `runs/kg_workflow_refactor_a100_20260906/` 的 `verified/` workspace。命令通过专用 Agent 解释器执行，完整 argv 保存在 `launch-command.json`：

```bash
python3 -m kernelgen.cli config set run.max-workers 2 --eval-server http://127.0.0.1:21448
python3 -m kernelgen.cli run --mode kernelgen --definition kernelgenbench_neg \
  --catalog-name kernelgenbench --target-hardware 'NVIDIA A100-SXM4-40GB' \
  --eval-server http://127.0.0.1:21448 --runtime codex \
  --n-parallel 2 --n-epoch 2 --min-rounds 1 --max-round 1 --no-profile \
  --workspace /path/to/new/verified
```

独立解释器在清除 `PYTHONPATH` 后仍导入本轮 KG/KGS worktree；同时记录了实际 MCP 子进程使用该专用解释器。完整 host suite 为 1084 passed in 36.85s。设备门禁为 square reference-as-solution Preflight/Eval 通过，随后执行完整 neg Workflow；未裁剪 Catalog workload。

- 共享 Analyzer 只执行一次；2 个 epoch 共 4 个独立 Coder，各完成 1 个 measured round，每个 round 的 36/36 workload 均通过，具有有效 timing、conclusion 和非取消 STOP。全部 Coder 输出与 `.best_kernel.py` 均和 ledger 相符。
- 第二轮两个 Coder 的实际 prompt seed 与第一轮 ledger best 逐字一致；两个 synthesis prompt 的全局赢家路径、round 与代码均和当时 ledger 选择一致。静态检查候选的核心计算为 Triton load/neg/store，Torch 仅分配输出，不是框架算子 fallback。
- 4 次 Preflight 和 4 次 Eval 的 operation ID 登记与注销逐一配对；KGS 查询 8 项均为 SUCCEEDED，本地 active operation 登记为空。最终 9 个 scope 均为 SUCCEEDED/COMPLETED，CLI worker 退出码为 0，Agent lease 释放。
- 运行中实测 active=2，未超过两个 slot；结束为 device_slots=healthy=available=2，active=waiting=checking=broken=0，incidents=recovered=0。

逐 Coder 分数和全局 best 由脚本从 ledger/history 生成，见 [summary.md](../../runs/kg_workflow_refactor_a100_20260906/summary.md) 与 [summary.json](../../runs/kg_workflow_refactor_a100_20260906/summary.json)。本轮每个 Coder 仅 1 轮，用于流程回归，不作为性能调优成效或跨设备性能对比。

## Finalize 与重构前对照

调用公开 `KernelGenWorkflow.finalize_completed_epoch(inp, 2)` 两次，真实执行现有最终化逻辑。两次结果相同；所有 ledger、Coder output、best code、Analyzer checkpoint 和 synthesis checkpoint 的 SHA-256 均未变化，没有新增 Analyzer/Coder 调用或设备 Eval。legacy KB 路径仍各执行两次知识合并模型调用，这是既有语义；“结果幂等”不等于整个最终化流程没有任何模型调用。

随后用重构前 `KG v6.2.0@a90b154b9c0188410a66190c5c9c6bd30caef0c2` 的恢复/选择代码读取同一批实测 ledger，其公开 `KernelGenOutput` JSON 与当前最终化输出完全相同。该项是相同证据下的确定性投影对照，不是重构前、后各跑一次完整模型 E2E。额外验证了从 1R checkpoint 加载 resume 的 Analyzer、directions 和 best 身份，但没有重新执行完整的中断续跑 campaign。

本轮没有覆盖 Profile、V1 Knowledge reviewer、fork、运行中取消、完整中断续跑及 CLI Server 管理命令。这些边界不能因本次普通执行通过而写成已经真机复验。

## 导入问题与证据保留

首次 `current/` workspace 的主进程导入路径正确，但 MCP 配置使用系统 `/usr/bin/python3`；去除 `PYTHONPATH` 后该解释器导入旧主工作区，且此次运行没有新版本应产生的 Eval/operation 结构化事件。因此将其保留为导入隔离不足的现场，不计入正式 E2E 通过证据。之后使用专用解释器在全新 `verified/` workspace 重跑，确认实际 MCP 解释器、完整 Eval 事件和 operation 登记均正确；未修改旧 ledger 或覆盖旧 workspace。

此次还修正了验收脚本自身对 DebugJob 序列化、NextVerdict 字段及原始设备名/规范化身份的假设，没有把这些脚本错误计作产品回归。正式脚本为证据目录中的 `e2e.py`、`inspect_run.py`、`finalize_check.py`、`baseline_projection.py`、`summarize.py`。结果包括 `final-status.json`、`final-history.json`、`operation-states.json`、`seed-check.json`、`finalize-check.json`、`baseline-projection.json`、`kernelgen-output.json`；最后一个文件来自公开 finalize 返回值，不冒充原 CLI launcher 自动输出的文件。

停止前后保留 Server 日志、环境 Debug Job、scheduler 快照与进程检查。属于本轮的 Agent/Coder/MCP 均已退出，临时 KGS 经 PID 启动身份核对后关闭；容器和原有环境保持不动，原始实验数据保留。报告链接中的 runs 文件仅存在于本地实验归档，不随 Git clone 分发。

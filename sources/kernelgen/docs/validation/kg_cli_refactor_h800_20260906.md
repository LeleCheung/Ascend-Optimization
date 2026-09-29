# KG CLI 小重构 H800 验证（2026-09-06）

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

核心流程通过：单任务 SimpleOpt、取消后原 workspace 续跑、YAML Batch、KernelGen 双 Coder 单 epoch、真实终端 Ctrl+C、API 在途取消恢复以及 Profile。KernelGen 首次执行暴露 tolerance mode 默认值从空字符串变为 None 的回归，已在原 feature 修复；失败目录保留，使用新目录复验成功。

## 代码与环境

初始 KG 为 `feat/runtime-observability-control@d5d9471e`，KGS 为 `feat/cancellable-operations@c287ee2`，两个 worktree 起始均干净。主工作区的 inventory 修改和其他未提交文件未改动。按用户授权直接在现有 feature 中按主题提交。

模型流程使用 `KG v6.1.2@903a9832`（此前单任务和 Batch 使用 `64a847aa`）、`KGS v6.2.4@965316f53765b4cb4da399e8002232f084f8d3f4`、`Protocol v6.2` 开发快照。最终 KGS feature 的 `0e1f45bca41e9bebae7214ac31c8119e44a33329` 仅追加测试 mock 和部署文档修正，生产代码与上述已测 commit 无差异；KG 锁定清单指向该最终 commit。没有创建或移动 release tag。

连接取自当时 `tests/multi_device_batch_hosts.conf` 的 nvidia 行（现行清单为 `tests/hosts.md`）：`10.0.25.65`，容器 `codex_fib_nvidia_20260728`。设备事实来自远端 /status 与 Debug Job：NVIDIA H800，CUDA，Triton 3.6.0，Torch 2.8.0a0+5228986c39.nv25.05，CUDA runtime 12.9，driver 580.105.08；可见卡 0、1，2 个 slot，4 个 KGS 请求 worker，Agent endpoint 容量 2。SimpleOpt lease 权重 1，KernelGen 的 n_parallel=2 权重 2。Runtime 为本地 Codex CLI 0.153.4，model=inherit，没有显式覆盖 provider 模型；响应中未提供可核实的最终模型名。

远端使用新建的 system-site-packages venv，未修改系统运行时。新增轻量包为 annotated-doc 0.0.5、fastapi 0.141.1、starlette 1.6.0、typing_extensions 4.16.0、typing-inspection 0.4.4、uvicorn 0.52.4，并安装当前 KGS editable 包；pip 24.0 来自 venv 创建。受保护包安装前后版本未变化；Triton 由厂商 FlagTree 提供，distribution metadata 不叫 triton，模块版本另由 /status 和 Debug Job 核实。

## 验证结果

KG 完整 host suite：1012 passed。KGS 完整 host suite：317 passed、28 skipped（缺少 Torch 或设备 E2E 未显式启用）。测试显式核对两侧导入路径；KGS 单独使用完整 worktree 链接及干净的 FlagGems d64794e63b502cb836bc015a92a62c42de4be05a，修正了初次测试的顶层 tools 包冲突和相对路径问题，没有将这些环境错误当作生产代码缺陷。H800 补跑 operation、Bundle、management 和 PID 测试：38 passed，包括本机缺少 Torch 而跳过的 8 项 API 测试。

H800 API 门禁包括 square reference-as-solution Preflight/Eval、RUNTIME_ERROR、三个并发 Eval 在两个 slot 上排队与释放、运行中 Preflight cancellation 与强探针恢复。最终无 broken slot；受控取消产生的 incident 被同一 slot 成功恢复。单 workload NCU metrics Profile 完成，结果和下载的 artifact 已回收。Bundle PUT/GET/HEAD 与重复上传通过，evaluation_binding 仍为 false。

优化结果由脚本读取 CLI JSON 和 ledger 投影生成，最终输出独立保留，不手工录入加速比。见本地 [汇总 Markdown](../../runs/kg_cli_refactor_h800_20260906/summary.md)、[状态 JSON](../../runs/kg_cli_refactor_h800_20260906/cli-statuses.json) 和 [history JSON](../../runs/kg_cli_refactor_h800_20260906/cli-histories.json)。成功任务包括 simple、cancel-resume、batch 的 square/neg 和 kernelgen-retest；kernelgen 为保留的失败现场，foreground-cancel 为预期取消。KernelGen 的 2 个记录分别属于两个 Coder，不合并成一个 Coder 的轮次轴。

取消证据中 MODEL_INVOCATION_COMPLETED 先于 RUN_CANCELLED，session 被完整保留；续跑沿用原 workspace 和参数。真实终端 Ctrl+C 在等待 lease 时协作退出 130。所有最终 scope 均非 RUNNING/QUEUED，status/history 的 best 值一致，progress 文件不持久化性能与轮次缓存。

## 证据与清理

本地证据根目录为 `runs/kg_cli_refactor_h800_20260906/`，包含测试驱动脚本、YAML、每个任务独立 workspace、ledger、最终输出、best kernel、模型日志、Server 日志、API JSON 与 Profile artifacts。远端新目录为 `/data/xuyao/kernelgen_e2e_20260829/kg-cli-refactor-20260906/`；代码仅通过 Gitee 临时测试分支拉取，没有上传或覆盖 KG/KGS 源码树。

最终 commit 的新实例 h800-final 已通过首次启动、重复 start 返回 ALREADY_RUNNING、doctor、status --json、stop、configure 将请求 worker 从 4 改为 3，以及 install-flaggems 的精确 revision 校验。FlagGems 复用机器已有干净 checkout，没有替换其 revision；重启后 /status 的请求 worker 为 3，device_slots=healthy=available=2，active=waiting=checking=broken=0。Debug Job 复核最终 checkout 的导入路径和受保护运行时版本。

h800-refactor 与 h800-final 均已停止，本地代理退出；CLI 索引中的所有本轮 Agent/Coder 均非活动状态。临时 `test-kg-cli-refactor-h800-20260906` 分支已从本地、Gitee 和目标 clone 删除，目标 checkout 转为 detached 证据快照，提交仍由 `feat/cancellable-operations` 保存。host 测试用的临时 FlagGems worktree 已移除，原 FlagGems 工作区未改动。实验目录、失败现场和日志均保留；最终状态见 [final-server-status.json](../../runs/kg_cli_refactor_h800_20260906/final-server-status.json)、[final-environment.json](../../runs/kg_cli_refactor_h800_20260906/final-environment.json) 和 [final-stopped.json](../../runs/kg_cli_refactor_h800_20260906/final-stopped.json)。

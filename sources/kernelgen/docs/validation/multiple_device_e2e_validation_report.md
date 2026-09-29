# 多芯片 BatchSimpleOpt E2E 验证报告

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

本文只归档 KernelGen BatchSimpleOpt 的历史环境快照、E2E 结果、加速比和失败
原因。当前实验的安装、启动、监控、验收和回收命令统一见
[多设备实验与 E2E 验收手册](../operations/multiple_device_experiment_runbook.md)。

KernelGen Server 的部署与自测见
`kernelgen_server/docs/multiple_device_deploy_tutorial.md`；Server、Schema、
reference 和计时兼容性结果见
`kernelgen_server/docs/multiple_device_eval_validation_report.md`。

## 2026-09-13 Stage 编排重构昇腾 CLI 验证

公共 StageRunner 的纯分支 host 回归为 267 passed，临时 CLI 集成组合为 268 passed。真实 relu 两轮搜索通过，首次最终复验因 reference profiler 无效计时而失败；保留证据并显式重测后，经 CLI resume 完成五阶段，最终独立复验 33/33 PASSED、0.915892×。再次 resume 五阶段全部复用，20 个业务文件哈希不变，无新模型或 KGS 执行调用。真实抽取和 Catalog Review 跳过，新 JSON Schema 未纳入；精确代码组合、状态限制及恢复过程见 [专项报告](stage_orchestration_ascend_20260913.md)。

## 2026-09-11 Catalog 生命周期昇腾验证

`feat/catalog-optimize-ascend` 配合 KGS `feat/operator-bundle-execution`，完成 relu 的真实抽取、Catalog 审核、Bundle 上传及目标验证、SimpleOpt 和代码审核。五阶段 SUCCEEDED，3 个搜索轮次及最终独立复验均为 33/33 PASSED；公开 lifecycle 取消和同 workspace 续跑通过，原全卡 KGS 已恢复。两个非阻塞审核 P1、精确版本、自动生成的性能结果和验收边界见 [专项报告](catalog_optimize_ascend_20260911.md)。

## 2026-09-08 最终 best 独立复验 A100 验证

`feat/final-best-retest` 的 KG 开发快照与 KGS main 完成 219 项相关 host 检查，以及 A100 上 Codex SimpleOpt、强制最终复验、额外复测和恢复验证。最终输出及多 Coder 汇总采用最新确认时延；搜索 ledger 保留原始测量，未确认候选不能通过恢复路径晋升。实际版本、自动汇总的加速比、范围限制和原始产物见 [专项报告](final_best_retest_a100_20260908.md)。

## 2026-09-06 KernelGen Workflow 小重构 A100 验证

在用户指定的本地 `kernelgen-nvidia-cu128` 容器（实际 A100）上，以 KG `00228f8e` / KGS v6.3.0 / Protocol v6.2 完成双 Coder、双 epoch 的真机回归及双次 finalize-only 校验；4 个独立 measured round 均为 36/36 PASSED，seed/赢家身份、operation 释放和终态通过。重构前代码对同一批实测 ledger 的公开结果投影与当前输出一致。首轮 MCP 导入隔离不足的现场不计入通过证据，正式结果来自独立解释器的新 workspace。配置、范围限制与原始证据见 [专项报告](kernelgen_workflow_refactor_a100_20260906.md)。

## 2026-09-06 KG CLI 小重构 H800 验证

本轮完成共享参数、ledger 状态投影、协作取消及 KGS 管理代码迁移后的 host 与 H800 验收。SimpleOpt、取消后续跑、YAML Batch、KernelGen 双 Coder 单 epoch、在途取消恢复及 Profile 均通过；真机发现的 tolerance mode 默认值回归已修复并在新 workspace 复验。完整 commit、环境、测试计数、失败现场和结构化产物见 [专项验证报告](kg_cli_refactor_h800_20260906.md)。

## 2026-09-05 NVIDIA H800 E2E 前置三项复验

本轮依次完成精确 FlagGems 评测门禁、KernelGen 进度/取消 host 回归和 KGS 运行中 operation 取消真机验证。第一项使用 `KG v6.1.2@32e4f09515cbf9a77a78ec9640f741e04689239a / KGS v6.2.4@9f9ae9cf40f6094fbff47c13ec14540a40e2c9d4 / Protocol v6.2`，H800 容器实际加载干净的 FlagGems `d64794e63b502cb836bc015a92a62c42de4be05a`。`addmm_` native reference-as-solution 为 51/51 `PASSED`，adapter/native preflight 均为 15/15 `PASSED`，FlagGems adapter Eval 为 51/51 `PASSED`、geo mean 0.705947x；该数值只验证既有 FlagGems 实现可被正确注入和评测，不作为优化合格结果。结束后 scheduler 为 `device_slots=healthy=available=8`、`active=waiting=checking=broken=incidents=0`。

同环境的 `gcd` native reference-as-solution 为 35/35 `PASSED`，adapter/native preflight 均为 8/8 `PASSED`，但 adapter Eval 返回 `RUNTIME_ERROR`：FlagGems `tests/test_gcd.py` 仍通过 `use_gems()` 调用 `torch.gcd`，未使用候选感知的 `resolve_gems_op`，所以 call coverage 得到 `operator=null`。本轮没有放宽 coverage、改写 reference 或增加算子旁路；缺口已记录到忽略目录 `kernel_todo_v2/pytest_conversion_protocol_gaps.md`，待统一确认 pytest 转换方案。

第二项在 `0f8687a8b18ec1a52abb83721a9320cd81d7d4b1` 将 legacy KB merge 移入独立 `1R/knowledge-merger` scope，根进度在模型调用期间保持 `SYNTHESIZING`，并让 `RunCancelled` 绕过 reducer 的普通失败 fallback。定向回归为 48 passed，隔离导入当前 KG/KGS worktree 的完整 host 回归为 998 passed；新增用例覆盖 knowledge merger 成功终态、取消终态、双 epoch 根阶段稳定性和 reducer 取消传播。

第三项使用 `KG v6.1.2@0f8687a8b18ec1a52abb83721a9320cd81d7d4b1 / KGS v6.2.4@ad143f2d5b5f065f532c95c600eaf42885e2f008 / Protocol v6.2` 的发布后开发快照，通过一次性 Gitee 分支 `test-cancellable-kg-cli-h800-20260905` 部署 8-slot CUDA/Triton Server。`kernelgenbench_square` 的受控 preflight 在 `cuda:0` 进入 `RUNNING` 时 scheduler 为 `active=1, available=7`；`kg cancel` 成功转发 1 个 operation、失败数为 0，原 POST 返回 `OPERATION_CANCELLED`，KGS operation 终态为 `CANCELLED`，KG 根任务终态为 `state=CANCELLED, stage=CANCELLED`，worker 记录退出码 130。原 slot 强探针通过后 scheduler 恢复为 `device_slots=healthy=available=8`、`active=waiting=checking=broken=0`、`incidents=recovered=1`。临时远端服务目录、测试 worktree 和 Gitee 分支均已删除；结构化结果和回收前 Server 日志保存在忽略目录 `runs/kg_cli_daily_nvidia_20260905/cancel_e2e_20260905/`。本轮未安装、升级或替换 Torch、Triton 或 CUDA 运行时。

## 2026-09-05 NVIDIA H800 远端 deployment env 验证

本轮使用 `KG v6.1.2@32e4f09515cbf9a77a78ec9640f741e04689239a / KGS v6.2.4@9f9ae9cf40f6094fbff47c13ec14540a40e2c9d4 / Protocol v6.2`，验证 `--remote-env-file` 是否能隔离承载 H800 容器的 GitHub HTTPS 和 Gitee SSH 代理配置。测试前确认原实例 8/8 slot healthy/available 且 `active=waiting=0` 后停服；新实例 `h800-env-e2e-20260905` 使用独立远端根目录 `/data/xuyao/kernelgen_e2e_20260829/kg-cli-env-20260905/`、端口 `18329/21329` 和机器侧 `deployment.env.sh`，未接管或删除其他 checkout。

Gitee 路径通过：`kg server start` 自动加载 env 中的 `GIT_SSH_COMMAND`，经 inventory 指定的 Gitee CONNECT 代理从空目录完成 KGS clone、精确 detached checkout 和 editable 安装。目标 checkout 的 origin、HEAD、clean 状态及容器 ownership 均通过校验，独立 Python 进程实际从该新 checkout 导入 `kernelgen_server`。后台 KGS 未继承 `GIT_SSH_COMMAND` 或测试 HOME；启动后为 KGS v6.2.4、Protocol v6.2、CUDA/Triton、8 workers，scheduler 为 `device_slots=healthy=available=8`、`active=waiting=checking=broken=incidents=0`。

GitHub 路径也已通过：inventory 更新为经 H800 实测可用的鉴权代理端点，凭据只保存在容器内权限为 `600` 的机器侧 env，不写入 KG 配置、仓库或日志。一次性 `curl` 返回 HTTP 200，FlagGems `git ls-remote` 将目标分支解析到锁定 commit `d64794e63b502cb836bc015a92a62c42de4be05a`。首次完整 clone 在代理传输的 `index-pack` 阶段中断；确认磁盘、内存和仓库体积正常后，在机器 env 中通过 `GIT_CONFIG_*` 强制 Git HTTP/1.1，第二次 `kg server install-flaggems` 在 17 秒内完成。目标 checkout 的 origin、HEAD、clean 状态和容器 ownership 均通过校验，停止状态下重复安装直接复用同一 checkout。

安装 FlagGems 后再次启动 KGS，`server doctor` 通过且 `/status.evaluation_adapters` 同时包含 `native` 和 `flaggems`；scheduler 仍为 `device_slots=healthy=available=8`、`active=waiting=checking=broken=incidents=0`。后台 KGS 只继承精确 `KGS_FLAGGEMS_ROOT`，不继承 `HTTPS_PROXY`、`GIT_SSH_COMMAND`、`GIT_CONFIG_*` 或测试 HOME。重复 `server start` 返回 `ALREADY_RUNNING`；交接前已安全停止远端 KGS 和本地代理，原 `h800-final-20260905` 也保持停止。

本地完整 host 回归使用本 worktree 和配套 KGS client worktree，结果为 `996 passed`。测试源快照和机器侧配置副本位于忽略目录 `runs/kg_cli_env_h800_20260905/`；未安装、升级或替换 Torch、Triton、CUDA 或其他厂商运行时。

## 2026-09-05 NVIDIA H800 `kg` CLI 日常使用验证

本轮使用 `KG v6.1.2@af9c4bb8616b5094365fa9f274fcb0a7ceed7b1a / KGS v6.2.4@ad143f2d5b5f065f532c95c600eaf42885e2f008 / Protocol v6.2` 的发布后开发快照；KGS 通过临时 Gitee 分支 `test-cancellable-operations-nvidia-20260905` 部署到 NVIDIA H800 容器，仅监听远端 loopback，并由本地 SSH stdio HTTP proxy 访问。Server 使用 CUDA/Triton、8 个设备 slot 和 16 个请求 worker；Agent 使用 Codex CLI 0.149.1，按 endpoint 设置 `run.max-workers=1/2`。未安装、升级或替换 Torch、Triton 或 CUDA 运行时。

单任务后台 SimpleOpt、状态、结构化事件日志、原始日志、历史 best 查询、取消与续跑均通过。`kernelgenbench_square` 在 1 轮中 36/36 workload 通过，best geo mean 为 0.981114x；自定义 `warmup_ms=25`、`benchmark_ms=10`、`num_trials=2` 同时写入请求和运行时上下文。`kernelgenbench_neg` 的运行中取消在完整 Codex turn 后生效，随后原 workspace 复用两轮 ledger 续跑成功，best geo mean 为 1.004607x。排队取消未启动 Coder，直接以 130 退出；在途 preflight 取消向 KGS 转发 1 个 operation，任务收到 `RUN_CANCELLED`，`cuda:5` 强探针恢复后 Server 回到 8/8 healthy、available。

YAML Batch 在 Agent 并发 2 下同时运行 `kernelgenbench_square` 和 `kernelgenbench_neg`，两项均完成 1 轮并分别得到 0.983725x 和 0.914543x；前者的 reference Triton、guidance 以及 `30/12 ms、2 trials` 单项覆盖均进入 Coder 与评测上下文。额外提交的 `kernelgen --n-parallel=1 --n-epoch=1` 在两个 Batch lease 占满时保持排队，lease 释放后自动启动并以 0.968703x 完成。父 Batch 取消也覆盖了一个运行 child 和一个排队 child，最终均为 `CANCELLED`。请求 `n_parallel=3` 超过 endpoint 上限 2 时在提交阶段拒绝；前台未知算子同步退出 2，并保留 `INFRASTRUCTURE_ERROR` 状态。

本轮还确认五个 CLI 交付或观测问题：KGS 普通 wheel 不包含完整 `kernelgenbench` Catalog；workspace MCP 命令固定为 `python3`，因此只调用虚拟环境内 `kg` 的绝对路径但未激活 PATH 时会误用系统 KG；远程 `server start` 的默认 Catalog 会触发额外 GitHub clone，且 `server configure` 不能修正 Catalog；首次用同版本 editable KGS 替换已有安装时，启动进程保留旧 editable import 映射，需要重新执行一次；`kernelgen` 根任务已为 `SUCCEEDED` 后，`1R/shared_analysis` 子 scope 仍残留 `RUNNING/MODEL_INVOCATION`。前三项已在后续开发提交中完成代码修复和 host 回归，尚未重新执行设备 E2E。第四项已改为在独立的新 Python 进程中验证安装后的 import 映射，并在 H800 上从 KGS `v6.2.1` 的旧 editable checkout 单次 `kg server start` 自动恢复到锁定的 KGS `v6.2.4@9f9ae9cf40f6094fbff47c13ec14540a40e2c9d4`；启动后 8/8 slot healthy/available，Debug Job 确认实际 import 来自目标 checkout，受保护运行时版本未变化。第五项已补齐 shared analysis、synthesis 和 knowledge reviewer scope 的成功、失败与取消终态，并将根进度细化为结构化 epoch/stage；host 回归已覆盖双 epoch 阶段序列和残留 scope，尚未重新执行设备 E2E。测试工作区及原始日志保存在 `runs/kg_cli_daily_nvidia_20260905/`。

## 2026-09-02 KernelGen v6.1.2 / KernelGen Server v6.2.4 发布验证快照

当前推荐组合为 KernelGen `v6.1.2` 与 KernelGen Server `v6.2.4`，协议条件为 Server `/status.api_version=v6.2`。KGS `v6.2.4` 的 release commit 为 `9f9ae9cf40f6094fbff47c13ec14540a40e2c9d4`，功能基线为 `ec053859bbbca0fe91b78895e496121b8869fc6f`；该基线修复 FlagGems pytest 子进程屏蔽 system site initialization 的问题。KGS 根目录 `compatibility.yaml` 将 `v6.0` Catalog 固定为 adapter 默认模式、`v6.2` Catalog 固定为 native 默认模式，并把本组合作为 KG `v6.1.2` 的最新已验证 KGS；FlagGems 固定为 `YaooXu/FlagGems` 的 `feat/new-api-for-kernelgen-server@d64794e63b502cb836bc015a92a62c42de4be05a`。

精确 release tree 的 host 验证结果为：KernelGen `892 passed`；KernelGen Server 使用 detached、干净的 FlagGems `d64794e63b502cb836bc015a92a62c42de4be05a`，结果为 `283 passed, 20 skipped`，skip 来自当前 host 未安装 Torch 或显式关闭的设备 E2E。本轮未安装或升级 Torch、Triton、厂商运行时或其他 Python 包，也没有新增精确 release pair 的真机 E2E；目标芯片部署仍须执行启动探针、reference-as-solution、preflight、错误路径和资源释放检查。

## 2026-09-01 KernelGen v6.1.1 / KernelGen Server v6.2.3 发布验证快照

当前推荐组合为 KernelGen `v6.1.1` 与 KernelGen Server `v6.2.3`，协议条件为 Server `/status.api_version=v6.2`。KGS `v6.2.3` 的 release commit 为 `79f1d01`，直接父提交和功能基线为 `8a97d8a`；后者修复 FlagGems case listing 的并发执行和 600 秒 client timeout，release commit 只同步包版本、`/status.server_version` 与版本测试。

精确 release tree 的 host 验证结果为：KernelGen `892 passed`；KernelGen Server 在干净 detached FlagGems `d64794e` 上为 `282 passed, 20 skipped`，其中 skip 来自当前 host 未安装 Torch 或显式关闭的设备 E2E。本轮没有安装或升级 Torch、Triton、厂商运行时或其他 Python 包。

沐曦 MetaX C550 上的 Codex runtime SimpleOpt 功能 E2E 使用 KernelGen `8292f59` 与 KGS `v6.2.2@934c964`，`gcd` 35/35 workload 通过，9 轮后 best geo mean 为 0.680651x，结束后 8/8 slot available 且 `broken=incidents=0`。该结果证明 v6.1.1 所含 Codex runtime 祖先提交的设备闭环，但不是当前精确 release pair 的设备复验；本节不把它表述为 `v6.1.1 + v6.2.3` 真机结果。

## 2026-08-27 沐曦 KernelGenBench v6.2 SimpleOpt E2E

本轮验证 KernelGenBench 从 v5 Catalog 迁移到 v6.2 native per-operator package 后的
完整链路。KernelGen 基于 `feat/schema-v6.2` 的 `acee7ae` 加本轮改动；KernelGen
Server 使用 `feat/v6.2-native-oracle` 的 `ab849a5`，通过临时 Gitee 分支
`test-v62-kernelgenbench-simple-opt-20260827` 部署。迁移覆盖 210 个算子、57,603 个
correctness workload 和 28,198 个 timing workload；原 workload JSONL 内容保持不变。

本地 Agent 通过 SSH stdio HTTP proxy 访问沐曦容器内仅绑定 loopback 的 Server。
Server backend 为 `metax`、timing 为 Triton，8 个设备 slot、
`max-workers=16`；实验前后均为 `healthy=8`，
`active=waiting=checking=broken=0`、`incidents=0`。远端软件为 Python 3.12.11、
Torch 2.8.0+metax3.7.2.0、Triton 3.6.0；本地 Claude Code 为 2.1.238。
未安装或升级任何 Python 包、Torch、Triton 或厂商运行时。

`kernelgenbench_cos` 的 reference-as-solution 先通过 preflight 和 36/36 workload
评测，geo mean 为 0.999800x。正式 SimpleOpt 使用 `env.opus.sh` 中的
`claude-opus-5[1m]`；`env.sh` 中的 `deepseek-v4-flash[1m]` 因当前 token 无模型
权限返回 403，因此没有用于正式验收。结果如下：

| Definition | 结果 | rounds / best | `best_geo_mean` | `min_speedup` |
| :--- | :--- | :---: | ---: | ---: |
| `kernelgenbench_square` | `PASSED`，36/36 | 1 / 1 | 1.008885x | 0.875000x |
| `kernelgenbench_neg` | `PASSED`，36/36 | 2 / 2 | 0.922652x | 0.460976x |

`neg` 第一轮只在 shape `[16,128,64,60]` 的三种 dtype 上数值失败。Debug Job
复核到三者在 MetaX 上均为非连续 layout，stride 为
`(491520, 1, 7680, 128)`；候选使用 `empty_like(x)` 保留该 stride，随后 flatten
输出时得到独立副本。第二轮改为直接分配连续 flat output，再按原 shape reshape
返回后 36/36 通过。这证明 v6.2 correctness workload 能暴露 layout 语义问题，
且 timing workload 不代替 correctness 判定。两项最终 geo mean 均高于项目的
0.8 合格线。

独立 Debug Job shell 当时继承了指向旧 Server checkout 的 `PYTHONPATH`；需要在
诊断命令中显式把当前部署仓库放到 `sys.path` 首位。正式 Eval Server 进程和上述
权威评测均使用 `ab849a5`，不受该 shell import 顺序影响。

结果保存在
`runs/v62_kernelgenbench_simple_opt_20260827/`。

## 2026-08-05 昇腾 KernelGen Server v5.1.1 单卡 SimpleOpt E2E

本轮用 `flaggems_acosh_` 复核连续 Coder context、15 轮硬上限和 v5.1.1
远端评测链路。KernelGen 基于 `98b51fd` 加当前工作区改动；KernelGen Server
使用 tag `v5.1.1` 的 `459d924`。本地 Agent 通过 SSH stdio HTTP proxy 访问
远端容器内仅绑定 loopback 的 Server，backend 为 `npu`，计时严格使用
`torch_npu.profiler`，Server `max-workers=1`。

物理卡 0 在启动前已有其他用户的 `uvicorn` 进程，因此没有占用；实验使用允许
范围内第一张空闲卡 1。Server 启动、实验结束和回收前均为单个健康 slot，
最终 snapshot 为 `healthy=1`、`active=waiting=checking=broken=0`、
`incidents=0`。未安装或升级任何 Python 包、Torch、Triton、torch_npu 或厂商
运行时。

SimpleOpt 共完成 15 个 measured round，全部位于同一个 Coder session
`d6c7f00d-a62b-44b4-b448-f8591b5c6349`。第 15 轮
`finalize_round` 持久化 `max_round_reached`、返回 `continue=false` 并恢复
round 13 best；Coder 随即输出唯一最终报告，没有进入 round 16。之后单独启动
Distiller session。最终结果：

| 指标 | 结果 |
| :--- | ---: |
| workflow status | `PASSED` |
| measured rounds / best round | 15 / 13 |
| `best_geo_mean` | 1.139868x |
| f16 `64x64` | 1.177638x |
| f32 `4096x4096` | 0.813425x |
| bf16 `64x512x512` | 1.546087x |

同算子 2026-08-03 历史结果为 6 轮、best round 4、
`best_geo_mean=0.706454x`；本轮相对提高 61.35%。best kernel 使用固定 512
blocks 的 grid-stride loop、按 dtype 选择 2048/4096 block size，并采用
`num_warps=4, num_stages=2`。最终 `tmp/main.py` 与 `.best_kernel.py` 的
SHA256 均为
`a460171b7413d5c9a2702cbc335250d7715c273c27ae6b8e981d22f205d0040e`。

本轮同时暴露一个不影响最终结果的 lifecycle 细节：Coder 在 round 2 eval 后、
`finalize_round` 前修改了 candidate；finalization 的 KEEP/REVERT 转换按已评测
snapshot 恢复文件，覆盖该未评测修改，导致 round 3 与 round 2 的 solution
SHA256 和 evaluation fingerprint 相同。后续应明确禁止在 measured round
finalize 前编辑下一候选，或调整提示顺序，避免浪费一次评测。

结果保存在
`runs/simple_opt_v511_ascend_card1_20260805_111635/`。实验完成后本地 proxy 和
远端临时 Server 均已停止，物理卡 1 无遗留进程；物理卡 0 的既有进程未被触碰。

## 2026-08-03 多设备轨迹采集快照

本轮七台设备的 mitmproxy 监听端口为 `18211` 至 `18217`，轨迹分别写入 `/data/.cc_traj/kernelgen_<device>_batch_20260803/`。当时的验收不仅检查 mitmdump 进程和监听端口，还对每个端口执行一次真实 `claude -p`；只有 Claude 正常返回，且对应目录新增包含完整 `request`、`response` 的 `session_*.jsonl`，才判定该设备的轨迹链路有效。这些端口和路径仅是历史快照；后续实验已停用 mitmproxy 轨迹采集，不再复用这套配置。

## 2026-07-29 Agent CLI 环境快照

本轮 E2E 采用本地 Agent + 本地 Server。两个仓库均使用 editable install，没有
通过项目 `PYTHONPATH` 覆盖导入路径。Agent CLI 通过 npm 在目标 CPU 架构上安装，
没有跨机器复制二进制。

| 芯片 | CPU 架构 | 安装前 Node | 实际 Node / npm | npm 选择的平台包 | 结果 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 天数 BI-V150 | x86_64 | 无 | 22.23.1 / 10.9.8 | `claude-code-linux-x64` | 通过 |
| 海光 BW1000 | x86_64 | 无 | 22.23.1 / 10.9.8 | `claude-code-linux-x64` | 通过 |
| 摩尔 MTT S5000 | x86_64 | 20.20.2 | 22.23.1 / 10.9.8 | `claude-code-linux-x64` | 通过 |
| 昆仑芯 P800 | x86_64 | 无 | 22.23.1 / 10.9.8 | `claude-code-linux-x64` | 通过 |
| 华为昇腾 910B4-1 | aarch64 | 无 | 22.23.1 / 10.9.8 | `claude-code-linux-arm64` | 通过 |
| 沐曦 MetaX C550 | x86_64 | 无 | 22.23.1 / 10.9.8 | `claude-code-linux-x64` | 通过 |
| 平头哥 PPU-ZW810E | x86_64 | 24.18.0 | 24.18.0 / 11.16.0 | `claude-code-linux-x64` | 通过 |

七台均生成 `package-lock.json`，且 `claude --version` 为 `2.1.220`。x64 CLI
的 SHA256 为
`674f61f20ff306f3100cf9200e4c36c4b70278b5bef2884549819b942a89c863`；
昇腾 arm64 CLI 的 SHA256 为
`159e4a51d796f3bf14677577100f7efb845611b1ceaf0c30cbd8d4650d942185`。
海光首次从官方 npm registry 下载时发生 `ETIMEDOUT`；增加 npm fetch 重试和
600 秒 timeout 后通过，安装器已固化该容错参数。

本次只在实验目录新增隔离 Node、npm metadata/lock 和 Claude CLI，没有安装
Python 包，也没有修改系统 Node、Torch、Triton 或厂商运行时。

## 2026-07-30 平头哥 KernelGen Server v5 SimpleOpt E2E

在 `dsw-694941-65ff46648d-zptxf` 上使用当前未提交源码进行同机验证：

- PPU-ZW810E；Torch 2.10.0，Triton 3.5.0，Claude CLI 2.1.220。
- KernelGen Server backend 为 `thead`，端口 18207，暴露 16 张逻辑设备，
  `max-workers=6`，计时严格使用 Triton `do_bench`。
- FlagGems v5 catalog 由
  `kernelgen_server.builtin_catalog_path("flaggems-v5")` 定位。
- 两个仓库只执行了 `pip install -e . --no-deps --no-build-isolation`；
  没有安装、升级或替换 Torch、Triton 及厂商运行时。

启动 Agent 前，`flaggems_rsqrt` reference-as-solution 的 preflight 和 evaluate
均为 8/8 workload 通过。随后 SimpleOpt 完成 9 个有效 eval round，所有 round
均通过 correctness；第 9 轮连续两轮未改善，返回明确
`performance_plateau` STOP。最终退出码为 0，best 来自第 7 轮：

| 指标 | 结果 |
| :--- | ---: |
| `best_geo_mean` | 1.027903x |
| `min_speedup` | 1.000759x |
| `rsqrt-time-0` | 1.020656x |
| `rsqrt-time-1` | 1.063279x |
| `rsqrt-time-2` | 1.000759x |

best kernel 使用 fp32 upcast 后执行 fused `tl.math.rsqrt`，按输入规模选择
BLOCK_SIZE，并仅对总字节数不小于 64MB 的输入启用 `.cg`。初始候选使用了不存在的
`triton.empty_like`，随后又发现该 Triton 的 `tl.math.rsqrt` 不接受 fp16/bf16；
两者都由 Agent 根据 preflight 错误自行修复，未进入错误计时。一次
`@triton.jit(num_warps=8)` 尝试也在 preflight 阶段被拒绝，没有污染 ledger。

任务完成后 `optimize_definition_output.json`、best kernel 和 9 轮 ledger 均存在；
无遗留 Claude/MCP 进程，server `/status` 正常，server 日志没有异常。

## 2026-07-29 七机 BatchSimpleOpt 四卡 E2E

原始横向实验采用同机部署，每台 server 暴露 4 张空闲卡，BatchSimpleOpt 使用
`max-workers=6`，计时策略统一为 Triton `do_bench`。随后在华为昇腾上保持
相同卡数、worker 数、模型和算子集合，以严格 `torch_npu.profiler` 独立重跑；
无有效 profiler timing 时直接失败，不允许回退到 walltime。legacy 选择
`gelu`、`matmul_basic`，FlagGems v4 选择 6 个代表性算子；昆仑芯避开已知
问题，使用等价规模的安全算子集合。

| 芯片 | 物理卡 / 端口 | legacy | FlagGems v4 | 结论与注意事项 |
| :--- | :--- | :---: | :---: | :--- |
| 天数 BI-V150 | 2,3,4,5 / 18183 | 2/2 | 6/6 | 通过。必须使用带厂商 Triton 的 `/usr/local/bin/python3`；`/usr/bin/python3` 虽能导入 Torch，但没有已注册的 Triton builder。 |
| 海光 BW1000 | 2,3,4,5 / 18184 | 2/2 | 6/6 | 通过。只设置 `HIP_VISIBLE_DEVICES`；同时设置 `ROCR_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES` 会造成可见卡重复映射，逻辑卡 2/3 报 invalid device ordinal。 |
| 摩尔 MTT S5000 | 1,2,3,4 / 18185 | 2/2 | 6/6 | 通过。使用 `MTHREADS_VISIBLE_DEVICES` 和 `MUSA_VISIBLE_DEVICES`。 |
| 昆仑芯 P800 | 1,2,3,4 / 18187 | 2/2 | 6/6 | 通过。原始 v4 为 5/6，`var` 在最后 verdict 仍为 `CONTINUE` 时 Agent 提前结束；续跑至明确 `STOP` 后通过。 |
| 华为昇腾 910B4-1 | 1,2,3,4 / 18188、18190 | do_bench 2/2；profiler 2/2 | do_bench 6/6；profiler 6/6 | 两种计时策略均通过。原始 do_bench v4 为 5/6，`rsqrt` 因同一非终态提前结束而失败，独立重试通过；profiler 重跑一次完成 6/6，server 未出现无效 timing 或 fallback 错误。 |
| 沐曦 MetaX C550 | 0,1,2,3 / 18186 | 2/2 | 6/6 | 通过。使用原环境 `/opt/conda/bin/python3`；`/usr/bin/python3` 缺少 Uvicorn。`var` 两次复现非终态提前结束，续跑至明确 `STOP` 后通过。 |
| 平头哥 PPU-ZW810E | 0,1,3,4 / 18189 | 2/2 | 6/6 | 通过。正确入口对应 `dsw-694941-65ff46648d-zptxf`；四卡 server 使用严格 Triton `do_bench`，6 workers 正常共享 slot。此前 403 是当时模型接口返回的鉴权错误；相同 CLI 和配置现已恢复，不是芯片侧问题。 |

多卡验证期间，6 个 Agent worker 可以共享 4 个 server slot；多个时刻 4 个 slot
同时处于 busy，任务完成后均能释放并继续接收排队请求。因此 server 模式下
`max-workers` 不需要小于或等于设备数，本轮 4 卡、6 worker 配置工作正常。
昇腾 profiler 重跑也观察到 4 个 slot 同时 busy；最终 batch 退出码为 0，
所有 Agent 退出且 slot 恢复 idle，说明非 daemon profiler worker 没有破坏
多请求排队、资源释放或终止流程。

三台机器独立复现了同一个 workflow 问题：Coder 在已经产出可用 best kernel、
但最新 finalized verdict 仍为 `CONTINUE` 时返回。原逻辑会立即判定任务失败。
KernelGen 现允许在同一 workspace 中自动启动后续 Coder session，注入持久化
历史和 best kernel，默认最多 3 个 session；仍然只有明确 `STOP` 才判定成功，
其他编译、正确性或性能失败不会被掩盖。

本轮四卡重跑没有安装任何新包，也没有安装项目本身或改动 Torch、Triton 和厂商
运行时。此前同机部署只在隔离的 `venv --system-site-packages` 中从离线
wheelhouse 补充 KernelGen 的轻量运行依赖（Pydantic、PyYAML、MCP、Requests
及其传递依赖，Python 3.10 额外补 exceptiongroup）。

平头哥当时通过 JumpServer 连接，并用 `hostname` 确认目标为
`dsw-694941-65ff46648d-zptxf`。当前连接方式以实验手册和机器清单为准。本次三个
退出码均为 0；legacy 的 `gelu`、
`matmul_basic` 最终加速比分别为 1.050x、1.185x。测试没有安装新包，临时
18189 server 已停止，原有 18087 server 保持运行。

### FlagGems v4 E2E 加速比

加速比取每个 definition 最终通过结果的 `best_geo_mean`，即各 workload 相对
reference 延迟比的几何平均；大于 1 表示生成 kernel 更快。昆仑芯为避开已知
不支持项，用 `silu_backward`、`narrow_copy` 替换了
`softplus_backward`、`max_pool3d_with_indices`。

| v4 definition | 天数 | 海光 | 摩尔线程 | 昆仑芯 | 昇腾 do_bench | 昇腾 profiler | 沐曦 | 平头哥 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rsqrt` | 1.069x | 1.191x | 1.099x | 0.400x | 0.925x | 0.960x | 0.985x | 1.006x |
| `var` | 1.040x | 1.243x | 0.784x | 0.141x | 0.111x | 1.203x | 1.420x | 1.030x |
| `softplus_backward` | 1.097x | 1.085x | 1.449x | — | 0.691x | 1.160x | 1.113x | 1.117x |
| `silu_backward` | — | — | — | 0.444x | — | — | — | — |
| `max_unpool2d` | 2.340x | 1.312x | 2.762x | 14.254x | 0.054x | 0.286x | 1.880x | 2.500x |
| `max_pool3d_with_indices` | 0.028x | 3.272x | 2.326x | — | 0.038x | 0.035x | 3.765x | 2.784x |
| `narrow_copy` | — | — | — | 0.073x | — | — | — | — |
| `uniform_` | 0.615x | 0.781x | 0.825x | 8.874x | 4.851x | 0.009x | 1.820x | 0.763x |
| **6 个 definition 几何平均** | **0.604x** | **1.324x** | **1.370x** | **0.783x** | **0.299x** | **0.223x** | **1.648x** | **1.354x** |
| **超过 1x** | **4/6** | **5/6** | **4/6** | **2/6** | **1/6** | **2/6** | **5/6** | **5/6** |

原始 do_bench 横向实验七机共 42 个 v4 definition，其中 26 个超过 1x；
全部结果等权几何平均为 0.924x。昇腾 profiler 独立重跑的 6 个 definition
几何平均为 0.223x，其中 2/6 超过 1x；legacy 的 `gelu`、`matmul_basic`
分别为 0.987x、0.987x，2/2 PASSED。由于昆仑芯的算子集合不同，其汇总值
不适合与其他机器直接横向排名。

两列昇腾结果来自两次独立 BatchSimpleOpt workflow，而不是对同一份 best
kernel 的重复计时。`do_bench` 使用 steady-state median；profiler 使用逐轮
清 L2 后的硬件采集平均值。它们适合比较不同计时反馈下的 E2E 产出，不应被
解释为同一 kernel 的固定换算系数。

这里的 `PASSED` 表示 workflow 得到了 correctness 通过的独立 kernel 并正常
结束，并不承诺 `best_geo_mean > 1`。低于 1x 的结果仍应保留，用于区分 E2E
可用性与最终性能是否优于 reference。

## 2026-07-30 七机 BatchSimpleOpt v5 15 算子 E2E

本轮在 FlagGems v5 catalog 上对七台芯片各跑一次
`batch_simple_opt_v5_overlap15`，每台使用同一组 15 个 definition。run 名为
`batch_simple_opt_v5_overlap15_20260730_<chip>`，同机部署、`localhost`
访问 eval server，计时策略与 2026-07-29 一致：昇腾使用严格
`torch_npu.profiler`，其余芯片使用 Triton `do_bench`；无有效 timing 时直接
失败，不回退 walltime。overlap15 指 Agent 生成/分析与 server 设备执行重叠，
`max-workers` 大于设备数。

截至 2026-07-31 01:06，七台的所有 definition 均进入终态（无 `RUNNING` /
`STARTING`），Agent 全部退出、设备回到 idle。昇腾最后收尾：`tile`、
`scaled_dot_product_attention_backward` 收敛为 PASSED，`unbind_copy` 在 10
轮后仍 `PARTIAL_PASS` 判 FAILED。

### 终态通过与加速比汇总

| 芯片 | 计时 | 终态 PASSED | 其中 >1x | PASSED 几何平均 |
| :--- | :--- | :---: | :---: | ---: |
| 天数 BI-V150 | do_bench | 13/15 | 2 | 0.460x |
| 海光 BW1000 | do_bench | 13/15 | 5 | 0.601x |
| 摩尔 MTT S5000 | do_bench | 12/15 | 7 | 0.956x |
| 沐曦 MetaX C550 | do_bench | 12/15 | 6 | 0.728x |
| 昆仑芯 P800 | do_bench | 9/15 | 2 | 0.118x |
| 华为昇腾 910B | profiler | 8/15 | 4 | 0.239x |
| 平头哥 PPU-ZW810E | do_bench | 11/15 | 7 | 1.370x |

「终态 PASSED」是全部 workload 通过、拿到 headline 性能的 definition 数；
`PARTIAL_PASS/FAILED` 不计入。「PASSED 几何平均」只对该芯片已 PASSED 的
definition 的 `best_geo_mean` 取等权几何平均，因此不同芯片的分母不同，且昇腾
使用 profiler 计时，这些汇总值不适合作为跨芯片绝对性能排名，只反映各芯片相对
自身 reference 的 E2E 产出。

### 各算子 best_geo_mean

`—` 表示该算子在该芯片最终为 `PARTIAL_PASS/FAILED`，没有全 workload 的有效
geo-mean。

| v5 definition | 天数 | 海光 | 摩尔线程 | 沐曦 | 昆仑芯 | 昇腾 | 平头哥 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `addmm_` | 0.118x | 0.493x | 0.252x | 0.430x | — | — | 0.919x |
| `avg_pool3d_backward` | 0.680x | 1.127x | 2.044x | 1.378x | 0.161x | 0.005x | 1.334x |
| `conv_transpose1d` | 0.156x | — | — | — | — | — | 0.784x |
| `cudnn_convolution` | 0.125x | 0.042x | — | 0.831x | — | — | — |
| `einsum` | 2.682x | 3.715x | 1.624x | 3.615x | — | 4.535x | — |
| `gcd` | 0.755x | 1.083x | 1.196x | 0.634x | 0.099x | 6.319x | 1.153x |
| `index_copy_` | 0.567x | 0.912x | 0.892x | 1.301x | 0.023x | 0.481x | 1.123x |
| `linear` | 0.128x | 0.208x | 0.147x | 0.351x | 0.305x | 0.001x | 0.622x |
| `median` | 2.710x | 4.606x | 4.205x | 1.113x | 0.063x | — | 4.613x |
| `nonzero_numpy` | — | 0.172x | 2.448x | 0.061x | 1.005x | 1.000x | — |
| `permute_copy` | 0.864x | 0.948x | 1.300x | 0.124x | 0.227x | — | 1.267x |
| `renorm_` | — | — | — | — | — | — | 0.919x |
| `scaled_dot_product_attention_backward` | 0.416x | 0.390x | 0.476x | — | — | 0.183x | — |
| `tile` | 0.374x | 2.468x | 0.118x | 1.090x | 0.001x | 1.060x | 1.905x |
| `unbind_copy` | 0.493x | 0.104x | 5.917x | 4.649x | 2.559x | — | 4.042x |

亮点：昇腾 `gcd` 6.319x、`einsum` 4.535x；摩尔线程 `unbind_copy` 5.917x；
海光、摩尔线程、沐曦、平头哥 `median` 均在 4x 以上。

### 各芯片未通过算子

统一 15 算子集下，七台合计 15 个「至少一台 PASSED」的算子中，
`renorm_` 是唯一在全部七台都失败的（各芯片均 10 轮 `PARTIAL_PASS`），需要在
第二阶段作为难算子归档进入后续攻关。其余高频失败为
`conv_transpose1d`（仅天数、平头哥通过）和 `cudnn_convolution`（仅天数、
海光、沐曦通过）。

| 芯片 | 未通过算子 |
| :--- | :--- |
| 天数 | `nonzero_numpy`、`renorm_` |
| 海光 | `conv_transpose1d`、`renorm_` |
| 摩尔线程 | `conv_transpose1d`、`cudnn_convolution`、`renorm_` |
| 沐曦 | `conv_transpose1d`、`renorm_`、`scaled_dot_product_attention_backward` |
| 昆仑芯 | `addmm_`、`conv_transpose1d`、`cudnn_convolution`、`einsum`、`renorm_`、`scaled_dot_product_attention_backward` |
| 昇腾 | `addmm_`、`conv_transpose1d`、`cudnn_convolution`、`median`、`permute_copy`、`renorm_`、`unbind_copy` |
| 平头哥 | `cudnn_convolution`、`einsum`、`nonzero_numpy`、`scaled_dot_product_attention_backward` |

注：昆仑芯 `einsum` 为 0 round FAILED（未产出任何 measured round），其余
FAILED 均为 10 轮耗尽仍 `PARTIAL_PASS`。这些结论是 E2E 链路可用性与优化产出的
快照。未通过项需区分基建问题与优化问题：属于抽取语义或芯片 capability 的按
`flaggems_definition_workload_guide.md` 的判定边界复核，属于优化难点的收集进入
第三阶段。

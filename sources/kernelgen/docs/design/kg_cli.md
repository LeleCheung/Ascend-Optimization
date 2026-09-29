# KG CLI 简化设计

状态：第一版本地 CLI 与 H800 YAML Batch E2E 已验证。

## 目标

`kg` 为专业算子开发者提供本地入口，复用现有 Workflow、Runtime 和 workspace，不改变 KernelGen 的执行模型。第一版只解决三个问题：方便启动优化、通过 campaign 路径查询和取消、自动准备并管理本地或远端 KGS。

CLI 不实现 Web、全局任务注册中心或中央 daemon。KGS 声明 capability 时支持在途操作取消；这不等于直接终止 Agent。优化 run 使用独立后台进程，不依赖常驻管理服务；新增的 lifecycle dummy 入口暂时只在当前进程前台执行。

## 命令面

优化入口的 `--target-hardware` 是可选约束，不预设芯片。未指定时，CLI、Batch 和 Python launcher 将 `null` 保留到 OperatorOptimize，后者使用 KGS `/status` 解析并冻结实际目标；显式约束不匹配或 Server 缺失设备信息时仍失败。请求及 history 顶层的 `target_hardware` 表示用户约束，可以为 `null`；实际评测设备以冻结的目标快照和各 ledger 为准，续跑仍校验原快照。

```text
kg run [--mode simple_opt|kernelgen] ... [--foreground]
kg extract --operator <operator> (--flaggems-repo <checkout> | --pr-url <url>) [--workspace <new-workspace>]
kg definition --flaggems-repo <checkout> --pytest-path <path> [--operator <operator>] [--workspace <workspace>]
kg run --mode lifecycle --dummy --definition <operator> [--stages ...] [--optimize-mode simple_opt|kernelgen]
kg run --batch-file <batch.yaml> [--workspace <batch-workspace>]
kg status <workspace> [--detail]
kg history <single-run-workspace> [--json]
kg logs <workspace> [--follow] [--after <sequence>] [--raw]
kg cancel <workspace> [--reason <text>]
kg resume <workspace> [--foreground]
kg list [--json]
kg config set run.max-workers <count> --eval-server <kgs-url>

kg server start <name> [--target local|remote ...]
kg server list [--json]
kg server configure <name> ...
kg server status <name> [--json]
kg server logs <name> [--follow]
kg server stop <name>
kg server install-flaggems <name> [--revision <branch-or-full-commit>]
kg server doctor <name>
```

规划中的自定义算子 CLI 绑定尚未实现，不属于当前命令面；KGS 已提供 Bundle 上传、校验和查询，配套执行绑定已实现，但是否可用必须读取目标 `/status.capabilities.operator_bundle_upload.evaluation_binding`：

```text
kg run --mode simple_opt|kernelgen --operator-dir <v6.2-operator-dir> --server <server-name>
```

`kg run --mode simple_opt|kernelgen` 默认启动独立后台进程并立即返回，campaign workspace 是本地任务句柄。`--foreground` 用于调试并直接输出日志；前台模式收到 `Ctrl-C` 时写入协作式取消请求，并等待当前安全边界退出。后台任务通过 `kg cancel <workspace>` 请求取消；若 KGS 声明 `operation_cancel` capability，CLI 同时转发 workspace 中登记的 Reference/Preflight/Eval/Profile operation ID。

`kg run --batch-file <batch.yaml>` 一次提交 YAML 中的全部算子。Batch 本身不创建第二套执行器；每个算子仍转换为独立 `RunRequest` 和后台 worker，并共同使用按 KGS endpoint 配置的 worker lease。Batch workspace 保存子任务索引，实际单算子 workspace 位于 `<batch-workspace>/definitions/<definition>/`。

CLI 使用标准库 `argparse`，通过 `pyproject.toml` 注册 `kg = "kernelgen.cli:main"`，不增加 CLI 框架依赖。文本输出面向人，任务 status 用 `--detail` 输出完整 JSON，其余命令保留 `--json`。

CLI 的配置、运行索引、并发租约、Server 实例状态和相关 lock 默认位于执行命令时的当前工作区 `.kernelgen/`。同一组任务和 Server 管理命令应从同一个工作区执行；需要跨目录共享或固定状态目录时，显式设置 `KERNELGEN_CLI_HOME=/path/to/state`。已有其他目录中的状态不会自动迁移。

## 逐功能设计方案

本节定义每个用户功能的输入、核心流程、状态归属和失败边界。除“自定义算子 Bundle”明确区分已完成的 KGS 存储接口与待实现的 CLI 执行绑定外，其余内容描述第一版命令契约；后续章节说明这些功能共用的后台进程、并发、Batch、运行控制和部署机制。

### `kg run`：提交单算子任务

用户通过 `--definition` 选择 KGS Catalog 中的算子，通过可选的 `--mode` 选择 `simple_opt` 或 `kernelgen`。默认使用 Gems adapter、kernelgen、1 个 Coder、1 个 epoch、每个 Coder 每个 epoch 最多 10 rounds。核心默认常量由 `data/constants.py` 供输入模型和 `framework/run_options.py` 共用；后者统一参数名称、别名、类型和 CLI/YAML 校验，最终合并结果按模式校验。显式参数和保存的旧请求不被新默认值覆盖。CLI 不支持 `-- ...` 任意透传，不开放 `--clean` 或凭据参数；`--foreground` 等进程管理参数只属于 KG。模式专属参数不得放进同时用于另一模式的 Batch defaults，应放到对应 operator 条目。

两种模式都支持 `--reference-code-path` 和可选 `--reference-code-prompt-path`，仅注入任意语言的只读设计参考，允许历史实现存在编译、正确性或性能问题。`--seed-code-path`（别名 `--seed-triton-path`）是未预验证的 Triton 初始候选；不继承历史成绩，也不改变 Catalog oracle、workload 或 timing baseline。reference 与 seed 可同时提供。旧 reference-triton 参数别名仍在输入边界归一化。

CLI 将解析后的参数一次转换为不含凭据的 `RunRequest` v2，直接调用 OperatorOptimize，不再把新任务的参数序列化为 launcher argv 再解析。`--mode simple_opt|kernelgen` 只选择内部优化器，不新增 `catalog_optimize` mode。`--catalog-name` 与 `--catalog-path` 互斥，仅两者均省略时补默认 Catalog；前者从 KGS 导出契约，后者冻结本地 Native 内容或 [新导出的 Gems Definition](workflows/gems_adapter_definition.md)，并内部上传 Bundle。Gems adapter、安装式 Native 与本地 Native 均支持两种优化器。Gems 的多 Coder/epoch 复用同一冻结的 pytest 契约，不转换成 Native，不伪造 reference/workload。未知或不适用的参数不能静默丢弃。

新任务按 `prepare_catalog → review_tests → optimize → code_review` 执行；所有来源默认审核测试契约，只有显式 `--skip-review`（Batch/Python：`skip_review: true`）跳过，旧抽取审核和已安装来源都不自动免审。Native 检查目标 reference readiness，Gems 在生成前执行原 benchmark 的 core `--reference-only`，随后仍由候选 Preflight 和原 pytest 完成正确性与计时验证。详见 [统一测试契约审核](workflows/review_tests.md)。只有 Optimize 申请一次 Coder lease，CLI supervisor 不再提前占用；既有优化器仍负责轮次、epoch、Knowledge、Eval 和最终确认。后台提交成功只表示 `SUBMITTED`，完成与否以后续 `kg status`、ledger 和审核结果为准。

优化 ledger 位于 `stages/optimize/work/`（KernelGen 再按 epoch/Coder 分目录）；status/history 读取该绑定下的真实 ledger，普通步骤没有空的 round/epoch 字段。旧 simple_opt/kernelgen v1 请求保留原 launcher 和目录用于原地续跑，不把旧 campaign 重解释成新流程。CLI 通过 `cli.optimization.run_operator_optimization()` 初始化 Runtime 并调用 OperatorOptimize；Python 集成直接使用 `workflows.optimization.OperatorOptimizeWorkflow` 及其输入模型，不再依赖已删除的 Catalog example launcher。

已有 KG run 的 workspace、同一 workspace 的活动进程、非法模式参数、缺失输入文件或任务申请的 worker 数超过 endpoint 上限时，提交在启动 Coder 前失败。默认后台运行；`--foreground` 只用于交互调试，不改变 workspace、状态和取消语义。

### 已删除的组合入口

不再提供 `ExtractAndOptimizeWorkflow`、`kg run --mode extract_and_optimize` 或其 `--input` 参数路径。先独立调用 CatalogExtract，再将产物路径交给 `kg run --mode simple_opt|kernelgen --catalog-path <path>`。历史组合 workspace 保持不动，查看或续跑需使用原 KG 版本，不自动迁移成新的优化任务。

### `kg definition`：从已有 pytest 导出 Gems Definition

通过 `cli.api.export_gems_definition()` 复用 `GemsAdapterDefinitionWorkflow`，只读取干净、已提交的本地 Gems checkout，导出 ABI 和原始测试来源信息。不调用模型、不执行 pytest、不连接 KGS，不复制 Native oracle/workload。必填 `--flaggems-repo` 与 `--pytest-path`；pytest 路径可相对源码仓库，聚合测试文件用可选的 `--operator` 指定算子。

输出默认位于当前 CLI home 的 `definitions/<id>`，可用 `--workspace` 覆盖，但必须在 Gems checkout 外。同一来源可以重复导出到同一 workspace，改变来源必须用新目录。成功打印 JSON，包含 workspace、catalog_path、definition_path、operator 和来源 hash/commit；退出码为成功 0、输入或执行错误 2、中断 130。该入口只前台运行，不创建 RunRequest、run index、Coder lease 或另一套持久状态，不支持 Batch、后台提交或 `kg status/resume`。导出不等于验证通过，优化仍由 `kg run --definition <operator> --catalog-path <catalog_path>` 提交。用法与来源一致性约束见 [Gems Definition](workflows/gems_adapter_definition.md)。

### `kg extract`：前台独立抽取

此入口已纳入维护者授权重发的 v6.4.0；最初指向 `0082e613` 的旧 tag 不包含它，需要刷新 tag。通过 `cli.api.extract_catalog()` 直接调用 CatalogExtractWorkflow，支持本地 FlagGems/FlagGems-vllm checkout 或受支持的 PR URL，源码输入二选一。`--operator` 必填；不暴露 case list 参数，Workflow 自动采集并执行既有审核修订循环。`--max-review-rounds` 默认 3、范围 1–10；`--runtime claude|codex` 默认读取 KERNELGEN_RUNTIME 或 claude；`--model` 使用各 provider 的现有配置，`--timeout` 默认 900 秒，作用于每次模型调用而非整个 Workflow。

默认 workspace 为当前 CLI home 下的 `extracts/<id>`，也可指定尚不存在的 `--workspace`，不能位于源码 checkout 内。原子创建 workspace 防止两个 CLI 同时接受同一目录；失败和审核阻断现场保留，不覆盖、不自动续跑。采集配置继续来自 `kg config set extract.*`，不要求目标 KGS，也不占优化 Coder lease。Runtime 对源码和审核证据只读，不挂载优化 MCP。

成功打印 workspace、catalog_path、operator_dir、case_list_path、review_path 和 `target_validation=NOT_RUN`；这不等于目标芯片评测通过。退出码为成功 0、审核预算耗尽 1、输入或执行错误 2、协作取消 130。Ctrl+C 复用完整模型输出后的取消安全点，不直接 kill Agent。

本入口仅前台运行，不创建优化 RunRequest 或 run index，不接入 `kg status/history/logs/cancel/resume/list`、Batch 或后台提交。过程和审核证据仍由原 Workflow 在 workspace 记录；未提供另一套持久状态。已发布目录交给 `kg run --catalog-path`，不恢复已删除的组合 Workflow。详情见 [抽取与采集配置](workflows/flaggems_pr_extract.md)。

### `kg run --mode lifecycle`：前台 dummy 生命周期

该模式复用 [OperatorDevelopmentWorkflow](workflows/operator_lifecycle.md) 和串行 `run_campaign()`，不经过优化任务的 `RunRequest`、后台 runner 或 Coder lease。`--dummy` 必须显式提供；`--definition add` 表示单算子，`--operators add mul` 表示算子列表，两者互斥。`--workspace` 是 campaign 根目录，省略时仍通过 CLI 默认规则分配到当前工作区 `.kernelgen/runs/`（或 `KERNELGEN_CLI_HOME` 对应目录）。算子名目前仅是模拟标识，不查询 Catalog 或 KGS。

`--stages` 选择阶段并按固定顺序执行；`--optimize-mode` 只配置选中的优化阶段，默认 `simple_opt`，可选 `kernelgen`。参数定义和前台执行/退出码处理由 `cli/lifecycle.py` 提供，与 Python 示例共用，不复制编排逻辑。普通优化模式拒绝 lifecycle 专属参数；lifecycle 拒绝 `--batch-file`、`--eval-server`、轮次等尚未接入的真实优化参数，不静默忽略。

当前始终前台串行执行，显式加 `--foreground` 也不改变行为；`Ctrl-C` 沿用协作式取消，在当前 dummy 完整结果写入后退出。输出是带 `simulated=true` 的 campaign JSON，退出码：成功 0、模拟失败或等待 1、参数错误 2、取消 130。续跑使用原命令加 `--resume`，必须保留相同 workspace、算子列表、阶段选择和优化模式。

这不是完整的 CLI 后台生命周期接入：不创建 run index，暂不接入 `kg status/history/logs/cancel/resume/list`。查询和跨进程取消仍使用配套 Python 示例的 `status`/`cancel`，详见生命周期文档。真实 Agent、并发调度和后台提交后续再接入；原有 `simple_opt`、`kernelgen` 和 YAML Batch 执行语义不变；状态查询已统一迁移为 v2.0。

### `kg run --batch-file`：提交 YAML Batch

Batch 先完整解析 YAML，合并 defaults、命令行共享参数和 operator 覆盖，并在启动任何任务前校验重复 definition、workspace 名冲突和所有本地文件。每个 operator 独立生成 `RunRequest` 和 `<batch-workspace>/definitions/<definition>/`，Batch 根目录只保存不可变的子任务索引，不充当第二套执行器。

子任务按 YAML 顺序启动独立后台进程，并与其他单任务共同竞争各自 KGS endpoint 的 worker lease。中途提交失败时，CLI 向已经启动的子任务发出取消请求并保留现场；不会删除子 workspace，也不会用 Batch 顶层结果覆盖子任务 ledger。第一版不支持 Batch 前台运行和 Batch 根续跑，续跑必须针对未完成的子任务 workspace。

### 自定义算子 Bundle：KGS 执行绑定已实现，目录快捷参数待实现

官方权威算子继续由版本化 Catalog 仓库提供；用户自带的 v6.2 native 算子不写入 Catalog checkout。KGS 已实现 `HEAD/GET/PUT /operator-bundles/{sha256}` 和 Python `upload_operator_bundle()`：客户端把包含 `definition.json`、`oracle.py`、`correctness.jsonl`、`timing.jsonl` 及可选 assets 的目录打成确定性 tar，以 tar 的 SHA-256 形成内容寻址 `bundle_id`，相同内容只上传一次。KGS 在原子安装前校验摘要、大小、归档路径、文件类型和 v6.2 契约，并拒绝绝对路径、`..`、符号链接、硬链接、特殊文件和不完整算子；Bundle 保存在独立临时存储，不修改官方 Catalog 或 KGS checkout。

KGS 已把 `bundle_id` 绑定到 inspect/Preflight/Eval/Profile，OperatorOptimizeWorkflow 已消费此能力。旧实例可能仍返回 `evaluation_binding=false`，必须拒绝执行，不能把“上传成功”解释成“可以优化”。尚未实现的是 `--operator-dir` 快捷入口及由任务 lease/保留期驱动的清理。后续落地时，`--operator-dir` 先检查 capability 为可绑定，再通过 `--server <name>` 解析出的 loopback endpoint 上传并把 `bundle_id` 与摘要固化到 `RunRequest`；远端实例仍经 SSH stdio HTTP proxy 传输，不使用 `scp`。Bundle 至少保留到任务及允许的续跑窗口结束，`--operator-dir` 与 `--definition/--catalog-name` 互斥，`--server` 与 `--eval-server` 互斥。

### `kg status`：查询任务状态

`--detail` 输出完整 JSON，统一使用状态 Schema v2.0；替代原 `kg status --json`，不保留旧参数别名。Workflow 显式声明适用的进度类型，普通 Extract/Review 不再携带 epoch/round/performance 字段，详见 [进度 Schema](runtime/stage_progress.md)。`kg history/list/server status --json` 不受这次参数重命名影响。

单任务状态由 `run-progress.json`、后台进程启动身份、ledger 和最终输出投影得到，不解析日志，也不为查询建立 KGS 连接。状态同时返回稳定的根生命周期和细粒度 `stage`、epoch、round、best performance 与 scopes；进程已消失但 Workflow 未写终态时只派生显示 `INTERRUPTED`，不改写原始 ledger。

传入 Batch workspace 时，CLI 读取子任务索引并逐个投影状态，再汇总 counts 和 Batch 状态；单个子任务失败不能覆盖其他子任务事实。默认文本输出用于人工查看，`--detail` 输出带 `schema_version` 的完整 JSON，供 Web 或脚本复用。

### `kg history`：查询逐轮优化历史

该命令只读单任务 workspace 中的 ledger，并按 scope 输出互不拼接的性能序列。顶层给出全局 `best_scope`、`best_round` 和 `best_geo_mean`，每轮保留 correctness/timing 结果对应的状态、几何平均加速比、最小加速比和是否成为最终最佳等紧凑字段。

尚未产生有效 measured round 时成功返回空 series；不存在或损坏的 ledger 报错，不从模型日志猜测性能。Batch 根目录不合并不同算子的历史，调用方先用 `kg status <batch-workspace>` 获得子 workspace 后分别查询。

### `kg logs`：查看结构化事件或原始日志

默认读取 `run-events.jsonl`，按单调递增的 `sequence` 输出事件；`--after` 支持增量消费，`--follow` 轮询新增事件并在进程终止后退出。`--raw` 读取 `.kernelgen/runner.log`，`--lines` 只限制首次显示的尾部行数，主要用于诊断未进入结构化运行控制面的启动错误。

结构化事件是产品展示的首选来源，原始日志可能包含模型和工具上下文，只继承 workspace 的本地访问边界。Batch 根目录没有混合日志，避免并行子任务交错；用户必须从 Batch status 选择具体子 workspace。

### `kg cancel`：协作式取消

CLI 在目标 workspace 写入带 generation 的幂等取消请求，不直接 kill Agent 进程。排队任务在取得 worker lease 或启动 Workflow 前结束；运行中的 Runtime 在一次完整模型输出结束后检查取消，从而保留 session 的最后一条完整输出，并停止启动下一轮或下一阶段。

Preflight/Eval/Profile 提交期间登记 KGS operation ID；若 `/status.capabilities` 声明 `operation_cancel`，CLI 同时向这些 operation 发送取消请求，KGS 负责让排队 operation 终止，或让运行 operation 进入隔离回收和 slot 强探针流程。转发失败只记录结构化警告，不撤销本地取消请求。Batch cancel 对仍活动的子任务逐一执行同一流程；任务已经终止时 CLI 不伪造新的 `CANCELLED` 状态。

### `kg resume`：原 workspace 续跑

续跑只接受无活动进程的单任务 workspace，读取原 `RunRequest`，恢复可继续的 ledger 状态并以新的 cancellation generation 清除旧取消标记，然后重新进入同一 endpoint 的 worker lease 队列。它不修改 definition、模式、模型参数或 workspace，也不执行 `--clean`；campaign 已记录的 KGS、framework 和输入快照必须保持不变，不能在续跑中自动升级。

第一版不支持 Batch 根续跑，也不自动重启机器重启前的任务。调用方根据 Batch status 选择 `INTERRUPTED`、`CANCELLED` 或其他需要人工复核的子任务逐一续跑；已有活动进程时拒绝启动第二个 Coder。

### `kg list`：列出本地任务

该命令扫描 CLI 的轻量 workspace 索引并即时计算状态，自定义 workspace 也在提交时登记。属于同一 Batch 的子任务折叠为一个根条目；损坏或已移动且无法读取的索引项被跳过，不阻塞其他任务展示。

索引只解决发现问题，不保存权威执行状态、不分配资源，也不扫描其他 `KERNELGEN_CLI_HOME`。`kg list` 不访问 KGS 或 SSH，因此适合在 Server 离线时查看本地历史。

### `kg config set run.max-workers`：配置 Agent 并发

配置以规范化 KGS endpoint 为资源池键，保存在当前 CLI home。SimpleOpt 占 1 个 lease，KernelGen 占 `n_parallel` 个 lease；跨进程的文件锁和 PID 启动身份保证原子分配与僵尸 lease 回收。修改上限只影响后续分配，不终止现有任务；单任务申请量大于上限时直接拒绝，而不是永久排队。

该值限制本机 Coder 并发，不是 KGS `--max-workers`，也不代表设备 slot 数。KGS 请求线程和每卡最多一个 Eval/Profile 仍分别由 Server worker 配置和 scheduler 管理。

### `kg config set extract.*`：配置 case-list 采集环境

`extract.host`、`extract.container`、`extract.python` 分别指定 SSH Host alias、已运行的 Docker 容器和容器内 Python 绝对路径。三项保存在同一个 `.kernelgen/config.json`，与 Agent 并发配置共用配置锁，但不按 KGS endpoint 分组。缺项时自动 case-list 采集明确失败，不回退本机 Python；KG 和模型 Agent 仍在发起机器运行，不改变目标 KGS 的评测环境。配置命令与产物说明见 [SSH + Docker 采集执行器](workflows/flaggems_pr_extract.md#统一-ssh--docker-采集执行器)。

### `kg server start`：创建、部署或恢复实例

可加 `--install-gems`（别名 `--install_gems`）在启动前准备 FlagGems，local/remote 均支持，复用 `install-flaggems` 的安装路径并在同一个实例生命周期锁内完成。已有 FlagGems 配置直接复用，不自动追踪分支或升级；尚未配置但服务已运行时要求先停止，不隐式重启。安装失败不会继续启动。远端推荐版本从目标机的锁定 KGS checkout 读取，不要求本地存在 KGS 仓库。显式切换版本仍使用 `install-flaggems --revision`。

首次调用以实例名固化 local/remote 目标、backend、设备、端口、Python、KGS 锁定版本和连接参数，并完成原 `init` 与 `start` 两步。命令同步执行 checkout/校验、必要的轻量安装、后台进程启动和 `/status` readiness 检查；成功返回时 Server 已可用，但 KGS 进程在后台持续运行。普通重复调用是幂等恢复操作，不更新版本、不重新选择 Catalog，也不自动安装 FlagGems。

远端模式在目标容器内准备和启动 KGS，在开发机启动 loopback SSH stdio HTTP proxy；两端端口都必须显式给出。checkout 不干净、commit 不符、Protocol/backend/worker/slot 不符、端口或设备与活动实例冲突、代理失败或受保护运行时依赖发生变化时启动失败并保留可诊断现场，不 reset 仓库、不暴露公网端口。

remote 实例不要求开发机存在 KGS checkout：配置创建时只记录 KG lock 的 release、commit 和 Protocol 期望值，不在本地 clone 或读取 KGS compatibility.yaml。实际 checkout 和自描述校验由远端部署负责，运行时能力仍从远端 /status 获取；本地模式保留原 checkout 校验。这不取消 KG 自身的 Python/client 依赖，也不改变远端 exact commit、clean、loopback 或 SSH stdio 约束。

### `kg server configure`：修改已停止实例

该命令只更新持久化实例配置，不部署也不启动。backend、设备、timing、worker、端口以及远端连接、容器、remote Python、目录和部署环境文件可以在本地代理与远端 KGS 都停止后修改；相同值返回 `UNCHANGED`，影响部署的远端字段变更会使下一次 start 重新执行幂等部署检查。

实例的 local/remote 身份以及 KG/KGS release 不能原地切换，这类变化使用新实例名。配置中保存 SSH 基础命令但公共 JSON 不返回它，密码、token 和环境文件内容不得进入配置。

### `kg server list`：列出命名实例

该命令只读取本地实例目录和受校验的进程身份，不批量建立 SSH 连接。远端实例只能快速显示 `PROXY_RUNNING` 或 `REMOTE_UNKNOWN`，不能把本地代理状态解释成远端 KGS 的权威状态；需要精确判断时查询具体实例。

### `kg server status`：检查实例与 scheduler

本地实例读取受管进程和 loopback `/status`；远端实例额外通过一次性 SSH 核实远端 PID，再通过本地代理读取同一个 KGS `/status`。输出区分 `RUNNING`、`STOPPED` 和 `UNREACHABLE`，并展示 release、`api_version`、backend 与 scheduler 的 active、waiting、healthy、checking、broken 等事实。

状态校验只按 `/status.api_version` 判断 Protocol 兼容，并按 capabilities 判断可选功能，不从 KG/KGS release 推导协议能力。`--json` 隐去 SSH 命令和凭据；网络错误与 Server 已停止分开表达。

### `kg server logs`：查看 KGS 日志

本地实例读取受管 KGS 日志，远端实例通过一次性 SSH helper 读取目标状态目录中的日志；`--lines` 控制初始尾部，`--follow` 持续到相应进程停止。该命令不混入 Agent/Runtime 日志，任务侧问题仍使用 `kg logs <workspace>`。

### `kg server stop`：停止实例

该命令按已记录的 PID 和进程启动身份停止正确的受管对象，远端实例先停止远端 KGS，再停止本地代理；宽限期结束后允许对同一已验证进程强制终止。它不删除 checkout、实例配置、日志或 FlagGems，也不代替 `kg cancel`。运维流程应先通过 `kg server status` 确认 scheduler `active=0`，再停止或重启实例，避免中断仍在设备上执行的请求。

### `kg server install-flaggems`：准备统一 FlagGems 环境

该命令要求实例两端都已停止，在 KGS 实际执行环境中准备 clean FlagGems checkout。首次不传 `--revision` 时，采用实例锁定 KGS 的 `compatibility.yaml` 中的 repository、branch 和推荐 exact commit；也可用 `--revision <分支或完整小写 commit>` 显式选择同一 repository 的其他版本。分支在本地 KG 通过 Git 解析一次，因此本地需要该仓库的读取权限；实际 checkout 仍在 KGS 执行环境中准备，远端部署环境文件不会用于本地解析。已知完整 commit 时无需解析分支。

成功后把 root、实际 commit 和可选 branch 保存到实例配置，checkout 按 commit 分目录保留。再次不传 `--revision` 或执行 `start`、`doctor` 时只使用已保存的 commit，不自动追踪分支；再次显式传入分支才会选择其当前 HEAD。安装失败不覆盖原实例选择，也不覆盖旧 checkout。它不接受 Catalog 名、不安装 Torch/Triton/厂商运行时，也不把 Server 绑定到某个 Catalog；下一次 start 注入 `KGS_FLAGGEMS_ROOT` 并检查 adapter capability。

选择其他 Gems 版本不代表它已通过 KGS/Catalog 验证。配套新 KGS 的 FlagGems Adapter Catalog 不再声明或校验固定 framework revision，旧 Adapter manifest 的 framework 字段也不参与绑定；执行版本以 KG 实例选定的实际 checkout 为准。更新后重新 inspect 和验证 workload/baseline，不因 commit 不同就要求重建 Adapter Catalog；仍检查源码工作树、测试资产、ABI、报告格式及已有 fingerprint 一致性约束。FlagGems-backed Native Catalog 的冻结 oracle 仍须匹配其 `framework_revision`，不能仅改 manifest 冒充兼容。已有 campaign 续跑保留原 Gems 和 Catalog 快照；并行验证新版本应使用新实例和新 campaign。

部署边界：当前 KG 锁定 KGS v6.3.1，已包含 Adapter revision 放宽。旧 KGS v6.3.0 仍有 exact 校验，不会自动升级；已有实例保留创建时的 KGS commit，新实验使用按当前锁定清单创建的新实例。不得手工覆盖旧 checkout 或仅删除旧 KGS 的 manifest 字段来替代配套升级。

### `kg server doctor`：只读诊断

该命令校验锁定 KGS checkout、editable import、backend 必需模块、可选 FlagGems checkout，并在实例运行时检查 `/status` 的 release、Protocol、worker 和健康 slot。结果为 `OK` 或逐项 problem，方便部署前和故障后定位；doctor 不自动修复、不升级依赖、不启动或停止实例。

## 后台运行

后台模式不引入中央 daemon。CLI 为本次 run 启动一个脱离终端会话的子进程，将 stdin 连接到空设备，并把 stdout/stderr 写入 campaign 的 `.kernelgen/runner.log`。提交成功表示后台进程已经启动，不表示优化成功。

`kg run` 返回最少信息：

```text
workspace: /path/to/campaign
pid: 12345
status: SUBMITTED
```

进程元数据写入 `.kernelgen/run-process.json`，至少包含 PID、进程启动身份、启动时间、工作目录、worker 占用量和日志路径。可重放且不含凭据的调用参数单独写入 `.kernelgen/run-request.json`；认证 token 只通过环境变量传递，不写入 workspace。查询进程时必须校验启动身份，不能只相信 PID，避免 PID 被系统复用后误判或误杀其他进程。

同一个单算子 workspace 同时只允许一个活动 Coder。`kg run` 在启动前获取 workspace 运行锁；发现活动进程时拒绝重复启动。进程退出后保留 workspace、ledger、事件和日志，不自动清理。

后台进程意外退出且 Workflow 尚未写入终态时，`kg status` 显示派生状态 `INTERRUPTED`，但不覆盖 ledger。机器重启后不会自动拉起任务；用户检查现场后通过显式 resume 续跑。

## 按 KGS 实例的全局并发

每次提交独立 campaign 后，Batch 内部的 `max_workers` 不再能够限制所有后台进程的总并发。第一版增加本机共享、按 KGS endpoint 隔离的 run worker 配额，但不引入中央 daemon：

```bash
kg config set run.max-workers 4 --eval-server http://127.0.0.1:8000
kg config set run.max-workers 8 --eval-server http://127.0.0.1:18000
```

后台进程执行准备与审核，进入 Optimize 后才竞争共享 Coder lease。等待时 Optimize scope 显示 `QUEUED/WAITING_CODER_LEASE`，取得容量后才启动 Analyzer/Coder；准备和目标 readiness 不占 Coder 配额。`kg cancel` 可以协作式取消等待中的任务。

配额按 Coder 并发数计量：`simple_opt` 占用 1 个 worker，`kernelgen --n-parallel N` 占用 N 个 worker。每个任务根据规范化后的 `--eval-server` 进入对应 KGS 池，同一 endpoint 内共享上限，不同 endpoint 互不占用；`localhost` 与 `127.0.0.1` 视为同一主机，省略的 HTTP/HTTPS 端口会补成 80/443，未配置的 KGS 池默认上限为 1。若单个任务请求量大于所在 KGS 池上限，提交时直接拒绝。`--target-hardware` 用于与 `/status.target.device` 核对芯片身份，不作为资源池键。新任务由公共 Optimize 边界持有并在 finally 释放配额，不另外建立调度器。

共享 lease 必须原子分配并记录 KGS endpoint 池、进程启动身份和占用量；进程正常退出时释放，进程消失时可安全回收。修改某个 KGS 池的上限不终止已经运行的 campaign，只影响该池后续排队任务。

`kg list` 扫描默认 runs 根目录并显示 `QUEUED/RUNNING/CANCEL_REQUESTED/SUCCEEDED/FAILED/INTERRUPTED`。使用自定义 workspace 时，提交记录仍写入本地轻量索引以便发现；该索引只用于列举路径，不负责调度，也不是常驻任务服务。

这里按 endpoint 设置的 `run.max-workers` 与 KGS `--max-workers` 是两个独立参数：前者限制本机针对该 KGS 同时运行的 Coder，后者限制该 KGS 进程的 HTTP 请求执行线程。设备侧同一张卡最多执行一个 Eval/Profile，仍由 KGS scheduler 的 device slot 保证。

## YAML Batch

Batch 文件只接受 `.yaml` 或 `.yml`，Schema v1 使用 `defaults` 表达共享参数，`operators` 表达每个算子的覆盖参数：

```yaml
version: 1
workspace: runs/h800-batch

defaults:
  mode: simple_opt
  runtime: codex
  eval_server: http://127.0.0.1:20308
  target_hardware: NVIDIA H800
  catalog_name: kernelgenbench
  min_rounds: 1
  max_round: 10
  warmup_ms: 1000
  benchmark_ms: 100
  num_trials: 1
  profile: false

operators:
  - definition: kernelgenbench_square
    reference_triton_path: references/square.py
    reference_triton_prompt_path: references/square.md

  - definition: kernelgenbench_sin

  - definition: flash_attention
    mode: kernelgen
    n_parallel: 3
    n_epoch: 2
    seed_code_path: seeds/flash_attention.py
```

`workspace` 和 YAML 内所有路径（包括 `catalog_path`）都相对于 YAML 文件所在目录解析；命令行 `--workspace` 可以覆盖顶层 workspace。参数优先级为内置及环境默认值、YAML `defaults`、显式命令行共享参数、单个 operator，越靠后优先级越高。SimpleOpt 的 `warmup_ms`、`benchmark_ms` 和 `num_trials` 分别控制预热、计时和试验次数，不能放进混合 KernelGen 条目的共享 defaults。两种模式都支持 `reference_code_path`、`reference_code_prompt_path` 和 `seed_code_path`；未知字段、重复 definition、缺失的 reference/seed 文件会在启动任何子任务前报错。认证 token 不允许写入 YAML，仍只通过 provider 环境或登录状态传递。

`kg status <batch-workspace>` 聚合 Batch 状态并列出全部子任务；`kg cancel <batch-workspace>` 向仍活动的子任务写入协作式取消请求；`kg list` 将同一 Batch 的子任务折叠为一个 Batch 条目。详细日志和中断续跑仍以子任务 workspace 为单位。Batch 暂不支持 `--foreground` 和顶层 `kg resume`。

现有 BatchSimpleOpt 暂时保留为旧脚本兼容入口，不参与 `kg` 的 YAML Batch 调度；调用方迁移完成后将其移除，避免长期维护两套并发和状态模型。

## 优化历史

`kg history <single-run-workspace> [--json]` 从单任务 workspace 内的 `.ledger.json` 读取逐轮性能历史。它只返回状态、性能值和时间等紧凑投影，不返回 workload 明细、候选代码、模型日志或 profile artifact；这些完整事实仍保留在 ledger 中。

JSON Schema v1 顶层包含 `run_id`、模式、算子、目标硬件、workspace、全任务 `best_scope/best_round/best_geo_mean`、series 数量、总轮数和 `series`。每个 series 对应一个独立 ledger，并包含 workspace 相对 `scope`、相对 `ledger_path`、该 series 的最佳结果以及逐轮记录。逐轮记录至少包含 `round`、`status`、`geo_mean`、`min_speedup`、`best_geo_mean_so_far`、`is_new_best`、`is_final_best`、`is_hack` 和 `evaluated_at`。

新任务 SimpleOpt ledger 的 scope 为 `stages/optimize/work`，KernelGen 为 `stages/optimize/work/1R/agent0` 等；每个 ledger 独立形成一条 series，不跨 Coder 拼接轮次。旧 v1 任务仍使用 `.` 或 `1R/agent0` 等原路径，下面保留旧任务的 JSON 示例。顶层最佳值指出全局最佳所在的 `best_scope`。合法任务尚未产生 measured round 时命令成功返回空 `series`，最佳字段为 `null`。Batch 根不提供混合历史，先用 `kg status <batch-workspace>` 获取子任务 workspace，再逐个查询。

```json
{
  "schema_version": "1.0",
  "kind": "run_history",
  "run_id": "2e35a47c87d0430791bdcaaa75bc342a",
  "mode": "simple_opt",
  "definition": "kernelgenbench_square",
  "target_hardware": "NVIDIA H800",
  "workspace": "/path/to/kernelgenbench_square",
  "best_scope": ".",
  "best_round": 1,
  "best_geo_mean": 0.9802126621498061,
  "series_count": 1,
  "round_count": 1,
  "series": [
    {
      "scope": ".",
      "ledger_path": ".ledger.json",
      "definition": "kernelgenbench_square",
      "target_hardware": "H800",
      "implementation_language": "triton",
      "best_round": 1,
      "best_geo_mean": 0.9802126621498061,
      "round_count": 1,
      "rounds": [
        {
          "round": 1,
          "status": "PASSED",
          "geo_mean": 0.9802126621498061,
          "min_speedup": 0.9067796322768129,
          "best_geo_mean_so_far": 0.9802126621498061,
          "is_new_best": true,
          "is_final_best": true,
          "is_hack": false,
          "evaluated_at": "2026-09-04T15:34:02.173178Z"
        }
      ]
    }
  ]
}
```

## 运行控制

`status`、`logs` 和 `cancel` 直接调用 `kernelgen.framework.run_control`，不解析终端日志：

- `status` 与 `history` 共用 ledger 轮次和性能事实；`run-progress.json` 仅提供实时阶段、epoch 和 scope 生命周期，进程身份与 cancellation generation 各自独立。最终输出不回填或覆盖 ledger 性能事实。
- `logs` 默认按 `run-events.jsonl` 的 `sequence` 增量读取；`--follow` 使用轮询，不引入常驻服务。诊断原始进程输出时读取 `.kernelgen/runner.log`。
- `cancel` 调用 `request_cancel()`；重复请求保持幂等。
- 显式续跑使用当前 cancellation generation 调用 `clear_cancellation()`，防止旧进程清除新的取消请求。

KernelGen 模式的根进度保持稳定的生命周期 `state`，并用独立的 `stage` 与 `current_epoch/total_epochs` 表达细粒度活动。活动阶段依次为 `PREPARING`、`SHARED_ANALYSIS`、`CODER_RUNNING`、`EPOCH_FINALIZING`、`SYNTHESIZING`、`PROMOTING_BEST` 和 `COMPLETED`；例如 `state=RUNNING, stage=CODER_RUNNING, current_epoch=2` 表示第二个 epoch 正在运行 Coder。`scopes` 继续记录 `2R/agent0`、`1R/shared_analysis`、`1R/knowledge-merger`、`1R/synthesis` 等子任务，所有实际启动过的子任务必须在成功、失败或取消时写入终态。知识合并的模型调用只更新 `knowledge-merger` scope，根任务在该阶段保持 `SYNTHESIZING`；根任务终态不得用于掩盖残留的 `RUNNING` scope。

## 命名实例与首次启动

KGS 不再使用单独的准备命令。`kg server start <name>` 在实例不存在时完成配置、部署和后台启动；实例存在时只按持久配置检查或恢复进程。每个实例的配置、PID 身份和日志位于当前工作区 `.kernelgen/servers/<name>/`，本地 KGS checkout 默认位于当前工作区 `.kernelgen/kgs/<release>/`；远端 checkout 位于目标环境的 `remote_kgs_root`，不要求本地 checkout。实例不绑定 Catalog，但会记录显式安装的 FlagGems commit/path。

首次启动按固定顺序执行：读取 `deployment/kgs.lock.yaml`；从清单指定的 Gitee 仓库准备精确 KGS tag/commit；读取 KGS 自描述文件；按本地或远端目标安装 KGS 轻量依赖；后台启动进程；最后读取 `/status` 校验 Protocol、backend、worker 和设备 slot；release/commit 在 checkout 阶段核对，readiness 还用 server_version 核对实例身份，不以 release 推导协议能力。整个流程不选择 Catalog，不 clone 外部 framework，也不安装或升级 PyTorch、Triton、厂商扩展和设备运行时。远端 KGS checkout 必须由目标容器内的部署进程创建，不支持复用因宿主机 clone 而具有不同 ownership 的挂载仓库；发现此类目录时停止并报告，不通过 Git `safe.directory` 接管。KGS editable 安装统一使用实例配置的 Python 和 `pip --no-build-isolation`，直接复用容器已准备的 `setuptools`、`wheel` 等构建工具；目标环境必须预先满足 KGS 的 build-system 依赖，该选项不跳过缺失的普通运行依赖。

本地实例的最小输入为：

```bash
kg server start local-npu --target local --backend npu --devices 0,1 --timing auto --port 8000
```

Server 始终只监听 loopback。已有 checkout 与锁定 commit 一致时复用；目录不一致、有未提交修改或来源不明时停止并报告，不覆盖、不 reset。普通重复启动不执行 git pull、不选择新 tag，也不重复安装依赖。

保留已有 checkout 的受控入口，不提供离线部署模式：

```text
--kgs-root <path>        使用已有 KGS checkout，仍执行版本和 clean 状态校验
```

第一版不提供自动选择“最新 KGS”的参数。显式版本覆盖若后续增加，也必须来自已验证清单，不能按 Git tag 的 SemVer 大小推断。

## 本地与远端启动

本地首次启动使用 `--target local`，并在启动前验证本地 Torch、Triton 和厂商扩展。远端首次启动使用 `--target remote`，必须显式提供远端 loopback 端口和本地 loopback 代理端口：

```bash
kg server start ascend-01 \
  --target remote \
  --backend npu \
  --devices 0,1 \
  --remote-port 18306 \
  --listen-port 19606 \
  --ssh-command 'ssh -p 2224 -i /path/to/deploy-key user@host' \
  --container kernelgen-ascend \
  --remote-env-file /data/kernelgen/deployment.env.sh
```

`--ssh-command` 只能是不含远端命令的基础 SSH 连接命令，不得申请 TTY，也不应内嵌密码或 token。CLI 强制使用 batch mode、connect timeout 和固定 keepalive。`--container -` 表示直接使用 SSH 主机；其他值作为 Docker 容器名，远端命令通过 `sudo docker exec -i` 执行。

`--remote-env-file` 可选地记录目标执行环境中的机器侧部署环境文件。远端 helper 在 KGS checkout、KGS pip 安装和 FlagGems checkout 期间通过非交互 Bash 加载该文件，支持由机器自行提供 GitHub HTTPS 代理、Gitee SSH `GIT_SSH_COMMAND`、部署 key 路径和 `NO_PROXY`；环境文件标准输出会被丢弃，失败信息不回显文件输出或变量值。部署环境在每次准备操作结束后恢复，不注入后台 KGS，也不成为 `status`、`logs` 或 `stop` 的运行前提。实例停止后可以通过 `kg server configure <name> --remote-env-file <path>` 修改；修改后下一次 `start` 重新执行幂等部署检查。

远端启动依次执行：

1. 按需加载目标机器的部署环境文件，在目标容器中从锁定的 Gitee repository 准备精确 KGS tag/commit，不复制或覆盖本地 KG 工作树。
2. 调用精确 checkout 中的 `kernelgen_server/management.py`，先从该 checkout 的 `client/` 安装纯客户端包，再安装完整 KGS；由 KGS 检查两包的实际导入位置、记录安装前后版本，并确认受保护的 Torch/Triton/厂商包没有变化。KG 不再发送管理脚本源码，不保存 `remote_deployed` 或安装许可状态。
3. 在容器内后台启动只监听 `127.0.0.1:<remote-port>` 的 KGS，记录远端 PID 和启动身份。
4. 在开发机启动一个只监听 `127.0.0.1:<listen-port>` 的长期 SSH stdio HTTP mux proxy。默认 `max_streams = KGS max_workers + 2`。
5. 通过本地代理读取远端 `/status`，校验 Protocol、capabilities、backend、worker 数和健康且空闲的设备 slot。release 和 commit 在 checkout 阶段校验；readiness 另核对 server_version 是否匹配实例配置，该身份检查不是 wire protocol 兼容判断。代理启动失败时保留已完成部署并启动的远端 KGS，用户再次执行 `start <name>` 即可恢复代理。

`kg server start <name>` 的恢复矩阵是：KGS 与代理都存活时幂等返回；仅代理退出时恢复代理；KGS 已退出时清理残留代理并重新启动两者。`kg server status <name>` 通过一次性 SSH 和本地 endpoint 同时核实两端状态；`logs` 读取远端 KGS 日志；`stop` 先校验远端 PID 启动身份再停止 KGS，随后停止本地代理。SSH 连接命令保存在本机权限受限的实例配置中，不在 `status --json` 中返回。

实例配置只允许在两端都停止后通过 `kg server configure <name> ...` 修改；`--target` 和 KG/KGS 版本选择仍须使用新的实例名。Catalog 由 `kg run` 的请求选择，不进入 Server 实例配置；未来 Catalog checkout 应在其实际执行侧复用同一部署环境约定，而不是重新绑定到 Server。FlagGems 是可选的 Server 环境扩展，版本选择、持久化和 Catalog 匹配边界见上文 `kg server install-flaggems`；后续 `start` 注入已选 checkout 的 `KGS_FLAGGEMS_ROOT`。`kg server list` 不批量建立 SSH 连接，因此只报告本地可见的 `PROXY_RUNNING` 或 `REMOTE_UNKNOWN`，远端精确状态需查询具体实例。同一开发机 loopback 端口以及同一主机/容器内的设备 token 不允许被两个活动实例重复占用。

## 版本与配置归属

每项事实只保留一个机器可读来源：

KG 的 Agent、CLI 和 Web service 只导入 `kernelgen_client` 的协议模型、HTTP 请求、Catalog/Bundle 准备和纯来源发现；设备执行、Bundle 存储、Profiler backend 与调度仅在目标 `kernelgen_server` 中。KG 包依赖的客户端 VCS URL 与 `deployment/kgs.lock.yaml` 使用同一 KGS exact commit，并由版本测试核对；本地或远端 Server 的 checkout 和导入身份仍由 `kg server start` 校验。客户端包版本不是新的 wire compatibility 判断，运行时仍只看 `/status.api_version` 和 capabilities。

| 事实 | 唯一来源 |
| --- | --- |
| 当前 KG 推荐部署哪个 KGS tag/commit | KG `deployment/kgs.lock.yaml` |
| KGS 自身 release 和 Protocol version | KGS 精简自描述文件及运行时 `/status` |
| Catalog `api_version`、evaluator、layout | Catalog `manifest.json` |
| FlagGems repository 和首次安装的默认 revision | KGS `compatibility.yaml` 的 `frameworks.flaggems` |
| KGS 实例实际选择的 FlagGems commit 和路径 | KG 实例配置；显式 `install-flaggems --revision` 成功后更新，运行时校验实际 checkout |
| Native 冻结 oracle 要求的 framework revision | Native Catalog `manifest.json`；必须与实际 checkout 一致。新 KGS 的 Adapter Catalog 不重复绑定版本 |
| 已验证组合和验证范围 | KG 部署锁定清单及验证报告 |

KG release、KGS release 和 Protocol version 保持独立。KG 的锁定清单记录一组已验证组合，但不通过任何一个版本号编码或推导另一个版本。

KG 锁定清单的最小结构为：

```yaml
schema_version: 1
kg_release: vX.Y.Z
kgs:
  repository: git@gitee.com:BaaiAC/kernelgen_server.git
  release: vA.B.C
  commit: <full-commit>
validated_protocol: vM.N
validation_scope: host
```

KGS 后续自描述文件只保留自身事实，不再列出所有 KG release，也不重复 Catalog 信息；KGS 自带 evaluator 所要求的统一 framework 版本仍由 `frameworks` 声明：

```yaml
schema_version: 2
server_release:
  version: vA.B.C
  protocol_version: vM.N
supported_catalog_api_versions:
  - v6.0
  - v6.2
frameworks:
  flaggems:
    repository: <repository>
    branch: <source-branch>
    revision: <full-commit>
    revision_policy: exact
```

已经发布的 KGS tag 及其 `compatibility.yaml schema_version: 1` 保持不动。首次 `kg server start <name>` 兼容读取旧 Schema，但忽略其中历史 `kernelgen_releases` / `latest_validated_kgs` 反向矩阵：配套 KGS repository、release 和 exact commit 只由 KG 锁定清单决定，不要求旧 KGS 为新的 KG release 补登记。仍须校验 checkout 来源、精确 commit、clean 状态及 KGS 自描述与锁定清单一致，运行时协议与功能兼容仍只看 `/status.api_version` 和 capabilities；FlagGems 默认 revision 仍由 KGS 推荐，显式选择后的实际 commit 由 KG 实例记录。新的 KGS release 使用精简 Schema，不再增长交叉版本矩阵。

## 实现边界

CLI 只做参数解析、环境准备、后台进程生命周期、调用现有 Python 接口和呈现结果。新任务的 `kg run` 直接构造 `RunRequest` 并调用 `OperatorOptimizeWorkflow`；旧 v1 请求保留原 example launcher 的 `main(argv)` 用于历史 workspace 续跑，不将新任务转换为 launcher 参数。

KGS 部署逻辑独立成模块，由 `server start/list/configure/status/logs/stop/install-flaggems/doctor` 共用。实例配置记录目标、连接、端口、锁定版本和已启用的可选环境扩展，进程状态记录 PID 启动身份和日志路径；设备能力与 scheduler 状态始终以当前 `/status` 响应为准。

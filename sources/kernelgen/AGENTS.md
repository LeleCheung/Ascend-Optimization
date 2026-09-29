# KernelGen Agent 接手检查表

本文只保留接手任务时必须遵守的约束和文档入口。部署、实验命令和历史结果分别由
对应文档维护，不在这里复制。

## 先读什么

1. [项目上手指南](docs/ONBOARDING.md)：安装、核心概念和最小本地示例。
2. [远端 KGS 部署](docs/operations/deployment/remote_server.md)：当前连接、端口和容器配置；配套 KGS 的 `docs/multiple_device_deploy_tutorial.md` 用于服务端参数与自测，旧环境快照不覆盖当前机器清单。
3. [多设备实验与 E2E 验收手册](docs/operations/multiple_device_experiment_runbook.md)：机器
   连接、两种运行模式、BatchSimpleOpt、监控和结果回收。
4. [KernelGen E2E 验证报告](docs/validation/multiple_device_e2e_validation_report.md)：历史
   结果与已知问题。
5. `kernelgen_server/docs/multiple_device_eval_validation_report.md`：Server Eval
   和算子兼容性结果。
6. [实验结果整理手册](docs/operations/experiments/experiment_result_maintenance.md)：正常完成口径、best/last code
   归档、静态审查、原始路径和汇总脚本。

## 接手后先检查

```bash
cd /data/akg_kernel_bench_lite/kernelgen
git status --short --branch
git log -5 --oneline --decorate
sed -n '1,160p' tests/hosts.md
```

- 工作区可能有其他人的未提交修改。只修改和暂存本任务文件，禁止
  `git reset --hard`、`git checkout -- .` 和全量 `git add .`。
- 版本管理统一为三个相互独立的概念，不得用其中一个版本号编码另外两个版本或跨仓库兼容关系：
  - **KernelGen release**：写作 `KG vX.Y.Z`，只表示 KernelGen 软件发布版本，由 KernelGen 的 Git tag、commit 和包元数据记录。
  - **KernelGen Server release**：写作 `KGS vX.Y.Z`，只表示 KernelGen Server 软件发布版本，由 Server 的 Git tag、commit 和包元数据记录。
  - **Protocol version**：写作 `Protocol vX.Y`，表示 Agent 与 Server 的 wire protocol 及其 Schema 兼容版本；当前协议字段仍为 `api_version`，但文档和沟通中不再把 API version、Schema version 当成独立版本概念。
- Agent 只按 `/status.api_version` 判断协议兼容性，具体功能是否可用按 `/status` capabilities 判断，不得根据 KG 或 KGS release 推断；KGS release 不作为 `/status` 的协议兼容字段。
- 正式新实验的配套 KGS repository、release 和 exact commit 由当前 KG 的 `deployment/kgs.lock.yaml` 唯一决定，不得按 tag 的 SemVer 大小或 KGS 历史 `kernelgen_releases` / `latest_validated_kgs` 反向矩阵选择，不要求旧 KGS 为新的 KG release 补登记。仍必须读取锁定 KGS 的 `compatibility.yaml` 校验自身 release、Protocol 和 framework 来源策略，并读取 Catalog `manifest.json` 核对输入契约。Catalog `api_version=v6.0` 默认使用 FlagGems adapter/flat layout，`api_version=v6.2` 默认使用 native/per-operator layout；这项部署选择规则不替代 `/status.api_version` 和 capabilities 的运行时协议检查。FlagGems 按清单指定的 repository 和 revision_policy 准备：branch policy 在新实验的服务准备阶段解析默认分支 HEAD，exact policy 使用清单 commit。已配置默认分支的实例在后续 `kg server start` 时重新检查 HEAD；仅已停止实例可以切换到新 checkout，不自动停止或替换运行中的服务。每次准备固定并记录实际 commit，checkout 保持 clean；`kg run`、Eval/Profile 和 `kg resume` 不触发更新。用户可通过 `kg server install-flaggems <name> --revision <分支或完整小写commit>` 显式选择版本；完整 commit 或非默认分支不会被普通 start 切回默认分支。支持运行时版本选择的新 KGS 不再用 Adapter Catalog 的 `framework_revision` 限制 Gems；更新后重新 inspect 和验证 baseline，不因 commit 不同就要求重建 Adapter Catalog。FlagGems-backed Native Catalog 仍须匹配其冻结 oracle 的 exact revision，不得只改 manifest 绕过校验。旧实例保留其原 KGS commit，不自动升级。中断续跑必须沿用 workspace 已记录的 KGS、framework 和 Catalog 快照，不得在 campaign 中途自动升级；具体边界见 [Gems 来源策略](docs/design/gems_source_policy.md)。
- KG 和 KGS release 各自遵循 SemVer：修复递增 patch，向后兼容的新功能递增 minor，破坏性软件变更递增 major；只修改一侧软件时只递增该仓库版本，即使 Protocol 长期不变也不得让软件版本依附于 Protocol 版本序列。向后兼容的协议扩展可以递增 Protocol minor 并保留旧协议，破坏性协议变化必须递增 Protocol major。
- 每次发布和验证统一记录 `KG vX.Y.Z@commit / KGS vX.Y.Z@commit / Protocol vX.Y`，同时写明兼容条件和实际通过测试的两个仓库 tag/commit；ONBOARDING 固定当前推荐组合，验证报告记录实验实际组合。下一发布版本由发布计划决定，不在本文硬编码；不得移动已经发布的 tag，除非用户明确要求。
- `kernelgen_server` 代码同步到目标机必须通过 Gitee。首次真机验证可以从当前开发
  分支创建临时 `test-*` 分支，目标机只拉取该临时分支；临时分支不得成为长期部署
  分支。若验证发现问题，将修复合并或拣选回创建它的当前开发分支，复验后删除本地和
  Gitee 上的临时分支及对应 worktree；若验证没有发现问题，同样删除临时分支，后续
  部署直接使用当前开发分支。创建临时分支前记录当前开发分支名和 commit，避免把
  “当前分支”解释为操作时偶然 checkout 的其他分支。禁止用 tar、scp 或同步脚本直接
  覆盖目标机上的 Server 工作树。
- 目标机访问 Git 仓库时只使用仓库级代理和项目专用只读部署 key，不通过 `export`
  修改进程或系统全局代理，不使用部署 key 访问授权范围外的服务。具体代理、key
  路径和 SSH 参数以多设备实验手册及 `tests/hosts.md` 为准；
  不得打印、复制或提交凭据和私钥内容。
- 为避免同步 KernelGen 代码，多设备优化优先使用“本地 Agent + 远端 Server”，
  并且必须通过项目的 SSH stdio HTTP proxy 访问远端 loopback 端口；不得让远端
  Debug Server 监听公网或直接暴露端口。单机交付验收仍默认“本地 Agent + 本地
  Server”，但用户明确授权的多设备 BatchSimpleOpt 可以采用远端模式。
- 本地环境可能设置全局 Squid `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`。访问 SSH
  stdio proxy 的 `127.0.0.1:<port>` 时，`curl` 使用 `--noproxy '*'`；Python/Agent
  命令为当前进程设置 `NO_PROXY=no_proxy=127.0.0.1,localhost`，并取消该进程继承的
  大小写 HTTP/HTTPS/ALL proxy 变量。不得为此修改系统全局代理配置。若多个本地
  proxy 端口同时返回 HTTP 502，先用
  `curl --noproxy '*' http://127.0.0.1:<port>/status` 复查；全局 Squid 返回的
  502 不代表 SSH 隧道或远端 Server 已退出。
- `env.sh`、`env.opus.sh` 可能含模型配置或凭据。可以 `source`，不得打印、提交或
  写入日志。
- 后续实验不再通过 mitmproxy 收集模型请求轨迹，不部署中间代理或额外 CA 证书，也不把轨迹文件数量作为实验验收项；历史轨迹配置和结果只保留在验证报告中。
- `env.sh` 或 `env.opus.sh` 因本地模型配置产生未提交修改时，不以此阻塞其他主题的
  干净 worktree 检查，但不得暂存或提交这些文件；该例外不得扩展到源码、测试、
  adapter 或其他实验逻辑文件。
- 远程操作前以 `tests/hosts.md` 为准；历史报告中的 IP、
  容器、端口和卡号都只是当时快照。
- `kernel_todo_v1/` 和 `kernel_todo_v2/` 是 Git 忽略的本地实验数据目录，不随仓库
  分发。使用依赖这些目录的脚本前，必须从授权的实验归档恢复输入，或按对应工具文档
  重新生成；脚本中的默认路径只是本地工作约定，不能假设全新 clone 已包含数据。

## 分支管理、合并与删除规范

- 开始任务前先 `git fetch <remote> --prune`，从团队约定的最新目标分支创建 feature：KernelGen 默认基于 `origin/dev`，KernelGen Server 默认基于 `origin/main`，其他仓库以其约定开发分支为准。禁止直接 push 共享目标分支或移动 tag，除非用户明确授权。
- 一个 feature 只承载一个可独立评审的主题。与当前主题无关、可被其他分支复用的公共改动必须从最新目标分支另建 feature 并单独提 PR；公共 PR 合入后，仍活跃且尚未合入的业务 feature 才可以 fetch 并 merge 最新目标分支。跨仓库改动在各仓库分别建分支、提交、验证和提 PR，不用一个仓库的分支替代另一个仓库的版本管理。
- 同时开发多个可独立评审的主题时，必须从同一个最新目标分支分别创建 feature，并为每个 feature 建立独立 worktree；每个 worktree 只承载、提交和验证自己的主题，分别提 PR。同一分支不得同时被多个 worktree checkout，不得复制工作目录或从已关闭的旧 feature 派生新分支来并行开发。多个主题需要修改同一文件时，按 hunk 精确迁移各自改动；若后续主题依赖先行主题，先合入先行 PR，再由仍未合入的活跃 feature 同步最新目标分支。
- 如果多个独立主题已经在同一分支或 worktree 中一起开发，必须在提交前将各主题的文件或 hunk 迁移到分别基于最新目标分支创建的 feature worktree，并在各自分支独立提交、验证和提 PR；禁止先创建包含多个主题的混合提交，再依赖后续拆分来恢复分支边界。迁移过程中保留原工作区未提交内容，不得用全量 stash、reset、checkout 或全量暂存处理其他人的改动。
- 只有尚未合入的活跃 feature 可以同步目标分支。同步前先检查工作区和分支拓扑，临时保存时只处理本任务路径并恢复原有未提交内容；禁止对已共享分支强制 push，除非协作者明确同意。合并冲突必须同时保留目标分支的最新公共语义和 feature 的净新增能力，并运行冲突区域及任务相关测试。
- feature PR 无论以 merge、squash 还是 rebase 方式合入，源分支都立即视为关闭，不得继续开发、再次 merge 目标分支、fast-forward 为目标分支或复用同名分支。后续工作先 fetch，再从最新目标分支创建新的 follow-up feature；旧分支中仍需保留的未合入提交只能通过明确的 cherry-pick 或重新整理迁移。
- feature 分支是临时开发指针，不是合并后的功能档案。提 PR 时必须使用可检索的标题，并在描述中记录源分支名、解决的问题、主要改动、验证结果和关联设计文档；不得使用“update”“fix”等无法识别功能边界的泛化标题。merge/squash 后保留在目标分支的提交标题必须概括功能并带 PR 编号，提交正文至少记录原因和验证，使删除源分支后仍能从 PR 和主线历史还原用途。
- 发布时必须在 release notes 中按功能汇总已合入 PR，记录“PR 编号、原分支名、目标分支提交和功能摘要”的对应关系；重要架构变更还必须链接设计或决策文档。日常追溯以 PR 和目标分支提交为准，不依赖已合入分支继续存在。
- PR 合入后先确认改动已经进入目标分支，并确认上述 PR 和提交追溯信息完整，再删除源分支。普通 merge/rebase 可以检查祖先关系；squash 必须同时核对 PR 状态以及 patch、目标文件树或等价提交，不能只因原提交不是祖先就判断未合入。确认后删除关联 worktree、本地分支和远端分支，并执行 fetch/prune；优先使用 `git branch -d`，只有在 squash 等已确认合入但拓扑不相连时才使用 `git branch -D`。PR 未合入或仅关闭时，不得删除仍含唯一改动的分支，必须先保留或迁移需要的提交。
- 分支重命名前确认新名称在本地和远端都不存在；先重命名本地分支并把完整 HEAD 推送到新远端、设置 upstream，验证两端 commit 一致后再删除旧远端分支。重命名不得丢失工作区未提交内容，旧分支删除后不再复用。

## 提交前门禁

- 只用 `git add -- <本任务文件...>` 显式暂存当前主题，禁止 `git add .`；随后检查 `git status --short --branch`、`git diff --cached --name-status` 和完整的 `git diff --cached`，确认暂存区不含其他人的修改或跨主题文件。
- 对暂存内容运行 `git diff --cached --check` 和与改动直接相关的测试；文档改动至少检查链接目标、命令和版本描述，代码改动不得仅凭静态阅读提交。
- 在独立 worktree 中运行 Python 测试前，先检查被测模块的 `__file__` 确实位于当前 worktree。本项目从仓库父目录发现 `kernelgen` 包，非标准目录名的 worktree 可能被已有 editable install 重定向到主工作区；此时必须通过显式导入路径或临时父目录中的 `kernelgen` 链接隔离测试，不能把导入其他工作区所得结果作为门禁证据。
- 提交前检查暂存文件中是否包含 token、密码、带用户信息的代理 URL、私钥、`env.sh` 或其他本地凭据。敏感信息检查不得把匹配内容打印到终端或日志；发现命中时只报告文件名并从暂存区移除，确认清理后再提交。

## 不可违反的实验约束

- 本地 Agent + 远端 Server 通过项目的 SSH stdio HTTP 代理访问；Agent、MCP、
  Catalog 和 workspace 留在本地，编译、执行、计时和 Debug Job 位于目标机。
  Agent 必须先读取远端 `/status`，所有目标环境查询都通过远端
  `submit_debug_job`，不得用 Agent 本机 Torch/Triton 环境推断目标能力。
- 禁止安装、升级或替换 `torch`、`triton`、`torch_npu` 和厂商运行时。确需安装
  轻量依赖时，记录安装前后版本和新增包。
- Server Debug Job 是可信任环境中的任意命令执行接口。Server 只绑定 loopback；
  不得在没有网络隔离的情况下启用远端 Debug。
- Server 隔离子进程保持 `daemon=False`。昇腾 profiler 需要创建子进程。
- Server 的设备队列必须与可见设备一一对应：同一时刻每张卡最多运行一个 Eval、
  Profile 或 Debug Job。`--max-workers` 是请求执行线程上限，可以大于设备数，但
  不得复制设备 token。启动后检查 `/status.scheduler`：
  `device_slots == healthy == 设备数`、`checking=broken=0`、
  `max_active <= device_slots`；空闲时必须为
  `active=0`、`waiting=0`、`available=device_slots`。Server 启动时会逐卡执行
  最小计算和同步预检；请求 `TIMEOUT` 或隔离 worker 异常退出后，当前 slot 先进入
  `checking`，独立探针通过则恢复，失败才标为 `broken`。普通算子
  `RUNTIME_ERROR` 不触发探针。Server 不再跨 slot 自动重试：强探针通过则恢复原
  slot，超时请求返回 `TIMEOUT`，worker 异常退出保留原错误；强探针失败才把原
  slot 标为 `broken` 并返回 `SUSPECTED_DEVICE_ERROR`。该结果写入 ledger 供后续
  排查，Agent 不得将其归因于 candidate 或继续修改代码；完成该轮
  `finalize_round` 后再次读取 `/status.scheduler`。只有 `broken>0` 才表示强探针
  确认 slot 不可用并需要人工恢复。`TIMEOUT` 且 slot 已恢复表示设备计算链仍健康，
  由操作方决定是否增大 timeout 后重跑，Agent 不得自行把请求转发到其他 slot。
  按实验手册规定的调度窗口边界检查 `/status.scheduler`。若 `broken>0`，当前窗口
  只使用其余健康 slot，完成后停止开启新窗口并人工检查；若 `incidents` 重复增长但
  探针均恢复，也在当前窗口结束后人工复测。不要由 Server 自动执行物理卡复位。
  事故背景见
  `kernelgen_server/docs/device_slot_isolation_incident.md`。
- Server 启动时所有可见设备并行运行强探针，覆盖矩阵乘、softmax、同步和目标
  计时链路；启动耗时应接近最慢单卡探针，而不是各卡耗时之和。
- 昆仑芯存在已确认的厂商编译器/运行时进程级退出和超时问题，默认不得为其启动
  BatchSimpleOpt 队列或强模型大规模生成。只有用户明确要求诊断时，才允许使用独立
  workspace 做单算子、低并发实验；恢复阈值和操作步骤见多设备实验手册及
  `kernelgen_server/docs/device_slot_isolation_incident.md`。
- 一个单算子 workspace 同时只能有一个 Coder。全新的 campaign 使用新的 run
  name 和 Batch workspace，不在旧 workspace 上通过 `--clean` 重跑；同一
  campaign 的中断续跑则必须复用原 Batch workspace，真正的隔离单位是
  `workspace/definitions/<definition>`。不同算子可以在各自目录并行；同一算子
  不得并发启动两个 Coder。已有最终输出的目录保持不动，已有 ledger 但未完成的
  目录原地续跑，从未进入 Coder、只有初始化文件的目录也优先原地重排。现有目录
  默认不得删除或移动；确需从头重跑时先保留旧证据并获得用户确认。批次续跑只输入
  未完成算子，汇总以每个算子的 `optimize_definition_output.json` 和 ledger 为
  准；不得用一次重试生成的顶层 `batch_simple_opt_definition_output.json` 覆盖或
  代替整个 campaign 的累计结果。
- BatchSimpleOpt 的 Agent `--max-workers` 可以高于设备数，因为生成和分析阶段不占卡；它与 Server 设备 slot 不是同一个概念。一个 campaign 可以包含全部符合条件的 definition，`2N` 等限制只描述当前并发调度窗口，不要求拆成多个 campaign 或复制 Batch workspace。窗口大小、Server workers 和监控频率按实验手册及当次资源状态确定，并记录采样依据；不得通过让多个 Eval 同卡运行来推算容量。

## 开发和验证顺序

1. 先运行与改动直接相关的 host 单元测试。
2. Server 或 Schema 改动先做 reference-as-solution、preflight、
   `RUNTIME_ERROR`、并发和资源释放验证。
3. 在一台代表性芯片完成最小 SimpleOpt/BatchSimpleOpt E2E。
4. 再按任务范围扩展到其他芯片。
5. 回收结果并使用汇总脚本生成 Markdown/JSON，不手工抄写加速比。

常用 host 测试：

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q tests
```

多设备操作和完整命令统一见
[多设备实验与 E2E 验收手册](docs/operations/multiple_device_experiment_runbook.md)。

## 判定边界

- FlagGems pytest 转换或待测代码注入若无法由当前协议准确表达，不得通过修改 reference 语义、删除 workload、增加算子专用旁路或其他强制适配方式伪装成通过。将算子、失败证据、协议缺口、影响范围和可选处理方案记录到本地 `kernel_todo_v2/pytest_conversion_protocol_gaps.md`（文件不存在时创建），保留为待确认状态，继续处理不受影响的算子，待用户统一确认后再修改协议或测试。
- `PASSED` 表示全部 workload 正确并具有有效 headline timing，不等于
  `best_geo_mean > 1`。
- 当前多设备优化把 `best_geo_mean >= 0.8` 作为结果合格线；数值正确和有效计时仍
  是前置条件。若 workflow 的 `PASSED` 仅表示 correctness/timing 完整，汇总时必须
  按该阈值另行分类，不能把所有 `PASSED` 都计为优化合格。
- 达到当次配置的轮数上限后仍为 `PARTIAL_PASS/FAILED`，通常是优化未收敛，不应直接归因于
  Server/Schema。
- 0 round `FAILED` 优先检查 Agent/API、preflight、reference、Schema 和芯片
  capability。
- 厂商明确的 `NOT IMPLEMENTED`、缺少设备 kernel 或不支持参数属于硬件 API
  能力；不要通过修改 reference 语义伪装成通过。
- 直接调用 `torch.ops.*` 的候选可能是框架 fallback。即使数值通过，也要确认
  best kernel 是目标实现。

## 交接前检查

- 记录两个仓库 commit/tag、容器、解释器、核心包版本、卡号、backend、timing、
  slot、Agent workers、模型、run name 和 workspace。
- 列出新增 Python/npm 包，确认 Torch、Triton 和厂商运行时未变化。
- 保留启动命令、Server/Agent 日志、最终 JSON、ledger 和 best kernel。
- 确认没有遗留 Agent/Coder，Server slot 已 idle；回收完成后关闭临时 Server。
- 将操作方法更新到实验手册，将实际结果更新到验证报告，不在本文追加历史流水账。

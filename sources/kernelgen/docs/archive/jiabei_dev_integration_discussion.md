# dev-xy 与 jiabei-dev 功能差异及 SimpleOpt 集成讨论稿

> 状态：讨论稿
> 对比日期：2026-08-05
> 当前侧：`dev-xy`，本文已包含 K-06、S-11、S-13、S-14 和 S-17 适配，release base tag `v5.1.1`
> jiabei 侧：`origin/jiabei-dev`，`2141e65`（2026-08-03 已 fetch 并核对远端）
> 共同祖先：`1b1f05d4f40eee05784bdd2a85bf952c07617e03`
> jiabei 真实运行基线：昇腾宿主机
> `/data/jiabei/kernelgen-campaigns/campaign-02-test`（本文统计快照截至
> 2026-08-03 20:26）

本文用于和 jiabei 确认后续整合边界，不是整分支合并方案，也不表示已经决定采用
`jiabei-dev` 的全部实现。当前工作区没有执行 merge 或 cherry-pick；已经确认的
能力按模块适配到当前实现。

> 2026-08-04 实施状态：S-07 ledger `3.0`、V1 Knowledge runtime/corpus、
> Knowledge Distiller、K-05 campaign/finalize、K-06 solution registry 和 K-08
> role/Skill 均已完成适配。SimpleOpt/BatchSimpleOpt 明确排除 V1 KB：没有
> Knowledge 配置、workspace state、KB/Skill 复制或 KB Agent 工具。

> 2026-08-07 后续决策：本文原有 K-04 边界被部分更新。默认 SimpleOpt 和
> BatchSimpleOpt 仍不使用 V1 KB；单 Definition SimpleOpt 新增显式
> `--knowledge-catalog-path PATH` 模式。该模式 materialize
> Knowledge state，选择 Knowledge Coder/Profile/Distiller，并保留本地
> CandidateDraft，但不执行 epoch publication、Solution promotion、fork 或多
> Agent synthesis。本文后续“SimpleOpt 不接入 V1 KB”的历史比较均应结合本注释
> 理解。

本文所说的 campaign，是一次完整的 KernelGen 多算子、多 epoch、多 Agent 批量
实验目录，不是 Git 分支。`campaign-02-test` 保存了每个算子下各 epoch 的
Analyzer、Agent、Synthesis workspace，以及 ledger、best kernel、Profile、
Knowledge retrieval 和 publish 结果。本文主统计只计算活动 workspace；
`_archive/` 中的中断现场单列，不能混入活动结果。

本次不是只检查某一段新增功能，而是从共同祖先重新扫描
jiabei 侧 `1b1f05d..2141e65` 的 87 个提交和当前侧
`1b1f05d..999287e` 的 41 个提交，并以两边最终代码和上述真实 campaign 为准
复核中间演进。审计范围
包括 prompt、Agent 输出合同、ledger schema/migration、lineage、错误恢复、
workspace 隔离和工具暴露面；不能因为提交标题是 fix/refactor 就忽略其运行语义。

**已确认前提：SimpleOpt 以当前分支为准。** 这包括它的职责边界、公开输入输出、
`OptimizeDefinitionWorkflow`、BatchSimpleOpt、当前 `kernelgen_server` v5.1
适配、round lifecycle、anti-hack、硬 `max_round`、设备事故停止语义和已验证的
多设备行为。
`jiabei-dev` 只作为新增能力和后续 KernelGenWorkflow/KB 设计的参考来源；任何
共享模块的引入都必须适配当前 SimpleOpt，不能反向要求 SimpleOpt 迁回 jiabei 的
旧 Workflow 或旧 Server 接口。

## 1. 结论

1. **SimpleOpt 以当前分支为唯一实现基线。** 当前 SimpleOpt 已经收敛为
   “一个现有 Catalog Definition 对应一个独立优化任务”，并已适配独立的
   `kernelgen_server` v5.1、BatchSimpleOpt、多设备运行和严格的 round lifecycle。
   `jiabei-dev` 的 SimpleOpt 仍把 PyTorch 提取、多个 Definition 和并行优化放在
   同一个 Workflow 中，不能覆盖当前实现。
2. **从 jiabei 选择性适配与 SimpleOpt 直接相关的能力。** Debug Job 已适配为
   单工具同步调用，静默输出时的 Runtime 超时修复和结构化 `TIMEOUT` 也已完成。
   Server Status 也已注册为只读 MCP，Coder role 要求每个 session 开始时调用
   一次；当前还已完成 v5.1 `api_version` 协议绑定、scheduler
   状态暴露和 `SUSPECTED_DEVICE_ERROR` 停止处理。最新 jiabei 也已收敛为单个同步
   Debug 工具，差异已从“工具形态”缩小为旧 `eval_service` adapter 与 v5.1
   事故语义。下一步可独立适配 Agent 输出合同、Distiller 局部修复和标准化单次
   运行报告。
3. **round lifecycle 只在 Profile 策略上采用 jiabei 模式，其余 SimpleOpt 门禁保留
   当前语义。** 当前代码已改为仅对非 hack 的 new best 发出请求，由 Agent 按诊断
   价值决定是否执行，pending Profile 不阻塞 conclusion、下一轮或 STOP，收尾时
   对最终 best 做一次 best-effort Profile。最新 jiabei 也已恢复 content-bound
   Preflight 门禁，但当前分支的
   一次性 receipt 和 v5.1 协议绑定更完整，继续以当前实现为准。Conclusion、
   candidate transition、Python-owned STOP、`max_round=15`、默认 3-round
   plateau 和设备事故停止语义均继续沿用当前分支。
4. **结构化 KB 已按 jiabei 设计接入 KernelGenWorkflow，不接入
   SimpleOpt/BatchSimpleOpt。** 当前已引入 V1 Catalog/corpus、Knowledge Skill、
   MCP、workspace materializer、Knowledge Coder/Profile/Distiller、epoch
   Publisher 和 lineage 审计；K-08 保持 jiabei 的 prompt/Skill 指导与 Python
   非阻塞语义。K-05 的 finalize recovery/campaign barrier 和 K-06 的
   `fresh/fork/resume`、`flaggems-v5-v5.1` benchmark identity、严格更优
   solution promotion 均已完成。
5. **不建议直接合并整个分支。** 两边的 Server API、SimpleOpt 职责、ledger
   schema 和 lifecycle 已分别演进。适合按模块移植设计和测试，不适合用冲突解决
   代替设计决策。

## 2. 相关性标记

| 标记 | 含义 |
|---|---|
| **SimpleOpt-直接** | 会改变当前 SimpleOpt/BatchSimpleOpt 的输入、状态机、结果或 Agent 工具 |
| **SimpleOpt-共享** | 属于公共 Agent/MCP/Runtime 基础设施，SimpleOpt 会使用，但不改变其职责 |
| **后续专题** | 主要属于 KernelGenWorkflow 或 KB，本轮只登记接口和分歧 |

## 3. 模块总览

| 模块 | 相关性 | 当前分支 | `jiabei-dev` | 初步处理意见 |
|---|---|---|---|---|
| SimpleOpt 职责和批处理 | **SimpleOpt-直接** | 单 Definition 优化；BatchSimpleOpt 另行编排并默认错峰启动 | 提取算子并并行优化多个 Definition | 保留当前边界 |
| Native Agent 与 MCP 基础设施 | **SimpleOpt-共享** | 普通与 Knowledge 角色隔离；SimpleOpt 仅获得普通 Coder/Profile 工具，KernelGen 可显式获得 KB MCP | 基础设施完整，多了 KB、Debug 和 `next` MCP | 已按角色选择性适配，不覆盖普通 SimpleOpt 权限 |
| Round lifecycle、Profile、STOP | **SimpleOpt-直接** | Profile 已改为 non-hack new best 按需执行、非阻塞，最终 best 由 Workflow best effort 补齐；Python 继续负责其他状态转换和最终 STOP | new best Profile 由 Agent 按需执行、非阻塞；最终 best 由 Workflow best effort 补齐 | **已完成**：Profile 采用 jiabei；其余 lifecycle 保留当前 |
| Implementation Profile 与目标设备 | **SimpleOpt-直接** | Triton Profile 完整；默认目标仍为 Ascend910B | Profile 基本相同；SimpleOpt 目标默认空 | 保留 Profile，统一目标来源 |
| Distiller 与运行报告 | **SimpleOpt-直接** | SimpleOpt 保留 legacy report-only Distiller；Knowledge KernelGen 使用固定报告、Python metadata、Profile 路径和逐 Candidate repair | 固定报告、Python 元数据、按路径读取 Profile；无效 Candidate 逐个修复和跳过 | Knowledge 链路已适配；不向 SimpleOpt 引入结构化 KB |
| Debug Job、Server Status 与设备事故 | **SimpleOpt-直接** | 同步 `submit_debug_job`、只读 `get_server_status`、scheduler 状态和 `SUSPECTED_DEVICE_ERROR` 停止语义 | 同样只暴露同步 `submit_debug_job`；仍是旧 Server adapter，无直接 Status MCP 或 v5.1 事故语义 | 保留当前 adapter；复用约束与审计设计 |
| Runtime、日志和 Agent 合同 | **SimpleOpt-共享** | 已有日志镜像、thinking 去重和静默超时修复 | 另有环境隔离、`Literal`/约束渲染、多 fenced JSON 容错 | 组合移植终态修复，不替换进程生命周期 |
| KernelGen 多 Agent/Epoch/Campaign | **后续专题** | 已适配当前 Server、OptimizeDefinition、finalize recovery、campaign barrier 和 exact-scope Solution | 有 finalize、campaign、KB barrier、按需 Analyzer 和 solution registry | **已完成 K-05/K-06 适配** |
| 结构化 KB 与治理 | **后续专题** | V1 runtime/corpus、Knowledge 角色、结构化 Distiller、epoch Publisher、lineage、campaign 和 Solution Registry 已接入 KernelGenWorkflow；SimpleOpt 后续增加显式查询模式 | 完整 V1 查询、统一 Skill、实验决策双路检索、发布、审计和治理 | **主体适配完成**；SimpleOpt 只复用查询、Profile 和本地 Distiller 链路 |

### 3.1 已完成适配记录

以下能力已经在当前分支落地，不再只是讨论项：

| 能力 | 当前实现方法 | 验证状态 |
|---|---|---|
| Server Status MCP | 无参数 `get_server_status` 从 workspace ToolContext 读取固定 Server URL，原样返回 `/status` 的 target/software/devices/timing/profile/debug/scheduler；Coder role 要求 session 开始时调用一次 | MCP schema、endpoint 绑定和错误返回 host 测试覆盖 |
| 同步 Debug Job MCP | Coder 只获得 `submit_debug_job`；MCP 从 ToolContext 读取 Server URL 和目标设备，一次调用完成 capability 校验、源码上传、等待终态、日志落盘和 artifact 下载 | Host schema/adapter 测试通过；Ascend910B 真机通过 |
| Server 同步 Debug API | `POST /debug/jobs` 等待任务完成并直接返回 `SUCCEEDED/FAILED/TIMEOUT/CANCELLED`；GET/DELETE 仅作为兼容和运维接口保留 | 已合入 `kernelgen_server` Gitee `main`：`622a0de` |
| 结构化执行超时 | Server 捕获隔离子进程 `TimeoutError`，通过 HTTP 200 返回 `status: TIMEOUT`；evaluate 和 preflight 不再把预期超时包装成 HTTP 500 | Server timeout 单测及 910B Debug 超时测试通过 |
| 静默 Runtime 超时 | CLI stdout 由后台线程读取，主循环每 100 ms 检查 idle/hard timeout；超时或中断时清理整个 CLI/MCP 进程组 | 静默 stdout、stderr 活跃、EOF 竞态和子进程清理测试通过 |
| Server v5.1 协议绑定 | 从 `kernelgen_server.KERNELGEN_API_VERSION` 取得 API 版本；`/status` 和 preflight receipt 只以 `api_version` 作为协议身份，并绑定 backend/timing；Server 软件版本由 Git tag/commit 单独记录；ledger evaluation 使用 `api_version` 并兼容读取旧字段 | preflight、eval、profile、ledger 和 adapter host 测试覆盖 |
| 设备可疑错误停止 | preflight 的 `SUSPECTED_DEVICE_ERROR` 不创建 round；eval 的同名结果写入 ledger，Coder 先以 `finalize_round` 持久化 `reason_code=suspected_device_error` 和 STOP，再读取一次 scheduler snapshot，且不把事件归因于 candidate | finalize_round、preflight、eval、MCP 和 Coder role host 测试覆盖 |
| Batch 启动错峰 | `run_parallel` 支持 `launch_interval_seconds`；BatchSimpleOpt 默认 1 秒，减少远端网关瞬时连接峰值 | BatchSimpleOpt 和通用并行框架 host 测试覆盖 |
| 默认 plateau 早停 | SimpleOpt、ExtractOpt 和 KernelGen 的 `early_stop_rounds` 默认统一为 3；只统计通过但未刷新 best 的 round，硬 `max_round=15` 不变 | stop policy 及各 Workflow host 测试覆盖 |
| 四层时间预算 | Coder hard/idle 默认 `1800/1500s`，Server execution 默认 `600s`，HTTP 配置默认 `1200s`；v5.1 eval/preflight 实际 transport 下限为 `2 × execution + 90s`，默认即 1290 秒 | SimpleOpt/BatchSimpleOpt 参数、ToolContext 和 adapter host 测试覆盖 |
| Profile 按需与最终 best 收尾 | 只对非 hack 的 new best 标记 pending；Coder 根据诊断价值决定是否立即调用；pending 不阻塞 conclusion/candidate transition/STOP/下一轮；Distill 前 best effort 补最终 best，失败尽量写 terminal evidence | Profile/ledger/MCP/Coder/Workflow/真实 stdio 脚本均已更新 |
| Round finalization 与目标身份 | `record_round` 已改名为 `finalize_round`，继续原子执行 conclusion、candidate transition 和 STOP；启动 Coder 前从 v5.1 `/status` 构造严格 TargetContext，device/backend 缺失或输入不一致时拒绝运行 | 全量 host 回归通过 |
| Coder 上下文连续性 | 正常情况下所有 round 都在一次 Coder invoke 中完成；若 Coder 在 `finalize_round` 返回 CONTINUE 后提前输出报告，Workflow 必须用 session ID 续接同一个 Claude 对话；无法 resume 时直接失败，不启动全新 Coder context | Runtime、Agent 和 SimpleOpt host 测试覆盖；昇腾单卡 E2E 验证 15 轮均在同一 Coder session，并在硬上限后立即报告 |
| Ledger `3.0` 基线加扩展 | schema 以 jiabei `2.6` 为主，只叠加当前 strict lifecycle 和执行链路实际消费的 `candidate_path`、workload `phase/skip_reason`、`api_version`、anti-hack、`timing_skipped`、fingerprint 和 `next_verdict`；迁移器支持当前 `2.0` 和 jiabei `2.1..2.6`，歧义 `2.0` 要求显式来源且转换不覆盖旧文件 | migration、ledger、trajectory 和 Knowledge lineage host 测试覆盖 |
| KernelGen V1 Knowledge 隔离 | 仅显式 Knowledge-enabled 的 KernelGenWorkflow materialize state 并使用 `kernel-knowledge-coder`、`kernel-knowledge-profile-analyzer`、`kernel-knowledge-distiller`；SimpleOpt/BatchSimpleOpt 保持普通角色且 Batch 不再复制 `kb/` | role/doctor/Coder/Batch/Knowledge Distiller 测试覆盖 |
| K-08 与结构化 Distiller | Knowledge Coder/Profile 按 jiabei prompt/Skill 执行同问题 Concept+Source 双路检索和 `4/2/200` 指导；Python/MCP 不加 eval 门禁。Knowledge Distiller 只接收 workspace 实际 retrieval provenance，逐条校验/修复 Candidate 并由 epoch Publisher 写 Catalog | Knowledge MCP、runtime、Distiller、lineage 和 golden query 测试覆盖 |

KernelGen 侧的目标诊断、Debug/Status 和 Runtime 修复由 `8e363eb` 落地；随后
`9185f37` 对齐 Server v5.1，`e041dab` 增加设备可疑错误的终止语义，`3302727`
增加并行启动错峰，`ba87e8e` 将默认 plateau 调整为 3 个 round。当前发布点为
`e42d252`（`v5.1.1`）；`9bd2fe0` 完成 Profile lifecycle 对齐，`999287e`
完成 `finalize_round` 改名和严格 TargetContext。KernelGen 和 KernelGen Server
的 release tag 仍固定为 `v5.1.1`，Server API 版本仍为 `v5.1`。
`e8938e8` 的 FlagGems 抽取强化以及 `6220cd6`、
`6db7549`、`e42d252` 的多设备工具/文档更新不改变本文已经确认的 SimpleOpt
职责边界。

Server 侧同步 Debug 功能提交为 `354cc9b`，随后经测试兼容提交 `725e7e3` 和验证
文档提交 `622a0de` 合入 Gitee `main`；当前正式配套基线为 KernelGen Server
release tag `v5.1.1`，API 版本为 `v5.1`。

### 3.2 共同祖先后的细粒度修复复核

以下不是 commit 标题汇总，而是按最终行为归并的复核结果。它们会影响集成正确性，
即使多数不属于新功能：

| 复核面 | `jiabei-dev` 最终行为 | 集成判断 |
|---|---|---|
| Prompt 与语言隔离（`e4db79a`、`2c94c36`、`de7c978`） | Triton prompt 不再混入 Gluon 指令；Knowledge 查询规范集中到 Skill；role/Skill 改用自然 Markdown 换行 | 语言隔离和集中规范应保留；Skill 名称另行统一 |
| Agent 输出合同（`fe66a44`、`9b1020d`） | `Literal` 显示真实允许值，Pydantic 长度/数值/正则约束进入 prompt；多个 fenced JSON 候选逆序尝试 | 与当前更强进程清理和现有 JSON 解码组合，不整文件替换 |
| Distiller 容错（`8eee8d9`、`896c2be`） | 两份报告分别校验；每个无效 `CandidateDraft` 最多单独修复两次，仍失败只跳过该 Candidate；`claim_stance` 明确是相对 Concept claim，不是相对单轮实验结果 | 属于低耦合正确性修复；先移植报告和逐候选容错 |
| Ledger 与停止计数（`cdfaa29`、`7f4ec56`、`2150ecd`、`f971849`） | correctness failure 不消耗 performance plateau；knowledge use、evaluation lineage、solution SHA 逐步纳入；2.0 到 2.6 多次删除重复字段 | 停止计数与当前一致；schema 已分叉，迁移不能直接复制 |
| Provenance 与检索审计（`48a29f3`、`4695e01`、`09f1617`、`c242909`、`04cff40`） | 精确 Source revision/locator、统一 query event、usage feedback、measured source application、单 current Concept 均由 Python 维护 | 作为 KB 整体引入，禁止 Agent 自报权威 metadata |
| Profile、目标与归档（`1361fb3`、`eb38677`、`0d58f06`、`c7e4009`） | 查询前校验 draft profile finding；TargetContext 消费 Server target metadata；保留嵌套 archive manifest；归一化 Ascend device alias | 适配到 v5.1 `/status` 和当前 archive layout |
| Workspace 与 Runtime（`fa958a8`、`c1d7c1c`、`9e9d726`） | 清除继承的 workspace 变量后重新绑定；支持只读 `--add-dir`；自定义 provider 临时交换 settings 后恢复 | 前两项可独立采用；provider 配置仍保留当前 env 方案 |
| Debug 工具收敛（`c891c2a`、`fb97ceb`、`71f8d36`、`3df1983`） | 经 submit/get/cancel、wait-only 等中间形态后，最终只向 Coder 暴露一个返回终态的 `submit_debug_job` | 最终形态与当前一致；只需比较 adapter、环境变量和事故状态 |

这轮复核还确认了一个原则：不能按提交顺序把某个中间接口当作分支终态。例如 Debug
工具的三工具形态已经被后续提交撤销；Concept revision 的多版本存储也已收敛为
“Catalog 中只保留 current、历史交给 Git”。后文均描述 `2141e65` 的最终代码，
必要时才保留演进背景。

最新尾部提交中，`4441ffc` 主要删除无消费者的 `CandidateOutbox`、过期 review/
refactoring 报告，并把“保持外部行为”写成 KB 瘦身门槛；它不是新增运行能力。
`6016817`、`cfbf38b`、`f3b00b7` 主要更新设计说明和检索 roadmap，其中 BM25/RRF
仍是 TODO。相反，`de7c978` 虽然标题是 docs，却直接修改运行时加载的 Agent/Skill
prompt，仍属于需要回归验证的行为改动。

### 3.3 `f3b00b7..2141e65` 增量复核

| 提交 | 最终行为 | 与当前分支的关系和风险 |
|---|---|---|
| `0c67af8` | 每次编辑 candidate 后必须重新 Preflight；成功结果写 `.kernelgen/preflight/state.json`，eval 遇到缺失、失败或 identity 不匹配时返回 `PREFLIGHT_REQUIRED` 且不记 round | S-02 方向已对齐；但 jiabei state 不会在 eval 后消费，identity 也不绑定 v5.1 Server version/API/backend/timing 或 immutable candidate snapshot，仍保留当前一次性 receipt |
| `633ab35` | Python 只验证 ProfileAnalysis 路径在 workspace 内且文件是 JSON，再把路径交给有 `Read` 权限的 Distiller；Agent 必须逐个读取，不能继续追踪 raw profiler artifact | 减少主 prompt 体积并支持跨 round 对比；“确实读取全部文件”目前由 prompt 约束，移植时需决定是否增加工具审计 |
| `2141e65` | Coder/Profile Analyzer 对会影响非 baseline 实验的 KB 决策，使用同一问题同时查询 Concept 和 Source；Source 限 4 个 hit，最多读 2 个片段、每个不超过 200 行；只有 Coder 重新读取并实际采用后才能写 `knowledge_uses` | 属于 KB 专题，不改变无 V1 KB 的当前 BatchSimpleOpt；双路调用和 200 行限制目前只是 Skill/role 规则，Python 未强制，底层 `get_source` 仍允许单次最多 400 行 |

在 `origin/jiabei-dev` 的干净快照上运行这三个提交直接涉及的 preflight、Distiller、
doctor、Coder 和 MCP host 测试，共 42 项，全部通过。测试覆盖 prompt、schema 和
主要 adapter 行为；没有覆盖“Agent 是否真的读取每个 Profile 文件”或“双路检索
是否每次都发生”这两项模型行为。

## 4. SimpleOpt 与 BatchSimpleOpt

**相关性：SimpleOpt-直接**

### 当前分支

- [`SimpleOptWorkflow`](../../workflows/simple_opt.py) 接收 `definition_name`，从
  `kernelgen_server` Catalog 加载一个 Definition 和对应 workloads，然后调用
  [`OptimizeDefinitionWorkflow`](../../workflows/optimize_definition/workflow.py)。
- SimpleOpt 本身不做提取、Analyzer、同 Definition 多 Agent fan-out 或多 epoch。
- [`BatchSimpleOptDefinitionWorkflow`](../../workflows/batch_simple_opt_definition.py)
  负责并行运行多个相互独立的 SimpleOpt workspace；一个任务失败时仍尽量回收其他
  Definition 的结果。任务提交默认间隔 1 秒，可通过
  `launch_interval_seconds` 调整，避免远端网关连接突发。
- 最终 `OptimizeDefinitionOutput` 从 ledger 和 `.best_kernel.py` 组装，而不是
  信任 Coder 的叙述性输出。
- 评测链路使用独立的 `kernelgen_server` v5.1 Catalog、API 和 client。
- FlagGems 抽取仍是 SimpleOpt 上游的独立 Workflow；`e8938e8` 强化了 v5.1
  manifest、source-backed workload 和 reference-as-solution 自检，没有把抽取
  重新放回 SimpleOpt。

### `jiabei-dev`

- `origin/jiabei-dev:workflows/simple_opt.py` 接收 PyTorch operator，先运行
  `PyTorchV5ExtractorAgent`，把提取结果写入旧 trace layout，再并行启动多个
  `CoderAndDistillWorkflow`。
- SimpleOpt 同时承担“提取、Definition 展开、批处理、优化和 KB 发布”。
- 输出是一个包含多个 `PerDefinitionResult` 的 `SimpleOptOutput`。
- 评测和 Profile 适配层仍依赖旧的 `flashinfer_bench`/`eval_service` 接口。

### 初步结论

- 保留当前职责划分：

  ```text
  提取/导入 Definition
          ↓
  SimpleOpt：一个 Definition、一个 workspace
          ↓
  BatchSimpleOpt：多个独立 Definition
          ↓
  KernelGenWorkflow：同一 Definition 的多 Agent、多 epoch 搜索
  ```

- jiabei 的 Knowledge materializer、结构化 Distiller 和 MCP 已通过
  `KernelGenWorkflow` 的显式 Knowledge 配置及 workspace materializer 接入；
  共享 `OptimizeDefinitionWorkflow` 只在 `knowledge_enabled=true` 时选择
  Knowledge 角色和结构化 outbox。`workflows/simple_opt.py` 不暴露该开关。
- jiabei 的并行编排若复用当前 `run_parallel`，必须保留调用侧显式配置的错峰语义；
  通用并行框架本身仍以 0 秒为默认，BatchSimpleOpt 的 1 秒默认属于 Workflow
  策略。

### 已确认边界与后续讨论

1. 上述四层职责已经确认；后续不再把 extractor/batch/multi-epoch 逻辑放回
   SimpleOpt。
2. **2026-08-07 更新**：BatchSimpleOpt 仍不接入；SimpleOpt 通过可选的
   `knowledge_catalog_path` 显式启用查询，不采用 `KernelGenWorkflow` 隐藏的
   构造配置。
3. Knowledge Distiller 的合同失败按 jiabei 语义 best effort，不丢弃已完成优化；
   SimpleOpt 的 legacy Distiller 成功门禁仍按当前逻辑，S-09 后续单独决定。

## 5. Native Agent 与 MCP 基础设施

**相关性：SimpleOpt-共享**

### 已经相同的基础能力

两边都已经具备：

- `.claude/agents/*.md` Native Agent；
- `.mcp.json` stdio MCP 注册；
- `ClaudeRuntime` 对 Agent frontmatter、子 Agent 和 MCP 工具名的校验；
- workspace-bound ToolContext；
- Coder 与 ProfileAnalyzer 的权限分离；
- MCP schema/stdio/doctor/host tests。

因此，“Native Agent 基础设施”本身不需要从 jiabei 重做。

### 当前工具差异

| 工具组 | 当前分支 | `jiabei-dev` |
|---|---|---|
| 正式评测 | `preflight_kernel`、`eval_round`、`finalize_round` | jiabei 仍使用 `preflight_kernel`、`eval_round`、可选 `record_round` 和独立 `next` |
| Stateless 验证 | `eval_only` | 无 |
| Profile | `get_profile_context`、`profile_workloads`、`record_profile_analysis` | 同类工具 |
| Debug | 单个同步 `submit_debug_job` | 最终同样只暴露单个同步 `submit_debug_job` |
| KB | 无 | `query_knowledge`、`get_knowledge`、`query_sources`、`get_source`、`search_rounds` |
| STOP | `finalize_round` 原子返回 verdict；设备可疑错误强制 STOP；`next` 不注册为 MCP | 独立 `next` MCP |
| Server Status | `get_server_status`，包含 scheduler 健康状态 | **无直接 Status MCP** |

当前分支还包含 FlagGems extractor、PyTorch extractor 和 PR submitter 等
`jiabei-dev` 没有的 Agent，因此不能整目录覆盖 `.claude/agents`。

### Workspace 边界差异

- 当前 [`data/tool_context.py`](../../data/tool_context.py) 在缺少
  `KERNELGEN_WORKSPACE` 时允许退回 `CLAUDE_PROJECT_DIR` 或 cwd。
- `jiabei-dev` 要求显式 `KERNELGEN_WORKSPACE`，缺失时拒绝执行。

正常 SimpleOpt 会显式绑定 workspace，但严格模式能降低 MCP 被从错误 cwd
直接启动时串写 ledger 的风险。

### 初步结论

- 保留当前 Native Agent 和 `eval_only`。
- Debug 两边最终工具面都已收敛为单次终态返回；当前仍保留自己的 v5.1 adapter。
  Status 已作为无参数只读工具注册；KB MCP 后续讨论，不覆盖现有 MCP server。
- 建议采用 jiabei 的“必须显式 workspace”规则，但需要先确认现有 host 工具和
  手工 MCP 调试是否依赖 cwd fallback。

## 6. Round lifecycle、Profile 与停止策略

**相关性：SimpleOpt-直接；当前最大设计分歧**

### 对比

| 阶段 | 当前分支 | `jiabei-dev` 最终状态 |
|---|---|---|
| Preflight | 每次正式 eval 前强制；一次性 receipt 绑定完整 Definition/workload、immutable candidate、v5.1 Server identity/backend/timing，eval 时复核并消费 | 每次 candidate edit 后强制；`state.json` 绑定 candidate SHA、Definition 名、workload UUID 和部分 ToolContext；缺失或 stale 时拒绝 eval，但成功 state 不会在 eval 后消费 |
| ExperimentPlan | 首轮 baseline、后续非 baseline；profile provenance 有跨轮校验 | 只做结构校验，规范主要由 Agent 遵循 |
| Profile 触发 | **已对齐**：只标记非 hack 的 new best；Coder 可暂不执行，收尾时 best effort 补最终 best | 只标记新 best；Coder 可暂不执行，收尾时 best effort 补最终 best |
| Profile 门禁 | **已对齐**：pending profile 不阻止 conclusion、candidate transition、STOP 或下一轮 | pending profile 不阻止 conclusion、`next` 或下一轮 |
| Conclusion lifecycle | 每个 measured round 都必须调用 `finalize_round`；缺失时禁止下一次 eval，Workflow 也拒绝未完成 round | `record_round` 一旦调用必须提交合法 `RoundConclusion`，但该调用本身可选；缺失只作为 recommendation，不阻止 `next` 或下一次 eval |
| Candidate 转换 | `finalize_round` 由 Python 原子执行 KEEP/REVERT/REPAIR | Coder 根据结果手工恢复 `.best_kernel.py` |
| STOP | `finalize_round` 原子保存 `next_verdict` 并返回；设备可疑错误强制 STOP；STOP 后工具拒绝继续 | 独立 `next` 动态计算，不写每轮 verdict |
| Coder 提前退出 | Workflow 最多执行 `max_coder_sessions` 次 Coder 调用；必须通过 session ID 续接同一个 Claude 对话，直到持久化 STOP；缺少 session ID 或 Runtime 不支持 resume 时直接失败 | 一次 Coder invoke 后进入收尾，没有同等 terminal STOP 检查 |
| 平台期 | 只统计 `PASSED` 且未刷新 best 的轮次；SimpleOpt 默认阈值 3 | 同一计数规则；SimpleOpt 默认阈值 2 |
| 正确性失败 | 记录 measured round，但不计入 plateau | 相同 |
| 硬上限 | 默认 `max_round=15`，所有 measured round 都计数 | StopConfig 没有硬 `max_round` |
| 设备可疑错误 | preflight 不记 round；eval 记审计 round，完成 conclusion 后强制 STOP | 无 `SUSPECTED_DEVICE_ERROR` 对等状态 |
| Anti-hack | `is_hack` round 不能成为 best，保留诊断证据 | 没有当前分支这套字段和 best gate |

当前状态机见 [`round_lifecycle.md`](../design/workflows/round_lifecycle.md)。

jiabei 的 `SimpleOptInput.profile_enabled` 字段描述仍写着 mandatory profile
subagent，但最终 Coder prompt、MCP 提示、`next` 和
`CoderAndDistillWorkflow._ensure_best_profile()` 已实现上表的“new best 按需、
最终 best 收尾补齐”语义。后续移植时应同步修正这个残留字段描述，避免再次向 Agent
注入相反合同。

### 真实 campaign 对 Conclusion 的复核

这里需要区分两种“必填”：

- 当前分支调用 `finalize_round`、jiabei 调用 `record_round` 时，参数都必须是
  通过 schema 校验的 `RoundConclusion`；例如 `round_num`、
  `expectation_status`、`root_cause` 和 `next_suggestion` 等必填字段不能缺失。
- 是否要求**每个 measured round 在进入下一轮前都完成该调用**，
  两个分支并不一致：当前分支强制，jiabei 不强制。S-04 讨论的是后一层 lifecycle
  门禁。

对昇腾宿主机
`/data/jiabei/kernelgen-campaigns/campaign-02-test` 的 schema `2.6` ledger
只读统计显示：当前 48 个非归档 ledger 共 521 个 measured round，其中 371 个
带 Conclusion、150 个不带；21 个 ledger 存在缺失 Conclusion 的 round，141 个
缺失 round 后面仍成功产生了后续 round。另有 4 个 `profile.status=completed`
的 round 没有 Conclusion，说明 Profile 完成也不会使 Conclusion 自动变成
lifecycle 门禁。归档的 2 个中断 ledger 另有 54 个 round，其中 49 个不带
Conclusion，不计入上述主统计。

代表性实例
`abl_t1_gelu/1R/agent1/.ledger.json` 的 round 1、2 缺少 Conclusion，但 ledger
继续产生了 round 3 至 11；该 workspace 的实际 Coder prompt 也明确写明：
`record_round` 只在 measured round 产生有用结论时调用，缺失不阻止 `next` 或
下一次 evaluation。因此 jiabei 的 Conclusion 在真实运行中也是稀疏证据，不是
每轮强制事实。

### 对 SimpleOpt 的影响

- 两边现在都无法跳过 Preflight。当前一次性 receipt 还会绑定并复核 v5.1
  Server/Catalog/immutable candidate 身份；jiabei 的可复用 state 只保证 candidate
  和部分请求上下文未变，不能替代当前 receipt。已确认保留当前消费语义：一次成功
  Preflight 只授权一次 eval；下一次 eval 即使 candidate 未修改，也必须重新
  Preflight。
- 已采用 jiabei 的可选 Profile；当前分支仍要求每个 measured round 调用
  `finalize_round` 持久化 Conclusion，并继续由 Python 原子完成 candidate
  transition 和 STOP。因此只放松 Profile，不把 Conclusion、其他顺序与恢复责任
  交给模型。
- jiabei 只让正确的非 best 轮次消耗 plateau，这一规则本身合理；但没有硬 round
  上限时，连续编译或数值失败可能长期不触发 STOP。
- 当前 `max_coder_sessions` 能处理 Coder 提前返回；Claude Runtime 必须续接原
  session，若 Runtime 不支持 resume 或未取得 session ID 则直接失败，不用 ledger
  摘要启动新上下文。jiabei 的收尾逻辑更关注最终 best Profile 和 Distiller，
  不保证 Coder 已获得 terminal STOP。
- 当前默认 plateau 阈值从 2 调整为 3，降低偶然两轮未提升导致过早停止的概率；
  这只是默认参数变化，没有改变 Python-owned STOP 或硬上限。
- `SUSPECTED_DEVICE_ERROR` 是基础设施、共享运行时或设备事故证据，不是 candidate
  评测证据。Agent 必须先完成该 round 的 `finalize_round`，再停止修改和重试。

### Ledger 不只是版本号不同

- `3.0` 实施前，当前分支 schema 是 `2.0`，不自动迁移旧 workspace。
- jiabei schema 是 `2.6`，提供从其历史 `2.0` 到 `2.6` 的迁移。
- jiabei 的迁移不是单纯加字段：
  - `2.0 -> 2.1` 增加 language/knowledge uses；
  - `2.1 -> 2.3` 调整 query/application 和 evaluation lineage；
  - `2.3 -> 2.4` 删除 evaluation fingerprint；
  - `2.4 -> 2.6` 删除 plan 风险/工作量估算和 conclusion 中未被消费的字段。
- 两边的“2.0”已经包含不同字段。当前 solution/evaluation/round 还有
  `candidate_path`、anti-hack 和 `next_verdict` 等字段；jiabei 的迁移器并不等价于
  “可迁移当前 dev-xy 的 2.0”。
- jiabei 最终会在加载后重新计算 plateau，且只让 `PASSED` 的 non-best round
  消耗计数；evaluation lineage 已从 target fingerprint 收敛到结构化 scope 和
  solution SHA。

因此后续不能只把版本号改为 `2.6`。需要定义显式的跨分支 schema 识别和迁移，
并验证旧 workspace 的 resume、best kernel、knowledge usage 和 STOP 语义。
当前 v5.1 ledger 使用 `api_version`，jiabei evaluation 仍沿用旧 Server
`schema_version`，合并 schema 时还必须先完成 adapter 字段映射。

### S-07 已实施的合并边界

当前已新建共同的 ledger schema `3.0`，没有把当前 `2.0` 伪装成 jiabei `2.6`
的前驱。`3.0` 以 jiabei `2.6` 为主：

- 只叠加当前分支仍有运行消费者的执行事实和门禁：`candidate_path`、workload
  `phase/skip_reason`、`api_version`、anti-hack、`timing_skipped`、
  evaluation fingerprint 和 `next_verdict`；
- 引入 jiabei 为知识提取新增的事实：`evaluated_at`、
  `experiment_parent_round_num`、明确命名的
  `performance_baseline_round_num` 和 `knowledge_uses`；
- PlanSource 使用 `origin/parent_round_num`；删除
  `target_workloads`、`risks`、estimated range、`analysis_path/detail`；
  RoundConclusion 删除 `architecture_tag`、`suggestion_followed`、
  `key_numbers`、composition 和 diagnostic 占位字段。旧输入和旧 ledger
  可以读取，但这些字段在迁移时丢弃，不再写入 `3.0`。

迁移器是 side-effect-free 的纯转换：读取旧 ledger 后生成 `3.0` 内存对象，
不原地覆盖旧文件。jiabei `2.1..2.6` 可以直接识别；同名 `2.0` 先用
`next_verdict/is_hack/candidate_path` 等当前侧特征和 `knowledge_uses` 等 jiabei
侧特征判别。没有足够特征的歧义 `2.0` 必须要求显式指定来源，不能猜测。迁移测试
覆盖两侧 fixture、best/STOP 恢复和迁移前后测量事实不丢失；真实 campaign 的
批量迁移仍需在昇腾实例目录做最终验收。

本轮细节复核发现 jiabei HEAD 的 Coder prompt 没有跟随 schema `2.6` 更新：
`eval_round` 示例仍提交已删除的 `target_workloads`、`risks`、变化百分比和旧
`source.kind/round_num/analysis_path`；`record_round` 说明仍要求已经从
`RoundConclusion` 删除的 `architecture_tag`。当前普通/Knowledge Coder prompt
已经改为 `origin/parent_round_num` 和精简 Conclusion；模型层继续只读兼容旧
占位字段，并增加 schema 回归测试。

### 需要和 jiabei 对齐的分歧

以下条目不是重新决定 SimpleOpt；SimpleOpt 的答案已经是“保留当前分支”。需要确认
的是 KernelGenWorkflow 是否采用不同策略，以及共享的 MCP、ledger 和 KB adapter
怎样避免把另一套策略带入 SimpleOpt。

1. **已确认**：Preflight 在所有正式 eval Workflow 中始终开启，是每次正式 eval
   的强制资格门禁，不提供切换到可选诊断模式的开关；SimpleOpt、BatchSimpleOpt
   和 KernelGenWorkflow 使用同一规则。Receipt 为一次性授权，eval 后立即消费；
   再次 eval 无论 candidate 是否变化都必须重新 Preflight。
2. **已确认并已实现**：`profile_enabled` 作为总开关；只对非 hack 的 new best
   发出 Profile 请求，由 Agent 判断是否有诊断价值；pending Profile 不阻塞后续
   lifecycle，Workflow 收尾时对最终 best 做一次 best-effort Profile。
3. **已澄清**：当前的 `finalize_round` 和 jiabei 的 `record_round` 都要求合法
   Conclusion payload；但只有当前 SimpleOpt 强制每个 measured round 完成
   finalization，jiabei 允许缺失。当前 SimpleOpt 保持每轮强制。
4. **已确认**：KEEP/REVERT/REPAIR 继续由 Python 在 `finalize_round` 原子执行。
5. **已确认**：不恢复独立 `next` MCP；verdict 持久化在 measured round。
6. **已确认**：保留当前 `max_round`、anti-hack 和 `max_coder_sessions`。
7. **已确认**：SimpleOpt 每轮都有 Conclusion；Profile evidence 可以按需产生。
8. **已确认并已实现**：Ledger `3.0` 使用 jiabei `2.6` 基线加当前运行字段，
   并提供双来源 side-effect-free migration；歧义 `2.0` 不猜测来源。

### SimpleOpt 已确认的处理

所有正式 eval Workflow 都保持 preflight 始终开启的当前强门禁。SimpleOpt 另外
保持 Python-owned candidate transition、持久化 STOP、anti-hack、硬
`max_round`、默认 3-round plateau、`max_coder_sessions` 和
`SUSPECTED_DEVICE_ERROR` 强制停止，BatchSimpleOpt 继承这些规则。

Profile 例外已经落地：`profile_required = profile_enabled and is_new_best`，
其中 anti-hack round 不能成为 new best；Profile 不再阻塞 `finalize_round`、
preflight、STOP 或下一次 eval；Coder prompt 改为按诊断价值执行；
`OptimizeDefinitionWorkflow` 在 Distill 前对最终 best 做一次 best-effort
Profile。Profiler 失败会尽量记录 terminal `failed` evidence，但不使已有有效
优化结果失败。Preflight、conclusion、candidate transition 和 STOP 门禁未放松。

## 7. Implementation Profile 与目标设备身份

**相关性：SimpleOpt-直接**

### 已经一致的部分

两边的 [`data/implementation.py`](../../data/implementation.py) 基本相同：

- 当前端到端只注册 `triton`；
- 使用 `@triton.jit`、`main.py::run` 和 `kernel[grid](...)`；
- host wrapper 只能做分配、元数据读取、grid 计算和 launch；
- 禁止用 `torch.*` 或 Tensor compute 作为 kernel fallback；
- Implementation Profile 会注入 Analyzer、Coder、Distiller、ToolContext 和
  ledger。

Implementation Profile 不需要从 jiabei 重写。以后增加 AscendC/CUDA C++ 等语言时，
应该新增完整 Profile 和 Server capability，而不是只扩 Enum。

### 目标硬件差异

- 当前 `SimpleOptInput`、SimpleOpt CLI 和 BatchSimpleOpt CLI 默认
  `Ascend910B`。
- jiabei 的 SimpleOpt 默认目标为空，但 KernelGen example/campaign 仍默认
  `Ascend910B`。
- jiabei 的 KB `TargetContext` 优先读取 eval `/status.target` 和
  `/status.software`，orchestrator 字符串只作降级上下文。
- 当前独立 Server 已经返回 `target` 和 `software`；当前
  `kernelgen_server_adapter` 会校验 orchestrator 期望 backend 与 Server 实际
  backend，并通过 `get_server_status` 把完整状态交给 Coder。

`Ascend910B` 目前是可覆盖的默认值，不是所有运行都被强制改成 910B。已有多机脚本
显式传入目标硬件时不会使用该默认值；风险发生在非昇腾运行遗漏该参数时。

### 已实现的处理

S-08 以 jiabei 的 `TargetContext` 构造流程为主，并适配当前 Server v5.1
`/status.target` 和 `/status.software`：

- backend、vendor、architecture、device、capabilities 和 software 以 Server
  实际返回为准，不增加 expected/actual 双层 ledger schema；
- `/status.target.device` 是正式 eval 和知识提取的必需事实；缺失时直接报错，
  禁止使用输入的 `target_hardware` fallback；
- 输入的 `target_hardware` 继续作为一致性声明传入运行流程；当它非空时，Python
  将它与 Server device 归一化后比较，不一致时在正式 eval 前返回
  `TARGET_HARDWARE_MISMATCH`；校验通过后，Coder、ledger 和 Distiller 统一使用
  归一化后的 Server 实际 device；
- `910B`、`Ascend 910B` 和 `Ascend910B` 等已知 alias 可以归一为同一设备，
  但不同设备不能靠 fallback 或覆盖继续运行。

因此知识 scope 使用经过上述门禁的 Server `TargetContext`；输入
`target_hardware` 是一致性声明，不是实际设备信息的替代来源。生产入口是否继续
提供 `Ascend910B` 默认值可在实现时单独清理，不影响“Server device 必填且不得
fallback”的合同。

当前实现已在启动 Coder 前读取 `/status`，把归一化后的实际设备和完整
`TargetContext` 写入 workspace ToolContext，并让 Coder、ledger、Distiller 使用
该实际设备。Preflight receipt 绑定归一化 device；eval 和 Profile 再次访问 Server
时重复执行同一校验，避免运行中切换到另一设备。缺少 backend 也以
`TARGET_BACKEND_MISSING` 拒绝运行。

## 8. Distiller 与标准化单次运行报告

**相关性：SimpleOpt-直接；结构化 Concept 发布属于后续专题**

### 当前分支

- Distiller 输出自由格式 `candidate_experience` 和 `candidate_detailed`。
- Workflow 直接写 `.new_experience.md` 和 `.new_detailed.md`。
- 没有固定章节、`R<n>` 引用校验或 Python-owned run metadata。
- **S-09 已按 jiabei 修改**：Distiller 连续修复后仍产生
  `AgentContractError` 时，保留 ledger、best kernel 和权威优化结果；网络、文件
  系统或程序逻辑等其他异常仍然上抛。

### `jiabei-dev`

- 两份 Markdown 使用固定章节：
  - Experience：Lessons、Remaining Bottlenecks、Next Experiments；
  - Detailed：Strategy Evolution、Cross-Round Analysis、Failure Analysis、
    Open Questions。
- 报告必须引用真实存在的 ledger round；Python 注入 Definition、hardware、
  language、round count、best 和最终状态。
- Python 校验 ledger 中的 ProfileAnalysis 路径没有逃逸 workspace、文件存在且是
  JSON，只把 `{round_num, path}` 列表放入 prompt；Distiller 获得 `Read` 权限并
  逐个按需读取这些结构化文件。原始 profiler artifact 不进入 prompt，也不允许
  Distiller 沿 analysis 内的 artifact path 继续读取。
- Distiller 的其他知识工具限于读取 `available_provenance` 中已经检索到的精确
  Concept/Source，不能做开放式查询。
- Distiller contract 错误按 best effort 处理，不丢弃已完成优化；其他运行时异常
  不在该降级范围内。
- 两份 Markdown report 属于同一个 `DistillOutput` 合同；任一报告格式错误都会
  触发整个 Distiller 响应的 repair，最多尝试两次，并非分别保留或分别修复。
- 另外产生结构化 `CandidateDraft`，供 V1 KB Publisher 使用。每个无效 Candidate
  独立送入聚焦 repair prompt，最多重试两次；仍无效只跳过这一条，不再丢弃整个
  Distiller 响应中的其他 Candidate。
- `ObservationIntent.claim_stance` 明确表示证据相对“候选 Concept claim”的支持或
  反对，不表示相对单轮实验假设的成败；读取旧 `stance` 字段时保留兼容别名。

### 当前实施

以下能力已接入 Knowledge-enabled KernelGenWorkflow：

- 固定 Markdown 章节；
- `R<n>` 引用校验；
- Python-owned metadata；
- 只传经 Python 检查的 ProfileAnalysis 路径，由 Distiller 读取结构化 analysis，
  不读取 raw profiler artifact；
- 普通和 Knowledge Distiller 的合同失败都不改变权威优化状态，其他异常继续
  上抛；
- 两份报告按整包合同 repair；Candidate 逐条修复/跳过。

`CandidateDraft`、`claim_stance`、Source/Concept provenance 和 Publisher 已随
V1 KB 一并引入。SimpleOpt/BatchSimpleOpt 继续使用原有 report-only Distiller，
不选择 Knowledge Distiller、不生成 Candidate outbox，也不发布公共 V1 Catalog。
**S-15 已确认完全沿用 jiabei 方案**：报告整包最多 repair 两次；报告合法后，
每条无效 Candidate 再独立 repair 最多两次，仍无效只跳过该条，不增加其他局部
保留或失败门禁。真实 `campaign-02-test` 的 48 个活动 Agent workspace 最终都
生成两份报告；6 个 workspace 触发过报告整包 repair，115 条 Candidate 修复成功，
24 条修复失败后被单独跳过。

**S-16 已确认完全沿用 jiabei 方案**：按路径加载，依靠 role 要求读取全部文件；
不增加 Python/tool 读取审计、不把 analysis 正文注入 prompt，也不增加新的失败
门禁。只有后续发现会造成证据错配、越权读取 raw artifact 或流程失败的重大 bug，
才重新讨论额外约束。

真实 `campaign-02-test` 的 48 个活动 Agent workspace 共引用 120 个
ProfileAnalysis；文件全部存在、为合法 JSON、未逃逸 workspace，且没有 round
错配或孤立文件。该 campaign materialize 的仍是旧 Distiller，54 次运行均为
`tools=0`，所以只能证明历史输入文件完整，不能作为新版 S-16“实际执行 Read”的
验收依据。

### 需要讨论

1. 两份 Markdown 是否长期保留；jiabei TODO 已提出最终收敛为一个结构化 run
   summary。
2. **已确认**：SimpleOpt/BatchSimpleOpt 不发布 V1 KB；公共 Catalog 只由显式
   Knowledge-enabled KernelGenWorkflow 的 Python Publisher 写入。

## 9. Debug Job、Server Status 与设备事故

**相关性：SimpleOpt-直接**

### 当前分支

- Debug 方面只注册一个 `submit_debug_job`，并默认授权给 Coder。它在一次 MCP
  调用内完成提交、等待终态、日志返回和 artifact 下载。
- workspace adapter 已按当前 `kernelgen_server.client` 重写，保留 jiabei 的
  文件边界、快照、ownership、输出截断和 artifact 校验设计。
- 当前 Server 的 `POST /debug/jobs` 与 evaluate 一样直接等待终态；MCP 主路径
  不再调用 GET 轮询。
- 后续所有 KernelGen Server 部署默认开启 Debug Jobs；adapter 仍读取 capability
  作为旧部署和配置错误的防御性校验，但 Coder 正常流程不设计 disabled 分支。
- KernelGen MCP 注册无参数、只读的 `get_server_status`。Coder role 要求每个
  session 开始、读取或修改候选之前调用一次。
- Status MCP 返回 Server 的原始权威 payload，包括 hardware、software、devices、
  timing、profile、debug capability 和 scheduler 健康状态。

### 当前实现方法

KernelGen 侧调用链为：

```text
Coder
  -> mcp__kernelgen__get_server_status()
  -> workspace ToolContext 中固定的 eval_server_url
  -> kernelgen_server.client.status()
  -> GET /status
  -> 返回原始权威环境信息

Coder
  -> mcp__kernelgen__submit_debug_job
  -> tools/debug_job.py::submit_workspace_debug_job()
  -> kernelgen_server.client.submit_debug_job()
  -> POST /debug/jobs
  -> 等待 Server 终态
  -> 下载并校验 artifact
  -> 返回终态、日志和本地审计路径
```

具体约束和落盘行为：

- Agent 只提交 `purpose`、argv 形式的 `command`、`tmp/` 下的 UTF-8 文件列表、
  非保留环境变量和 `timeout_seconds`；Server URL、Definition、目标硬件和
  transport timeout 均来自 ToolContext。
- 最多上传 64 个文件，总计不超过 2 MiB；拒绝绝对路径、`..`、重复路径、保留的
  设备可见性变量和 Debug 环境变量。
- 调用前读取 `/status`，检查 `debug.enabled` 和预期 backend；这不是向 Agent
  发起的 Status MCP 调用。
- `timeout_seconds` 范围为 1 到 1800 秒；HTTP budget 取
  `max(eval_transport_timeout_seconds, timeout_seconds + 10)`。
- 请求、源码快照、终态、事件和完整 stdout/stderr 分别保存到
  `.kernelgen/debug-jobs/<job_id>/`；MCP 响应中的 stdout/stderr 各最多返回末尾
  32 KiB，完整日志仍保留在文件中。
- Debug 脚本写入 `$KGS_DEBUG_ARTIFACTS` 的文件由客户端下载到
  `.kernelgen/debug-jobs/<job_id>/artifacts/`，并保留 Server 返回的大小和
  SHA-256 元数据。
- `tools/debug_job.py` 仍保留 workspace-bound get/cancel adapter，便于兼容和
  host 测试，但它们没有注册为 MCP，Coder 主路径不会轮询。

### `jiabei-dev`

最新代码经过多轮接口收敛后，给 Coder **只注册一个**
`submit_debug_job`。该调用直接返回
`SUCCEEDED/FAILED/TIMEOUT/CANCELLED` 终态；`get_debug_job`、
`wait_debug_job` 和 `cancel_debug_job` 均不再暴露为 MCP。其 workspace adapter
仍提供：

- 只能上传 `tmp/` 下的 UTF-8 文本；
- 文件数量和总大小限制；
- 源文件 SHA-256 和精确快照；
- job ownership；
- stdout/stderr 截断返回、完整落盘；
- artifact 校验和本地审计。

这意味着两边最终 Debug 工具形态已经一致，不应再按早期的三工具/long-polling
提交评估差异。但是 `jiabei-dev` **没有**直接注册查看 `/status` 的 MCP，Status
只被 Debug、Profile 和 KB bridge 内部消费；它也没有 v5.1 scheduler incident
处理。

### 已完成的当前 Server API 适配

`origin/jiabei-dev:tools/debug_job.py` 最终虽已改成单次终态调用，仍调用旧
`eval_service.debug_client`，并使用旧 `FIB_DEBUG_ARTIFACTS` 和
`trace_set_key` 合同。当前独立 Server 的
[`kernelgen_server.client`](../../../kernelgen_server/kernelgen_server/client.py)
已经改为：

- `status(server_url)`；
- `submit_debug_job(DebugJobRequest, server_url)`；
- `submit_debug_job` 直接返回终态，而不是先返回 `QUEUED`；
- 返回 `DebugJob` Pydantic model；
- artifact 下载返回本地路径映射；
- debug artifact 环境变量使用 `KGS_DEBUG_ARTIFACTS`；
- 不再把 `trace_set_key` 作为 Debug Job 参数。

### 已确认结论

1. 已保留 jiabei 的 workspace 约束、审计、ownership、输出截断和 artifact
   校验设计。两边终态均已移除 Agent 主路径的异步轮询。
2. 已以当前 `kernelgen_server.client` 重写适配层和测试。
3. Debug 工具方面只向 Coder 暴露同步 `submit_debug_job`；当前底层 get/cancel
   client 能力保留，但不注册为 MCP。后续 Server 部署统一默认开启 Debug Jobs。
4. Debug 结果不创建 round，不刷新 best，也不能代替正式 preflight/eval。
5. 当前新增 `get_server_status`，工具无参数，Server
   URL 只能来自 ToolContext，正常返回原始 `/status`，失败返回
   `status=ERROR/reason=SERVER_STATUS_FAILED`。Coder 每个 session 开始时只调用
   一次；遇到 `SUSPECTED_DEVICE_ERROR` 后再调用一次保存 incident snapshot。
   Knowledge bridge 已复用同一严格 Server 身份：backend/device 缺失或输入不一致
   直接失败，不使用 orchestrator fallback。

### v5.1 设备事故语义

- Server 在请求超时或隔离 worker 异常退出后先把 slot 置为 `checking`，独立探针
  通过后恢复，只有探针失败才计入 `scheduler.broken`。
- Server 排除首个失败 slot，在另一健康 slot 上原样重试一次；两次都超时或无响应
  时返回 `SUSPECTED_DEVICE_ERROR`。它不等于“卡已损坏”。
- preflight 收到该状态时不创建 round，也不能以修改 candidate 的方式处理；正式
  eval 收到该状态时保留一个审计 round，但不会成为 best 或触发 profile，
  `finalize_round` 随后持久化 STOP。
- 只有 `/status.scheduler.broken > 0` 能确认探针失败的不可用 slot。
  `broken == 0` 时应记录为未确认的请求、共享运行时或设备异常，并由操作方在本轮
  外降低并发或增加 timeout 后重跑。

因此，已引入的 ledger migration、Run Archive、KB observation 和报告生成必须
继续保留该状态及 incident snapshot，不能把它折叠为普通
`RUNTIME_ERROR/TIMEOUT`，也不能写成 candidate 失败。

当前还有一个同分支内的提示顺序不一致需要修正：最新 [`AGENTS.md`](../../AGENTS.md)
要求 eval 返回该状态后先完成 `finalize_round`，再读取 `/status.scheduler`；
[Coder role](../../.claude/agents/kernel-coder.md) 当前仍提示先调用
`get_server_status`、再
`finalize_round`。权威操作约束以前者为准，后续应同步 Coder role 和对应 host test，
固定为：

```text
eval_round(SUSPECTED_DEVICE_ERROR)
        -> finalize_round(not_evaluable)
        -> get_server_status incident snapshot
        -> STOP
```

### 已完成验证

- Ascend910B 物理卡 3 上运行了真实 NPU Tensor 脚本：
  `torch.ones(1, device=KGS_DEVICE)` 返回 `1.0`。
- 同步提交返回 `SUCCEEDED`、`device=npu:0`、`exit_code=0`；artifact
  `device.json` 记录 `physical_visibility=3`，并成功下载到本地 workspace。
- `sleep(5)` 配合 `timeout_seconds=1` 直接返回终态 `TIMEOUT`，没有客户端
  GET 轮询。
- 远端 `tests/live_debug_validation.py` 通过；同时完成 NPU 绑定、stdout、
  artifact 和 timeout 验证。

### 需要讨论

1. 同一 workspace 是否限制为一个 active Debug Job？
2. BatchSimpleOpt 是否需要 Debug Job 配额，避免多个 Agent 占满正式 eval/profile
   的共享设备 slot？
3. 是否在客户端强制 loopback/受控代理，还是只依赖部署约束？
4. 结构化 KB/Run Archive 如何记录 `SUSPECTED_DEVICE_ERROR` 和 scheduler
   snapshot，而不把未确认事故写成 candidate 或硬件结论？
5. 将 Coder role 的 incident snapshot 调用顺序改为 `finalize_round` 之后，并补充
   顺序约束测试。

Debug Job 是可信环境中的任意命令执行接口。无论最终工具策略如何，Server 必须保持
loopback 或等效网络隔离；正式计时期间不应并行运行 Debug Job。

## 10. Runtime、日志和 Agent 合同

**相关性：SimpleOpt-共享**

### 当前已经具备

- `mirror_to_console`；
- workspace 内的人类可读 `claude-runtime.log`，不额外保存 provider 原始 stream；
- streamed/aggregate thinking 去重；
- session resume 和 stderr drain；
- 静默 stdout 下仍按时触发的 idle/hard timeout；
- 超时、中断或解析异常后的 CLI/MCP 进程组清理。
- Server v5.1 `api_version` 协议身份、scheduler 状态和
  `SUSPECTED_DEVICE_ERROR` 工具合同。

因此 jiabei 的“日志和 console 管理”主体已经被当前分支吸收。

### 已适配和仍待讨论的修复

1. **静默 stdout 超时修复（已适配）**：当前 `_pump()` 已使用后台 reader 和
   queue，完全静默时主循环仍能检查 idle/hard timeout；stderr 活动会刷新 idle
   判断，但不会绕过 hard timeout。同时补充了 EOF 竞态和 CLI/MCP 进程组清理。
2. **父 Agent 环境清理（已适配）**：当前已按 jiabei 清除继承的
   `CLAUDECODE`、`KERNELGEN_WORKSPACE`、`CLAUDE_PROJECT_DIR`，再绑定当前
   workspace，避免嵌套 Agent 串用父 workspace。
3. **额外只读目录**：jiabei 支持 `--add-dir`，主要服务外部 KB Catalog 和 epoch
   目录。
4. **输出合同（已适配）**：当前已按 jiabei 把 `Literal` 的真实允许值以及
   Pydantic `pattern/min/max/gt/ge/lt/le` 约束渲染给 Agent，修复只展示 Python
   类型名、模型无法得知枚举和边界的问题。
5. **JSON 容错（已组合适配）**：当前按逆序尝试多个 fenced JSON 候选，再尝试
   整体响应，避免最后一个无效代码块遮蔽前面的正确输出；每个候选继续使用当前
   `JSONDecoder.raw_decode()` 判断真实 JSON 边界，保留字符串内嵌 code fence
   的能力，并覆盖多候选、嵌套 fence 和纯 JSON 测试。
6. **Runtime 配置方式**：当前通过子进程环境传递 API endpoint/token，并已在
   Runtime 边界把显式 token 或旧 `ANTHROPIC_AUTH_TOKEN` 统一映射为
   `ANTHROPIC_API_KEY`，确保子进程不会同时收到两种认证变量；jiabei 临时交换
   `.claude/settings.local.json`，结束后恢复，且仍写入旧认证变量。

### 初步结论

- 1、2、4、5 已独立引入并保留 host tests；其中 4、5 采用 jiabei 的合同/候选
  策略和当前 JSON 边界解析的组合实现。
- 外部 V1 Catalog 已通过 workspace state 和 MCP service 引用，不需要把整个
  Catalog 作为 `--add-dir` 暴露给 Agent；若未来出现非 MCP 的只读资料再单独评估。
- API endpoint/token 已确认保留当前“仅子进程 env、不写 workspace settings”的
  方式。token 只以 `ANTHROPIC_API_KEY` 传给 Claude 子进程，并删除旧
  `ANTHROPIC_AUTH_TOKEN`，避免 zyapi 选择错误认证方式。
- 不用 jiabei Runtime 整体替换当前实现：当前的 `start_new_session`、进程组终止、
  静默超时和 EOF 清理更完整，应只移植环境清理、合同渲染和 JSON 候选策略。

当前 SimpleOpt 和 BatchSimpleOpt 示例的四层默认时间预算为：

| 预算 | 配置默认值 | 实际行为 |
|---|---:|---|
| Coder hard timeout | 1800 秒 | 一个 Coder session 的绝对最长时间 |
| Coder idle timeout | 1500 秒 | stdout/stderr 都静默时终止 session |
| Server execution timeout | 600 秒 | 已取得设备 slot 后，每次隔离 preflight/eval 尝试的执行上限 |
| HTTP transport timeout | 1200 秒 | adapter 取 `max(配置值, 2 × execution + 90s)`，默认实际为 1290 秒，以覆盖两次 slot 尝试和健康检查 |

`eval_transport_timeout_seconds` 必须大于 `eval_timeout_seconds`。隔离执行达到
600 秒时，Server 会按 v5.1 slot 隔离策略处理；两次尝试及恢复检查仍未得到结果时
可返回 `SUSPECTED_DEVICE_ERROR`。只有排队、两次执行和恢复检查的总耗时超过实际
transport budget 时才表现为 transport error。Debug Job 不走双 slot retry floor，
其 transport budget 仍为
`max(eval_transport_timeout_seconds, timeout_seconds + 10)`。

## 11. KernelGenWorkflow、Multi-Epoch 与 Campaign

**相关性：后续专题**

### 当前分支

- Analyzer 一次冷启动；
- 同一 Definition 的多个 `OptimizeDefinitionWorkflow` 独立运行；
- epoch reducer 是 legacy canonical KB 的唯一写者；
- EpochSummary 产生下一 epoch 方向；
- 已适配当前 Catalog、Server 和严格 lifecycle。

### `jiabei-dev`

- 保留多 Agent/multi-epoch 主体；
- 增加 Knowledge bridge、epoch publish barrier、Run Archive；
- 支持只重做 KB publication/synthesis 的 `finalize_epoch`；
- 增加多 operator campaign runner、resume 和 campaign summary；
- Analyzer 不再由 Python 预加载一大段 knowledge context；它在 materialized
  `shared_analysis` workspace 中通过统一 Knowledge Skill/MCP 按具体问题检索；
- 新增 exact-scope best solution registry。`KernelGenInput.start_mode` 支持
  `fresh/fork/resume`：`fork` 必须命中 definition、benchmark、backend、
  architecture、device、language 完全一致且 checksum 正确的 solution；
- 完成运行后在 Catalog transaction lock 内归档 workspace，只在新
  `geo_mean` 严格更高时替换该 scope 的 current solution，并记录
  `parent_solution_ref`、run/workspace/round/archive 和 solution SHA；
- 后期 lifecycle、CoderAndDistill 和 ledger 已按 jiabei 的可选 gate 设计调整。

### 最新代码中发现的恢复路径缺陷

`workflows/kernel_gen/workflow.py::_load_completed_result()` 在没有任何 completed agent ledger
时构造 fallback 结果，但使用了该方法作用域中不存在的 `def_name`。正常 fresh
路径会在外层先建初始结果，不一定触发；resume/finalize 或空 archive 恢复路径可能
直接得到 `NameError`，掩盖原本应返回的“无完成结果”状态。当前适配版已把
definition name 作为显式参数传入并在 fallback 使用该参数，不再复制这个缺陷；
`finalize_epoch` recovery 已在 K-05 中接入并覆盖空/不完整 Ledger 检查。

### 已确认并完成的 K-06 适配

- 正式 v5.1 SimpleOpt、BatchSimpleOpt、KernelGen 和 MCP `eval_only` 统一使用
  `catalog_name`；默认值为 `flaggems-v5`，内部通过
  `kernelgen_server.builtin_catalog_path()` 定位。正式 Coder input、
  ToolContext、eval/profile/preflight/debug adapter 已删除旧 `trace_root`/
  `trace_set_key` 字段；本地路径概念只留给后续单独整理的 ExtractOpt 和历史转换；
- Solution benchmark identity 不再复用 `trace_set_key`，而是由 Catalog
  manifest 的 `name + api_version` 生成；当前值为 `flaggems-v5-v5.1`，不增加
  Definition/workload 内容哈希；
- Knowledge-enabled KernelGen 支持 `fresh/fork/resume`。`fork` 必须命中
  Definition、benchmark、backend、architecture、device、language 完全一致且
  checksum 正确的 Solution，并把其代码作为第一 epoch 所有 Agent 的共同 seed；
  `resume` 只读取当前 workspace checkpoint；
- 运行或 `finalize_epoch` 完成后，从权威 ledger 对应的真实获胜 workspace 归档并
  promotion；只有 `geo_mean` 严格更高才替换 exact-scope current Solution，保留
  parent solution、run/workspace/round/archive、solution SHA 和 incident 审计；
- exact-scope Registry 只属于显式 Knowledge-enabled KernelGen。SimpleOpt 和
  BatchSimpleOpt 不读取、不发布公共 Solution；
- CLI 路径参数已明确命名为 `--knowledge-catalog-path` 和
  `--knowledge-run-archive-path`，不保留旧别名。

## 12. 结构化 KB 与治理

**相关性：KernelGenWorkflow/KB；SimpleOpt 仅显式接入查询和本地候选**

### 当前分支实施状态

- 仓库已引入完整 V1 `knowledge/` runtime、内置 `kb/` corpus、统一
  `kernelgen-knowledge` Skill、Knowledge MCP、运维 CLI、schema 和 golden tests；
- ledger 已升级为 jiabei 基线加当前执行字段的 `3.0`，可以记录精确 Concept/Source
  `knowledge_uses`、evaluation timestamp、experiment parent 和 performance
  baseline；
- 仅显式 Knowledge-enabled 的 KernelGenWorkflow 创建
  `.kernelgen/knowledge/state.json`，并选择
  `kernel-knowledge-coder`、`kernel-knowledge-profile-analyzer` 和
  `kernel-knowledge-distiller`；
- Knowledge Distiller 从真实 ledger、已验证 ProfileAnalysis 路径和 workspace
  retrieval provenance 生成固定报告及 `CandidateDraft`，epoch Publisher 负责
  Observation/Candidate/Catalog 写入；
- legacy KernelGen reducer 与 V1 Publisher 互斥。默认 SimpleOpt 和
  BatchSimpleOpt 使用普通角色；显式 KB-enabled SimpleOpt 会复制 Knowledge
  Skill、产生 state/outbox，但不进入 Publisher；
- K-05 的 campaign/finalize recovery 和 K-06 的完整 fork/promotion 工作流均已
  接入；Solution benchmark identity 使用 `flaggems-v5-v5.1`。

### `jiabei-dev`

实现了一个完整 V1 子系统：

- `knowledge/` 35 个代码文件；
- `kb/` 约 5,500 个文件、约 60.6 MiB；
- 57 个 Concept；
- 5,412 个 vendored source 文件；
- Concept、Source、Candidate、Observation、TargetContext、scope、evidence 和
  retrieval contracts；
- SQLite 派生索引、scope-aware Concept query、Source search 和基于 FTS5/BM25
  的历史 round search；
- `query_knowledge/get_knowledge/query_sources/get_source/search_rounds` MCP；
- `.claude/skills/kernelgen-knowledge/SKILL.md` 作为各角色共用的稳定查询协议：
  Analyzer/Epoch Summary 采用 Concept first、必要时 Source fallback；Coder 和
  Profile Analyzer 在 KB-guided non-baseline 实验决策前对同一问题同时查询
  Concept/Source。所有角色都必须使用返回的 canonical ref/query ID/package/path，
  并检查 payload 内的 `status: ERROR`；
- 双路 Source 检索在 prompt 中限制为 `max_results=4`、最多读取 2 个局部片段、
  每个不超过 200 行；Profile Analyzer 只交回精确 ref，只有 Coder 重新读取并实际
  采用后才记录 `ExperimentPlan.knowledge_uses`；
- `ExperimentPlan.knowledge_uses`、retrieval log 和 usage feedback；
- Publisher、Run Archive、锁、幂等发布、审计、回滚、重建索引、revalidation 和
  metrics；
- exact-scope solution registry 与 `fresh/fork/resume` lineage；
- ledger `2.6` 及 jiabei 自己历史版本的迁移。

基础设施是多后端的，但当前 57 个种子 Concept 主要覆盖
Ascend/AscendC/Triton-Ascend。其他芯片需要明确标记为 coverage gap，不能把
Ascend 经验当成 direct match。

### 真实 campaign 对 K-08 的复核

`campaign-02-test` 能证明旧流程的实际行为，但不能验证 `2141e65` 新增的 K-08
prompt。48 个活动 Agent workspace 都是在该 prompt materialize 之前创建的；
其中没有一个同时包含 `Dual Retrieval for Experiment Decisions` 和
`dual retrieval is required`。因此下面的数据只能用来说明为什么需要可审计约束，
不能据此宣称新 prompt 已经生效或已经失败。

活动 campaign 的 48 个 schema `2.6` ledger 共 521 个 measured round。54 个
round 声明了 62 条 `knowledge_uses`，其中 55 条 Concept、7 条 Source；按 jiabei
自己的 retrieval log 规则重建 lineage，只有 1 条 `complete`，12 条
`detail_not_read`，49 条 `query_missing`。

旧 prompt 下，48 个 Agent workspace 中 40 个曾同时使用 Concept 和 Source 查询，
但没有 workspace 出现两类查询使用完全相同问题；124 次成功 Source query 中
118 次设置的 `max_results` 大于 4。Agent workspace 另有 131 次成功
`get_source`、21 次失败调用；按闭区间计算有 10 次读取超过 200 行，10 个 Source
query ID 关联了超过 2 次成功读取。这也与代码一致：底层
`query_sources` 默认 12、允许最大 50，`get_source` 单次硬限制仍是 400 行。

### K-08 已确认的执行边界

已按 jiabei 当前做法实现，不增加额外约束：

- 只有 Knowledge-enabled KernelGen 的 Coder/Profile Analyzer role 和
  Knowledge Skill 要求 KB-guided
  non-baseline 实验使用同一问题查询 Concept 和 Source，并在 prompt 中遵循
  `4 hits / 2 fragments / 200 lines`；
- 普通 `kernel-coder`/`kernel-profile-analyzer` 不包含 Knowledge Skill 或 KB
  工具，SimpleOpt/BatchSimpleOpt 因此不执行 K-08；
- Python/MCP 不把上述要求实现为 eval 门禁，也不额外收紧底层
  `query_sources/get_source` 的现有参数范围；
- `ExperimentPlan.knowledge_uses` 继续只记录 Agent 声明为实际采用或调整的证据，
  Python 根据 retrieval log 派生
  `complete/query_missing/detail_not_read` 供审计；
- lineage 不完整不阻止正式 eval，也不新增 publication、promotion 或效果权重
  门禁。

### 已知设计分歧

1. **KnowledgeMode 语义**：
   **已实施 K-01 收敛**：新运行只有“无 `KnowledgeConfig` 即禁用”和
   `read_write_v1` 两态；旧 workspace state mode 不再受支持。
2. **Skill 的名称和合同**：
   **已实施**统一 `kernelgen-knowledge` Skill-over-MCP；没有恢复旧
   `query-kb` 目录路由。Agent role 只声明何时使用 Skill，canonical
   Source/query 规则集中维护。
3. **Source corpus 位置**：
   **当前实施选择**是把完整离线 corpus 纳入仓库基线；正式可写运行仍建议复制到
   独立 Catalog/run archive，后续可再评估 artifact 拆分。
4. **公共写入策略**：
   **已确认** SimpleOpt/BatchSimpleOpt 不读写公共 V1 Catalog；只有显式配置的
   KernelGenWorkflow 可以进入 `read_write_v1`。
5. **Ledger 迁移**：
   **已实施** schema `3.0` 双来源迁移；保留测量和运行事实，丢弃 jiabei 已移除
   的旧占位字段；歧义 `2.0` 必须显式指定来源。
6. **目标归一化**：
   需要用当前 `/status` 在所有已支持芯片上补 golden query，不能只验证
   A100/Ascend。
7. **Legacy 双 Markdown 链路**：
   V1 run 的 legacy reducer 已禁用，但 legacy KernelGen 模式仍保留 Markdown
   reducer；是否最终删除仍属 `KB-A06`。
8. **Solution registry 的消费方**：
   SimpleOpt 已明确不消费公共 best；KernelGenWorkflow 的完整 fork/promotion、
   benchmark identity、正确性门禁和 recovery 仍属 K-06。
9. **Concept 检索演进**：
   当前 Concept route 主要依赖结构化 scope/exact token，Source 主要依赖 alias、
   substring/path；TODO `KB-R01` 计划为 Concept 增加 BM25 并用 RRF 融合，再保留
   scope/evidence/usage rerank。Embedding/graph 仍是后续方向，不应按已实现能力
   对外承诺。
10. **双路检索的门禁位置**：
    **已确认采用 jiabei 当前语义**：prompt/Skill 提供 `4/2/200` 指导，
    Python/MCP 不增加硬门禁，lineage 只作审计。

### 当前结论

V1 models/schema/scope/query、Skill/MCP/materializer、Python-owned
authority/evidence、usage lineage、ledger migration、Publisher/Run Archive、
corpus、golden queries、维护 CLI、campaign 和 exact-scope Solution 已按模块适配，
没有复制 jiabei 的 SimpleOpt/KernelGenWorkflow 文件。
SimpleOpt/BatchSimpleOpt 明确保持无 V1 KB。公共 Runtime/Agent 对比项均已确认，
后续工作集中在真实多芯片 E2E。

## 13. 与 jiabei 的待确认清单

| ID | 决策 | 当前分支 `dev-xy` | `jiabei-dev@2141e65` | 建议或确认状态 | 影响范围 |
|---|---|---|---|---|---|
| S-01 | SimpleOpt 是否固定为单 Definition | 一个已有 Catalog Definition 对应一个独立任务；提取、Batch、KernelGen 分层 | SimpleOpt 内含 PyTorch 提取、多个 Definition 展开和并行优化 | **已确认**：保留当前分层，不合并 jiabei 的 SimpleOpt | SimpleOpt |
| S-02 | 正式 eval 是否强制 preflight receipt | Preflight 始终开启；一次性 receipt 绑定并复核完整 candidate/Catalog/v5.1 Server identity，eval 使用 immutable snapshot 后立即消费 receipt | Preflight 也已强制；成功 `state.json` 绑定 candidate SHA 和部分请求 context，缺失/stale 时拒绝 eval，但 state 不会在 eval 后消费，也不绑定 v5.1 协议身份 | **已确认**：保留当前分支；一次 Preflight 只授权一次 eval，下一次 eval 即使 candidate 未变也必须重新 Preflight | SimpleOpt/BatchSimpleOpt/KernelGenWorkflow |
| S-03 | Profile 的触发和门禁 | 已实现：只对非 hack 的 new best 发出请求；Agent 按诊断价值决定是否执行，pending 不阻塞 lifecycle；收尾时对最终 best 做 best-effort Profile | 只对 new best 发出请求；Agent 按诊断价值决定是否执行，pending 不阻塞 lifecycle；收尾时对最终 best 做 best-effort Profile | **已完成，采用 jiabei 模式**：保留总开关，按需 Profile new best，最终 best 收尾补齐；失败只记录 evidence，不使优化任务失败 | SimpleOpt/BatchSimpleOpt/KernelGenWorkflow |
| S-04 | 每个 measured round 是否必须调用 finalization 工具持久化 Conclusion | 必须调用 `finalize_round`；缺失时禁止下一次 eval，Workflow 拒绝未完成 round；调用参数必须通过 `RoundConclusion` schema | 调用 `record_round` 时也必须提交合法 `RoundConclusion`，但调用本身可选；真实 campaign 的 521 个非归档 round 中有 150 个缺失 Conclusion，且 141 个随后仍产生后续 round | **SimpleOpt 已确认**：保持当前 lifecycle 门禁，每轮必须完成 finalization；当前工具重命名为 `finalize_round`，不采用 jiabei 的可选 `record_round` | SimpleOpt |
| S-05 | KEEP/REVERT/REPAIR 和 STOP 的所有者 | `finalize_round` 由 Python 原子修改 candidate、保存 `next_verdict` 并返回 STOP | Coder 手工恢复 candidate；独立 `next` 动态计算且不保存 per-round verdict | **SimpleOpt 已确认**：保持 Python-owned transition 和持久化 STOP | SimpleOpt |
| S-06 | 硬 `max_round`、anti-hack、Coder 恢复和 plateau | 默认 `max_round=15`、plateau=3；有 anti-hack、`max_coder_sessions` 和 terminal STOP 检查 | plateau=2；无对等硬 round 上限、anti-hack、Coder 恢复和 terminal STOP 检查 | **SimpleOpt 已确认**：保留当前全部约束 | SimpleOpt |
| S-07 | Ledger 下一版本和跨分支迁移 | **工作区已实现 schema `3.0`**：以 jiabei `2.6` 为主，只叠加 `candidate_path`、phased workload、v5.1/anti-hack/timing、fingerprint 和持久化 verdict | schema `2.6`，包含 evaluation/experiment/knowledge lineage，并已删除旧 Plan/Conclusion 占位字段 | **已完成**：采用 jiabei 字段基线和双来源 side-effect-free migration；歧义 `2.0` 不猜测 | SimpleOpt/KernelGenWorkflow/KB |
| S-08 | 目标身份的来源 | **已实现**：启动 Coder 前从 v5.1 `/status` 构造 TargetContext；device/backend 缺失直接失败；输入 target_hardware 非空时做归一化一致性校验；Preflight、eval 和 Profile 重复校验 | `TargetContext` 优先旧 Eval Service status，device 缺失时 fallback 到输入 `target_hardware`；不校验输入与实际 device 是否一致 | **已完成，采用 jiabei 主流程并收紧门禁**：Server TargetContext 是实际身份的唯一来源；禁止输入 fallback；不一致在正式 eval 前失败 | SimpleOpt/KB |
| S-09 | Distiller 是否在成功关键路径 | **已实现**：普通和 Knowledge Distiller 的 `AgentContractError` 不影响 ledger/best/优化结果；其他异常继续上抛 | 同样仅将 Distiller 合同错误按 best effort 处理 | **已确认完全沿用 jiabei**：报告不是权威优化成功门禁，但不吞运行时和程序错误 | SimpleOpt/KernelGenWorkflow |
| S-10 | Debug 和 Status MCP | 同步 `submit_debug_job`；另有只读 `get_server_status`，暴露 scheduler | 最终也只暴露同步 `submit_debug_job`；仍是旧 adapter，且无直接 Status MCP | **已确认**：保留当前 v5.1 adapter 和 Status；复用 jiabei 的文件/审计约束 | SimpleOpt |
| S-11 | Runtime endpoint/token 配置 | **已修复**：仅通过子进程环境传递，不写 workspace settings；显式 token 和旧 `ANTHROPIC_AUTH_TOKEN` 均统一为 `ANTHROPIC_API_KEY`，并从子进程删除旧变量 | 临时交换 `.claude/settings.local.json`，结束后恢复，且写入旧 `ANTHROPIC_AUTH_TOKEN` | **已确认**：保留当前 env 方案并修复 zyapi 双认证变量 403；不移植 settings 交换 | 公共 Runtime |
| S-12 | Server v5.1 设备事故状态 | 完整处理 `SUSPECTED_DEVICE_ERROR`、scheduler snapshot、审计 round 和强制 STOP | 没有对等状态和 scheduler incident 语义 | **已确认**：保留当前实现，事故不得归因于 candidate | SimpleOpt/KB |
| S-13 | 设备事故 snapshot 调用顺序 | **已修正**：普通与 Knowledge Coder 均先 `finalize_round` 持久化强制 STOP，再调用一次 `get_server_status` 获取 scheduler snapshot，随后停止 | 无直接 Status MCP，也没有该事故流程 | **已完成**：统一为 `finalize_round -> get_server_status -> STOP`，preflight 因不产生 measured round 而直接读取 snapshot | SimpleOpt/Agent |
| S-14 | Agent 输出合同与 JSON 容错 | **已完成组合适配**：完整渲染 `Literal`/Pydantic 边界；倒序尝试多个 fenced JSON，同时用 `JSONDecoder` 保留嵌套 fence；清理父 workspace 环境并保留当前静默超时、进程组终止和 EOF 清理 | 渲染 `Literal` 和 Pydantic 边界；逆序尝试多个 fenced JSON；清理父 workspace 环境，但进程生命周期弱于当前 | **已完成**：采用 jiabei 的合同、候选和环境隔离修复，保留当前 decoder 边界识别与进程生命周期 | 公共 Runtime |
| S-15 | Distiller 局部失败策略 | 报告整包最多 repair 两次；Knowledge Distiller 对每个无效 Candidate 再独立 repair 两次，失败只跳过该条；SimpleOpt 保留 legacy report-only Distiller | 同一整包报告 repair 和逐 Candidate 修复/跳过语义 | **已确认完全沿用 jiabei**：不增加额外局部保留或失败门禁；SimpleOpt 不引入结构化 Candidate | SimpleOpt/KernelGenWorkflow/KB |
| S-16 | Distiller 如何获取 ProfileAnalysis | Knowledge Distiller 由 Python 校验 analysis path/JSON，只传相对路径并要求 `Read`，禁止追踪 raw artifact；SimpleOpt legacy Distiller 不使用 KB | 同一相对路径模式 | **已确认完全沿用 jiabei**：不增加读取审计、正文注入或新门禁；仅重大 bug 例外；SimpleOpt 不引入 Knowledge Distiller | KernelGenWorkflow/KB |
| S-17 | 多 round Coder 是否保持同一模型上下文 | **已实现组合方案**：正常 round 本就在一次 Coder invoke 内；提前返回且 verdict 为 CONTINUE 时，通过 Runtime 捕获的 session ID 续接同一个 Claude 对话，同时保留 `finalize_round`、terminal STOP 和 `max_coder_sessions` 门禁；无法 resume 时直接失败，不启动全新 Coder context | 所有 round 依赖一次 Coder invoke 自然共享上下文；Coder 一旦提前返回，Workflow 直接进入收尾，没有恢复和 terminal STOP 校验 | **已完成并通过 E2E**：采用 jiabei 的上下文连续性目标，保留当前 Python-owned lifecycle；昇腾 v5.1.1 单卡 15 轮均在同一 Coder session，第 15 轮 STOP 后没有 round 16 | SimpleOpt/BatchSimpleOpt/公共 Runtime |
| K-01 | KnowledgeMode | **已实现收敛**：无 `KnowledgeConfig` 即禁用；有配置和 workspace state 时只能是 `read_write_v1` | 保留 `legacy/shadow/read_v1/read_write_v1`，且 `read_v1` 实际也写 V1 | **已确认偏离 jiabei 修复语义冲突**：只保留 `read_write_v1`，不兼容旧 mode | KB |
| K-02 | Knowledge Skill 名称和合同 | 已引入统一 `kernelgen-knowledge` Skill-over-MCP；只暴露给 Knowledge 角色 | 提供统一 Skill 和严格 Source/query ID 规则 | **已完成**：不恢复旧 `query-kb` 路由 | KernelGenWorkflow/KB/Agent |
| K-03 | vendored Source corpus 是否入主仓 | 已引入完整离线 `kb/` corpus，当前验证为 57 Concepts、5 Source packages、5,470 entries | 主仓包含约 5,500 个 KB 文件 | **已完成当前选择**：仓库保留离线基线；正式可写 Catalog 独立放置 | KB |
| K-04 | SimpleOpt 是否写公共 KB | 默认 SimpleOpt/BatchSimpleOpt 不接入；显式 KB-enabled SimpleOpt 可查询并生成本地 state/outbox，但不发布公共 Catalog | V1 mode 可记录 usage、发布 Candidate/Observation 并写 Catalog | **2026-08-07 更新**：允许单 Definition SimpleOpt 显式查询；公共发布仍只属于 KernelGenWorkflow | SimpleOpt/KB |
| K-05 | KernelGen epoch publish/finalize/campaign | **已实现**统一 finalize barrier：发布必须为 `published/noop`，多 Agent 必须有 synthesis checkpoint；支持从完整 Ledger 重放 finalize，campaign 按 epoch 隔离 incomplete 算子 | 同样具有 publish barrier、`finalize_epoch` 和 campaign/resume，但依赖旧 Server/lifecycle | **已确认并完成**：沿用 jiabei 收尾与 campaign 逻辑，保留当前 v5.1、Ledger 3.0 和严格 round lifecycle | KernelGenWorkflow |
| K-06 | exact-scope solution registry | **已实现**：只限 Knowledge-enabled KernelGen；支持完整 `fresh/fork/resume`、精确 seed、`flaggems-v5-v5.1` benchmark identity 和严格更优 promotion；SimpleOpt 不消费 | 支持完整 `fresh/fork/resume` 和严格更优 promotion，但 benchmark identity 仍复用旧 `trace_set_key` | **已完成**：采用 jiabei lifecycle，并按当前 v5.1 Catalog 修正 identity | KernelGenWorkflow/KB |
| K-07 | empty recovery 的 `def_name` 缺陷 | 当前 `_load_completed_result()` 显式接收并使用 `def_name`，空结果不会引用未定义变量 | jiabei fallback 引用未定义的 `def_name`，可能在 resume/finalize 抛 `NameError` | **已规避并纳入 K-05 recovery**：没有复制该缺陷 | KernelGenWorkflow |
| K-08 | KB-guided 实验是否强制双路检索 | Knowledge Coder/Profile role 和 Skill 已实现同问题 Concept+Source 与 `4/2/200` 指导；Python/MCP 不阻塞 eval，lineage 只审计；普通角色无 KB 工具 | 同一 prompt/Skill 指导和 Python 非阻塞语义 | **已完成**：只作用于 Knowledge-enabled KernelGen；现有 jiabei campaign 是旧 prompt，不能作为新 K-08 验收 | KernelGenWorkflow/KB/Agent |

## 14. 建议讨论顺序

1. `S-01..S-17` 和 `K-01..K-08` 已确定并完成主体适配。
2. K-07 已随 K-05 recovery 规避，不需额外移植；当前没有剩余公共对比项。
3. 在一台代表性芯片做显式 `read_write_v1` 最小 KernelGenWorkflow E2E，确认普通
   SimpleOpt/BatchSimpleOpt workspace 不出现 Knowledge state、outbox 或 KB 调用。
4. 最后按多芯片实际 `/status` 补 scope/golden query 验证。

这个顺序可以避免用 jiabei 的旧 SimpleOpt/旧 Server 适配层覆盖当前已经验证的
多设备 SimpleOpt，同时为后续 KernelGenWorkflow 和 KB 接入保留清晰接口。

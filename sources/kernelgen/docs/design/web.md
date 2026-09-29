# KernelGen Web

## 背景

KernelGen Web 承接两类算子优化请求，并将请求转换为可验证、可调度和可追踪的 KernelGen workflow：

1. FlagGems 验收算子使用 FlagGems adapter。已有 pytest 时，先按照 `FlagGems-master/docs/content/zh-cn/testing/kernelgen-integration.md` 转换协议；对应 Agent 尚在开发。没有 pytest 时，需要先自动构造 pytest；对应 Agent 尚未开始开发。
2. 通用算子使用 v6.2 native 协议。提交者提供 `definition.json`、`oracle.py`、`correctness.jsonl` 和 `timing.jsonl` 四个文件。

## 用户需求

1. 提交者可以创建算子优化申请，选择 FlagGems 或通用算子，并选择一个或多个目标平台。FlagGems 请求至少提供算子名和源码版本；通用算子请求必须提供完整的 v6.2 四文件 schema。
2. 提交者必须选择优化模式：普通优化使用 SimpleOpt，面向单算子的有限轮次优化；极致优化使用 KernelGen，执行包含多 epoch、Knowledge 和 profiler 的完整优化流程。页面应说明两种模式的预计耗时和资源差异，所选模式随请求输入快照固定，执行中不可切换。
3. 提交者可以查看自己请求的输入、校验结果、执行进度和最终产物，也可以撤销自己的请求。排队中的请求立即撤销；运行中的请求进入“取消中”，停止启动新一轮。当前 Agent invocation 保留完整末次输出，已结束调用的自动重试先检查取消；KGS 声明取消 capability 时主动取消正在排队或运行的 Reference/Preflight/Eval/Profile，否则等待当前 Server 调用自然结束。
4. 多平台请求按平台拆分为独立执行任务。每个平台分别展示状态、轮次、加速比、失败原因和结果，部分平台失败不能覆盖其他平台已经完成的结果。
5. 所有请求展示当前阶段、排队位置、已运行轮数、当前最佳加速比、开始时间、最近更新时间和预计剩余时间。预计剩余时间无法可靠计算时应显示“暂不可估算”，不能提供虚假精度。
6. 请求完成或准入失败时通知提交者。失败信息必须包含失败阶段、可操作的原因和是否允许修正后重提，不能只返回 Agent 的原始文本或堆栈。

## 准入与验证需求

1. 请求进入优化队列前必须固定输入快照并生成唯一请求 ID。已进入校验或执行的输入不可原地修改；修改输入应创建新版本或新请求，以保证结果可复现。
2. v6.2 请求先进行文件完整性、JSON/JSONL 语法、协议版本、公共 ABI、workload 和 oracle 静态校验，再在所选目标平台的 KernelGen Server 上执行权威验证。只有 baseline 可执行、全部 correctness workload 的 validator 通过且 timing workload 得到有效计时后，才能进入优化队列。
3. FlagGems 请求必须记录源码仓库和 commit。已有 pytest 的请求在协议转换后验证 pytest 覆盖范围、benchmark workload 与生成 schema 的一致性；没有 pytest 的请求在自动构造能力交付前明确标记为“暂不支持”，不能直接跳过 correctness 校验进入优化。
4. 每个平台在入队前校验 Server 的 `api_version`、`target.device`、backend、timing 配置和 scheduler 健康状态。Web 服务不能使用自身的 Torch/Triton 环境推断目标平台能力。
5. 准入失败必须保留原始输入、校验日志和结构化错误。提交者修正后以新 attempt 重试，历史证据不得被覆盖或删除。

## 调度与执行需求

1. 后台调度器定期领取已通过准入的请求，根据请求类型选择 pytest 构造、协议转换和 schema 验证流程，再根据优化模式启动 SimpleOpt 或 KernelGen。领取操作必须具备租约或等价的原子机制，避免服务重启或多调度实例导致同一任务重复执行。
2. 每个执行任务使用独立 workspace；同一算子 workspace 同时只能有一个 Coder。中断续跑复用原 workspace 和 ledger，不使用 `--clean`，重新执行则创建新 attempt 并保留旧产物。
3. 调度必须匹配请求所需平台与健康设备。Server 侧每张可见设备同一时刻最多运行一个 Eval、Profile 或 Debug Job；`checking` 或 `broken` slot 不得分配新任务。
4. 任务只能在 Server scheduler 满足设备数一致且无异常状态时切换批次。运行中发现 `broken > 0` 时，不再向坏 slot 派发请求，并在当前组结束后暂停该设备的后续组，等待人工处理。
5. Web 撤销、超时或进程重启不得触发已转发 Eval、Profile 或 Debug 请求的自动重放。无法确认是否执行过的调用应标记为待核查，由 ledger 和 Server 记录完成对账后再决定是否续跑。
6. 当前已知会触发设备隔离问题的平台应支持管理员禁用批量队列。昆仑芯默认不得进入 BatchSimpleOpt；只有明确授权的独立单算子诊断请求可以被调度。

## 状态与结果需求

请求状态至少包括“待校验、校验中、校验失败、排队中、优化中、取消中、已取消、已完成、优化失败、基础设施异常”。状态转换由 workflow、ledger 和 Server 返回的结构化事实驱动，不能根据日志关键字或模型自行总结判定。

优化中的任务应展示当前轮次和最大轮次、当前 best round、`best_geo_mean`、最近一次 correctness/timing 结果以及停止原因。命令退出或 Web 服务重启后，系统应从 `optimize_definition_output.json` 和 `.ledger.json` 恢复状态。

最终结果必须区分以下概念：

- `PASSED`：全部 workload 数值正确且具有有效 headline timing。
- 性能提升：`best_geo_mean > 1`。
- 当前批量优化合格：在 correctness 和有效计时成立的前提下，`best_geo_mean >= 0.8`。

Web 页面不能把 `PASSED` 显示为“已加速”。最终结果还需确认 best kernel 是目标实现，而不是直接调用 `torch.ops.*` 等框架 fallback。

## 机器与运维需求

1. 机器页面展示平台、Server/API 版本、最近心跳、设备总数以及 scheduler 的 `active`、`waiting`、`available`、`checking`、`broken` 和 `incidents`。机器失联和设备 slot 异常必须与普通任务失败分开告警。
2. 管理员可以暂停或恢复平台队列、调整 Agent 并发上限、查看请求与设备的绑定关系，并在保留证据的前提下发起续跑。Web 不提供自动物理卡复位能力。
3. 运行记录必须保存请求输入快照、request/attempt ID、KernelGen 与 KernelGen Server 的 commit/tag、API 版本、平台和设备、backend、timing、模型、workspace、启动参数、日志、ledger、最终 JSON 和 best kernel。
4. KernelGen Server 只绑定目标机 loopback。Web 服务不得向浏览器或公网暴露 Debug Job 接口；访问远端 Server 必须使用受控的项目 SSH stdio HTTP 代理。模型 token、endpoint、SSH 私钥及环境变量不得写入页面、日志或结果包。

## 最小交付范围

第一阶段优先交付通用 v6.2 请求的提交、准入验证、单平台 SimpleOpt/KernelGen 模式调度、进度展示、撤销和结果下载。FlagGems 已有 pytest 的协议转换可在对应 Agent 稳定后接入；自动构造 pytest 和多平台并行调度作为后续能力，但数据模型和状态设计应从第一阶段起支持优化模式、请求版本、平台子任务和多次 attempt。

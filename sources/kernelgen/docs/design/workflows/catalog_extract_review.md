# CatalogExtract：公共抽取与有界审核修订

同一源码算子先由 `CatalogExtractWorkflow` 生成并审核一份 Native Catalog，再将相同内容交给不同目标的 `OperatorOptimizeWorkflow`。不为每个芯片分别抽取，也不根据单个目标的能力删减公共 workload。

## 输入与执行

输入仍为 `operator`、互斥的 `flaggems_repo` / `pr_url`，以及可选 `case_list_path`。新增 `max_review_rounds`，默认 3，范围 1～10，表示包含首次抽取在内的审核轮数上限。未提供清单时仍复用已有采集流程；本改动不代表已实现硬件中立的完整 case 枚举，采集环境影响必须保留在来源证据中。

执行顺序为源码和用例准备一次 → 抽取 → 静态结构/覆盖校验 → 只读 Catalog 审核 → 必要时反馈修订 → 再审核。后续修订使用原抽取会话并重新输出完整契约，不让 reviewer 修改文件。BaseAgent 自身的输出契约修复与这里的语义审核轮次互相独立。

每轮产物保存在 `attempts/01/`、`attempts/02/` 等目录，包括 Catalog、extraction.json、accuracy_coverage.json 和 review/review.json。审核报告绑定内容摘要、必读证据及其文件摘要；缺少必读证据或出现 P0 转换错误不通过。finding 的 `category` 为 `conversion`、`source_quality` 或 `target_capability`：后两者仅记录，即使 priority=P0 也不触发抽取修订；P1 不阻塞。未分类 finding 保守按 conversion 判定。源证据在抽取/修订前后必须保持一致；审核期间也不得改动 Catalog 或证据。取消在完整模型输出后检查，不强杀 Agent。

仅审核通过后复制该轮原样内容到最终 `catalog/`，输出既有 `catalog_path`、`operator_dir`、`source_evidence` 等字段，并增加 `review_path`。达到上限仍未通过时，`review-loop.json` 记录 NEEDS_FIX，Run Control 显示 WAITING_REVIEW，并抛出带报告路径的 `CatalogReviewRequired`；不发布最终 Catalog，也不启动目标验证或优化。运行时、解析或持久化异常不是“审核不通过”，不能静默算成通过或当作普通 P0 无限重试。

发布时在 `catalog/` 同级写入 `catalog.review.json`，只记录算子、原审核报告路径及摘要，不把审核元数据打进 Operator Bundle。原 `attempts/NN/review/review.json` 仍是抽取保真审核的事实来源。当前 `OperatorOptimizeWorkflow` 对新任务默认重新审核目标测试契约；抽取阶段的审核报告和已安装来源都不构成免审条件。只有用户显式传入 `skip_review=true` 才跳过模型测试审核，目标 reference 验证仍执行，见 [统一测试契约审核](review_tests.md)。

直接提供未审核 Catalog，或记录缺失、内容变化、证据失效时，仍执行普通语义审核。`catalog.review.json` 记录抽取阶段的结论，不替代每个新优化任务对目标测试契约的审核；旧实验如使用 `semantic_review=REUSED_ACCEPTED`，只能按其原版本和冻结计划读取或续跑，不自动迁移为当前策略。

当前会话内支持自动迭代与协作式取消；跨进程恢复抽取会话尚未实现。已存在 attempts 的 workspace 保留证据，不覆盖尝试目录。需要重新启动完整抽取时使用新的 workspace；不能手改已接受 Catalog 后沿用其审核身份。

## 审核边界

抽取只负责忠实转换原 pytest，不替代 pytest review：原测试覆盖不足、缺少边界或 dtype、对原 reference 设计的质量建议属于 source_quality，留给 pytest review，不要求抽取 Agent 扩充测试。源 reference 的 float64、TO_CPU/support_fp64 分支和结果 cast-back 按源语义保留；静态结构校验不再一律禁止 correctness_run 中的 float64 转换。修改源码精度、删已有 workload、改变 timing baseline、漏掉原有梯度断言仍属于 conversion，而非测试质量建议。模型分类不是语义正确性的证明，报告必须给出源要求与抽取差异的具体证据。

抽取发布使用 `ReviewOutput.blocks_catalog()` 判定转换保真性；下游 `review_tests` 使用独立的测试契约审核策略，旧抽取结论不自动复用。报告及内容摘要仍完整保留；目标验证独立执行，不因 target_capability 在抽取审核中非阻塞就忽略真实 readiness 失败。不增加新的长期编排服务，不自动重写或重跑历史实验。

审核对象是选定 public operator 的源码语义、ABI、oracle 独立性、workload 和容差忠实性。遗漏源测试要求、改变副作用/alias、循环正确性 oracle 或放宽容差属于阻塞问题。同文件其他算子不自动扩大当前范围；原测试未覆盖的额外边界、凑够 100 case、新增 NaN/负零等通常是建议，不作为新通过条件。

目标 API/dtype、编译器、计时和 profiler 可用性由逐目标 readiness 判定。公共 Catalog 不因某卡不支持就删除用例。源码要求但协议无法表达的行为应明确记录为协议缺口，不能通过改 reference 或测试语义伪造通过。精度与环境 setup 要区分是否影响选定算子、是否发生在计时 callable 内；不能机械复制无关全局配置，也不能仅为让 Preflight 通过而删除数学上必需的设置。

角色规则位于 `.kernelgen/agents/kernel-artifact-reviewer.md`。这是 Native Catalog 语义审核，不等同于执行原生 Gems pytest 的 `pytest-review` skill。静态审核通过也不代表 reference 已在某芯片执行成功；后续 OperatorOptimize 必须继续做 Bundle 绑定和目标验证。

## 已确认设计：KGS 内置源参考精度与测例适用策略

2026-09-15 确认：KGS 不为获取 Gems 的 `support_fp64` 而依赖 Gems 安装或运行时导入。将 Gems 的对应厂商配置复制为 KGS 随包分发的静态配置，记录来源仓库、exact commit 和源路径，由 KGS 版本管理；不在运行时自动同步 Gems，不增加用户 CLI 配置项。当前核对的来源为 FlagGems `d64794e63b502cb836bc015a92a62c42de4be05a` 中的 `src/flag_gems/runtime/backend/` 厂商描述配置。

该配置表示源 pytest 使用的参考精度与测例适用策略，不是硬件能力探针结果。KGS 根据目标环境实际识别的厂商解析策略；未知厂商明确返回未知，不沿用源描述类默认 `True` 来猜测。不得用 Agent 本机 Torch、一次 float64 分配成功或临时环境变量替代该策略。

抽取的 Native reference 保留源 `support_fp64`、`TO_CPU`、upcast 和 cast-back 的分支语义，通过明确的执行契约消费 KGS 提供的策略，不要求安装 Gems。同一份配置同时覆盖源测例使用的 `support_fp64`、`support_int64` 等适用条件，不再为测例选择维护另一套厂商判断。仅迁移源参考精度和测例适用所需的配置，不改写 candidate dtype、timing baseline，也不影响不使用该策略的 Native Catalog。配置来源与冻结 oracle 必须可追溯；不静默用另一版策略替换已记录的源语义。

抽取时保留选定算子的源 pytest 全部条件分支及对应测例，连同源适用条件一起写入公共 Catalog；不能根据抽取机器的配置提前删除 float64、int64 等条件测例。“全部”仅指源 pytest 已定义的覆盖，不要求扩充源测试，也不把原本有条件的 dtype 强制扩展到所有测试函数。单次 pytest collection 只体现当时环境下的展开结果，不能单独作为全部条件测例已完整抽取的证明。

执行时由目标 KGS 依据源条件和目标配置判定：条件满足则执行；条件不满足则保留该测例并记录 SKIP 及原因；所需配置未知则明确报告无法判定，不能猜测执行或静默跳过。配置声明适用但实际运行失败仍保留真实执行结果，不能事后改成条件性 SKIP。同一份 Catalog 供多芯片复用，不生成按芯片删减用例的副本。

Review 检查源测例、参数和条件是否忠实保留，不自行推断设备支持情况；参考精度选择与测例适用判断共用上述策略，但各自保留源语义。首版条件表示及执行接口见下文；现有 Schema 无法表达时记录协议缺口，不提前删除测例或伪造通过。

协议兼容仍只看 `/status.api_version`，可选支持仍只看 `/status.capabilities`。本节不是已部署 Server 的能力声明，也不据此将受阻 Catalog 标为可运行。KG 改动在活跃的 `fix/advisory-code-review` 分支开发，配套 KGS 改动按跨仓库分支规范独立提交和验证。

## 已确认设计：报告阻塞并提前终止

2026-09-15 确认：允许抽取或审核 Agent 报告无法通过修改当前 Catalog 解决的协议或环境阻塞，提前结束该算子的本次抽取，不必耗尽修订轮次。漏测例、参数映射错误、reference 转换错误等可修复的 conversion 问题仍走原有有界修订流程；普通审核不通过不自动等同于阻塞。

阻塞报告必须结构化记录源行为要求、具体源码或执行证据、缺失的接口或环境条件，以及解除阻塞所需的变化。Workflow 校验报告和证据引用完整后结束本次尝试，不能仅依据自然语言日志中的“不支持”判定；报告完整不等于其技术判断已获独立验证。证据不足的判断按现有流程处理，不能伪造阻塞结论。单个目标不支持某 dtype 等不妨碍忠实抽取的 target_capability 建议，仍不得据此终止公共 Catalog 抽取。

提前终止保留当前产物、会话和阻塞证据，明确区别于成功、普通审核拒绝及 Runtime 异常；不发布 accepted Catalog、不启动下游优化、不自动反复重试同一阻塞，也不阻止其他独立算子继续处理。停止发生在完整模型输出结束并完成结果持久化之后，不直接 kill Agent。具体结果字段与 Run Control 状态映射见下文，不新增 RunState 枚举。

本次另确认暂时跳过 `asinh_`：其 Gems `*args, **kwargs` 入口隐藏了单 Tensor 参数契约，当前抽取入口无法解析；保留失败证据，不新增专用签名适配，不计为审核失败或优化失败。此处仅记录后续处理决定，不改写历史实验状态。

## 2026-09-15 开发实现与验证边界

上述决策的首版实现位于 KG `fix/advisory-code-review` 与配套 KGS `feat/native-source-policy`，尚未发布，不能据此假设已部署 Server 具备能力。后续真实抽取验证使用 Codex Runtime（`kg extract --runtime codex`），不修改其他 Workflow 的 Runtime 默认值。

源策略身份只保存在 `Definition.source_policy_id`；oracle 使用 `kernelgen_server.runtime.source_policy.source_flag(name)`，workload 使用 `source_condition.flags` 保存布尔期望。KGS 按执行作用域绑定逻辑 backend 和 Definition 策略，支持 fp64、bf16、int64 标志的 AND 条件；不引入通用表达式解释器，暂不能表达的复杂条件仍报告缺口。无条件 Catalog 无需此能力。KG 在上传/绑定所需 Catalog 时核对 `/status.capabilities.native_source_policy`，KGS 的 Native benchmark fingerprint 绑定策略快照。条件行显式生成，不使用可能错误继承条件的批量 dtype expansion。

Agent 的 `blocker` 与完整抽取资产互斥，包含类型、源行为要求、证据路径、缺失契约和解除条件。Workflow 检查引用属于已绑定证据；审核侧仍须完成全部必读证据。有效报告持久化后以 `PENDING / BLOCKED` 显示，抛出独立的 `CatalogExtractionBlocked`，CLI 非零退出并输出报告路径，不发布最终 Catalog。该状态表示本次执行已退出、等待解除阻塞，不表示后台 Agent 仍运行。取消请求仍在完整响应后的安全点优先处理。无效报告按现有错误路径处理，不伪造阻塞身份；技术判断仍是 Agent 报告而非独立验证结论。

当前验证包括 host Schema、工作流、CLI 和容器内 CPU 引擎测试；容器内测试用逻辑 backend 模拟策略选择，不是对应芯片的实测能力。正式目标部署与历史算子批量重跑尚未完成，不更新旧实验结果，也不移动 release tag 或修改锁定部署组合来宣称已经验收。

后续已补充 [昇腾单卡验真](../../validation/native_source_policy_ascend_20260915.md)：独立临时 KGS 上，真实 `fix` Catalog 33/33 通过，并验证 fp64=false 的 NPU float32 分支、条件 SKIP、ALL_SKIP 后 benchmark、错误与单卡排队。该验证不升级原服务，不等同于全部历史算子或其他芯片已通过。

真实 Codex 抽取复测：`fix` 使用固定 Gems `d64794e63b502cb836bc015a92a62c42de4be05a` 和既有冻结 timing 清单，在新 workspace 首轮抽取与审核通过，得到 18 个正确性测例（其中 6 个保留 `support_bf16` 条件）和 15 个原 timing 测例。Oracle 的参考精度使用 `source_flag("support_fp64")`，没有分配探针。审核仍记录容差证据不足和原实现移植性两项 P1 建议；accepted 不表示这些问题已经独立验证，更不表示目标 readiness 已通过。本地证据位置为 `runs/kernel_todo_v2/source-policy-codex-qhGxfy/fix/`，不随仓库分发，未覆盖旧实验目录。

## 昇腾旧 Catalog 修复与规则收敛

2026-09-15 将 15 个旧 Catalog 修复到新的本地实验目录，并用 Codex 真实重抽 `adaptive_max_pool2d`，后者首轮审核接受且没有发现项。精度选择改为执行期 `source_flag`；`linear`、`addmm_`、`baddbmm_` 不再为读取标志导入 Gems。`take` 补回条件 float64/int64，移除源固定 fp32/fp16 测试中误扩展的 bf16，正确性共 186 项；`diagonal_copy` 补回 45 项条件 int64，共 392 项，使用 KGS `source_vendor()` 保留 cambricon/int16 的 CPU 输入构造分支。全部原始输入和旧审核证据保留，15 项的 ABI、effects、timing 清单均逐项核对未改变。

抽取角色及注入提示统一为“条件矩阵显式列举，优先于无条件 dtype 压缩”，禁止以未解决的语义差异注释代替 blocker，也不把已支持的源精度分支误报为协议缺口。模型输出校验要求导入源策略的 oracle 声明 `source_policy_id`，即使其 workload 没有条件字段。审核同步检查 metadata、执行期查询、源 case 范围及 fallback 边界。

CPU 执行检查另发现 `batch_norm_backward` 的旧 `valid` 同时有错误拆包和缺少显式 verdict 两项问题，修复了新 Catalog 并同步角色规则。KGS 的 `valid` 实参取决于实际返回：tuple/list 转成顶层列表，Tensor/dict 包成单元素列表；不由 Definition 输出名数量决定。成功路径必须返回 bool 或 verdict mapping，不能沿用 pytest 仅做 assert 后隐式返回 None 的习惯。该修复未改变 KGS 现有 hook 语义，配套测试锁定单个声明输出下的 tuple 行为。

验证：KG 207 项相关 host 测试、KGS 119 项相关测试通过；16 个修复 Catalog 均通过 CPU 代表性 reference-as-solution 检查。这不是昇腾全量 readiness 或性能结果；人工修复的 15 项没有伪造模型 accepted 记录。完整本地索引与逐项状态为 `runs/kernel_todo_v2/ascend-catalog-repair-ot19eX/index.json`、`REPORT.md`，不随仓库分发。

仍保留 `rnn_relu` 的真实协议阻塞：源 pytest 要求对候选执行 backward 并比较 input/hx/params 梯度，而当前 `valid` 仅收到 detached CPU 值，没有候选 callable 或计算图。调整精度、放宽审核和新增厂商查询都不能补上该能力；本轮未引入新的 autograd 验证协议，也没有发布 forward-only Catalog 冒充完整转换。`asinh_` 仍按用户既定决定跳过。未启动二阶段优化、升级常驻 Server 或修改正式 lock/tag。

# Catalog 抽取与优化

## 两个独立入口

`CatalogExtractWorkflow` 接受 Gems checkout 或支持的 PR URL、算子名，以及可选 `case_list_path`；未提供清单时由临时 Agent pytest 在配置的 SSH + Docker 采集环境生成。Workflow 调用抽取 Agent、计算覆盖报告，并执行有上限的语义审核与反馈修订，只有审核通过才发布唯一的 Native Catalog。输出保留 `extraction` 与 `accuracy_coverage`，返回 `catalog_path`、`operator_dir`、`source_evidence`、实际 `case_list_path` 和 `review_path`；launcher 不再重复落盘。它不连接目标 KGS，不承担逐芯片执行验证。循环、审计和审核边界见 [CatalogExtract 审核修订](catalog_extract_review.md)，来源支持见 [PR 抽取说明](flaggems_pr_extract.md)。

`OperatorOptimizeWorkflow` 接受互斥的 `catalog_path` 或 `catalog_name`，以及 `operator` 和 `optimization`，不接受源码抽取参数、不调用抽取 Agent。新流程为 `prepare_catalog → review_tests → optimize → code_review`：prepare 冻结来源、上传适用的本地输入并保存目标契约与审核证据；所有来源默认执行测试契约审核，仅显式 `skip_review=true` 跳过。上传和来源核对不代表审核通过，审核阻断时不执行 reference 或候选。安装式 Catalog 从目标 KGS 获取完整契约和源码，不读取本地同名副本、不上传 Bundle。Native 和 Gems adapter 均支持 SimpleOpt 与 KernelGen；Gems 始终由原 pytest 评测。具体能力门禁、恢复边界和输入示例见 [Catalog 输入来源](catalog_input_sources.md) 与 [统一审核](review_tests.md)。

不再提供 ExtractAndOptimizeWorkflow。调用方独立执行 CatalogExtract，将产物路径交给一个或多个 OperatorOptimize；KG 不额外保存组合任务或跨目标调度状态。旧组合 campaign 使用原 KG 版本查看和续跑，不自动迁移；独立 OperatorOptimize 的执行计划和 `stages/optimize/work` 路径保持不变。

已有 Catalog 的最小真实输入：

```json
{
  "catalog_path": "/path/to/prepared-catalog",
  "operator": "relu",
  "optimization": {
    "definition_name": "relu",
    "eval_server_url": "http://127.0.0.1:8000"
  }
}
```

```bash
kg run --catalog-path /path/to/prepared-catalog --definition relu \
  --workspace ./campaign/operators/relu --eval-server http://127.0.0.1:8000
```

远端 KGS 的 URL 使用本地 SSH stdio proxy 地址。日常优化统一使用 `kg run`，支持后台状态、取消、续跑和 YAML Batch；默认 KernelGen，也可显式选择 `--mode simple_opt`。上面的 JSON 展示 Python Workflow 输入结构，不是 CLI 的 `--input` 文件；独立 Catalog Python launcher 已删除。Workflow 的 `dummy: true` 仅用于不需要真实 Catalog、模型或 Server 的编排测试。模式和代码输入见 [优化模式与代码输入](catalog_optimizer_modes.md)。

本地 Catalog 的执行计划冻结输入路径、选中算子的 Bundle 内容与 manifest 摘要以及目标环境。续跑需要原输入仍可读取且匹配，workspace 内的快照也必须通过 artifact 校验；修改输入或快照不能沿用已完成的审核/验证。安装式输入冻结目标返回的 Definition、workloads 和 benchmark fingerprint，后续评测核对目标契约，不从本地 Catalog 补齐。其他算子的内容不是本任务的输入真源。正确性 reference 与性能 baseline 继续遵循 Catalog 契约，不按来源重新猜测或合并。

流程拆分本身不增加来源抽取器；独立 Extract 的 Gems PR 输入和自动采集由后续 [PR 抽取能力](flaggems_pr_extract.md) 接入。ATen/vLLM/SGLang/AscendC/Torch 抽取器、内存指标、硬件中立完整 case 枚举和多 GPU 编排仍未由本流程实现。其他来源可以先准备符合现有协议的 Catalog，再进入优化；当前协议无法表达的原生 baseline 依赖仍需另行开发，不承诺任意来源都已可运行。来源选择收敛在 Extract 内部，不进入 Optimize。

原拆分时的历史 host 门禁为 183 passed：`test_catalog_optimize`、`test_extract_and_optimize`、`test_flaggems_v62_batch_extract`、`test_operator_development`、`test_run_control`、`test_cli_lifecycle`、`test_optimizer_inputs`、`test_worker_pool`、`test_simple_opt_package`、`test_workflow_names`（均位于 `tests/`，文件后缀 `.py`）。包括真实阶段编排搭配模拟 Agent/KGS/优化器的全链路，以及摘要篡改、取消和续跑；未重新执行模型或芯片 E2E。验证在已有 `kernelgen-nvidia-cu128` 容器内完成，显式核对 KG feature worktree 与 KGS 主线 `c309cb351d2a473e79dcad80a90df4363d5b9fb7` 的导入路径，没有安装依赖或修改 KGS 锁定清单。下文昇腾结果是拆分前的历史验收，不是此次测试结果。

## 原组合流程的验收与约束

状态：开发快照已通过 2026-09-11 昇腾单算子五阶段真实 E2E，包括在途取消和原 workspace 续跑，见 [验证报告](../../validation/catalog_optimize_ascend_20260911.md)。这不是全量算子或跨设备验收；审核发现的非阻塞 P1 仍保留，未验证能力不能视为已经交付。当前验收不包含 PR 输入解析、PR 提交、多目标并发或自动准备镜像。

## 2026-09-11 原始目标（历史背景）

在一个固定 Gems checkout、固定 KGS 容器与 Python 环境中，按顺序执行 `extract_catalog → review_catalog → distribute_catalog → optimize → code_review`。先在昇腾以单算子、SimpleOpt、低并发和 profiler 计时验证。KG、模型 Runtime、Catalog 和 workspace 留在本地，通过 SSH stdio HTTP proxy 访问远端 loopback KGS。不得修改运行中的既有 Server、升级 Torch/Triton/厂商运行时，或用本机环境推断目标能力。

抽取复用 FlagGemsV62ExtractorAgent，输入包括固定源码和目标生成的 case list；Catalog review 检查源码语义、oracle、workload 与容差。分发使用内容寻址 Operator Bundle，只有 KGS 明确返回 `operator_bundle_upload.evaluation_binding=true` 才允许执行。上传后的目标验证负责加载、reference-as-solution 和计时就绪；它不能替代上传前的语义审核。优化复用现有 Workflow，实测事实仍归 ledger；代码审核绑定准确代码摘要，修改代码后必须重新评测。

生命周期复用现有阶段控制、PID 身份绑定的 owner、cancellation generation、scopes、结果凭据和续跑校验。不新增 campaign 状态副本，不把日志关键字当作状态。优化阶段续跑复用原来的工作目录、session 和 ledger，阶段尝试报告独立保留。规划中的多目标执行按目标分别保存分发结果、优化目录和 ledger。

## 当前验收边界

当前 CLI 与 Batch 实测见 [统一优化入口验证](../../validation/unified_catalog_cli_a100_20260914.md)。OperatorOptimize 的 Catalog 审核仍是只读门禁，阻断时返回 WAITING；审核修订循环属于独立 CatalogExtract。代码审核采用 advisory 策略：P0/P1/P2 均完整写入绑定候选摘要的 review.json，不因发现的优先级阻断 Workflow，也不改写 ledger 的实测结论。报告路径随 Workflow 结果保留，可作为后续 epoch 的修复依据；当前代码审核位于整个 optimize 之后，不会自动倒灌给已结束的 epoch。缺少必读证据仍须重审，模型执行错误和取消保持原语义，不能把未完成审核记录为完成。reference-as-solution 是目标就绪证据，不写入候选优化 ledger。正确性、计时、Profile 和设备健康门禁不受 advisory 策略影响。

## 待改进：抽取契约与 Workflow 封装

状态：2026-09-11 确认设计方向；Catalog 落盘与已有 Catalog 优化入口已按上文拆分，PR 源码准备和 SSH + Docker 自动采集已增量接入，硬件中立的完整枚举仍待验证。本节保留此前实验背景，不把昇腾结果扩展为跨设备验证。

这里有两个相关但可独立评审的需求：硬件中立抽取解决输入契约与事实归属，Extract Workflow 整体封装解决调用边界和自动编排。先明确前者的契约，再据此完成后者，避免把“必须依赖目标 timing 清单”的现有路径固化为新接口；封装完成不能单独证明抽取已经硬件中立。

### 需求一：硬件中立的 Catalog 抽取与逐目标验证

原 2026-09-11 实验要求手工提供目标生成的 `case_list_path`；现已可通过 SSH + Docker 自动采集，仍不等于硬件中立完整枚举通过验收。当时 relu 的原 benchmark `--list-cases` 由测试准备步骤通过昇腾 Debug Job 执行，抽取 Agent 本身不负责采集；15 个 timing 用例的 ID、shape、dtype 经对照一致，18 个 correctness 用例来自源码抽取和审核，没有在目标端实际采集 accuracy pytest 的完整展开清单。这不足以保证任意算子在动态参数、设备条件和 skip 下都保持用例一致。

改进后的职责分为两部分：

- 本地 KG 从固定源码 revision 抽取并审核一份硬件中立的 Catalog。“完整”指原始测试声明的用例及其生成规则、适用条件，不是自行扩充所有 shape/dtype。保留 correctness/timing、参数、输入生成、oracle、容差和条件性 skip，不依据本机 Torch/Triton 或设备能力删减用例。
- 各目标 KGS 验证同一份 Catalog 的 dtype、API、输入生成、reference、正确性与计时链路，分别保存目标验证报告，不修改通用 Catalog。报告区分通过、已确认不支持、原测试条件性跳过、执行失败和未验证；reference 抽取错误、运行时故障或超时不能一概归为硬件不支持。

KG 可以运行在 CPU 服务节点，负责 Web/API、编排、抽取、审核和 workspace；模型可通过独立服务调用，编译及设备执行留在 KGS。“本地抽取”不等于在任意 CPU 环境直接运行原 pytest：Gems 的 dtype 集合和参数展开可能依赖设备能力，单纯执行本机 collect 会把本机限制带入 Catalog。源测试需要目标运行时才能确定的动态部分，应保留条件或生成规则并标记待目标核验，不能用猜测的枚举结果宣称完整。常驻服务不要求引入 KG 中央 daemon，每个 run 仍可保持独立进程；远端继续使用 loopback KGS 与 SSH stdio proxy。

一致性检查应覆盖 correctness 和 timing，由确定性代码比较原始测试与 Catalog 的用例身份、参数、shape/dtype、输入生成及适用范围；输入名不同必须显式映射，不能因字段名不匹配就跳过比较。对动态展开、fixture 或 skip 条件，必要时通过目标 KGS 采集原 pytest 的实际结果作为对照证据；清单一致不能替代 oracle、输入分布和容差的语义审核。无法表达或无法证明一致的部分保留待确认状态，不删用例、不改 reference 来伪装通过。

通用 Catalog 是测试契约的真源；目标报告是该契约在特定环境中的适用性与执行事实。目标报告绑定源码 revision、Bundle 摘要、实际执行环境和用例身份，并纳入原 lifecycle 的阶段凭据；优化实测仍归各目标自己的 ledger。后续若允许仅针对目标验证通过的用例生成，必须显式冻结执行范围，同时保留完整 Catalog 总数及未执行用例和原因，不能将部分通过报告为全量通过。现有执行接口是否需要扩展用例选择应另行设计，本次不增加静默过滤路径。

后续验收至少覆盖：CPU 节点抽取不受本机 BF16 能力影响；两个目标消费相同 Catalog 摘要但产生各自报告；correctness/timing 的漏项、额外项或参数差异被阻止；条件性 skip 与真实失败分开记录；续跑不能沿用其他 Bundle 或环境的验证凭据。先完善契约和一致性门禁，再接入自动采集与多目标调度，不混入此次已完成的固定环境 E2E。

### 需求二：Extract Workflow 整体封装

`CatalogExtractWorkflow` 已支持自行准备 case list、执行 Agent 抽取、计算覆盖报告并在独立 workspace 落盘 Catalog；调用方直接消费其 Catalog 产物，OperatorOptimize 的 prepare 负责打包上传。PR 入口还封装固定源码准备；当前自动采集保留原环境条件，不等于已完成硬件中立的完整用例契约。

目标是让 Extract Workflow 统一编排源码 checkout/revision 校验、源码收集、用例准备、Agent 抽取、确定性校验和 Catalog 落盘，返回明确的产物路径、覆盖报告及来源信息。调用方不再必须先手工生成清单；外层 lifecycle 的抽取阶段只调用 Workflow 并登记结果，Bundle 打包/上传仍可由后续分发环节负责，独立 Catalog Review 与目标 reference 验证不并入抽取。

用例准备和必要的原 pytest 采集由 Python 编排层调用 helper，执行成功并保存清单、源码 revision、采集参数及相关目标环境信息后，再将证据路径或内容传给抽取/审核 Agent。不能由 Agent 自由决定是否采集、选择哪些用例或解释失败为成功。需要目标运行时的采集通过 KGS 完成，作为逐目标核验证据，不重新成为通用 Catalog 的唯一真源；其与抽取阶段的具体时序由需求一的契约决定。

内部职责保持分离：Workflow 管步骤及取消/续跑衔接，采集 helper 管 pytest 执行，Agent 管测试语义转换，validator 管确定性一致性检查，持久化函数管产物落盘。复用现有函数，不把所有逻辑堆进 Agent 的 `preprocess()`，也不在外层 lifecycle 再实现一份相同准备流程。

封装验收应证明：调用方一次调用即可得到抽取产物；必须采集的步骤失败或证据不一致时不继续调用 Agent 或发布成功结果；取消/续跑不丢失已完成的采集证据；证据只有在源码、参数和相关环境匹配时才可复用。具体参数模型、采集协议与多目标调度后续单独设计，本次仅记录需求。

### 接口预留：统一源码输入与 PR URL

状态：GitHub FlagGems/FlagGems-vllm PR 入口已按 [PR 抽取说明](flaggems_pr_extract.md) 增量实现，`CatalogExtractWorkflow` 可用 `pr_url` 替代 `flaggems_repo`，另需提供 `operator`；`case_list_path` 可选，省略时自动采集。Python 准备并固定 PR base/head，PR Agent 只增加上下文，复用原语义抽取器及校验。下文统一 `source + operator` 模型、硬件中立完整用例契约仍是规划接口；优化入口不接受 PR URL，并非已经全部实现。

已有 Gems checkout 的输入示意：

```json
{
  "source": {
    "kind": "checkout",
    "path": "/path/to/FlagGems",
    "revision": "<完整 commit>"
  },
  "operator": "relu"
}
```

PR 来源的输入示意：

```json
{
  "source": {
    "kind": "pull_request",
    "url": "https://github.com/owner/FlagGems/pull/123"
  },
  "operator": "relu"
}
```

以上为拟议输入，不是当前可执行配置。两种 source 互斥校验，不同时接受 checkout 路径与 PR URL。PR URL 由 Python 解析为仓库、PR 编号及准确的 base/head commit，准备独立、干净的本地 checkout；默认抽取 PR head，不隐式合并目标分支。URL 用于来源追溯，实际抽取内容由冻结的 commit 决定。首次解析结果纳入执行计划与阶段凭据，续跑不自动追踪 PR 更新；采用更新后的 head 必须建立新的计划与验证证据。准备源码不自动执行 PR 提供的安装脚本，也不覆盖调用方已有工作树。

一个 PR 可能涉及多个算子，初期仍要求显式指定单个 `operator`，不猜测要抽取的对象；自动发现算子列表与批量调度后续单独接入。URL 支持范围首先限定为 Gems 仓库及其 fork，并按明确支持的托管平台解析；不是任意仓库或任意 URL 都自动具有 Gems 抽取能力，不支持的来源明确拒绝。认证由部署侧提供，不将凭据写入 URL、执行计划或 Catalog。

`case_list_path` 不再是调用方必须手工准备的输入，而是 Workflow 管理的证据产物；具体如何生成通用测试契约及采集目标清单，遵循需求一，不因新增 PR 来源退回到目标 timing 清单唯一真源。workspace 与 Runtime 由编排器显式传入，目标 KGS 配置属于逐目标验证上下文，不绑定通用 Catalog 的源码身份。

接口验收应覆盖：同一源码 commit 和相同抽取配置从 checkout/PR 两种入口进入时，解析为相同源码内容并复用相同后续步骤；PR head 更新后续跑仍固定原 commit；缺失算子、不支持的 URL、鉴权或拉取失败不启动抽取；源码准备不改动已有工作树。Agent 生成本身不保证逐字节可复现，不以两次模型生成的 Bundle 摘要必须相同作为入口等价性的判据。

## 已完成的小重构与历史验证

2026-09-11：以下公共改动已分别合入主线，本业务分支已全部同步并接入。它们不是待开发能力，也不意味着上述硬件中立抽取、整体封装或 PR 来源已经实现。

- !101 统一优化输入：SimpleOpt、KernelGen 和本流程共同调用 `workflows/optimization/single_coder/inputs.py::build_optimizer_input()`。调用方保留各自的 Catalog/reference、analysis/seed 和 Bundle 快照准备，不复制 Schema 或改变 DPS、timeout、轮次默认语义。详见 [共享输入契约](optimizer_inputs.md)。
- !102 迁移 Coder 租约：CLI 与本流程共同使用 `framework/worker_pool.py`；进程身份、锁和状态路径复用 `framework/local_state.py`。本流程的优化阶段占一个 Coder 名额，不与 KGS 请求线程或 device slot 合并，不改变当前目录资源池隔离和完整模型输出后的取消安全点。详见 [公共租约](../runtime/worker_pool.md)。
- !103 拆分 SimpleOpt：`workflows/simple_opt/` 按 contracts、preparation、knowledge、workflow 分工，公开入口不变；Coder 循环、Profile、最终复测和 distillation 仍在共享优化执行层。
- !104–!105 及业务接入完成名称调整：`CatalogExtractWorkflow` 是现有抽取能力，`ExtractAndOptimizeWorkflow` 是本五阶段业务流程，`OperatorDevelopmentWorkflow` 是仍以 dummy 验证的八阶段开发流程，`SingleCoderOptimizationWorkflow` 是单 Coder 优化执行层。目录与 Python 类型同步更名，CLI、现有 launcher 和 workspace 持久化标识保持兼容；详见 [命名映射与验证](workflow_names.md)。

上述历史实现已由 [Workflow build](../runtime/stage_orchestration.md) 取代：业务 Workflow 声明具名调用并返回 WorkflowResult，公共执行层管理 receipt、取消与恢复；不再继承 OperatorDevelopmentWorkflow，也不再提供 StageRunner。

最新集成 host 回归为 310 passed，覆盖抽取、Bundle 阶段、SimpleOpt/KernelGen、公共输入、租约及 CLI lifecycle；三类更名前 workspace 的完成、等待与取消续跑对照通过。该结果不包括此前记录的 16 项 ledger/Coder 历史失败，也不等于重构后重新做过真机 E2E；精确范围与命令见 [命名验证记录](workflow_names.md#extractandoptimize-业务编排命名)。

提 PR 前再次扩大回归：KG 360 passed，在上述 310 项基础上加入 agent roles、evaluation snapshot、KGS adapter、CLI/Batch 与 Run Control。KGS 在已有本地 `kernelgen-nvidia-cu128` 容器中 62 passed，覆盖 Bundle 执行/存储、Schema、Profile 生命周期、operation cancel、native adapter、隔离进程组和 device pool。宿主机没有 Torch，初次同组测试为 49 passed / 3 failed / 9 skipped，失败均为缺少 Torch 导入；容器复跑消除了这些环境缺项。两边导入路径均指向配套 worktree，未安装依赖或启动真实设备任务。容器解释器为 `/data/jiabei/venvs/kernelgen-cu132-torch211-triton360/bin/python3`，Torch 2.11.0+cu130、Triton 3.6.0；这次 host 结果与前述昇腾实测环境分开记录。

配套 KGS 执行绑定已通过 [KGS !47](https://gitee.com/BaaiAC/kernelgen_server/pulls/47) 合入，当时 `deployment/kgs.lock.yaml` 固定主线 commit `cfef13154608963e08a5b096a2c13d9a1a3148a3`（历史验证快照；新部署以当前 lock 文件为准），文件树与已验证的 feature `a5749f5f` 相同。该 commit 是保留现有版本元数据的开发快照，不是新的发布 tag；未移动 KGS v6.3.1 tag。源码与 release 的正式发布另按 SemVer 计划处理。运行时始终以 `/status.api_version` 和 capabilities 为准，旧实例 `evaluation_binding=false` 时明确拒绝本流程；已有 campaign 不自动更换 KGS。

最终锁定 commit 复验：从 `cfef1315` 创建独立 detached worktree 并核对导入路径，KG 上述 360 项、额外 `tests/test_install_kgs.py tests/test_cli_server.py` 的 63 项，以及 KGS 同组 62 项全部通过。锁定 release/Protocol 与该 checkout 的 `compatibility.yaml`、包元数据一致；未重新进行芯片 E2E，未安装或更改运行时依赖。

## 后续设计：一个 KGS 管理多个执行环境

此节仅记录设计，不属于本次实现。任务选择一个预先登记的执行环境，部署配置将其解析成固定镜像 digest，或者已有容器中的准确 Python 虚拟环境。不要让同一个长期 Python 进程在请求之间切换环境，也不在评测前临时升级核心运行时。

```text
本地 KG → SSH stdio proxy → KGS 管理端
                            ├─ 环境 A：worker 容器 A
                            └─ 环境 B：worker 容器 B
```

KGS 管理端负责请求、Bundle、物理设备租约、任务状态、取消和 worker 生命周期；worker 负责实际环境探测、加载、Preflight、Eval、Profile 和设备健康探针。worker 需要配套的 KGS 执行代码、轻量依赖、Torch、编译器与厂商运行时，按任务增加 Gems 和 profiler；不需要独立 HTTP Server、KG 或模型 Runtime。

管理端可以部署在宿主机，也可以部署在能访问宿主机容器引擎的管理容器。后一种方式创建的是并列 worker 容器，不要求 Docker-in-Docker。Docker socket 通常意味着宿主机级权限，只能交给可信管理端，不能暴露给执行候选的 worker。挂载必须使用宿主机可解析的路径。镜像由操作方授权并验证驱动兼容，不允许任务任意提供特权、宿主机挂载或 Docker 启动参数。

同一容器内也可使用不同虚拟环境，以指定解释器启动独立 worker。虚拟环境主要隔离 Python 包，不完整隔离系统库、CANN/厂商工具链、驱动或设备；差异较大的底层环境使用不同镜像，镜像仍需兼容宿主机驱动。建议按一次算子优化运行复用 worker 环境，每次评测继续使用隔离子进程，避免每轮重新创建容器。

物理设备必须在宿主机范围内互斥。不同容器里的 `npu:0` 可能是同一物理卡，不能分别分配同一设备 token。现有 KG 远端设备锁包含容器标识，而且位于各自本地 CLI 状态目录，不能当作跨容器、跨 KG 客户端的全局租约。第一版多环境部署采用不重叠的静态分卡或先释放再切换；后续动态租约使用稳定的宿主机/物理设备身份，进程异常与取消后完成清理和健康验证才释放。

执行计划冻结实际镜像、解释器、核心依赖、KGS/Gems commit 和 Bundle 摘要。部署配置表达要求，目标环境探测表达事实，两者不得混用。管理端 `/status` 的 Python/Torch 环境不能冒充 worker 环境；worker 能力必须有明确的查询契约。续跑禁止静默更换环境，环境变化要求重新验证并使用独立优化证据。KG release、KGS release 和 Protocol 仍相互独立，协议与功能判断分别使用 `api_version` 和 capabilities。

## 接入顺序

先完成固定环境的 Bundle 执行绑定与单算子生命周期 E2E，再独立设计环境描述和 worker 查询/执行协议，最后接入容器生命周期及宿主机设备租约。不将上述执行架构改造混入当前 Bundle 上传或昇腾验收。

相关文档：[Operator lifecycle](operator_lifecycle.md)、[Run Control](../runtime/run_control.md)、[多设备操作手册](../../operations/multiple_device_experiment_runbook.md)。

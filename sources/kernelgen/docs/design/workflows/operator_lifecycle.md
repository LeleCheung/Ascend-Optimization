# OperatorDevelopmentWorkflow：整体流程与接入状态

源码已更名为 `OperatorDevelopmentWorkflow` / `workflows/operator_development/`；CLI 与持久化标识保持兼容，详见 [Workflow 命名边界](workflow_names.md)。下述 dummy 功能基线不因更名而扩展。

另有已接入 CLI 后台的 [OperatorOptimizeWorkflow](catalog_optimize.md)，多环境 worker 仍是规划。两者共用 [Workflow build 执行层](../runtime/stage_orchestration.md) 和 Run Control，不通过业务子类继承复用；本文的八阶段 `operator_lifecycle` 仍要求 dummy，不因 Catalog 流程完成真实 E2E 而自动支持真实执行。

实现基线：`dev@4a2753ee`（[PR !98](https://gitee.com/BaaiAC/kernelgen/pulls/98)，2026-09-09）。当前已完成生命周期编排和运行控制的 dummy 验证，可通过 `kg run --mode lifecycle` 前台启动；八个业务阶段均未接入真实执行。本文同时记录目标流程和待办，标为“目标”“待接入”的内容不是已完成能力。现有 `simple_opt`/`kernelgen` 启动方式、KG/KGS 发布版本和 Protocol 均不改变。

## 整体流程

目标是把一个算子从测试准备、生成优化到代码交付的过程放进同一套 workspace 和 run control，而不是重新实现各个 Agent。用户指定一个 campaign 根目录和算子列表，编排器分配每个算子的 workspace，再显式传递各阶段的 workspace、上游报告路径和 run control。子阶段不猜目录，不从其他任务扫描输入。

当前已实现的调用链如下；多个算子串行执行，图中的 `workflow_call` 默认始终为 `DummyWorkflowCall`：

```text
kg run --mode lifecycle --dummy
  → cli/lifecycle.py：共用参数适配、输出和退出码
  → run_campaign()：校验算子列表及共用计划，保存 campaign 索引
      → operators/add/：OperatorDevelopmentWorkflow.run()
          → 校验单算子计划，领取 PID 身份绑定的执行所有权
          → 按固定顺序遍历选中的 stage
              → 校验并复用已成功报告，或分配 stages/<stage>/attempts/01/
              → 显式传入 WorkflowContext，执行 dummy
              → 原子保存 result.json，事件记录报告摘要
              → 取消安全点 → 更新 stage scope 和单算子根进度
          → 返回完成、失败、等待或取消；释放执行所有权
      → operators/mul/：相同流程、独立控制状态
  → campaign_status()：读取各算子的状态并汇总，不另存一份 campaign 状态
```

业务目标链为“pytest 生成（如果缺少）→ pytest review → 算子生成/优化 → 代码 review → 本地 CI → PR 提交 → PR review + fix → PR 状态跟进”。当前只实现选定阶段的固定顺序执行，不会自动判断是否已有 pytest；审核反馈后的修复、重新测试和再次提交也还不是实际循环。未来的 PR 跟进需要区分等待审核、请求修改、CI 失败、已合并和关闭未合并，不能把“PR 已创建”当成生命周期成功。

## 各阶段职责与完成情况

可选阶段及固定执行顺序如下，选中的阶段通过同一个可注入的 `workflow_call(WorkflowContext) -> WorkflowResult` 执行。默认 `DummyWorkflowCall` 不导入 Agent、不创建 Runtime、不调用 GPU/KGS、不安装依赖、不执行 Git 或 PR API。

下表产物均指未来真实产物；当前每个阶段实际只生成 JSON 模拟报告。

| 阶段 | 目标职责与产物 | 现有组件与待接入项 |
|---|---|---|
| `pytest_generate` | 接收算子需求与参考输入，缺少 pytest 时生成测试；pytest 暂留本地 workspace，不提交 Git | [TestWriterAgent](../../../agents/test_writer/__init__.py) 已存在，未接入；不使用 PytestConverterAgent。当前 Agent 面向显式 FlagGems checkout 写测试及 benchmark，需先适配本地阶段产物、已有测试复用条件和目标环境执行边界，不能只换调用名 |
| `pytest_review` | 生成前验证测试链路，产出 `READY/NEEDS_FIX`、P0 证据及非阻塞 P1 建议 | [pytest review](pytest_review.md) 已有最小验证设计与脚本，未接入 lifecycle；按该设计复用，不额外假设已有独立 Reviewer Agent |
| `optimize` | 调用 SimpleOpt 或 KernelGen，产出候选代码、优化 ledger 及最终结果 | [统一优化入口](../../../workflows/optimization/workflow.py) 已支持两种引擎；目前只传递 `optimize.mode` 到 dummy 报告，真实输入、Runtime、KGS、参数与结果适配均待接入 |
| `code_review` | 对确定版本的候选代码进行规范及实现审查，产出带代码摘要的问题报告 | lifecycle 专用执行适配和结果契约尚未实现；确定性源码准入检查与 Preflight 保持职责清晰，不在两处复制同一规则 |
| `local_ci` | 在确定的集成代码及目标环境上运行必要检查，保留命令、日志、版本和结果 | 独立阶段契约及执行适配尚未实现；不能把优化 Eval 通过或 PRSubmitter 的静态检查当成完整集成 CI 通过 |
| `pr_submit` | 提交经过审核和 CI 的准确代码版本，记录仓库、分支、commit 与 PR 地址 | [PRWorkflow](../../../workflows/pr_workflow.py) 和 [PRSubmitterAgent](../../../agents/pr_submitter/__init__.py) 已存在，未接入；原流程会创建/清理 Git worktree 并集成代码，需要先明确与上游审核快照、阶段 workspace 和提交授权的衔接 |
| `pr_review_fix` | 读取指定 PR head 的反馈并修复，重新审核/测试后更新同一 PR | 反馈读取、修复循环、回到 Review/CI、迭代上限及外部操作幂等性尚未实现 |
| `pr_followup` | 查询托管平台的审核、CI 和合并状态；等待时保留可恢复检查点 | 平台状态适配、定时唤醒及真实终止条件尚未实现；当前只模拟完成或等待，不轮询平台 |

这是固定顺序的控制面试验，不是通用 DAG 框架。真实 Review-Fix 循环、按反馈回到 CI、PR head 绑定、人工审批和定时跟进尚未实现；不能把一次线性 dummy 运行当作这些能力已通过 E2E。当前也不探测已有 pytest，不校验真实代码或 pytest 的阶段依赖；后续接入真实 adapter 时必须显式提供或引用所需产物，缺少输入时报错，不自动补跑被排除的阶段。

## 输入输出与模块分工

当前 `OperatorDevelopmentInput` 只有 `operator`、`stages`、`optimize`、`dummy` 和 `resume`；单算子 workspace 通过构造器 `cwd` 指定，不是 Agent 自行推导的输入。`operator` 现在只是安全路径标识，不能等同于已解析的 ATen 名、Catalog Definition 或可执行算子包。真实算子需求/源码、已有 pytest、Catalog/参考快照、目标设备及提交目标等业务输入尚未定义，不能把这些数据随意塞进当前参数后假定已生效。

`run_campaign(workspace, operators, ...)` 是批量入口；所有算子共用阶段选择和优化配置，每个算子独立运行一个 `OperatorDevelopmentWorkflow`。单算子直接调用也保留。`WorkflowContext` 明确传递算子、阶段、attempt、阶段 workspace、上游报告路径和 scoped RunControl；业务子类 `OperatorWorkflowContext` 另外携带优化配置。后续 adapter 负责转换为原 Agent/Workflow 的既有接口，不把算法逻辑搬进 lifecycle。

每个具名调用返回 `WorkflowResult(state, output)`，公共 Workflow 执行层持久化带输入摘要的 receipt。单算子返回 `OperatorDevelopmentOutput`，即 `WorkflowResult[WorkflowSummary]`，通过 `result.state` 和 `result.output.reports` 读取执行状态及报告。CLI 仍返回各算子状态的 campaign 汇总。具体组合接口见 [Workflow build 设计](../runtime/stage_orchestration.md)。本八阶段入口仍只接受 `dummy=true`，不能仅删掉检查就宣称支持真实执行。

| 模块入口 | 负责什么 | 不负责什么 |
|---|---|---|
| [CLI 适配](../../../cli/lifecycle.py) | 参数、调用前台批量入口、输出和退出码；与 Python 示例共用 | Agent 调度、真实优化、后台进程 |
| [campaign.py](../../../workflows/operator_development/campaign.py) | 算子索引、统一配置、分配单算子 workspace、串行调用及状态汇总 | 复制单算子阶段循环或持久化第二份任务状态 |
| [workflow.py](../../../workflows/operator_development/workflow.py) | 单算子输入和 build 调用声明；owner、路径、状态、取消及恢复委托公共执行层 | 重新实现 TestWriter、优化算法、CI 或 PR 平台 |
| [contracts.py](../../../workflows/operator_development/contracts.py) | 输入校验、阶段顺序、上下文/报告契约和 dummy runner | 推断远端设备能力或真实产物依赖 |
| [run control](../../../framework/run_control.py) | 原子状态、事件、scope、取消 generation 和已有跨进程控制 primitives | 判断代码质量、性能优劣或 PR 是否合并 |

## 阶段选择与优化模式

`OperatorDevelopmentInput.stages` 不传时选择全部阶段，也可以只选一个或多个阶段。输入顺序会归一化为上表顺序；空列表、重复项和未知阶段报错。未选中的阶段不创建目录或 scope，不计入总任务数，也不生成虚假的完成报告。`WorkflowContext.inputs` 只包含本次计划中已执行成功的上游报告，不扫描其他 workspace。

`optimize.mode` 选择 `simple_opt` 或 `kernelgen`，复用现有 `RunMode` 枚举，不把两种模式拆成两个 stage。选中 `optimize` 而不传配置时默认 `simple_opt`；未选中 `optimize` 却传入配置时报错。当前配置只有 `mode`，不接受尚未接入的轮次、并发等参数；后续复用已有优化参数契约，不另建一套手工映射。只有优化调用的 `OperatorWorkflowContext.optimize` 非空，dummy 仅把模式写入报告且保持 `called=false`，不会真正分发到优化 Workflow。

```python
from kernelgen.workflows.operator_development import OperatorDevelopmentWorkflow, run_campaign

workflow = OperatorDevelopmentWorkflow(cwd="/tmp/operator-life-selected-add")
workflow.run({
    "operator": "add",
    "stages": ["optimize", "code_review"],
    "optimize": {"mode": "kernelgen"},
    "dummy": True,
})

# 同样的选择可用于串行 campaign；尚无并发调度或后台提交。
run_campaign(
    "/tmp/operator-life-selected-batch", ["add", "mul"],
    stages=["optimize", "code_review"],
    optimize={"mode": "simple_opt"}, dummy=True,
)
```

阶段选择和解析后的优化配置是不可变执行计划的一部分。resume 时传入同一选择和模式；仅调整选择列表的书写顺序不改变计划，增加或减少阶段、切换模式则拒绝续跑，应使用新的执行 workspace。campaign 索引保存共用执行配置，确保尚未启动的算子也不能在续跑时偷偷换模式；子算子计划记录实际执行快照，不引入第二份运行状态。当前计划 Schema 为 `1.1`，此前不含模式的 `1.0` dummy 计划不自动迁移，旧报告保持不动，新演示使用新目录。

现有 `kg run --mode simple_opt|kernelgen` 命令无需修改；新增的 `kg run --mode lifecycle` 直接调用此 dummy 编排，不把原优化入口重定向到 lifecycle。优化 `RunMode` 枚举仍只包含 `simple_opt`/`kernelgen`，lifecycle 不进入后台优化请求及 worker 权重计算。未来 CLI 内部即使复用生命周期编排，也应保留现有优化命令和参数语义；SimpleOpt、KernelGen 仍可独立调用。

## 目录与状态归属

```text
campaign/
├── .kernelgen/operator-lifecycle-campaign.json   # 不可变算子索引及共用执行配置
└── operators/add/
    ├── .kernelgen/
    │   ├── operator-lifecycle.json              # 不可变阶段选择、优化配置、算子和模拟标记
    │   ├── lifecycle-owner.json                 # 执行期间的 PID + 启动身份
    │   ├── run-progress.json                    # 此算子根状态及阶段 scopes
    │   ├── run-control.json                     # cancellation generation
    │   └── run-events.jsonl                     # 共用结构化事件流
    └── stages/pytest_generate/attempts/01/
        ├── .kernelgen/run-control-ref.json      # 显式关联单算子根及阶段 scope
        └── result.json                          # 内容绑定的模拟阶段报告
```

其他阶段使用同样的 `stages/<stage>/attempts/<attempt>/` 规则；编号从 `01` 开始，至少两位，超过 `99` 后自然增长，按数值选择最新 attempt。不同算子不共享控制文件。campaign 根不维护第二份任务状态，`campaign_status()` 从各算子实时汇总。`operator-lifecycle.json` 只保存计划，不缓存状态。owner 文件只表达执行所有权，不保存成功/失败；正常返回或异常时释放，进程异常退出后的 PID 必须核对启动身份，不能仅检查进程号。

阶段使用 `stages/<stage>` scope，子阶段的完成只更新该 scope，不提前把整个算子标为成功。根的 `completed_tasks` 表示完成阶段数，不是优化轮数。没有生成任何 `.ledger.json`，不会把模拟数据注入优化历史。

pytest 后续是 workspace 的本地产物，生成阶段不做 Git 提交。本轮只有 JSON 模拟报告，没有生成可执行 pytest。真实本地 pytest 接入 KGS 评测仍是独立待办，不因目录编排完成而自动可用。

例如选中 `pytest_generate`、`pytest_review` 和 `optimize` 时，这三个阶段都位于同一个 `campaign/operators/add/` 下，分别使用自己的 `stages/<stage>/attempts/01/`。SimpleOpt 和 KernelGen 共用 `stages/optimize/` 这个阶段入口，内部目录仍由原 Workflow 约定。当前其中只有 dummy 报告，没有正式候选代码或 ledger。

## 结果、失败和续跑

此 dummy 入口的 `WorkflowResult.simulated=true`；返回状态使用 `SUCCEEDED/FAILED/WAITING/CANCELLED`，不再使用 `DUMMY_*` 前缀。必须同时检查 simulated，禁止把模拟成功解释为 correctness、性能、CI 或真实 PR 通过。RunControl 沿用现有执行状态枚举，并标记 `mode=dummy`。磁盘 receipt 的历史 `outcome/data` 编码只在公共执行层读写，用于保留旧任务恢复证据，不作为新的业务返回接口。

- 阶段失败或抛异常：根及当前 scope 失败，不启动后续阶段。普通失败结果允许 campaign 继续处理其他算子；未处理异常则停止本次调用，保留现场。
- 等待：scope 和根为 `PENDING`，`stage=WAITING_<阶段>`；释放 owner 后退出，不靠一个后台进程常驻等评审。只有显式 resume 才重试该阶段。
- 取消：沿用 `run-control.json` 的 generation，阶段开始前和完整结果写入后检查；不轮询、不中断半条输出、不 kill Agent。当前阶段与根确认取消，后续阶段不启动。campaign 收到一个取消结果后不再启动剩余算子。
- 续跑：必须显式 `resume=True`，算子、阶段选择、优化配置和模拟模式不得变化。成功报告绑定计划、所有上游报告的 SHA-256，并由 `LIFECYCLE_STAGE_RESULT` 事件记录报告自身摘要。读取时校验，变更或丢失证据直接报错，不默默重新认定通过。
- 重试：失败、等待或中断的阶段使用新的 attempt 目录，旧目录不删除、不覆盖。完整结果及对应事件已提交后，即使尚未更新成功进度就被取消或退出，也可据此恢复完成投影。只有文件、没有提交事件的孤立结果不作为成功检查点，保留后重试；因此不承诺未来远端副作用的 exactly-once。
- 进程异常退出：`lifecycle_status()` 根据 owner PID 身份将残留 RUNNING/CANCEL_REQUESTED 派生为 `INTERRUPTED`，不改写原始进度。resume 重新领取 owner；活跃 owner 拒绝重复执行。

报告是阶段结果证据，progress 是执行与展示投影，事件携带报告提交身份；没有另外的 `completed_stages.json`。目前只支持恢复同一个 campaign 路径，不支持搬迁 workspace。真正的优化结果仍应由优化 ledger 提供，PR 事实仍应由托管平台提供。

接入真实优化时，必须区分“恢复同一次优化”和“重新执行整个阶段”：前者复用原阶段 workspace、session 和 ledger，不因一次 lifecycle resume 就新建 attempt 重跑；后者才分配新 attempt 并保留旧证据。当前 dummy 的失败重试规则不等于真实优化的续跑实现，这项适配仍未完成。

真实设备能力、计时和设备 slot 仍以 KGS 返回为准；远端执行使用本地 Agent + SSH stdio proxy + 远端 loopback KGS。已有 Runtime 模型完整输出安全点、active operation 持久登记及 KGS cancellation 必须在真实 adapter 中接通，不能因 lifecycle 已有取消状态就声称远端操作也被取消。当前 dummy 遇到残留 active KGS operations 会拒绝执行，不进行远端清理。

## 最小使用示例

安装了当前源码后，可以直接使用 `kg run --mode lifecycle`。当前只支持前台串行 dummy，必须显式传 `--dummy`；不用加 `--foreground`，也不启动后台进程或领取 Coder lease。`--workspace` 指定 campaign 根，不传时自动在当前工作区 `.kernelgen/runs/` 下分配（遵循 `KERNELGEN_CLI_HOME`）。单算子用 `--definition add`，多个算子用 `--operators add mul`，两者互斥。

```bash
kg run --mode lifecycle --dummy --definition add

kg run --mode lifecycle --dummy \
  --workspace /tmp/operator-life-demo --operators add mul
```

只模拟优化和代码审核，可在新目录运行：

```bash
kg run --mode lifecycle --dummy \
  --workspace /tmp/operator-life-selected --definition add \
  --stages optimize code_review --optimize-mode kernelgen
```

`--stages`、`--optimize-mode`、`--resume` 和 dummy 故障注入参数仅用于 `kg run --mode lifecycle`，不能传给普通优化模式；反过来，lifecycle 暂不支持 `--batch-file`、KGS 地址或真实优化参数。续跑时保留算子列表、workspace 和阶段/模式选择，并追加 `--resume`。

验证失败后续跑，使用另一个全新目录：

```bash
kg run --mode lifecycle --dummy \
  --workspace /tmp/operator-life-retry --operators add mul \
  --fail-stage code_review

kg run --mode lifecycle --dummy \
  --workspace /tmp/operator-life-retry --operators add mul --resume
```

同样支持 `--wait-stage pr_followup` 和 `--cancel-stage pytest_review` 注入模拟情形；resume 时去掉相应注入选项。原 Python 示例的 `run` 子命令仍可使用，与 `kg` 共用 `cli/lifecycle.py` 的参数和执行适配。生命周期暂不登记到 `kg list`，也不接入 `kg status/history/logs/cancel/resume`；旧 CLI 的 RunRequest 和 status Schema 保持不变。查询和跨进程取消暂用 Python 示例：`status` 接收 campaign 根，`cancel` 接收单算子 workspace。

```bash
python3 -m kernelgen.examples.operator_lifecycle.run_example status /tmp/operator-life-demo

python3 -m kernelgen.examples.operator_lifecycle.run_example cancel \
  /tmp/operator-life-demo/operators/add --reason "stop this operator"
```

已经终止的算子拒绝再次取消；dummy 默认很快结束，上述命令不保证能赶上活动阶段。退出码：全部模拟成功为 0，失败或等待为 1，参数/执行异常为 2，取消为 130。查询结果始终含 `simulated: true`。

## 尚未完成与接入顺序

已经完成的是控制流程：单/多算子输入、固定顺序的阶段选择、优化模式配置、显式 workspace/scope、报告校验、协作取消、失败/等待/中断后的恢复，以及前台 `kg` 入口。以下工作仍未完成，按接入顺序推进：

| 顺序 | 待完成内容 | 接入前必须明确/验证 |
|---|---|---|
| 1. 测试准备 | 真实 TestWriter、已有 pytest 复用、pytest review | 输入/输出快照、本地 pytest 产物位置、目标环境执行；测试链路未执行或全部 skip 不可按通过放行；本阶段不提交 Git |
| 2. 生成优化 | SimpleOpt/KernelGen 的真实 adapter | 复用原参数和资源约束；ledger 性能真源；阶段正常续跑复用原 workspace；真实模型和 KGS operation 的取消/恢复 |
| 3. 审核与交付 | 代码 Review、本地 CI、PRSubmitter 适配、PR review-fix 及跟进 | 被审核/测试的源码摘要与实际提交一致；代码变化后重新验证；写远端前明确授权；重复调用不重复创建 PR；等待、关闭与合并的业务终态清晰 |
| 4. 批量与产品化 | YAML Batch、每算子配置、并发调度、CLI 后台提交及管理命令、Web 对接 | 等真实 Agent 接入稳定后再做；复用单算子 primitive，不新增另一套 Batch 执行器；Agent Coder lease、KGS 请求线程和设备 slot 分开管理 |

当前 `--operators add mul` 已经支持批量输入，不代表已经支持 YAML 或并发。`kg run --mode lifecycle` 已可启动及原命令加 `--resume` 续跑，但 `kg status/history/logs/cancel/resume/list` 尚未接入生命周期。Web 也尚无 lifecycle job 映射、事件消费或唤醒接口；以后需要将 job 关联到 campaign 及具体算子 workspace，不能仅绑定 campaign 根后把多个算子的进度混为一个任务，更不应复制一套阶段状态真源。

每接入一个真实阶段，先验证输入输出、成功/失败、取消和新进程恢复，再串联下一阶段。全流程真实 E2E、真实 Review-Fix 循环、远端副作用恢复及并发 E2E 均未完成；现有 dummy 测试只证明编排和控制行为。相关独立设计以 [pytest review](pytest_review.md)、[运行控制](../runtime/run_control.md) 和 [KG CLI](../kg_cli.md) 为准，不在本文另建一套相同规则。

## 验证记录

`tests/test_operator_lifecycle.py` 用真实文件锁、PID 身份、run control、子进程和报告测试：完整阶段顺序、多算子隔离、每个阶段失败、异常、取消安全边界、generation、等待/续跑、已完成阶段不重复执行、报告篡改/丢失、owner 冲突、PID reuse、真实子进程退出恢复、跨进程查询和 launcher。没有真实模型、设备或 Git/PR 操作。

2026-09-09 初版（`6f3e356d`）host 验证：新增 34 项测试全部通过；全量为 1216 passed、22 failed。独立检出基线 `dev@777e6457` 得到 1182 passed、22 failed，按 JUnit 测试标识逐项比较，所有既有测试结果一致。22 项既有失败集中在 Coder 最终复验、KernelGen 赢家选择及 ledger best source 测试，本分支未修改这些逻辑。另通过独立命令行进程验证两个算子在模拟代码审核失败后续跑完成全部 8 个阶段，保留原报告。测试前已核对导入路径指向被测 worktree；未安装依赖、调用模型或启动设备服务。

阶段选择与模式配置接入（`1f182a2f`）后，lifecycle 测试为 66 passed；连同 `test_run_control.py`、`test_cli.py`、`test_cli_batch.py`、`test_cli_resume.py`、`test_cli_history.py`、`test_cli_server.py` 和 `test_cli_runtime.py` 共 200 passed。新增覆盖单阶段执行、规范顺序、未选中阶段无状态/产物、两种优化模式的配置传递、非法配置、子集流程失败/等待/取消后的续跑，以及 campaign 尚未启动算子的计划冻结；独立示例进程验证只选优化和代码审核并指定 `kernelgen` 时失败后续跑。当时未重跑全量套件，也未修改 CLI 或真实优化实现。

`kg run --mode lifecycle` 接入后，上述回归套件加 `test_cli_lifecycle.py` 共 221 passed。新增验证当前进程前台执行、默认 workspace、单/多算子、退出码及续跑、模式专属参数拒绝、原优化提交路径，以及真实子进程 SIGINT 后完整 dummy 报告和取消状态。当前 host 的 PATH 没有安装 `kg` 可执行文件，跨进程测试直接调用其注册入口 `kernelgen.cli:main`，已确认导入此 feature worktree；未为测试安装依赖或调用真实业务阶段。

后续先接 TestWriter 的纯产物接口和 pytest Review，再接优化 Workflow。真实 adapter 必须处理目标环境归属、输入快照、资源 lease 和外部操作幂等性；接入前不能去掉本轮的 dummy 限制。状态语义详见 [run control](../runtime/run_control.md)，现有用户命令详见 [KG CLI](../kg_cli.md)。

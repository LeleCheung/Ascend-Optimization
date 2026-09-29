# Workflow 的 build 组合与统一返回

## 设计结论

业务层只使用 Workflow、具名调用和 `WorkflowResult`，不再维护独立的 Stage、StageResult、StageRunner 或业务 `*Stages` 类。`build(inp)` 声明执行顺序；每个调用可以执行一个子 Workflow，也可以是包装普通函数或 Agent 的薄函数。只有需要 workspace、持久结果和恢复的业务调用才进入 build；轻量计算仍直接调用。

当前 OperatorDevelopment 和 CatalogOptimize 使用此组合接口。SimpleOpt、KernelGen、CatalogExtract 等已有算法 Workflow 的业务输出保持不变，由调用处显式包装为 `WorkflowResult`；不对全仓 Workflow 做隐式状态推断或破坏性返回值替换。它们后续可单独采用类型化包装，不能仅凭返回了对象就默认业务通过。

## 开发入口

- [framework/workflow.py](../../../framework/workflow.py)：Workflow 基类的 `build()` 和 `run_build()`；没有持久编排需求的 Workflow 继续实现自己的 `_execute()`。
- [framework/workflow_result.py](../../../framework/workflow_result.py)：`WorkflowResult[T]`、`WorkflowContext`、编排汇总与磁盘 receipt 编解码。
- [framework/workflow_execution.py](../../../framework/workflow_execution.py)：私有公共执行机制，负责独占、输入校验、attempt、结果提交、RunControl、取消和恢复；业务代码不实例化执行器。
- [OperatorDevelopmentWorkflow](../../../workflows/operator_development/workflow.py)：build 声明选中的 dummy 调用。
- [OperatorOptimizeWorkflow](../../../workflows/optimization/workflow.py)：build 声明准备 Catalog 与共享优化流程。
- [Catalog 操作](../../../workflows/optimization/operations.py)：普通调用函数，负责实际子 Workflow/Agent 调用和结果映射；没有业务 Stage 类。

CatalogOptimize 直接继承 Workflow，以 build 声明 `prepare_catalog → review_tests → optimize → code_review`，不再保留 CatalogPipelineWorkflow 中间基类。CatalogExtract 独立输出 Catalog 路径，由调用方交给优化流程，不新增父子恢复记录或跨任务调度器。

Catalog 输入不再继承 dummy OperatorDevelopment 的阶段选择配置，也不提供 `stages`、`stage_order` 或第二份顺序常量；当前固定流程的顺序仅由 build 决定。公共执行层从本次 build 派生持久计划的历史 `stages` 字段，仍能校验旧 workspace。OperatorDevelopment 原有的用户步骤选择能力保留，不因这次 Catalog 清理而删除。

## 声明和返回

示意代码：

```python
def build(self, inp):
    return (
        ("extract_catalog", self.extract_catalog),
        ("review_tests", self.review_tests),
    )

def review_tests(self, context):
    catalog = context.outputs["extract_catalog"]
    report = review_tests(catalog)
    return WorkflowResult(
        state="WAITING" if report.has_blockers else "SUCCEEDED",
        output=report,
    )
```

`build` 中的名字标识一次调用，不是类名；同一个 Workflow 可在不同名字下调用。调用拿到 `WorkflowContext`，其中包含算子、scope 名、独立 attempt workspace、上游 receipt 路径和 RunControl。`context.outputs` 从上游成功 receipt 派生结果映射，不增加输出索引文件。业务输出在直接调用期间可以保留 Pydantic 类型，持久化后按 JSON 读取；需要类型化的下游显式按自己的输入模型验证。

`WorkflowResult[T]` 只有一份执行状态：`state=SUCCEEDED/FAILED/WAITING/CANCELLED`；`output` 保存业务结果，`message` 说明停止原因，`simulated` 区分 dummy。返回 WAITING 的 Review 可以携带完整报告，表示等待处理而不是执行异常。普通函数也必须明确包装，不接受裸 dict 并自动判定成功。

组合 Workflow 的公开返回也是这个格式，业务 output 为 `WorkflowSummary(operator, workspace, reports)`，不重复保存执行状态。旧调用方从 `result.status`、`result.reports` 改为 `result.state`、`result.output.reports`。Dummy 使用正常执行状态加 `simulated=true`，不再把 DUMMY 编进返回状态；模拟成功不代表真实业务验收。后续整合的 CLI/campaign 状态查询使用 [进度 Schema v2](stage_progress.md)，RunControl 的取消和 ledger 事实归属不变。

## 运行状态与恢复

运行中进度继续由 RunControl 提供，返回结果只表达调用结束时的状态。公共执行层依据 WorkflowResult 写终态；普通包装函数不重复写终态、复制取消处理或管理 owner。异常由公共执行层记录失败并抛出，模型输出的完整性仍由 Runtime safe point 保证。

| 情况 | 行为 |
| --- | --- |
| 成功且产物有效 | 跳过调用，沿用已提交结果 |
| 普通调用失败、等待或中断 | 新建 `attempts/02` 等目录完整重跑，不恢复旧 Agent session |
| Optimize 未完成 | 新 attempt 保存调用结果；适配函数继续使用 `stages/optimize/work`，由优化器自身 ledger、session 和停止规则恢复 |
| Catalog 或目标快照变化 | 拒绝沿用旧计划和优化 ledger，使用新 workspace |

Optimize 不新增通用 resume 参数或第二个 round 计数。到达停止条件后，重试调用不代表重新开始搜索；最终复验仍遵循显式 retest 规则。当前不新增同 workspace 的已完成调用强制重跑或下游失效传播开关；需要完整重跑时使用新 workspace，保留旧证据。

持久路径 `.kernelgen/operator-lifecycle.json`、`stages/<name>/attempts/01/result.json`、事件名称、PID reuse 校验、cancellation generation 和结果提交哈希保持不变。磁盘 receipt 内的 `result.outcome/data` 仅是保留旧证据的私有编码；统一在边界映射为 `WorkflowResult.state/output`，不再给业务提供第二个 StageResult，也不同时持久化两份状态。已提交 receipt 丢失、篡改或产物不一致时仍拒绝静默重跑。

## 验证与范围

`tests/test_workflow_composition.py` 验证 build 顺序、类型化子 Workflow 输出、普通函数包装、非法返回拒绝、失败／等待／取消后的跳过和重跑，以及既有 receipt 编码。生命周期、Catalog、抽取和 CLI 测试继续覆盖跨进程状态、独占、safe point、Optimize 稳定 workspace、产物校验和恢复。

这次不改变算法、KGS 协议、lease 权重或评测事实来源，不引入 DAG／并发调度。历史 PR 中的 CLI extract_and_optimize 接入（现已删除）及 `refactor/stage-progress-schema` 的进度 Schema 已按用户要求整合进本 PR；新 Schema 通过 call_progress_kinds 适配 build，不恢复旧 Stage 抽象。先前 [Stage 昇腾 E2E](../../validation/stage_orchestration_ascend_20260913.md) 属于旧提交记录；新返回接口、build 编排及真实 CLI 后台的结果见 [Workflow build 昇腾 E2E](../../validation/workflow_build_ascend_20260913.md)。

2026-09-13 本次 host 回归为 289 passed（包含原 Workflow、evaluation snapshot、生命周期、CLI、SimpleOpt 和 KernelGen 相关测试；一个现有 Starlette 弃用警告）。另用旧 `48bcb5d2` CLI 集成 checkout 在临时目录生成 Optimize 已完成、Code Review 失败的 dummy 任务，再由当前实现跨进程 resume：最终 SUCCEEDED，两个旧 receipt 字节不变，Code Review 创建 attempt 02。该检查只调用本地 dummy，证据目录为 `/tmp/kg-workflow-build-resume-nszZac/run`，不随 Git 分发；没有安装依赖或操作真实实验 workspace。

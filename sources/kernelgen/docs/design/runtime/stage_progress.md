# Stage-owned progress Schema

## 职责

Run Control 继续统一管理 state、stage、mode、stop_reason、message、updated_at、取消代次、事件和 scope 层级。Workflow/stage 显式选择自己的 progress_kind；CLI 不通过阶段名称、日志关键词或 null 值推断其含义。此改动不改变任务运行、取消和性能判定。

framework/progress_schema.py 提供四类有类型和范围校验的模型：basic 没有计数字段；rounds 包含优化轮次及 ledger 实测指标；tasks 包含子任务完成/失败/取消计数；epochs 包含 epoch、该 epoch 的任务计数和最佳性能。新增进度类型须定义模型并接入已知类型表，不能将任意字典直接作为阶段进度。

组合 Workflow 可以通过 call_progress_kinds 声明 build 中各调用的进度类型；未声明的调用使用 basic。OperatorOptimizeWorkflow 根据优化模式声明 optimize 为 rounds 或 epochs，其余调用为 basic；编排根节点使用 tasks。共享优化生命周期默认 rounds，KernelGenWorkflow 显式选择 epochs，独立 Coder scope 使用 rounds。普通函数也可以通过 context.control 显式选择类型，不由 CLI 维护阶段名映射。

## 唯一事实来源

progress_kind 是同一 run-progress.json 内的 Schema 选择元数据，不另建状态文件。现有控制字段继续由 Run Control 写入；current_round、completed_rounds、best_round、best_geo_mean、last_evaluation_status 仍不落入 run-progress.json，写接口仍拒绝它们。CLI 先完成 ledger 投影，再按声明的模型输出对应字段；v2 不再复制一份可写的 progress 字典。模型定义同时供内部控制记录复用，避免重复维护字段类型和默认值。

## CLI/API 输出

`kg status WORKSPACE --detail` 直接返回 JSON v2.0；不再提供 --json 别名、v1 输出、版本选择参数或 legacy 投影。cli.api.status、run_status、batch_status 及 lifecycle_status 统一使用新版结构，Batch 子任务同样为 v2.0。文本输出从相同结构读取适用的进度字段；调用方需要同步更新，不保留旧字段兼容层。其他命令的 --json 参数不变。

Review scope 示例：

```json
{
  "state": "RUNNING",
  "stage": "review_tests",
  "mode": "catalog_optimize",
  "stop_reason": "",
  "message": "Executing stage",
  "updated_at": "2026-09-12T00:00:00Z",
  "progress_kind": "basic",
  "progress": {}
}
```

相同位置的 rounds scope 则只展示 current_round、completed_rounds、max_round、best_round、best_geo_mean、last_evaluation_status，不展示 epoch 或任务计数。适用但尚无事实的字段仍可为 null，例如尚未完成第一轮的 current_round；不可将 null 等同于失败。

## 状态声明

没有显式选择进度类型的记录使用 basic，不根据旧字段推断类型，查询不会改写历史实验产物。新运行会声明类型；显式 resume 时，build 编排重新声明调用类型但保留已完成 receipt，不为更新展示字段重跑业务。旧代码不得与新代码在同一 workspace 并发运行。

## 验证

Host 测试覆盖独立 scope 类型、跨实例读写、新版默认输出、不允许直接写入 ledger 指标、未知类型拒绝、取消控制保持不变，以及 CLI、Batch、SimpleOpt、KernelGen、生命周期回归。测试不启动 KGS 或模型。

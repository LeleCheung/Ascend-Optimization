# KernelGen 运行控制面

KernelGen 内部通过 `kernelgen.framework.run_control` 暴露与前端无关的运行控制接口。CLI、测试工具或未来服务层只负责调用这些接口，不需要解析终端文本，也不需要侵入 Workflow、Runtime 或 MCP 工具。

## 接口边界

`RunControl` 组合了三类稳定能力：

- `ProgressStore`：通过 `progress()` 读取当前快照，通过 `update_progress()` 更新运行或子 workspace 的进度。
- `EventStream`：通过 `read_events(after_sequence=...)` 增量回放事件，通过 `record_event()` 写入结构化事件。
- `CancellationController`：通过 `request_cancel()` 幂等请求取消，通过 `checkpoint()` 在安全边界停止，通过 `clear_cancellation(expected_generation=...)` 为显式续跑清除旧请求。

默认实现 `WorkspaceRunControl` 使用 workspace 文件持久化状态，因此主进程、CLI Runtime 和 MCP 子进程不需要共享内存。调用示例：

```python
from kernelgen.framework.run_control import WorkspaceRunControl

control = WorkspaceRunControl("/path/to/run", source="cli")
snapshot = control.progress()
events = control.read_events(after_sequence=120, limit=100)
control.request_cancel("operator requested stop")
```

若子 workspace 位于运行根目录下，它会自动发现最近的运行根；外部分配的 workspace 使用 `link_workspace(path, scope=...)` 显式关联。

## 持久化合同

所有控制面文件位于运行根目录的 `.kernelgen/`：

| 文件 | 语义 |
| --- | --- |
| `run-progress.json` | 当前运行和各子 workspace 的进度投影，可覆盖更新 |
| `run-events.jsonl` | 仅追加的结构化事件流，`sequence` 在同一次运行内单调递增 |
| `run-control.json` | 幂等取消请求及其 generation |
| `active-server-operations.json` | 当前可转发给 KGS 的 Reference/Preflight/Eval/Profile operation ID 与 endpoint |
| `run-root.json` | 运行根标记 |
| `run-control-ref.json` | 非后代 workspace 指向运行根的显式引用 |

上述 Schema 当前均为 `1.0`。Schema 版本只描述本地控制面文件，不替代 Protocol version，也不用于推断 KG 或 KGS release 兼容性。

进度状态为 `PENDING`、`QUEUED`、`RUNNING`、`CANCEL_REQUESTED`、`CANCELLED`、`SUCCEEDED`、`FAILED` 或 `INFRASTRUCTURE_ERROR`。`stage` 描述当前阶段，例如 `CODING`、`PREFLIGHT`、`EVALUATING`、`PROFILING` 和 `OPTIMIZING_EPOCH`；并行执行另外维护 `total_tasks`、`completed_tasks`、`failed_tasks` 和 `cancelled_tasks`。`run-progress.json` 不持久化轮次、best round、best geo mean、评测状态或取消布尔值；`kg status` 和 `kg history` 共用 ledger 性能投影，取消请求从 generation 文件派生。旧 progress 中的这些缓存字段在读取时忽略，对外 status 使用进度 Schema v2.0（见 [进度 Schema](stage_progress.md)）；并行 Coder 的轮次仅在各自 scope 中展示，不拼接成根轮次轴。

事件包含 `event_type`、`source`、`scope`、`stage`、`level`、`visibility`、`message` 和结构化 `data`。Runtime 的逐行进展以 `RUNTIME_LOG` 事件记录并标记为 `DEBUG`，调用方可按可见性过滤；这类日志可能包含模型和工具上下文，应沿用 workspace 的本地访问权限，不应默认上传。

## 取消语义

取消是以 workspace 为权威来源的协作式请求，并可在 KGS 显式声明 capability 时向设备操作转发：

1. `request_cancel()` 立即把根状态改为 `CANCEL_REQUESTED`，重复请求返回同一个 generation。
2. `kg cancel` 读取 `active-server-operations.json`，对仍活动的 KGS operation 调用取消接口；排队操作不取得 slot，运行中的操作结束子进程树并在强探针后释放 slot。转发失败不撤销本地取消请求，Runtime 仍会在后续安全边界停止。
3. Runtime 不在输出流中轮询取消；当前 Agent invocation 的完整输出和 session ID 落盘后，才在 `AFTER_MODEL_INVOCATION` 安全边界退出。若一次调用因超时或 provider 可恢复错误而结束，在启动下一次自动重试前还会于 `BEFORE_MODEL_RETRY` 检查取消。
4. `finalize_round` 若看到取消请求，会把当前已完成轮次写成 `user_cancelled` supervisor stop，避免 Coder 开始下一轮。
5. 顶层 Workflow 确认不再启动新工作后调用 `acknowledge_cancellation()`，状态转为 `CANCELLED`。显式续跑必须用当前 generation 调用 `clear_cancellation()`，防止旧控制端误清除更新的请求。

`run_parallel(..., cancellation_token=...)` 会在 workspace 分配前检查取消，因此线程池中尚未开始的任务不会启动；已经运行的任务仍在自己的安全边界退出。

KGS 的 Preflight/Eval/Profile POST 仍是同步结果接口；取消控制面通过独立 operation ID、`GET /operations/{id}` 和 `DELETE /operations/{id}` 提供，不要求把正常提交改成轮询式异步 API。客户端只能依据 `/status.capabilities.operation_cancel` 启用该路径，不得根据 KGS release 推断。

手工取消、Batch 部分提交失败回收和前台 Ctrl+C 共用取消服务。SIGINT handler 只入队通知，独立线程写入取消意图并转发 KGS，不打断模型输出读取。active operation 登记反映远端生命周期；成功响应、明确拒绝或查询到远端终态后才注销，网络失联不等于执行结束。DELETE 先于 POST 到达而返回 404 时只有限重试 DELETE，不重试执行 POST；无法确认终态的记录保留。续跑在 workspace 提交锁内先检查 owner 并核实残留 operation，之后才恢复 ledger 和清除旧 cancellation generation。

## 已接入位置

- `OperatorDevelopmentWorkflow` 的 [dummy 生命周期试验](../workflows/operator_lifecycle.md) 复用每算子根控制与阶段 scope，验证等待、失败、取消和续跑；不代表真实 pytest、Review、CI 或 PR 已接入。
- 新任务由 `OperatorOptimizeWorkflow`、`SingleCoderOptimizationWorkflow` 和 `KernelGenWorkflow` 写入对应 scope 生命周期；历史 SimpleOpt/Python Batch 的 Run Control 保留在 `workflows/legacy/`，不迁移原目录或重写事件。
- `CLIRuntime` 写入模型调用、重试、失败和 `RUNTIME_LOG` 事件，并在调用前后检查取消。
- `preflight_kernel`、`eval_round`、`profile_workloads` 在远端调用前检查取消并写入阶段事件。
- `finalize_round` 更新实时阶段并发出轮次事件，把取消请求固化为 ledger 中安全的轮次终止事实，不复制性能缓存到 progress。

调用方应优先读取 `run-progress.json` 展示当前状态，使用 `run-events.jsonl` 的 `sequence` 游标流式展示事件，并在进程重启或事件缺口时重新读取 progress 与 ledger 完成恢复。

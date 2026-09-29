# 设备槽位隔离事故复盘

## 结论

2026-08-02 确认 KernelGen Server 的设备调度存在重大隔离缺陷：当
`--max-workers` 大于可见设备数时，旧实现会为同一逻辑设备创建多个 slot，使多个
Eval、Profile 或 Debug Job 进程同时使用同一张物理卡。

修复后的约束是：

- 一张可见设备只对应一个设备 token；
- 同一时刻，每张设备最多运行一个 Eval、Profile 或 Debug Job；
- `max_workers` 只控制请求执行线程上限，不再决定设备 token 数；
- worker 数大于设备数时，多余请求在 Server 内排队。

修复提交最初发布在 Gitee 测试分支
`test-device-slot-isolation-20260802`，提交为 `983b3b4`。

## 如何发现

在昇腾 910B 的 4 张可见卡上使用严格 `torch_npu.profiler` 测试
`flaggems_floor`：

| Server workers | 同时请求数 | 结果 |
| ---: | ---: | --- |
| 4 | 4 | 全部通过 |
| 8 | 6 | 1 个 `RUNTIME_ERROR` |
| 8 | 8 | 1 个 `RUNTIME_ERROR` |

失败请求不是数值错误或算子 API 不支持。Profiler 已执行并创建结果目录，但导出的
CSV 中没有可解析的算子统计行，Server 日志为 `No valid ops found`。该算子此前在
独立 profiler capability 测试中通过，因此问题与算子定义无关。

对照实验把 Server 恢复为 4 workers，仍同时提交 6 个相同请求。前 4 个请求执行，
后 2 个请求排队，6 个请求最终全部通过。这说明失败由设备并发调度触发，而不是请求
总数、SSH 代理或 `flaggems_floor` 本身触发。

昇腾 profiler 对并发采集更敏感，因此首先暴露问题；旧调度对所有 backend 都有
影响。即使 `do_bench` 没有报错，同卡并发仍会污染延迟和加速比，不能认为结果有效。

## 根因

旧版 `create_app()` 同时用 `worker_count` 创建线程池和设备 slot：

```python
worker_count = max_workers or len(devices)
slots = asyncio.Queue()
for index in range(worker_count):
    slots.put_nowait(devices[index % len(devices)])
```

例如 4 张卡、8 workers 会生成：

```text
npu:0, npu:1, npu:2, npu:3,
npu:0, npu:1, npu:2, npu:3
```

队列中的两个 `npu:0` 是两个独立 token。两个请求可以同时取走它们，并各自启动一个
隔离 eval 子进程。这里没有产生新的逻辑卡，只是错误地把同一逻辑设备登记了两次。

根因是混淆了两种不同资源：

- worker 是 Server 可承载的请求执行线程；
- device token 是独占的加速器执行资源。

Agent 的并发数可以大于卡数，因为 Agent 在生成和分析阶段不占卡；这不意味着
Server 可以让多个 eval 进程同时占用一张卡。

## 修复

设备队列改为只按可见设备创建 token：

```python
slots = asyncio.Queue()
for device in devices:
    slots.put_nowait(device)
```

线程池仍按 `max_workers` 创建。请求先取得 worker 容量，再取得独占设备 token；
任务结束或失败后归还两者。Eval、Profile 和 Debug Job 统一使用
`acquire_device()` / `release_device()`，避免某个入口绕过隔离。

`/status` 新增调度状态：

```json
{
  "workers": 8,
  "scheduler": {
    "device_slots": 4,
    "max_active": 4,
    "active": 4,
    "waiting": 2,
    "available": 0
  }
}
```

这里 `workers=8` 不表示 8 个设备任务可以同时运行；`max_active=4` 才是设备侧并发
上限。

## 回归验证

新增 `tests/test_server_device_slots.py`，使用 2 个伪设备和 4 workers 同时提交 4 个
请求，并在第一批请求被阻塞时断言：

- `active == 2`；
- `waiting == 2`；
- `available == 0`；
- 任一设备的同时活跃任务数不超过 1；
- 第一批释放后，第二批请求正常完成。

该测试已在天数、海光、摩尔线程、沐曦、昆仑芯、昇腾和平头哥七个目标环境的 Server
Python 中通过。

修复后又在昇腾真机上启动 4 张卡、8 workers 的 Server，同时提交 6 个完全相同的
`flaggems_floor` profiler 请求。6 个请求全部通过；观测到
`max_active=4`、`max_waiting=2`、`min_available=0`，证明后两个请求在 Server 内
排队，没有再复制 NPU token。

真机部署还必须检查 `/status.scheduler.device_slots` 等于授权设备数，并用并发
reference-as-solution 请求确认执行结束后：

```text
active=0, waiting=0, available=device_slots
```

上述空闲断言适用于所有请求正常退出的情况。请求 `TIMEOUT` 或隔离 worker 异常
退出后，slot 先进入 `checking`，Server 在全新进程中执行矩阵乘、softmax、结果
校验、同步和目标计时链路组成的强探针。探针失败立即转为 `broken` 并返回
`SUSPECTED_DEVICE_ERROR`；探针通过便恢复 slot，超时返回 `TIMEOUT`。普通算子
`RUNTIME_ERROR` 不触发健康检查。Server 不再跨 slot 自动重试同一请求，防止一个
异常算子连续污染多张卡。服务启动时也执行同样的逐设备强探针，只有通过的 slot
才进入队列。

## 使用和验收要求

- 不再通过限制 `max_workers <= 设备数` 掩盖问题；Server 本身必须保证设备独占。
- 调整 BatchSimpleOpt 的 Agent `--max-workers` 时，观察
  `/status.scheduler.waiting`。短暂排队正常，持续增长且设备始终满载表示 Agent
  并发已经超过当前 Server 的有效承载能力。
- 性能结果必须来自修复后的 Server。旧版本在
  `max_workers > device_count` 条件下产生的计时数据应视为可能受同卡并发污染。
- 昇腾仍使用严格 `torch_npu.profiler`，其他设备仍使用
  `triton.testing.do_bench`；设备隔离修复不改变计时策略。

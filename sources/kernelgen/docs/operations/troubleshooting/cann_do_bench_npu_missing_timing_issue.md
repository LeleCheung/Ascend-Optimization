# Issue 标题

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

`[do_bench_npu] _collect_prof_result 假设每次调用只产生一条 kernel 记录，导致部分算子返回 inf、越界或漏算耗时`

## 问题描述

在 Ascend 910B4-1 上使用 `triton.backends.ascend.testing.do_bench_npu` 测量 PyTorch/复合算子时，部分调用能够正常返回结果并完成同步，但 `do_bench_npu` 无法返回有效 latency。实测表现包括返回 `float("inf")`、抛出 `IndexError: single positional indexer is out-of-bounds`，以及同一个算子只有部分 shape 能返回时间。

问题出现在 `_collect_prof_result` 对 `kernel_details.csv` 的解释方式。当前实现按 `func_idx * (warmup + active) + iteration` 计算 CSV 行号，等价于假设每次 Python callable 调用恰好产生一条 profiler kernel 记录：

```python
expected_rows = num_funcs * (num_warmup + num_active)

for func_idx in np.arange(0, num_funcs):
    for active_index in np.arange(0, num_active):
        row_index = func_idx * (num_warmup + num_active) + num_warmup + active_index
        time_cost[func_idx] += filter_df.iloc[row_index]["Duration(us)"]
```

这个假设对通用算子并不成立：view/meta 算子可能不启动 NPU kernel，CPU fallback 算子不会产生 NPU kernel，复合算子一次调用可能启动多个 kernel，某些执行还可能因为融合或 shape 不同产生数量不同的 kernel。由此会出现三类问题：

- 没有生成 `kernel_details.csv` 时直接返回 `float("inf")`，调用方无法区分“无 NPU kernel”“CPU fallback”和“profiler 异常”。
- CSV 行数少于预期时，位置索引抛出 `IndexError`。
- 一次调用产生多条 kernel 记录时，当前位置索引只取其中一部分，返回值可能不是该 callable 的完整设备耗时。

`target_kernel_name` 可以约束单个已知 Triton kernel，但无法解决通用 PyTorch/复合 callable 的零 kernel、多 kernel 和动态 kernel 数量问题。

## 环境信息

- 芯片：Ascend 910B4-1
- Driver：25.2.0
- CANN：9.0.0
- Python：3.11.15
- PyTorch：2.9.0
- torch-npu：2.9.0.post2
- triton-ascend：3.5.1

## 复现方式

下面的脚本分别覆盖零 kernel callable、可能包含多个 kernel 的 callable，以及实际扫描中出现部分 shape 无有效计时的 `narrow_copy`。请同时检查返回值和保留目录中的 `kernel_details.csv`：

```python
import os

import torch
import torch_npu
from triton.backends.ascend.testing import do_bench_npu

torch.npu.set_device(0)
x = torch.randn((4096, 4096), dtype=torch.float16, device="npu:0")
x_small = torch.randn((64, 64), dtype=torch.float16, device="npu:0")

def view_only():
    return x.transpose(0, 1)

def composite():
    return torch.cos(torch.sin(x))

def copy_op():
    return torch.narrow_copy(x_small, 0, 16, 32)

for name, fn in {
    "view_only": view_only,
    "composite": composite,
    "narrow_copy": copy_op,
}.items():
    path = os.path.abspath(f"profile_{name}")
    latency = do_bench_npu(
        fn,
        warmup=0,
        active=1,
        prof_dir=path,
        keep_res=True,
    )
    print(name, latency, path)
```

需要重点核对：

1. `view_only` 没有 NPU kernel 时是否只返回 `inf`，且没有结构化原因。
2. `composite` 的返回 latency 是否等于本次调用产生的全部目标 kernel 的 `Duration(us)` 之和，而不是某一行的耗时。
3. 修改 `active` 或输入 shape 后，是否出现 CSV 行数与调用次数不一致、越界或 `inf`。

## 实测结果

我们使用相同的 `do_bench_npu` 调用方式对同一环境中的 193 个算子进行了完整扫描，每个算子设置 900 秒超时，使用 8 张 NPU 并发执行。结果为 118 个算子的全部用例获得有效计时，75 个算子未通过；其中有 53 个算子的 653 个用例明确得到了无效 latency，其余失败还包括算子 API/运行错误和超时，不归因于本问题。

代表性结果如下：

| 算子 | 无效 case / 总 case | 表现 |
| --- | ---: | --- |
| `fmax` | 15 / 15 | 底层明确 CPU fallback，`do_bench_npu` 返回 `inf`，下游记录为 `1.7976931348623157e+308` |
| `scatter_reduce_` | 20 / 20 | 调用完成但全部 case 无有效 NPU latency，未出现 CPU fallback 警告 |
| `narrow_copy` | 6 / 15 | 同一算子部分 shape 返回有限值，部分 shape 返回 `inf` |
| `permute_copy` | 3 / 9 | 同一算子部分 shape 返回有限值，部分 shape 返回 `inf` |
| `einsum` | 12 / 69 无效，随后失败 | `_collect_prof_result` 抛出 `IndexError: single positional indexer is out-of-bounds` |

另一次直接解析 `torch_npu.profiler` 产物的扫描中，193 个算子里有 54 个算子没有得到有效设备 kernel CSV，`einsum` 则表现为 69 个 case 中仅 57 个获得有效 kernel timing。这批数据包含 CPU fallback 和 view/no-kernel 等预期情况，因此并不意味着 54 个都是 profiler 缺陷；它说明 `do_bench_npu` 需要明确处理这些合法的零 kernel 场景，而不能依赖固定行数并统一退化为 `inf` 或越界。

## 期望行为

- `do_bench_npu` 不应假设每次 callable 调用只对应一条 kernel 记录。
- 对一次调用产生多个 NPU kernel 的情况，应按 invocation/step/correlation 信息聚合该调用的全部目标 kernel 耗时。
- 对零 NPU kernel、CPU fallback、profiler 未产出文件和产物解析失败，应返回不同的明确状态或抛出包含原因的专用异常，不应静默返回 `inf`。
- CSV 记录数量与预期不一致时不应出现裸 `IndexError`；错误信息至少应包含 profiler 路径、预期调用次数、实际记录数和筛选条件。
- 若当前 profiler 产物不足以把 kernel 可靠归属到每次 callable invocation，希望社区说明推荐的关联字段或官方计时方式。

## 建议修复方向

建议使用 profiler 的 step、correlation ID 或等价边界把记录按实际 invocation 分组，再对每个 measured invocation 内的目标 NPU kernel 求和；warmup 应按 invocation 丢弃，而不是按 CSV 行数丢弃。若当前 `kernel_details.csv` 缺少可靠关联字段，可以在 profiling 采集阶段为每次 invocation 增加可识别的 record function/step 标记，或提供只计量指定 kernel 集合的官方接口。

对于确实不启动 NPU kernel 的调用，建议新增类似 `NoNpuKernelError(reason="view_only" | "cpu_fallback" | "no_profiler_record")` 的明确结果，并保留 profiler 目录供排查。这样上层 benchmark 可以选择跳过、改用其他计时语义或报告不支持，而不会把最大浮点数当成正常 latency。

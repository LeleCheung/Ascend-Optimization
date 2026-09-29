# 多设备实验结果整理手册

本文记录本地 `kernel_todo_v1/` 的结果汇总流程。该目录用于保存实验输入、汇总产物
和候选代码，已由 Git 忽略且不随仓库分发。使用本流程前，必须从授权的结果归档恢复
所需文件，或通过扫描命令在本地重新生成；不要把本地实验数据提交到源码仓库。

配套工具：

```text
scripts/experiments/organize_kernel_todo_results.py
```

该工具只读取 JSON 和代码文本，不导入候选，也不执行 Torch、Triton 或远端 Eval。
其 `--manifest` 默认值为本地 `kernel_todo_v1/result_sources.json`，也可以显式传入
其他路径。

## 1. 收录边界

只收录正常完成的 BatchSimpleOpt 任务：

- `PASSED`：轮数大于 0，存在数值型 `best_geo_mean` 和非空 best code。
- `FAILED`：轮数大于 0，最后一轮是 `PARTIAL_PASS`，有实际 workload 结果和最后
  candidate，但始终没有全量正确候选。这类任务保留最后一轮代码，标为 `last`，
  不能称为 best。

以下情况全部排除，不写入 `result_sources.json`：

- 厂商 API 不支持或 workload 本身不可执行。
- `rounds=0`，没有生成或评测候选。
- JumpServer/SSH 网关、鉴权、模型额度或 Agent 输出解析错误。
- 最后一轮出现 `RUNTIME_ERROR`、`TIMEOUT`、`SUSPECTED_DEVICE_ERROR` 或
  `SERVER_STATUS_FAILED`。
- 缺少 `.ledger.json` 或 `optimize_definition_output.json` 的不完整任务。

判断依据必须是 per-definition 的 `.ledger.json` 和
`optimize_definition_output.json`，不能只看 batch 进程退出码或目录是否存在。

## 2. 扫描新实验

在结果回收机的 KernelGen 仓库中执行：

```bash
python3 scripts/experiments/organize_kernel_todo_results.py scan \
  --device tianshu \
  --run-root runs/<run-name>/tianshu \
  --model 'deepseek-v4-pro[1m]' \
  --tested-at 2026-08-04 \
  --output /tmp/tianshu-result-scan.json

python3 -m json.tool /tmp/tianshu-result-scan.json >/dev/null
```

扫描结果分为：

- `included`：满足正常完成口径的候选条目。
- `excluded`：不完整或基础设施/API 中断，并保留排除原因。
- `duplicates`：同一设备、同一算子存在多个正常结果。

脚本不会自动解决重复实验。出现 `duplicates` 时，人工比较模型、日期、运行配置、
计时策略和原始 ledger，明确选择一条；禁止简单按加速比最大值覆盖，因为不同版本、
计时策略或 workload 可能不可比。

确认后，将选中的条目加入本地 `kernel_todo_v1/result_sources.json` 对应设备的
`results` 数组，并填写 `review`、`analysis` 和 `next_step`。每个设备每个算子只能
保留一条当前结论。

## 3. best/last code 选择

- 正常 `PASSED`：从 `ledger.best_code` 取代码，分析 `best_round` 对应 evaluation。
- 正常跑满但 `FAILED`：从 `ledger.rounds[-1].solution.code` 取最后候选，分析最后
  evaluation 的失败 workload。
- 不得用最后一轮覆盖一个更早的正确 best，也不得把不正确的最后候选放入
  `codes/` 后标成 best。

原始路径统一记录为仓库相对路径：

```text
runs/<run-name>/<device>/.../<definition>/.ledger.json
runs/<run-name>/<device>/.../<definition>/optimize_definition_output.json
```

`runs/` 通常是本地大文件且不提交 Git。因此原始链接主要用于结果回收机追溯；真正
需要长期保留的代码应进入授权的实验归档，而不是源码仓库。

## 4. 静态初析

`Server is_hack=false` 只是当时策略未命中，不能代替代码审查。至少检查：

- 是否直接引用 `gen_inputs` 的数值范围、固定 boundaries、固定 shape 或测试数据。
- 是否删掉了当前 workload 不会经过的通用算法分支。
- 是否通过 `torch.ops`、`torch.linalg`、`torch.sort` 等完成目标计算。
- 是否包含 `.cpu()`、`.numpy()`、`.item()` 等 host sync。
- Reference 是否只是 view/元数据操作，而 candidate 实际分配并物化输出。
- 极端加速比是否来自 reference 路径开销、计时异常或 workload 特化。

低加速结果优先按以下方向归因：

- `addmm`、`linear`、卷积、Attention：厂商库基线已经高度优化，检查 tensor-core、
  implicit-GEMM、tile、layout、`num_warps` 和 shape/dtype 分桶。
- `histc`、`nonzero`、索引、归约：检查原子冲突、scan/prefix-sum、中间张量和
  host sync。
- `broadcast_to`、`conj`、`unsqueeze_`：先判断 Reference 是否为 view；语义成本
  不对称时不应继续用物化 Triton kernel 追加速比。
- Elementwise/RNG：检查 kernel launch、转换、临时张量、向量宽度和 RNG 状态。

没有正确 best 的任务只做静态分析：列出最后一轮失败 workload、误差类型、代码
结构和可能修复方向，不为整理结果重新执行代码。

## 5. 生成和校验

准备并审查本地 manifest 后执行：

```bash
python3 scripts/experiments/organize_kernel_todo_results.py render
python3 scripts/experiments/organize_kernel_todo_results.py verify
git diff --check -- scripts/experiments/organize_kernel_todo_results.py
```

`render` 在本地 `kernel_todo_v1/` 生成或更新：

- 每个设备的 `README.md`、`results.md` 和 `analysis.md`。
- 每个算子的 `codes/<operator>.py`。

`verify` 会检查：

- manifest 中的任务仍满足“正常完成”口径。
- operator、轮数和三位小数加速比与 ledger 一致。
- best/last code 选择正确，已整理代码与 ledger 完全相同。
- 每个设备没有重复算子，`codes/` 没有缺失或多余文件。

本地 `kernel_todo_v1/README.md` 还包含 Todo 和 Reference 总数，这些数字不属于单次
workflow artifact；新增结果后需要按 manifest 统计手工更新，并再次核对设备合计。

## 6. 经验与约束

- 不能用“目录存在”判断完成；实践中存在 `rounds=0`、网关失败但生成了 output 的
  任务。
- 正常 `FAILED` 与基础设施 `FAILED` 必须分开。正常跑满的数值失败可以保留 last
  code，API/网关中断必须排除。
- 同一算子存在多个正常结果时，必须人工比较配置和来源，不能按最大加速比自动选择。
- `is_hack=false` 不能取代静态审查；仍需检查 workload 细节引用、Torch 计算、
  host sync 和固定 shape 分支。
- 每条结果同时保留代码、ledger 和 workflow output 的归档位置，才能区分真实性能
  问题、数值未通过和基础设施失败。

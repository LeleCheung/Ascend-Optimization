# V6 Schema 与 FlagGems Adapter 规范

> FlagGems 上游需要提供的接口、当前 POC 和 master 合入建议，见计划中的上游文档
> `flaggems_kernelgen_integration.md`。本文保留 V6
> Schema、Server adapter 合同和详细设计记录。

## V6 的新含义

V6 不再尝试把所有 pytest 翻译成一套通用 Definition/Workload DSL。它先定义一份
所有 evaluator 共用的纯算子 Definition，再允许不同 runner 使用各自的评测资产，
最终共享一个结果协议：

```text
                         ┌─ native reference/Workload ── simplified runner ─┐
common V6 Definition ────┤                                                  ├─ EvaluateResponse
                         └─ FlagGems native pytest ───── framework adapter ─┘
```

这样做的判断边界是：

- 评测只有稳定的输入 recipe、reference、correctness 和 timing 时，使用简化 V6；
- 评测依赖框架 fixture、参数化、skip、特殊断言、framework object 或复杂 benchmark
  harness 时，使用框架 adapter，不再把这些机制重新实现一遍。

Agent 只依赖公共 `EvaluateResponse`，优化循环、ledger、best code 和结果汇总不因 runner
不同而分叉。

## Agent/MCP 与 adapter 边界

Coder 不选择 runner，也不构造 Server 请求。Workspace 的 `ToolContext` 只保存
`catalog_name + definition` binding；该配置是 orchestrator 拥有的权威状态，不是
MCP 调用参数。Server 从 Catalog manifest 读取 evaluator 并完成内部路由。

```json
{"catalog_name": "kernelbench-v6", "definition": "add"}
```

框架 adapter 使用同一个内部绑定点：

```json
{"catalog_name": "flaggems-adapter-definitions", "definition": "addmm_"}
```

`evaluator` 由 Catalog manifest 决定，不复制为模型可编辑字段。公开 operator 恒等于
`Definition.name`，benchmark level 也是 Catalog 级配置，不在每个 binding 中重复。

Coder 始终只看到稳定的 workspace-bound MCP 工具：

```python
preflight_kernel(kernel_path="tmp/main.py")
eval_round(experiment_plan={...}, kernel_path="tmp/main.py")
```

`experiment_plan` 必填，`kernel_path` 默认为 `tmp/main.py`。runner、Definition、
Workload、Server URL、设备和计时策略均不暴露为模型可填字段。
MCP 读取 `ToolContext` 后只提交已绑定的 Catalog/Definition 和 candidate：

```text
preflight_kernel / eval_round
              │
              ▼
     workspace ToolContext
              │
              │
              ▼
     Catalog EvaluatorBinding
              │
       ┌─────┴─────┐
       ▼           ▼
     native      FlagGems adapter
       └─────┬─────┘
              ▼
       common EvaluateResponse
```

Server 按 binding 内部分派 adapter，不定义 `SuiteEvaluateRequest`或
`/evaluate_suite`。统一的是 MCP 工具面和返回协议，不是迫使所有
runner 共用一个巨大的请求 union。`eval_only` 同样只接收 binding 和 candidate，
由 Server 内部选择 evaluator。

## 公共 V6 Definition 与 Native evaluator

Definition 只保存：

- 真实有序 ABI：参数名、顺序、`kind`、必填状态和默认值；
- 可选的非权威 `type_hint` 字符串，只用于 Coder 理解参数；
- 逻辑输出名；
- 简单 `mutates` 与 `returns_alias_of`；
- 面向 Coder 的稳定算子描述。

```json
{
  "api_version": "v6.0",
  "name": "addmm_",
  "description": "In-place matrix multiply-add with keyword-only alpha and beta.",
  "parameters": [
    {"name": "self", "required": true, "type_hint": "Tensor"},
    {"name": "mat1", "required": true, "type_hint": "Tensor"},
    {"name": "mat2", "required": true, "type_hint": "Tensor"},
    {"name": "beta", "kind": "keyword_only", "required": false, "default": 1, "type_hint": "Number"},
    {"name": "alpha", "kind": "keyword_only", "required": false, "default": 1, "type_hint": "Number"}
  ],
  "outputs": ["out"],
  "effects": {
    "mutates": ["self"],
    "returns_alias_of": {"out": "self"}
  }
}
```

`parameters` 可以携带 `Tensor`、`List[Tensor]`、Tuple、device、Generator、Optional
或其他任意 `type_hint` 字符串，但 V6 不解析这些表达式，也不据此生成输入或校验
candidate annotation。只有参数名、顺序、`kind`、`required` 和 `default` 属于强校验
ABI。`type_hint` 只用于 prompt/文档；没有可靠源码 annotation 时省略，抽取 Agent
不得自行猜测。复杂类型和任意长度容器仍由可信输入生成代码或框架 pytest 处理。

V6 只为真实公开 ABI 保留一个受限的 `var_positional` kind，用来表达
`broadcast_tensors(*tensors)` 这类入口；它不增加元素类型 DSL，输入仍由 native
`gen_inputs` 或框架 pytest 物化。V6 不接受未解析的 `**kwargs`，也不支持条件 effects、
预期异常或 Workload `call` 表达式。若 FlagGems 实现只是用 `*args/**kwargs` 包装一个
固定 ATen schema，Definition 必须记录固定 schema，而不是把实现细节当成公开 ABI。

Definition 不携带 `reference`、`reference_device`、`gen_inputs`、`valid`、Workload、
pytest 路径、dispatcher key 或 adapter kind。这些是 evaluator 的执行资产，不是算子
契约。这样 KernelBench、FlashInfer-bench 和 FlagGems 抽取得到的 Definition 使用
完全相同的 schema。

KernelBench/FlashInfer-bench 的 native evaluator 在 Catalog 中把同一 Definition 与
独立的 native evaluation assets 绑定。assets 包含可信 reference 模块、
`reference_device`、correctness/timing Workload 和容差；它们不进入 Coder 可编辑的
Definition。

### Catalog 目录约定

一个 Catalog 只允许一种 evaluator。Server 递归扫描 `definitions/`，并按相同相对
路径和文件 stem 自动关联 native assets；manifest 不保存逐算子列表、路径或数量。

Native Catalog：

```text
kernelbench-v6/
├── manifest.json
├── definitions/
│   └── gemm/addmm_.json
├── references/
│   └── gemm/addmm_.py
└── workloads/
    └── gemm/
        ├── addmm_.correctness.jsonl
        └── addmm_.timing.jsonl
```

```json
{
  "api_version": "v6.0",
  "evaluator": "native"
}
```

路径解析规则固定为：

```text
definitions/<relative>.json
references/<relative>.py
workloads/<relative>.correctness.jsonl
workloads/<relative>.timing.jsonl
```

`reference_device` 由可信 reference 模块声明：

```python
REFERENCE_DEVICE = "target"
```

Workload 数量由 Server 读取 JSONL 后统计，不写入 manifest。缺失同名 reference 或
必需 Workload 时 Catalog 加载直接失败。

Workload 只保存名称、JSON generator context、seed 和可选容差：

```json
{
  "name": "case-0",
  "inputs": {
    "shape": [16, 128],
    "dtype": "float16",
    "num_pages": 32
  },
  "seed": 0
}
```

普通 Tensor/标量可继续使用 `random`、`scalar`、`literal` recipe。native reference
模块可固定导出：

```python
def gen_inputs(ctx, device):
    # Tensor List、变长 Tuple、device、Generator、框架对象在这里生成。
    return {"actual_parameter": ...}

def run(actual_parameter, ...):
    ...

def correctness(actual_parameter, ...):
    # 可选；必须与 run ABI 完全一致。
    ...

def valid(ref_outputs, sol_outputs, inputs, ctx):
    # 可选固定校验入口。
    ...
```

Server 根据 Definition 参数顺序直接调用 native reference 和 candidate，不执行
`eval(call)`。candidate 仍固定导出 `run`，以兼容既有轨迹和生成格式；Server 严格
检查 reference/candidate 的参数名、顺序、kind、默认值与 Definition 一致。

### Native 加载 Workload 的目标

Native adapter 从 Catalog 加载 Definition、reference assets 和
correctness/timing Workload，是为了构造一份
确定、可重现的评测计划：

- 枚举需要生成输入和执行的 case，并区分 correctness 与 timing；
- 与 candidate、Definition 及 settings 一起构造精确的 preflight/evaluate 请求；
- 绑定 preflight receipt 与正式 eval，防止两个阶段使用不同 case；
- 将 `per_workload` 结果、timing headline 和 profile case 对齐到稳定身份。

Workload 不是给模型的 prompt，也不是框架测试的通用中间语言。它只属于
native adapter；FlagGems、SGLang 或 vLLM adapter 应保留自身的 case 发现与执行
机制。

## FlagGems adapter

FlagGems 对接所需接口、当前 POC、master 合入判断和修改建议，统一见计划中的上游文档
`flaggems_kernelgen_integration.md`。本文后续只定义
各 adapter 共用的 Case List 和 Server 合同。

FlagGems 内置 pytest 接口通过 `--candidate-code-path <file>` 加载待测文件的 `run()`，通过 `--candidate-report-path <file>` 输出实际绑定算子及按 pytest node/performance case 统计的调用覆盖。调用覆盖用于 correctness 和 preflight；正式 performance 只传待测代码路径并用 benchmark 报告确认路由，避免把逐调用计数开销带入 latency。KernelGen Server 的 FlagGems adapter 只负责物化待测文件、构造上述参数并解析报告，不再携带私有 pytest 注入 plugin，也不再修改 FlagGems dispatcher 注册表。

## 统一 Case List Schema

Native 与框架 adapter 必须向 Server 通用层提供相同的逻辑能力：

```python
list_timing_cases() -> CaseList
evaluate_all() -> EvaluateResponse
build_profile_command(case_id: str) -> ProfileCommand
```

`list_timing_cases()` 由 `inspect().case_list` 暴露；`evaluate_all()` 对应完整
`evaluate()`，不接受 case 过滤。选择性执行只用于 profiling。这里约定的是
公共语义，不要求 Native 与 FlagGems 使用相同底层 runner。Agent/MCP 只看
公共 timing `case_id` 和语义信息，不接触 native Workload name 或 pytest nodeid。

公共返回格式固定为 `kernelgen.case-list/v1`：

```json
{
  "schema_version": "kernelgen.case-list/v1",
  "adapter_kind": "flaggems",
  "operator": "addmm_",
  "benchmark_fingerprint": "sha256:...",
  "cases": [
    {
      "case_id": "benchmark/test_addmm_.py::test_addmm_::core::float16::0",
      "phase": "timing",
      "ordinal": 0,
      "profile_eligible": true,
      "dtype": "torch.float16",
      "shape": {"b": 16, "m": 4096, "n": 4096, "k": 4096},
      "params": {"b_column_major": true}
    }
  ]
}
```

约束如下：

- `case_id` 在 `benchmark_fingerprint` 内唯一，是 Agent 选择、profiling、ledger 和
  `EvaluateResponse.per_workload[].uuid` 共用的身份；即使 shape 相同也不得去重；
- `dtype`、`shape` 和 `params` 原样描述 benchmark 在生成 Tensor 前已有的循环坐标；
  不要求 adapter 将其展开成每个 ABI Tensor 的 shape；
- 公开 ABI、参数名、默认值和 mutation 只读取公共 Definition，不在 Case List 中重复；
  非默认动态调用值若属于 benchmark 枚举，也作为普通 `params` 保留；
- `b_column_major` 属于原生 benchmark 循环参数，不属于 `addmm_` ABI；
- Case List 是描述和选择协议，不是输入重建 DSL。List Tensor、变长 Tuple、fixture、
  随机数和 framework 对象继续由各 adapter 的原生 runner 处理；
- Native adapter 可以把 `case_id` 直接定义为 Workload name；FlagGems adapter 必须
  原样透传 FlagGems `case_id`，不得再合成第二个 ID 或解析其中的 pytest 信息。

Native 不需要 pytest。它直接从 Catalog 中已有的 timing Workload 生成
上述 Case List。FlagGems 从原生 benchmark family 的 case 坐标生成同一格式，并在
profiling 时直接接受同一个 `case_id`。
两者的 Definition schema、reference 资产和输入生成实现均不因此改变。

## Adapter 最小合同

V6 只要求 adapter 提供稳定 case ID、preflight 和正式评测；profiling 是可选
能力。所有 adapter 共用公共 Definition，但不要求共用 `EvaluationBundle`、Workload
或 pytest 表示。初步内部接口为：

```python
class EvaluatorAdapter:
    def inspect(self) -> AdapterManifest: ...
    def preflight(self, request: BoundEvaluateRequest) -> PreflightResult: ...
    def evaluate(self, request: BoundEvaluateRequest) -> EvaluateResponse: ...

    # 只在 manifest.capabilities.profile == True 时必须实现。
    def build_profile_command(
        self, request: ProfileRequest, options, artifact_dir, device
    ) -> ProfileCommand: ...
```

| 接口 | 要求 | 职责 |
| --- | --- | --- |
| `inspect` | 必须 | 返回 candidate 合同、能力、benchmark fingerprint 和完整有序 timing case |
| `preflight` | 必须 | 检查精确 candidate 是否可进入正式评测，不计时、不写 round |
| `evaluate` | 必须 | 执行全部绑定 case，返回公共 `EvaluateResponse` |
| `build_profile_command` | 可选 | 只当 `capabilities.profile=true` 时实现，为一个 `case_id` 构造精确 profiler 命令 |

Adapter 返回的 `PreflightResult` 使用最小公共格式：

```json
{
  "status": "PASSED",
  "stage": "complete",
  "log": ""
}
```

`status` 至少覆盖 `PASSED`、`FAILED`、`TIMEOUT` 和
`SUSPECTED_DEVICE_ERROR`；失败细节放在 `stage/log`，不为每个框架扩展
公共字段。receipt 由通用层在 `PASSED` 后生成，不由 adapter 自行写入。

`inspect()` 同时完成 timing case 枚举和 benchmark 身份固定；正式 adapter
必须在执行前返回完整有序 `kernelgen.case-list/v1`，不能用一次
evaluate 的 report 事后补齐。correctness case 不进入该列表。
`AdapterManifest` 只在外层增加 candidate contract 和 capability，例如：

```json
{
  "kind": "flaggems",
  "adapter_version": "1",
  "benchmark_fingerprint": "sha256:...",
  "candidate_contract": {
    "entrypoint": "run",
    "signature": "(self, mat1, mat2, *, beta=1, alpha=1)"
  },
  "capabilities": {
    "preflight": true,
    "profile": true
  },
  "case_list": {
    "schema_version": "kernelgen.case-list/v1",
    "adapter_kind": "flaggems",
    "operator": "addmm_",
    "benchmark_fingerprint": "sha256:...",
    "cases": ["...见上一节..."]
  }
}
```

格式约束如下：

- `case_id` 在一个 benchmark fingerprint 内全局唯一，是 MCP、ledger 和
  `EvaluateResponse` 使用的公共身份；
- `phase` 固定为 `timing`；
- `dtype/shape/params` 只用于展示和分析，不得作为重放选择器；ABI 和默认调用参数
  由公共 Definition 提供；
- `EvaluateResponse.per_workload[].uuid` 必须等于对应 `case_id`；字段名
  `per_workload` 为稳定响应协议，其语义是 adapter case；
- `benchmark_fingerprint` 覆盖 adapter binding、权威测试源和有序 case 列表。

Native adapter 的 `case_id` 是 timing Workload name；`inspect()` 从 Catalog 加载
Definition 和 timing Workload，并保留现有 `EvaluationBundle` 作为 native 内部类。

Pytest adapter 不能假设“一个 nodeid 就是一个 timing case”。例如 FlagGems
`benchmark/test_addmm_.py::test_addmm_` 只收集为一个 pytest item，但
`BlasBenchmark.run()` 在其内部遍历 3 种 dtype 和每种 5 个 shape。因此
FlagGems 必须生成包含 pytest 节点身份和内部循环坐标的全局 `case_id`，并由自身 CLI
把它定位到唯一 case。KGS 将该 ID 视为不透明字符串。FlagGems 必须提供两个对等能力：

```text
list_timing_cases()      -> 枚举 pytest item 内部的所有 case_id
run_case(case_id)         -> 精确执行一个全局 case_id
```

已经由 `pytest.parametrize` 展开的测试可以只使用完整 nodeid；在一个
test function 内部循环多个 shape/dtype 的 benchmark 必须额外提供
`case_id` 过滤。不使用 `-k` 模糊匹配，也不用 shape 单独作为身份；
shape 相同但源 metadata 不同的 case 仍必须拥有不同 `case_id`。

`preflight_kernel(kernel_path)` 的 MCP 名称和生命周期门禁通用，内部检查
不强行相同：native 负责 Definition/native assets，并只对全部 timing Workload 做
candidate-only smoke；FlagGems
按同一 Definition 校验 ABI，并对 `inspect()` 枚举的全部 core timing case
各执行一次 candidate。FlagGems preflight 不运行 correctness pytest、reference、
assertion、warmup 或计时，不发布 correctness 或 performance 结论。

通用层在 preflight 成功后仅保存最小一次性 receipt：

```json
{
  "status": "PASSED",
  "adapter_kind": "flaggems",
  "adapter_version": "1",
  "candidate_sha256": "...",
  "benchmark_fingerprint": "sha256:...",
  "service_signature": "sha256:..."
}
```

`eval_round` 重新执行 `inspect()` 并验证 receipt。candidate、benchmark 或 Server
身份变化都要求重新 preflight。通用层只负责分派、receipt、ledger、
`kernelgen.case-list/v1` 和公共结果，不引入通用 `PreparedEvaluation` 或用于重建
框架输入的 Workload schema。

### Profiling 边界

Native 继续使用 Workload 重放和现有 profiler。FlagGems 直接接受 timing `case_id`，
在自身 pytest harness 中定位 case、保留原输入生成，并在 profiler capture range 内
只调用 candidate。Server 负责 NCU/`msprof`、
timeout、artifact 和 report 归档，不把 pytest case 翻译成 native Workload。

FlagGems harness 已输出并接受稳定的全局 timing `case_id`；Server 原样保留该 ID，
无需维护 nodeid 映射或使用 `dtype + index` 合成身份。`--preflight-only`、
`--profile-only`、binding 分派、
receipt 和 profiler command 已接入；NCU/`msprof` 是否成功仍取决于目标机 profiler
资源和权限。

## 抽取 Agent 的职责变化

FlagGems extractor 只生成与 native 相同 schema 的公共 Definition：

1. 定位公开 callable，静态提取并强校验 Python ABI；有可靠源码 annotation 时原样
   保存为可选 `type_hint`，没有时省略而不推断；
2. 提取逻辑输出、简单 mutation/alias 和面向 Coder 的描述；
3. 保存必要的只读来源证据，供复核 Definition 是否来自目标 FlagGems revision；
4. 不生成 reference、gen_inputs、valid、逐条 Workload 或 adapter registry。

pytest 文件、marker、`core` 命令和待测代码直调覆盖检查属于 FlagGems 原生测试接口及 Server adapter 的确定性维护/验收流程，不属于 LLM 抽取 Agent。当前缺少正式 FlagGems API 时，可由独立扫描脚本生成 Server-owned 兼容映射；它不能修改 Definition。

List Tensor、变长 Tuple、device、Generator、framework fixture、custom valid 等复杂度均
留在 FlagGems pytest 内，不再映射成通用 schema 字段。

## 部署要求

FlagGems adapter 需要目标镜像预装固定 FlagGems checkout 所需依赖，并通过
`KGS_FLAGGEMS_ROOT` 指向源码树。KernelGen/Server 不自动安装或升级 Torch、Triton、
厂商运行时或 FlagGems 依赖。

所有启用同一 FlagGems evaluator binding 的设备必须使用相同的权威 FlagGems commit 和
tests/benchmark fingerprint；各设备的 FlagTree 和厂商运行时允许使用不同的兼容
版本。只比较 `pip` 版本字符串不够，Server 必须检查 checkout revision、测试源指纹
和工作树状态。不一致时返回 `ADAPTER_INCOMPATIBLE`，不能静默运行容器中另一版本的
pytest。

普通 KernelBench/FlashInfer-bench runner 不依赖 FlagGems；未安装 FlagGems 的目标镜像
仍可使用 native adapter；只有绑定 FlagGems Catalog 的请求会在 adapter
依赖缺失时失败。

当前 POC 实现位于 KernelGen Server：

- `agents/extractor/flaggems/`：生成并强校验公共 V6 Definition；
- `kernelgen_server/data/simple-v6-test`：native 自动配对 fixture；
- `data/flaggems-adapter-definitions/`：由 `kernel_todo` 去重得到的 170 份 FlagGems
  Definition Catalog；
- `kernelgen_server/evaluation/adapters/flaggems/`：FlagGems adapter、测试资产发现和原生 pytest CLI 调用。

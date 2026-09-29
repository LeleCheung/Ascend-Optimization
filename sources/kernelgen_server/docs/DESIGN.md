# KernelGen Server 设计

## V6 的边界

KernelGen Server 是可信 candidate 的设备执行与结果归一化边界。V6 先定义所有
evaluator 共用的纯算子 Definition，再允许不同评测源采用不同 runner，不要求所有
框架先翻译成同一套测试 DSL：

```text
                         ┌─ native evaluation assets ─ simplified runner ─┐
common V6 Definition ────┤                                                ├─ EvaluateResponse
                         └─ FlagGems native pytest ─── framework adapter ─┘
```

Definition 只包含有序 ABI、逻辑输出、简单 mutation/alias、稳定描述和可选的非权威
`type_hint`。它不包含 reference、输入生成、Workload、pytest 路径、dispatcher key
或 adapter kind。
这条边界把“算子契约”“测试语义归谁维护”和“Agent 如何消费结果”分开；Agent 始终
消费同一个 Definition 和扁平结果模型。

## Agent/MCP 边界与 evaluator 选择

Evaluator 由 workflow 在初始化 workspace 时选择，并作为权威 `ToolContext`
状态写入 workspace。Coder 既不选择 evaluator，也不提交 runner、
Definition、Workload、Server URL、设备或计时策略。

Coder 只看到两个稳定的 workspace-bound MCP 操作：

```python
preflight_kernel(kernel_path="tmp/main.py")
eval_round(experiment_plan={...}, kernel_path="tmp/main.py")
```

MCP 只提交 workspace 内已绑定的 Catalog/Definition 和 candidate。Server 根据
Catalog manifest 解析 `EvaluatorBinding`，再分派给 native 或 FlagGems adapter。
这个分派对模型不可见；所有 evaluator 最终返回同一
`EvaluateResponse`。V6 不定义 `SuiteEvaluateRequest`，也不提供
`/evaluate_suite`。

因此 V6 不定义同时包含 native 与所有 framework 字段的巨大请求 union。
Server 内部的多种请求 schema 可以不同；统一面是 MCP 工具合同、生命周期门禁、
ledger 和响应协议。

### Catalog 约束与自动配对

本节解释设计边界；面向 Catalog 作者的规范化必填/可选清单、文件模板和验收流程见
[v6.2 Native Catalog 接入规范](v6.2_native_catalog_authoring.md)。

一个 Catalog 只允许一种 evaluator。`manifest.json` 只保存 Catalog 级配置，不维护
逐算子 operator 列表、文件路径或 Workload 数量。v6.2 Server 扫描 per-operator
Native 资产：

```text
ops/[<group>/]<Definition.name>/definition.json
ops/[<group>/]<Definition.name>/oracle.py
ops/[<group>/]<Definition.name>/correctness.jsonl
ops/[<group>/]<Definition.name>/timing.jsonl
ops/[<group>/]<Definition.name>/correctness_full.jsonl  # optional inactive archive
ops/[<group>/]<Definition.name>/timing_full.jsonl       # optional inactive archive
ops/[<group>/]<Definition.name>/assets/...  # optional
```

Catalog 可以省略 `<group>`，也可以增加恰好一层分组目录。分组仅整理磁盘资产，
不参与 `Definition.name` 或 wire protocol 的 operator identity。

`oracle.py` 是唯一 source，提供共享 `run()`、可选的 `correctness_run()`/
`timing_run()` phase override 和可选 `torch_run()`。v6.0 catalog 仍递归扫描
`definitions/` 并自动配对：

```text
definitions/<relative>.json
references/<relative>.py
workloads/<relative>.correctness.jsonl
workloads/<relative>.timing.jsonl
```

oracle 需要 Python helper、共享库或配置时，只能放在同一算子的可选 `assets/` 子目录。
Server 拒绝算子目录中的其他同级文件和所有符号链接，导入时只临时暴露该算子目录与
`assets/`；资产路径和内容哈希进入 native benchmark fingerprint。这样依赖不会污染
其他算子的 import namespace，也不会依赖目标机绝对路径。

完整参数空间需要生成稳定子集时，原始 Workload 可以保存在可选的
`correctness_full.jsonl` 和 `timing_full.jsonl`，活跃样本仍使用无后缀文件名。
`*_full.jsonl` 不参与评测或 fingerprint；采样算法、上限和完整/活跃数量记录在
manifest 中。

Native manifest 的最小形式为：

```json
{"api_version": "v6.2", "evaluator": "native", "layout": "per-operator"}
```

当某个 timing reference 必须调用 FlagGems benchmark 自身的 baseline callable 时，该 native catalog 还声明 `framework="flaggems"`、仓库、来源分支和完整 `framework_revision`。adapter registry 在加载 oracle 前要求 FlagGems checkout HEAD 与该 revision 完全一致且相关源码工作区干净，并把该 checkout 的 `src` 置于导入路径首位；不能使用镜像中偶然安装的其他版本，也不能把目标机绝对路径写入 oracle。

固定 checkout 若存在已确认的厂商 harness 兼容问题，修复必须在 candidate 加载前由
可信 adapter/reference 初始化执行，并同时限定算子和厂商；candidate 不得修改 baseline
来使 benchmark 可运行。当前仅 MetaX `linear_backward` 有该例外：忽略配置中无语义的
`SPLIT_K=1`；其余 tuning 选择保持 FlagGems 原生行为。

FlagGems manifest 的最小形式为：

```json
{
  "api_version": "v6.0",
  "evaluator": "flaggems",
  "benchmark_level": "core"
}
```

FlagGems Adapter Catalog 不选择 Gems 版本：checkout 由 KG 实例或操作方通过 `KGS_FLAGGEMS_ROOT` 选择，旧 Adapter manifest 中的 framework repository/branch/revision 字段不再参与绑定。Adapter 读取实际 Git HEAD、校验相关源码工作树 clean，并将实际 commit、测试源码和枚举结果纳入 benchmark fingerprint；更新 commit 不要求仅为版本号重新生成 Adapter Catalog，但测试资产、ABI 和报告格式仍须兼容。一次 campaign 固定已选 checkout，不自动追踪分支，Profile 拒绝 inspect 后 fingerprint 改变的请求。上述放宽只适用于 evaluator=flaggems；前述 FlagGems-backed native oracle 仍要求其 Catalog 的 exact revision。

FlagGems Catalog 只有 `definitions/`，公开 operator 恒等于 `Definition.name`。
KernelBench、FlashInfer-bench、FlagGems 和其他 framework 使用独立 Catalog，不允许在
一个 manifest 内逐算子混用 evaluator。Workload 数量由 Server 读取 JSONL 后统计；
native reference 的 `REFERENCE_DEVICE` 常量保存在同名可信 Python 模块中。
`Definition.name` 在 Catalog 内必须唯一；缺失必需同名资产或发现无法配对的多余资产
时，Catalog 加载失败。

## Simplified workload runner

`POST /evaluate` 接收 `EvaluateRequest`。公共 Definition 描述有序 Python ABI、逻辑
输出和简单 mutation/alias；native evaluation assets 独立提供 reference 模块、
`reference_device`、Workload 和容差。候选固定导出 `run`。

参数的 `type_hint` 是任意字符串，只用于 Coder prompt 和文档。Server 不解析
`Tensor`、`List[Tensor]`、Tuple、device、Generator 或 Optional 表达式，不根据它
生成输入，也不检查 candidate 的 Python annotation。ABI 强校验只覆盖参数名、顺序、
`kind`、`required` 和 `default`；实际值由 native `gen_inputs` 或框架 pytest 负责。

v6.2 oracle 可以固定导出 `gen_inputs(ctx, device)`、同 ABI 的共享 `run(...)`、可选
`correctness_run(...)`/`timing_run(...)` phase override、可选 `torch_run(...)` 和
`valid(ref_outputs, sol_outputs, inputs, ctx)`。复杂 Tensor List、变长 Tuple、device、
Generator 或框架对象在可信 `gen_inputs` 中生成，不进入 schema 类型系统。
correctness 固定解析为 `correctness_run ?? run`，timing 固定解析为
`timing_run ?? run`；两种语义一致时只定义 `run()`，避免复制相同 reference code。
`inputs` 是按 Definition 参数名索引的映射。存在 `valid` 时，它替代 correctness
返回值和已声明 mutation 结果的逐值比较，以表达随机/统计类算子的 framework
规则；Server 仍强制返值和 mutation 的 PyTree 结构、leaf 类型、Tensor shape/dtype、
未声明 mutation 检查与 alias 契约。v6.0 的 `valid`
仍接收参数值列表，且只在默认逐值比较通过后执行。
少数 framework correctness 明确采用 `len + zip` 等非 PyTree 返回契约时，oracle
可用 `VALID_OWNS_RETURN_CONTRACT = True` 将返回值的结构、leaf、shape/dtype 和数值
检查一并交给 `valid`；默认仍保持上述严格门禁，mutation 与 alias 也不受该开关影响。
Torch fallback 使用对应的 `TORCH_VALID_OWNS_RETURN_CONTRACT` 与 `torch_valid`。

普通 random recipe 允许 v6.2 私有的 `device: "target"` 物化提示。它不进入公共
Definition 类型系统，只要求 native runner 在已分配的调用设备上直接生成输入；省略
时沿用 CPU base 后克隆的兼容行为。该提示用于忠实保留 framework benchmark 的设备
分配路径，不能填写物理 device id。

同一 recipe 可声明 `distribution`、`source_dtype` 和顺序 `transforms`，用于保留来源
benchmark 的均匀/整数采样、生成 dtype、三角/对称矩阵、数值变换和 RNG 调用顺序。
这些约束由公共 Workload runtime 执行，不为单个 Catalog 增加 evaluator adapter。

v6.2 Workload 可通过绝对 `input_path` 从 safetensors 按参数名直接物化输入 Tensor，
也可在 correctness 中通过绝对 `output_path` 按逻辑输出名加载 golden Tensor。后者
跳过对应 correctness reference，但不改变 candidate 的 mutation/alias 门禁；不能被
输出 alias 表达的 mutation 不允许使用该模式。timing 禁止 `output_path`，reference
latency 始终由 KGS 在目标设备计时，不从外部 metadata 读取。

正式 Evaluate 先对全部 workload 运行 primary readiness；任一失败时从头验证
`torch_run()`，成功后整轮统一使用 fallback。Candidate 错误不会触发切换。随后执行
严格 ABI 校验和公共结果聚合。correctness workload 分别物化 reference/candidate
输入并执行 correctness/effects 门禁，返回值与输入递归按 PyTree 处理。v6.2 timing
workload 与 framework benchmark 一致，只要求 reference/candidate 可执行并产生合法
目标计时，不比较输出、mutation 或 alias；v6.0 继续保留旧 timing gate。

Native 加载 Definition、reference assets 和 Workload，生成确定的评测计划：枚举
correctness/timing case，构造 preflight/evaluate 请求，绑定两个阶段的 case 身份，并将
`per_workload`、headline 和 profile case 对齐。Workload 是 native adapter 的内部
评测计划，不暴露给模型，也不作为框架 adapter 必须翻译到的通用 IR。

## FlagGems framework adapter

FlagGems 不需要 suite 对象。Server 从受信 Catalog 加载 Definition，使用
`Definition.name` 作为公开 operator，从 Catalog manifest 读取 `core`
benchmark level 和所需 FlagGems revision，然后由 adapter 按 pytest marker
确定性定位一个或多个 accuracy/benchmark 文件；共享文件仍用 marker 隔离，不能把
同文件其他算子的 case 混入结果。Definition 不保存 pytest 路径、dispatcher key 或
预期 case 数。

FlagGems adapter 的生命周期固定为：

1. `inspect()` 只枚举全部 `core` timing case，返回有序 Case List 和
   benchmark fingerprint；
2. `preflight()` 严格检查 candidate ABI，然后对每个 timing case 生成输入、
   调用 candidate 一次并同步；
3. `evaluate()` 完整执行 accuracy pytest 和 `core` benchmark pytest，并将
   两份报告归一为公共 `EvaluateResponse`；
4. `build_profile_command()` 把公共 timing `case_id` 原样传给 FlagGems
   `--case-id`，只在 profiler capture 内执行 candidate。

`preflight()` 不运行 correctness pytest，不执行 reference、assertion、warmup、
重复计时或加速比计算。这与 native preflight 的“candidate-only smoke”语义
一致，同时避免要求任意 correctness Python 测试拆分 reference/assertion。
correctness 和 performance 结论只由 `evaluate()` 产生。

Candidate 调用契约固定为 direct `gems_op`。correctness 和 timing 保留各自 pytest 的参数化、输入、reference/assertion 和计时/report，但都由 FlagGems 内置 `--candidate-code-path` 加载待测文件，并通过同一个 process-local resolver 直接执行。KernelGen 优化主路径不通过 `_FULL_CONFIG`、`use_gems()`、`FULL_CONFIG_BY_FUNC` 或 pytest 已缓存的 `self.gems_op` 选择 candidate。

公共和自定义 Benchmark 必须由 Benchmark 基类解析 candidate，correctness 必须通过 `resolve_gems_op()` 使用同一 callable。每个非 skip correctness nodeid 和每个 timing `case_id` 都必须报告 candidate 身份与调用覆盖；pytest 退出码、注册表匹配或全局总调用数不能单独作为执行成功的证据。Dispatcher 注册链路只在最终交付 smoke 中复核，不进入 KernelGen 优化评分主路径。

调用验证采用最小证据：correctness 只记录每个非 skip pytest `nodeid` 是否覆盖，
preflight 要求每个全局 `case_id` 恰好调用一次；正式 timing 不安装跟踪
wrapper，只校验 report 的 `candidate_source=override`；profiling 单独输出一个标量
调用数，用于核对 `warmup + iterations`。Server 不再为 correctness/preflight
维护全局总调用计数，这些内部证据也不进入公共协议。

### Candidate 加载、baseline 与注册

Server 必须把待测文件物化、FlagGems 原生加载、基线和框架注册作为四个独立阶段：

1. Server 只把请求中的 source 物化到隔离目录，并向 pytest 传入 `--candidate-code-path <entry>`；
2. FlagGems 内置 pytest plugin 导入 `run` callable，并在第一次 `resolve_gems_op()` 时绑定唯一公开算子；它不修改 `torch.ops`、`_FULL_CONFIG` 或 dispatcher；
3. baseline 由受信 FlagGems harness 固定，不从请求或 candidate 推导。概念字段为 `baseline_op`；现有实现即使仍命名为 `torch_op`，也可以指向 Torch、外部框架、组合 reference 或加载待测代码前保存的原始 FlagGems callable；
4. 缺失 Torch schema、backend kernel 和 custom namespace 的注册由固定 revision 的 FlagGems bootstrap/patch 负责。Server 不复制注册表，也不根据算子名猜测 schema、namespace 或 dispatch key。

candidate override 不能影响 baseline。尤其禁止先把 candidate 注册成 Torch Op，
再让 benchmark 把同一实现同时作为 baseline 和 candidate；这种执行不产生可信的
correctness 对比或 speedup。若 harness 无法提供独立可信 baseline，adapter 必须
fail closed 或明确返回无有效性能结论，不能用 candidate 自比较补齐结果。

主评分 direct `gems_op` 不依赖 dispatcher。`run_dispatcher_smoke()` 是独立的交付检查，
使用 FlagGems 原生初始化验证 ATen/Tensor/custom-op 调用链。FlagGems 已支持的缺失
Torch Op 由它自己注册；FlagGems 未提供注册的 Op 不由 `_load_candidate()` 临时补齐。
因此某个目标没有原生 Torch API 时，只要 harness 仍有独立 baseline，主路径可以继续；
只有交付要求 dispatcher API 时才必须额外通过 registration smoke。

### FlagGems 原生 harness 与 Server adapter 的接口边界

FlagGems 不实现 KernelGen 的 HTTP、ledger、receipt、设备队列或公共响应。它保留原生 pytest/benchmark 语义，并向 adapter 提供以下最小接口：

```python
resolve_gems_op(operator) -> Callable
pytest ... --candidate-code-path <file>
pytest ... --candidate-code-path <file> --candidate-report-path <json>
list_timing_cases(operator, benchmark_level="core") -> list[NativeTimingCase]
preflight_timing(operator, benchmark_level="core") -> NativeRunReport
run_correctness(operator, report_path=...) -> NativeRunReport
run_timing(operator, benchmark_level="core", report_path=...) -> NativeRunReport
run_profile_case(operator, case_id, options) -> None
run_dispatcher_smoke(operator) -> None
```

`NativeTimingCase.case_id` 只要求在对应 benchmark fingerprint 内全局唯一且稳定，
并保留原始顺序；它不是 native `Workload`。Server 原样保留该 ID。
`preflight_timing()` 必须走与
`list_timing_cases()` 相同的全部 case 序列，但每个 case 只物化并调用
candidate 一次。
`--candidate-code-path` 安装 correctness/timing 共用的 candidate；`resolve_gems_op()` 是测试内部唯一 candidate 解析入口。正确性、preflight 和 profiling 按需启用调用覆盖；正式 timing 直接计时 `run` callable，不安装逐调用计数 wrapper。
`run_dispatcher_smoke()` 只属于交付验收，不参与优化评分，并且只使用 FlagGems 原生
bootstrap/patch 建立的注册链路。

当前接入成本如下：

| 能力 | 当前支持 | 难度/侵入性 |
| --- | --- | --- |
| 完整 accuracy/performance 和 JSON report | 已支持 | 无新增实现，直接运行原 pytest |
| candidate override | FlagGems 已提供内置 `--candidate-code-path`；算子 pytest 分批接入 direct `gems_op` | 低；公共 Benchmark 已具备 resolver，correctness 按 nodeid、timing 按 case_id 验证覆盖 |
| baseline | 原生 benchmark 已持有 callable，部分字段仍名为 `torch_op` | 低；按 `baseline_op` 语义审查，保证它来自受信 harness 且不受 override 影响 |
| Torch/custom-op 注册 | 由固定 FlagGems checkout 的 bootstrap/patch 提供 | Server 不实现注册；只在独立 dispatcher smoke 中验证，缺失时明确失败 |
| timing 内部 case 枚举 | 公共 family 与 `kernel_todo` 涉及的 legacy/custom benchmark 已迁移 | 170 个算子的 marker 精确收集得到 2644 个 `core` case；新增自定义 loop 仍需实现两阶段接口 |
| timing 内部 case 精确选择 | 支持重复精确 `--case-id`；170 个 Definition 均已通过 KGS inspect 归一化 | 保留原 input factory，只增加 family 级 case 坐标枚举 |
| timing report 稳定 `case_id` | 已支持全局 ID，pytest nodeid 已编码在 ID 中 | Server 原样透传 `case_id` |
| timing candidate-only preflight | 已实现统一 `--preflight-only` | 复用两阶段 case 接口，不修改算子 pytest |
| profile command | 已实现精确 `--profile-only`，由 Server 包裹 profiler | FlagGems 不管理 profiler |

当前 V6 catalog 已完成 direct `gems_op` 迁移：correctness 按非 skip nodeid、
preflight/profile 按 `case_id` 检查 candidate 覆盖，正式 timing 通过报告中的
`candidate_source=override` 验证路由且不在计时区间加入跟踪 wrapper。Server 不再
提供 `_FULL_CONFIG`/dispatcher 注入回退；仍未迁移的 case 会被 candidate 覆盖检查
直接拒绝。这不改变 Definition、MCP 或公共协议。

2026-08-16 在 A100-SXM4-40GB、Torch 2.8.0+cu128、Triton 3.4.0 上对
direct FlagGems callable 与持久 `use_gems()` dispatcher 路径使用同一
`triton.testing.do_bench` 对比。`addmm_` 15 个 `core` case 的
`dispatcher/direct` 逐 case 几何平均为 `1.0066`；direct latency 小于
`0.2 ms` 的 8 case 中位差为 `+2.13 us`/`+1.98%`，大 case 差异在
噪声内。补测 `softmax 1024x1024` 的 dispatcher 约慢 `0.67 us`/
`5.0%`，而 `4096x4096` 无显著差异。因此 direct 调用是有意识的
工程取舍：大算子影响很小，短 kernel 可出现 `2%-5%` 的相对差异；
微小算子必须在最终 dispatcher benchmark 复核。该数值只是 A100 快照，
其他 backend 需要独立验证。

`build_profile_command(candidate, case_id, context, options)` 属于 Server 的 FlagGems
adapter，而不是 FlagGems 上游职责。它返回 `argv/cwd/env/capture`；Server 继续负责
设备 slot、隔离、timeout、`msprof`/NCU、artifact 和报告归档。命令必须调用
FlagGems 的精确单 case 重放入口，沿用原 input generator，并确保 profiler capture
range 内只执行 candidate。

旧 `suites/`、`SuiteEvaluateRequest`、`/evaluate_suite` 和为 `addmm_` 硬编码
Definition/ABI/case 数的 registry 已删除。当前实现只使用 Catalog
`EvaluatorBinding` 和 adapter 自动发现，不再维护第二份 ABI 真值。

## Adapter 最小合同

公共编排层不定义通用 `EvaluationBundle` 或通用 pytest 格式。每个 adapter
只需实现：

```python
class EvaluatorAdapter:
    def inspect(self) -> AdapterManifest: ...
    def preflight(self, request: BoundEvaluateRequest) -> PreflightResult: ...
    def evaluate(self, request: BoundEvaluateRequest) -> EvaluateResponse: ...

    # Optional; required only when the manifest advertises profile=true.
    def build_profile_command(
        self, request: ProfileRequest, options, artifact_dir, device
    ) -> ProfileCommand: ...
```

`inspect`、`preflight` 和 `evaluate` 是必须接口；`build_profile_command` 是可选接口，
只在 manifest 声明 `profile=true` 时实现。`preflight` 返回最小公共结果：

```json
{
  "status": "PASSED",
  "stage": "complete",
  "log": ""
}
```

`status` 至少覆盖 `PASSED`、`FAILED`、`TIMEOUT` 和
`SUSPECTED_DEVICE_ERROR`。Adapter 不写 ledger 或 receipt；通用编排层只在
preflight `PASSED` 后生成一次性 receipt。

`inspect()` 是唯一通用 case 发现入口，逻辑上提供
`list_timing_cases() -> kernelgen.case-list/v1`。返回值包含 adapter
kind/version、candidate 入口与签名、capability、benchmark fingerprint 和执行前
可得的完整有序 timing case 列表；不枚举 correctness nodeid，也不得依赖
一次 evaluate 的 report 事后补齐。`evaluate()` 始终执行全部绑定的
correctness 和 timing case；按 case 选择只属于 `build_profile_command()`。初步格式：

```json
{
  "kind": "flaggems",
  "adapter_version": "1",
  "benchmark_fingerprint": "sha256:...",
  "candidate_contract": {
    "entrypoint": "run",
    "signature": "(self, mat1, mat2, *, beta=1, alpha=1)"
  },
  "capabilities": {"preflight": true, "profile": true},
  "case_list": {
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
}
```

`case_id` 在一个 benchmark fingerprint 内全局唯一。`dtype/shape/params` 原样保存
原生 benchmark 在生成 Tensor 前已有的循环坐标，只用于 Agent 展示和选择，不负责
重建 Tensor。公开 ABI、默认参数和 mutation 由公共 Definition 提供，不在 Case List
重复保存。
公共响应仍保留
`per_workload` 字段，但每条 `WorkloadResult.uuid` 必须等于 adapter 返回的
`case_id`。

Native 从 Catalog Workload name 生成 Case List；FlagGems 生成包含 pytest 节点身份
和内部循环坐标的全局 `case_id`，并由自身 `--case-id` 完成定位。Server 不解析、不
重新编号，也不维护 `case_id -> native selector` 映射。两者都必须实现精确重放，
不使用 `-k` 或 shape 单独作为选择器。shape 相同的不同源 case 仍保留不同
`case_id`。

Native 与 FlagGems 共用 timing Case List Schema 和选择语义，但不共用输入
实现：Native 从 timing Workload 生成 Case List，FlagGems 调用原 benchmark
input factory。List Tensor、变长 Tuple、fixture、随机数和 framework 对象不进入
Case List Schema。case 枚举只需
提供生成 Tensor 前可知的 dtype、原始 shape 坐标和其他循环参数；真实 Tensor 仅在
选中 case 后由对应 runner 创建。

Workspace evaluator binding 使用稳定外壳和 adapter-owned `config`：

```json
{
  "evaluator": {
    "kind": "flaggems",
    "adapter_version": "1",
    "config": {
      "definition": "addmm_",
      "catalog_name": "flaggems-v6"
    }
  }
}
```

`kind` 是 workspace 初始化时从 Catalog manifest 复制的权威值。单算子 binding 不再
重复 operator 或 benchmark level。

Native 保留现有 `EvaluationBundle` 作为内部类；FlagGems 保留 pytest collection
和自己的全局 case 选择实现。两者共用 Definition，但 native assets 与
FlagGems pytest 不相互转换。Native preflight 只对全部 timing Workload 执行
candidate-only smoke；FlagGems preflight 按 Definition 校验 ABI，并对
`inspect()` 枚举的全部 core timing case 各执行一次 candidate。FlagGems
preflight 不收集或运行 correctness pytest，不做 assertion 或计时。

通用层只保存 `adapter_kind`、`adapter_version`、candidate SHA-256、
`benchmark_fingerprint` 和 Server `service_signature` 构成的一次性 receipt。
`eval_round` 重新执行 `inspect()` 并验证这些字段，不引入通用
`PreparedEvaluation` 或用于重建框架输入的 Workload schema；通用层只保存
`kernelgen.case-list/v1` 的描述和身份字段。

Profiling 是 adapter capability。Native 保持 `profile=true` 和现有 Workload
重放。FlagGems 实现 `build_profile_command()` 后也声明 `profile=true`：它把 timing
`case_id` 原样传给 FlagGems `--case-id`，使用原 benchmark 物化输入，
并确保 profiler capture range 内只调用 candidate。Server 继续负责 NCU/
`msprof`、timeout、artifact 和 report 归档，不为 profiling 把 pytest case 翻译成
native Workload。

## 公共执行基础设施

两种 runner 共用设备 slot、线程队列、非 daemon spawn 隔离、请求审计、超时处理、
强健康探针和 `EvaluateResponse` 聚合。每张可见设备同一时刻最多运行一个 Eval、
Profile 或 Debug Job；`--max-workers` 可以大于设备数，但不会复制设备 token。

请求 `TIMEOUT` 或隔离 worker 异常退出后，当前 slot 进入 `checking` 并在同一设备上
运行强探针。探针通过则恢复 slot 且不跨卡重试；失败才标记 `broken` 并返回
`SUSPECTED_DEVICE_ERROR`。普通 candidate `RUNTIME_ERROR` 不触发设备探针。

## 计时

非昇腾正式评测使用 `triton.testing.do_bench`；昇腾正式评测使用严格
`torch_npu.profiler`，不回退 wall time。简化 runner 由 device adapter 执行计时；
FlagGems adapter 复用原 benchmark 自己的计时实现并强制 `--level core`。

## 安全与依赖

candidate、reference、native pytest 和 Debug Job 都是可信代码，不是安全沙箱。Server
只应绑定 loopback 并仅接收可信客户端。

Server 不安装 PyTorch、Triton、厂商运行时或框架依赖。启用 FlagGems adapter 的镜像
必须预先具备固定 FlagGems checkout 及其依赖；`KGS_FLAGGEMS_ROOT` 用于选择源码树。
所有设备必须使用同一个权威 FlagGems revision 和 tests/benchmark fingerprint；厂商运行时可以按设备使用不同的兼容版本。Server 不以 Python 包版本字符串
推断一致性，而是校验 checkout commit、测试源指纹和工作树状态。缺失、dirty 或不匹配
时 adapter fail closed，返回 `ADAPTER_INCOMPATIBLE`，不得继续执行其他版本的 pytest。

## 主要源码

| 路径 | 作用 |
| --- | --- |
| `protocol/schema.py` | 公共 Definition、binding、case 和响应 schema。 |
| `evaluation/engine.py` | 简化 Workload runner。 |
| `evaluation/workload_runtime.py` | Workload 输入物化、ABI 调用绑定与 `gen_inputs`。 |
| `evaluation/result.py` | 两种 runner 共用的权威聚合。 |
| `evaluation/adapters/base.py` | evaluator adapter 生命周期合同。 |
| `evaluation/adapters/registry.py` | 按 Catalog evaluator 类型分派 adapter。 |
| `evaluation/adapters/flaggems/adapter.py` | FlagGems inspect/preflight/evaluate/profile 和报告归一化。 |
| `evaluation/adapters/flaggems/discovery.py` | 按 Definition 名和 pytest marker 确定性发现一个或多个 pytest 资产。 |
| `runtime/isolated.py` | spawn 隔离及 runner 路由。 |
| `runtime/device_pool.py` | 设备独占、排队、探针与 slot 状态。 |
| `kernelgen_server/data/simple-v6-test` | 当前旧 POC fixture；需从逐算子 manifest/内嵌 reference 迁移到目录自动配对。 |
| `data/flaggems-adapter-definitions` | `kernel_todo` 当前 170 个唯一算子的 Definition-only Catalog；评测由 FlagGems adapter 执行。 |
| `data/flaggems-native` | 原生逐算子 FlagGems v6.2 Catalog。 |
| `data/kernelgenbench` | 已迁移的 KernelGenBench v6.2 Catalog。 |
| `data/.old` | 当前 Server 不支持的历史 Catalog；只用于迁移和溯源。 |

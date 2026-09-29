# 数据协议 v6.2

V6 的含义是“多种 evaluator，共享绑定、case 发现和结果协议”。Agent 不上传
framework pytest，也不把复杂测试翻译成通用 Workload DSL。

面向算子/Catalog 提供方的目录模板、必填与可选资产、oracle hook、Workload recipe、
交付证据和验收检查表统一见
[v6.2 Native Catalog 接入规范](v6.2_native_catalog_authoring.md)。本文保留 wire model
与 evaluator 语义说明，不重复维护接入清单。

## 公共入口

所有 evaluator 共用四个入口：

- `POST /inspect`：返回 ABI、adapter capability、benchmark fingerprint 和全部
  timing case；
- `POST /preflight`：严格检查 ABI，并让 candidate 在每个 timing case 上完整执行
  一次；不执行 reference、assertion、correctness pytest 或计时；
- `POST /evaluate`：执行 evaluator 的完整 correctness 与 performance 流程；
- `POST /profile`：按 `case_id` 精确重放一个 timing case，并由 Server 统一管理
  profiler、超时和产物。

请求通过同一个 binding 选择受信 Catalog 记录：

```json
{
  "catalog_name": "flaggems-adapter-definitions",
  "definition": "addmm_"
}
```

Catalog manifest 的 `evaluator` 决定内部使用 `native` 或 `flaggems` adapter。Agent、
MCP 和 Coder 不传 adapter 类型，也不需要知道不同 runner 的私有请求字段。

## Definition

Definition 是 prompt 与 candidate 的公共 ABI，不携带 pytest、reference source 或
Workload：

```json
{
  "api_version": "v6.2",
  "name": "addmm_",
  "parameters": [
    {"name": "self", "required": true, "type_hint": "Tensor"},
    {"name": "mat1", "required": true, "type_hint": "Tensor"},
    {"name": "mat2", "required": true, "type_hint": "Tensor"},
    {
      "name": "beta",
      "kind": "keyword_only",
      "required": false,
      "default": 1,
      "type_hint": "Number"
    },
    {
      "name": "alpha",
      "kind": "keyword_only",
      "required": false,
      "default": 1,
      "type_hint": "Number"
    }
  ],
  "outputs": ["out"],
  "effects": {
    "mutates": ["self"],
    "returns_alias_of": {"out": "self"}
  }
}
```

`parameters` 是唯一 ABI。`kind` 支持 `positional_only`、
`positional_or_keyword`、`var_positional` 和 `keyword_only`；可选参数必须显式保存默认值。
`type_hint` 只给 Agent 阅读，不驱动 Server 的类型分派。V6 不为 Tensor、
`List[Tensor]`、任意 Tuple、device 等继续增加类型 DSL。`var_positional` 只表达真实的
`*args` 调用边界，不解析元素类型；未解析的 `**kwargs` 不被接受。

## Native evaluator

v6.2 Native Catalog 使用 per-operator 资产：

```text
ops/[<group>/]<Definition.name>/definition.json
ops/[<group>/]<Definition.name>/oracle.py
ops/[<group>/]<Definition.name>/correctness.jsonl
ops/[<group>/]<Definition.name>/timing.jsonl
ops/[<group>/]<Definition.name>/correctness_full.jsonl  # optional inactive archive
ops/[<group>/]<Definition.name>/timing_full.jsonl       # optional inactive archive
ops/[<group>/]<Definition.name>/assets/...  # optional private runtime files
```

`<group>` 是可选的一层目录，只用于整理 Catalog；它不进入
`Definition.name`，也不改变 wire protocol 中的 operator identity。禁止继续增加更深
的目录层级。

manifest 固定为：

```json
{"api_version": "v6.2", "evaluator": "native", "layout": "per-operator"}
```

若 native timing reference 必须复用 FlagGems benchmark 自身的 callable，manifest 额外声明 `framework="flaggems"`、`framework_repository`、`framework_branch` 和完整 40 位 `framework_revision`。Server 在加载 oracle 前校验 checkout HEAD 与该 revision 完全一致、相关源码工作区干净，并只从其 `src` 导入；oracle 不保存机器路径，也不能回退到镜像中碰巧安装的其他 FlagGems 版本。

`definition.json` 是纯 ABI。`oracle.py` 是唯一 reference source，顶层必须只声明一次
`REFERENCE_DEVICE`。ABI 一致的 `run()` 是 correctness/timing 的共享 primary；可选
`correctness_run()` 和 `timing_run()` 分别覆盖对应阶段；可选 `torch_run()` 是整轮
fallback。三个逻辑角色分别称为
`correctness_reference`、`timing_reference` 和 `torch_reference`，但不拆成三个文件。
解析顺序固定为：correctness 使用 `correctness_run ?? run`，timing 使用
`timing_run ?? run`。两个阶段语义一致时只需一个 `run()`。

oracle 的额外 Python、`.so` 或配置只能作为普通文件放在该算子的 `assets/` 下；除
可选的两个 `_full.jsonl` 归档外，禁止符号链接和算子目录中的其他同级文件。KGS 从原目录加载 oracle，使 `__file__` 相对路径
和私有 import 可用，并把全部 assets 的路径与内容纳入 fingerprint。assets 不是公共
Definition 字段，也不由 Candidate 通过 wire request 上传。

`correctness_full.jsonl` 和 `timing_full.jsonl` 只保存采样前的完整 Workload。Server
不加载、不执行它们；benchmark fingerprint 和评测结果只由无后缀的活跃 Workload
决定。

Evaluate 在 candidate 运行前检查全部 primary workload；普通 workload smoke 对应
reference，带 `output_path` 的 correctness workload 改为验证并加载 golden。任一
primary 执行失败时从头 smoke 全部 `torch_run()` workload。只有 fallback 全部成功才
整轮切换，不能按 workload 混用。
Candidate 错误、数值/effects 错误、TIMEOUT、worker 退出和设备错误不触发 fallback。
Preflight 仍然只运行 timing candidate。

v6.2 的 `valid(ref_outputs, sol_outputs, inputs, ctx)` 接收按 Definition 参数名索引的
`inputs` 映射。声明 `valid` 后，其 verdict 取代 correctness 返回值和已声明 mutation
结果的逐值比较；KGS 仍先强制 PyTree 结构、leaf 类型、Tensor shape/dtype、
未声明 mutation 检查与 return-alias。这使随机原地算子可以由 `valid` 实现统计校验。
若上游 correctness 明确忽略返回容器类型或采用其他非 PyTree 契约，oracle 可同时声明
`VALID_OWNS_RETURN_CONTRACT = True`。此时只有返回值的结构、leaf 类型、shape/dtype
和数值检查交给 `valid` 完整负责；未声明 mutation 与 return-alias 仍由 KGS 强制。
该常量必须是布尔值且必须与 `valid` 同时存在，默认值为 `False`。Torch fallback
对应的独立开关是 `TORCH_VALID_OWNS_RETURN_CONTRACT`，并要求 `torch_valid`。
v6.0 继续传入按 ABI 排列的参数值列表，并保留先逐值比较、再调用 `valid` 的行为。

v6.0 flat Native Catalog 继续兼容：

Native Catalog 按 Definition 的相对路径自动配对受信资产：

```text
definitions/<relative>.json
references/<relative>.py
workloads/<relative>.correctness.jsonl
workloads/<relative>.timing.jsonl
```

legacy reference 模块可提供 `run`、`gen_inputs`、`correctness` 和 `valid`。复杂对象在
`gen_inputs(ctx, device)` 中生成；Workload 只保存 JSON context、seed 和可选容差。
Native `inspect` 把 timing Workload 映射成公共 case list；Preflight 只执行这些
timing Workload，correctness Workload 只在正式 Evaluate 中运行。

基础 random recipe 默认先在 CPU 生成再为 reference/candidate 克隆。若原始测试明确
使用 `torch.randn(..., device=device)`，v6.2 可写
`{"type":"random","shape":[...],"dtype":"float16","device":"target"}`。
Server 会在每次调用物化前重置 seed，并直接使用分配到的调用设备默认 Generator；这样
既保留 reference/candidate 的确定性，又不把 target 分配路径偷偷替换成 CPU staging。
`device` 只接受 `cpu` 或 `target`，缺省仍为 `cpu`，因此 v6.0/旧 Catalog 行为不变。

random recipe 还可用 `distribution=normal|uniform|integer`、`source_dtype` 和顺序
`transforms` 保留来源 benchmark 的输入语义。支持的变换为 `cast`、基础标量四则/反向
运算、`transpose`、`symmetrize_sum`、`triu`、`tril`、`softmax` 与带嵌套 random
recipe 的 `multiply_random`；后者的 `before=true` 保留随机数在主 Tensor 前生成的
顺序。运行器与独立交付验证工具使用同一物化规则。

v6.2 还继承文件型 Workload：顶层 `input_path` 指向 safetensors，标为
`type="safetensor"` 的参数按同名 key 读取；correctness 顶层 `output_path` 按
`Definition.outputs` 的名称读取 golden Tensor，并跳过该 workload 的 correctness
reference。两个路径都必须是目标环境可读的绝对路径。`output_path` 不允许用于 timing；
timing 即使从 `input_path` 读取真实输入，reference/candidate latency 仍由 KGS 在同一
设备现场测量，不读取文件 metadata 中的外部 latency。

## FlagGems evaluator

FlagGems Adapter Catalog 只保存 Definition 和 `benchmark_level=core`，不声明必须使用的 framework revision。checkout 由 KG 实例或操作方选择，旧 Adapter manifest 的 framework 字段不参与绑定。adapter 在目标环境中直接运行 FlagGems 原始 accuracy pytest 与 benchmark pytest，保留原始参数化、输入生成、skip、容差、mutation 断言、shape 顺序和计时实现。

`inspect` 枚举 benchmark 内部 case，并原样保留 FlagGems 提供的全局 `case_id`。
pytest nodeid 已包含在这个不透明 ID 中，Server 不再维护第二层映射；Agent 只看到
dtype、shape、布局意图和调用参数等可用描述。Preflight 使用相同的全部 core case
序列，但每个 case 只物化输入并调用 candidate 一次。Evaluate 执行完整 correctness
和 core timing。

## Case List

`inspect` 返回 `kernelgen.case-list/v1`：

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
        "dtype": "float16",
        "shape": [[16, 32], [16, 64], [64, 32]],
        "params": {"b_column_major": false}
      }
    ]
  }
}
```

`benchmark_fingerprint` 绑定 framework revision、测试源码和枚举结果。Preflight、
Evaluate snapshot 与 Profile 必须使用同一 fingerprint，避免测试更新后重放错 case。
FlagGems Adapter 使用实际 checkout HEAD，不再与 Catalog 中的旧 `framework_revision` 比较；相关源码工作树仍必须干净，实际 commit 参与 fingerprint。不同实验可显式选择不同 Gems 版本，不在已有 campaign 中途自动更新；测试资产缺失、ABI 或报告格式不兼容仍报错。FlagGems-backed native Catalog 的冻结 oracle 不在此放宽范围，仍按其 manifest 校验 exact revision。厂商差异由 KGS 的受信 backend/reference 兼容层处理，不通过未提交修改引入每台机器不同的 pytest 语义。

## 公共返回协议

`/evaluate` 对所有 adapter 返回同一个 `EvaluateResponse`：状态、case 数量、权威
headline、扁平的 correctness/timing `per_workload`、hack 检查、错误日志和设备信息。
`reference_source` 为 `primary` 或 `torch_fallback`，记录本轮正式评测统一使用的 source。
只有全部 case 通过时才发布 headline 性能指标。超时和设备强探针失败仍分别返回
`TIMEOUT` 与 `SUSPECTED_DEVICE_ERROR`，slot 隔离语义与 evaluator 无关。

FlagGems correctness 用例全部 `skipped` 时，Eval 仍按原流程执行 benchmark；若其余检查正常完成，HTTP 200 返回 `status=ALL_SKIP`，保留现有计时字段、逐用例结果和 skip 原因。此时的计时仅供参考，不表示正确性通过，不能据此选为最佳候选。部分 skip 的既有行为不变；benchmark 失败及 pytest 异常仍按原有路径处理，空报告不属于 `ALL_SKIP`。客户端需使用包含 `EvaluationStatus.ALL_SKIP` 的 KGS 协议模型。

FlagGems correctness 的 pytest `failed` 结果根据原始 `reason` 分类：识别到导入、编译或执行异常（非 `AssertionError` 的 `*Error` / `*Exception`）时，逐用例状态为 `RUNTIME_ERROR`；断言失败和无法识别异常类型的文本保持 `INCORRECT_NUMERICAL`。原始 `reason` 保留在逐用例 `log`，整体状态仍按公共聚合规则计算，不将任意一条运行异常强行覆盖为整体状态；例如通过与失败用例混合时仍为 `PARTIAL_PASS`。

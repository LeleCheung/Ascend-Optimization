# Native Catalog 源策略与条件测例

Native Catalog 可声明 `Definition.source_policy_id`，引用 KGS 随包分发的源 pytest 厂商策略。该值表示源策略身份，不是 KG release、KGS release 或 Protocol version。当前提供 `flaggems/d64794e63b502cb836bc015a92a62c42de4be05a`，来源和全部厂商值见 [source_policy.json](../kernelgen_server/runtime/source_policy.json)。数据从固定 FlagGems 源码的 VendorDescriptor 默认值及厂商覆盖值复制，运行时不安装、导入或自动更新 Gems。

## 契约

只有依赖源策略的 Definition 才设置 `source_policy_id`。Oracle 通过 `from kernelgen_server.runtime.source_policy import source_flag` 和 `source_flag("support_fp64")` 读取标志，策略身份仅保存在 Definition，不复制到每个 workload 或 oracle 常量。KGS 在 Eval、Preflight 和 Profile 执行期间绑定当前逻辑 backend 与该身份，退出时恢复上下文；CUDA-compatible 厂商使用各自逻辑 backend，不按设备字符串 `cuda:0` 判断。此配置表达源测试的精度/适用规则，并非实测硬件能力。

workload 可带 `"source_condition": {"flags": {"support_fp64": true}}`。当前支持 `support_fp64`、`support_bf16`、`support_int64` 的布尔期望，多个标志为 AND。抽取必须保留源测试的测例和条件，不按抽取机器预先删除。任意厂商表达式、复杂 OR 或其他无法忠实表示的条件仍须报告协议缺口，不能简化条件或制造通过。

同一模块的 `source_vendor()` 返回当前策略作用域绑定的源厂商名称，用于保留源输入生成分支，例如 cambricon/int16 先在 CPU 生成再传入目标设备。它复用 `source_flag` 的策略身份和未知 backend 校验，不导入 Gems，也不按 CUDA-compatible 设备字符串猜厂商；必须在执行 hook 内调用，不能在模块导入时读取。该 helper 不增加 workload 的通用厂商条件表达式。

条件不满足时，在分配输入、调用 reference 或 candidate 前记录 `SKIP` 和原因；配置未知、策略身份不支持或实际执行失败不能转换成 SKIP，也不能通过 oracle fallback 绕过未知策略。源 `TO_CPU`、upcast、cast-back 和容差仍由 oracle 忠实表达，精度分支本身不等同于输入 dtype 的跳过条件。

Eval 保留每个 SKIP 测例，`num_passed` 不包含 SKIP；其余适用测例全部通过时可为 PASSED。全部 correctness 测例 SKIP 时仍继续适用的 benchmark，若其余执行成功则返回 ALL_SKIP，逐测例 timing 保留，但不发布 PASSED 的 headline 加速比。全部 timing 测例 SKIP 时同样不能宣称已测得 PASSED，其他执行成功时返回 ALL_SKIP。真实错误仍按原聚合规则返回，不能被 ALL_SKIP 遮蔽。Preflight 仍只 smoke timing 测例；它不承担 correctness 验证。显式 Profile 一个不适用的测例会被拒绝，不产生成功的 Profile 记录。

## 发现与快照

客户端通过 `/status.capabilities.native_source_policy` 核对 `enabled`、`workload_conditions`、`policy_id` 及目标 `flags`；未知厂商返回 `vendor=null, flags=null`。不能根据软件 release 推断支持，也不能用 Agent 本机的 Torch 探针替代该配置。未使用源策略的 Catalog 不要求此能力。Protocol 兼容仍按 `/status.api_version` 判断。

使用策略的 Native Catalog，其 benchmark fingerprint 同时绑定目标策略快照；换策略或目标厂商后必须重新 inspect，不能复用旧计时身份。新增配置必须保留来源可追溯性，不静默改变已有策略 ID 下的源语义。

## 验证

相关测试为 `tests/test_source_policy.py`、`tests/test_source_policy_engine.py` 和 `tests/test_operator_contract.py`。引擎测试使用 CPU 测试设备模拟逻辑 backend，覆盖条件判断早于输入分配、未知策略、ALL_SKIP 后继续 benchmark、真实错误不转 SKIP 及上下文隔离；这不代表已在对应厂商芯片验真。正式部署仍须完成目标设备验证。

2026-09-15 已在昇腾 910B4-1 物理 7 号卡的独立临时 KGS `9af6ef8`（KG `da48958b`，Protocol v6.2，profiler timing）完成验真：oracle 断言 `support_fp64=false` 后实际使用 NPU float32；条件 fp64 correctness/timing 均 SKIP；真实抽取的 `fix` Catalog 18 correctness + 15 timing 全通过。ALL_SKIP 时仍运行 benchmark；故意的执行异常返回 RUNTIME_ERROR；两请求并发最多一个 active、另一个 waiting，最终 slot 健康空闲。禁止 Gems 导入的独立 Debug Job 也能读取策略。该开发验证不意味着新 release、其他厂商或全部历史 Catalog 已验收，详细报告位于配套 KG 的 `docs/validation/native_source_policy_ascend_20260915.md`，原始证据为本地 `runs/kernel_todo_v2/source-policy-npu-OoqmNZ/`。

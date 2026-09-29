# 历史候选 preflight 源码门禁回放（2026-09-08 更新）

当前通用门禁检出已标注问题 20/22（90.9%），115 份正常对照均未被拒绝（误杀 0/115）。`lift_fresh` 辅助 launch 和 `unsqueeze_` 额外 `squeeze_()` 保留为两个未检出案例，不增加算子专用规则。此前 a944e37 增加的两个专用检查已移除；当时的 22/22 仅保留为历史证据，不代表当前门禁覆盖率。

## 范围与结果

扫描 348 份历史候选，另有 21 条台账记录缺少源码，已扫描源码 SHA256 均与历史观察记录一致。计数单位为“厂商 × 算子”台账行。已标注问题 22 份、正常对照 115 份，其余 211 份未作为正常标签，不计入误杀率分母。

标签独立来自本地 `runs/kernel_todo_v2/regression_audit_20260907_01/non-ascend-full-attribution/attribution.json`，不是根据检测结果反推。正常对照同时满足 `attribution_state=no_failure_observed`、`native_status=PASSED`、`native_candidate_hash_verified=true`，只代表此前选定原生 core 测试的验证范围。昇腾源码参与全量扫描；已标注问题和正常对照沿用此前昇腾外归因范围。

| 问题类别 | 当前检出/样本数 |
|---|---:|
| 候选修改 Gems 注册／测试接口 | 9/9 |
| 候选修改 Torch reference／全局运算 | 7/7 |
| 候选修改 compiler／JIT 私有接口 | 6/6 |
| lift_fresh 用零 grid 冒充 Triton 计算 | 1/1 |
| lift_fresh 启动不参与结果的 side-stream kernel | 0/1 |
| unsqueeze_ 额外 squeeze_ 改变输入语义 | 0/1 |

前三类重叠，去重为 19 份，加上通用零 grid 规则拒绝的 1 份，共 20 份。Gems 专属限制仍仅对 Gems adapter 启用；native 保留 Torch、编译器与评测全局状态的通用保护。

| 厂商 | 已标注问题：检出/总数 | 正常对照：误杀/总数 |
|---|---:|---:|
| 摩尔线程 | 5/6 | 0/29 |
| 沐曦 | 6/6 | 0/37 |
| 平头哥 | 6/7 | 0/19 |
| 天数 | 3/3 | 0/30 |

实际调用 FlagGems adapter `preflight()`，以哨兵替代后续 `_prepare_workdir`：20 份被拒绝的候选均返回 `FAILED / candidate_admission / is_hack=true`；115 份正常对照和两个未检出案例均到达后续准备边界。这仅验证源码准入阶段，没有执行候选 import、pytest `--level core`、GPU 编译、精度或计时，不等于这些候选完整 preflight 或复测通过。

## 两个未检出案例的处理边界

摩尔线程 `lift_fresh`（moer:26）在 side stream 启动只写 scratch 的 kernel，最后返回输入。入口“返回输入且存在 launch”的专用模式能够命中这份源码，但不能普遍证明 kernel 不参与结果或计时行为不正确，因此移除该专用检测。

平头哥 `unsqueeze_`（pingtouge:42）在高 rank 时额外调用输入 `squeeze_()`，改变语义。针对该算子直接禁止此方法只能覆盖特定写法，因此同样移除，不将 preflight 扩展成逐算子语义检测器。

既有 metadata 白名单保持不变：仅对已审核的最小 `lift_fresh` / `unsqueeze_` 形式豁免缺少 JIT／launch 的检查。不添加新的专用黑名单，也不扩大白名单的职责；语义、alias、输入效果和计时正确性由对应执行验证承担。策略见 [候选准入与 metadata 白名单](metadata_admission_policy.md)。

此前昇腾外 P0 涉及 40 条去重记录，当前源码门禁可拦截其中 20 条（50%）；其余 20 条包括上述两个案例及已有测试注入、reference、编译、精度和计时问题。覆盖的是被拒绝的候选行，不能据此推断该行涉及的所有独立问题机制均已解决。未扩大 BLOCK，也不把源码门禁通过作为任意 Python 无副作用的证明。

## 验证与复现

实现位于 KGS v6.3.1 开发分支 `feat/metadata-admission-policy`；配套 KG v6.2.1 开发分支 `feat/candidate-admission@3ad4ccd1`，Protocol v6.2。此次仅撤销 a944e37 的三处源码/测试改动并更新文档，未新增 release/tag、未修改 KG 流程、未启动 Server/Coder 或安装依赖。相关 admission、metadata、FlagGems adapter 与既有 hack 检测 host 测试 96 项通过。

当前证据目录为 `/data/akg_kernel_bench_lite/kernelgen/runs/preflight-history-replay-generic-20260908/`，包含 `generic.json`、`summary.json`、`integration.json`、`unit-tests.log` 和复现脚本。原来的 `runs/preflight-history-replay-20260907/` 保持不动，保存曾加入专用规则时的历史结果；独立标签使用该目录的 `gold_labels.json`。这些输入及 kernel_todo_v2 不随新 clone 分发，复现前需恢复授权归档。

在当前 KGS feature 根目录运行：

```bash
PYTHONPATH=. python3 -c "import kernelgen_server; print(kernelgen_server.__file__)"
PYTHONPATH=. python3 /data/akg_kernel_bench_lite/kernelgen/runs/preflight-history-replay-generic-20260908/scan.py
PYTHONPATH=. python3 /data/akg_kernel_bench_lite/kernelgen/runs/preflight-history-replay-generic-20260908/verify.py
PYTHONPATH=. python3 -m pytest -q tests/test_candidate_admission.py tests/test_metadata_policy.py tests/test_flaggems_adapter.py tests/test_hack_detection.py tests/test_hack_detection_blacklists.py tests/test_hack_detection_extended.py tests/test_hack_detection_namespace_exclusion.py
```

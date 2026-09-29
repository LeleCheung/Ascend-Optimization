---
name: pytest-review
description: 在 KernelGen 生成前，用原生 FlagGems pytest 验证 reference-as-solution、候选注入和目标测试链路；覆盖不足通常是非阻塞建议，不批准性能结果，也不改变运行时 BLOCK 状态。
---

# pytest review

## 目标

回答“测试链路是否可用于开始生成”，不是判断算子是否已优化或是否覆盖所有输入。使用原生 Gems pytest，benchmark 固定 `--level core`；KGS 只提供受调度的目标执行环境，设备、backend 和版本事实来自 `/status` 与 Debug Job。

## 执行顺序

1. 固定 Gems commit、目标环境、算子入口、测试路径和 reference source。候选必须由正式注入工具生成；协议缺口要记录，不能修改测试、reference、容差或增加旁路绕过。
2. 按 [checks.md](references/checks.md) 运行 `probe_pytest.py --phase reference-as-solution`，传入 `--candidate` 和 `--reference-source`。探针只验证身份、实际调用、执行/skip、阶段和异常，不自行注入，也不替代 Preflight。
3. 输出 `READY` 或 `NEEDS_FIX`，保存命令、日志、JSON、源码 hash、入口和目标版本。`READY` 允许 P1 建议；全 skip、零执行、注入证据缺失、异常或无法证明实际入口必须是 `NEEDS_FIX`。

## 判定规则

| 类别 | 处理 | 阻塞 |
|---|---|---|
| API/依赖/后端缺失 | `API_UNAVAILABLE` | 是 |
| 候选未执行、入口错误、baseline 被替换 | `CANDIDATE_NOT_INJECTED` | 是 |
| 实际编译失败 | `COMPILATION_FAILED`，保留源码和异常 | 是 |
| 独立证据确认 reference 错误 | `REFERENCE_SEMANTICS_INVALID` | 是 |
| 全 skip、零执行、执行异常 | 保留原始证据，`NEEDS_FIX` | 是 |
| workload/边界/部分 skip/覆盖不足/性能可比性 | `suggestions`，通常 P1 | 否 |

未知异常不要强行分类；普通 `AssertionError` 不能自动判定 reference 错误。Reference-as-solution 通过不代表 reference 数学语义正确，也不要求加速比达到 0.8；候选性能、最终 best 和 ledger 由后续阶段负责。

## 覆盖与低精度

对已在库且应支持低精度的 ATen/Gems 算子，记录来源、开发单位和 KT2 标记，优先处理 KT2 且有开发单位的条目。以 PyTorch `native_functions.yaml`、Gems 注册和 CUDA 实测共同确认 dtype 支持。支持的 `int8`、`uint8`、`float8_e4m3`、`float8_e5m2` 必须有对应 pytest case；正在开发的 ATen 算子同样适用，不能只按名称猜测。

有效 case 少于 100 时，建议补到至少 100，覆盖适用的数值范围、0～5 维 shape、支持 dtype、广播和参数边界。推荐范围为 `[-1,1]`、`[0,1]`、`[-1,0]`、`[0,dtype_max]`、`[dtype_min,0]`；固定维度、原生不支持的 dtype/shape/广播或不适用参数按原生限制处理，不强行生成全部组合。缺失 case 通常是 P1 建议，只有产品或算子契约明确要求时才升级阻塞。

## 禁止事项

- 不修改原始 pytest、reference、candidate、容差或 skip 来伪造通过。
- 不让 Agent 编造 case、correctness 清单或性能数字。
- 不用 adapter 结果替代原生 pytest 证据。
- 不把 review 当作 Preflight、性能验收、恶意代码沙箱或运行时 BLOCK 开关。

Coder 只修候选；不修 Gems、Torch、编译器或评测全局状态。源码扫描和 `reference`/`candidate` 深入模式只在明确要求的回归排查中使用。

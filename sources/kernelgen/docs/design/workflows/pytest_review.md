# Pytest review：生成前的最小验证

状态：2026-09-08 用户已确认设计；现已在本分支实现 skill 的最小流程和探针 reference-as-solution 模式，输出 `READY / NEEDS_FIX`。生成流程尚未自动调用该 skill，reference 语义错误仍需独立证据确认。验证使用 host 合成 pytest 场景，未新增设备运行。

## 目标与范围

Pytest review 在 Coder 开始生成前，运行一次 reference-as-solution，确认目标环境和测试链路具备开始生成的条件。重点发现 API 缺失、candidate 注入失败、编译失败和已确认的 reference 语义错误等 P0 问题。Workload、边界覆盖和性能路径等 P1 改进只给建议，不强制阻塞，不要求先完成全面测试审查。

本设计替代此前“完整审查对象、reference、输入、覆盖、计时和汇总后才放行”的宽泛准入方案。不新增审核 Agent，使用现有 pytest review skill 与确定性执行观测；不要求为每个算子增加专用检查规则。

## 最小执行流程

1. 固定目标环境、Gems revision 和算子测试入口。把 reference 实现作为 solution，通过正式的 candidate path/entry 注入，运行 Gems 原生 pytest，使用 `--level core`，保留原有测试和断言。
2. 在这次验证中记录实际执行对象、solution 注入及调用证据、pytest 执行/skip 数和失败阶段。原生对照实现保持独立，不能把 baseline 同时替换成 solution；不能仅凭同名函数、相同源码 hash 或 pytest 退出码为 0 判断注入成功。测试调用和 benchmark 调用分别观测，防止只在正确性阶段命中候选。
3. 输出 `READY` 或 `NEEDS_FIX`，附 P0 证据及 P1 建议。一次验证不通过就保留现场并交相应维护方，不交给 Coder 修补框架或评测代码；必要的后续诊断不计作增加常规 review 步骤。

验证在受调度的目标环境执行，远端设备遵守现有 KGS Debug Job 与设备队列约束，不以 Agent 本机环境推断目标能力。这里的“一次”指一次完整 reference-as-solution 验证，不是只调用一次 kernel，也不是取消 pytest 自身的参数化或 benchmark warmup。

## 状态与信号

| 结果状态 | 判定 |
|---|---|
| `READY` | 验证实际执行并通过，正式注入链路获得执行证据，未发现阻塞问题；允许附带 P1 建议 |
| `NEEDS_FIX` | 发现 P0，或验证未有效完成，例如全部 skip、零有效执行、运行中断或缺少必要的注入证据；不能按通过放行 |

`NEEDS_FIX` 是生成前准备状态，不是 Coder 运行中的 `BLOCK`，也不表示故障一定在候选、Gems 或编译器某一方。未完成验证或未定位的失败保留原始原因，不为了填分类而猜测具体故障。

| P0 信号 | 触发依据 | 后续责任 |
|---|---|---|
| `API_UNAVAILABLE` | 必要 API 缺失或目标环境明确不支持，导致验证不能执行 | 按实际缺失项交 Gems、Torch/厂商 runtime 或测试准备维护方 |
| `CANDIDATE_NOT_INJECTED` | 应执行传入 solution 的阶段未命中正式注入入口，或执行了错误对象 | 我方注入工具、adapter 或 Gems 测试入口维护方；按实际失败位置定责 |
| `COMPILATION_FAILED` | 验证中出现编译失败，保存实际编译源码和完整异常 | 先注明 reference/solution 哪一侧及哪个编译阶段失败，再由实现或编译器维护方诊断；不得一律判为编译器 bug |
| `REFERENCE_SEMANTICS_INVALID` | 独立断言、可信对照或具体反例证实 reference 语义错误 | reference 实现或对应测试语义维护方 |

每个问题记录证据、责任方和下一步处理。责任方尚不明确时标明待定位，保留实际异常与失败阶段；同一次失败不因缺失 API 进一步导致编译失败而重复计算影响算子数。

## Reference 语义检查的能力边界

Reference-as-solution 主要证明测试链路能执行，不能自动证明 reference 数学或接口语义正确。若两侧使用同一份错误实现，可能一致通过；只有独立断言、可信对照或反例能够证实时，才上报 `REFERENCE_SEMANTICS_INVALID`。数值不一致本身也不能直接证明 reference 错，必须保留对照关系。

不为此新增独立语义审核流程，也不因为“尚未独立证明全部 reference 语义”而长期卡住 `READY`。但已观察到的失败不能以此理由忽略；基线被误覆盖等已发现的身份错误必须修复，不能利用自比较伪装通过。

## P1 建议不阻塞

Workload 参数分布、随机种子与完整参数记录、边界用例、主要性能路径覆盖、不同测试集合或计时口径的可比性等问题，作为建议记录，不要求先补齐才开始生成。发现现有断言实际失败、对象错误或验证未执行时，仍按上述阻塞规则处理，不能将真实故障降为覆盖建议。

例如 `expand_copy` 的性能点只测同形状复制，没有覆盖真实扩展：只给出“补充真实扩展性能点、明确当前性能覆盖范围”的 P1 建议。只要本次 reference-as-solution 有效通过，结果仍为 `READY`。这替代此前因该覆盖缺口而给出 `NEEDS_FIX` 的示例；全部 skip 则仍属于验证未执行，不能套用 P1 放行。

Reference-as-solution 的加速比不要求达到 0.8，也不用于选 best。低性能或缺少额外性能覆盖本身不阻止开始生成；正式候选最终验收仍遵守当次配置的正确性、有效计时和性能阈值。

## 阶段职责

| 阶段 | 职责 |
|---|---|
| Pytest review | 一次 reference-as-solution，输出准备状态、P0 证据和非阻塞的 P1 建议 |
| Preflight | 候选源码准入与适用的 hack/全局状态修改约束，不承担数学正确性和性能覆盖证明 |
| Coder 运行 | 修复候选算法、数值、访存和合法实现配置；运行中才暴露且不属于候选职责的问题走已有 BLOCK 设计 |
| 最终性能验收 | 双边计时检查、reference 漂移识别和最终 best 独立复验，不塞进本次 review |

现有实现入口：[pytest review skill](../../../.kernelgen/skills/pytest-review/SKILL.md)、[检查脚本用法](../../../.kernelgen/skills/pytest-review/references/checks.md)。默认使用 `--phase reference-as-solution --reference-source <prepared-reference>`；原 `reference`/`candidate` 深入审计模式保留兼容，非默认前置步骤。最小模式的执行证据和旧审计提示分别保留；P1 提示不覆盖准备状态。

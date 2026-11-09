# 独立 HG-KDA 全量回归

**原 25 项已全部完成独立入口验证，修复两项性能回落后均达标；新增 `uniform` 达标，当前原合同及语义口径为 26/30。累计积分另有 1 项通过独立数学参考合同。** 完整当前表见[30 项状态](current-summary.md)，机器可读状态见[当前索引](../../current.json)。

固定的 25 个候选全部通过独立 CLI 完成评测：776/776 个正式用例、2366/2366 个补充检查通过，23 项相对 CANN 超过 0.8×。服务器与 Windows 对原始源码、合同、请求、补测和计时结果的独立核验均通过。

| 性能未过线项 | 拆分前记录 | 本轮固定候选 |
| --- | ---: | ---: |
| `negative_` | 0.814× | 0.653× |
| `hardswish_out` | 0.862× | 0.778× |

两项均通过正确性及两侧补测，问题为性能回落。所有正式性能采用 KGS 完整调用 walltime，保持原候选、合同、容差和计时设置，每项一次正式请求。后续修复使用新源码身份单独记录。

- [25 项完整表格](fixed-25/summary-final.md)
- [服务器最终审计](fixed-25/audit-final.json)、[Windows 独立审计](windows-audit.json)
- [逐 case 耗时](fixed-25/summary-final.csv)
- [新旧耗时分别比较](historical-drift.json)：区分候选变化与 CANN 变化；历史汇总分数和所附单次原件分开记录
- [批次索引](index.json)、[运行与复现说明](../../../docs/regression-20261109.md)
- [异机原件核验](portable-archive-audit.json)：重算 25 项结论并逐项检查传输清单，包含 LFS 实体

`fixed-25/` 保存本批原件及传输清单，不含运行缓存。复现需要当前 HG-KDA 工具、匹配的 NPU 环境，以及恢复说明中记录的历史参考库和宿主桥；不直接重启旧优化控制器。

原 `benchmarks/index.json` 记录拆分前验收，本批结果独立保存；目录重构和入口通过不等于全部原性能仍然过线。后续参考资格修复和新候选结果在各自批次中记录。

## 回归后的实验

| 算子及批次 | 相对 CANN | 正式评测 | 两侧补测 | 结论 |
| --- | ---: | --- | --- | --- |
| `uniform` 原生参考资格修复 | 2.399× | 18/18 | 各 31/31 | 原 Triton 候选新增合格的 CANN Uniform＋Clamp 参考，达到验收线 |
| `negative_` 完整分块候选 | 0.806× | 30/30 | 各 63/63 | 宿主缓存＋完整块固定长度搬运，恢复达标 |
| `hardswish_out` 按类型分块候选 | 0.867× | 30/30 | 各 42/42 | 归一化门控、低精度计算及缓冲预算约束，恢复达标 |
| `cumulative_trapezoid` 数学参考合同 | 1.012× | 33/33 | 各 36/36 | 独立数学参考合同通过，原 NPU 默认合同未宣布通过 |

正式原件见 [uniform](uniform-native-reference/summarize-final.md)、[negative_](negative-complete-block/summarize-final.md)、[hardswish_out](hardswish-native-low-precision/summarize-final.md)及[累计积分](trapezoid-mathematical-reference/summarize-final.md)。四项均通过原件审计及 Windows 镜像核验，控制器已停止各自专用 KGS。

`uniform` 的参考资格检查为 31/31；完整性能包括原生组合及其适配开销，候选源码保持不变。累计积分参考为 CANN Add＋Muls＋FP32 Cumsum＋Cast，独立数学资格检查 36/36。两项参考变更均明确记录，不将更换参考带来的分数变化宣称为候选优化收益。

固定回归的 23/25 结论保持不变；两项修复使用新的源码身份。`negative_` 的中间宿主缓存版本仅为 0.695×，未通过，原件保留于 [中间实验](negative-host-cache/summarize-final.md)。编译失败、UB 需求及补测发现的 BF16 大输入溢出保存在 `compile-failures/`，这些版本没有正式性能成绩。

## 尚未闭环的原生参考

- `expm1_`：官方默认／高精度模式仍未通过小值数学资格。
- `fractional_max_pool3d`：尚缺同语义的随机窗口及索引参考。
- `linalg_inv_ex`：原生 Inverse 缺逐矩阵 LU `info` 输出。

研究范围、版本与证据见[CANN 参考可用性](../../../docs/native-reference-availability-20261109.md)。累计积分的新数学合同与上述三项分开记录，仍不计入原默认合同达标数。

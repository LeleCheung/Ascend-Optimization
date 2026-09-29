# CatalogExtract 审核迭代：PPU 实际算子验证

本页保留两次验证：最初 Deepseek 抽取受输出 ABI 缺口阻断；修复 ABI 后，用户临时指定 Codex 替代不可用的 Opus，Codex 已产出通过源码 review 的 Catalog，并交给独立 CatalogOptimize 做 PPU 实测。下面先保留初次阻断证据，再记录 Codex 验证，不能把二者混为一次自动修订成功。

## 初次 Deepseek 验证结论

真实抽取 → 审核 → 两次修订 → 再审核链路完成，3 轮上限生效；最终仍有 P0，正确停在 `WAITING_REVIEW`，未发布最终 Catalog，也没有启动 CatalogOptimize。此结果验证了有界迭代与阻断，不是完整优化 E2E 通过。

## 输入与环境

算子为平头哥历史未达标的 `scaled_dot_product_cudnn_attention`，历史结果 0.731×。固定 Gems checkout 为 `d64794e63b502cb836bc015a92a62c42de4be05a`。独立 CatalogExtract 使用 `feat/catalog-extract-review-loop` 开发代码，基线 KG `d92e7ccb`；模型为 Claude Runtime / `deepseek-v4-flash[1m]`。配套 KGS 为 v6.3.4@`00dfc8692a6927d052b0c6da238cdde7908363c4`，Protocol v6.2，未改变锁定清单。

PPU 现有实例为 `kt2-stage2-ppu-EdHzTP`，远端 loopback 22107、本地 SSH stdio proxy 23107，目标 PPU-ZW810E，16 个健康 slot。通过 KGS Debug Job 真实执行原 benchmark 的 `--list-cases`，得到 20 个 timing case；这是 PPU 环境采集的清单，不代表硬件中立的完整枚举。源码、原 pytest 和 case list 在整个迭代中未修改；未安装依赖或变更厂商运行时。没有候选评测、Profile 或历史成绩导入。

## 证据与根因

本地实验根目录：`runs/kernel_todo_v2/ppu-sdpa-extract-e2e-20260914-8YGmqB/`（实验数据不随仓库分发）。`case-list-job.json` 和 `cases.json` 记录真实采集；`extract-02/attempts/01..03/review/review.json` 保留各轮内容摘要、必读文件和审核意见；`extract-02/review-loop.json` 记录第 3 轮 NEEDS_FIX；`extract-02-error.json` 为 `CatalogReviewRequired`。最早的 `extract/` 尝试仅因测试 runtime_factory 没有复制角色目录而失败，修复测试入口后新建 `extract-02/`，没有覆盖旧证据。

三轮共同的 P0：源码返回 9 元组，原 pytest 还检查 logsumexp shape、max_q 和 max_k；确定性 `extract_flaggems_definition()` 从 `_OUTPUTS` 表取值，未登记的该算子回退为 `["out"]`。抽取 prompt 又要求 Agent 不得改变这份 Definition，因此语义修订无法修复被 Python 固定的输出结构。Catalog 仍只比较主输出，但覆盖报告声称完整。第 2 轮已移除无关 TF32 setup，不再有该项意见；输出契约 P0 未解决。

这不是 KGS 设备故障，也不是本轮取消/迭代失效。目前只证实 KG 抽取的 ABI 推导缺口，不能据此声称 Protocol v6.2 不支持多输出。后续应单独完善确定性输出契约与源断言映射，确认 tuple、shape-only 辅助检查和 timing callable 的表达方式，再重新抽取和审核；不能仅增大轮数、降低审核标准、删掉原断言，或把旧单输出实现当作已验证 seed 来伪造通过。

## Host 门禁与后续

抽取侧 112 passed：`test_catalog_extract_review.py`、`test_flaggems_v62_batch_extract.py`、`test_flaggems_pr_extract.py`、`test_gems_source_profiles.py`、`test_case_collection.py`、`test_catalog_optimize.py`、`test_extract_and_optimize.py`。包括修订通过、耗尽、证据缺失/篡改和完整审核输出后的取消。导入通过临时父目录的 `kernelgen` 链接固定到 feature worktree，KGS 导入上述锁定主工作区。

CatalogOptimize 的 SimpleOpt/KernelGen 接线另在独立 feature 回归；本次没有得到可接受的 SDPA Catalog，因此该算子的下游真实优化尚未验证。后续应先补齐抽取契约，再用同一份冻结 Catalog 分别验证两种优化模式，保留 seed 未验证与 reference 只读语义。

## Codex 接续验证

输出 ABI 已由 [!126](https://gitee.com/BaaiAC/kernelgen/pulls/126) 修复为原实现的 9 项返回值。Opus 尝试因 API 401 失败并协作取消，用户随后指定 Codex。本次抽取使用 Codex CLI 0.153.4 / `gpt-6-astra`；独立 review 及下游优化仍使用 Claude Runtime / `deepseek-v4-flash[1m]`。抽取代码为 `feat/catalog-extract-review-loop@70d72b57`，包含 dev 的 !126 和 !127；KGS、固定 Gems 源码及 20 个原始 timing case 均未变化。

实验根目录为 `runs/kernel_todo_v2/ppu-sdpa-codex-e2e-20260914-wJ3jTx/`。首次 `extract/` 产物被抽取 review 以 P1 接受，但独立下游 review 发现 correctness 返回单 Tensor、Definition 声明 9 项的真实问题并阻断。把该反馈交给 Codex 后，在新 `extract-02/` workspace 重新抽取，未覆盖旧产物。这是测试编排器手工反馈后重启抽取，不代表已经实现“下游失败自动回到上游”的跨 Workflow 修订机制。

第二份产物 `extract-02/catalog` 的 Bundle ID 为 `sha256:9bdc43819de5214f9631cff7e8c1bee2d3d7c723e185a7c98dfa07d1c6b529b2`。保留 17 个 correctness case、20 个 timing case；`correctness_run` 提供 9 项返回结构，`valid` 对照源 pytest 数值检查主输出并检查原有辅助 shape/max_q/max_k 断言，不对源测试未比较的未初始化辅助值追加数值检查；`timing_run` 保留原 benchmark 的单 Tensor SDPA baseline。原 pytest 的 bias 随机输入乘 0.1，未新增 float64 转换，未添加或删除原 case。

独立 CatalogOptimize 最初只拿到 Catalog 文件，未得到源 pytest/benchmark 证据，review 对辅助输出及 bias 生成产生误判。测试入口 `optimize_with_evidence.py` 仅为真实只读 reviewer 补充固定源码路径和覆盖报告；没有预填 review 结果、修改 Catalog 或 mock 任何执行步骤。补充证据后独立 review 通过。该现象说明两 Workflow 之间还需完善 provenance 交接，当前不能宣称完全无人介入的一次性链路。

PPU 随后真实通过 Preflight 和 reference-as-solution Eval（37/37 workload），并完成 KernelGen 的 2 个 epoch、4 个 Coder、8 个搜索轮次和 4 次最终独立复测；最终 CatalogOptimize 返回 `SUCCEEDED`，优化输出 `PASSED`，代码 review 无 P0。接线期间发现的通用 KG 问题分别由 !128（readiness 候选不应包含 evaluator 钩子）、!129（快照保留显式 None 默认值）、!130（gen_inputs 配方不等于调用实参）修复并经 PR 合入 dev。下游运行日志与 ledger 位于同一实验根的 `optimize-02/`；根目录 `summarize.py` 自动生成 `summary.json` / `summary.md`，记录确认后的成绩与资源回收情况。

验收边界：可选 Profile 仍因 KG 未使用 Bundle 绑定而失败，未取得性能计数器；Python lifecycle_status 的部分 Coder scope 轮次字段仍未派生 ledger；代码 review 保留数值稳定性和未覆盖参数/广播语义等 P1。这次主执行链路通过，不代表上述能力均已修复或完整通用算子正确性已获证明。结束时 PPU 16 个 slot 全部 healthy/available，active/waiting/checking/broken/incidents 为 0，Coder lease 与 active operation 登记为空，既有 Server 保持运行。

补充 host 门禁：抽取/契约相关 121 passed；Codex Runtime、CLI Runtime、生命周期和抽取审核相关 73 passed。未安装包、修改模型凭据或移动发布 tag。

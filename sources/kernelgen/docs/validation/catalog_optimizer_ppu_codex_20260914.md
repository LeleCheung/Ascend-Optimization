# Codex CatalogExtract → PPU CatalogOptimize 验证

结论：真实 Codex 抽取产物通过 review 后，独立 CatalogOptimize → KernelGen 完成全部调用并返回 `SUCCEEDED`，优化输出 `PASSED`。2 个 epoch、每个 epoch 2 个 Coder、每个 Coder 2 个搜索轮次，共 8 轮；其中 7 轮 PASSED、1 轮因辅助输出类型不匹配 PARTIAL_PASS，4 个 Coder 的最终独立复测全部 VERIFIED。代码 review 没有 P0，保留数值稳定性与未覆盖参数等 P1。通过范围是当前冻结的 17 correctness +20 timing 套件和主生命周期链路，不代表通用算子全部语义或所有可选能力通过。

## 环境与输入

用户因 Opus API 暂时不可用，明确指定 Codex 抽取。CatalogExtract 使用 Codex CLI 0.153.4 / `gpt-6-astra`；真实 review 和下游 KernelGen 使用 Claude Runtime / `deepseek-v4-flash[1m]`。没有模拟、跳过 review 或手工改写生成后的 Catalog。

KG 抽取分支为 `feat/catalog-extract-review-loop@70d72b57`，优化分支为 `feat/catalog-pipeline-optimizer-inputs@b61e0bd197e985f883733df6ae1c3dfe4cc20329`；后者已同步 dev 的 !128–!130。当前发布基线为 KG v6.3.1，以上是未发布的 feature/修复代码，不应把此结果直接归入旧 tag。KGS 为锁定的 v6.3.4@`00dfc8692a6927d052b0c6da238cdde7908363c4`，Protocol v6.2，Server 代码和部署未变更。

目标为 PPU-ZW810E / backend `thead` / timing `triton`；使用既有 `kt2-stage2-ppu-EdHzTP`，远端 loopback 22107，经 SSH stdio proxy 的本地 23107 访问。现有 Server 配置为 16 个 slot；Agent Coder 并发为 2，二者不是同一种资源。目标软件为 Triton 3.5.0、compiler 3.5.0+git43e6ef4d、ppu-sdk 2.1.3-ac7131、driver 1.3.2-d7f5a2，均从 KGS 返回事实读取，不由 Agent 本机推断。没有安装包、修改 Torch/Triton/厂商运行时、重启其他 KGS 或移动 tag。

算子为 PPU 历史未达标的 `scaled_dot_product_cudnn_attention`。固定 Gems commit `d64794e63b502cb836bc015a92a62c42de4be05a`；最终抽取 Bundle 为 `sha256:9bdc43819de5214f9631cff7e8c1bee2d3d7c723e185a7c98dfa07d1c6b529b2`，保留 17 个 correctness 和 20 个 timing case。历史 Triton 作为未预验证的 seed 输入，sha256 为 `088aa45c0f47667e095e4991ecb197b3a24275e068fea10d0e7d93e656d340fd`，历史 0.731× 不作为本次成绩导入。优化配置：`mode=kernelgen`、`n_parallel=2`、`n_epoch=2`、`min_rounds=max_round=2`、`profile_enabled=true`。本次目标是流程验真，不要求性能达到既定实验合格线。

## 运行证据与边界

本地实验根目录为 `runs/kernel_todo_v2/ppu-sdpa-codex-e2e-20260914-wJ3jTx/`，实验输入、日志及产物不随仓库分发。`extract-02/catalog` 为最终抽取产物，`optimize-02/` 为唯一的下游 workspace；修复接线后均在该 workspace 续跑，没有覆盖旧失败记录或变更冻结输入。

这次运行的是独立 CatalogExtractWorkflow 与 CatalogOptimizeWorkflow 的 Python 入口，不是 `kg run` CLI 后台提交测试。首次 Codex 产物存在单 Tensor/9 输出 ABI 不一致，真实下游 review 阻断后，测试编排器手工将反馈交给 Codex，在新抽取 workspace 修订；不是跨 Workflow 自动回退能力。独立 CatalogOptimize 缺少原源码 evidence 时产生了辅助输出/bias 生成方面的误判；本地 `optimize_with_evidence.py` 仅为真实只读 reviewer 补充固定源码路径及覆盖报告，不预填审核结果，也不替换任何 Workflow 调用。补充证据后 review 通过。

readiness 的原始证据位于 `optimize-02/stages/distribute_catalog/attempts/`：01 为复制 evaluator 钩子导致的 candidate admission 失败；02、03 已通过目标端完整 Preflight/Eval，但后续快照构造失败；04 成功完成交接并进入 KernelGen。`optimize-02/stages/optimize/work/1R/agent*/.ledger.json` 和 `2R/agent*/.ledger.json` 保存真实搜索轮次，独立复测保存于各 Coder 的 `.kernelgen/retests/`，最终成绩必须使用确认后的输出，不能取 Debug Job 调参日志里的最佳点。

实验根中的 `summarize.py` 从各 Coder ledger、final-verification、Workflow status 和 KGS `/status` 自动生成 `summary.json` / `summary.md`，具体加速比以这些生成文件和 `optimize-02/stages/optimize/work/kernelgen_output.json` 为准。最终候选为 `2R/agent0/.best_kernel.py`，SHA-256 `18b716193199c984043be08a4dfc33048b791217607d153bf0a1b4fea14e9e29`。`final-status.json` 记录终态与 `process_alive=false`，`server-after.json` 记录 16/16 healthy、active/waiting/checking/broken/incidents 全部为 0、available=16；active Server operation 登记为空，Coder pool used=0、lease 为空。保留既有 PPU Server，未因测试停止其他服务。

## 已通过 PR 修复的接线问题

| PR | 原分支 | dev 提交 | 原因与修复 |
| --- | --- | --- | --- |
| [!128](https://gitee.com/BaaiAC/kernelgen/pulls/128) | `fix/catalog-reference-readiness` | `db55eaff` | readiness 把 oracle 的 `gen_inputs` TF32 设置复制进候选，触发候选环境修改拒绝。仅从候选副本移除 evaluator 钩子，原 Bundle、oracle 和检查不变；独立模块导出 `correctness_run`，不重绑其内部 `run`。 |
| [!129](https://gitee.com/BaaiAC/kernelgen/pulls/129) | `fix/catalog-snapshot-null-defaults` | `fdf36207` | `exclude_none=True` 丢失 `attn_bias=None`、`scale=None` 的合法默认值。改为保留显式 null，同时不为 required 参数补默认值。 |
| [!130](https://gitee.com/BaaiAC/kernelgen/pulls/130) | `fix/catalog-input-factory-snapshots` | `29a59e31` | KG 把 `gen_inputs` 的 `case` 配方误判为函数实参。带显式输入工厂时保留配方，由 KGS 实际生成并绑定验证；无工厂时仍严格校验静态实参，名称唯一性等检查不变。 |

三个修复均通过独立 PR squash 合入 dev，未直接 push dev；确认 PR 状态和源/目标文件树相同后清理了对应临时 worktree 与分支。

## 尚未计为通过的部分

- 可选 Profile：KG `tools/profile_round.py` 仍用 `catalog_name=uploaded-…` 构造绑定，未使用冻结的 Bundle ID，目标端因此返回 catalog unavailable；未获得计数器或 profiler artifact。已保留真实 failed analysis，不能据此作性能瓶颈结论，也不能把它宣称为已验证能力。
- 单算子子 scope 的状态展示：生命周期/epoch 任务计数能推进，但 `lifecycle_status` 中部分 Coder scope 的 `completed_rounds`、best 字段仍为初始值，与真实 ledger 不一致。这里只观察到 Python status 入口的现象，尚未验收 CLI 后台 `kg status`；实测轮数与成绩以 ledger 和独立复测为准。
- 独立抽取→优化的源码 evidence 交接、跨 Workflow 自动修订尚未完成产品化；本次明确使用了测试入口补充证据，不能称为全自动一次运行成功。
- 最终代码 review 保留 P1：去掉 softmax running-max 后的大幅值稳定性、未数值验证的辅助输出、未覆盖的 dropout/debug-mask 参数和广播形式等。当前全套 case 通过不应被解释为这些未覆盖调用均正确；本次不扩大测试覆盖或修改原 reference 来处理它们。

## Host 门禁

优化 feature 同步全部公共修复后，155 项相关 host 测试通过：`test_snapshot_input_factory`、`test_evaluation_snapshot`、`test_catalog_optimizer_modes`、`test_catalog_reference_readiness`、`test_catalog_optimize`、`test_extract_and_optimize`、`test_kernelgen_uploaded_snapshot`、`test_cli_lifecycle`、`test_kernel_gen`、`test_worker_pool`、`test_workflow_names`。测试通过临时父目录中的 `kernelgen` 链接固定导入当前 feature，KGS 导入锁定主工作区；先核对了两个包的 `__file__`。没有把其他 editable install 的结果作为门禁。

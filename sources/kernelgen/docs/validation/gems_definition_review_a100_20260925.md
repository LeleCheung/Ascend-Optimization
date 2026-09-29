# A100：Gems 新 pytest → Definition → KernelGen CLI 完整验收

结论：`relu6_` 的真实后台流程已通过。该算子在 Gems 中有实现、正确性 pytest 和 benchmark，但不在本次 KGS 内置 `flaggems-adapter-definitions` 中。`kg definition` 从干净 Gems checkout 导出本地 Catalog，`kg run --catalog-path` 自动上传 Bundle；默认测试审核、15 个 core baseline case、候选生成、Preflight、33 个原始 pytest/benchmark case、独立最终复验、Profile 和 advisory 代码审核均完成。`kg status --detail` 的终态是 `SUCCEEDED`，四个阶段 4/4 完成、runner exit 0。该结果只证明这一个算子在本次 A100 环境与冻结测试契约下跑通。

## 实际组合和资源

- KG：`codex/unified-review-tests` 开发快照，包含 `kg definition`、统一 `review_tests`、可读源码证据和取消重试修复；实测 KGS 为 v6.3.4 开发快照 `ba05292bfad7234dd94636567f4290404e1321cf`，Protocol v6.2。随后 KGS !64 以 squash 合入 `main@3d00e93c592847b9fc3cf8f03a5345a3466dc7ae`，核对文件树与实测提交完全一致，KG 锁定清单已更新至合并提交。KG/KGS 软件 release 标签没有因本实验修改。
- Gems：官方 `kernelgen-dev@4772d816bc5d52d37c4718f36adf377d81261f83`，包含已合入 master 的 `--reference-only`。真实进程核对 `flag_gems.__file__` 和 `kernelgen_server.__file__` 分别来自本次冻结 Gems worktree 和配套 KGS feature worktree。
- 本地 `kernelgen-nvidia-cu128` 容器，NVIDIA A100-SXM4-40GB；解释器 `/data/jiabei/venvs/kernelgen-cu132-torch211-triton360/bin/python3`，Torch 2.11.0、Triton 3.6.0。目标 KGS 仅监听容器 loopback；现有 KGS Debug Job 为每次临时 KGS 独占分配一个单卡 slot。同一时间临时 KGS 只看到一张卡，调度 `max_active=1`。未安装、升级或替换受保护运行时包。
- 本次为本地 Agent + 本地目标容器的 A100 验收。H20 有原驻留进程，因此未在 H20 共用设备；远端 SSH stdio 部署及其他芯片能力不由本结果证明。

## 输入、执行和结果

`kg definition --flaggems-repo <本次冻结 checkout> --pytest-path tests/test_relu6_.py --workspace <definition-workspace>` 输出 `definition/catalog`，Definition SHA-256 为 `4c032e428e136edb97a413e7c749ad9ac0078ce2860c4ab7282d2d6f7e8eaab4`。KGS 内置 `definitions/relu6_.json` 确认不存在。`kg run --mode kernelgen --definition relu6_ --catalog-path <definition/catalog> --eval-server <临时 loopback KGS> --max-round 1 --min-rounds 1 --n-epoch 1 --n-parallel 1 --timeout 600 --workspace <optimize-workspace>` 真实后台运行，无 seed、无 dummy、无 `--skip-review`。Bundle ID 为 `sha256:099b6144e10fc497eb77e93dbd4aaa1cd7b9c495a39fb5a4724afcb89963ffef`；prepare、review 和优化器使用同一冻结 binding。

| 验收项 | 实测 |
| --- | --- |
| 测试契约审核 | ACCEPTED，缺失依赖 0；原 pytest 覆盖不足作为 source-quality 建议记录 |
| 生成前 baseline | KGS `/reference` PASSED，core 15/15；FP16、FP32、BF16 各 5 |
| 候选 | Coder 生成纯 Triton 原地 `relu6_`，Preflight PASSED；没有 Torch op fallback |
| 搜索 round 1 | Eval PASSED，33/33；ledger `geo_mean=1.0029776829246704` |
| 独立最终复验 | PASSED，`geo_mean=1.0106983850179179`；与搜索成绩分别保留 |
| Profile | NCU Profile 调用完成，ProfileAnalysis 落库，ledger profile=`completed` |
| 代码审核 | advisory 审核完成，报告绑定 best kernel 摘要；2 条建议，不改写评测结论 |
| CLI 状态与历史 | 181 份过程中 status/scheduler 采样；终态 `SUCCEEDED`、4/4 阶段完成、exit 0；`kg history` 只有 1 个 measured round，与 ledger 一致 |
| 清理 | Agent/Coder 均退出；临时 KGS 及父 Debug Job 已结束；父 scheduler 恢复 8/8 healthy、active/waiting/checking/broken=0、incidents=0 |

从同一任务的旧 workspace 恢复时，prepare 和 review_tests 回执复用，稳定优化目录的 ledger 仍只有原 round 1；成功的 optimize 回执复用后只运行末端代码审核。没有用新任务覆盖旧 ledger，也没有将最后复验分数写回搜索 ledger。

## 实验中发现并修复的问题

首次模型审核反复读取 `test-sources.json` 中承载 `benchmark/base.py` 的超长 JSON 字符串行，Read 工具报告 29,156 tokens，超过单次 25,000-token 上限，按行分页也无法读取。KG 现在保留原 JSON 作为冻结事实，并在每个审核 attempt 派生带摘要索引的逐文件原样文本；测试审核和代码审核共用可读证据，完整阅读门禁不降低。第二次审核指出 KGS 漏导出 `benchmark/consts.py`、`profile_hook.py`、`core_shapes.yaml`；KGS 已补齐这几项及 case/input 辅助文件，再审核通过。两个旧 attempt 保留为失败证据。

设备预约有明确期限，第一次优化在 ProfileAnalysis 记录后收到协作取消，于安全点 exit 130；恢复后保留 round 1 与 ProfileAnalysis。旧 Runtime 对结束且可恢复的模型调用会在已有取消请求下继续超时重试；KG !164 已增加“上次调用结束、下次重试开始前”安全点，不中断当前模型输出。旧代码审核 attempt 因继承原始 JSON 再次超长；改为复用已登记的可读测试证据，原任务恢复后仅重跑代码审核并完成。预约超时的父 Debug Job `FAILED` 只是测试 harness 的期限终止，不是算子 Eval 或 KGS slot 故障。

## 证据与范围

本机原始证据目录：`/data/akg_kernel_bench_lite/kernelgen/runs/gems-definition-review-e2e-86KA2I/`。其中 `definition/catalog/` 为输入，`reference-smoke/` 为独立 KGS 接口测试，`optimize-complete-evidence/` 为成功任务，`observations/` 为真实 CLI status/history 与 Server scheduler 过程快照，`acceptance.json` 是逐项核对冻结源码摘要、四阶段回执、ledger、最终复验和进程退出后的验收报告，`target-environment-verified.json` 记录实际包版本和导入路径。被中断的原 `optimize/` workspace 保留以供排查，不是成功任务。

Host/API 回归在现有本地环境中分别达到 KG 309 项、KGS 103 项，包含公共 Runtime 取消与审核路径；Gems 同步 master 后 117 项 core 测试通过。它们补充真实设备证据，但不说明所有 Gems 算子或国产芯片都已通过。跨芯片复测应在独立 Workflow 中使用同一冻结 Definition 与 best candidate，按每台 KGS 的实际状态分别记录结果。

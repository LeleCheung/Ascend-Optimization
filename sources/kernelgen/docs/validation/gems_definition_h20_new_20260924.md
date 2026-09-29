# H20：Gems Definition → OperatorOptimize 真实后台 E2E

结论：本轮已通过。新生成的 Gems adapter Definition 被实际上传并用于 KGS inspect/Preflight/Eval/Profile；随后通过标准 CLI 后台提交运行 OperatorOptimize → KernelGen，最终 `SUCCEEDED`，4/4 阶段完成，进程 exit code 0。不是 dummy、不是 Native workload 转换，也不是同名内置 Definition 的替代测试。

## 实际组合与配置

- KG v6.5.1 开发快照 `02f1d9aeca90f8c328ec751ec95346283f8fb066` / KGS v6.3.4 开发快照 `6ea41df44402fedae13f555ef084ef4e7e306583` / Protocol v6.2。本轮没有发布新 tag。
- Gems 官方仓库 commit `299e2659df543b35ac9dbede78e2818ad9e30590`。启动时分支名为 kernelgen-dev-new，现已改名 kernelgen-dev；本轮固定 checkout 未随重命名或分支更新而变化。
- H20 `115.191.21.142`，容器 `kernelgen-nvidia-ngc-2504`，官方镜像 `nvcr.io/nvidia/pytorch:25.04-py3`；物理卡 7，共享原驻留进程但不停止它。共享卡上的数字用于流程验收，不作独占环境的性能结论。
- 解释器 `/data/xuyao/kernelgen_hosts_20260923/venv/bin/python`；Torch `2.7.0a0+79aa17489c.nv25.04`、Triton `3.2.0`，本轮没有新增包，受保护依赖前后快照相同。
- KGS loopback 18308，经本地 SSH stdio proxy 19808 访问；1 个 slot、4 个请求线程。KG 在本地运行，模型请求为 `deepseek-v4-flash[1m]`，Runtime 为 Claude CLI。
- 优化配置：kernelgen、1 Coder、1 epoch、max-round 10、Profile 开启，无 seed/reference code。实际 6 轮后由连续 3 轮无提升的停止策略结束；max-round 是上限，不要求凑满。

## 输入与测量事实

输入为同一 clean checkout 的 `tests/test_negative.py`。GemsAdapterDefinitionWorkflow 在 CPU Agent 主机只读源码生成 Definition；ABI 参数 x，输出 out，没有本地导入 Torch/Gems，也没有改写源测试。输出 Definition SHA-256 为 `df065f68daaeee841d18135e68591f53a7edeea882c1d09ccb10f3862f5ae010`，实际 Bundle 为 `sha256:84667c402e9e7d17439c209ba593afa6d089590dec800b4550c893fff040c2d9`。

prepare、KernelGen、Coder 三份冻结执行契约均绑定同一 Bundle，evaluator 均为 flaggems。KGS 使用原始 18 个正确性 case 和 15 个 core timing case；每轮均 PASSED 33/33，候选调用覆盖由原生 pytest 报告校验。静态审查确认 best code 采用 Triton kernel，没有调用 torch.neg/torch.ops 代替实现。

| round | Eval | 搜索阶段 geo_mean |
|---|---|---|
| 1 | PASSED | 1.001225 |
| 2 | PASSED | 1.036185 |
| 3 | PASSED | 1.039885 |
| 4 | PASSED | 1.023711 |
| 5 | PASSED | 1.029308 |
| 6 | PASSED | 1.038772 |

搜索 best 为第 3 轮，1.039885×；独立最终复验为 **PASSED / 1.037000×**。两者分别保存，不用一次复验覆盖搜索 ledger。任务历时约 42.27 分钟，包含模型、Profile 和收尾，并非纯 Eval 时间。

## Profile、状态与失败恢复

手写独立 smoke 候选的 Preflight、Eval 33/33 和 NCU metrics Profile 全部通过，但该代码未作为优化 seed。真实 Coder 流程记录 4 次 PROFILE_COMPLETED 聚合事件，落地 15 份 `.ncu-rep`；聚合事件数量不等于单 case HTTP 请求数量。

`kg status` 和 `kg status --detail` 的过程快照覆盖 RUNNING、共享分析、Coder、Preflight/Eval/Profile、代码审核及最终 SUCCEEDED。basic scope 没有虚构 round/epoch 字段；history 的轮数、best round 和搜索分数与 ledger 对齐。最终 best code、solution SHA、final-verification、KernelGen 输出一致，active server operations 已清空。

模型任务退出后，再并发提交一次预期失败 Eval 和一次合法 Preflight。错误被结构化为 RUNTIME_ERROR，合法请求 PASSED，scheduler 全程 active/max_active 不超过 1；结束后 active/waiting/checking/broken 均为 0，healthy/available/device_slots 均为 1，incidents 未增加。临时 KGS/proxy 已按 PID 启动身份关闭，原驻留进程和全部证据保留。

## 验收边界

advisory code review 完成并记录三条 P1：大 stride 索引的未覆盖边界、性能收益较小且存在测量噪声、源 pytest 缺少非连续输入等覆盖。这不等于候选在任意输入上已获生产认证；本轮结论仅为给定原始 pytest 契约下的真实 E2E 跑通。也没有验证新分支全部 68 组算子或所有芯片。

Host 回归为 KG 144 passed / 1 skipped（Agent 主机无 Torch），配套 KGS 在 H20 环境 151 passed。Gems 新分支的独立迁移验证为 79 项回归通过、54,347 个默认正确性 case 与 10,160 个 quick case 成功收集、1,618 个唯一 benchmark case ID；收集结果不是正确性运行结果。

## 证据与查看命令

实验目录：`/data/akg_kernel_bench_lite/kernelgen/runs/gems-definition-h20-new-20260924`，内含 acceptance.json、extra-checks.json、cleanup.json、观测快照、完整请求、Definition、smoke 结果、进程身份与复现脚本。工作区为 `/data/akg_kernel_bench_lite/kernelgen/runs/gems-definition-h20-new-20260924/optimize`，best code 位于 `stages/optimize/work/1R/agent0/.best_kernel.py`，性能以同目录 `.ledger.json` 和 `.kernelgen/final-verification.json` 为准。

实验时使用的 feature checkout 查看命令如下（临时 worktree 清理后不再可用，不会启动任务）：

```bash
PYTHONPATH=/tmp/kg-definition-import-poiiZQ:/data/akg_kernel_bench_lite/worktrees/kgs-gems-definition-bundles python3 -m kernelgen.cli.main status /data/akg_kernel_bench_lite/kernelgen/runs/gems-definition-h20-new-20260924/optimize --detail
```

这次 feature 的代码尚未因本报告自动发布；源分支版本和运行证据不能当作未来 main/dev 任意 HEAD 的验证。新的 Gems 自动更新策略独立开发，KGS 仍保留 exact commit 锁定。

## 主线整合复验（2026-09-24）

来源策略已通过 KG !160 / KGS !62 合入，Bundle 实现通过 KGS !63 合入。KG Definition 分支同步主线后改用正式 CLI 初始化模块，并锁定 KGS `bfea8b4f2963b05faafc3786d8cb94b5d2ce0a24`。KGS 运行实现与上述真机快照一致；新策略只在实验前准备阶段生效，不改变本轮 Gems `299e2659` 的实测事实。联合 host 回归为 KG 180 passed / 1 skipped、KGS 45 passed / 11 skipped；跳过均因本机缺 Torch。整合时未重新启动模型或真机服务，不能将这些 host 测试当作新一轮 E2E。

使用已安装的合入后 KG 查看保留的实验记录：

```bash
kg status /data/akg_kernel_bench_lite/kernelgen/runs/gems-definition-h20-new-20260924/optimize --detail
```

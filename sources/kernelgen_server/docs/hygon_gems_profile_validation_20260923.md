# 海光 Native/Gems Profile 修复与验证（2026-09-23）

结论：海光 Gems `/profile` 的 completion marker 失败已修复，Native/Gems 均返回 `completed` 并具有有效 PMC 记录；故意失败的 candidate 仍返回 `failed`，其原始 pytest 错误保留在 `hipprof.log`。本报告只覆盖海光，不扩大为其他厂商或全部算子的验收结论。

## 原因与修复

1. 海光 profiler 原先用报告目录覆盖 `ProfileCommand.cwd`。Native 命令使用绝对路径，不易暴露问题；Gems 命令使用仓库相对路径 `benchmark/test_addmm_.py`，因此 pytest 找不到 conftest，报 `unrecognized arguments: --profile-only --case-id ...`。hipprof 又将子进程失败包装成退出码 0，KGS 的 completion marker 校验正确拒绝了该结果。修复保留 evaluator 指定的 cwd，报告路径仍为绝对路径，不删除 marker 校验。
2. 此版本 DTK 的 PMC CSV 实际写到 evaluator cwd，不跟随 `-d/-o`。现在按 hipprof 输出的进程 PID 回收本次 `pmc_results_<pid>.csv` 到独立 artifact 目录；不扫描或回收同目录其他任务的文件。回收使用跨文件系统安全的移动，适用于 Gems checkout 与报告位于不同挂载的部署。
3. PMC 中的 C++ linker symbol 与 trace 中的展开函数名不同。现在通过目标环境已有的 `c++filt` 展开后精确匹配，保留原始 symbol，不使用子串匹配；仍只选择 measured invocation 对应的尾部计数器，不把全部 setup/warmup 记录混入结果。缺少展开工具时不猜测匹配，仍如实报告缺少有效 PMC。
4. 通用进程执行器在 marker 缺失时保留 wrapper 原始输出，由海光 profiler 落盘，避免再次只有泛化错误而丢失 pytest 失败证据。

## 实测环境与范围

- KG v6.5.1 所在主工作区仅作为 SSH stdio proxy 发起端，没有修改其 CLI 或用户工作区内容。
- KGS v6.3.4 开发分支 `feat/flaggems-profile-hook`：原失败版本 `2d3170d`，cwd 修复 `8bb3809`，PMC 与失败日志修复 `44f1cdec689df42fdc62cdea30e9c16191836eba`；Protocol v6.2。代码通过 Gitee 同步，没有覆盖厂商运行时。
- FlagGems `feat/kernelgen-profile-hook@962327660fd735a1abad22061e19b6d3e4130a44`，无需修改 Gems。
- 目标 `10.232.2.26`，容器 `codex_fib_hygon_20260728`，物理卡 6，`HIP_VISIBLE_DEVICES=6`，KGS 中为 `cuda:0`；单 slot、单 worker。KGS 仅监听远端 `127.0.0.1:24122`，本地通过 SSH stdio proxy 的 `127.0.0.1:25122` 请求。
- 解释器 `/workspace/kernelgen_e2e_20260728/deployments/stage2-kg631-EdHzTP/venv/bin/python`；目标 `/status` 报告 Triton 3.6.0、FlagTree 0.6.1a1+hcu3.6、ROCm 6.1.25065、driver 6.3.30-V1.4.1a。未安装或升级依赖。

验证设置为 `level=metrics, warmup=1, iterations=1, timeout_sec=300`。Native 使用 `kernelgenbench_add` 的 reference-as-solution，Gems 使用 `addmm_` 的真实 override candidate，调用 Gems `addmm_kernel`。这是两种 adapter 的协议、wrapper、产物与资源释放验证，不是同一算子的性能对照，也没有把 profiler timing 当作 Eval latency。

| 请求 | Profile ID | 状态 | PMC 结果 |
| --- | --- | --- | --- |
| Native | `84897f822c8245c3962081b48b405d6b` | completed | 原始 4 条，选择 measured 2 条；2 个目标 kernel |
| Gems | `e45870ee1623445e981de789694f92da` | completed | 原始 10 条，选择 measured 1 条；`addmm_kernel` |
| Gems 故意抛出 RuntimeError | `ee791d62e11b479c8e4ae2af1a4788c1` | failed（符合预期） | 未伪装成功，日志包含 `intentional-profile-failure` |

两个成功结果均包含 `VALUBusy`、`LDSInsts`、`LDSBankConflict`、`L1MemoryBusy`、`L1MemoryStalled`、`L2WriteUnitStalled`、`L2CacheHit` 等原始计数器；此前的 `hipprof exported no valid measured PMC records` 警告消失。所有请求结束后均满足 `active=waiting=broken=checking=0, available=healthy=device_slots=1`。

Host 回归在隔离复制的测试入口运行，导入路径已核对为本 feature worktree。宿主机无 Torch 时为 71 passed / 8 skipped；在已有 A100 容器的 Python 环境补齐同一测试集后为 79 passed。随后补充跨挂载回收的 `EXDEV` 注入回归，最终为 **80 passed**，包括 cancellation、process control、Gems runner、vendor parsers、profile lifecycle 和新增 cwd/PMC 回收/失败日志回归。修复前新增相对路径回归失败，绝对路径 Native 回归通过，修复后两类均通过。跨挂载的移动回退由 host 故障注入验证；真机记录对应上述 `44f1cde`，没有声称真机环境使用了跨挂载部署。

## 证据与收尾

本地证据目录为 KG 工作区的 `runs/hygon-profile-fix-20260923/`，包含请求、完整 `/profile` 响应、前后 `/status`、下载的厂商 CSV/trace/数据库、复现输出及 `remote-evidence.tar.gz`。其中 `verify.py` 经项目 SSH stdio proxy 发起请求，`collect.py` 从 artifact API 下载产物；实验脚本不属于产品接口。

远端现场保留在 `/workspace/kernelgen_e2e_20260728/hygon-profile-fix-ujHD83/`。本次 cwd-only 验证写入 Gems/Catalog 目录的 3 个确切 PID 的 PMC 文件已移入该目录的 `pre-fix-pmc/`；没有删除其他任务文件。临时 KGS 和本地代理已停止，旧服务未改动，原始证据未删除。

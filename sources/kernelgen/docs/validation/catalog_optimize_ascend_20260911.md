# 2026-09-11 Catalog 生命周期昇腾验证

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

## 结论与范围

固定环境中的 `extract_catalog → review_catalog → distribute_catalog → optimize → code_review` 五阶段真实 E2E 通过，根状态 SUCCEEDED、completed_tasks=5、进程退出码 0。独立复验全部 workload 通过且达到项目当前 0.8 合格线，但候选没有快于 Torch reference。Catalog 审核和代码审核各留有一个非阻塞 P1，不能把本次有限用例通过理解为无缺陷或跨设备验收。使用真实模型、上传的 native Catalog 和昇腾 KGS，不含 PR、Batch、多目标并发或动态镜像/虚拟环境。实现与后续多环境边界见 [设计文档](../design/workflows/catalog_optimize.md)。

## 实际版本与环境

最终续跑使用 `KG v6.2.1@fb5523221b4c67013d9a65f94bdccadd7acad779 / KGS v6.3.1@a5749f5fc4994be1cf5e770c77951eccc56eaa32 / Protocol v6.2` 开发快照，不是新的 release tag。KG 分支为 `feat/catalog-optimize-ascend`，KGS 分支为 `feat/operator-bundle-execution`；目标 KGS 通过 Gitee 临时分支 `test-bundle-ascend-20260911` 精确部署，没有向目标复制 Server 源码。KG 的锁定清单指向该 KGS commit；运行时仍只根据 `api_version` 和 capabilities 判断兼容性。

连接依据为当时的 `tests/multi_device_batch_hosts.conf` 中 ascend 条目（现行清单为 `tests/hosts.md`），目标容器 `codex_fib_ascend_20260821`，芯片 Ascend 910B4-1。经用户授权暂停原有全卡 KGS，再以同一容器、同一解释器启动单卡测试实例；没有并存两个占用同一卡的 KGS。测试使用物理卡 0、backend=npu、timing=profiler、1 device slot、2 request workers、1 Coder lease。KG、模型 Runtime、Catalog 和 workspace 均在本地；通过 `127.0.0.1:19606` SSH stdio proxy 访问远端 `127.0.0.1:18306`，没有公开远端端口。

目标解释器为 `/home/secure/xuyao/kernelgen_e2e_ascend_20260821/deployments/ascend-managed-20260910/venv/bin/python`。实际模块版本为 Torch 2.9.0+cpu、Triton 3.5.1、torch_npu 2.9.0.post2；KGS 返回 CANN 9.0.0、driver 25.2.0、architecture dav-c220-cube。模型 Runtime 为 Claude Code，实际模型 `deepseek-v4-pro[1m]`。测试没有更换这些核心运行时；Server 管理记录的 10 项依赖 before/after 完全一致、added=[]，没有新增 Python/npm 包。

## 输入与执行

算子 `relu` 从干净的 Gems `d64794e63b502cb836bc015a92a62c42de4be05a` 抽取；目标侧通过 Debug Job 执行原 benchmark 的 `--level core --list-cases`，得到全部 15 个 timing workload。抽取产物包含 18 个 correctness workload，覆盖 float16、float32、bfloat16；没有删减大尺寸 workload。目标 Gems 源码另行 clone 并固定 commit，不改原实例的 Gems 安装。

配置为 SimpleOpt：max_round=3、min_rounds=1、early_stop_rounds=1、max_coder_sessions=1、warmup_ms=100、benchmark_ms=100、num_trials=1、eval_timeout_seconds=1500。前台入口为 `python3 -m kernelgen.examples.catalog_optimize.run_example run --workspace <workspace> --input <input.json> --runtime claude`，续跑追加 `--resume`。现有 `kg run --mode lifecycle --dummy` 没有改成真实 Catalog 入口。

抽取与 Catalog 审核各只产生一份成功凭据，取消后的两次续跑均复用这些凭据。上传的 Bundle 为 `sha256:15c7aed656cb6e622ae44249bbcff2d827944313e7f1eb90a5fd729a475924ff`，目标 inspect 指纹为 `sha256:f85dc519487b590fc2dc7ebe5920d8bd929b5a11fe31298859f6cab2bc9f6c1a`。完整 reference-as-solution Preflight/Eval 已通过；这属于分发阶段的就绪验证，不写入优化 ledger，也不作为候选优化成绩。

## 优化结果

| 评测 | 状态 | 通过/全部 | 加速比 |
|---|---|---:|---:|
| 搜索 round 1 | PASSED | 33/33 | 0.337333x |
| 搜索 round 2 | PASSED | 33/33 | 0.815429x |
| 搜索 round 3 | PASSED | 33/33 | 0.861689x |
| 最终确认 | PASSED | 33/33 | 0.861807x |

本表由 `summarize.py` 从 ledger、最终输出和独立复验结果生成。搜索只执行 3 轮；最终输出采用独立复验确认的分数，不回退采用搜索分数。确认代码 SHA-256 为 `44cd4d0fa157a9f8316a2cd671d4b471b7d62316fda212728fb901531986cfb9`，与代码审核绑定的摘要、`.best_kernel.py` 以及最终输出 best_code 一致。

首版候选因 launch grid 超过 CANN coreDim 上限被 Preflight 拒绝，Coder 通过远端 Debug Job 诊断并修改候选后才进入正式 Eval；后续调优同样重新经过 Preflight。诊断数据没有代替正式 profiler 计时，也未修改 Catalog、oracle 或 workload。最终代码审核按现有 P0 阻塞/P1 非阻塞门禁完成，没有自动应用审核建议。

## 取消、恢复和故障覆盖

KGS Bundle smoke 验证包括 inspect、Preflight、reference-as-solution、故意触发的 `RUNTIME_ERROR` 和两个并发 Eval：正常请求均通过，故障请求保持 `RUNTIME_ERROR`，并发采样为最多 active=1、waiting=1，没有复制设备 slot。

第一次在途 Eval 取消后，KGS 强探针恢复原 slot，incidents=recovered=1、broken=0，但暴露了 KG 将远端取消误记为 FAILED 的问题；同时补齐公开 lifecycle cancel 对 active KGS operation 的转发。修复后的真实复验通过公开 `cancel_lifecycle()` 取消在途 Preflight：进程退出码 130、根状态 CANCELLED、completed_tasks=2，active operation 登记清空。再次续跑仍复用同一个单算子 workspace，generation 更新，未重做已经成功的抽取与审核。失败和取消尝试均保留，没有清空 workspace。

另一个本次修复是 typed Eval response 的 JSON 序列化：KGS evaluate 返回模型对象，分发阶段须先调用 `model_dump(mode="json")`。修复后已完成真实完整 reference 评测。

## 服务恢复与资源释放

最终测试实例的 scheduler 为 healthy=available=1、active=waiting=checking=broken=incidents=0。随后测试实例按 PID/start identity 校验停止，原 `ascend-20260910` 实例恢复为 `7f1110be26e1d3a37195dbf5fee96cc99ca516bb`；Debug Job 再次确认原 KGS 导入路径、解释器及 Torch/Triton/torch_npu 版本，均与切换前一致，原实例配置未变。原服务最终 8/8 卡健康空闲，active=waiting=checking=broken=incidents=0；SSH proxy 保留，旧服务仍返回 evaluation_binding=false，没有被静默升级。

Lifecycle owner 已释放，active-server-operations 为空，Coder lease 文件无本次租约；五份成功阶段凭据及其文件摘要通过汇总脚本校验。本地、Gitee 及目标 clone 中的 `test-bundle-ascend-20260911` 临时指针已删除，目标测试源码保留在 clean、detached 的精确 commit 上，正式 feature 分支和实验产物保留。没有删除原 campaign 或取消/失败证据。

## 尚未扩展的边界与现场问题

Catalog 审核给出一个非阻塞 P1：抽取 oracle 丢失 Gems 原有 fp64 capability guard。目标 Debug Job 确认本次昇腾 `.to(torch.float64)` 实际降为 float32，与源测试当前采用的 reference dtype 一致，因此本次目标没有由此失败；该结果不证明跨设备语义完全等价。没有手改冻结 oracle 或放宽容差来通过验证。后续推广到其他目标前应修正抽取并使用新的执行计划复验。

取消 NPU profiler 时，已有 profiler 实现可能在 KGS checkout 留下 `profile_results_<timestamp>`，从而被 clean checkout 校验正确拦截。此次仅将确定的生成目录保存在独立 Server state 的 `cancelled-profile-evidence/`，没有删除证据、增加 Git ignore 或削弱启动检查；该独立问题没有混入 Bundle 执行代码。

代码审核另给出一个 P1：`.best_kernel.py:38` 的 FP32 tile 选择使用向下取整，`:60` 的 launch 分支使用向上取整；在 `65535*8192 < numel < 65536*8192` 的范围可能误选较大 tile 的 grid-stride 路径。Reviewer 根据代码注释指出潜在编译风险，本次没有在这个额外尺寸区间执行真机复验，因此不能把潜在失败表述为已经复现。没有修改已评测 best code；后续修复需要新的评测证据。

测试启动曾因覆盖 PYTHONPATH 丢失容器的 CANN 路径，修正一次性测试控制脚本为前置测试 KGS 路径并保留原路径后，目标 architecture metadata 完整。此修正不涉及目标依赖安装。

真实 `/profile` 接口、其他算子/芯片、KernelGen 多 Coder、自动 review + fix、PR 和多环境 worker 不在本次 E2E 范围。Eval 使用真实 NPU profiler 计时，不等于独立 Profile 接口已完成真机验收。

## Host 回归与原始证据

KG 相关 host 回归为 347 passed，覆盖 Catalog/lifecycle、snapshot、KGS adapter、SimpleOpt/retest、CLI/Batch/取消与 Runtime；KGS 相关 host 回归为 77 passed，覆盖 Bundle/schema、native adapter、隔离执行、设备 slot、Profile 生命周期和 operation cancellation。KG 使用显式 PYTHONPATH，KGS engine 测试在本地 `kernelgen-nvidia-cu128` 容器中运行；两者均先检查导入路径来自对应 feature worktree。模拟设备的 host 检查不计为昇腾实测。

原始证据位于本地忽略目录 `/data/akg_kernel_bench_lite/kernelgen/.kernelgen/experiments/catalog-ascend-20260911`，不随 Git 分发。单算子 workspace 为该目录下的 `campaign/operators/relu`。`run-command-*.json`、`run-exit-*.json` 保存每次代码版本、完整命令和退出码；`workflow.log`、各 stage 的 `attempts/01/` 等目录保存模型日志、Catalog、审核及目标结果；`stages/optimize/work/` 保存原优化 session、ledger、最终输出和 best code。

`control.py` 保存原配置和进程身份，负责切换及恢复；`server_checks.py` 保存 Server smoke 结果；`cancellation-check-*.json` 和 `cancelled-progress-02.json` 保存真实取消证据；`summarize.py` 从结构化结果和 ledger 生成验证摘要，不导入或执行候选。远端测试目录为 `/home/secure/xuyao/kernelgen_e2e_ascend_20260821/catalog-e2e-20260911`，其中 `server-state/` 保留 Server 日志与取消 profiler 的证据。

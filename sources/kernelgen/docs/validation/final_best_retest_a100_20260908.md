# 2026-09-08 最终 best 复验 A100 验证

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

## 结论

`feat/final-best-retest` 完成 219 项相关 host 回归和 A100 上 Codex SimpleOpt 验证。强制复验采用新测得的双边时延，额外复测使旧输出失效，恢复后采用最新确认分数；搜索 ledger 始终只有 1 轮。多 Coder 汇总与恢复仅选择最终确认结果，不重新采用搜索高分。设计见 [最终 best 独立复验与 Agent 复测请求](../design/workflows/final_best_retest.md)。

## 实际版本与环境

组合为 `KG v6.2.1@af9086c2 / KGS v6.3.1@a02cb3883e93d179102fb92452e30476f9f401e2 / Protocol v6.2` 的开发快照；首次 Codex 生成与强制复验执行于 KG `4099df89`，补齐 epoch 汇总后在 `af9086c2` 完成额外复测、workflow 恢复和真实产物汇总校验。软件版本来自当前包元数据，未创建 release tag。本轮是开发功能验收，不宣称已成为 compatibility.yaml 的正式推荐发布组合，也未修改现有发布 pin。

本地容器为 `kernelgen-nvidia-cu128`，实际镜像 `kernelgen-nvidia-cu132-nsight2026.1:base`，image ID 为 `sha256:e5806e3c985b763ff6e3ebc95c4f0bca7b54f1c1e4ba4c145d73721349b0a53c`。使用物理卡 0（A100-SXM4-40GB），`CUDA_VISIBLE_DEVICES=0`，Server backend=cuda、timing=triton、1 slot、max-workers=2；Agent workers=1，runtime=codex，继承配置后实际模型为 gpt-6-astra。通过 Server Debug Job 记录 Python 3.12.3、Torch 2.11.0+cu130、Triton 3.6.0、CUDA 13.0，解释器为 `/data/jiabei/venvs/kernelgen-cu132-torch211-triton360/bin/python3`。没有新增 Python/npm 包，没有安装、升级或替换 Torch、Triton 或厂商运行时。

KGS 从 Gitee 的 main 克隆并确认上述 commit；独立服务只绑定 `127.0.0.1:18853`。启动前卡 0 无运行负载；调度和每次 Eval 的 acquire/release 均保留在 request audit。结束时 `device_slots=healthy=available=1`，`active=waiting=checking=broken=incidents=0`，Agent 为 SUCCEEDED、COMPLETED，active-server-operations 为空；随后向该专用服务进程发送 SIGTERM 关闭。

## 实测结果

| 阶段 | 结果 | 有效 workload | 加速比 |
|---|---|---:|---:|
| 搜索 round 1 | PASSED | 36/36 | 1.048449x |
| 自动最终复验 | PASSED | 36/36 | 0.967197x |
| 额外复测 | PASSED | 36/36 | 0.974114x |

使用内置 native Catalog `kernelgenbench` 的 `kernelgenbench_square`，warmup_ms=10、benchmark_ms=10、num_trials=1、max_round=min_rounds=1，profile 关闭。为 Codex 传入了既有 square Triton 作为 reference Triton；本轮不依赖 Gems，不代表 Gems 全量或跨厂商复测通过。候选 preflight 覆盖 9 个 specialization；每次评测包含 27 个正确性结果和 9 个计时结果，逐点保留 reference_latency_ms、latency_ms 及新旧比例。

首次生成与最终复验由真实 Codex SimpleOpt 驱动；额外请求由验证脚本调用与 MCP 相同的 `request_retest` 实现，以首次双边计时变化为证据，随后通过正常 SimpleOpt 恢复流程导出。该验证证明工具与恢复链路，不声称模型自主识别了异常并主动发起请求。恢复复用已完成的 Coder ledger，只运行结果蒸馏，没有再生成候选或重复强制复验。

KGS audit 中存在三个不同 evaluate request ID，全部 request SHA 相同：`f667dc19e8f920aabd66afe613e26bb1a1965e29230d40f549dfc41da78749af`。本轮直接验证的是请求独立且完整输入一致；新隔离子进程由现有 KGS executor 保证，audit 未暴露 worker PID，因此不将 PID 差异描述为本轮直接观测结果。最后输出 `best_geo_mean=0.9741135187353857`、`search_best_geo_mean=1.0484491048471312`，复验摘要为 replayed，epoch 读取同样得到最新确认分数。

## 验证命令与证据

在 feature worktree 中使用显式父目录 PYTHONPATH，并断言 KG 和 MCP 模块从该 worktree 导入。相关 host 命令如下，结果为 219 passed：

```bash
PYTHONPATH=/data/akg_kernel_bench_lite/worktrees/kg-final-best-retest python3 -m pytest -q tests/test_retest.py tests/test_kernel_gen.py tests/test_simple_opt.py tests/test_eval_round.py tests/test_profile_round.py tests/test_mcp_server.py tests/test_coder_agent.py tests/test_agent_roles.py tests/test_batch_simple_opt_definition.py tests/test_knowledge_distiller_reports.py tests/test_workflow_lifecycle.py tests/test_coder_completion.py tests/test_mcp_stdio.py --tb=short
```

原始归档位于本地忽略目录 `/data/akg_kernel_bench_lite/kernelgen/runs/final-best-retest-validation-pnciqge5`，不随 Git 分发。`run_simpleopt_isolated.py` 和 `run_extra_retest_resume.py` 保存输入、runtime 和导入隔离方式；`simpleopt.log`、`extra-retest-resume.log`、`server.log`、`audit/` 保存执行记录；`environment-job.json`、`model-metadata.json`、`mcp-import-proof.json` 保存目标环境与导入证据。`summarize_validation.py` 从 JSON/audit 断言并生成 `validation-summary.json`，本表数值由该摘要生成。

`simpleopt-isolated/.kernelgen/evals/round-0001/` 保留不可变搜索快照，`.kernelgen/retests/` 保留强制和额外复测的完整结果；`initial-output.json` 保存追加复测前的成功输出，`extra-retest-result.json` 与 `resume-result.json` 保存后续阶段。复测前后 ledger 字节一致、追加请求后旧公开分数清空、恢复后不增加设备请求均由驱动脚本断言。新功能未关闭旧快照缺证据的限制，也未用这一个算子验证 2 倍异常阈值在各厂商上的误报率。

# 候选准入验证（2026-09-07）

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

本轮验证通过：KG 的 prompt／receipt、KGS 的统一准入，以及实际 Codex SimpleOpt 流程均有独立证据。设计见 [候选准入设计](../design/candidate_admission.md)。本次是开发分支功能验证，不是新的 release 或厂商性能结论。

## 版本与环境

验证组合为 KG v6.2.1@6f4fb552 / KGS v6.3.1@dc020d4 / Protocol v6.2；两个 commit 均为基于对应 release 的开发提交，未移动发布 tag。KG 分支 `feat/candidate-admission`，KGS 分支 `feat/metadata-admission-policy`。Server 通过 Gitee 临时分支 `test-candidate-admission-20260907` 的 checkout 启动，Catalog 为内置 native `kernelgenbench`；本次未运行外部 Gems，不能据此声明外部 framework release 兼容性。

容器名 `kernelgen-nvidia-cu128`，实际镜像 `kernelgen-nvidia-cu132-nsight2026.1:base`，image ID `sha256:e5806e3c985b763ff6e3ebc95c4f0bca7b54f1c1e4ba4c145d73721349b0a53c`。物理 0 卡 NVIDIA A100-SXM4-40GB，逻辑 cuda:0；Python 3.12.3，解释器 `/data/jiabei/venvs/kernelgen-cu132-torch211-triton360/bin/python3`，Torch 2.11.0+cu130，Triton 3.6.0，CUDA runtime 13.0，driver 580.126.20。未安装或更换任何包。

本地 Agent + 本地容器 Server，地址 `http://127.0.0.1:18847`，仅监听 loopback；一个设备 slot、两个请求线程、一个 Coder，timing=triton。SimpleOpt runtime=codex，实际模型 `gpt-6-astra`，max_round=1，warmup_ms=10，benchmark_ms=10，num_trials=1，profile=false。传入了已验证的 Triton square 作为 reference Triton。独立 run/workspace 为 `/data/akg_kernel_bench_lite/kernelgen/runs/candidate-admission-validation/simpleopt-isolated`。

## 结果

| 验证 | 结果 |
| --- | --- |
| KG preflight、eval_round、Coder、Server adapter host 测试 | 62 passed |
| KGS admission、metadata、hack、FlagGems adapter host 测试 | 93 passed |
| 目标 Debug Job engine/native/admission/metadata 测试 | 50 passed |
| metadata recipe 的 CUDA 对象 identity、storage、shape | lift_fresh / unsqueeze_ 均通过 |
| native reference-as-solution 与 Triton square | preflight 通过，evaluate 36/36 通过 |
| Torch 全局改写、仅零 grid | preflight 与直接 evaluate 均拒绝，导入标记不存在 |
| profile 绕过尝试 | 保护状态修改在导入前被拒绝 |
| 并发、普通 RUNTIME_ERROR、worker 退出后的恢复 | 通过，单 slot max_active=1 |
| Codex SimpleOpt | PASSED，1 round，36/36 workload，best_geo_mean=0.999946585 |

以上计数分别属于不同测试层级，有重叠用例，不相加为独立测试总数。SimpleOpt 用于流程验证，加速比接近 1，不是优化提升证据。测试有意执行一次 os._exit(17)，产生 incidents=1/recovered=1；强探针恢复成功，最终 broken=0、active=waiting=0、available=1。

最终消费的 receipt schema=3.0，规则 hash=`f0952f8789c36ec9df10dcc829d6b66d29c809cc47e99d307dc1a3f3b0b9421b`。`mcp-import-proof.json` 确认子进程导入当前 feature 的 `tools/preflight.py`。首次 `simpleopt/` 运行虽然 PASSED，但 MCP 经 editable install 导入了主工作区的旧模块并生成 schema 2.0 receipt，因此保留证据且不计入新 KG 门禁验收；重跑使用新的 workspace 和仅用于验证的显式 MCP 启动脚本，未覆盖旧结果。

## 证据与复现

本地归档根目录为 `/data/akg_kernel_bench_lite/kernelgen/runs/candidate-admission-validation`，不随 Git clone 分发。`summary.json` 由 `summarize_validation.py` 从原始结果生成；`debug-download/validation.json`、`live-server.json`、`admission-live.json`、`simpleopt-isolated/optimize_definition_output.json`、其 `.ledger.json`／`.best_kernel.py`／receipt、`mcp-import-proof.json`、`model-proof.json` 与 audit/log 共同构成证据。`run_debug_validation.py`、`run_admission_validation.py`、`run_simpleopt_isolated.py` 保留了目标请求和生成参数。

在相同开发 commit 和目标环境启动 loopback Server 后，可运行 KGS 自带的验证命令；Catalog 路径指向该 Server checkout 的 `data/kernelgenbench`：

```bash
python3 tests/live_server_validation.py --server http://127.0.0.1:18847 --catalog data/kernelgenbench --definition kernelgenbench_square --expected-backend cuda --expected-device-count 1 --expected-workers 2 --concurrency 2 --warmup-ms 10 --benchmark-ms 10 --output /tmp/candidate-admission-live.json
```

KG host 验证应先确认 `kernelgen.__file__` 位于当前 worktree，再运行 `python3 -m pytest -q tests/test_preflight.py tests/test_eval_round.py tests/test_coder_agent.py tests/test_kernelgen_server_adapter.py`。访问 loopback 时按实验手册取消当前进程继承的 HTTP/HTTPS/ALL proxy 并设置 NO_PROXY/no_proxy；模型凭据只从已有环境读取，不写入复现文档或提交。

确定性门禁覆盖已实现的显式源码模式，接收者类型不明的静态命中保留为审核信号；不声称检测所有动态写入。独立 Gems 原生 pytest runner 仍须显式接入准入能力，不能从 KGS 两条路径通过推断它已自动获得保护。

收尾已确认 Server slot idle，临时 Server 已关闭，loopback 端口不再监听；临时 Gitee 分支与验证 checkout 已删除。两个 feature 分支及其工作树保留供评审，未合入共享目标分支；旧结果和本次 audit／Debug／Coder 证据保留。

后续按用户要求将 Gems 专属约束限定到 Gems adapter：KG prompt／native snapshot／SimpleOpt 相关 host 测试 65 项通过，KGS 路径选择／通用保护／metadata／Gems preflight、evaluate、profile 拒绝测试 52 项通过。本次范围调整为 host 验证；上面的 A100 与 Codex 结果对应所列的历史实现 commit，不将其表述为调整后代码的真机复验。

准入职责进一步收敛为仅 preflight：KG v6.2.1@7c91e40d / KGS v6.3.1@b173f32 / Protocol v6.2（开发提交）。evaluate/profile 不重复检测，KG 保留源码绑定 receipt 检查。KG 39 项 host 测试、KGS 51 项 host 测试通过；host 缺少 Torch 的三个模块通过 Gitee 临时 checkout 在现有 kernelgen-nvidia-cu128 容器中补跑 CPU 测试，24 项通过，CUDA_VISIBLE_DEVICES 为空，未启动 Server／Coder 或占用 GPU，未安装依赖。证据位于本地 runs/preflight-only-validation-20260907/summary.json 与 cpu-tests.log；历史 evaluate/profile 拦截测试已不再代表当前预期行为，当前正反用例确认它们不调用准入检测器。

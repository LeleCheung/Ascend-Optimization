# H20：Pytest Definition 与 OperatorOptimize 接线验证（进行中）

结论：Definition 导出、上传绑定和 host 回归已通过；真实 Gems Preflight 因源分支尚未同步已合入 master 的接口而失败，尚未启动模型优化，不能宣称 KernelGen E2E 通过。

## 实际组合

- KG：`codex/gems-pytest-definition`，基于 `dev@452a9c0b`；公共优化入口重构已由 !158 合入。
- KGS：`codex/gems-definition-bundles@fc3d2860aae74deade1207f6a8c930bd8289df81`，v6.3.4 开发快照；Protocol v6.2。源码经 Gitee 克隆到新目录，没有覆盖原 KGS checkout。
- Gems：官方 `kernelgen-dev@ab36a82ca0830f8520fdc25a81665d66ebd96cf6`，本地导出与远端执行使用同一 clean commit。
- 目标：H20 `115.191.21.142`，容器 `kernelgen-nvidia-ngc-2504`，官方镜像 `nvcr.io/nvidia/pytorch:25.04-py3`；沿用单卡 7 的临时共享验证方式，不停止驻留进程。
- Python：`/data/xuyao/kernelgen_hosts_20260923/venv/bin/python`；Torch `2.7.0a0+79aa17489c.nv25.04`、Triton `3.2.0`。项目虚拟环境新增/更新 packaging、SQLAlchemy、greenlet，受保护的 Torch/Triton/CUDA 相关 distribution 前后快照相同。镜像 pip constraint 原本固定 packaging 23.2，本次仅对上述轻量依赖安装命令清除该 constraint，没有修改系统配置。
- KGS：远端 loopback 18308、本地 SSH stdio proxy 19808；1 个健康 slot、请求线程 4、NCU capability 可用。该 capability 不代表本次已经实际 Profile 成功。

## 验证范围与结果

KG 的 Definition、统一 CLI/默认值、Catalog 输入、Native 恢复、evaluation snapshot 与请求构造相关组合先后通过 102 项（1 skip）及补充组合 99 项（1 skip），两组大量重叠，不相加计数。skip 是本机缺少 Torch。测试导入路径核对为当前 KG/KGS feature worktree。

继续补充“新 Definition → KernelGen → advisory code review”编排及恢复测试后，含原始 source extractor 的回归组合为 **144 passed、1 skipped**：实际导出、打包、安装和状态管理保留，GPU 与模型调用使用 test doubles。验证 Optimize 失败后仅恢复优化、不重复上传/输入审核，以及完成后 resume 不重跑；这仍不是模型或 GPU E2E 证据。复查官方远端 kernelgen-dev 仍为上述 ab36a82 commit。

KGS 在 CPU Agent 主机的纯 host 组合通过 130 项，排除 1 个依赖 Torch 的 API 测试。扩展组合曾有 4 个 `ModuleNotFoundError: torch` 和 10 个 skip；之后在 H20 现成运行时原样执行相关 7 个 test 文件，连同新增 Gems Bundle HTTP contract 测试共 **151 passed**，无失败或跳过。这些是接口/单元测试，不替代真实 pytest E2E。

真实输入为 `tests/test_negative.py`。GemsAdapterDefinitionWorkflow 从源码导出参数 `x`、输出 `out` 的纯 v6.0 Definition，记录 correctness、benchmark、实现文件摘要；没有生成 Native oracle 或 workload。通过 Bundle upload 和 operator-contract 确认 KGS 使用上传的 Definition，而不是同名内置 Catalog。`inspect` 成功列出原 benchmark 的 15 个 timing cases。

随后用独立的纯 Triton negative smoke 候选执行 Preflight。pytest 返回 1 个 benchmark 测试函数通过，但 KGS 返回 `RUNTIME_ERROR / timing_candidate_smoke`：`FlagGems preflight report must use schema flaggems.preflight/v1`。不能仅凭该 pytest 测试函数成功推断全部 candidate case 覆盖已得到验证，因此未继续 Eval、Profile 或模型 KernelGen。

## 源分支缺口与下一步

实际读取官方 master `e3ce56673c66dd3bf0b085b9b965b7e61d70dc65` 与 kernelgen-dev 源文件，发现 master 已有而 kernelgen-dev 尚缺：

| 接口 | master 文件 | KGS 用途 |
|---|---|---|
| `flaggems.preflight/v1` 报告 | `benchmark/conftest.py` | 验证每个 timing case 实际运行了候选 |
| `candidate_calls` | `tests/conftest.py` | 防止正确性测试漏用注入的候选 |
| `pytest_flaggems_profile_scope` | `benchmark/conftest.py` | 在 pytest 进程内执行 KGS profiling 前后处理 |

需要先通过官方 kernelgen-dev 的同步 PR 或由维护者同步 master，再选定新 commit、重新导出 Definition、更新配套清单及远端 clean checkout，重跑完整 Smoke 和默认 1 Coder / 1 epoch / max-round 10 的 OperatorOptimize。已向用户询问同步 PR 的授权，不自行修改官方共享分支，不放宽 KGS 门禁，不用 master 的测试结果冒充 kernelgen-dev 验证。

后续 MultipleDeviceTestWorkflow 按已确认顺序放在 H20 真正完成之后，本轮没有实现或启动其他芯片测试。

## 证据位置

本地实验目录：`/data/akg_kernel_bench_lite/kernelgen/runs/gems-definition-h20-20260924/`；包括 prepare/start/exercise/cleanup 脚本、受保护依赖核对摘要、Definition、Bundle 元数据、operator-contract、inspect、Preflight JSON、host 测试日志、源接口对比和清理记录。没有在该目录保存代理或模型凭据。

远端目录：`/data/xuyao/kernelgen_hosts_20260923/h20-gems-definition-20260924/`；包含 clean KGS/Gems checkout、host-tests.xml、依赖前后快照、Server 日志和 Bundle。停止前确认 scheduler 空闲；仅按 PID 启动身份关闭本次临时 KGS/proxy，保留全部失败证据和既有驻留进程。

# A100 Catalog prepare/review E2E — 2026-09-14

## 范围和代码

验证 `CatalogOptimizeWorkflow` 新的 `prepare_catalog → review_catalog → optimize → code_review`，包括本地 Native Catalog 打包上传、真实语义审核、目标 reference Preflight/Eval、原优化器及代码审核。输入是已有 Catalog，不重新抽取或生成 pytest；入口是 Python launcher，不是尚未统一的 `kg run` 后台提交。Profile 关闭，真实 E2E 不替代 pytest 生成、Gems 原生测试、断点续跑或多芯片验收。

KG v6.3.1 开发快照 `f37e6e03`（分支 `refactor/catalog-prepare-review`）配套锁定 KGS v6.3.4 `00dfc8692a6927d052b0c6da238cdde7908363c4`，Protocol v6.2。未移动 tag、修改锁定清单或更新远端服务。此 KGS 支持 Bundle evaluation binding，但没有 `operator_contract`，因此真机输入使用 `catalog_path`，不以 release 推断安装式契约导出能力。

容器内单元/配套 HTTP 回归另使用 KGS `49fcf4ecb86a2c3ec100b5194dba276100232b5e`（已包含 operator_contract），138 passed，含此前宿主机缺 Torch 跳过的测试。HTTP 路由和 Schema 为真实实现，设备执行由测试模拟；不与下面使用锁定 KGS 的真机结果混为同一次运行。两个测试环境均先确认了 KG/KGS 的导入路径，KG 指向当前 worktree。

## 环境和输入

使用现有 `kernelgen-nvidia-cu128` 容器，host 网络，挂载 `/data`。名称不是实际运行时版本：解释器 `/data/jiabei/venvs/kernelgen-cu132-torch211-triton360/bin/python3`，Torch 2.11.0、Triton 3.6.0、Pydantic 2.13.4、pytest 9.1.1；KGS `/status` 报告 CUDA runtime 13.0、driver 580.126.20。启动前后上述包版本未变，未安装或升级受保护运行时。

现场确认 8 张卡无计算进程后仅使用物理卡 7：NVIDIA A100-SXM4-40GB，UUID `GPU-9a15bd7a-66f2-027a-13bd-8ffa6f96f710`。临时命名实例 `catalog-prepare-nv` 通过容器内 `kg server start catalog-prepare-nv --target local --devices 7 --backend cuda --timing triton --port 24217 --max-workers 2` 启动，同时显式指定 `--kgs-root /data/akg_kernel_bench_lite/kernelgen_server` 和上述解释器的 `--python` 路径。KGS 只绑定 loopback，进程可见的 `cuda:0` 对应物理卡 7；设备 slot 为 1，不是两个请求线程各占一张卡。

本地 KG/模型通过同一宿主机 loopback 访问容器 KGS；不是跨机器远端部署，未新增公网端口。独立 CLI 状态目录、workspace 和日志都位于 `runs/catalog-prepare-nv-e2e-44YfQF/`，不触碰旧实验。模型由安全 source 本地配置加载，Claude runtime、`deepseek-v4-flash[1m]`。

Catalog 为锁定 KGS checkout 的 `data/kernelgenbench`，选择 `kernelgenbench_square`，完整保留 27 个 correctness workload、9 个 timing workload，不删减或修改 reference。两种优化模式使用独立 workspace：SimpleOpt 两轮；KernelGen 一个 epoch、两个 Coder，每个两轮。共用容量 2 的独立 Coder lease pool，KernelGen 需等待两个名额，设备请求始终经单 slot 串行调度。

## 证据与查询

实验根目录包含输入 `simple_opt.json`、`kernelgen.json`、可重现命令 `run.sh`、Agent 日志、Server config/process/log、冻结的 Catalog/Bundle/评测快照、review 报告、ledger、最终输出及候选代码。`observe.py` 保存结构化运行状态和 scheduler 快照；`summarize.py` 从原始 ledger 与最终输出生成 `summary.json` 和 `summary.md`，不手工填性能结果。

查询此次 Workflow（不将独立 Python workspace 伪装成已注册 CLI run）：

```bash
PYTHONPATH=/data/akg_kernel_bench_lite/kernelgen/runs/catalog-prepare-nv-e2e-44YfQF/imports:/data/akg_kernel_bench_lite/kernelgen_server \
python3 -m kernelgen.examples.catalog_optimize.run_example status \
  --workspace /data/akg_kernel_bench_lite/kernelgen/runs/catalog-prepare-nv-e2e-44YfQF/simple_opt
```

KernelGen 将末尾目录改为 `kernelgen`。上述导入链接是当前 worktree 的本地验收路径，不是正式安装要求。

## 最终结果

SimpleOpt 与 KernelGen 均为真实 `SUCCEEDED`，最终性能输出均为 `PASSED`。SimpleOpt 两个 measured round、KernelGen 两个 Coder 各两个 measured round，总计六轮全部 36/36 workload 通过。准确性能值见实验目录中自动生成的 `summary.json` 和 `summary.md`；搜索最佳值与最终复测值分别保留，不用其中一个覆盖另一个。

两次 Catalog review 和两次代码 review 均真实调用模型，无 dummy 或 skip_review。报告保留 P1 意见：重复测例、来源/容差证据不完整、非连续输入分支覆盖、计时量化和直接使用 Triton 内部启动接口等。部分属于模型待核对的建议，不据此自动修改共享 Catalog 或改变评测语义。此次证明当前声明 workload 下的完整执行链路，不证明候选覆盖任意输入，也不将几个百分点的加速视为已具备稳定统计优势。

SimpleOpt 中一个补充 Debug Job 因 Triton 不支持 `cache_modifier=".cs"` 失败，属于模型实验脚本编译失败；正式评测仍通过，KGS 无 device incident，没有通过升级依赖或修改 reference 绕过。模型在轮内做了多次诊断，因此两轮预算不等于两次模型调用或固定时长。

额外对已完成 SimpleOpt 执行 `--resume`，返回 `SUCCEEDED`，复用四份 `attempts/01/result.json`，没有新增 attempt、重复上传或模型调用；不将其声称为真实中途取消/恢复测试。中断、审核阻断和 readiness 失败恢复已由本分支单元测试覆盖。

观测到 prepare 完成后先进入 review，语义审核通过再执行目标 reference 验证，随后 optimize，最后 code_review；状态不再含 distribute_catalog。KernelGen 等待 SimpleOpt 释放名额后取得两个 Coder lease；设备调度实际出现 `active=1, waiting=1`，始终只用一个 slot。终态两个 Workflow `process_alive=false`，各自活动 Server operation 为零，独立 pool 的 leases 为空，KGS `healthy=available=1`、`active=waiting=checking=broken=incidents=0`。完成取证后通过 `kg server stop catalog-prepare-nv` 停止临时实例，保留全部日志和输入产物，不影响其他实例。

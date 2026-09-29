# Unified Catalog CLI — A100 验证（2026-09-14）

## 范围与入口

本次验证统一优化入口 `kg run --mode simple_opt|kernelgen`：新请求直接调用 CatalogOptimize，执行 `prepare_catalog → review_catalog → optimize → code_review`，不再经过 launcher argv 往返。Python 两个优化 example 使用同一参数归一化和同一入口；CLI YAML Batch 仍为每个算子建立独立请求、后台进程和 workspace。旧 v1 CLI 请求保留原 launcher/ledger 布局用于续跑，不迁移旧实验目录。

独立 CatalogExtractWorkflow 仍是生产公共 Catalog 的 Python Workflow；本次真实抽取后将同一冻结产物交给 `kg run --batch-file` 消费，不要求先完成其他算子的抽取。不新增常驻编排服务。pytest 生成按本次范围排除；原生 pytest 的准备检查、抽取/审核、Native 与 Gems 优化、代码审核、CLI 状态、取消和恢复分别记录证据。原生 pytest probe 的 `READY` 不等于语义审核或性能验收通过。

## 代码与环境

KG v6.3.1 开发分支 `feat/unified-catalog-run`，统一入口代码提交 `3cb7a462`，基于 dev `1fdd476e72c514f0f4f3ddcd6b0fd99618f5df37`；配套 KGS v6.3.4 开发快照 `49fcf4ecb86a2c3ec100b5194dba276100232b5e`（!55），Protocol v6.2。本次同步 `deployment/kgs.lock.yaml` 的 exact commit，不移动两个仓库的任何 tag。运行时已核对 `operator_contract.enabled`、`operator_bundle_upload.evaluation_binding` 和 operation cancellation；不根据 release 推断功能。实验期间公共修复分别合入后复测，最终实现和验收经 !138 提交，不将期间所有任务称为同一个已发布 clean tag 的运行。

复用 `kernelgen-nvidia-cu128` 容器，只为 KGS 使用空闲物理卡 7（逻辑 `cuda:0`，NVIDIA A100-SXM4-40GB）。临时实例名 `unified-catalog-nv`，loopback `127.0.0.1:24218`，backend=cuda、timing=triton、请求线程 2、设备 slot 1。独立 Agent Coder pool 上限 2，SimpleOpt 权重 1、KernelGen 权重 2；准备、审核和 KGS readiness 不占 Coder 名额。

容器名不代表实际版本：解释器 `/data/jiabei/venvs/kernelgen-cu132-torch211-triton360/bin/python3`，Torch 2.11.0、Triton 3.6.0、Pydantic 2.13.4；KGS 报告 CUDA runtime 13.0、driver 580.126.20。本次没有安装依赖或替换 Torch/Triton/厂商运行时。Agent/CLI 留在本机，GPU 执行通过容器 KGS。独立抽取使用已配置的 SSH + Docker collection executor（SSH 回到同一 A100 主机），不把这种测试宣称为 CPU-only 抽取验证，也不把 collection executor 的设备选择等同于 KGS 的卡 7 隔离。

优化与代码审核使用 Claude Runtime / `deepseek-v4-flash[1m]`；抽取的 CaseListAgent/Extract Agent 使用 Codex / `gpt-6-astra`，Catalog 审核使用 Claude / `deepseek-v4-flash[1m]`。模型配置通过 source 本地配置读取，凭据未保存到请求或本报告。

## 真实流程证据

实验根目录：`/data/akg_kernel_bench_lite/kernelgen/runs/unified-catalog-cli-e2e-MYavP4`。保留 `kg.sh`、输入 YAML/JSON、后台进程记录、Server 日志、冻结契约、逐次审核记录、ledger、最终复测记录及 best code。`observe.py` 调用真实 `kg status --detail` / `kg history --json` 并采集 scheduler、lease、active operations；`summarize.py` 从这些结构化记录和 ledger 生成 `summary.json` / `summary.md`，不从日志关键词或手填性能值判断成功。

| 场景 | 实验子目录 | 实际配置 |
| --- | --- | --- |
| 安装式 Native / SimpleOpt | `installed-simple` | kernelgenbench_square，2 轮；完成后再次 resume |
| 安装式 Native / KernelGen | `installed-kernelgen` | kernelgenbench_square，1 epoch × 2 Coder × 2 轮；真实取消进行中的 Eval 后原地 resume |
| Gems adapter / SimpleOpt | `gems-simple` | asin，2 轮，原 Gems pytest/benchmark 经 adapter 执行 |
| 本地 Native Bundle / KernelGen | `native-batch/definitions/kernelgenbench_square` | 1 epoch × 2 Coder × 2 轮，使用前一 SimpleOpt winner 作为未预验证 seed，不导入旧成绩 |
| 刚抽取的 Native / SimpleOpt | `native-batch/definitions/relu` | 独立抽取产物，2 轮；与上项同一 CLI YAML Batch |
| Bundle / Profile | `bundle-profile` | square，1 轮，真实 Profile；与普通无 Profile 任务分开验收 |

`extract-03` 真实执行原始 `benchmark/test_relu.py::test_relu --list-cases --level core`，没有用户提供的 case_list_path，也没有手工改写 case 集合。CaseListAgent 产出执行计划，Python collection executor 经 SSH + Docker 执行，随后抽取并审核；首轮审核通过，产出 `extract-03/catalog`。真实实验没有触发第二轮修订，不将单元测试覆盖的迭代/阻断路径称为真机已触发。来源是独立干净 FlagGems checkout `d64794e63b502cb836bc015a92a62c42de4be05a`；实际 Gems adapter 的来源则为 `/data/akg_kernel_bench_lite/FlagGems-master` @`2784b5c416f16260b7dad4e05e4c4868acf034c1`，两者不混用。

按照仓库 `pytest-review` Skill 执行原生 asin reference-as-solution probe：accuracy + benchmark 共 19 个实际调用通过，结果为 `pytest-02-readiness.json` 的 `READY`。保留报告中的 unverified 项，未把该 probe 当成 oracle 语义认证。首轮虽然测试通过，但 Gems 默认写入自身目录的 `accuracy_result.json` 触发来源变化检查；第二轮通过正式 `--record json --output <实验目录文件>` 将报告写到源码外，通过原检查，没有删除检查或修改 pytest 内容。

## 本次修复及回归

独立公共修复先分别 PR：!135（`refactor/catalog-prepare-review`）将上传合入 prepare、统一 review/readiness 顺序；!136（`fix/collection-source-filter`）在读取和传输前排除 Git 元数据、缓存及环境文件，接收端保留同一安全过滤，不放宽真实源码变更/越界 symlink 检查；!137（`fix/profile-eval-binding`）把实际 Eval binding 写入既有 immutable round identity/fingerprint，Profile 使用该绑定，不再把 Bundle 的内部标签当作安装式 Catalog 名称。

统一 CLI 同时修复了嵌套 ledger 查找及 KernelGen 父 scope 最佳值投影；普通步骤的公开 status 使用 basic payload，不显示空 round/epoch 字段。优化轮次仍只来源于 ledger，审核结果不伪造 measured round。新请求按一次类型化输入调用，旧请求只走旧 launcher；优化 lease 仅在公共 Optimize 边界获取和释放。

最终 host 主回归 463 passed、1 skipped；同组容器回归 464 passed。额外 host 的 pytest-review、NativeToFlagGems、Knowledge 与 uploaded snapshot 回归 181 passed。所有测试先核对 KG/KGS `__file__` 位于本次 worktree/配套 checkout；host 跳过的含 Torch 测试在容器执行通过。未执行全仓库无关测试，也不声称修复历史全量失败。公开参数、请求 v2、两种 launcher、批量路径、取消/恢复、MCP 权限、lease、Profile/retest/preflight 和 Schema 均有相关回归。

Knowledge 使用冻结 Catalog 的 benchmark identity，不再去本地猜同名 Catalog；默认 run ID 从 campaign 路径稳定派生，避免所有嵌套优化目录都叫 `work` 而冲突，KernelGen 的显式 knowledge_run_id 仍保留优先级。相关转发和现有 Knowledge 行为由 host 测试覆盖，本次不把它们称为启用 Knowledge 的模型 E2E。

Native baseline 额外测量、NativeToFlagGems 真机回迁及多芯片复制不是本次统一入口的隐含步骤：前者仍按后续独立策略接入，回迁本次只有 host 测试；本报告不把它们计为真实 E2E 已通过。pytest 生成未运行，没有 dummy 生成结果充当验收。

## 最终结果与回收

最终结构化门禁通过：6 个真实 CLI 子任务全部 `SUCCEEDED`，共 15 个 measured round 全部 `PASSED`；KernelGen 两个任务各有两个独立 Coder ledger，每个 2 轮，未把并行轮次拼接成单个 Coder 的 4 轮。性能数值见自动生成的 `summary.json` / `summary.md`。Profile 实验的最佳加速比略小于 1，属于正确且可计时但没有加速，不能因 Workflow 成功就宣称性能提升；几个百分点的差异也不视为稳定统计优势。

Bundle Profile 的实际请求 binding 与该轮 Eval identity 一致，candidate SHA 一致；下载的 NCU artifacts 非空、无 download_error，并包含真实 `_square_kernel` invocation。ledger Profile 状态为 `completed`，分析文件为 `bundle-profile/stages/optimize/work/.kernelgen/profile-analysis/round-0001.json`；最终代码审核真实完成，无 dummy 或 skip_review。保留 P1 覆盖范围、计时精度和性能表述建议，不因此修改 reference/workload 或伪造 measured 数据。

真实 `kg cancel` 在 KernelGen 已记录 2 个 measured round 时发生：一项 active Eval 取消成功，进程在安全点退出，观测到 `CANCELLED/process_alive=false`、活动操作为零、pool 使用量为零。KGS 对该取消执行强探针并恢复，因此累计 `incidents=1/recovered=1`，不是未恢复设备故障。随后 `kg resume` 沿用原 workspace，generation 从 1 进入 2，保留此前轮次并最终完成原定总计 4 轮；已完成 SimpleOpt 的 resume 没有增加轮次或新的步骤 attempt。

所有 6 个后台进程最终退出，各 workspace 的 active operations 为空，Coder lease 为零；KGS `healthy=available=1`、`active=waiting=checking=broken=0`、`max_active=1`。前后环境 probe 输出一致。之后只停止临时 `unified-catalog-nv` 实例，确认端口 24218 关闭，并验证 KGS 停止后 `kg status` / `--detail` 仍能查询 Batch 和单任务。保留全部实验数据，未停止其他 KGS、重启容器或清理用户主工作区。

## 查询命令

正式安装本次版本后：

```bash
kg status /data/akg_kernel_bench_lite/kernelgen/runs/unified-catalog-cli-e2e-MYavP4/native-batch
kg status /data/akg_kernel_bench_lite/kernelgen/runs/unified-catalog-cli-e2e-MYavP4/installed-kernelgen --detail
kg history /data/akg_kernel_bench_lite/kernelgen/runs/unified-catalog-cli-e2e-MYavP4/installed-kernelgen --json
```

尚未更新本机安装时，可用实验根目录的 `bash .../kg.sh status <workspace> --detail` 指向测试代码；不要用旧 editable install 的 CLI 判断新请求 Schema 是否支持。状态和历史查询不需要 KGS 在线，也不会启动新的优化。

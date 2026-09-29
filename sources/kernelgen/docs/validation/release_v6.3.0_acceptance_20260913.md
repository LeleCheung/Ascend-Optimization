# KG v6.3.0 发布后使用验收：天数单卡

当前结论：天数单卡上的真实 CLI SimpleOpt、KernelGen、prepared Catalog 的 ExtractAndOptimize 和两算子 Batch 均已成功，包含最终独立复测；相关 host 回归为 254 passed。部署发现安装前置条件缺漏、remote doctor 误查本地 checkout，以及本地代理端口冲突时 start 可能误报成功，因此不能宣称原版部署验收全部通过。首轮应用 E2E 经旧 SSH proxy 访问本次新 KGS，独立代理部署另行复验；真实源码抽取和 Gems 安装未包含在本轮真机通过范围内。已有 22 项主线单测失败按本次要求不处理，见 [发布说明](../releases/v6.3.0.md)。

## 被测版本与隔离

实测为 KG v6.3.0@fef3e2efc88a6bc891f421b55ec6b1f4daef7cd2 / KGS v6.3.2@ca456d64deb30a5b90eafdebb1845c5feed180d7 / Protocol v6.2。KG 从 Gitee 的 v6.3.0 tag 全新 clone；KGS 由该 clone 的 `scripts/install_kgs.py` 按 lock 获取，不改源码、不移动 tag。文档修正单独位于 `docs/release-v6.3.0-validation`，未作为受测发布代码。

证据根为本机忽略目录 `/data/akg_kernel_bench_lite/kernelgen/.kernelgen/experiments/release-v6.3.0-JouxH3`：`kernelgen/`、`kernelgen_server/` 为独立 checkout，`venv/` 保留首次安装现场，`venv-documented/` 验证补正文档后的安装，`host-tests.xml` 保存测试结果，`environment.json` 保存完整包版本与导入路径。临时目录不是新用户需要复制的路径；不包含在 Git 分发中。

## 已完成的检查

| 检查 | 结果与边界 |
| --- | --- |
| 全新 CPU venv 安装 KG | 成功，Python 3.12；未安装 Torch、Triton 或厂商扩展 |
| 按原文运行 KGS 安装脚本 | 失败：`--no-build-isolation` 下缺少 setuptools；原错误包装只显示子命令退出码 |
| 补齐构建依赖后重试 | 成功，KGS checkout exact commit、clean 状态与两个包的实际导入路径匹配 |
| `--copies` 虚拟环境重验 | 安装和 Python 路径选择通过；避免默认 symlink venv 被本地 Server 解析为 `/usr/bin/python3.12` |
| 重复 KGS 安装、`pip check` | 成功，重复安装前后记录的依赖版本不变，无依赖冲突 |
| 模型连接 | 使用既有 Claude CLI 和已授权的本地模型配置，真实调用 `deepseek-v4-flash[1m]` 返回 `KG_RELEASE_OK`；不代表 Coder/MCP/Eval 全链路通过 |
| 生命周期 dummy 与原命令 `--resume` | 8/8 模拟调用成功，普通调用进度为空字典；未调用模型或设备。该模式本来不接入 `kg status`，不能用模拟成功代表优化成功 |
| 发布相关 host 回归 | 254 passed、0 skipped；没有运行或屏蔽那 22 项旧失败所属的测试文件 |

Host 命令在独立 KG clone 中执行，解释器为 `../venv-documented/bin/python`，未使用主工作区的 editable 安装：

```bash
PYTHONDONTWRITEBYTECODE=1 ../venv-documented/bin/python -m pytest -q \
  tests/test_version.py tests/test_install_kgs.py tests/test_cli*.py \
  tests/test_run_control.py tests/test_workflow_composition.py \
  tests/test_catalog_optimize.py tests/test_extract_and_optimize.py tests/test_retest.py \
  --junitxml=../host-tests.xml --tb=short
```

这是 host 测试，模型/KGS 的模拟属于单元测试边界。新 CPU venv 的主要依赖为 mcp 1.30.0、pydantic 2.13.5、FastAPI 0.141.1、uvicorn 0.52.4、safetensors 0.8.0；额外准备 setuptools 84.0.0、wheel 0.48.0、packaging 26.3 和 pytest 9.1.1。完整新增包及版本见 `environment.json`，没有修改现有宿主机或设备容器的包。

## 文档修正

补充 ONBOARDING 的虚拟环境激活及发布版的构建依赖、`--copies` 临时解决方法；补上 ExtractAndOptimize CLI 必填的 `--definition`。组合 Workflow 当前仍要求源码字段和 `case_list_path`，不能套用独立 CatalogExtract 的自动采集输入契约。Runtime 指南中的历史 KGS 反向推荐规则改为当前 KG lock 唯一选择。后续代码修复及其简化安装路径单独记录于 [部署修复复验](deployment_fixes_20260913.md)，不改变本报告对原发布版本的判定。

## 本轮真机验收范围

按用户追加授权在天数创建独立临时 KGS，使用简单算子、最多两轮，核对 Preflight、完整 correctness/timing、最终复测、ledger/输出、`kg status --detail`、history、logs、list、取消/恢复和 Batch 独立 workspace。以下流程真实调用模型与设备；ExtractAndOptimize 明确走 prepared Catalog 输入分支，未调用源码抽取 Agent，Catalog Review 和代码 Review 都是真实执行，`dummy=false`、`skip_review=false`。

| CLI 任务 | 输入与配置 | 结果 |
| --- | --- | --- |
| `simple_opt` | kernelgenbench_square，2 轮，warmup/benchmark 各 100 ms | SUCCEEDED，最终复测 VERIFIED |
| `kernelgen` | kernelgenbench_square，2 个 Coder、1 个 epoch、每个最多 2 轮，保留默认 Profile | SUCCEEDED；共享分析、两个 Coder、Profile、各自复测、Knowledge 合并与综合选择完成 |
| `extract_and_optimize` | 已准备 relu Catalog，2 轮，真实 Catalog Review 和代码 Review | SUCCEEDED，5/5 调用完成；上传 Bundle、绑定 Preflight/Eval、优化及最终复测通过 |
| YAML Batch | square、sin，各 1 轮，每算子独立进程和 workspace | 根目录 SUCCEEDED，2/2 子任务成功且最终复测 VERIFIED |

全部使用 deepseek-v4-flash[1m] 与 Claude Runtime。Coder 配额先设为 2，确认权重为 2 的 KernelGen 在已有一个 SimpleOpt lease 时排队；后调为 3 推进其余任务。KGS 请求线程始终为 2、slot 始终为 1，没有修改设备范围。原版 KernelGen 不接受 SimpleOpt 专用的 `warmup_ms/benchmark_ms/eval_timeout_seconds`；本轮 KernelGen 使用其自身默认计时参数和 `--timeout 1800`，不能把两种模式的参数混用。

取消验收覆盖 KernelGen 排队取消、Batch 整批排队取消、逐子任务恢复，以及 sin 在真实模型生成中取消后恢复。运行中取消发生在 04:59:25 UTC，完整模型结果输出后记录 MODEL_INVOCATION_COMPLETED，随后 04:59:47 UTC 记录 CANCEL_OBSERVED/RUN_CANCELLED，进程以 130 退出；未 kill Agent，未留下 measured round。05:00:42 UTC 从同一 workspace 恢复，最终成功。该样本在 BEFORE_PREFLIGHT 安全点停止，转发的活动 KGS operation 数为 0，不能据此声称已实测在途 Eval/Profile 的 cancellation。Batch 根目录 resume 明确不支持，逐个恢复子 workspace 与文档一致。

`observe.py` 保存每次 CLI text/detail/history、后期 list 和 Server scheduler；`check_evidence.py` 校验 Schema 2.0、basic scope 的空 progress、单卡 max_active，以及每个成功优化输出的最终复测状态和 best code 文件一致性。完整结果为证据根 `evidence-check.json`，性能数字由脚本读取输出生成，不手抄；本轮共享卡，数字不作为独占性能基准。最终全部 5 个独立 run 进程均已退出且 exit_code=0，worker-leases.json 的 leases 为空。

本轮检查时 A100 卡 7 的 GPU 利用率为 100%；已有昇腾 KGS 虽 idle，但仍管理全部 8 张卡，且运行的是旧发布元数据 v6.3.1。因此没有占卡、停止共享服务或在重叠设备上启动另一实例。需要本轮设备授权或明确协调后的资源切分，再继续部署/E2E；原共享服务保持不动。

## 天数目标准备与设备冲突

用户随后指定天数验真并允许重启 KGS。已按 inventory 登录 `10.31.28.29`，目标容器为 `codex_fib_iluvatar_20260728`，镜像为 `harbor.baai.ac.cn/flaggems/iluvatar-flaggems-test-bi-v150-flagtree0.6.0-iluvatar3.6:202607161508`。容器采用 host PID 和 host network；所以容器里能看到的进程、端口不都属于它，不能根据 `ps` 就直接停止服务。

通过进程 mount namespace 与 Docker init PID 对照，确认旧 KGS 9001/PID 526134 属于 `kb_yy`，旧 KGS 8000/PID 913094 属于 `ixprof_bi150_test`。二者均报告 Protocol v5.1、16 个 slot、12 healthy/4 broken、active=waiting=0，异常 slot 为 0、8、10、11，历史原因是启动探针超时。另一个 PID 1145072 属于 `gosim_server`，运行 9999 端口的独立评测 API，持有另外 12 张卡的设备上下文；`/status` 为 404，`/health` 查询超时，无法据此确认可暂停。没有停止这些其他容器的服务，也没有因利用率为 0 就启动重叠实例。

目标容器经持久化只读部署 key 访问 Gitee 成功，v6.3.2 tag 解析到锁定 commit ca456d64deb30a5b90eafdebb1845c5feed180d7。为受管 KGS 创建了独立 `--copies --system-site-packages` Python 环境：`/workspace/kernelgen_e2e_20260728/deployments/kg-release-v630-JouxH3/venv`，以及只引用现有部署 key/known_hosts 的权限 600 部署环境文件；未复制私钥、未修改全局代理。`packages-before.json` 保存在同一独立部署根。

用户随后明确授权“不用释放，临时用一张卡启动测试”，因此选择当时利用率 0% 的物理卡 15 临时共享，不停止上述服务。本次仅验收功能与有效计时，不将加速比视为独占环境性能基准。通过原版 CLI 创建实例 `tianshu-v630-jouxh3`，远端 loopback 18301、请求 workers=2、device slot=1，首轮 KGS PID 3648162。本地初始配置 19601；`start` 曾报告新代理 PID 4091261/RUNNING，但收尾核实该进程已退出，19601 实际一直由既有代理 PID 3615951 监听，不能把首次 start 的返回当作新代理存活证明。

`/status` 确认 v6.3.2 / Protocol v6.2、backend=iluvatar、device=Iluvatar BI-V150、timing=triton、Triton 3.6.0、FlagTree 0.6.0+iluvatar3.6、driver 4.4.0。physical 15 经可见设备映射为逻辑 `cuda:0`。启动后 healthy=available=1，active=waiting=checking=broken=0；operation cancellation 和 Bundle evaluation binding 均已开启。

配套原版 `tests/live_server_validation.py` 使用完整 kernelgenbench_square workload、concurrency=2、warmup=50 ms、benchmark=20 ms、timeout=300 s，结果 `passed=true`，完整响应保存在证据根 `server-smoke.json`。覆盖 reference Preflight/Eval、Triton candidate、并发排队、预期 RUNTIME_ERROR、隔离 worker `os._exit(17)` 及强探针恢复；最终 incidents=recovered=1、broken=0、slot idle。没有执行物理卡 reset。

真实后台 SimpleOpt 已提交至证据根 `runs/simple-opt`，模型为 deepseek-v4-flash[1m]，square、min_rounds=max_round=2、warmup=benchmark=100 ms。已观测到 `kg status --detail` Schema 2.0 的 MODEL_INVOCATION → EVALUATING 与实际 KGS active=1，不是 dummy。后续采样统一保存在 `observations/`，其中该目录是验收脚本额外创建的证据，不是正式 Workflow 的固定产物。

部署问题：原版 `cli/server.py::_command_doctor` 在判断 remote 前无条件检查 `config.kgs_root`，无本地部署 checkout 的合法远端实例因此报告 `not a Git checkout`。远端 KGS 和实际评测均正常；不可把这个误报算作 doctor 验收通过，也不应为消除误报额外 clone 不需要的本地部署目录。

收尾 stop/start 进一步暴露代理 readiness 问题：新代理日志为 `OSError: [Errno 98] Address already in use`，而旧代理 PID 3615951 自 2026-08-29 起监听 19601，连接的同样是天数目标容器的 loopback 18301。首次 readiness 存在借旧代理 HTTP 健康响应误报成功的窗口。首轮应用请求仍符合 SSH stdio 边界，且 `/status` 与本次新 KGS 的版本、单卡和 scheduler 匹配，故应用 E2E 结果有效；但首次“新代理部署成功”结论撤回。没有停止或接管旧代理，改用 `ss` 确认空闲的 21631，通过 stop/configure/start 修改仅本次实例，重新验证独立代理。

独立端口复验通过：`ss` 确认 21631 由受管 proxy PID 80427 监听，远端新 KGS PID 3715399，`kg server status --json` 为 RUNNING；重复 start 返回 ALREADY_RUNNING 且 PID 不变。`managed-proxy-server-smoke.json` 保存经该代理的 reference Preflight/Eval 与 Triton 测试通过结果，这次未重复 resilience 故障注入。额外真实 CLI `runs/managed-proxy-smoke` 使用 square、一轮，SUCCEEDED 且最终复测 VERIFIED；该任务从提交到退出都核对新代理仍存活。累计 24 次状态采样的 Schema/单卡断言通过，7 个优化输出均通过复测和 best code 文件一致性检查。

远端 `packages-diff.json` 对比部署前后记录的发行包版本没有变化，Torch 2.7.1+corex.4.4.0 和 FlagTree 0.6.0+iluvatar3.6 保持不变。所有正式结果、取消事件、模型 session、ledger、retest 和 Profile 产物保留在各 workspace；没有覆盖旧实验或移动 tag。

收尾 `kg server stop tianshu-v630-jouxh3` 后再次查询为 STOPPED，21631 已无监听，worker leases 为空。仅关闭本次临时 KGS 和其受管代理；旧代理 PID 3615951 仍存活，其他容器服务没有停止或修改。新增一轮补测也已正常退出，不遗留本次 Agent/Coder。

## 查看本次结果

以下是本机验收现场的命令，不是新用户需要复制的默认路径；status 查询在 KGS 停止后仍可读取持久化结果。

```bash
source /data/akg_kernel_bench_lite/kernelgen/.kernelgen/experiments/release-v6.3.0-JouxH3/venv-documented/bin/activate
cd /data/akg_kernel_bench_lite/kernelgen/.kernelgen/experiments/release-v6.3.0-JouxH3/kernelgen
kg status ../runs/simple-opt --detail
kg status ../runs/kernelgen --detail
kg status ../runs/extract-prepared --detail
kg status ../runs/batch --detail
kg history ../runs/simple-opt --json
kg list --json
```

## 尚未覆盖与后续修复

本轮没有验证真实 Gems 源码抽取、自动 list-case、PR 抽取、Gems 安装、跨 epoch 恢复、已完成组合调用 receipt 的 resume，以及正在执行的 KGS operation cancellation。KernelGen 仅 1 个 epoch，不宣称覆盖多 epoch 全部路径；已有 22 项 host 失败仍未处理。目标现有 FlagGems 是 site-packages 安装而非可验证 exact clean checkout，不能直接用它验收冻结 Gems oracle，也未修改其版本。

安装器应自行准备 setuptools/wheel 或清晰报告缺少构建后端；本地 Server 解释器不能通过 resolve 丢失 venv；remote doctor 的 checkout 校验应放回远端管理入口；远端 start 不能把占用端口上的另一个代理响应视为自身 readiness。上述问题已在独立修复分支处理，提交与回归范围见 [部署修复复验](deployment_fixes_20260913.md)；本报告原始 KG 提交 `fef3e2ef` 和 KGS v6.3.2 仍保留这些部署问题；用户随后授权重发最终 KG v6.3.0，发布说明记录了 tag 替换与新配套 KGS v6.3.3，不改变此处历史证据。

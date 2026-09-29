# Enflame CLI 单卡验收（2026-09-13）

结论：CLI 的 Enflame backend 已通过燧原真实远端启动、状态、幂等启动、doctor 和停止测试。仅使用物理卡 7，未执行 Agent 生成、Catalog Eval 或 Profile，不能把本记录当作完整优化 E2E 验收。

## 代码与环境

KG 使用基于 `cb511e36` 的 `feat/enflame-server-cli` 开发快照；配套 KGS 使用 `feat/enflame-managed-server` 的精确 commit `eda70bb905184f9b0dd59ac93df83968773c8554`，Protocol v6.2。包元数据仍为 KG v6.3.0 / KGS v6.3.3，但这两项新增管理能力不属于已经发布的 tag。实测时锁定该 KGS 开发 commit；正式发布需另行确定版本并更新发布说明。

合入更新：配套 [KGS !53](https://gitee.com/BaaiAC/kernelgen_server/pulls/53) 已 squash 合入 `main`，主线 commit 为 `8a3cfaa3dbd227bcaf0986023ccd6ce6e1dccb25`，与上述实测 commit 的完整文件树一致。KG 锁定清单已改为该主线 commit；原始实测记录仍保留开发 commit，不将合并后 host 复测描述为再次真机测试。

目标采用 inventory 中的 `suiyuan`，容器为 `codex_fib_suiyuan_20260820`，镜像为 `harbor.baai.ac.cn/flaggems/suiyuan-flaggems-test-s60-flagtree0.6.1-enflame3.6-pujiang:202608201756`，镜像 ID 为 `sha256:567f72620957e91d6aa65a09627f1b204a12ddfb57ce4da2b505255b5db7d6c8`。这些是本次快照，不替代 inventory 的后续连接配置。

测试实例名 `enflame-MxjqDG`，远端部署根目录 `/home/secure/xuyao/kernelgen_e2e_suiyuan_20260901/deployments/enflame-cli-MxjqDG`。KGS 通过 Gitee 临时分支 checkout 到上述 exact commit；验收后保留 feature 分支，删除本地及远端临时 `test-enflame-start-20260913` 分支，不覆盖既有 Server checkout。

## 部署与结果

真实 CLI 使用 `--target remote --backend enflame --devices 7 --timing triton --max-workers 2`，远端 loopback 端口 `21909`，本地 SSH stdio proxy 端口 `22909`，readiness timeout 为 180 秒。容器系统 Python 3.12.3 首次安装触发 PEP 668，CLI 报错且未启动 KGS；未绕过系统保护。改用部署目录下 `venv/bin/python`：以 `python3 -m venv --without-pip --system-site-packages <venv>` 创建项目环境，继承系统 pip 和厂商运行时，并通过 `kg server configure <name> --remote-python <venv>/bin/python` 选择它。该镜像缺少 ensurepip，因此没有使用默认 venv bootstrap。

| 检查 | 实测结果 |
| --- | --- |
| `kg server start` | RUNNING，启动强探针通过 |
| `kg server status --json` | `backend=enflame`、`devices=[gcu:0]`、`timing=triton`、Protocol v6.2 |
| scheduler | `device_slots=healthy=available=max_active=1`，`checking=broken=active=waiting=incidents=0` |
| 重复 `start` | ALREADY_RUNNING，远端 PID 和本地 proxy PID 不变 |
| `kg server doctor` | OK |
| `kg server stop` 及后续 status | STOPPED，本地 22909 和远端 21909 均关闭 |

`TOPS_VISIBLE_DEVICES=7` 将物理卡 7 映射为进程内 `gcu:0`；请求线程数 2 没有产生额外设备 token。目标 `/status` 报告 ZIXIAOC200（display name S60）、Triton 3.6.0、FlagTree 0.6.1+enflame3.6、TOPS runtime 1.9、driver 1.9.10。Profile supported=false，不作 Profile 可用性承诺。

安装前后 Torch `2.10.0+cpu`、`torch-gcu 2.10.0+3.7.20260408`、FlagTree `0.6.1+enflame3.6` 均未变化。Triton 由 FlagTree 提供，没有独立 triton distribution；其 distribution 元数据为空不表示运行时缺失。项目 venv 新增 KGS editable 包、FastAPI 0.141.1、Uvicorn 0.52.4、Pydantic 2.13.5、safetensors 0.8.0，以及 annotated-doc 0.0.5、pydantic_core 2.46.5、annotated-types 0.8.0、anyio 4.15.1、typing-inspection 0.4.4、h11 0.16.0、starlette 1.6.0；未修改系统运行时。初次轻量依赖安装耗时数分钟，不属于设备探针耗时。

## Host 测试与证据

测试前确认 `kernelgen.__file__` 和 `kernelgen_server.__file__` 分别来自两个 Enflame feature worktree，未使用主目录 editable install。

- KG：`tests/test_cli*.py tests/test_install_kgs.py tests/test_version.py`，181 passed。
- KGS：`tests/test_management.py tests/test_backend_registry.py tests/test_status_metadata.py`，16 passed、2 skipped；跳过项由于 host 未安装 Torch。management 自身 15 项全部通过。
- 新增用例覆盖 start/configure 的 backend 选择、本地/远端设备变量隔离、继承的 TOPS/CUDA/NPU 变量清理、`torch_gcu` 必需导入及安装前后保护。

本地原始证据在 `/data/akg_kernel_bench_lite/kernelgen/.kernelgen/experiments/enflame-cli-MxjqDG`：`cli-results.json` 保留第一次 PEP 668 失败，`cli-venv-results.json` 保存成功流程的命令和完整状态，`environment-after.json` 保存版本与端口回收检查，`test_cli.py`、`collect.py` 保存验证方法。远端日志保留在部署目录的 `server/kernelgen-server.log`；测试 KGS 已停止，原容器和其他服务保持运行。上述路径是本地实验证据，不随仓库分发。

# KernelGen Server

KernelGen Server 在异构加速器上执行可信客户端提交的 Python/Triton 算子，完成
preflight、正确性验证、计时、profiling 和调试任务。V6 同时提供简化 Workload
runner 和框架原生 adapter，并统一返回 `EvaluateResponse`。

当前发布为 KGS v6.5.0，Protocol 保持 v6.2；配套 KG v6.7.0。独立 client 包、安装边界与真机验证见 [v6.5.0 发布说明](docs/releases/v6.5.0.md)；实际新实验仍以 KG 锁定清单中的 KGS exact commit 为准。

## 安装

目标镜像必须已经提供与芯片匹配的 PyTorch、Triton 和厂商运行时。本项目不会安装
或升级这些核心依赖。

```bash
git clone git@gitee.com:BaaiAC/kernelgen_server.git
cd kernelgen_server
python3 -m pip install -e ./client
python3 -m pip install -e ".[server]"
```

`client/` 是独立发行包，只包含协议模型、Catalog/Bundle 准备、HTTP 调用及纯来源发现逻辑，不包含 KGS 设备执行器；`server` extra 只增加 FastAPI 和 Uvicorn。先从同一 checkout 安装 client，确保两个包的实现来自相同提交。验收环境依赖已经齐全时，可以增加 `--no-deps --no-build-isolation`。

正式运行前按 [版本兼容与运行选择规则](docs/version_compatibility.md)读取当前 KG 的锁定清单选择 KGS exact commit，再读取根目录 `compatibility.yaml` 核对自身描述：`v6.0` Catalog 默认走 FlagGems adapter，`v6.2` Catalog 默认走 native evaluator。不按历史反向矩阵或 tag 大小推断配套版本。

## 启动

先用厂商可见设备变量限制 Server 可使用的卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m kernelgen_server.server \
  --host 127.0.0.1 \
  --port 8000 \
  --backend cuda \
  --timing triton \
  --max-workers 4
```

支持的 backend 包括 `cuda`、`npu`、`musa`、`mlu`、`hygon`、`metax`、
`iluvatar`、`kunlunxin` 和 `thead`。worker 数可以大于设备数，但设备 token 始终
与可见设备一一对应；每张卡同一时刻最多执行一个 Eval、Profile 或 Debug Job，多余
请求在 Server 内排队。通过 `/status.scheduler` 检查真实占用和排队状态。
启动后应确认 `healthy=device_slots`、`checking=broken=0`。请求超时或隔离 worker
异常退出后，Server 会先隔离 slot 并执行强健康探针；探针通过则恢复且不跨卡重试，
探针失败则立即标为 `broken`、返回 `SUSPECTED_DEVICE_ERROR` 并停止复用。

非昇腾设备的 `--timing triton` 使用 `triton.testing.do_bench`。昇腾正式性能
使用 `--timing auto` 或 `--timing profiler` 选择严格
`torch_npu.profiler`。计时失败不会回退 wall time。

确认服务：

```bash
curl -fsS http://127.0.0.1:8000/status
```

需要保留厂商问题证据时，启动命令增加
`--request-audit-root /data/kernelgen_request_audit/<run_name>`。Server 会为
每次启动创建独立 session，完整保存所有通过协议校验的 `/preflight`、`/evaluate`
请求，以及设备分配、超时、worker 异常、恢复探针和响应事件。审计默认关闭；启用后
若原始请求无法在设备调度前落盘，该请求不会执行。实际 session 路径可从
`/status.request_audit` 获取。

## 最小评测

```python
from kernelgen_server import (
    BoundEvaluateRequest,
    Catalog,
    EvaluationSettings,
    EvaluatorBinding,
    Implementation,
    InspectRequest,
    SourceFile,
    builtin_catalog_path,
)
from kernelgen_server.client import evaluate, inspect, preflight

catalog = Catalog(builtin_catalog_path("simple-v6-test"))
operator = catalog.load("identity")
implementation = Implementation(
    name="reference-as-solution",
    definition=operator.definition.name,
    language="python",
    entrypoint="main.py::run",
    sources=[SourceFile(path="main.py", content=operator.definition.reference)],
)
binding = EvaluatorBinding(
    catalog_name="simple-v6-test",
    definition="identity",
)
request = BoundEvaluateRequest(
    binding=binding,
    implementation=implementation,
    settings=EvaluationSettings(warmup_ms=1000, benchmark_ms=100),
)

server = "http://127.0.0.1:8000"
print(inspect(InspectRequest(binding=binding), server))
print(preflight(request, server))
print(evaluate(request, server))
```

Native 和 FlagGems 共用 `POST /inspect`、`POST /preflight`、`POST /evaluate` 和 `POST /profile`。请求只包含 Catalog binding、candidate 与设置；Server 根据 Catalog manifest 内部选择 adapter。`data/flaggems-adapter-definitions` 保存公开 Definition，具体 accuracy/timing case 由 KGS 在固定 FlagGems checkout 上按 pytest marker 动态发现并归一化；FlagGems v2 的全局 `case_id` 原样用于精确 profiling。

支持取消的客户端从 `/status.capabilities.operation_cancel` 判断能力，并在 Preflight/Eval/Profile 请求中携带 `X-KernelGen-Operation-Id`。`DELETE /operations/{operation_id}` 请求取消，`GET /operations/{operation_id}` 查询 `QUEUED/RUNNING/CANCEL_REQUESTED/CANCELLED/SUCCEEDED/FAILED` 状态；该扩展不改变 Protocol v6.2 请求体 Schema。

KGS `v6.3.1` 的 FlagGems Adapter 不再绑定 Catalog 中的固定 commit，直接使用 KG 实例或操作方通过 `KGS_FLAGGEMS_ROOT` 选定的 checkout；默认推荐值仍来自 `compatibility.yaml`。源码工作树必须干净，实际 commit 参与 benchmark fingerprint；更新后重新 inspect 和验证 baseline，不在 campaign 中途自动跟随分支。FlagGems-backed Native Catalog 的冻结 oracle 仍须匹配其 exact revision。部署不会修改 Torch、Triton 或厂商运行时。算子数、case 数和真机结果以当前 Catalog、`/status` 与验证报告为准，不在 README 中维护易过期快照。

Server 会执行客户端提交的源码和 Debug Job，不是安全沙箱。默认只监听 loopback，
只允许可信客户端访问。

## 上传用户 v6.2 native 算子

KGS 提供内容寻址的 Operator Bundle 上传接口。Python 客户端可调用 `upload_operator_bundle("./my_operator", server)` 自动打包、查询缓存并按需上传；Server 通过 `--operator-bundle-root` 配置临时存储目录。当前阶段尚未将 `bundle_id` 接入评测 binding，完整格式、HTTP 接口、安全校验和能力边界见[用户算子上传接口](docs/operator_bundle_upload.md)。

## 独立验证 v6.2 Catalog

`tools/validate_v62_native_catalog.py` 是可单文件交付给厂商的验证器，静态检查只依赖
Python 标准库，不需要安装或导入 KGS、Torch、Triton。加入 `--runtime` 后，它会在
独立子进程中真实导入 oracle、加载算子私有 `assets/`，并 smoke 全部 workload：

```bash
python3 tools/validate_v62_native_catalog.py data/<catalog> --mode standard
python3 tools/validate_v62_native_catalog.py data/<catalog> \
  --runtime --target-device cuda:0 --output validation-report.json
```

运行检查需要厂商目标环境已经具备 oracle 自身依赖。完整规则和报告口径见
[v6.2 Native Catalog 接入规范](docs/v6.2_native_catalog_authoring.md)。

## 文档

- [版本兼容与运行选择规则](docs/version_compatibility.md)：KG/KGS 发布组合、`v6.0` adapter 与 `v6.2` native 模式，以及新实验和续跑的版本选择规则。
- [v6.2 Native Catalog 接入规范](docs/v6.2_native_catalog_authoring.md)：算子提供方
  必须/可选资产、文件契约、模板、验收流程和提交检查表。
- [v6.2 Profiling 合并关键决策](docs/v6.2_profiling_integration_decisions.md)：记录
  profiling 增强分支迁入 v6.2 时的功能来源、已确认方案和原因。
- [多芯片部署与自测教程](docs/multiple_device_deploy_tutorial.md)：完整部署、设备
  映射、容错和全 Catalog 自测。
- [多芯片 Eval 验证报告](docs/multiple_device_eval_validation_report.md)：历史
  环境、兼容性问题和结果。
- [设计文档](docs/DESIGN.md)：协议、执行、计时、Debug Job 和源码结构。
- [设备槽位隔离事故复盘](docs/device_slot_isolation_incident.md)：worker 与设备
  token 混淆的影响、修复和回归要求。
- [Schema 说明](docs/schema.md)：Definition、Workload、Implementation 和输出语义。
- [用户算子上传接口](docs/operator_bundle_upload.md)：v6.2 native 单算子 Bundle 的打包、查询、上传和临时存储契约。

KernelGen Agent 的 BatchSimpleOpt 启动、远端 Server 代理、监控与结果回收见
`kernelgen/docs/multiple_device_experiment_runbook.md`。

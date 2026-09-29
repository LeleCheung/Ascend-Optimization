# 多芯片部署与自测教程

本文面向第一次在新加速器环境中部署 KernelGen Server 的用户，只描述可复用的
部署、自测和清理步骤。已经验证过的环境、逐芯片问题和全量结果见
[多芯片 Eval 验证报告](multiple_device_eval_validation_report.md)。协议和内部实现
分别见 [Schema 说明](schema.md)与[设计文档](DESIGN.md)。

## 1. 部署原则

- 使用厂商提供且 PyTorch、Triton、驱动能够配套工作的验收镜像。平头哥 PAI DSW
  本身已经是容器，不再嵌套 Docker。
- 启动前确认空闲物理卡，并通过厂商可见设备变量限制容器或 Server。
- 先安装 `client/`，再使用 `pip install -e .` 安装 Server；同一环境中的 KernelGen 只需导入 `kernelgen_client.xxx`。
- 不安装、升级或替换 PyTorch、Triton、厂商扩展和驱动。缺少 HTTP 依赖时，只在
  当前环境中补充 Pydantic、FastAPI 和 Uvicorn。
- 默认采用本地 Agent + 本地 Server，客户端访问 `127.0.0.1`；远端模式同样保持 Server 监听 loopback，只由 KernelGen 的 SSH stdio HTTP proxy 提供开发机入口。
- Server 会执行客户端提交的 Python 源码，不是安全沙箱，只能向可信客户端开放。

需要从开发机访问目标芯片上的 Server 时，按本文 4.1 节启动 KernelGen 提供的 SSH stdio HTTP proxy；完整适用边界见 `kernelgen/docs/multiple_device_experiment_runbook.md`。

以下变量在全文中复用：

```bash
export DEPLOY_BASE=/path/to/deploy
export KGS_REPO="$DEPLOY_BASE/kernelgen_server"
export SERVER_PORT=18080
export SERVER_URL="http://127.0.0.1:$SERVER_PORT"
export CATALOG="$KGS_REPO/data/kernelgenbench"
```

## 2. 选择后端和设备

| 芯片 | `--backend` | 可见设备变量 | 运行时设备 | 正式计时建议 |
| --- | --- | --- | --- | --- |
| NVIDIA | `cuda` | `CUDA_VISIBLE_DEVICES` | `cuda` | `triton` |
| 天数 | `iluvatar` | `CUDA_VISIBLE_DEVICES` | `cuda` | `triton` |
| 海光 | `hygon` | `HIP_VISIBLE_DEVICES` | `cuda` | `triton` |
| 摩尔线程 | `musa` | `MTHREADS_VISIBLE_DEVICES`、`MUSA_VISIBLE_DEVICES` | `musa` | `triton` |
| 沐曦 | `metax` | `CUDA_VISIBLE_DEVICES` | `cuda` | `triton` |
| 昆仑芯 | `kunlunxin` | `CUDA_VISIBLE_DEVICES` | `cuda` | `triton` |
| 华为昇腾 | `npu` | `ASCEND_RT_VISIBLE_DEVICES` | `npu` | `auto`/`profiler` |
| 平头哥 | `thead` | `CUDA_VISIBLE_DEVICES` | `cuda` | `triton` |
| 燧原 | `enflame` | `TOPS_VISIBLE_DEVICES` | `gcu` | `triton` |
| 寒武纪 | `mlu` | 容器设备挂载及厂商环境 | `mlu` | `triton` |

注意：

- 海光建议只设置 `HIP_VISIBLE_DEVICES`，避免多个可见设备变量产生重复映射。
- 摩尔线程的两个可见设备变量应设置为相同列表。
- 昇腾目标 Triton 的 `do_bench` 延迟数值存在已知问题。Schema/reference
  可执行性测试可以显式使用 `--timing triton`，但正式性能计时应使用
  `--timing auto` 或 `--timing profiler`，采集失败时严禁回退到 wall time。
- 平头哥的非交互环境应确认 `PPU_SDK`、`CUDA_PATH` 和厂商工具路径。
- `backend` 表示厂商逻辑身份，不一定等于 PyTorch 设备类型。例如天数、海光、
  沐曦、昆仑芯和平头哥均通过 `cuda:<index>` 执行，但不能归类为 NVIDIA。

启动前记录环境，不要通过安装核心包修补错误镜像：

昇腾使用自定义 CANN 安装目录时，先加载该安装的 `set_env.sh`，再启动 KGS。`/status.software.runtime_version` 读取 `ASCEND_HOME_PATH`（其次 `ASCEND_TOOLKIT_HOME`）下的 `share/info/runtime/version.info`；仅未选择目录时使用系统默认目录。选定目录缺少版本信息时报告 unknown，不回退成另一个已安装 CANN 的版本；`/status.metadata.sources` 保留信息来源。Triton 语言版本在隔离子进程中按 Torch → Triton 顺序导入读取，与 compiler distribution 的名称和版本独立。

```bash
hostname
python3 --version
python3 -c "import torch, triton; print(torch.__version__, triton.__version__)"
python3 -m pip check || true
```

## 3. 准备源码和轻量依赖

本节使用 KGS v6.3.2 tag。通过 KG CLI 管理部署时，以当前 KG 锁定清单中的精确 commit 为准；版本、兼容条件和验证范围见 [发布说明](releases/v6.3.2.md)。

```bash
cd "$DEPLOY_BASE"
git clone <kernelgen-server-repository> kernelgen_server
cd "$KGS_REPO"

export KGS_RELEASE=v6.3.2
git checkout "$KGS_RELEASE"
sed -n '1,120p' compatibility.yaml
```

正式新实验必须先按[版本兼容与运行选择规则](version_compatibility.md)核对 KG 锁定清单与 Catalog `api_version`。`v6.0` Catalog 默认走 adapter，`v6.2` Catalog 默认走 native。中断续跑沿用 workspace 已记录的 KGS release，不在 campaign 中途自动升级。

运行 FlagGems 时必须准备并记录一个固定 checkout。以下命令使用兼容清单推荐值；KGS v6.3.1 的 Adapter 也可使用 KG 实例显式选择的其他 commit（例如停止实例后执行 `kg server install-flaggems <name> --revision <分支或完整commit>`）。FlagGems-backed native Catalog 仍必须匹配其 manifest 的 exact revision。新行为的边界见[设计文档](DESIGN.md)。

```bash
export FLAGGEMS_ROOT="$DEPLOY_BASE/FlagGems"
git clone --branch feat/new-api-for-kernelgen-server \
  https://github.com/YaooXu/FlagGems.git "$FLAGGEMS_ROOT"
git -C "$FLAGGEMS_ROOT" checkout --detach \
  d64794e63b502cb836bc015a92a62c42de4be05a
test "$(git -C "$FLAGGEMS_ROOT" rev-parse HEAD)" = \
  d64794e63b502cb836bc015a92a62c42de4be05a
export KGS_FLAGGEMS_ROOT="$FLAGGEMS_ROOT"
```

分支名不能替代实际 commit 记录，不允许在 campaign 中途自动跟随分支更新。KGS v6.3.1 的 Adapter 从选定 checkout 读取实际 HEAD，不再要求等于 Catalog 的历史 revision；仍拒绝影响 Python/pytest 配置的未提交修改，并用实际 commit、源码和 case 枚举生成 fingerprint。更新后先重新 inspect 和验证 baseline；缺失测试资产或接口不兼容不能视作通过。旧 KGS v6.3.0 仍执行 Catalog exact 校验，其 tag 保持不变。

默认安装到运行 KernelGen 和 Server 的同一个 Python 环境：

```bash
export SERVER_PYTHON="$(command -v python3)"
"$SERVER_PYTHON" -m pip install -e ./client
"$SERVER_PYTHON" -m pip install -e .
```

Client 的必需依赖是 Pydantic 和 Requests；Server 不声明 PyTorch、Triton 或任何厂商扩展，因此不会
通过本项目替换核心运行时。安装后先验证 KernelGen 所需的导入：

```bash
"$SERVER_PYTHON" -c "import kernelgen_server; print(kernelgen_server.__file__)"
(cd /tmp && "$SERVER_PYTHON" -c \
  "import kernelgen_server; print(kernelgen_server.__file__)")
```

第二条命令必须仍指向当前仓库。若它指向旧的 KernelGen 工作区，通常是另一个
editable 安装把包含同名 `kernelgen_server` 目录的源码根写入了 `.pth`，从而
遮蔽本项目。应修正冲突的 editable 安装后再启动服务，不要用 `PYTHONPATH` 临时
改变优先级。

启动 HTTP 服务还需要 FastAPI 和 Uvicorn。先检查：

```bash
"$SERVER_PYTHON" -c "import fastapi, uvicorn"
```

只有上述导入失败时才补充 `server` extra：

```bash
"$SERVER_PYTHON" -m pip install -e ".[server]"
```

如果验收流程要求隔离 HTTP 依赖，可以创建继承厂商核心包的环境，但 KernelGen
Agent 也必须使用该 Python，才能直接导入 Server：

```bash
python3 -m venv --system-site-packages "$DEPLOY_BASE/server-venv"
export SERVER_PYTHON="$DEPLOY_BASE/server-venv/bin/python"
"$SERVER_PYTHON" -m pip install -e ./client
"$SERVER_PYTHON" -m pip install -e ".[server]"
```

无论采用哪种方式，都应记录本次新增和升级的轻量依赖版本。

## 4. 启动 Server

下面以四张 CUDA 兼容设备、六个并发 worker 为例：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export BACKEND=metax
export TIMING=triton
export DEVICE_COUNT=4
export MAX_WORKERS=6

mkdir -p "$DEPLOY_BASE/runs"
cd "$KGS_REPO"
nohup setsid "$SERVER_PYTHON" -u -m kernelgen_server.server \
  --backend "$BACKEND" \
  --timing "$TIMING" \
  --max-workers "$MAX_WORKERS" \
  --port "$SERVER_PORT" \
  --profile-artifact-root "$DEPLOY_BASE/runs/profiles" \
  >"$DEPLOY_BASE/runs/kernelgen-server.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$DEPLOY_BASE/runs/kernelgen-server.pid"
```

`max_workers` 是请求执行线程上限，可以大于设备数，但不会复制设备 token。设备
slot 与可见设备一一对应，同一张卡同一时刻最多执行一个 Eval、Profile 或 Debug
Job；超出设备数的请求在 Server 内排队。要让本次服务覆盖全部可见设备，应设置
`MAX_WORKERS >= DEVICE_COUNT`。历史缺陷与回归方法见
[设备槽位隔离事故复盘](device_slot_isolation_incident.md)。

确认配置：

```bash
curl -sS "$SERVER_URL/status"
```

检查：

- `backend` 与 `$BACKEND` 一致；
- `server_version` 与 checkout 的 KGS tag 一致，`api_version` 与 Agent 要求的协议版本一致；
- `evaluation_adapters` 及 profile/debug 能力满足本轮任务；
- 本轮需要在途取消时，要求 `capabilities.operation_cancel.enabled=true`，且 `operations` 包含实际需要取消的操作；已发布 v6.2.4 tag 不含此开发分支的新能力，必须使用 KG 锁定清单中的配套精确开发 commit。普通无取消实验不把此可选 capability 作为启动必要条件；
- `devices` 数量与 `$DEVICE_COUNT` 一致；
- `workers` 与 `$MAX_WORKERS` 一致；
- `scheduler.device_slots` 与 `$DEVICE_COUNT` 一致，且
  `scheduler.healthy=DEVICE_COUNT`、`scheduler.checking=0`、
  `scheduler.broken=0`、
  `scheduler.max_active <= scheduler.device_slots`；
- 空闲时 `scheduler.active=0`、`scheduler.waiting=0`、
  `scheduler.available=DEVICE_COUNT`；
- `timing` 与 `$TIMING` 一致；
- `target`、`software` 与当前验收镜像一致；无法探测的字段可以为空；
- 未枚举到不在本次授权范围内的设备。

### 4.1 从开发机访问远端 loopback Server

远端模式仍必须让 Server 监听 `127.0.0.1`，不得为了开发机访问而改成公网地址。SSH stdio HTTP proxy 位于 KernelGen 仓库的 `scripts/remote_server/`，不属于 KernelGen Server 的测试脚本。

在开发机的 KernelGen 仓库根目录启动长期代理：

```bash
cd /path/to/kernelgen
bash scripts/remote_server/run_persistent_remote_http_proxy.sh \
  --device ppu \
  --listen-port 19607 \
  --max-streams 10
```

入口默认从 `tests/multi_device_batch_hosts.conf` 读取目标地址、JumpServer 登录、容器和 Server 端口；临时清单使用 `--inventory /path/to/hosts.conf`，临时 Server 端口使用 `--remote-port <port>` 覆盖。`--max-streams` 是单条 SSH 会话内的逻辑 HTTP 流上限，通常设为 Server `--max-workers + 2`，为 `/status` 和短 Debug 请求保留余量；它不会增加 SSH 连接数。

另一个终端只为当前进程清除可能劫持 loopback 请求的全局 HTTP proxy，再检查 Server：

```bash
export KERNELGEN_SERVER_URL=http://127.0.0.1:19607
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

curl --noproxy '*' -fsS "$KERNELGEN_SERVER_URL/status"
```

长期代理固定使用 `ServerAliveInterval=6`、`ServerAliveCountMax=30` 和 `TCPKeepAlive=yes`。SSH 异常断开时，所有在途 Eval、Profile 或 Debug 请求返回连接失败且不会自动重放；只有后续新请求会重新建立一条 SSH 会话。停止前台代理即断开 SSH，不改变远端 Server。兼容入口 `scripts/remote_server/run_remote_http_proxy.sh` 会为每个本地连接新建 SSH，不用于高并发实验。完整的多设备适用边界、错峰启动、状态判断和结果回收见 KernelGen 的 `docs/multiple_device_experiment_runbook.md`。

Server 启动时会对每张可见设备执行 FP32 矩阵乘、softmax、结果校验和同步；
非昇腾的 Triton 模式还会执行短时 `do_bench`。某张卡未通过时，对应
`scheduler.slots[].state` 为 `broken`，不会进入调度池；所有卡都未通过则服务
拒绝启动。所有可见设备的启动探针并行执行，启动耗时接近最慢单卡探针。可用
`--health-probe-timeout` 调整启动和故障后单卡探针超时，默认 30 秒；旧参数名
`--startup-probe-timeout` 仍兼容。

运行中请求 `TIMEOUT` 或隔离 worker 异常退出后，当前 slot 先进入 `checking`，
并在全新进程中运行同一强探针。通过则恢复调度：超时返回 `TIMEOUT`，worker
异常退出保留原错误；探针失败则进入 `broken` 并返回
`SUSPECTED_DEVICE_ERROR`。Server 不会把同一请求转发到另一张卡，避免异常请求
扩大污染范围。普通 `RUNTIME_ERROR` 不触发探针。

`scheduler.incidents` 是本次 Server 生命周期内需要强探针确认的超时、异常退出或运行中取消总数，`recovered` 是强探针通过并恢复 slot 的次数。排队阶段取消尚未接触设备，不增加这两个计数。
Server 不自动复位物理卡；发现
`broken>0` 后只会停止使用对应 slot，当前一组任务可在其余健康设备上完成。该组
结束后不要启动下一组，确认不会影响其他用户，再按厂商流程重启对应运行时和
Server。若全部 slot 都坏，新请求会返回 HTTP 503，本组也只能提前终止。

需要向厂商提供逐请求证据时，为本轮使用唯一目录并在启动参数中增加：

```bash
export REQUEST_AUDIT_ROOT="$DEPLOY_BASE/runs/request-audit-<run-name>"
# 在 kernelgen_server.server 参数中增加：
--request-audit-root "$REQUEST_AUDIT_ROOT"
```

该目录按 Server 生命周期创建 `server_<session>/`，其中 `manifest.json` 记录
启动配置，`events.jsonl` 关联设备分配、超时/崩溃和强探针事件，
`requests/<request_id>/` 保存完整请求以及响应或异常。启用后先检查
`/status.request_audit.enabled=true` 并保存其 `session_root`。审计只覆盖
`/preflight` 和 `/evaluate`；Server stdout/stderr 仍须单独落盘。

### 4.2 取消排队或运行中的评测操作

支持取消的客户端先从 `/status.capabilities.operation_cancel` 判断能力，不根据 KG 或 KGS release 推断。提交 `POST /preflight`、`POST /evaluate` 或 `POST /profile` 时，在 `X-KernelGen-Operation-Id` Header 中传入客户端生成的唯一 ID；请求体和 Protocol v6.2 Schema 不变。

```bash
export OPERATION_ID=example-eval-001
curl -sS -X DELETE "$SERVER_URL/operations/$OPERATION_ID"
curl -sS "$SERVER_URL/operations/$OPERATION_ID"
```

`DELETE` 是幂等的取消请求：排队操作在取得 slot 前结束，运行中的 Eval/Preflight 会终止隔离进程树，Profile 会终止当前 vendor 工具进程树；运行中操作随后对原 slot 执行强探针，通过后才重新入队。原始 POST 在确认取消后返回 HTTP 409，`detail.code=OPERATION_CANCELLED`。若取消与正常完成竞态，已经完成的操作保持 `SUCCEEDED`，不会伪装成 `CANCELLED`。终态查询在 Server 内保留一小时，Server 重启后不持久化。

## 5. Agent 调试任务

Debug Job 默认启用：

```bash
nohup setsid "$SERVER_PYTHON" -u -m kernelgen_server.server \
  --host 127.0.0.1 \
  --backend "$BACKEND" \
  --timing "$TIMING" \
  --max-workers "$MAX_WORKERS" \
  --debug-artifact-root "$DEPLOY_BASE/runs/debug-jobs" \
  --port "$SERVER_PORT" \
  >"$DEPLOY_BASE/runs/kernelgen-server.log" 2>&1 </dev/null &
```

Debug Job 与正式 eval/profile 共用设备 slot，不需要另建常驻设备进程。它能执行任意
命令，不是安全沙箱；同机部署时必须继续监听 `127.0.0.1`。不需要该能力时传入
`--disable-debug-jobs`。服务会拒绝客户端覆盖设备可见性变量，防止任务绕过本次服务
被授权的卡。

启用后执行统一自测：

```bash
"$SERVER_PYTHON" tests/live_debug_validation.py \
  --server "$SERVER_URL" \
  --expected-backend "$BACKEND" \
  --expected-device-count "$DEVICE_COUNT" \
  --expected-workers "$MAX_WORKERS" \
  --output "$DEPLOY_BASE/runs/debug-smoke.json"
```

## 6. 快速功能与容错自测

以下命令验证状态端点、reference-as-solution、preflight、正确性、计时、可移植
Triton kernel、并发 `RUNTIME_ERROR`、worker 硬退出隔离和失败后恢复。脚本使用
v6.2 `EvaluatorBinding`，Definition、oracle 和 workload 均由 Server 根据 catalog
绑定加载，客户端不再内嵌或裁剪这些可信评测资产。每个算子提交一次完整请求；
Server 在同一请求内分别解析 `correctness_run ?? run` 和
`timing_run ?? run`，primary reference 不可执行且存在 `torch_run` 时，脚本再以
整轮 Torch fallback 做 reference-as-solution：

```bash
cd "$KGS_REPO"
"$SERVER_PYTHON" tests/live_server_validation.py \
  --server "$SERVER_URL" \
  --catalog "$CATALOG" \
  --expected-backend "$BACKEND" \
  --expected-device-count "$DEVICE_COUNT" \
  --expected-workers "$MAX_WORKERS" \
  --expected-timing "$TIMING" \
  --definition kernelgenbench_square \
  --concurrency "$MAX_WORKERS" \
  --warmup-ms 50 \
  --benchmark-ms 20 \
  --request-timeout 300 \
  --output "$DEPLOY_BASE/runs/server-smoke.json"
```

`--catalog` 是运行脚本所用的本地 catalog 副本；绑定名默认读取 manifest 的 `name`，
没有该字段时使用 catalog 目录名。若本地目录名与 Server 内置目录名不同，显式传入
`--catalog-name <server-catalog-name>`。该脚本只接受 native catalog；FlagGems
adapter 使用对应的 adapter E2E 测试。

v6.2 bound request 总是执行 Server catalog 中该算子的全部 workload，因此快速自测
应选择 workload 较小且有可移植 Triton candidate 的
`kernelgenbench_square`，不能再用客户端参数缩减 workload。Triton 与容错阶段默认
使用该算子；也可以用 `--smoke-definition` 显式选择，当前可移植 Triton smoke 只为
`kernelgenbench_square` 提供。

`--concurrency` 必须至少等于设备数，并且本教程要求 worker 数不小于设备数，才能
验证请求确实分发到全部可见设备。命令应返回 0；失败后不要通过重启 Server 掩盖隔离
或恢复问题。

## 7. KernelGenBench v6.2 全量 reference 自测

下面的快速命令使用短计时预算验证全部 210 个 Definition，以及经过确定性覆盖采样的
15353 个活跃 correctness workload 和 11695 个活跃 timing workload。完整的
57603/28198 条 `_full.jsonl` 归档不由 Server 执行：

```bash
"$SERVER_PYTHON" tests/live_server_validation.py \
  --server "$SERVER_URL" \
  --catalog "$CATALOG" \
  --expected-backend "$BACKEND" \
  --expected-device-count "$DEVICE_COUNT" \
  --expected-workers "$MAX_WORKERS" \
  --expected-timing "$TIMING" \
  --all-definitions \
  --parallel-definitions "$MAX_WORKERS" \
  --skip-triton \
  --skip-resilience \
  --warmup-ms 5 \
  --benchmark-ms 5 \
  --request-timeout 900 \
  --output "$DEPLOY_BASE/runs/kernelgenbench-v62-full.json"
```

这里的 `--skip-triton` 只跳过额外的可移植 Triton smoke；每个 Definition 的完整
bound request 仍会覆盖 catalog 中全部 correctness/timing workload，并按 phase
选择对应 oracle，reference timing 严格使用 Server 配置的 `do_bench`。5 ms 是快速
可执行性预算，
不能用来发布性能或加速比。正式测量应使用默认的 `warmup_ms=1000` 和
`benchmark_ms=100`，或经过统一评审的预算。

不同厂商的原生 PyTorch API 能力不同，全量脚本非零退出不一定表示 Server 失败。
需要按 Definition 区分：

- Schema、输入构造、序列化或隔离进程错误；
- 厂商 API 未实现或上游 pytest 已有跳过条件；
- 超大 workload 的显存不足或并发资源竞争；
- 数值不一致；
- 请求级超时。

已确认无法在目标设备执行的 Definition 可以通过重复传入
`--skip-definition <name>` 跳过，但必须在验证报告中记录原因。

## 8. 停止服务与留档

确认没有客户端仍在使用服务后，只停止本次 PID 文件指向的进程：

```bash
server_pid="$(cat "$DEPLOY_BASE/runs/kernelgen-server.pid")"
ps -o pid,args -p "$server_pid"
kill "$server_pid"
```

记录以下信息：

- 主机、芯片、驱动、镜像、容器启动参数和物理卡；
- Python、PyTorch、Triton、厂商扩展及新增轻量包；
- Server backend、计时策略、设备数和 worker 数；
- smoke、容错和全量结果 JSON；
- 所有失败、定向复测及最终资源释放状态。

新的多芯片问题和结论统一追加到
[多芯片 Eval 验证报告](multiple_device_eval_validation_report.md)，不要继续扩充
本教程。

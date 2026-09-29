# 910B 旧版部署与运行指南

> **历史文档。** 本文基于 flashinfer-bench v4 和旧 Eval Service，仅用于复盘，
> 不适用于当前 kernelgen_server。当前部署见
> `kernelgen_server/docs/multiple_device_deploy_tutorial.md`，当前 E2E 操作见
> `kernelgen/docs/operations/multiple_device_experiment_runbook.md`。

> 在 Ascend 910B NPU 上部署 eval server + 运行 KernelGen 优化流程

---

## 前置条件

- 910B 机器上有 Docker 容器（如 `xuyao_sglang`），已配置 CANN 环境
- 容器内能访问 NPU 设备
- flashinfer-bench 使用 KernelGen 已验证的 `v4.0` tag（commit `cb2d092`）

---

## Phase 1: 配置 eval server

### 1.1 确认 flashinfer-bench 代码

```bash
# 在 910B docker 内
cd /data/jiabei/flashinfer-bench
git fetch --tags origin
git checkout v4.0
git describe --tags --exact-match
# 期望: v4.0
test "$(git rev-parse HEAD)" = \
  "cb2d0921c633dd580ab7d1625a2e8719737421a4"
test -f eval_service/server.py
```

### 1.2 启动 server

```bash
# 在 910B docker 内
cd /data/jiabei/flashinfer-bench
PYTHONPATH=$(pwd):$PYTHONPATH \
python3 -u eval_service/server.py \
  --backend npu \
  --perf profiler \
  --trace-root /data/jiabei \
  --port 8000 &
```

### 1.3 验证 server

```bash
# 在 910B docker 内
curl http://localhost:8000/status
# 期望同时包含:
# - backend: "npu"
# - service_instance_id: 当前服务进程身份
# - preflight.schema_version / preflight.policy_version
# - gpus: 设备状态
# 应看到容器实际分配的 NPU 设备，且全部 idle
```

如果 `/status` 没有 `service_instance_id` 或 `preflight`，先确认代码是
flashinfer-bench `v4.0` 且运行中的 server 已重启；否则 KernelGen 的
`preflight_kernel` 无法生成有效 receipt。

### 1.4 验证 preflight 功能

preflight 使用与 `/evaluate_on_workloads` 相同的完整
Solution/Definition/Workloads payload，但只执行静态策略、目标机编译和一次
smoke launch：

```bash
python3 - <<'PY'
import requests

payload = {
    "solution": {
        "name": "test_relu",
        "definition": "test_relu",
        "author": "deployment-check",
        "spec": {
            "language": "python",
            "target_hardware": ["Ascend910B"],
            "entry_point": "main.py::run",
            "destination_passing_style": False,
        },
        "sources": [{
            "path": "main.py",
            "content": "import torch\ndef run(x): return torch.relu(x)",
        }],
    },
    "definition": {
        "name": "test_relu",
        "op_type": "pointwise",
        "tags": ["schema:workload-shape-dtype-v4"],
        "axes": {},
        "inputs": {"x": {"shape": "dynamic", "dtype": "dynamic"}},
        "outputs": {"output": {"shape": "dynamic", "dtype": "dynamic"}},
        "reference": "import torch\ndef run(x): return torch.relu(x)\n",
    },
    "definition_name": "test_relu",
    "workloads": [{
        "definition": "test_relu",
        "workload": {
            "uuid": "test-0",
            "axes": {},
            "inputs": {
                "x": {
                    "type": "random",
                    "shape": [1024],
                    "dtype": "float32",
                }
            },
        },
        "solution": None,
        "evaluation": None,
    }],
    "trace_set_key": "",
}

result = requests.post(
    "http://localhost:8000/preflight",
    json=payload,
    timeout=600,
).json()
print(result)
assert result["status"] == "PASSED", result
assert result["stage"] == "complete", result
assert result["static"]["status"] == "PASSED", result
assert result["service_instance_id"], result
PY
```

preflight 不检查数值正确性和性能。这个检查通过后，还要继续验证正式 eval。

### 1.5 验证 eval 功能

```bash
# 在 910B docker 内
python3 -c "
import requests, json
resp = requests.post('http://localhost:8000/evaluate_on_workloads', json={
    'solution': {
        'name': 'test_relu', 'definition': 'test_relu', 'author': 'test',
        'spec': {'language': 'python', 'target_hardware': ['Ascend910B'],
                 'entry_point': 'main.py::run', 'destination_passing_style': False},
        'sources': [{'path': 'main.py', 'content': 'import torch\ndef run(x): return torch.relu(x)'}]
    },
    'definition': {
        'name': 'test_relu', 'op_type': 'pointwise',
        'tags': ['schema:workload-shape-dtype-v4'], 'axes': {},
        'inputs': {'x': {'shape': 'dynamic', 'dtype': 'dynamic'}},
        'outputs': {'output': {'shape': 'dynamic', 'dtype': 'dynamic'}},
        'reference': 'import torch\ndef run(x): return torch.relu(x)\n'
    },
    'definition_name': 'test_relu',
    'workloads': [{'definition': 'test_relu', 'workload': {'uuid': 'test-0', 'axes': {}, 'inputs': {'x': {'type': 'random', 'shape': [1024], 'dtype': 'float32'}}}, 'solution': None, 'evaluation': None}],
    'trace_set_key': ''
}, timeout=60)
r = resp.json()
t = list(r['traces_by_uuid'].values())[0]
print(f'Status: {t[\"evaluation\"][\"status\"]}')
print(f'Speedup: {t[\"evaluation\"].get(\"performance\", {}).get(\"speedup_factor\")}')
"
# 期望输出: Status: PASSED, Speedup: ~1.0
```

---

## Phase 2: 配置 KernelGen 环境

### 2.1 安装依赖

```bash
# 在 910B docker 内

# Node.js + npm (Claude Code 2.1.220 要求 Node.js >= 22)
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs openssh-client
node -v   # 期望: v22.x 或更高
npm -v    # 期望: 10.x

# Claude Code：安装到部署目录，不修改全局 npm 包
mkdir -p /data/jiabei/runtime/claude
test -f /data/jiabei/runtime/claude/package.json ||
  npm --prefix /data/jiabei/runtime/claude init --yes
npm --prefix /data/jiabei/runtime/claude install \
  --save-exact --no-audit --no-fund \
  --fetch-retries 5 --fetch-timeout 600000 \
  @anthropic-ai/claude-code@2.1.220
export PATH=/data/jiabei/runtime/claude/node_modules/.bin:$PATH
claude --version   # 期望: 2.1.220 (Claude Code)

# Python 依赖
cd /data/jiabei/kernelgen
pip install -r requirements.txt
```

### 2.2 验证 Claude Code

```bash
# 在 910B docker 内
claude --version
# 期望: 2.1.220 (Claude Code)

# 如果需要登录 (首次使用)
claude auth login
```

### 2.3 获取 KernelGen 代码

```bash
# 安装 SSH key (首次)
mkdir -p /root/.ssh
# 把你的 id_ed25519 拷贝到 /root/.ssh/id_ed25519
chmod 600 /root/.ssh/id_ed25519
ssh-keyscan gitee.com >> /root/.ssh/known_hosts 2>/dev/null

git clone git@gitee.com:BaaiAC/kernelgen.git /data/jiabei/kernelgen
cd /data/jiabei/kernelgen
git switch dev
```

### 2.4 验证 KernelGen

```bash
# 在 910B docker 内
cd /data/jiabei/kernelgen
PYTHONPATH=/data/jiabei python3 -c "
from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.data.tool_context import ToolContext
from kernelgen.mcp_server.server import mcp
from kernelgen.workflows.simple_opt import SimpleOptWorkflow
print('✅ All imports OK')
"
```

### 2.5 验证原生 Agent 与 MCP 配置

```bash
# 在 910B docker 内
cd /data/jiabei/kernelgen

# 检查 .claude/ 原生 Agent、.mcp.json 和 MCP 工具声明是否一致
PYTHONPATH=/data/jiabei python3 -m kernelgen.tools.doctor

# 期望输出:
# KernelGen native-agent configuration is healthy: /data/jiabei/kernelgen
```

当前架构不再把 `eval_round` 当作用户直接调用的独立 CLI。Coder 通过
stdio MCP 调用
`preflight_kernel → eval_round → finalize_round(finalize)`；当 eval 返回
`profile_required=true` 时，由原生 `kernel-profile-analyzer` subagent 独占
`get_profile_context/profile_workloads/record_profile_analysis`。

MCP server 把 definition、workloads、target、eval endpoint、ledger、
preflight receipt 和 profile/conclusion gate 绑定到同一个 workspace。模型
不能把工具重定向到其他 agent 目录。`preflight_kernel` 内部调用本机 eval
service 的 `/preflight`，不是另起一个远端 MCP 服务。

---

## Phase 3: 运行 SimpleOptWorkflow

### 3.1 完整 e2e 测试

```bash
# 在 910B docker 内
cd /data/jiabei/kernelgen

source env.sh
export DEFINITION_NAME=flaggems_rsqrt
export FIB_TRACE_ROOT=/data/jiabei/kernelgen_server/data/flaggems-v5
export FIB_EVAL_SERVER=http://localhost:8000
export WT="/data/jiabei/kernelgen/runs/e2e_simple_opt/${DEFINITION_NAME}"

python3 -u examples/simple_opt/run_example.py \
  --definition-name "$DEFINITION_NAME" \
  --trace-root "$FIB_TRACE_ROOT" \
  --workspace "$WT" \
  --eval-server "$FIB_EVAL_SERVER" \
  --target-hardware Ascend910B \
  --model "$MODEL" \
  --early-stop-rounds 2 \
  --min-rounds 2 \
  --max-round 15 \
  --clean
```

example 入口直接按 `definition_name` 从 KernelGen Server catalog 加载 Definition
与 Workload，不执行 extractor。不设置 `FIB_TRACE_ROOT` 时使用
`kernelgen_server` 提供的 FlagGems v5 catalog。
SimpleOpt 默认关闭 profile，passing round 会直接调用 `finalize_round`。

### 3.2 验证 e2e 结果

```bash
echo "=== Workspace structure ==="
ls -la "$WT"

echo ""
echo "=== Ledger ==="
python3 -c "import json, os; l=json.load(open(os.path.join(os.environ['WT'], '.ledger.json'))); print('rounds=', len(l.get('rounds', [])), 'best=', l.get('best_geo_mean'))"

echo ""
echo "=== Best kernel ==="
sed -n '1,160p' "$WT/.best_kernel.py"

test -f "$WT/.kernelgen/tool-context.json"
test -f "$WT/.best_kernel.py"
test -f "$WT/optimize_definition_output.json"
```

`optimize_definition_output.json` 与 SimpleOpt 的返回对象字段一致，包含后续 PR Agent
需要的真实 `best_code`、`best_geo_mean`、round 数和 workspace。

### 3.3 批量运行与监控

多个已有 Definition 可以通过独立批量入口并行运行，单算子
`SimpleOptWorkflow` 的教程契约保持不变：

```bash
export BATCH_WORKSPACE=/data/jiabei/kernelgen/runs/batch_simple_opt

python3 -u examples/batch_simple_opt_definition/run_example.py \
  -n flaggems_rsqrt \
  -n flaggems_repeat \
  --trace-root /data/jiabei/kernelgen_server/data/flaggems-v5 \
  --workspace "$BATCH_WORKSPACE" \
  --eval-server http://localhost:8000 \
  --max-workers 2 \
  --clean
```

batch 的汇总文件为 `batch_simple_opt_definition_output.json`。如果某个
SimpleOpt 子任务抛出异常，其他已完成任务的
`optimize_definition_output.json` 仍会被收集，异常任务在汇总中标为
`FAILED`，示例入口最终返回非零状态。

长批次使用只读监控工具；默认间隔为 180 秒，并且只在状态变化时打印：

```bash
python3 -m kernelgen.tools.monitor_batch_simple_opt \
  --workspace "$BATCH_WORKSPACE" \
  -n flaggems_rsqrt \
  -n flaggems_gelu \
  --server-url http://localhost:8000
```

---

## Phase 4: 运行 KernelGenWorkflow 完整 E2E

下面以 KernelGen Server catalog 中的 `flaggems_rsqrt` 为例。
`--trace-root` 直接传 catalog 根目录，`--trace-set-key` 留空。

### 4.1 首次运行 1R

```bash
cd /data/jiabei/kernelgen

ANTHROPIC_AUTH_TOKEN='<your-token>' \
FIB_EVAL_SERVER=http://localhost:8000 \
python3 -u examples/kernel_gen/run_example.py \
  --definition flaggems_rsqrt \
  --trace-root /data/jiabei/kernelgen_server/data/flaggems-v5 \
  --target-hardware Ascend910B \
  --eval-server http://localhost:8000 \
  --model deepseek-v4-pro[1m] \
  --n-parallel 2 \
  --n-epoch 1 \
  --workspace /data/jiabei/kernelgen/runs/flaggems_rsqrt_e2e \
  --timeout 3600 \
  --clean
```

执行顺序：

```text
1R/shared_analysis
  → 1R/agent0 和 1R/agent1 并行优化
  → epoch KB reducer
  → 1R/synthesis/synthesis.json
```

每个 Coder 内部的正式 round 顺序是：

```text
edit → preflight_kernel → eval_round
     → [ProfileAnalyzeAgent] → finalize_round(finalize)
```

preflight 失败只会写
`.kernelgen/preflight/attempts.jsonl`，不会增加 `.ledger.json` 的 round 数。

### 4.2 检查 1R 产物

```bash
RUN=/data/jiabei/kernelgen/runs/flaggems_rsqrt_e2e

test -f "$RUN/1R/shared_analysis/analysis.json"
test -f "$RUN/1R/agent0/.ledger.json"
test -f "$RUN/1R/agent1/.ledger.json"
test -f "$RUN/1R/synthesis/synthesis.json"

git -C "$RUN/kb" log --oneline --max-count=5
find "$RUN/kb/experience/by_definition/pointwise/flaggems_rsqrt/Ascend910B" \
  -maxdepth 1 -type f -print
```

canonical KB 路径必须使用 definition 的真实 `op_type=pointwise`，正常情况下
包含 `experience.md` 和 `detailed.md`。agent/synthesis 内的 `kb/` 只是内容
快照，不应包含 `.git`。

### 4.3 查看完整日志

```bash
RUN=/data/jiabei/kernelgen/runs/flaggems_rsqrt_e2e

# 人类可读日志：完整 thinking、文本、工具输入和工具结果
tail -f "$RUN/1R/agent0/.kernelgen/claude-runtime.log"

# synthesis 日志
less "$RUN/1R/synthesis/.kernelgen/claude-runtime.log"

# provider 原始 stream-json；只在协议排查时查看
less "$RUN/1R/agent0/.kernelgen/claude-stream.jsonl"
```

`claude-runtime.log` 会对 partial stream 和最终聚合消息去重，不需要再从 tmux
屏幕里拼接 `Read/Glob` 摘要。

多 agent 并行时，共享启动终端只输出 epoch/agent 生命周期摘要，避免多个 Runtime
的多行 thinking/tool 内容交错。每个 agent 的完整内容仍实时写入其独立的
`.kernelgen/claude-runtime.log`；需要跟踪某个 agent 时直接 `tail -f` 该文件。

### 4.4 从 checkpoint 运行 2R

1R 完成后，2R 直接读取：

```text
1R/shared_analysis/analysis.json
1R/synthesis/synthesis.json
1R/agent*/.ledger.json
```

运行：

```bash
cd /data/jiabei/kernelgen

ANTHROPIC_AUTH_TOKEN='<your-token>' \
FIB_EVAL_SERVER=http://localhost:8000 \
python3 -u examples/kernel_gen/run_example.py \
  --definition flaggems_rsqrt \
  --trace-root /data/jiabei/kernelgen_server/data/flaggems-v5 \
  --target-hardware Ascend910B \
  --eval-server http://localhost:8000 \
  --model deepseek-v4-pro[1m] \
  --n-parallel 2 \
  --start-epoch 2 \
  --n-epoch 2 \
  --workspace /data/jiabei/kernelgen/runs/flaggems_rsqrt_e2e
```

`--n-epoch 2` 表示最终 epoch 编号是 2；不是“从当前位置再跑两个 epoch”。
续跑时不要使用 `--clean`。

---

## Phase 5: 后续更新代码

```bash
cd /data/jiabei/kernelgen
git pull origin dev
```

---

## 环境变量速查

| 变量 | 值 | 说明 |
|---|---|---|
| `FIB_EVAL_SERVER` | `http://localhost:8000` | eval server 地址 |
| `FIB_TRACE_ROOT` | `/data/jiabei/kernelgen_server/data/flaggems-v5` | 正式 FlagGems v5 catalog |
| `FIB_TRACE_SET_KEY` | 空字符串 | `FIB_TRACE_ROOT` 已直接指向 catalog 根目录 |
| `MODEL` | `deepseek-v4-pro[1m]` | LLM 模型名 |
| `ANTHROPIC_BASE_URL` | provider endpoint | Claude Runtime API 地址 |
| `ANTHROPIC_AUTH_TOKEN` | provider token | 通过环境变量传入，不要写入仓库 |

---

## 常见问题

### Q: `FileNotFoundError: 'claude'`
A: 没装 Claude Code。执行:
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
mkdir -p /data/jiabei/runtime/claude
test -f /data/jiabei/runtime/claude/package.json ||
  npm --prefix /data/jiabei/runtime/claude init --yes
npm --prefix /data/jiabei/runtime/claude install \
  --save-exact --no-audit --no-fund \
  --fetch-retries 5 --fetch-timeout 600000 \
  @anthropic-ai/claude-code@2.1.220
export PATH=/data/jiabei/runtime/claude/node_modules/.bin:$PATH
```

### Q: `ModuleNotFoundError: No module named 'requests'`
A: `pip install requests`

### Q: `504 Gateway Timeout`
A: 从远程访问 server 时遇到（中间有网关）。解决: 在 910B 本机跑 agent，用 `--eval-server http://localhost:8000`。

### Q: `Value-returning style callable: expected N params`
A: Agent 写的 kernel 没有遵循 DPS 接口。DPS 要求 `run(...inputs, output)` 接收 output 参数并写入，而不是 `return output`。检查 definition 的 `dps_param_count` 和 kernel 的 `run()` 签名是否匹配。

### Q: `PREFLIGHT_REQUIRED`
A: 当前候选没有有效 receipt。常见原因是跳过 preflight、通过后又编辑了
`tmp/main.py`、receipt 已被一次 eval 消费，或 eval server 已重启。对当前
代码重新调用 `preflight_kernel`，确认 `status=PASSED` 后立即调用
`eval_round`。

### Q: preflight 一直 `FAILED`
A: 先看 MCP 返回的 `static.errors`；静态通过后，再看
`target.per_workload` 中具体 specialization 的 compile/smoke 错误。失败不是
正式 round，不要调用 `finalize_round`，修复代码后直接再次
preflight。

### Q: tmux 里只看到 `Read`、`Glob`，如何看完整过程？
A: 查看对应 workspace 的 `.kernelgen/claude-runtime.log`。其中保留完整
thinking、工具参数与结果；`.kernelgen/claude-stream.jsonl` 是未经整理的
provider 原始事件。

### Q: Server 进程挂了 / 需要重启
A:
```bash
ps aux | grep server.py | grep -v grep | awk '{print $2}' | xargs kill
find /data/jiabei/flashinfer-bench -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
PYTHONPATH=/data/jiabei/flashinfer-bench:$PYTHONPATH \
python3 -u eval_service/server.py \
  --backend npu --perf profiler --trace-root /data/jiabei --port 8000 &
```

### Q: `pip install` 需要哪些包？
A: KernelGen 的 Python 依赖:
```bash
pip install -r requirements.txt

# 还要运行 host 测试时
pip install -r requirements-dev.txt
```
Server 的依赖（flashinfer-bench 已有）: `fastapi uvicorn requests pydantic safetensors`

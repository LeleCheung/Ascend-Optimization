# KGRunner 设计文档

> 版本: 1.0 | 日期: 2026-07-16 | 仓库: gitee.com/BaaiAC/kernelgen (dev 分支)

## 1. 概述

KGRunner 是一个 **Code Agent 编排框架**，用于并行调度 AI 代码生成/优化任务。核心能力：

- 多 GPU 设备池化管理
- Workspace 隔离（git worktree / 临时目录）
- Code Agent 生命周期管理（启动/恢复/终止）
- 可插拔 backend（Claude Code / Codex / 其他）
- 并行 fan-out 执行

## 2. 核心 API

只有两个执行原语：

| API | 作用 | 说明 |
|-----|------|------|
| `run(agent, input)` | 执行单个 agent | 自带 retry/resume/timeout |
| `run_parallel(agent, inputs)` | 并行执行多个任务 | GPU 池约束并发度 |

### 最简用法

```python
from kgrunner import run_parallel, load_agent, GPUPool

agent = load_agent("agents/auto_gen")      # 读 agent.yaml + config.yaml
pool = GPUPool()                            # 自动检测 GPU

results = run_parallel(agent, "ops.jsonl", gpu=pool) # JSONL 输入，并行执行
```

## 3. Agent 定义

一个 agent 是一个目录：

```
agents/auto_gen/
├── agent.yaml          # IO schema + 资源声明
├── config.yaml         # backend + workspace + 运行参数
└── prompts/
    └── generate_op.md  # prompt 模板 ({{VAR}} 替换)
```

### agent.yaml

```yaml
name: auto_gen
version: 1.1.0
description: 为 FlagGems 生成新算子

prompt_template: prompts/generate_op.md

resources:
  gpu: 1
  workspace: true
  workspace_name_key: OPERATOR    # 用哪个 input 字段命名 workspace

inputs:
  OPERATOR: { type: string, required: true, desc: "算子名" }

outputs:
  operator:        { type: string }
  status:          { type: "enum(success|failed)" }
  accuracy_passed: { type: bool, optional: true }
  error_message:   { type: string, optional: true }
```

### config.yaml

```yaml
backend:
  type: claude          # claude / codex / 其他
  bin: claude           # 可执行文件路径
  budget: null          # 单次预算上限 (USD)

workspace:
  type: git_worktree    # git_worktree / tempdir / directory
  repo_dir: /path/to/FlagGems
  base_branch: master
  branch_prefix: auto-gen

timeout: 1800           # 单次执行超时 (秒)
max_retries: 3          # 失败后重试次数
max_resumes: 5          # API 中断后恢复次数
```

## 4. 架构

```
┌─────────────────────────────────────────────────┐
│  Workflow (Python script / callable)             │
│  run_parallel(agent, inputs, gpu=pool)           │
├─────────────────────────────────────────────────┤
│  run_parallel() — 并行调度                        │
│  ├── 解析 JSONL 输入                             │
│  ├── 自动创建 Workspace (从 agent config)        │
│  ├── ThreadPoolExecutor (并发度 = GPU 数)        │
│  └── 每个 task → run()                          │
├─────────────────────────────────────────────────┤
│  run() — 单次执行                                │
│  ├── 渲染 prompt 模板                            │
│  ├── Backend.spawn() 启动子进程                  │
│  ├── 等待完成 / timeout 杀进程                   │
│  ├── Backend.extract_text() → Layer 1 解析       │
│  ├── output_parser() → Layer 2 解析              │
│  ├── 失败 → detect_resumable_error → resume      │
│  └── 失败 → retry                               │
├─────────────────────────────────────────────────┤
│  Backend (可插拔)          │  Workspace (可插拔)  │
│  ├── ClaudeBackend         │  ├── GitWorktree    │
│  ├── CodexBackend (未来)   │  ├── TempDir        │
│  └── ...                   │  └── Directory      │
├─────────────────────────────────────────────────┤
│  GPUPool — 设备池 (lock-file 原子分配)           │
│  ├── 自动检测平台 (11 厂商)                      │
│  ├── 阻塞 acquire / release                     │
│  └── 设备可见性环境变量                          │
└─────────────────────────────────────────────────┘
```

## 5. Backend 抽象

`CodeAgentBackend` ABC 定义 6 个方法：

| 方法 | 职责 |
|------|------|
| `spawn(prompt, cwd, device_id)` | 启动子进程 |
| `resume(session_id, cwd)` | 恢复中断的 session |
| `kill(proc)` | 终止子进程 |
| `extract_text(proc)` | Layer 1: 从 backend 格式提取纯文本 |
| `detect_resumable_error(proc)` | 判断是否可恢复 |
| `extract_session_id(proc)` | 提取 session ID |

当前实现：`ClaudeBackend`（CC stream-json JSONL 格式）。

## 6. 两层输出解析

| 层 | 职责 | 谁实现 |
|----|------|--------|
| Layer 1 | 从 backend 特定格式提取纯文本 | Backend（`extract_text`） |
| Layer 2 | 从纯文本提取业务 JSON | `output_parser`（用户可替换） |

默认 Layer 2：在文本中找 ```json 代码块，用 brace-counting 提取 JSON 对象。

## 7. 设备管理

```python
pool = GPUPool()                    # 零参数：自动检测平台 + 设备列表
pool = GPUPool(device_ids=[4, 5])   # 指定设备
pool = GPUPool(vendor="ascend")     # 指定厂商
```

检测顺序（参考 FlagGems DeviceDetector）：
1. 环境变量：`GEMS_VENDOR` / `FLAGGEMS_VENDOR` / `KGRUNNER_VENDOR`
2. 系统命令探测：`nvidia-smi` / `npu-smi` / `cnmon` / ...

支持 11 个厂商，设备可见性环境变量由框架自动设置（多变量支持，如 ascend 需同时设 `ASCEND_RT_VISIBLE_DEVICES` + `NPU_VISIBLE_DEVICES`）。

## 8. Workspace

```python
class Workspace(ABC):
    def allocate(self, name: str) -> Path   # 基类：创建 + 记录
    def deallocate(self, name: str)         # 基类：清理 + 删记录
    def _create(self, name: str) -> Path    # 子类实现
    def _cleanup(self, path: Path)          # 子类实现
```

| 策略 | 适用场景 |
|------|----------|
| `GitWorktree(repo_dir, base_branch)` | 单 git repo 并行开发 |
| `TempDir(base_dir)` | 独立临时目录 |
| `Directory(path)` | Pass-through（不创建不清理） |

`run_parallel()` 默认不清理 workspace（`cleanup_workspace=False`），方便用户检查产出。

## 9. 失败处理

| 情况 | 框架行为 |
|------|----------|
| Agent 返回 `status=failed` | 重跑（restart），最多 `max_retries` 次 |
| API stream error（连接中断） | 恢复（`--resume <session_id>`），最多 `max_resumes` 次 |
| 执行超时 | 杀进程，进入 retry 流程 |
| 输出不可解析 | 检查是否 resumable → resume 或 retry |

Resume 只在 `output_parser` 返回 None（解析失败）**且** backend 检测到可恢复错误时触发。正常完成的 agent 即使输出里提到 "API Error" 也不会误触发。

## 10. Callable 模式

对于复杂 pipeline（循环、条件、串联），用 Python 函数编排多次 `run()` 调用：

```python
def optimize_loop(input_data, gpu_id=None, workspace=None):
    for i in range(20):
        result = run(optimize_agent, input_data, gpu=gpu_id, workspace=workspace)
        save_version(workspace, i, result)         # 保存历史
        update_performance_md(workspace)           # 下一轮 CC 能看到
        if result.get("speedup", 0) >= target:
            break
    return result

results = run_parallel(optimize_loop, "ops.jsonl", gpu=pool)
```

Callable 通过 workspace 文件系统在迭代间传递状态 — 每轮 CC 是新 session，但能读到之前的历史。

## 11. 包结构

```
kgrunner/
├── __init__.py          # 公共 API 导出
├── agent.py             # AgentDef + BackendConfig + WorkspaceConfig + 加载
├── codeagent/
│   ├── base.py          # CodeAgentBackend ABC + default_output_parser
│   └── claude.py        # ClaudeBackend 实现
├── workspace/
│   └── base.py          # Workspace ABC + GitWorktree/TempDir/Directory
├── device.py            # GPUPool
├── platform.py          # PlatformInfo + 11 厂商注册表 + detect_platform
├── run.py               # run() 原语
├── map.py               # run_parallel() 并行调度
├── output.py            # CC 输出解析工具
├── template.py          # {{VAR}} 模板渲染
├── progress.py          # TaskTracker
├── config.py            # 全局配置
└── tests/
    └── test_run.py      # 11 个测试（callable/agent/retry/resume/timeout/parser）
```

## 12. 使用示例

### 场景 1：批量生成算子

```bash
echo '{"OPERATOR": "relu"}' > ops.jsonl
echo '{"OPERATOR": "gelu"}' >> ops.jsonl

python kgrunner/examples/auto_gen/run_example.py --input ops.jsonl --gpu-ids 0 1 2 3
```

### 场景 2：迭代优化

```bash
echo '{"OPERATOR": "softmax"}' > ops.jsonl

python kgrunner/examples/optimize_loop/run_example.py --input ops.jsonl --gpu-ids 4 5 --max-iters 20 --target-speedup 2.0
```

### 场景 3：自定义 workflow

```python
from kgrunner import run, run_parallel, load_agent, GPUPool

analyze = load_agent("agents/analyze")
code = load_agent("agents/code")

def pipeline(input_data, gpu_id=None, workspace=None):
    analysis = run(analyze, input_data, gpu=gpu_id, workspace=workspace)
    result = run(code, {**input_data, **analysis}, gpu=gpu_id, workspace=workspace)
    return result

results = run_parallel(pipeline, "tasks.jsonl", gpu=GPUPool())
```

## 13. 设计原则

1. **Agent 自包含** — 一个目录 = agent.yaml + config.yaml + prompt，框架自动解析
2. **框架管资源** — GPU 池化、workspace 分配、设备环境变量，用户不操心
3. **两层解析** — backend 管传输格式，用户管业务格式
4. **Python 即编排语言** — 循环/条件/fan-out 用 Python callable，不发明 DSL
5. **默认保留产出** — workspace 不自动清理，方便检查
6. **Backend 可插拔** — Claude/Codex/其他，实现 ABC 即可接入

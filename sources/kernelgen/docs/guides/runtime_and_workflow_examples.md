# Runtime 配置与进阶 Workflow 示例

本文保存首次 Example 之外的模型配置、参考代码与 Knowledge 用法。第一次安装和运行请从 [ONBOARDING](../ONBOARDING.md) 开始；任务管理和 Batch YAML 见 [KG CLI 文档](../design/kg_cli.md)。以下命令在 KG 仓库根目录执行，先完成上手指南中的安装和 Server 启动；示例中的环境变量须按实际实例设置，不读取或打印凭据。

## 1. 模型 Runtime 与 Catalog 准备

运行 SimpleOpt、BatchSimpleOpt 或 KernelGen campaign 需要 Claude Code CLI 或 Codex CLI；入口通过 `--runtime claude|codex` 选择，默认仍为 `claude`。Claude Code 通过仓库脚本在目标机器安装对应 CPU 架构的版本，不从其他机器复制二进制：

```bash
export DEPLOY_BASE=/path/to/deploy
bash tests/install_claude_cli.sh \
  --base "$DEPLOY_BASE" \
  --version 2.1.220 \
  --node-version 22.23.1

export PATH="$DEPLOY_BASE/runtime/node-bin:$DEPLOY_BASE/runtime/claude/node_modules/.bin:$PATH"
claude --version
source env.sh

# env.sh 使用 KernelGen 配置名；启动 Claude Code 前转换为其认证变量。
export ANTHROPIC_API_KEY="$ANTHROPIC_AUTH_TOKEN"
unset ANTHROPIC_AUTH_TOKEN
export ANTHROPIC_MODEL="$MODEL"
```

`env.sh` 中的 endpoint 和 token 不得打印、提交或写入实验报告。不要同时保留 `ANTHROPIC_AUTH_TOKEN` 和 `ANTHROPIC_API_KEY`；zyapi 使用后者。

使用 Codex 时，先按 [Codex CLI 官方文档](https://developers.openai.com/codex/cli) 安装并验证 `codex --version`。交互环境可运行 `codex login`，自动化环境可设置 `CODEX_API_KEY`；自定义兼容 endpoint 使用 `OPENAI_BASE_URL`，模型可用 `CODEX_MODEL` 指定。KernelGen 不修改用户的 `CODEX_HOME`，workspace 内的 `.kernelgen/mcp.json` 会在启动时映射为本次 `codex exec` 的临时 `mcp_servers.*` 配置。

```bash
export KERNELGEN_RUNTIME=codex
codex --version
```

Codex runtime 使用 JSONL 事件流、workspace-write sandbox、`approval_policy=never` 和持久化 session resume；人类可读日志写入各 agent workspace 的 `.kernelgen/codex-runtime.log`。Agent 静态角色单一维护在 `.kernelgen/agents/*.md`，通用 Agent Skills 单一维护在 `.kernelgen/skills/<name>/`，MCP 启动定义单一维护在 `.kernelgen/mcp.json`。Workspace 初始化时会把 KernelGen MCP 绑定到当前 `kg` 进程的 Python 解释器，并自动生成 Claude `.claude/agents/*.md`、Claude `.claude/skills/<name>/`、Claude `.mcp.json`、Codex `.codex/agents/*.toml` 和 Codex `.agents/skills/<name>/`；因此可以直接调用虚拟环境内 `kg` 的绝对路径，无需依赖 `PATH` 选择 MCP 的 Python。不要手工修改或提交这些运行时生成物。Codex 的 `.agents/skills` 路径遵循 [OpenAI 官方 Skills 约定](https://developers.openai.com/codex/skills)。

通过 `--catalog-name` 选择 KGS 已安装的 Catalog，或通过 `--catalog-path` 提供本地 Native Catalog，由 KG 内部冻结并上传，两者互斥。先读取目标 Catalog 的 `manifest.json`：`api_version=v6.0` 默认走 FlagGems adapter/flat layout，`api_version=v6.2` 默认走 native/per-operator layout。配套 KGS release 和 exact commit 只由当前 KG 的 `deployment/kgs.lock.yaml` 决定，KGS `compatibility.yaml` 用于核对自身协议和框架描述，不使用历史反向推荐矩阵选择版本。

验证 KernelGenBench native Catalog 时使用：

```bash
export KERNELGEN_CATALOG_NAME=kernelgenbench
```

验证原生 FlagGems Catalog 时改为：

```bash
export KERNELGEN_CATALOG_NAME=flaggems-native
```

验证 FlagGems adapter Catalog 时改为：

```bash
export KERNELGEN_CATALOG_NAME=flaggems-adapter-definitions
```

KernelGen 只通过 Server 的 `/status.api_version` 和 Catalog 接口读取运行时契约，不依赖 Server 仓库中的本地数据路径。运行时只按 `/status.api_version` 判断 Protocol 兼容，按 capabilities 判断功能，并在 `evaluation_adapters` 中确认本轮需要的 evaluator；软件 release 不替代这些检查。`v6.0` adapter Catalog 不要求旧版 Server，也不会把 wire protocol 降为 v6.0。

## 2. 提供只读参考代码

如果有可借鉴的 Triton、CUDA、Ascend C 或其他语言实现，可以在 `kg run --mode simple_opt ...` 或 `kg run --mode kernelgen ...` 中增加以下参数；对应 Python launcher 使用完全相同的参数：

```bash
--reference-code-path /path/to/reference.cu \
--reference-code-prompt-path /path/to/reference_context.md
```

第一个文件提供只读参考代码；第二个文件说明代码在哪款芯片和哪套 evaluator 上验证、哪些设计可以迁移，以及当前任务需要重新验证的 ABI、workload、API 和性能假设。说明文件不能脱离参考代码单独传入。旧 `--reference-triton-path` / `--reference-triton-prompt-path` 名称保留为输入别名。

Definition、workload、reference 语义和 validator 始终是权威契约。参考代码不会成为 初始候选或 timing baseline；历史加速比也不能与当前 evaluator 的结果直接比较。 Coder 只能复用算法、分块和调优思路，不能修改 Torch/FlagGems API、pytest、 dispatcher、reference、validator 或 timing baseline 来制造通过，所有改写仍须通过 当前远端 preflight 和 eval。

参考代码允许是上一阶段未通过编译、正确性或性能未达标的实现，说明文件可记录其失败信息；输入仅作文本读取，不导入或执行，不因提供代码就认定它通过验证。KernelGen 会将它传给每个 epoch 的所有 Coder，并与从 ledger 选出的 epoch seed 分开。`--seed-code-path`（别名 `--seed-triton-path`）为两种模式提供未预验证的 Triton 初始候选上下文，不继承历史成绩或触发独立 baseline 测量；实际候选仍通过原 Preflight/Eval。reference 与 seed 可同时提供。

直接调用底层 Python `KernelGenWorkflow.run()` 时，输入字段为 `reference_code_source`（源码文本）和可选 `reference_code_prompt`（说明文本），而不是路径；统一 CatalogOptimize 入口负责文件的有界读取和冻结。YAML Batch 两种模式均可设置 `reference_code_path`、`reference_code_prompt_path` 和 `seed_code_path`，相对路径以 YAML 所在目录为准。续跑沿用原输入，不把只读参考写入 ledger 作为已测结果。

新 Batch 使用 `kg run --batch-file`，在 YAML 的每个 operator 中显式填写 `reference_code_path` 和可选 `reference_code_prompt_path`，不另建 Python Batch 执行流程。仅历史 campaign 使用 `python3 -m kernelgen.cli.legacy_batch_simple_opt`，其重复 reference 参数和 `tests/run_batch_simple_opt_chip_e2e.sh` 的目录扫描保留旧行为；该脚本不是新实验推荐入口。

## 3. 让 SimpleOpt 查询现有 Knowledge Catalog

SimpleOpt 默认不使用 Knowledge。需要让单算子 Coder 查询一个现有 V1 Knowledge Catalog 时，先验证 Catalog：

```bash
export KNOWLEDGE_CATALOG_PATH=/path/to/knowledge-catalog
python3 -m kernelgen.tools.kb validate --kb "$KNOWLEDGE_CATALOG_PATH"
```

验证结果的 `valid` 必须为 `true`。设置 `KERNELGEN_SERVER_URL`、`KERNELGEN_CATALOG_NAME` 和与 `/status.target.device` 一致的 `TARGET_HARDWARE`，随后使用全新的 workspace 启动：

```bash
cd /path/to/kernelgen

definition=gcd
workspace="$PWD/runs/simple_opt_kb/${definition}_$(date +%Y%m%d_%H%M%S)"

kg run --mode simple_opt --foreground \
  --definition "$definition" \
  --catalog-name "$KERNELGEN_CATALOG_NAME" \
  --workspace "$workspace" \
  --eval-server "$KERNELGEN_SERVER_URL" \
  --target-hardware "$TARGET_HARDWARE" \
  --knowledge-catalog-path "$KNOWLEDGE_CATALOG_PATH" \
  --model "$MODEL" \
  --max-round 15
```

运行结束后检查 Knowledge 上下文、检索记录和 ledger：

```bash
test -f "$workspace/stages/optimize/work/.kernelgen/knowledge/state.json"
test -f "$workspace/stages/optimize/work/.kernelgen/knowledge/retrieval-log.jsonl"

jq '{status, best_geo_mean, rounds}' \
  "$workspace/stages/optimize/work/optimize_definition_output.json"
jq '[.rounds[].plan.knowledge_uses[]?] | length' \
  "$workspace/stages/optimize/work/.ledger.json"
```

提供该参数会在优化 workspace（`stages/optimize/work/`）写入 Knowledge 查询上下文，选择 Knowledge Coder/Profile/Distiller，并在 Distiller 成功时将 CandidateDraft 保存在其中的 `.kernelgen/knowledge/candidates.jsonl`。检索记录存在不表示知识实际影响了优化；只有进入 ExperimentPlan 的条目才计入 ledger 的 `knowledge_uses`，所以该值允许为 `0`。SimpleOpt 不修改公共 Catalog，也不执行 KernelGen 的 epoch publication、Solution promotion、fork 或多 Agent synthesis；这些完整流程见 [V1 KB 快速使用](KB_QUICKSTART.md)。不提供该参数时，默认不使用 Knowledge。

同一 SimpleOpt workspace 不能同时启动两个 Coder。上述前台运行返回 0 表示得到 correctness 和 timing 都完整的 best kernel，不表示它一定快于 reference；后台运行应另行检查 `kg status` 和 ledger。

运行多个算子、调整 workers、监控 15 分钟长任务和回收结果时，不在这里拼接命令， 直接执行 [多设备实验与 E2E 验收手册](../operations/multiple_device_experiment_runbook.md)。

## 4. 其他 Workflow 的 Python 入口

普通算子生成和优化优先使用 `kg run --mode simple_opt|kernelgen` 或 `kg run --batch-file`。以下工作流超出这些命令的范围，从仓库根目录用 Python 启动，先通过 `--help` 查看各自参数：

| 目标 | 入口 |
|---|---|
| 从 PyTorch/FlagGems 抽取并优化 | `python3 examples/extract_opt/run_example.py --help` |
| 多算子、共享知识库的 epoch campaign | `python3 examples/kernel_gen/run_campaign.py --help` |
| 将通过的 BatchSimpleOpt 结果提交上游 | `python3 examples/batch_pr/run_example.py --help` |

抽取输入契约见 [FlagGems 抽取指南](flaggems_definition_workload_guide.md)，Knowledge campaign 见 [V1 KB 快速使用](KB_QUICKSTART.md)。Python BatchSimpleOpt 的最终汇总文件不等同于 KG YAML Batch 子任务索引；任务结果与归档口径见 [实验结果整理手册](../operations/experiments/experiment_result_maintenance.md)。

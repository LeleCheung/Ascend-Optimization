# FlagGems PR 抽取入口

支持输入公开的 `https://github.com/flagos-ai/FlagGems/pull/<number>` 和 `https://github.com/flagos-ai/FlagGems-vllm/pull/<number>`，固定源码后复用同一个 Native v6.2 语义抽取器。当前仅支持这两个仓库的 PR（包括来自 fork 的 PR），不扩展为任意 GitHub 仓库、Gitee PR 或自动多算子发现。

`source_profile.py` 集中维护 repository、Python package 和 framework 名称：FlagGems 使用 `flag_gems`，FlagGems-vllm 使用 `flaggems_vllm`。源码根目录和 Git URL 由该配置派生。普通 checkout 按唯一存在的支持包目录选择，目录缺失或同时存在两个包时拒绝猜测；PR 还校验包布局与冻结的 repository 一致。两者共用源码发现、公开 ABI、Agent、oracle 校验和 Catalog 落盘，不复制 Agent。

## 调用与职责

`CatalogExtractWorkflow` 的源码输入二选一：`flaggems_repo` 或 `pr_url`，另需明确 `operator`。用户无需准备或传入 `case_list_path`：Workflow 内部生成临时采集 pytest 并执行，再把清单交给语义抽取 Agent。已有清单仍可通过可选的 `case_list_path` 复用。输出保留 `extraction/accuracy_coverage`，统一返回已落盘的 `catalog_path`、`operator_dir`、`source_evidence` 和实际使用的 `case_list_path`；持久化 Workflow 名称保持不变。

`pr_source.prepare_pull_request()` 由 Python 调用 GitHub API，先记录 PR repository、number、base/head 和 merge-base 的精确 commit，再在独立的 `source/head`、`source/base` 准备只读语义的 checkout。不会 merge PR、修改已有用户仓库、执行安装脚本、初始化 submodule 或运行 PR pytest。Git hooks 被禁用，拒绝 submodule，以及绝对、越界、悬空、循环或不指向受 Git 跟踪普通文件的 symlink；允许仓库内部文档的相对链接。失败保留现场；重试读取已有 `source/source.json`，不重新解析变化后的 PR。源码 HEAD、origin、tracked/untracked/ignored 状态不符则拒绝复用。

`FlagGemsPRExtractorAgent` 只增加固定 PR 上下文，复用 `FlagGemsV62ExtractorAgent` 的角色、输出 Schema、校验和有限修复循环。完整 head 源码是抽取输入；base 用于理解改动，diff 路径不是完整源码的替代品。新增单个 `timing_reference=flaggems` 策略在 prompt 和持久化校验两处使用同一个输入值，普通 checkout 默认 `source` 不变。

当前 PR 性能基线固定为 **PR head 的 Gems 实现**，用于继续优化 PR 内的算子；不是 PR base，不引用 PR 描述中的性能数字，也不沿用 benchmark 中可能存在的 vLLM 对照。正确性 reference 仍按源码 pytest 独立抽取。切换为 base 基线涉及 ABI 和运行时版本绑定，需要另行确认，不能仅替换一行 revision。

## #5623 示例

重发的 v6.4.0 已包含 `kg extract`（前台），推荐使用：

```bash
kg extract --pr-url https://github.com/flagos-ai/FlagGems/pull/5623 \
  --operator fused_experts_impl
```

本地 checkout 则用 `kg extract --flaggems-repo /path/to/FlagGems --operator relu`。可添加 `--runtime codex --model <model>`；不提供 case list 参数。默认生成新的 `.kernelgen/extracts/<id>`，指定 `--workspace` 时该目录必须尚不存在且位于源码 checkout 外。审核失败或取消后保留证据，下一次使用新 workspace，不承诺抽取内部 session 续跑。成功输出的 Catalog 路径可供多个目标分别运行 `kg run`。共享执行边界为 `cli.api.extract_catalog()`，不启动 KGS、申请 Coder lease 或提交后台优化任务。

以下 Python 入口仍保留，其源码准备和 Workflow 行为不变；最初指向 `0082e613` 的 v6.4.0 不包含新 CLI，需按发布说明刷新 tag：

只准备源码，不调用模型或设备：

```bash
python3 -m kernelgen.examples.flaggems_pr_extract.run_example prepare \
  --pr-url https://github.com/flagos-ai/FlagGems/pull/5623 \
  --workspace ./pr5623
```

自动准备 timing case list 并生成独立 Catalog：

```bash
python3 -m kernelgen.examples.flaggems_pr_extract.run_example extract \
  --pr-url https://github.com/flagos-ai/FlagGems/pull/5623 \
  --operator fused_experts_impl \
  --workspace ./pr5623
```

示例默认使用已经认证的 Claude Runtime，`--model` 和 `--timeout` 可选，不安装模型工具或依赖。Python 调用方可自行提供现有 Runtime factory：

```python
from kernelgen.workflows.catalog_extract import CatalogExtractWorkflow

output = CatalogExtractWorkflow(cwd="./pr5623", runtime_factory=runtime_factory).run({
    "pr_url": "https://github.com/flagos-ai/FlagGems/pull/5623",
    "operator": "fused_experts_impl",
})
```

Workflow 把 case list 内容摘要和选中算子写入 `catalog-extract-input.json`；模型返回后再次校验 case list，PR 输入另校验源码。Workflow 沿用现有持久化函数生成唯一的 `catalog/`，写入 `accuracy_coverage.json`、`extraction.json`，并负责 manifest 的 framework、repository、PR ref 和 exact head commit 绑定。`source_evidence` 包含源码发现结果、实际清单和 PR `source.json`。launcher 只调用 Workflow 并打印路径，不再重复准备源码、转换输入、保存 Catalog 或维护另一把抽取锁。成功输出只表示 Catalog 抽取与静态校验完成，`target_validation=NOT_RUN`；直接调用 Workflow 和 launcher 都会拒绝覆盖已有 Catalog 或抽取结果，失败现场不自动清理。

## 没有 list-case 时

`CaseListAgent` 读取原始 pytest、benchmark 及其 helper，优先包装已有导出接口；没有接口时生成 `collect_cases(source_root, operator)`，接入原 benchmark 的真实参数展开与 input iterator，拦截计时而非执行评测。Python 将其写入独立 `case_collection/01/collector.py`，通过固定的 `test_collect_cases.py` 执行。Agent 不手写结果清单，不修改冻结源码。下一次修复使用 `02/`，每次调用最多尝试两次，之前的源码和日志保留。

Python 保存 `pytest.log`、`cases.json` 和 `executed-source.json`，要求有原 benchmark Python 函数实际执行、结果非空且符合现有 timing case-list Schema。skip、缺依赖、超时、静态结果或 Schema 错误均不进入抽取；失败路径交给 Agent 作有限修复，不用 fake torch/Gems、猜测的 capability 或 correctness 清单替代真实 timing 清单。成功 receipt 绑定清单摘要，重试先检查源码/config 内容摘要和清单摘要，不静默使用过期数据。

临时 pytest 通过下文的 SSH + Docker 执行器运行，默认执行超时 300 秒，执行 workspace 必须位于 source checkout 之外。子进程不继承模型凭据或 pytest 自动加载插件。它是用户授权的源码执行，不是恶意 Python 的安全沙箱；执行轨迹也不等于数学证明全部用例覆盖。执行环境必须具备原 builder 所需依赖；环境准备不自动执行，失败时保留日志，不安装或升级 Torch/Triton/厂商运行时。采集结果不证明目标能力或 correctness/性能通过。

自动采集接入后的 host 回归为 **316 passed**，在下文原有测试集上加入 `test_case_collection` 并扩展不传 case-list 的 PR Workflow/launcher 测试。临时 pytest 子进程真实执行，模型提案与原 benchmark builder 使用 fixture；覆盖成功、有限修复、skip/空清单/重复 ID/错误 phase 拒绝、源码与清单变更拒绝、超时清理和凭据环境隔离。测试前已核对导入当前 feature worktree，没有安装依赖。

## 统一 SSH + Docker 采集执行器

KG 和模型 Agent 留在发起任务的机器，临时 pytest 统一通过 SSH 进入指定采集节点，再由 `docker exec` 使用指定容器解释器执行。不要求 CPU 主机安装 Gems 或 GPU 环境，也不要求容器安装模型 CLI。KG 即使运行在 A100 采集节点自身，也使用同一个 SSH Host alias，不切换成本地执行路径；因此 CPU 与 A100 部署可以使用相同配置。

配置保存在启动目录的 `.kernelgen/config.json`，沿用 `KERNELGEN_CLI_HOME` 覆盖和现有 `config.lock`，不放入 KGS 实例配置，也不增加每次抽取必填参数。使用同一目录启动后续 Python launcher；从其他目录启动时可显式设置相同的 `KERNELGEN_CLI_HOME`。先配置一次：

```bash
kg config set extract.host a100-extract
kg config set extract.container kernelgen-nvidia-cu128
kg config set extract.python /data/jiabei/venvs/kernelgen-cu132-torch211-triton360/bin/python3
```

`host` 是指向固定采集节点的 SSH 配置别名，不是 `localhost`；各发起机器的 SSH 配置需将它解析到同一节点。身份、私钥与跳板设置由 SSH 客户端配置维护，不写入 KG 配置。容器名和解释器路径是当前环境示例，不由容器名字推断 Torch/CUDA 版本。只通过 SSH 执行命令，不新增常驻 daemon、Docker API 暴露或网络监听端口。

Workflow 继续负责冻结输入、调用 CaseListAgent、有限修复和清单校验；采集执行器替换原本地 pytest 子进程执行边界，负责以下事项：

- 将冻结的源码快照、生成的 collector 和执行请求准备到远端独立任务目录，显式映射本地、远端宿主机与容器路径，不假定共享文件系统，不覆盖已有源码或任务产物。
- 使用指定解释器执行固定 pytest wrapper，保留超时、取消、退出状态和凭据隔离语义；每次采集显式导入对应快照，校验源码 revision 与内容完整性，不静默使用容器里另一版本的已安装 Gems。
- 将 case list、日志和执行证据回收到原 workspace，再由现有校验路径接受结果；receipt 记录实际采集节点、容器、解释器与源码身份。连接中断或产物回收不完整不算成功，保留现场供重试，不引入另一套业务状态真源。

缺少 SSH、Docker、容器、解释器或 builder 依赖时明确失败，不回退到发起机器 Python，不自动安装或升级 Torch、Triton 和厂商运行时。配置与传输失败不调用模型修复；实际 pytest 失败仍走原有限修复路径。已有有效 receipt 或显式传入的 case list 可以直接复用，无需重新连接采集节点。

`collection_executor.py` 通过 SSH stdio 向容器发送 stdlib-only 的 `collection_worker.py` 和源码快照，不要求远端安装 KG/KGS。容器内新建 `kg-case-collection-*` 临时目录，保留源码和失败现场；本地每个 attempt 保存 `execution.json`、`transport.jsonl`、`transport.log` 和回收的 pytest 产物。`execution.json` 记录请求的节点/容器、实际解释器、远端目录、源码完整摘要和可用的 Git revision，属于执行证据，不是第二份 Workflow 状态。源码不携带 `.git`、缓存或环境文件，拒绝越界链接，snapshot 上限为 128 MiB；Git checkout 必须 clean。远端按传输内容摘要验证安装后的快照，并在执行结束后再次验证；通过 `PYTHONPATH` 优先导入该任务快照的 `src` 和根目录。

模型输出完成后才进入采集执行阶段的 cancellation checkpoint，不改变 Runtime 的模型安全点。远端 supervisor 监测 SSH stdin EOF，并独立执行 300 秒默认 deadline，终止该次 pytest 进程组及其子进程，不停止容器或其他任务。连接中断不能视为即时远端取消确认：网络分区时以远端 deadline 兜底，现场路径优先从已收到的 `ready` 记录定位。SSH 断开或回收不完整不会生成成功 receipt。

实现后的相关 host 回归为 **372 passed**，覆盖抽取、CLI、worker lease 与 run-control；采集测试通过模拟 SSH 命令入口运行真实独立 worker 和 pytest，检查传输、路径映射、快照导入、环境隔离、超时与 EOF 取消。另在本地 `kernelgen-nvidia-cu128` 和上文解释器中完成 **2 passed** 的真实 Docker 集成测试，验证结果回收，以及关闭 Docker stdin 连接后远端 pytest 的终止证据；远端独立目录保留 `result.json`，供断连后核对退出原因。测试没有调用模型、执行 GPU 计算或安装依赖。

2026-09-12 补测 A100 发起端 → `ssh a100-extract` → 本机 `kernelgen-nvidia-cu128` 的完整真实链路，**3 passed in 9.41s**：结果回收、执行超时、取消后容器内 pytest 进程组退出。SSH alias 指向固定 A100 地址 `10.1.2.106`，主机返回 `bm-baai-dx-zone1-d-a100-40g-2-106`；未 mock SSH 命令，取消后的退出证据也通过 SSH 回读。测试在独立临时 CLI home 中通过 `kg config set extract.*` 写入并读取配置，不覆盖日常 `.kernelgen/config.json`。专用登录 key 和主机公钥校验留在机器的 SSH 配置中，不入库；本次 key 限制为本机来源，CPU 发起端需另外配置授权身份，不能直接复用该 key 的来源限制。

尚未从另一台 CPU 主机发起，也未在本次链路测试运行真实模型或 Gems 算子采集；这些测试只验收执行器。可用 `KG_COLLECTION_TEST_CONTAINER` 和 `KG_COLLECTION_TEST_PYTHON` 显式启用 `tests/test_collection_docker.py`，增加 `KG_COLLECTION_TEST_SSH_HOST=a100-extract` 即通过完整 SSH 链路运行；不设置 SSH 变量时只验证 Docker，默认 host 测试跳过真实容器执行。旧本地采集 receipt 没有完整快照与 revision 绑定，不自动迁移；保留旧证据，新采集使用新的 workspace。

“中立采集节点”只表示统一的源码准备与采集环境，不保证硬件无关的完整枚举。原 pytest 的 NVIDIA 条件、dtype 判断或 skip 仍可能影响结果，Ascend 专用依赖也可能无法在 A100 环境运行；必须保留这些条件和执行证据，不能移除 skip 或把采集结果视为所有目标设备的完整覆盖。各目标的 reference/dtype 验证及优化 Eval、Benchmark、Profile 仍交给对应 KGS，沿用 loopback 与 SSH stdio HTTP proxy 边界。此执行器不承担 KGS 部署，也不改变 Server 源码经 Gitee 同步的规则。

## 验证事实与未完成部分

2026-09-11 对 [FlagGems #5623](https://github.com/flagos-ai/FlagGems/pull/5623) 实际完成 GitHub 解析、两个干净 checkout 和静态源码发现：head 为 `60d8cc5fcb7d435e9118d591578e52cee2d6e7ad`，base/merge-base 为 `01433e8304d78ef6c16ce76fe68e51e16c0b4d66`，diff 共 15 个文件。选中实现为 `src/flag_gems/fused/fused_moe.py::fused_experts_impl`，发现两个 correctness 源文件和六个 benchmark 源文件，成功解析公开 ABI。此记录不代表 PR 当前 HEAD 永远不变。

该 checkout 的 benchmark 没有现成的 `--list-cases` / `flaggems.benchmark-case-list/v2` 导出能力，首次源码准备验证没有真实 case list，也没有调用模型生成该算子的 Catalog。后续已加入上文的 Agent 临时 pytest 采集路径，但尚未在 #5623 完成真实验收；硬件中立枚举、动态 pytest 参数与条件性 skip 的完整一致性验证仍待完善。不能从 PR 描述的少数性能案例拼造清单；源码摘要和执行文件轨迹也不能单独证明完整覆盖。

Host 验证：在已有 `kernelgen-nvidia-cu128` 容器、既有 `/data/jiabei/venvs/kernelgen-cu132-torch211-triton360/bin/python3` 中执行 220 项测试通过，覆盖 `test_flaggems_pr_extract`、`test_flaggems_v62_direct_agent`、`test_flaggems_v62_batch_extract`、`test_extract_and_optimize`、`test_agent_roles`、`test_workflow_names`、`test_operator_development` 和 `test_cli_lifecycle`。测试前核对 KG 来自本 feature worktree，KGS 来自主仓库 checkout；没有安装依赖。测试中的模型输出和 launcher 持久化使用 fixture，不等于 #5623 的真实模型抽取。真实 PR 源码的重复准备验证沿用原 commit，不进行新的网络解析。

抽取输出仍需 Catalog Review、目标 reference/dtype 验证和候选评测。本分支不修改 KGS、不安装 PR 版 Gems、不运行 GPU/NPU。当前 KGS Operator Bundle 不携带 Catalog 顶层 framework 绑定信息，所以本例 manifest 的 exact revision 也不能被误认为 Bundle 上传后已经强制生效；进入优化前需要保证运行时版本绑定，不能让另一版 Gems 静默充当此 baseline。

本功能最初基于 KG `origin/dev@fe8610f5` 开发，现已同步 !107 的 `origin/dev@84e6b0d3`，整合 `CatalogExtractWorkflow` 的 PR source dispatch 与已准备 Catalog 输出。FlagGems 抽取产物可通过 `OperatorOptimizeWorkflow` 的 `catalog_path` 输入继续准备优化，不需要重新持久化。当前不再提供组合 Workflow；PR URL 只属于独立抽取入口，优化通过 `kg run --catalog-path` 消费已发布产物。

衔接测试使用本地 Git fixture、固定模型回复、真实 Agent 校验与 Catalog 持久化，验证 FlagGems 产物可被 `OperatorOptimizeWorkflow.prepare_input()` 读取并制作相同摘要的快照，原 Catalog 不变。FlagGems-vllm 保留真实 `framework=flaggems_vllm` 标记；当前配套 KGS Native Catalog 读取器只接受未绑定或 `flaggems`，因此该产物目前不能直接进入优化，测试明确验证此拒绝行为。后续需独立扩展 KGS framework 绑定和运行时支持，不删除来源字段或改名为 FlagGems 来绕过限制。

2026-09-12 同步 !107 后回归 **384 passed in 40.73s**，覆盖 Catalog 组合与快照、两种 Gems PR 来源、自动采集、CLI、生命周期、worker lease 和 run-control。真实 A100 SSH → 本机 Docker 复验 **3 passed in 9.53s**。复验中发现并修复连接已结束后 stdin 关闭刷新缓冲触发 `BrokenPipeError` 的竞态，仅忽略该关闭错误，仍按完整返回记录判定执行是否成功；新增确定性回归。此轮没有修改 KGS、依赖或部署版本，也没有宣称真实 PR 模型抽取和目标优化 E2E 已通过。

## FlagGems-vllm 验证与边界

#731 `add_rms_norm` 和 #728 `swiglu` 使用同一个 launcher，只需替换 `--pr-url` 与 `--operator`。测试覆盖两种来源的 URL 解析、同编号不同仓库拒绝复用、源码布局、公共 ABI、Agent prompt、oracle 基线包名检查、落盘校验、Workflow 和 launcher manifest。现有 case-list Schema 保持 `flaggems.benchmark-case-list/v2`，不按仓库另造一套协议。

2026-09-11 使用真实 GitHub API 和 launcher `prepare` 验证两个 PR 均成功：#731 head `fab1c83d2bcf5a349ce105276e6b07bba6e09224`，#728 head `cea7824a84a19425c8a37126bbc29f18f89392ad`。两者 base 为 `5865ed53b183e6d3aff69ad1823622a11679d257`，merge-base 为 `1abdbac04592c679a4220094f8a69592a19658be`。源码 discovery 和 ABI 解析成功：分别为 `src/flaggems_vllm/ops/add_rms_norm.py` 和 `src/flaggems_vllm/ops/swiglu.py`；这是公共导出 ABI，不意味着使用本机推断了 Ascend dispatch。源码及 receipt 保存在本地实验目录 `pr-extraction-tests.rhWeCP/entrypoint731` 和 `entrypoint728`。

本次 host 回归 **301 passed**：`test_gems_source_profiles`、`test_flaggems_pr_extract`、`test_flaggems_extractor`、`test_flaggems_v62_direct_agent`、`test_flaggems_v62_batch_extract`、`test_extract_and_optimize`、`test_agent_roles`、`test_workflow_names`、`test_operator_development` 和 `test_cli_lifecycle`。使用 host `python3`、`PYTHONDONTWRITEBYTECODE=1` 和显式 `PYTHONPATH`，先核对 KG 导入来自 `kg-flaggems-pr-extraction/kernelgen`、KGS 导入来自配套主 checkout；没有安装依赖。模型输出和持久化验证仍使用 fixture，本次未启动真实模型、KGS 或设备任务。

这两个 PR 的真实 timing collection、模型 Catalog 抽取和昇腾 reference 验证尚未验收通过。此前 Agent 生成的临时静态脚本只枚举 correctness shape/dtype，并显式标注 `STATIC_DECLARED_NOT_PYTEST_COLLECTED`，不能用作 timing 输入。新增自动采集路径已用真实 pytest 子进程和 fixture 原始 builder 验证，模型提案使用 fixture；不把 host 接线测试称为这两个 PR 的 E2E。FlagGems-vllm 的运行时安装与精确版本绑定也不能假定由现有 KGS FlagGems 安装命令自动支持。

相关设计：[Catalog 抽取与优化](catalog_optimize.md)、[Workflow 命名](workflow_names.md)。

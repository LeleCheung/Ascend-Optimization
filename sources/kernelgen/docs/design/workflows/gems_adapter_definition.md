# Pytest 到 Gems adapter Definition

`GemsAdapterDefinitionWorkflow` 从已提交且干净的 Gems checkout 读取原始正确性 pytest、benchmark 和公开算子签名，输出一个本地 v6.0 Gems adapter Catalog。它不生成 pytest，不调用模型，不在 Agent 主机执行 Torch/Gems，也不抽取 Native oracle 或 workload。

## 输入与输出

输入是 `flaggems_repo`、`pytest_path` 和可选的 `operator`。pytest 路径可以相对仓库，也可以是仓库内绝对路径；默认按 `test_<operator>.py` 识别算子，聚合测试文件需要明确指定 operator。选中的 pytest 必须属于该算子的原始正确性 suite，同时必须存在 benchmark suite。没有确定公开 ABI 时明确失败，不从单个输入样例猜签名，也不把可变 kwargs 强行改成固定参数。

输出包含 `operator`、`catalog_path`、`definition_path`、`definition_sha256`、`source_revision` 和 `source_files`。Catalog 只有 Definition 和来源 manifest，不复制 Gems pytest 作为另一份测试真源。输出目录必须在 Gems checkout 外；相同来源可重复读取，输入改变必须使用新 workspace，不能改写旧实验。

## 连续调用

推荐使用 CLI，在 Gems checkout 外执行：

```bash
kg definition --flaggems-repo /path/to/FlagGems \
  --pytest-path tests/test_negative.py --workspace runs/negative/definition

kg run --definition negative --catalog-path runs/negative/definition/catalog
```

`--workspace` 可省略，默认在当前 CLI home 的 `definitions/<id>` 下创建输出（通常为 `.kernelgen/definitions/<id>`）；聚合测试文件用 `--operator` 指明算子。成功在 stdout 输出 JSON，包含 `workspace`、`operator`、`catalog_path`、`definition_path`、`definition_sha256`、`source_revision` 和 `source_files`；将其中的 `catalog_path` 交给 `kg run`。指定输出目录时，相同来源可重复导出，不同来源拒绝覆盖。该命令前台完成，不创建后台 run、不占 Coder lease、不接入 `kg status/resume`。成功退出 0，输入或执行错误退出 2，Ctrl+C 退出 130。不调用模型、执行 pytest 或连接 KGS，导出成功不代表测试通过；用 `kg server status <name> --json` 中的 `config.flaggems_commit` 核对目标 Gems 与 `source_revision` 一致。

Python 集成仍可直接调用同一个 Workflow：

```python
from kernelgen.workflows.gems_adapter_definition import GemsAdapterDefinitionWorkflow

prepared = GemsAdapterDefinitionWorkflow(cwd="runs/negative/definition").run({
    "flaggems_repo": "/path/to/FlagGems",
    "pytest_path": "tests/test_negative.py",
})
print(prepared.catalog_path)
```

将打印的 Catalog 路径交给统一 CLI，不再使用已删除的 Catalog example launcher：

```bash
kg run --definition negative --catalog-path <catalog_path> \
  --eval-server http://127.0.0.1:19808 --workspace runs/negative/optimize
```

`kg run` 负责模型 Runtime 配置，并直接调用 `OperatorOptimizeWorkflow`。默认 kernelgen、1 Coder、1 epoch、每 Coder 最多 10 rounds；显式参数仍可以覆盖，不需要用户操作 Bundle ID。`kg definition` 通过 `cli.api.export_gems_definition()` 直接复用 Definition Workflow；`kg extract` 仍专用于 Native Catalog 抽取，两者不混用。

## 服务端执行与审核边界

OperatorOptimize 将所选 Definition 和来源元数据打包到现有 Bundle 存储，通过 Bundle binding 导出并冻结 KGS 契约。KGS 必须声明 `operator_bundle_upload.evaluation_binding=true` 且 `evaluators` 包含 `flaggems`；否则提交优化前明确失败。Bundle 不复制 pytest 源码；KGS 从其匹配的 Gems checkout 加载原始 correctness/benchmark suite，并核对源文件 SHA 与 commit，避免优化误用同名内置 Definition。

来源核对只属于 prepare。新的 Gems 输入默认在 `review_tests` 审核目标原始 pytest/benchmark，随后通过 KGS 执行 core baseline `--reference-only`；只有显式 `skip_review=true` 才跳过模型审核，目标验证仍会执行，见 [统一测试契约审核](review_tests.md)。候选代码的 Preflight、正确性、Benchmark、可选 Profile、最终复验和 advisory code review 仍正常执行。性能事实来自 ledger，不从 Definition 导出或源码审核成功推断优化通过。

## 版本选择

默认来源是官方 `flagos-ai/FlagGems` 的 `kernelgen-dev`。配套 KGS `compatibility.yaml` 声明 branch policy，KG 在新实验的服务准备阶段解析并记录实际 commit，再使用同一 commit 导出 Definition。配置为默认分支的实例在后续 start 时检查最新版本，但不替换运行中的服务；显式完整 commit 和续跑保持原快照。分支定期合并 master 不触发运行中 campaign 自动升级。详见 [Gems 来源策略](../gems_source_policy.md)。

来源一致性约束针对本次导出的 ABI/pytest 输入，不是对所有 Gems adapter Catalog 重新施加统一 Gems 版本限制。已安装 Catalog 的原有输入路径保持不变。KG/KGS release、Protocol、Gems revision 仍独立；此次接口只按服务端 capability 判定可用性。

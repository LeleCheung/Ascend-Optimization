# Knowledge 模块

这个目录负责 KernelGen 的知识检索和发布。

当前只由显式启用 V1 Knowledge 的 `KernelGenWorkflow` 使用。
SimpleOpt/BatchSimpleOpt 不创建 Knowledge workspace state，也不调用本模块的
查询或发布服务。

一句话说明：

```text
Agent 查知识和提出建议，Python 用真实实验结果决定写入什么。
```

Agent 不能直接修改公共 Catalog。

## 1. 六个核心对象

| 对象 | 说人话解释 |
|---|---|
| Source | 固定版本的原始文档或代码 |
| Concept | 一条可复用的约束、方法、诊断或经验 |
| Query / Retrieval | Agent 搜了什么，以及真正打开了什么 |
| Candidate | Agent 建议发布的知识，还不是正式知识 |
| Observation | Python 从真实 Ledger 和 eval 结果冻结的实验事实 |
| Catalog | Source、Concept、Observation、索引和发布审计的集合 |

“检索到”不等于“使用了”。只有 Agent 真正读取详情，并在实验 round 中记录 `adopted`、`adapted` 或 `rejected`，系统才能分析这条知识有没有帮助。

## 2. 一次完整流程

```text
构建当前算子和硬件上下文
  → 查询 Concept 或 Source
  → 记录详情读取
  → Agent 写代码并提交 ExperimentPlan
  → eval service 产生真实结果
  → Ledger 记录知识使用和结果
  → Distiller 提交 Candidate
  → Publisher 校验 Candidate
  → Python 生成 Observation
  → 更新 Concept、索引和发布审计
```

几个边界必须记住：

- eval 结果只来自 eval service；
- Observation 只来自已经归档的 Ledger；
- Source 引用必须来自本 workspace 真正读取过的 Source；
- Concept relation 必须来自本 workspace 真正读取过的 Concept；
- 公共 Catalog 只有 Publisher 能写。

## 3. 目录怎么读

```text
knowledge/
├── models/                 Concept、Observation、Candidate、Query 和 Source 模型
├── contracts/              workspace、检索和发布的运行时返回契约
├── publishing/
│   ├── publisher.py        发布锁、staging、提交和失败恢复
│   ├── materialize.py      Candidate 和 Ledger 转为 Observation/Evidence
│   ├── merge.py            Concept 查重、证据合并、状态和 supersede
│   ├── validation.py       Candidate 发布契约
│   ├── lifecycle.py        deprecate、delete、revalidate 和 rollback
│   └── static_import.py    审核后的 Source → Candidate 导入
├── catalog.py              权威 Markdown/JSONL 读写和 Catalog Git 提交
├── index.py                可删除重建的 Concept/usage SQLite 索引
├── records.py              retrieval log 和 candidate outbox
├── query.py                Concept 查询和详情读取
├── sources.py              Source 查询和原文读取
├── scope.py                direct、analogy、incompatible 判定
├── ranking.py              根据真实使用效果调整排序
├── context.py              构建并写入 target/operator/query 上下文
├── run_archive.py          不可变运行归档
├── run_facts.py            从 Ledger 物化 Observation
├── round_index.py          历史 round 索引
├── metrics.py              知识使用效果统计
├── solutions.py            fresh/fork/resume 的 best solution registry
├── validation.py           整个 Catalog 的完整性校验
├── vocabulary.py           canonical 词表和 alias
├── layout.py               Catalog 与 workspace 的磁盘路径
├── config.py               KB 模式和 Catalog 路径
└── bootstrap.py            唯一服务组装入口
```

目录按真实功能命名，不再使用 `domain/application/adapters/ingestion` 这类需要先理解架构层次的通用目录。这里不记录容易过期的文件行数。

## 4. 想改功能时去哪里

| 目标 | 先看 |
|---|---|
| 改字段 | `models/`、`docs/design/knowledge/knowledge_concept_contract.md` |
| 改 scope 匹配 | `scope.py` |
| 改 Concept 检索 | `query.py`、`index.py`、`ranking.py` |
| 改 Source 检索 | `sources.py`、`vocabulary.py` |
| 改发布规则 | `publishing/` |
| 改运行事实归档 | `run_archive.py`、`run_facts.py` |
| 改词表 | `vocabulary.py` 和 Catalog 中的 canonical/alias YAML |
| 改 CLI | `tools/kb.py`，不要新建一批一命令一个文件的薄包装 |

## 5. 常用检查命令

从 KernelGen 仓库执行：

```bash
cd /data/akg_kernel_bench_lite/kernelgen

python3 -m kernelgen.tools.kb validate --kb kb

python3 -m kernelgen.tools.kb rebuild \
  --kb kb \
  --run-archive run-archive
```

查看全部 KB 运维子命令：

```bash
python3 -m kernelgen.tools.kb --help
```

当前统一入口包含：

```text
validate
rebuild
import
publish
revalidate
metrics
rollback
deprecate
delete
```

开发检查、审计和一次性迁移不放在 `tools/`：

```text
scripts/dev/
scripts/kb/
scripts/migrations/
```

## 6. 详细文档

- 字段含义和正文结构：`docs/design/knowledge/knowledge_concept_contract.md`
- 当前实现和数据流：`docs/design/knowledge/knowledge_base_implementation.md`
- 后续工作：`docs/design/knowledge/knowledge_base_todo.md`
- 词表和 Source → Concept：`docs/design/knowledge/kb_vocabulary_systematization.md`
- KernelGenWorkflow 快速运行：`docs/guides/KB_QUICKSTART.md`

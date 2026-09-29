# KernelGen Knowledge Base

这是 KernelGen 的正式知识 Catalog。完整设计、数据流、Schema 示例和检查命令只在
`docs/design/knowledge/knowledge_base_implementation.md` 维护。

核心目录：

- `sources/`：固定版本的原始博客、文档和仓库内容；
- `observations/`：被正式知识引用的冻结实验事实；
- `concepts/`：Agent 可查询的正式知识；
- `.derived/`：可删除重建的 Concept 与历史 round SQLite 索引；
- `publish/audit/`：Publisher 幂等审计；
- `schemas/`：从 Pydantic 生成的 JSON Schema。

Agent workspace 不复制 Catalog，只引用同 epoch 的不可变 snapshot。Agent
只能写自己的 query log 和 Candidate outbox；Publisher 是 Catalog 唯一写入者。
该规则只适用于显式启用 V1 Knowledge 的 KernelGenWorkflow；
SimpleOpt/BatchSimpleOpt 不会 materialize 或复制本 Catalog。

Static ingestion is dry-run by default:

```bash
python3 -m kernelgen.tools.kb import \
  --kb kb --plan kb/sources/ingestion/<source>.yaml
```

Use `--publish` only after reviewing the plan and dry-run Candidate IDs.

Repository sources are captured from the complete Git tree:

```bash
python3 tools/snapshot_git_source.py \
  --repo /path/to/triton-ascend \
  --kb kb \
  --package kb/sources/packages/triton-ascend-a60006ac63f3.yaml \
  --name triton-ascend \
  --revision a60006ac63f3f5068cc4b217ebd4e14ce0801f3e \
  --replace
```

The MCP `query_sources` tool searches bounded snippets from immutable
`indexed`/`direct`/`extract` documents and returns source classification plus
usage-policy metadata. Set `include_source_only=true` only when exact code,
test, or other raw repository material is required.

默认不启用 V1 Knowledge；KernelGenWorkflow 只有在显式传入
`KnowledgeConfig` 时启用 `read_write_v1`。

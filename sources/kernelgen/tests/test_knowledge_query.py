"""Knowledge query, retrieval, and dependency behavior tests."""

from __future__ import annotations

import sqlite3

from kernelgen.tests._knowledge_runtime_support import *  # noqa: F403
from kernelgen.tests._knowledge_runtime_support import (
    _DEFINITION,
    _evidence_state,
    _lifecycle_candidate,
    _materialize,
    _reference_body,
    _seed,
    _target_devices_match,
)


def _write_query_variant(
    catalog_root,
    base,
    *,
    concept_id,
    claim_key,
    title,
    summary,
    body,
    keywords=(),
    symptoms=(),
):
    concept = base.model_copy(
        update={
            "id": concept_id,
            "claim_key": claim_key,
            "title": title,
            "summary": summary,
            "body": body,
            "retrieval": base.retrieval.model_copy(
                update={
                    "keywords": list(keywords),
                    "symptoms": list(symptoms),
                    "techniques": [],
                }
            ),
            "managed": base.managed.model_copy(
                update={"content_hash": "sha256:" + "0" * 64}
            ),
        }
    )
    concept = concept.model_copy(
        update={
            "managed": concept.managed.model_copy(
                update={"content_hash": canonical_concept_hash(concept)}
            )
        }
    )
    FilesystemCatalog(catalog_root).write_concept(concept)
    return concept


def test_workspace_query_and_direct_detail(tmp_path):
    catalog_root = tmp_path / "catalog"
    concept = _seed(catalog_root)
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    _materialize(catalog_root, workspace)

    state = json.loads(
        KnowledgeLayout(workspace).state.read_text(encoding="utf-8")
    )
    shared_catalog = (workspace / state["catalog_ref"]).resolve()
    assert not (workspace / "kb").exists()
    assert shared_catalog == catalog_root.resolve()
    assert FilesystemCatalog(shared_catalog).snapshot()

    context = build_query_context(
        workspace,
        phase="initial",
        task="architecture_selection",
        question="How should optimization use two_stage_reduction?",
    )
    services = build_workspace_services(workspace)
    bundle = services.query.execute(context)
    repeated = services.query.execute(context)

    assert [item.concept_ref for item in bundle.direct] == [
        concept.id
    ]
    assert "keyword:optimization" in bundle.direct[0].matched_on
    assert "keyword:two_stage_reduction" in bundle.direct[0].matched_on
    assert bundle.coverage.gaps == []
    assert repeated.query_id != bundle.query_id
    assert repeated.model_copy(update={"query_id": bundle.query_id}) == bundle
    assert "target:portable" in bundle.direct[0].matched_on
    assert not KnowledgeLayout(workspace).query_log.exists()
    assert KnowledgeLayout(workspace).retrieval_log.is_file()
    query_record = services.audit.read(bundle.query_id)
    assert query_record.event_id == bundle.query_id
    assert query_record.origin == "application"
    assert (
        query_record.request["question"]
        == "How should optimization use two_stage_reduction?"
    )
    assert (
        query_record.request["operator_signature"]["definition_id"]
        == build_operator_signature(_DEFINITION).definition_id
    )
    assert query_record.request["target_context"]["device"] == "A100"
    documents = services.get.execute(
        bundle.query_id,
        [bundle.direct[0].concept_ref],
    )
    assert documents[0].body.startswith("# Claim")

    with pytest.raises(KeyError, match="concept ref not found"):
        services.get.execute(
            bundle.query_id,
            ["kg:method:not-returned"],
        )
    retrievals = services.retrieval.read_all()
    assert [
        (record.operation, record.status)
        for record in retrievals
    ] == [
        ("query_knowledge", "success"),
        ("query_knowledge", "success"),
        ("get_knowledge", "success"),
        ("get_knowledge", "error"),
    ]


def test_coverage_gaps_follow_query_task(tmp_path):
    catalog_root = tmp_path / "catalog"
    _seed(catalog_root)
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    _materialize(catalog_root, workspace)
    context = build_query_context(
        workspace,
        phase="initial",
        task="constraint_check",
        question="What constraint applies to this reduction?",
    )

    bundle = build_workspace_services(workspace).query.execute(context)

    assert bundle.direct == []
    assert bundle.coverage.gaps == [
        "no direct applicable knowledge",
        "no direct reference for constraint_check",
    ]


def test_retrieval_events_read_legacy_query_log(tmp_path):
    legacy_path = tmp_path / "query-log.jsonl"
    record = QueryRecord(
        query_id="query:legacy:1",
        snapshot="sha256:" + "a" * 64,
        context_fingerprint="sha256:" + "b" * 64,
        phase="initial",
        task="architecture_selection",
        question="legacy query",
        returned_refs=["kg:method:legacy"],
        result_levels={"kg:method:legacy": "direct"},
        created_at=datetime.now(timezone.utc),
    )
    FileQueryAudit(legacy_path).append(record)

    recovered = FileRetrievalAudit(
        tmp_path / "retrieval-log.jsonl",
        legacy_query_path=legacy_path,
    ).read(record.query_id)

    assert recovered.event_id == record.query_id
    assert recovered.snapshot == record.snapshot
    assert "context_fingerprint" not in record.model_dump(mode="json")
    assert recovered.returned_refs == record.returned_refs
    assert recovered.result_levels == record.result_levels


def test_stale_concept_is_not_returned(tmp_path):
    catalog_root = tmp_path / "catalog"
    concept = _seed(catalog_root)
    concept = concept.model_copy(
        update={
            "stale_after": date(2025, 1, 1),
        }
    )
    concept = concept.model_copy(
        update={
            "managed": concept.managed.model_copy(
                update={"content_hash": canonical_concept_hash(concept)}
            )
        }
    )
    FilesystemCatalog(catalog_root).write_concept(concept)
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    _materialize(catalog_root, workspace)

    bundle = build_workspace_services(workspace).query.execute(
        build_query_context(
            workspace,
            phase="initial",
            task="architecture_selection",
            question="How should this reduction be structured?",
        )
    )

    assert bundle.direct == []


def test_definition_explicit_scope_metadata_is_preserved():
    signature = build_operator_signature(
        {
            **_DEFINITION,
            "capabilities": ["async_copy"],
            "dataflow": ["load_reduce_store"],
            "layouts": ["row_major"],
            "numerics": {
                "exact": False,
                "allow_tf32": False,
                "max_abs_error": 1e-5,
            },
        }
    )

    assert signature.required_capabilities == ["async_copy"]
    assert signature.dataflow == ["load_reduce_store"]
    assert signature.layouts == ["row_major"]
    assert signature.numerics.exact is None
    assert signature.numerics.allow_tf32 is False
    assert signature.numerics.max_abs_error == 1e-5

    target = build_target_context(
        target_hardware="A100",
        implementation_language="triton",
        service_status={
            "backend": "cuda",
            "target": {
                "backend": "cuda",
                "device": "A100",
                "capabilities": ["async_copy"],
            },
            "profile": {"capabilities": ["instruction_listing"]},
        },
    )
    assert target.capabilities == ["async_copy"]


def test_workspace_provenance_uses_only_successful_detail_reads(tmp_path):
    workspace = tmp_path / "agent0"
    layout = KnowledgeLayout(workspace)
    layout.retrieval_log.parent.mkdir(parents=True)
    records = [
        {
            "status": "success",
            "operation": "query_sources",
            "returned_sources": ["source:docs::search-only.md"],
        },
        {
            "status": "success",
            "operation": "get_source",
            "returned_sources": [
                "source:docs@rev-7::used.md:L10-L20"
            ],
        },
        {
            "status": "success",
            "operation": "query_knowledge",
            "returned_refs": ["kg:method:search-only"],
        },
        {
            "status": "success",
            "operation": "get_knowledge",
            "returned_refs": ["kg:method:used"],
        },
        {
            "status": "error",
            "operation": "get_source",
            "returned_sources": ["source:docs::failed.md"],
        },
    ]
    layout.retrieval_log.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )

    sources, concept_ids = workspace_provenance(workspace)

    assert [
        (item.resource, item.revision, item.locator) for item in sources
    ] == [("source:docs", "rev-7", "used.md:L10-L20")]
    assert concept_ids == ["kg:method:used"]


def test_post_profile_context_accepts_validated_draft_findings(tmp_path):
    catalog_root = tmp_path / "catalog"
    _seed(catalog_root)
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    _materialize(catalog_root, workspace)
    Ledger(workspace).record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.0,
            "num_workloads": 1,
            "num_passed": 1,
            "per_workload": [{"uuid": "u0", "status": "PASSED"}],
        },
        "def kernel(): pass",
        {
            "kind": "baseline",
            "strategy": "collect profile evidence",
            "code_changes": "none",
            "hypothesis": "profile the current kernel",
            "expected_effect": {
                "metric": "geo_mean",
                "direction": "establish_baseline",
                "mechanism": "collect counters",
            },
            "source": {"kind": "baseline"},
            "knowledge_uses": [],
        },
        profile_enabled=True,
    )
    draft = ProfileFindingContext(
        category="compute_pipeline",
        label="scalar pipeline dominant",
        confidence="high",
        workload_uuids=["u0"],
    )

    context = build_query_context(
        workspace,
        phase="post_profile",
        task="next_experiment",
        question="How should scalar pipeline overhead be reduced?",
        round_num=1,
        draft_findings=[draft],
    )

    assert context.findings == [draft]


def test_sibling_workspaces_share_the_live_catalog(tmp_path):
    catalog_root = tmp_path / "catalog"
    _seed(catalog_root)
    first = tmp_path / "epoch" / "agent0"
    second = tmp_path / "epoch" / "agent1"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    _materialize(catalog_root, first)
    _materialize(catalog_root, second)

    first_state = json.loads(
        KnowledgeLayout(first).state.read_text(encoding="utf-8")
    )
    second_state = json.loads(
        KnowledgeLayout(second).state.read_text(encoding="utf-8")
    )
    first_catalog = (first / first_state["catalog_ref"]).resolve()
    second_catalog = (second / second_state["catalog_ref"]).resolve()

    assert first_catalog == second_catalog
    assert not (first / "kb").exists()
    assert not (second / "kb").exists()
    assert first_catalog == catalog_root.resolve()


def test_workspace_sees_canonical_catalog_updates(tmp_path):
    catalog_root = tmp_path / "catalog"
    concept = _seed(catalog_root)
    workspace = tmp_path / "epoch" / "agent0"
    workspace.mkdir(parents=True)
    _materialize(catalog_root, workspace)
    state = json.loads(
        KnowledgeLayout(workspace).state.read_text(encoding="utf-8")
    )
    shared_catalog = (workspace / state["catalog_ref"]).resolve()
    initial_snapshot = FilesystemCatalog(catalog_root).snapshot()

    updated = concept.model_copy(
        update={
            "summary": "Canonical catalog changed after the epoch started.",
            "managed": concept.managed.model_copy(
                update={"content_hash": "sha256:" + "0" * 64}
            ),
        }
    )
    updated = updated.model_copy(
        update={
            "managed": updated.managed.model_copy(
                update={"content_hash": canonical_concept_hash(updated)}
            )
        }
    )
    FilesystemCatalog(catalog_root).write_concept(updated)

    assert FilesystemCatalog(catalog_root).snapshot() != initial_snapshot
    current = FilesystemCatalog(catalog_root).snapshot()
    assert FilesystemCatalog(shared_catalog).snapshot() == current
    assert build_workspace_services(workspace).get.catalog.snapshot() == current


def test_catalog_overwrites_one_current_concept_file(tmp_path):
    catalog_root = tmp_path / "catalog"
    catalog = FilesystemCatalog(catalog_root)
    first = _seed(catalog_root)
    first_path = catalog.concept_path(first)

    second = first.model_copy(
        update={
            "summary": "Overwrite the one current Concept file.",
            "managed": first.managed.model_copy(
                update={"content_hash": "sha256:" + "0" * 64}
            ),
        }
    )
    second = second.model_copy(
        update={
            "managed": second.managed.model_copy(
                update={"content_hash": canonical_concept_hash(second)}
            )
        }
    )
    second_path = catalog.write_concept(second)

    assert second_path == first_path
    assert second_path.name == "method--two-stage-reduction.md"
    assert "revision:" not in second_path.read_text(encoding="utf-8")
    assert catalog.get_concept(second.id) == second
    assert catalog.iter_concepts() == [second]
    assert list((catalog_root / "concepts").glob("*.md")) == [second_path]
    assert validate_knowledge_base(catalog_root).valid
    assert catalog.write_concept(second) == second_path


def test_migration_keeps_latest_concept_and_removes_versioned_files(
    tmp_path,
    monkeypatch,
):
    catalog_root = tmp_path / "catalog"
    catalog = FilesystemCatalog(catalog_root)
    first = _seed(catalog_root)
    first_path = catalog.concept_path(first)
    first_version = first_path.with_name(
        first_path.stem + "@1.md"
    )
    first_path.rename(first_version)
    first_version.write_text(
        first_version.read_text(encoding="utf-8").replace(
            "\nkind:", "\nrevision: 1\nkind:", 1
        ),
        encoding="utf-8",
    )
    second = first.model_copy(
        update={
            "summary": "The latest Concept body wins migration.",
            "managed": first.managed.model_copy(
                update={"content_hash": "sha256:" + "0" * 64}
            ),
        }
    )
    second = second.model_copy(
        update={
            "managed": second.managed.model_copy(
                update={"content_hash": canonical_concept_hash(second)}
            )
        }
    )
    second_path = catalog.write_concept(second)
    second_version = second_path.with_name(
        second_path.stem + "@2.md"
    )
    second_path.rename(second_version)
    second_version.write_text(
        second_version.read_text(encoding="utf-8").replace(
            "\nkind:", "\nrevision: 2\nkind:", 1
        ),
        encoding="utf-8",
    )
    original = {
        path: path.read_bytes()
        for path in (first_version, second_version)
    }

    planned = migrate_single_concept_files(catalog_root)
    assert planned == {
        "status": "planned",
        "concepts": 1,
        "source_files": 2,
        "target_files": 1,
        "obsolete_files": 2,
        "source_revisions_unchanged": True,
    }

    monkeypatch.setattr(
        "kernelgen.scripts.migrations.migrate_single_concept_files."
        "SQLiteKnowledgeIndex.rebuild",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("index rebuild failed")
        ),
    )
    with pytest.raises(RuntimeError, match="index rebuild failed"):
        migrate_single_concept_files(catalog_root, apply=True)
    assert {
        path: path.read_bytes()
        for path in (first_version, second_version)
    } == original
    assert not second_path.exists()
    monkeypatch.undo()

    migrated = migrate_single_concept_files(catalog_root, apply=True)
    assert migrated["status"] == "migrated"
    assert migrated["valid"] is True
    assert list((catalog_root / "concepts").glob("*.md")) == [second_path]
    assert "revision:" not in second_path.read_text(encoding="utf-8")
    assert catalog.get_concept(second.id).summary == second.summary
    assert validate_knowledge_base(catalog_root).valid


def test_backend_scope_does_not_cross_target(tmp_path):
    catalog_root = tmp_path / "catalog"
    _seed(catalog_root, level="backend", backend="cuda")
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    _materialize(catalog_root, workspace, backend="npu")
    context = build_query_context(
        workspace,
        phase="initial",
        task="architecture_selection",
        question="reduction",
    )
    bundle = build_workspace_services(workspace).query.execute(context)
    assert bundle.direct == []
    assert bundle.analogies == []


def test_retrieval_phase_and_task_are_eligibility_filters(tmp_path):
    catalog_root = tmp_path / "catalog"
    concept = _seed(catalog_root)
    concept = concept.model_copy(
        update={
            "retrieval": concept.retrieval.model_copy(
                update={
                    "phases": ["post_profile"],
                    "tasks": ["diagnosis"],
                }
            )
        }
    )
    concept = concept.model_copy(
        update={
            "managed": concept.managed.model_copy(
                update={"content_hash": canonical_concept_hash(concept)}
            )
        }
    )
    FilesystemCatalog(catalog_root).write_concept(concept)
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    _materialize(catalog_root, workspace)
    context = build_query_context(
        workspace,
        phase="initial",
        task="architecture_selection",
        question="reduction",
    )
    bundle = build_workspace_services(workspace).query.execute(context)
    assert bundle.direct == []
    assert bundle.analogies == []



def test_knowledge_dependency_boundaries():
    root = Path(__file__).resolve().parents[1]

    def imports(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        return modules

    forbidden_core = (
        "kernelgen.agents",
        "kernelgen.framework",
        "kernelgen.mcp_server",
        "kernelgen.workflows",
    )
    knowledge_paths = sorted((root / "knowledge").glob("**/*.py"))
    for path in knowledge_paths:
        assert not any(
            module.startswith(forbidden_core)
            for module in imports(path)
        ), path
    assert not any(
        module.startswith("kernelgen.knowledge")
        for module in imports(root / "framework" / "parallel.py")
    )
    ledger_importers = [
        path.name
        for path in knowledge_paths
        if any(
            module.startswith("kernelgen.data.ledger")
            for module in imports(path)
        )
    ]
    assert ledger_importers == ["context.py", "run_facts.py"]


def test_recent_detail_read_refs_ignore_queries_dedupe_reads_and_fail_open(
    tmp_path,
):
    catalog_root = tmp_path / "catalog"
    concept = _seed(catalog_root)
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    _materialize(catalog_root, workspace)
    context = build_query_context(
        workspace,
        phase="initial",
        task="architecture_selection",
        question="reduction",
    )
    bundle = build_workspace_services(workspace).query.execute(context)
    assert recent_detail_read_refs(workspace, after=None) == []

    services = build_workspace_services(workspace)
    services.get.execute(bundle.query_id, [concept.id])
    services.get.execute(bundle.query_id, [concept.id])
    assert recent_detail_read_refs(workspace, after=None) == [concept.id]
    assert recent_detail_read_refs(
        workspace,
        after=datetime(2100, 1, 1, tzinfo=timezone.utc),
    ) == []

    KnowledgeLayout(workspace).retrieval_log.write_text(
        "{not-json}\n",
        encoding="utf-8",
    )
    assert recent_detail_read_refs(workspace, after=None) == []


def test_bm25_breaks_structured_ties_with_question_relevance(tmp_path):
    catalog_root = tmp_path / "catalog"
    base = _seed(catalog_root)
    relevant = base.model_copy(
        update={
            "summary": (
                "Accumulate partial values through a hierarchical reduction "
                "before the final output stage."
            ),
            "body": (
                "# Claim\n\nHierarchically accumulate partial values, then "
                "perform one final reduction."
            ),
            "managed": base.managed.model_copy(
                update={"content_hash": "sha256:" + "0" * 64}
            ),
        }
    )
    relevant = relevant.model_copy(
        update={
            "managed": relevant.managed.model_copy(
                update={"content_hash": canonical_concept_hash(relevant)}
            )
        }
    )
    FilesystemCatalog(catalog_root).write_concept(relevant)
    generic = _write_query_variant(
        catalog_root,
        base,
        concept_id="kg:method:aaa-generic-loop",
        claim_key="method.reduction.generic_loop",
        title="Generic loop scheduling",
        summary="Adjust loop scheduling parameters for a generic reduction kernel.",
        body="# Claim\n\nTune the loop schedule and launch geometry.",
    )
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    _materialize(catalog_root, workspace)

    bundle = build_workspace_services(workspace).query.execute(
        build_query_context(
            workspace,
            phase="initial",
            task="architecture_selection",
            question="How should hierarchical partials be accumulated?",
        )
    )

    assert bundle.direct[0].concept_ref == relevant.id
    assert generic.id in {item.concept_ref for item in bundle.direct}
    assert "lexical:english_bm25" in bundle.direct[0].matched_on


def test_chinese_query_expands_to_english_concept_terms(tmp_path):
    catalog_root = tmp_path / "catalog"
    (catalog_root / "aliases.yaml").parent.mkdir(parents=True, exist_ok=True)
    (catalog_root / "aliases.yaml").write_text(
        "symptoms:\n"
        "  memory_bandwidth:\n"
        "    - memory bandwidth\n"
        "    - 内存带宽\n"
        "techniques: {}\n",
        encoding="utf-8",
    )
    base = _seed(catalog_root)
    relevant = _write_query_variant(
        catalog_root,
        base,
        concept_id="kg:method:memory-bandwidth-routing",
        claim_key="method.memory_bandwidth.routing",
        title="Diagnose memory bandwidth saturation",
        summary="Route bandwidth-bound reductions toward fewer bytes and wider loads.",
        body="# Claim\n\nMeasure memory bandwidth before changing compute tiles.",
        symptoms=["memory_bandwidth"],
    )
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    _materialize(catalog_root, workspace)

    bundle = build_workspace_services(workspace).query.execute(
        build_query_context(
            workspace,
            phase="initial",
            task="architecture_selection",
            question="如何解决内存带宽瓶颈？",
        )
    )

    result = next(item for item in bundle.direct if item.concept_ref == relevant.id)
    assert "lexical:english_bm25" in result.matched_on


def test_english_query_expands_to_chinese_concept_text(tmp_path):
    catalog_root = tmp_path / "catalog"
    (catalog_root / "aliases.yaml").parent.mkdir(parents=True, exist_ok=True)
    (catalog_root / "aliases.yaml").write_text(
        "symptoms:\n"
        "  pipeline_stall:\n"
        "    - pipeline stall\n"
        "    - 流水线停顿\n"
        "techniques: {}\n",
        encoding="utf-8",
    )
    base = _seed(catalog_root)
    relevant = _write_query_variant(
        catalog_root,
        base,
        concept_id="kg:method:chinese-pipeline-stall",
        claim_key="method.pipeline_stall.chinese",
        title="诊断流水线停顿",
        summary=(
            "检查数据搬运与计算之间的流水线停顿，"
            "并验证双缓冲是否生效。"
        ),
        body="# Claim\n\n先定位流水线停顿，再调整缓冲级数。",
        symptoms=["pipeline_stall"],
    )
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    _materialize(catalog_root, workspace)

    bundle = build_workspace_services(workspace).query.execute(
        build_query_context(
            workspace,
            phase="initial",
            task="architecture_selection",
            question="How should a pipeline stall be diagnosed?",
        )
    )

    result = next(item for item in bundle.direct if item.concept_ref == relevant.id)
    assert "lexical:chinese_trigram_bm25" in result.matched_on


def test_legacy_index_schema_is_rebuilt(tmp_path):
    path = tmp_path / "knowledge.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO metadata VALUES('snapshot', 'legacy-snapshot')"
        )

    assert SQLiteKnowledgeIndex(path).snapshot() == ""

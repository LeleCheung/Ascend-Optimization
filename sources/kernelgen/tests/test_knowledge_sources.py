"""Knowledge Source search and detail-read behavior tests."""

from __future__ import annotations

from types import SimpleNamespace

from kernelgen.knowledge.models import (
    CandidateReviewDecision,
    KnowledgeUsageScope,
    KnowledgeReviewOutput,
    SourcePackage,
)
from kernelgen.knowledge.source_index import SQLiteSourceIndex
from kernelgen.knowledge.sources import (
    SourceSearcher,
    _path_token_boost,
    _target_backend_relevance,
)
from kernelgen.tests._knowledge_runtime_support import *  # noqa: F403
from kernelgen.tests._knowledge_runtime_support import (
    _DEFINITION,
    _evidence_state,
    _finalize_test_round,
    _lifecycle_candidate,
    _materialize,
    _reference_body,
    _seed,
    _target_devices_match,
)


def test_complete_source_snapshot_is_searchable_without_concept_conversion(tmp_path):
    kb = tmp_path / "kb"
    content = kb / "sources" / "content" / "repo" / "abc123"
    source = content / "python" / "backend.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def choose_core_count(problem_size):\n"
        "    # Combine logical grid tasks across physical cores.\n"
        "    return problem_size\n",
        encoding="utf-8",
    )
    unused_source = content / "python" / "unused.py"
    unused_source.write_text("unused reference\n", encoding="utf-8")
    data = source.read_bytes()
    unused_data = unused_source.read_bytes()
    manifest_path = kb / "sources" / "manifests" / "repo-abc123.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_package": "source:repo.abc123",
                "revision": "abc123",
                "object_format": "sha1",
                "entries": [
                    {
                        "path": "python/backend.py",
                        "mode": "100644",
                        "object_type": "blob",
                        "object_id": "a" * 40,
                        "size": len(data),
                        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                    },
                    {
                        "path": "python/unused.py",
                        "mode": "100644",
                        "object_type": "blob",
                        "object_id": "b" * 40,
                        "size": len(unused_data),
                        "sha256": (
                            "sha256:"
                            + hashlib.sha256(unused_data).hexdigest()
                        ),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    package = kb / "sources" / "packages" / "repo-abc123.yaml"
    package.parent.mkdir(parents=True)
    package.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "id": "source:repo.abc123",
                "source_type": "repository",
                "origin": "https://example.invalid/repo.git",
                "revision": "abc123",
                "content_root": "kb://sources/content/repo/abc123",
                "license": "MIT",
                "checksum": source_content_checksum(content),
                "manifest_root": "kb://sources/manifests/repo-abc123.json",
                "retrieved_at": "2026-07-27T00:00:00Z",
                "authority": "official",
                "allowed_hosts": ["example.invalid"],
            }
        ),
        encoding="utf-8",
    )
    plan = kb / "sources" / "ingestion" / "repo-abc123.yaml"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "source_package": "source:repo.abc123",
                "rules": [{"include": "*", "mode": "source_only", "kinds": []}],
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    workspace = tmp_path / "epoch" / "agent0"
    workspace.mkdir(parents=True)
    signature = build_operator_signature(_DEFINITION)
    target = build_target_context(
        target_hardware="A100",
        implementation_language="triton",
        service_status={
            "backend": "cuda",
            "target": {
                "backend": "cuda",
                "vendor": "nvidia",
                "device": "A100",
                "architecture": "sm80",
            },
            "software": {
                "language": "triton",
                "compiler": "triton",
                "compiler_version": "3.3.1",
                "runtime": "cuda",
                "runtime_version": "12.6",
                "driver_version": "560.35.03",
            },
        },
    )
    KnowledgeWorkspaceMaterializer(
        catalog_root=kb,
        mode=KnowledgeMode.READ_WRITE_V1,
        run_id="run-source",
        operator_signature=signature,
        target_context=target,
    ).materialize(workspace)
    services = build_workspace_services(workspace)
    result = services.sources.search(
        "logical grid physical cores",
        package_ids=["source:repo.abc123"],
        include_source_only=True,
    )
    assert not (workspace / "kb").exists()
    assert len(result.hits) == 1
    assert result.hits[0].path == "python/backend.py"
    assert result.hits[0].classification == "source_only"
    assert result.hits[0].usage_policy == "unknown"
    assert "physical cores" in result.hits[0].snippet
    document = services.sources.read(
        query_id=result.query_id,
        source_package=result.hits[0].source_package,
        path=result.hits[0].path,
        line_start=1,
        line_end=3,
    )
    assert document.path == "python/backend.py"
    assert document.line_start == 1
    assert document.line_end == 3
    assert document.total_lines == 3
    assert document.truncated is False
    assert "def choose_core_count" in document.content
    unused_document = services.sources.read(
        query_id=result.query_id,
        source_package=result.hits[0].source_package,
        path="python/unused.py",
        line_start=1,
        line_end=1,
    )
    ledger = Ledger(workspace)
    ledger.record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.0,
            "num_workloads": 1,
            "num_passed": 1,
            "per_workload": [
                {"uuid": "w0", "status": "PASSED", "speedup": 1.0}
            ],
        },
        "baseline",
        {
            "kind": "baseline",
            "strategy": "establish baseline",
            "code_changes": "implement baseline",
            "hypothesis": "measure baseline",
            "expected_effect": {
                "metric": "geo_mean",
                "direction": "establish_baseline",
                "mechanism": "measure",
            },
            "source": {"kind": "baseline"},
            "knowledge_uses": [],
        },
        definition_name=_DEFINITION["name"],
        target_hardware="A100",
    )
    _finalize_test_round(ledger, 1)
    ledger.record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.1,
            "num_workloads": 1,
            "num_passed": 1,
            "per_workload": [
                {"uuid": "w0", "status": "PASSED", "speedup": 1.1}
            ],
        },
        "source-guided",
        {
            "kind": "performance",
            "strategy": "apply documented core mapping",
            "code_changes": "map logical tasks to physical cores",
            "hypothesis": "the documented mapping reduces launch overhead",
            "expected_effect": {
                "metric": "geo_mean",
                "direction": "increase",
                "mechanism": "reduce launch overhead",
            },
            "source": {"kind": "knowledge_base"},
            "knowledge_uses": [
                {
                    "source_ref": {
                        "resource": document.source_package,
                        "revision": document.revision,
                        "locator": document.locator,
                    },
                    "query_event_id": result.query_id,
                    "role": "implementation",
                    "disposition": "adopted",
                    "application_note": "used the documented core mapping",
                    "affected_parts": ["launch_grid"],
                }
            ],
        },
    )
    ledger.finalize_round(
        2,
        round_conclusion(
            2,
            knowledge_assessments=[
                {
                    "source_ref": {
                        "resource": document.source_package,
                        "revision": document.revision,
                        "locator": document.locator,
                    },
                    "assessment": "confirmed",
                    "rationale": (
                        "The documented mapping reduced launch overhead "
                        "and improved the measured geo_mean."
                    ),
                }
            ],
        ),
    )
    state = WorkspaceKnowledgeState.model_validate_json(
        KnowledgeLayout(workspace).state.read_text(encoding="utf-8")
    )
    observation = KernelGenRunFactReader(
        KernelGenRunArchive(kb.parent / "run-archive")
    ).materialize_observation(
        workspace,
        state,
        signature,
        target,
        round_num=2,
        run_id="run-source",
    )
    assert observation.applications[0].source_ref.locator == document.locator
    assert (
        observation.applications[0].agent_assessment.assessment
        == "confirmed"
    )
    observation_path = (
        kb / "observations" / "by_run" / "run-source.jsonl"
    )
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text(
        observation.model_dump_json() + "\n",
        encoding="utf-8",
    )
    ranked_source = SourceSearcher(
        kb,
        usage_index=SQLiteKnowledgeIndex(
            CatalogLayout(kb).index
        ),
        usage_scope=KnowledgeUsageScope.from_context(
            signature,
            target,
        ),
    ).search(
        "logical grid physical cores",
        package_ids=["source:repo.abc123"],
        include_source_only=True,
    )
    assert ranked_source.hits[0].usage_summary.single_success_count == 1
    assert ranked_source.hits[0].usage_summary.regression_count == 0
    assert ranked_source.hits[0].usage_summary.agent_confirmed_count == 1
    assert ranked_source.hits[0].usage_summary.observation_refs == [
        observation.id
    ]
    assert (
        "same-scope single successes: 1"
        in ranked_source.hits[0].recommendation_reasons
    )
    assert (
        "Agent-confirmed applications: 1"
        in ranked_source.hits[0].recommendation_reasons
    )
    assert ranked_source.hits[0].usage_summary.retrieved_count == 1
    unused_summary = SQLiteKnowledgeIndex(
        CatalogLayout(kb).index
    ).usage(
        (
            f"{unused_document.source_package}@{unused_document.revision}"
            f"::{unused_document.locator}"
        ),
        KnowledgeUsageScope.from_context(signature, target),
    )
    assert unused_summary.retrieved_count == 1
    assert unused_summary.considered_count == 0
    assert unused_summary.applied_count == 0
    append_candidate_drafts(
        workspace,
        [
            CandidateDraft(
                proposed_kind="method",
                proposed_id="kg:method:documented-core-mapping",
                claim_key="method.launch.documented_core_mapping",
                title="Apply the documented core mapping",
                summary=(
                    "The documented logical-to-physical core mapping improved "
                    "the measured round."
                ),
                domains=["optimization"],
                retrieval={
                    "phases": ["post_evaluation"],
                    "tasks": ["next_experiment"],
                },
                body=(
                    "## Claim\n\nUse the documented core mapping.\n\n"
                    "## Evidence\n\nR2 applied the Source and improved performance.\n\n"
                        "## Applicability\n\nThis exact definition and target.\n\n"
                        "## Action\n\nMap logical tasks to physical cores.\n\n"
                        "### Expected Metric Change\n\n"
                        "Reduce launch overhead and improve geo_mean.\n\n"
                        "## Limits\n\nNo portability claim is made.\n\n"
                        "### Mechanism Requirements\n\n"
                        "The target must expose the documented core mapping."
                    ),
                observation_intents=[
                    {
                        "round_num": 2,
                        "stance": "supports",
                        "rationale": (
                            "R2 measured the Source-guided implementation."
                        ),
                    }
                ],
                source_refs=[
                    {
                        "resource": document.source_package,
                        "revision": document.revision,
                        "locator": document.locator,
                    }
                ],
            )
        ],
    )
    def reviewer(review_input):
        unit = review_input.units[0]
        return KnowledgeReviewOutput(
            decisions=[
                CandidateReviewDecision(
                    review_id=unit.review_id,
                    decision="approve",
                    merge_action="create_new",
                    evidence_refs=[unit.observations[0].id],
                    conflict_refs=[],
                    scope_assessment="compatible",
                    reusable=True,
                    confidence="high",
                    reason=(
                        "The exact-Scope measured round and cited Source support "
                        "the documented core-mapping claim."
                    ),
                )
            ]
        )

    published = BatchPublisher(
        kb,
        KernelGenRunFactReader(
            KernelGenRunArchive(kb.parent / "run-archive")
        ),
    ).publish(
        [workspace],
        run_id="run-source",
        batch_id="run-source-epoch-1",
        reviewer=reviewer,
    )
    assert published.created == [
        "kg:method:documented-core-mapping"
    ]
    metrics = build_knowledge_metrics(
        kb,
        kb.parent / "run-archive",
    )
    assert metrics["retrieval"] == {
        "query_events": 1,
        "queries_with_detail_read": 1,
        "query_to_detail_read_rate": 1.0,
    }
    assert metrics["funnel"] == {
        "retrieved": 2,
        "considered": 1,
        "applied": 1,
        "evaluated": 1,
    }
    assert metrics["performance_improvement"]["single"] == {
        "comparable_rounds": 1,
        "improved_rounds": 1,
        "improvement_rate": 1.0,
    }
    assert metrics["optimization_progress"] == [
        {
            "run_id": "run-source",
            "workspace_id": "epoch.agent0",
            "rounds": 2,
            "first_passed_round": 1,
            "best_round": 2,
        }
    ]
    source_metric = next(
        item
        for item in metrics["per_reference"]
        if item["reference"]
        == (
            f"{document.source_package}@{document.revision}"
            f"::{document.locator}"
        )
    )
    assert source_metric["scope"]["target_device"] == "A100"
    assert source_metric["performance_improved"] == 1
    assert source_metric["agent_confirmed"] == 1
    assert metrics["campaign_comparison"]["available"] is False
    promoted = FilesystemCatalog(kb).get_concept(published.created[0])
    assert promoted.evidence_state == "observed"
    assert promoted.sources[0].resource == document.source_package
    assert promoted.sources[0].revision == document.revision
    assert promoted.sources[0].locator == document.locator
    assert promoted.evidence[0].observation_ref == observation.id
    audit = json.loads(
        (kb / published.audit_ref).read_text(encoding="utf-8")
    )
    assert audit["application_links"] == [
        {
            "observation_ref": observation.id,
            "knowledge_ref": (
                f"{document.source_package}@{document.revision}"
                f"::{document.locator}"
            ),
            "disposition": "adopted",
            "lineage_status": "complete",
            "agent_assessment": {
                "assessment": "confirmed",
                "rationale": (
                    "The documented mapping reduced launch overhead "
                    "and improved the measured geo_mean."
                ),
            },
        }
    ]
    mapped = services.sources.search(
        "logical grid physical cores",
        package_ids=["source:repo.abc123"],
        include_source_only=True,
    )
    assert mapped.hits[0].promoted_concept_refs == [
        "kg:method:documented-core-mapping"
    ]
    assert "actionable promoted concepts: 1" in (
        mapped.hits[0].recommendation_reasons
    )
    assert services.retrieval.read_query(
        mapped.query_id,
        operation="query_sources",
    ).returned_refs == ["kg:method:documented-core-mapping"]
    retrievals = services.retrieval.read_all()
    assert retrievals[0].returned_sources == [
        "source:repo.abc123@abc123::python/backend.py:L1-L3"
    ]
    assert retrievals[1].returned_sources == [
        "source:repo.abc123@abc123::python/backend.py:L1-L3"
    ]
    assert retrievals[2].returned_sources == [
        "source:repo.abc123@abc123::python/unused.py:L1-L1"
    ]
    with pytest.raises(ValueError, match="not readable"):
        services.sources.read(
            query_id=result.query_id,
            source_package="source:repo.abc123",
            path="python/not-returned.py",
        )
    assert [
        (record.operation, record.status)
        for record in services.retrieval.read_all()
    ] == [
        ("query_sources", "success"),
        ("get_source", "success"),
        ("get_source", "success"),
        ("query_sources", "success"),
        ("get_source", "error"),
    ]
    with pytest.raises(ValueError, match="unknown source package id"):
        services.sources.search(
            "logical grid physical cores",
            package_ids=["repo.abc123"],
            include_source_only=True,
        )
    with pytest.raises(ValueError, match="must start with source-query:"):
        services.sources.read(
            query_id="query:not-a-source-query",
            source_package="source:repo.abc123",
            path="python/backend.py",
            line_start=1,
            line_end=3,
        )
    with pytest.raises(
        ValueError,
        match="must use canonical source:<id> form",
    ):
        services.sources.read(
            query_id="",
            source_package="repo.abc123",
            path="python/backend.py",
            line_start=1,
            line_end=3,
        )


def test_source_search_defaults_to_indexed_knowledge_documents(tmp_path):
    kb = tmp_path / "kb"
    content = kb / "sources" / "content" / "repo" / "abc123"
    indexed = content / "docs" / "guide.md"
    duplicate = content / "plugins" / "guide.md"
    source_only = content / "scripts" / "helper.py"
    hierarchical = content / "docs" / "hierarchy.md"
    chinese_html = content / "docs" / "pipeline.html"
    indexed.parent.mkdir(parents=True)
    duplicate.parent.mkdir(parents=True)
    source_only.parent.mkdir(parents=True)
    indexed.write_text("Vector tiling avoids scalar traffic.", encoding="utf-8")
    duplicate.write_text(indexed.read_text(encoding="utf-8"), encoding="utf-8")
    source_only.write_text("# vector tiling debug helper", encoding="utf-8")
    hierarchical.write_text(
        "Hierarchically accumulate partial values before the final stage.",
        encoding="utf-8",
    )
    chinese_html.write_text(
        "<html><body>检查流水线停顿并验证双缓冲。</body></html>",
        encoding="utf-8",
    )
    (kb / "aliases.yaml").write_text(
        "symptoms:\n"
        "  pipeline_stall:\n"
        "    - pipeline stall\n"
        "    - 流水线停顿\n"
        "techniques: {}\n",
        encoding="utf-8",
    )

    def entry(path):
        data = (content / path).read_bytes()
        return {
            "path": path,
            "mode": "100644",
            "object_type": "blob",
            "object_id": hashlib.sha1(data).hexdigest(),
            "size": len(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        }

    manifest = kb / "sources" / "manifests" / "repo-abc123.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_package": "source:repo.abc123",
                "revision": "abc123",
                "object_format": "sha1",
                "entries": [
                    entry("docs/guide.md"),
                    entry("docs/hierarchy.md"),
                    entry("docs/pipeline.html"),
                    entry("plugins/guide.md"),
                    entry("scripts/helper.py"),
                ],
            }
        ),
        encoding="utf-8",
    )
    package = kb / "sources" / "packages" / "repo-abc123.yaml"
    package.parent.mkdir(parents=True)
    package.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "id": "source:repo.abc123",
                "source_type": "repository",
                "origin": "https://example.invalid/repo.git",
                "revision": "abc123",
                "content_root": "kb://sources/content/repo/abc123",
                "license": "MIT",
                "checksum": source_content_checksum(content),
                "manifest_root": "kb://sources/manifests/repo-abc123.json",
                "retrieved_at": "2026-07-27T00:00:00Z",
                "authority": "official",
                "allowed_hosts": ["example.invalid"],
            }
        ),
        encoding="utf-8",
    )
    plan = kb / "sources" / "ingestion" / "repo-abc123.yaml"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "source_package": "source:repo.abc123",
                "rules": [
                    {"include": "*", "mode": "source_only", "kinds": []},
                    {"include": "docs/*.md", "mode": "indexed", "kinds": []},
                    {"include": "plugins/*.md", "mode": "indexed", "kinds": []},
                    {"include": "docs/*.html", "mode": "indexed", "kinds": []},
                ],
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    default = SourceSearcher(kb).search("vector tiling")
    raw = SourceSearcher(kb).search("vector tiling", include_source_only=True)
    morphology = SourceSearcher(kb).search(
        "hierarchical accumulated partials"
    )
    bilingual = SourceSearcher(kb).search("pipeline stall")

    assert [hit.path for hit in default.hits] == ["docs/guide.md"]
    assert default.hits[0].classification == "indexed"
    assert {hit.path for hit in raw.hits} == {
        "docs/guide.md",
        "plugins/guide.md",
        "scripts/helper.py",
    }
    assert [hit.path for hit in morphology.hits] == ["docs/hierarchy.md"]
    assert [hit.path for hit in bilingual.hits] == ["docs/pipeline.html"]
    source_index = SQLiteSourceIndex(CatalogLayout(kb).source_index, kb)
    assert CatalogLayout(kb).source_index.is_file()
    assert source_index.current_key() == source_index.expected_key()
    previous_key = source_index.current_key()
    plan_payload = yaml.safe_load(plan.read_text(encoding="utf-8"))
    plan_payload["rules"].append(
        {"include": "scripts/*.py", "mode": "indexed", "kinds": []}
    )
    plan.write_text(yaml.safe_dump(plan_payload), encoding="utf-8")

    refreshed = SourceSearcher(kb).search("vector tiling")

    assert {hit.path for hit in refreshed.hits} == {
        "docs/guide.md",
        "scripts/helper.py",
    }
    assert source_index.current_key() != previous_key
    assert source_index.current_key() == source_index.expected_key()


def test_source_path_boost_uses_informative_query_tokens_only():
    assert _path_token_boost(
        "templates/quantization_optimization.md",
        ["ascend", "optimization", "quantization", "triton"],
    ) == 4.0
    assert _path_token_boost(
        "docs/triton/python/language.md",
        ["ascend", "language", "python", "triton"],
    ) == 0.0
    assert _path_token_boost(
        "alpha/bravo/charlie/delta.md",
        ["alpha", "bravo", "charlie", "delta"],
    ) == 6.0


def _package_for_backend_policy(*, usage_policy, allowed_backends):
    return SourcePackage(
        id="source:backend-policy-test",
        source_type="documentation",
        origin="https://example.invalid/docs",
        revision="v1",
        content_root="kb://sources/content/backend-policy-test",
        license="test-only",
        search_mode="static_only",
        retrieved_at="2026-08-28T00:00:00Z",
        usage_policy=usage_policy,
        usage_notes=(
            "Only use on the declared backend."
            if usage_policy in {"restricted", "federated_only"}
            else ""
        ),
        allowed_backends=allowed_backends,
    )


def _usage_scope(target_backend):
    return KnowledgeUsageScope(
        definition_id="ks_test",
        target_backend=target_backend,
        target_device="test-device",
    )


def test_target_backend_relevance_boosts_matching_source():
    eligible, adjustment, reason = _target_backend_relevance(
        _package_for_backend_policy(
            usage_policy="restricted",
            allowed_backends=["iluvatar"],
        ),
        _usage_scope("iluvatar"),
    )

    assert eligible is True
    assert adjustment > 0
    assert reason == "target backend match: iluvatar"


def test_target_backend_relevance_blocks_mismatched_restricted_source():
    eligible, adjustment, reason = _target_backend_relevance(
        _package_for_backend_policy(
            usage_policy="restricted",
            allowed_backends=["hygon"],
        ),
        _usage_scope("iluvatar"),
    )

    assert eligible is False
    assert adjustment == 0.0
    assert reason == ""


def test_direct_source_read_cannot_bypass_restricted_backend_policy():
    package = _package_for_backend_policy(
        usage_policy="restricted",
        allowed_backends=["hygon"],
    ).model_copy(
        update={
            "search_mode": "searchable",
            "manifest_root": "kb://sources/manifests/backend-policy.json",
        }
    )
    searcher = object.__new__(SourceSearcher)
    searcher.catalog = SimpleNamespace(
        snapshot=lambda: "sha256:" + "0" * 64,
        iter_source_packages=lambda: [package],
    )
    searcher.usage_scope = _usage_scope("iluvatar")

    with pytest.raises(
        ValueError,
        match="not allowed for target backend iluvatar",
    ):
        searcher._read(
            query_id="source-query:test",
            source_package=package.id,
            path="guide.md",
            line_start=1,
            line_end=1,
        )


def test_target_backend_relevance_keeps_cross_backend_redistributable_analogy():
    eligible, adjustment, reason = _target_backend_relevance(
        _package_for_backend_policy(
            usage_policy="redistributable",
            allowed_backends=["ascend"],
        ),
        _usage_scope("iluvatar"),
    )

    assert eligible is True
    assert adjustment < 0
    assert reason == "cross-backend analogy; source scope: ascend"


def test_target_backend_relevance_keeps_generic_source_neutral():
    eligible, adjustment, reason = _target_backend_relevance(
        _package_for_backend_policy(
            usage_policy="unknown",
            allowed_backends=[],
        ),
        _usage_scope("iluvatar"),
    )

    assert eligible is True
    assert adjustment == 0.0
    assert reason == ""

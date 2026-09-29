"""Knowledge publication lineage and archive behavior tests."""

from __future__ import annotations

from datetime import datetime, timezone

from kernelgen.knowledge.contracts import RetrievalRecord
from kernelgen.knowledge.config import KnowledgeReviewerMode
from kernelgen.knowledge.models import (
    CandidateReviewDecision,
    KnowledgeApplication,
    KnowledgeReviewOutput,
)
from kernelgen.knowledge.records import FileRetrievalAudit
from kernelgen.knowledge.run_facts import _lineage_status
from kernelgen.tests._knowledge_runtime_support import *  # noqa: F403
from kernelgen.tests._knowledge_runtime_support import (
    _DEFINITION,
    _evidence_state,
    _finalize_test_round,
    _lifecycle_candidate,
    _materialize,
    _reference_body,
    _seed,
    _service_status,
    _target_devices_match,
)


def test_epoch_publisher_materializes_evidence_and_is_idempotent(tmp_path):
    catalog_root = tmp_path / "catalog"
    _seed(catalog_root)
    workspace = tmp_path / "1R" / "agent0"
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
        catalog_root=catalog_root,
        mode=KnowledgeMode.READ_WRITE_V1,
        run_id="run-1",
        operator_signature=signature,
        target_context=target,
    ).materialize(workspace)
    ledger = Ledger(workspace)
    ledger.record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.25,
            "num_workloads": 1,
            "num_passed": 1,
            "per_workload": [
                {
                    "uuid": "w0",
                    "status": "PASSED",
                    "speedup": 1.25,
                }
            ],
        },
        "def kernel(): pass",
        {
            "kind": "baseline",
            "strategy": "measure two-stage reduction",
            "code_changes": "implement two stages",
            "hypothesis": "two stages reduce long-reduction latency",
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
        implementation_language="triton",
    )
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint="test-round-0001",
        snapshot_path=".kernelgen/evals/round-0001",
    )
    append_candidate_drafts(
        workspace,
        [
            CandidateDraft(
                proposed_kind="experience",
                claim_key="experience.sum_rows.two_stage",
                title="Measured two-stage reduction",
                summary="Two stages improved this A100 definition.",
                domains=["optimization"],
                scope_hints={"motifs": ["two_stage_reduction"]},
                retrieval={
                    "phases": ["initial", "plateau"],
                    "tasks": ["architecture_selection", "next_experiment"],
                    "techniques": ["two_stage_reduction"],
                },
                body=(
                    "## Claim\n\nTwo stages improve this measured definition.\n\n"
                    "## Evidence\n\nR1 directly measured the two-stage implementation.\n\n"
                    "## Applicability\n\nThis definition and its constant workload.\n\n"
                    "## Action\n\nRetain the measured two-stage structure.\n\n"
                    "## Limits\n\nNo portability claim is made."
                ),
                observation_intents=[
                    {
                        "round_num": 1,
                        "stance": "supports",
                        "rationale": (
                            "The round directly measured the proposed "
                            "two-stage implementation."
                        ),
                        "confidence": "high",
                    }
                ],
            )
        ],
    )
    # A workspace created by the old implementation may still carry opaque
    # hashes as its operator and target identities. Reading it must normalize
    # those fields instead of rejecting an otherwise valid publication.
    layout = KnowledgeLayout(workspace)
    layout.operator_signature.write_text(
        signature.model_copy(
            update={"definition_id": "sha256:" + "a" * 64}
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    legacy_target = target.model_dump(mode="json")
    legacy_target["fingerprint"] = "sha256:" + "b" * 64
    layout.target_context.write_text(
        json.dumps(legacy_target, indent=2),
        encoding="utf-8",
    )
    archive_root = tmp_path / "run-archive"
    run_archive = KernelGenRunArchive(archive_root)
    publisher = BatchPublisher(
        catalog_root,
        KernelGenRunFactReader(run_archive),
    )
    review_packets = []

    def reviewer(review_input):
        review_packets.append(review_input)
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
                        "The measured round directly supports the exact-Scope "
                        "two-stage claim."
                    ),
                )
            ]
        )

    result = publisher.publish(
        [workspace],
        run_id="run-1",
        batch_id="run-1-epoch-1",
        reviewer=reviewer,
    )

    assert result.status == "published"
    assert result.reviewer_mode == "enforce"
    assert len(review_packets) == 1
    assert result.approved_candidates == result.processed_candidates
    assert result.deferred_candidates == []
    assert result.review_rejected_candidates == []
    assert len(result.created) == 1
    assert len(result.observations_added) == 1
    assert validate_knowledge_base(catalog_root).valid
    published_concept = FilesystemCatalog(catalog_root).get_concept(
        result.created[0]
    )
    assert published_concept.scope.target.level == "exact"
    assert published_concept.scope.target.devices == ["A100"]
    assert not hasattr(published_concept.scope.target, "target_fingerprint")
    assert published_concept.scope.target.capabilities == []
    assert published_concept.scope.operator.definition_ids == [
        signature.definition_id
    ]
    assert "two_stage_reduction" in published_concept.scope.operator.motifs
    assert [
        item.model_dump(mode="json")
        for item in published_concept.scope.workloads.all
    ] == [
        {"field": "reduction_size", "op": "eq", "value": 4096}
    ]
    assert published_concept.verified == []
    assert published_concept.evidence[0].stance == "supports"
    concept_path = next((catalog_root / "concepts").glob("experience--*.md"))
    rendered_concept = concept_path.read_text(encoding="utf-8")
    assert "\ntype:" not in rendered_concept
    assert "\ngenerated:" not in rendered_concept
    assert "\n  created_by: kernelgen/publisher-v1" in rendered_concept
    assert "target_fingerprint:" not in rendered_concept
    assert "\nverified:" not in rendered_concept
    assert "\nstale_after:" not in rendered_concept
    assert "\nrelations:" not in rendered_concept
    assert "\nnumerics:" not in rendered_concept
    observation_files = list(
        (catalog_root / "observations").glob("**/*.jsonl")
    )
    observation = json.loads(
        observation_files[0].read_text(encoding="utf-8")
    )
    assert observation["schema_version"] == "2.0"
    assert observation["ledger_locator"]["path"].startswith("archive://")
    assert observation["experiment_plan"]["strategy"] == (
        "measure two-stage reduction"
    )
    assert observation["code_changes"] == "implement two stages"
    assert observation["agent_conclusion"] == {}
    review_observation = review_packets[0].units[0].observations[0]
    assert review_observation.experiment_plan == observation["experiment_plan"]
    assert review_observation.code_changes == observation["code_changes"]
    assert review_observation.agent_conclusion == observation["agent_conclusion"]
    audit = json.loads(
        (catalog_root / result.audit_ref).read_text(encoding="utf-8")
    )
    assert audit["snapshot_before"] != audit["snapshot_after"]
    assert audit["reviewer_mode"] == "enforce"
    assert audit["result"]["created"] == result.created
    assert audit["result"]["observations_added"] == result.observations_added
    assert audit["application_links"] == []
    assert audit["review_decisions"][0]["decision"] == "approve"
    review_record = json.loads(
        (catalog_root / result.review_audit_ref).read_text(encoding="utf-8")
    )
    assert review_record["candidate_ids"] == result.approved_candidates
    assert review_record["reviewer_mode"] == "enforce"
    assert review_record["observation_refs"] == [observation["id"]]

    round_index = SQLiteRoundIndex(
        CatalogLayout(catalog_root).round_index
    )
    round_index.rebuild(archive_root)
    hits = round_index.search(
        "two-stage reduction",
        definition_id=signature.definition_id,
        target_backend=target.backend,
        target_architecture=target.architecture,
        target_device=target.device,
    )
    assert hits[0].target_backend == target.backend
    assert hits[0].target_architecture == target.architecture
    assert hits[0].target_device == target.device
    assert [(item.workspace_id, item.round_num) for item in hits] == [
        ("1R.agent0", 1)
    ]

    replay = publish_workspace_batch(
        catalog_root,
        archive_root,
        [workspace],
        run_id="run-1",
        batch_id="run-1-epoch-1",
    )
    assert replay.status == "noop"
    assert replay.reviewer_mode == "enforce"
    assert replay.created == []
    assert replay.updated == []
    assert replay.observations_added == []
    assert replay.approved_candidates == result.approved_candidates
    assert replay.review_audit_ref == result.review_audit_ref
    assert json.loads(
        (catalog_root / result.audit_ref).read_text(encoding="utf-8")
    )["result"]["status"] == "published"
    rebuilt = rebuild_indexes(catalog_root, archive_root)
    assert rebuilt["concepts"] == 2
    assert rebuilt["observations"] == 1
    assert Path(rebuilt["knowledge_index"]).is_file()
    assert Path(rebuilt["round_index"]).is_file()
    with pytest.raises(ValueError, match="requires git revert"):
        rollback_publish_batch(catalog_root, "run-1-epoch-1")
    assert FilesystemCatalog(catalog_root).snapshot() == audit["snapshot_after"]
    assert FilesystemCatalog(catalog_root).get_concept(result.created[0])


def test_epoch_publisher_persists_usage_without_candidate(tmp_path):
    catalog_root = tmp_path / "catalog"
    concept = _seed(catalog_root)
    workspace = tmp_path / "1R" / "agent0"
    workspace.mkdir(parents=True)
    _materialize(catalog_root, workspace)

    services = build_workspace_services(workspace)
    context = build_query_context(
        workspace,
        phase="initial",
        task="architecture_selection",
        question="reduction",
    )
    bundle = services.query.execute(context)
    concept_ref = concept.id
    services.get.execute(bundle.query_id, [concept_ref])

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
        implementation_language="triton",
    )
    _finalize_test_round(ledger, 1)
    ledger.record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.2,
            "num_workloads": 1,
            "num_passed": 1,
            "per_workload": [
                {"uuid": "w0", "status": "PASSED", "speedup": 1.2}
            ],
        },
        "knowledge-guided",
        {
            "kind": "performance",
            "strategy": "apply two-stage reduction",
            "code_changes": "split the reduction into two stages",
            "hypothesis": "two stages reduce long-reduction latency",
            "expected_effect": {
                "metric": "geo_mean",
                "direction": "increase",
                "mechanism": "reduce reduction latency",
            },
            "source": {"kind": "knowledge_base"},
            "knowledge_uses": [
                {
                    "concept_ref": concept_ref,
                    "query_event_id": bundle.query_id,
                    "role": "implementation",
                    "disposition": "adopted",
                    "application_note": "used the two-stage reduction structure",
                    "affected_parts": ["reduction"],
                }
            ],
        },
    )

    publisher = BatchPublisher(
        catalog_root,
        KernelGenRunFactReader(
            KernelGenRunArchive(tmp_path / "run-archive")
        ),
    )
    result = publisher.publish(
        [workspace],
        run_id="run-usage",
        batch_id="run-usage-epoch-1",
    )

    assert result.status == "published"
    assert result.created == []
    assert result.updated == []
    assert result.processed_candidates == []
    assert len(result.observations_added) == 1
    observations = list(
        FilesystemCatalog(catalog_root).iter_observations()
    )
    assert len(observations) == 1
    observation = observations[0]
    assert observation.round_num == 2
    assert observation.usage_mode == "single"
    assert observation.applications[0].concept_ref == concept_ref
    assert observation.applications[0].lineage_status == "complete"
    assert observation.knowledge_effect() == "performance_improved"

    ranked = services.query.execute(context)
    used = next(
        item for item in ranked.direct if item.concept_ref == concept_ref
    )
    assert used.usage_summary.considered_count == 1
    assert used.usage_summary.applied_count == 1
    assert used.usage_summary.single_success_count == 1
    assert used.usage_summary.regression_count == 0
    assert used.usage_summary.observation_refs == [observation.id]

    audit = json.loads(
        (catalog_root / result.audit_ref).read_text(encoding="utf-8")
    )
    assert audit["application_links"] == [
        {
            "observation_ref": observation.id,
            "knowledge_ref": concept_ref,
            "disposition": "adopted",
            "lineage_status": "complete",
        }
    ]
    replay = publisher.publish(
        [workspace],
        run_id="run-usage",
        batch_id="run-usage-epoch-1",
    )
    assert replay.status == "noop"
    assert replay.observations_added == []


def test_run_archive_keeps_nested_manifest_files(tmp_path):
    catalog_root = tmp_path / "catalog"
    _seed(catalog_root)
    workspace = tmp_path / "1R" / "agent0"
    workspace.mkdir(parents=True)
    signature = build_operator_signature(_DEFINITION)
    target = build_target_context(
        target_hardware="A100",
        implementation_language="triton",
        service_status=_service_status(),
    )
    KnowledgeWorkspaceMaterializer(
        catalog_root=catalog_root,
        mode=KnowledgeMode.READ_WRITE_V1,
        run_id="run-1",
        operator_signature=signature,
        target_context=target,
    ).materialize(workspace)
    ledger = Ledger(workspace)
    ledger.record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.0,
            "num_workloads": 1,
            "num_passed": 1,
            "per_workload": [{"uuid": "w0", "status": "PASSED"}],
        },
        "def kernel(): pass",
        {
            "kind": "baseline",
            "strategy": "establish the baseline",
            "code_changes": "none",
            "hypothesis": "the baseline is measurable",
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
        implementation_language="triton",
    )
    snapshot = workspace / ".kernelgen" / "evals" / "round-0001"
    snapshot.mkdir(parents=True)
    nested_manifest = snapshot / "manifest.json"
    nested_manifest.write_text(
        json.dumps({"kind": "nested-evaluation-manifest"}),
        encoding="utf-8",
    )
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint="test-round-0001",
        snapshot_path=".kernelgen/evals/round-0001",
    )
    state = WorkspaceKnowledgeState.model_validate_json(
        KnowledgeLayout(workspace).state.read_text(encoding="utf-8")
    )
    archive = KernelGenRunArchive(tmp_path / "run-archive")
    bundle = build_workspace_services(workspace).query.execute(
        build_query_context(
            workspace,
            phase="initial",
            task="architecture_selection",
            question="archive this query",
        )
    )

    first = archive.archive_workspace(
        workspace,
        state,
        signature,
        target,
        run_id="run-1",
    )
    archived_paths = {item["path"] for item in first.manifest["files"]}
    assert (
        "artifacts/.kernelgen/evals/round-0001/manifest.json"
        in archived_paths
    )
    assert "knowledge/retrieval-log.jsonl" in archived_paths
    archived_event = json.loads(
        (first.root / "knowledge" / "retrieval-log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert archived_event["event_id"] == bundle.query_id
    assert (
        archived_event["request"]["operator_signature"]["definition_id"]
        == signature.definition_id
    )
    replay = archive.archive_workspace(
        workspace,
        state,
        signature,
        target,
        run_id="run-1",
    )
    assert replay.snapshot_id == first.snapshot_id

    (first.root / "unexpected.txt").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        archive.archive_workspace(
            workspace,
            state,
            signature,
            target,
            run_id="run-1",
        )
    message = str(error.value)
    assert "missing_count=0" in message
    assert "unexpected_count=1" in message
    assert "unexpected_examples=['unexpected.txt']" in message
    assert "expected=[" not in message


def test_bridge_commits_published_batch_with_epoch_identity(
    tmp_path,
    monkeypatch,
):
    catalog_root = tmp_path / "catalog"
    _seed(catalog_root)
    workspace = tmp_path / "1R" / "agent0"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(
        knowledge_bridge_module,
        "_service_status",
        lambda _: _service_status(),
    )
    bridge = KernelGenKnowledgeBridge(
        config=KnowledgeConfig(
            mode=KnowledgeMode.READ_WRITE_V1,
            catalog_root=catalog_root,
        ),
        definition=_DEFINITION,
        target_hardware="A100",
        implementation_language="triton",
        eval_server_url="",
        run_id="run-1",
    )
    bridge.materializer().materialize(workspace)
    published = PublishResult(
        status="published",
        processed_candidates=["candidate-1"],
    )

    class FakePublisher:
        def __init__(self, catalog, fact_reader):
            assert catalog == catalog_root

        def publish(self, workspaces, **arguments):
            assert workspaces == [workspace]
            assert arguments["batch_id"] == "run-1-epoch-3"
            assert arguments["reviewer_mode"] == KnowledgeReviewerMode.OFF
            return published

    commits = []
    changed_path_snapshots = iter(
        [
            {"preexisting.txt"},
            {
                "preexisting.txt",
                "publish/audit/run-1-epoch-3.json",
            },
        ]
    )
    monkeypatch.setattr(
        knowledge_bridge_module,
        "catalog_changed_paths",
        lambda root: next(changed_path_snapshots),
    )
    monkeypatch.setattr(knowledge_bridge_module, "BatchPublisher", FakePublisher)
    monkeypatch.setattr(
        knowledge_bridge_module,
        "KernelGenRunArchive",
        lambda root: type(
            "FakeArchive",
            (),
            {"archive_workspace": lambda self, *args, **kwargs: None},
        )(),
    )
    monkeypatch.setattr(
        knowledge_bridge_module,
        "SQLiteRoundIndex",
        lambda path: type(
            "FakeRoundIndex",
            (),
            {"rebuild": lambda self, root: None},
        )(),
    )
    monkeypatch.setattr(
        knowledge_bridge_module,
        "commit_catalog",
        lambda root, message, *, paths: commits.append(
            (root, message, sorted(paths))
        ),
    )

    assert bridge.publish_epoch([workspace], epoch_num=3) == published
    assert (
        tmp_path / ".catalog.publish-transaction.lock"
    ).is_file()
    assert commits == [
        (
            catalog_root,
            "run-1-epoch-3: published 1 V1 candidates",
            ["publish/audit/run-1-epoch-3.json"],
        )
    ]


def test_bridge_serializes_complete_publish_transactions(
    tmp_path,
    monkeypatch,
):
    catalog_root = tmp_path / "catalog"
    workspace = tmp_path / "1R" / "agent0"
    state_path = KnowledgeLayout(workspace).state
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        knowledge_bridge_module,
        "_service_status",
        lambda _: _service_status(),
    )
    bridge = KernelGenKnowledgeBridge(
        config=KnowledgeConfig(
            mode=KnowledgeMode.READ_WRITE_V1,
            catalog_root=catalog_root,
        ),
        definition=_DEFINITION,
        target_hardware="A100",
        implementation_language="triton",
        eval_server_url="",
        run_id="run-1",
    )
    active = 0
    peak_active = 0
    counter_lock = threading.Lock()

    def fake_publish(self, workspaces, *, epoch_num):
        nonlocal active, peak_active
        with counter_lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.05)
        with counter_lock:
            active -= 1
        return epoch_num

    monkeypatch.setattr(
        KernelGenKnowledgeBridge,
        "_publish_epoch_locked",
        fake_publish,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda epoch: bridge.publish_epoch(
                    [workspace],
                    epoch_num=epoch,
                ),
                [1, 2],
            )
        )

    assert results == [1, 2]
    assert peak_active == 1



def test_observation_freezes_concept_and_source_use_lineage(tmp_path):
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
    build_workspace_services(workspace).get.execute(
        bundle.query_id,
        [concept.id],
    )
    plan = ExperimentPlan.model_validate(
        {
            "kind": "performance",
            "strategy": "apply retrieved reduction knowledge",
            "code_changes": "implement the retrieved reduction structure",
            "hypothesis": "the retrieved structure improves the reduction",
            "expected_effect": {
                "metric": "geo_mean",
                "direction": "increase",
                "mechanism": "reduce redundant reduction work",
            },
            "source": {"kind": "knowledge_base"},
            "knowledge_uses": [
                {
                    "concept_ref": concept.id,
                    "query_event_id": bundle.query_id,
                    "role": "hypothesis",
                    "disposition": "adopted",
                    "application_note": "used as the experiment hypothesis",
                    "affected_parts": ["reduction"],
                }
            ],
        }
    )
    validate_knowledge_uses(workspace, plan)

    forged = plan.model_copy(
        update={
            "knowledge_uses": [
                plan.knowledge_uses[0].model_copy(
                    update={"concept_ref": "kg:method:forged"}
                )
            ]
        }
    )
    validate_knowledge_uses(workspace, forged)

    ledger = Ledger(workspace)
    baseline_plan = ExperimentPlan.model_validate(
        {
            **plan.model_dump(mode="python"),
            "kind": "baseline",
            "strategy": "establish baseline",
            "code_changes": "implement baseline",
            "hypothesis": "measure baseline",
            "expected_effect": {
                "metric": "baseline",
                "direction": "establish_baseline",
                "mechanism": "measure",
            },
            "source": {"kind": "baseline"},
            "knowledge_uses": [],
        }
    )
    ledger.record_eval(
        {
            "status": "INCORRECT_NUMERICAL",
            "geo_mean": None,
            "per_workload": [
                {"uuid": "w0", "status": "INCORRECT_NUMERICAL"}
            ],
        },
        "incorrect",
        baseline_plan,
        definition_name=_DEFINITION["name"],
        target_hardware="A100",
    )
    _finalize_test_round(ledger, 1)
    concept_evaluated_at = datetime(
        2026,
        7,
        31,
        4,
        5,
        6,
        tzinfo=timezone.utc,
    )
    ledger.record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.2,
            "num_workloads": 1,
            "num_passed": 1,
            "per_workload": [
                {"uuid": "w0", "status": "PASSED", "speedup": 1.2}
            ],
        },
        "concept fix",
        plan,
        evaluated_at=concept_evaluated_at,
    )
    _finalize_test_round(ledger, 2)
    source_use = {
        "source_ref": {
            "resource": "source:triton-docs",
            "revision": "abc123",
            "locator": "language/load.md#mask",
        },
        "query_event_id": "source-query:" + "b" * 32,
        "role": "constraint",
        "disposition": "adopted",
        "application_note": "preserve documented mask semantics",
        "affected_parts": ["boundary_mask"],
    }
    source_plan = ExperimentPlan.model_validate(
        {
            **plan.model_dump(mode="python"),
            "knowledge_uses": [source_use],
        }
    )
    ledger.record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.5,
            "num_workloads": 1,
            "num_passed": 1,
            "per_workload": [
                {"uuid": "w0", "status": "PASSED", "speedup": 1.5}
            ],
        },
        "source improvement",
        source_plan,
    )
    _finalize_test_round(ledger, 3)
    combined_plan = ExperimentPlan.model_validate(
        {
            **plan.model_dump(mode="python"),
            "knowledge_uses": [
                plan.knowledge_uses[0].model_dump(mode="python"),
                source_use,
            ],
        }
    )
    ledger.record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.3,
            "num_workloads": 1,
            "num_passed": 1,
            "per_workload": [
                {"uuid": "w0", "status": "PASSED", "speedup": 1.3}
            ],
        },
        "combined regression",
        combined_plan,
    )
    _finalize_test_round(ledger, 4)
    ledger.record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.4,
            "num_workloads": 1,
            "num_passed": 1,
            "per_workload": [
                {"uuid": "w0", "status": "PASSED", "speedup": 1.4}
            ],
        },
        "rejected knowledge",
        source_plan,
    )
    _finalize_test_round(ledger, 5)
    raw_history = json.loads(ledger.path.read_text(encoding="utf-8"))
    historical_use = raw_history["rounds"][-1]["plan"]["knowledge_uses"][0]
    historical_use["disposition"] = "rejected"
    historical_use["application_note"] = "not applicable to this code path"
    ledger.path.write_text(json.dumps(raw_history), encoding="utf-8")
    ledger = Ledger(workspace)
    unread_bundle = build_workspace_services(workspace).query.execute(context)
    unread_plan = ExperimentPlan.model_validate(
        {
            **plan.model_dump(mode="python"),
            "knowledge_uses": [
                plan.knowledge_uses[0].model_copy(
                    update={"query_event_id": unread_bundle.query_id}
                ).model_dump(mode="python")
            ],
        }
    )
    ledger.record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.4,
            "num_workloads": 1,
            "num_passed": 1,
            "per_workload": [
                {"uuid": "w0", "status": "PASSED", "speedup": 1.4}
            ],
        },
        "query result not read in detail",
        unread_plan,
    )

    state = WorkspaceKnowledgeState.model_validate_json(
        KnowledgeLayout(workspace).state.read_text(encoding="utf-8")
    )
    signature = build_operator_signature(_DEFINITION)
    target = build_target_context(
        target_hardware="A100",
        implementation_language="triton",
        service_status=_service_status(),
    )
    reader = KernelGenRunFactReader(
        KernelGenRunArchive(catalog_root.parent / "run-archive")
    )
    concept_observation = reader.materialize_observation(
        workspace,
        state,
        signature,
        target,
        round_num=2,
        run_id="run-1",
    )
    assert concept_observation.evaluated_at == concept_evaluated_at
    assert concept_observation.target_context_ref.startswith("archive://")
    assert concept_observation.operator_signature_ref.startswith("archive://")
    assert concept_observation.experiment_parent.round_num == 1
    assert concept_observation.experiment_parent.status == "INCORRECT_NUMERICAL"
    assert concept_observation.comparison.performance_baseline_round_num is None
    assert concept_observation.usage_mode == "single"
    assert concept_observation.applications[0].concept_ref == concept.id
    assert concept_observation.applications[0].lineage_status == "complete"
    assert concept_observation.knowledge_effect() == "correctness_recovered"

    source_observation = reader.materialize_observation(
        workspace,
        state,
        signature,
        target,
        round_num=3,
        run_id="run-1",
    )
    assert source_observation.usage_mode == "single"
    assert source_observation.applications[0].source_ref.locator == (
        "language/load.md#mask"
    )
    assert source_observation.applications[0].lineage_status == "query_missing"
    assert (
        source_observation.comparison.performance_baseline_round_num == 2
    )
    assert source_observation.knowledge_effect() == "performance_improved"
    no_change = source_observation.model_copy(
        update={
            "comparison": source_observation.comparison.model_copy(
                update={"geo_mean_delta_pct": 0.0}
            )
        }
    )
    assert no_change.knowledge_effect() == "no_material_change"
    assert (
        source_observation.knowledge_effect(materiality_pct=30.0)
        == "no_material_change"
    )
    correctness_regression = source_observation.model_copy(
        update={
            "outcome": source_observation.outcome.model_copy(
                update={"status": "RUNTIME_ERROR"}
            )
        }
    )
    assert (
        correctness_regression.knowledge_effect()
        == "correctness_regressed"
    )

    combined_observation = reader.materialize_observation(
        workspace,
        state,
        signature,
        target,
        round_num=4,
        run_id="run-1",
    )
    assert combined_observation.usage_mode == "combined"
    assert combined_observation.knowledge_effect() == "performance_regressed"

    rejected_observation = reader.materialize_observation(
        workspace,
        state,
        signature,
        target,
        round_num=5,
        run_id="run-1",
    )
    assert rejected_observation.usage_mode == "none"
    assert rejected_observation.knowledge_effect() == "unclassified"

    unread_observation = reader.materialize_observation(
        workspace,
        state,
        signature,
        target,
        round_num=6,
        run_id="run-1",
    )
    assert unread_observation.applications[0].lineage_status == "detail_not_read"

    cold = concept.model_copy(
        update={
            "id": "kg:method:aaa-cold-reduction",
            "claim_key": "method.reduction.cold",
            "title": "Cold two-stage reduction",
        }
    )
    cold = cold.model_copy(
        update={
            "managed": cold.managed.model_copy(
                update={"content_hash": canonical_concept_hash(cold)}
            )
        }
    )
    FilesystemCatalog(catalog_root).write_concept(cold)
    other_scope = concept_observation.evaluation_scope.model_copy(
        update={"target_device": "H100"}
    )
    cross_scope = concept_observation.model_copy(
        update={
            "id": "kg:observation:other:other:round-2",
            "run_id": "other",
            "workspace_id": "other",
            "evaluation_scope": other_scope,
            "applications": [
                concept_observation.applications[0].model_copy(
                    update={"concept_ref": cold.id}
                )
            ],
        }
    )
    observation_path = (
        catalog_root / "observations" / "by_run" / "run-1.jsonl"
    )
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text(
        "".join(
            item.model_dump_json() + "\n"
            for item in (
                concept_observation,
                source_observation,
                combined_observation,
                rejected_observation,
                cross_scope,
            )
        ),
        encoding="utf-8",
    )

    ranked = build_workspace_services(workspace).query.execute(context)
    ranked_refs = [item.concept_ref for item in ranked.direct]
    assert ranked_refs.index(concept.id) < ranked_refs.index(
        cold.id
    )
    used = next(
        item
        for item in ranked.direct
        if item.concept_ref == concept.id
    )
    assert used.usage_summary.considered_count == 2
    assert used.usage_summary.retrieved_count == 1
    assert used.usage_summary.applied_count == 2
    assert used.usage_summary.single_success_count == 1
    assert used.usage_summary.combined_success_count == 0
    assert used.usage_summary.correctness_recovery_count == 1
    assert used.usage_summary.regression_count == 1
    assert used.usage_summary.observation_refs == sorted(
        [concept_observation.id, combined_observation.id]
    )
    assert "same-scope single successes: 1" in used.recommendation_reasons
    assert "same-scope regressions: 1" in used.recommendation_reasons
    cold_result = next(
        item
        for item in ranked.direct
        if item.concept_ref == cold.id
    )
    assert cold_result.usage_summary.evaluated_count == 0
    assert (
        "no same-scope evaluated use yet"
        in cold_result.recommendation_reasons
    )
    assert FilesystemCatalog(catalog_root).get_concept(
        concept.id
    ) == concept
    shutil.rmtree(CatalogLayout(catalog_root).derived)
    rebuilt = build_workspace_services(workspace).query.execute(context)
    rebuilt_used = next(
        item
        for item in rebuilt.direct
        if item.concept_ref == concept.id
    )
    assert rebuilt_used.usage_summary == used.usage_summary


def _retrieval(
    *,
    event_id: str,
    operation: str,
    query_id: str,
    returned_refs: list[str] | None = None,
    returned_sources: list[str] | None = None,
) -> RetrievalRecord:
    return RetrievalRecord(
        event_id=event_id,
        operation=operation,
        origin="mcp",
        status="success",
        query_id=query_id,
        returned_refs=returned_refs or [],
        returned_sources=returned_sources or [],
        created_at=datetime.now(timezone.utc),
    )


def test_concept_lineage_distinguishes_missing_query_detail_and_complete(
    tmp_path,
):
    audit = FileRetrievalAudit(tmp_path / "retrieval.jsonl")
    application = KnowledgeApplication(
        concept_ref="kg:method:reduction-test",
        query_event_id="query:round-2",
        role="implementation",
        disposition="adopted",
        application_note="selected a two-stage reduction",
        affected_parts=["strategy"],
    )

    assert _lineage_status(application, audit, audit.read_all()) == "query_missing"

    audit.append(
        _retrieval(
            event_id="query:round-2",
            operation="query_knowledge",
            query_id="query:round-2",
            returned_refs=["kg:method:reduction-test"],
        )
    )
    assert (
        _lineage_status(application, audit, audit.read_all())
        == "detail_not_read"
    )

    audit.append(
        _retrieval(
            event_id="retrieval:round-2",
            operation="get_knowledge",
            query_id="query:round-2",
            returned_refs=["kg:method:reduction-test"],
        )
    )
    assert _lineage_status(application, audit, audit.read_all()) == "complete"


def test_source_lineage_requires_the_exact_revision_and_locator(tmp_path):
    audit = FileRetrievalAudit(tmp_path / "retrieval.jsonl")
    application = KnowledgeApplication(
        source_ref={
            "resource": "source:triton-ascend",
            "revision": "abc123",
            "locator": "docs/reduction.md#L20-L40",
        },
        query_event_id="source-query:round-2",
        role="implementation",
        disposition="adapted",
        application_note="adapted the documented reduction pattern",
        affected_parts=["code_changes"],
    )
    audit.append(
        _retrieval(
            event_id="source-query:round-2",
            operation="query_sources",
            query_id="source-query:round-2",
        )
    )
    audit.append(
        _retrieval(
            event_id="retrieval:wrong-fragment",
            operation="get_source",
            query_id="source-query:round-2",
            returned_sources=[
                "source:triton-ascend@abc123::docs/reduction.md#L1-L10"
            ],
        )
    )

    assert (
        _lineage_status(application, audit, audit.read_all())
        == "detail_not_read"
    )

    audit.append(
        _retrieval(
            event_id="retrieval:right-fragment",
            operation="get_source",
            query_id="source-query:round-2",
            returned_sources=[
                (
                    "source:triton-ascend@abc123::"
                    "docs/reduction.md#L20-L40"
                )
            ],
        )
    )
    assert _lineage_status(application, audit, audit.read_all()) == "complete"

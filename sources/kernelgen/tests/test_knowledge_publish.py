"""Knowledge publication and lifecycle behavior tests."""

from __future__ import annotations

import subprocess

import pytest

from kernelgen.knowledge.catalog import (
    assert_catalog_paths_clean,
    catalog_changed_paths,
    commit_catalog,
)
from kernelgen.knowledge.publishing.materialize import (
    _canonicalize_candidate_symptoms,
)
from kernelgen.knowledge.vocabulary import canonical_failure_symptom
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


def test_publisher_updates_exact_target_without_new_concept(tmp_path):
    catalog_root = tmp_path / "catalog"
    original = _seed(catalog_root)
    candidate = _lifecycle_candidate(
        original,
        candidate_id="update-1",
        claim_key="method.reduction.reworded",
        title="Updated reduction",
        body="## Claim\n\nUse the revised reduction rule.",
        publish_action="update",
        target_concept_id=original.id,
    )

    changed, created, updated, contested, rejected = merge_concepts(
        FilesystemCatalog(catalog_root),
        [(candidate, [])],
        {},
    )

    assert not created
    assert not contested
    assert not rejected
    assert updated == [original.id]
    assert {item.id for item in changed} == {original.id}
    revised = FilesystemCatalog(catalog_root).get_concept(original.id)
    assert revised.claim_key == original.claim_key
    assert revised.title == "Updated reduction"
    assert revised.body.endswith("revised reduction rule.")


def test_publisher_supersedes_and_deprecates_previous_concept(tmp_path):
    catalog_root = tmp_path / "catalog"
    original = _seed(catalog_root)
    candidate = _lifecycle_candidate(
        original,
        candidate_id="create-1",
        claim_key="method.reduction.replacement",
        title="Replacement reduction",
        body="## Claim\n\nUse the replacement rule.",
        relations=[
            ConceptRelation(type="supersedes", target=original.id),
        ],
    )

    changed, created, updated, contested, rejected = merge_concepts(
        FilesystemCatalog(catalog_root),
        [(candidate, [])],
        {},
    )

    assert created == ["kg:method:method-reduction-replacement"]
    assert updated == [original.id]
    assert not contested
    assert not rejected
    assert {item.id for item in changed} == {
        original.id,
        "kg:method:method-reduction-replacement",
    }
    assert FilesystemCatalog(catalog_root).get_concept(original.id).status == "deprecated"


def test_concept_maintenance_deprecate_and_delete(tmp_path):
    catalog_root = tmp_path / "catalog"
    original = _seed(catalog_root)

    deprecated = deprecate_concept(
        catalog_root,
        original.id,
        reason="replaced by a narrower claim",
    )
    assert deprecated["status"] == "deprecated"
    assert FilesystemCatalog(catalog_root).get_concept(original.id).status == "deprecated"

    planned = delete_concept(
        catalog_root,
        original.id,
        reason="empty deprecated shell is no longer needed",
    )
    assert planned["status"] == "planned"
    assert Path(catalog_root / planned["path"]).is_file()

    deleted = delete_concept(
        catalog_root,
        original.id,
        reason="empty deprecated shell is no longer needed",
        apply=True,
    )
    assert deleted["status"] == "deleted"
    assert not Path(catalog_root / planned["path"]).exists()



def test_candidate_draft_gets_workspace_authority_fields(tmp_path):
    catalog_root = tmp_path / "catalog"
    _seed(catalog_root)
    workspace = tmp_path / "agent0"
    workspace.mkdir()
    _materialize(catalog_root, workspace)
    draft = CandidateDraft(
        proposed_kind="experience",
        claim_key="experience.sum_rows.a100",
        title="A100 sum rows",
        summary="Two stages were measured for this definition.",
        domains=["optimization"],
        scope_hints={"motifs": ["two_stage_reduction"]},
        body="# Claim\n\nTwo stages helped.",
        observation_intents=[
            {
                "round_num": 1,
                "stance": "supports",
                "rationale": "The round directly tests the claim.",
            }
        ],
    )

    ids = append_candidate_drafts(workspace, [draft])
    stored = build_workspace_services(workspace).outbox.read_all()

    assert len(ids) == 1
    assert "base_snapshot" not in stored[0].model_dump()
    assert "scope" not in stored[0].model_dump()
    assert stored[0].scope_hints.motifs == ["two_stage_reduction"]
    assert stored[0].created_by == "agent0"
    assert stored[0].candidate_id == ids[0]
    assert ids[0].endswith("candidate-0001")


def test_publish_is_not_blocked_by_unrelated_catalog_validation_issue(tmp_path):
    catalog_root = tmp_path / "catalog"
    stale = _seed(catalog_root)
    stale = stale.model_copy(
        update={
            "relations": [
                ConceptRelation(
                    type="related",
                    target="kg:diagnostic:not-imported",
                )
            ]
        }
    )
    stale = stale.model_copy(
        update={
            "managed": stale.managed.model_copy(
                update={"content_hash": canonical_concept_hash(stale)}
            )
        }
    )
    FilesystemCatalog(catalog_root).write_concept(stale)
    candidate = CandidateConcept(
        candidate_id="static:new-reference",
        proposed_kind="reference",
        proposed_id="kg:reference:new-rule",
        claim_key="reference.new.rule",
        title="New rule",
        summary="A new independently valid rule.",
        domains=["language"],
        scope=Scope(
            target=TargetScope(level="portable"),
            numerics={"exact": False},
        ),
        retrieval={
            "phases": ["initial"],
            "tasks": ["constraint_check"],
        },
        body=_reference_body("Use the new rule."),
        source_refs=[
            SourceReference(
                resource="https://example.invalid/new-rule",
                revision="2026-01-01",
                locator="https://example.invalid/new-rule",
                title="New rule",
            )
        ],
        created_by="static-test",
    )

    result = BatchPublisher(catalog_root, fact_reader=None).publish_static(
        [candidate],
        batch_id="static-new-rule",
    )

    assert result.status == "published"
    assert result.created == ["kg:reference:new-rule"]
    published = FilesystemCatalog(catalog_root).get_concept(result.created[0])
    assert published.scope.numerics.exact is None
    assert "exact: false" not in (
        catalog_root / "concepts" / "reference--new-rule.md"
    ).read_text(encoding="utf-8")
    report = validate_knowledge_base(catalog_root)
    assert report.valid is False
    assert any(
        "relation target does not exist" in issue.message
        for issue in report.issues
    )


def test_evidence_state_is_derived_from_observations():
    def evidence(workspace, round_num, stance):
        return ConceptEvidence(
            observation_ref=(
                f"kg:observation:run-1:{workspace}:round-{round_num}"
            ),
            stance=stance,
            rationale="Measured outcome.",
        )

    first = evidence("agent-a", 1, "supports")
    second = evidence("agent-b", 1, "supports")
    refute_a = evidence("agent-a", 2, "refutes")
    refute_b = evidence("agent-b", 2, "refutes")
    regression_support = ConceptEvidence(
        observation_ref="kg:observation:run-2:agent-a:round-2",
        stance="supports",
        rationale=(
            "The optimization hypothesis failed, which supports the negative "
            "Concept claim that this change regresses."
        ),
    )

    assert _evidence_state([first], []) == "observed"
    assert _evidence_state([first, second], []) == "corroborated"
    assert _evidence_state([first, refute_a], []) == "contested"
    assert _evidence_state([refute_a, refute_b], []) == "falsified"
    assert _evidence_state([regression_support], []) == "observed"
    assert _evidence_state([], [object()]) == "source_supported"


def test_publisher_rejects_new_relation_to_missing_concept(tmp_path):
    catalog_root = tmp_path / "catalog"
    candidate = CandidateConcept(
        candidate_id="static:invalid-relation",
        proposed_kind="reference",
        proposed_id="kg:reference:invalid-relation",
        claim_key="reference.invalid.relation",
        title="Invalid relation",
        summary="A missing relation target must reject this candidate.",
        domains=["language"],
        scope=Scope(target=TargetScope(level="portable")),
        retrieval={
            "phases": ["initial"],
            "tasks": ["constraint_check"],
        },
        body=_reference_body("This candidate must not publish."),
        relations=[
            ConceptRelation(
                type="requires",
                target="kg:method:not-present",
            )
        ],
        source_refs=[
            SourceReference(
                resource="https://example.invalid/rule",
                revision="2026-01-01",
                locator="https://example.invalid/rule",
                title="Rule",
            )
        ],
        created_by="source-ingest:test",
    )

    result = BatchPublisher(catalog_root, fact_reader=None).publish_static(
        [candidate],
        batch_id="invalid-relation",
    )

    assert result.status == "noop"
    assert result.created == []
    assert result.rejected == [
        "static:invalid-relation: relation targets do not exist: "
        "kg:method:not-present"
    ]



def test_revalidation_list_reports_environment_version_changes(tmp_path):
    catalog_root = tmp_path / "catalog"
    original = _seed(catalog_root)
    revised = original.model_copy(
        update={
            "scope": Scope(
                target=TargetScope(
                    level="device",
                    devices=["910B4-1"],
                    software={
                        "language": "triton",
                        "compiler": "triton-ascend",
                        "compiler_version": "3.2.0",
                        "runtime": "cann",
                        "runtime_version": "8.5.0",
                    },
                )
            ),
        }
    )
    revised = revised.model_copy(
        update={
            "managed": revised.managed.model_copy(
                update={"content_hash": canonical_concept_hash(revised)}
            )
        }
    )
    FilesystemCatalog(catalog_root).write_concept(revised)
    target = build_target_context(
        target_hardware="Ascend910B",
        implementation_language="triton",
        service_status={
            "backend": "ascend",
            "target": {"device": "Ascend 910B"},
            "software": {
                "language": "triton",
                "compiler": "triton-ascend",
                "compiler_version": "3.3.0",
                "runtime": "cann",
                "runtime_version": "8.5.0",
            },
        },
    )
    assert revalidation_candidates(catalog_root, target) == [
        {
            "concept_ref": revised.id,
            "changes": [
                {
                    "field": "compiler_version",
                    "recorded": "3.2.0",
                    "current": "3.3.0",
                }
            ],
        }
    ]
    assert FilesystemCatalog(catalog_root).get_concept(
        revised.id
    ) == revised


def test_static_source_ingestion_reuses_publisher_and_is_idempotent(
    tmp_path,
    monkeypatch,
):
    kb = tmp_path / "kb"
    content = kb / "sources" / "content" / "example" / "abc123"
    document = content / "docs" / "guide.md"
    document.parent.mkdir(parents=True)
    document.write_text(
        "# Guide\n\n"
        "## Stable Rule\n\nUse stable behavior.\n\n"
        "```python\n# This is a code comment, not a heading.\n```\n\n"
        "## Other Rule\n\nDo something else.\n",
        encoding="utf-8",
    )
    (content / "guide-link.md").symlink_to("docs/guide.md")
    package = kb / "sources" / "packages" / "example-abc123.yaml"
    package.parent.mkdir(parents=True)
    package.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "id": "source:example.abc123",
                "source_type": "repository",
                "origin": "https://example.invalid/repository.git",
                "revision": "abc123",
                "content_root": "kb://sources/content/example/abc123",
                "license": "test-only",
                "checksum": source_content_checksum(content),
                "retrieved_at": "2026-07-27T00:00:00Z",
                "authority": "official",
                "allowed_hosts": ["example.invalid"],
            }
        ),
        encoding="utf-8",
    )
    plan = kb / "sources" / "ingestion" / "example.yaml"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "source_package": "source:example.abc123",
                "rules": [
                    {
                        "include": "docs/*.md",
                        "mode": "direct",
                        "kinds": ["reference"],
                    }
                ],
                "entries": [
                    {
                        "source_path": "docs/guide.md",
                        "heading": "Stable Rule",
                        "mode": "direct",
                        "proposed_kind": "reference",
                        "proposed_id": "kg:reference:example-stable-rule",
                        "claim_key": "reference.example.stable_rule",
                        "title": "Example stable rule",
                        "summary": "Use the documented stable behavior.",
                        "domains": ["language"],
                        "scope": {
                            "target": {
                                "level": "portable",
                                "software": {"language": "example"},
                            }
                        },
                        "retrieval": {
                            "phases": ["initial"],
                            "tasks": ["constraint_check"],
                            "keywords": ["stable", "behavior"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    importer = StaticKnowledgeImporter.from_plan_file(kb, plan)
    candidates = importer.build_candidates()
    assert "## Stable Rule" in candidates[0].body
    assert "# This is a code comment" in candidates[0].body
    assert "## Other Rule" not in candidates[0].body

    result = importer.publish(
        batch_id="static-example-abc123",
    )
    assert result.status == "published"
    assert result.created == ["kg:reference:example-stable-rule"]
    assert result.observations_added == []
    assert validate_knowledge_base(kb).valid
    audit = json.loads((kb / result.audit_ref).read_text(encoding="utf-8"))
    assert audit["snapshot_before"] != audit["snapshot_after"]
    assert audit["application_links"] == []

    concept = FilesystemCatalog(kb).get_concept(
        "kg:reference:example-stable-rule"
    )
    original = concept
    assert concept.evidence_state == "source_supported"
    assert concept.sources[0].revision == "abc123"
    assert concept.sources[0].locator == "docs/guide.md#Stable Rule"
    replay = importer.publish(
        batch_id="static-example-abc123",
    )
    assert replay.status == "noop"
    assert replay.created == []
    assert replay.updated == []

    refreshed_candidate = candidates[0].model_copy(
        update={
            "candidate_id": "static:example-def456",
            "source_refs": [
                candidates[0].source_refs[0].model_copy(
                    update={"revision": "def456"}
                )
            ],
        }
    )
    refresh_result = BatchPublisher(kb, fact_reader=None).publish_static(
        [refreshed_candidate],
        batch_id="static-example-def456",
    )
    assert refresh_result.updated == ["kg:reference:example-stable-rule"]
    refreshed = FilesystemCatalog(kb).get_concept(
        "kg:reference:example-stable-rule"
    )
    assert [source.revision for source in refreshed.sources] == ["def456"]

    updated_candidate = refreshed_candidate.model_copy(
        update={
            "candidate_id": "static:updated-example",
            "summary": "Use the revised documented stable behavior.",
            "body": refreshed_candidate.body.replace(
                "Use the documented stable behavior.",
                "Use the revised documented stable behavior.",
            ),
            "scope": Scope(
                target=TargetScope(level="backend", backend="ascend")
            ),
        }
    )
    update_result = BatchPublisher(kb, fact_reader=None).publish_static(
        [updated_candidate],
        batch_id="static-example-abc123-plan-2",
    )
    assert update_result.updated == ["kg:reference:example-stable-rule"]
    updated = FilesystemCatalog(kb).get_concept(
        "kg:reference:example-stable-rule"
    )
    assert original.summary == "Use the documented stable behavior."
    assert updated.summary == "Use the revised documented stable behavior."
    assert updated.scope.target.backend == "ascend"
    assert [source.revision for source in updated.sources] == ["def456"]
    assert FilesystemCatalog(kb).iter_concepts() == [updated]

    failed_candidate = updated_candidate.model_copy(
        update={
            "candidate_id": "static:failed-example",
            "summary": "This revision must roll back.",
        }
    )

    def fail_rebuild(*_args, **_kwargs):
        raise RuntimeError("index rebuild failed")

    monkeypatch.setattr(
        "kernelgen.knowledge.publishing.publisher."
        "SQLiteKnowledgeIndex.rebuild",
        fail_rebuild,
    )
    with pytest.raises(RuntimeError, match="index rebuild failed"):
        BatchPublisher(kb, fact_reader=None).publish_static(
            [failed_candidate],
            batch_id="static-example-abc123-plan-3",
        )
    assert FilesystemCatalog(kb).get_concept(
        "kg:reference:example-stable-rule"
    ) == updated
    monkeypatch.undo()
    with pytest.raises(ValueError, match="requires git revert"):
        rollback_publish_batch(kb, "static-example-abc123-plan-2")
    assert FilesystemCatalog(kb).get_concept(
        "kg:reference:example-stable-rule"
    ) == updated
    assert original != updated


def test_partial_pass_maps_to_canonical_correctness_symptom(tmp_path):
    assert canonical_failure_symptom("PARTIAL_PASS") == "correctness_failure"
    original = _seed(tmp_path / "catalog")
    candidate = _lifecycle_candidate(
        original,
        candidate_id="legacy-partial-pass",
        claim_key="method.reduction.legacy-partial-pass",
        title="Legacy partial pass",
        body="## Claim\n\nNormalize the legacy evaluator status.",
    )
    candidate = candidate.model_copy(
        update={
            "retrieval": candidate.retrieval.model_copy(
                update={"symptoms": ["PARTIAL_PASS", "compile_error"]}
            )
        }
    )
    assert _canonicalize_candidate_symptoms(candidate).retrieval.symptoms == [
        "compile_error",
        "correctness_failure",
    ]


def test_commit_catalog_preserves_unrelated_dirty_paths(tmp_path):
    catalog = tmp_path / "catalog"
    catalog.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=str(catalog),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init")
    (catalog / "published.txt").write_text("old\n", encoding="utf-8")
    (catalog / "unrelated.txt").write_text("old\n", encoding="utf-8")
    git("add", "published.txt", "unrelated.txt")
    git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "seed",
    )

    (catalog / "published.txt").write_text("new\n", encoding="utf-8")
    (catalog / "unrelated.txt").write_text("dirty\n", encoding="utf-8")
    (catalog / "transaction-new.txt").write_text("new\n", encoding="utf-8")
    (catalog / "unrelated-untracked.txt").write_text(
        "dirty\n",
        encoding="utf-8",
    )
    git("add", "unrelated.txt")

    with pytest.raises(
        RuntimeError,
        match="pre-existing dirty paths: unrelated.txt",
    ):
        assert_catalog_paths_clean(catalog, {catalog / "unrelated.txt"})

    assert catalog_changed_paths(catalog) == {
        "published.txt",
        "transaction-new.txt",
        "unrelated-untracked.txt",
        "unrelated.txt",
    }
    assert commit_catalog(
        catalog,
        "publish transaction",
        paths={"published.txt", "transaction-new.txt"},
    )

    committed = set(
        git("show", "--format=", "--name-only", "HEAD").splitlines()
    )
    assert committed == {"published.txt", "transaction-new.txt"}
    status = git("status", "--short")
    assert "unrelated.txt" in status
    assert "unrelated-untracked.txt" in status
    assert "published.txt" not in status
    assert "transaction-new.txt" not in status

"""Host-only tests for KernelGen KB V1 machine contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kernelgen.knowledge.models import (
    CandidateConcept,
    Concept,
    ObservationRecord,
    KnowledgeBundle,
    KnowledgeApplication,
    OperatorSignature,
    ObservationIntent,
    QueryContext,
    SourcePackage,
    SourceReference,
    StaticIngestionPlan,
    TargetContext,
)
from kernelgen.knowledge.validation import (
    canonical_concept_hash,
    parse_concept,
    source_content_checksum,
    validate_knowledge_base,
)
from kernelgen.knowledge.source_rules import source_path_matches
from kernelgen.scripts.dev.export_knowledge_schemas import export_schemas


_HEX_A = "a" * 64
_HEX_B = "b" * 64
_NOW = datetime(2026, 7, 26, 10, 0, tzinfo=timezone.utc)


def _target(**overrides):
    value = {
        "schema_version": "1.0",
        "backend": "cuda",
        "vendor": "nvidia",
        "architecture": "sm80",
        "device": "NVIDIA-A100-SXM4-80GB",
        "capabilities": ["tensor_core"],
        "software": {
            "language": "triton",
            "language_version": "3.3.1",
            "compiler": "triton",
            "compiler_version": "3.3.1",
            "runtime": "cuda",
            "runtime_version": "12.6",
        },
        "source": "fixture",
    }
    value.update(overrides)
    return value


def _signature():
    return {
        "schema_version": "1.0",
        "definition_id": "flashinfer:rmsnorm:v1",
        "definition_name": "rmsnorm",
        "op_type": "normalization",
        "motifs": ["reduction", "elementwise"],
        "dataflow": ["reduce_then_broadcast"],
        "dtypes": ["float16"],
        "layouts": [],
        "workload_features": {"reduction_size": 8192},
    }


def _scope(level="backend", **target_overrides):
    target = {"level": level, "backend": "cuda", **target_overrides}
    if level == "portable":
        target.pop("backend", None)
    return {
        "target": target,
        "operator": {
            "definition_ids": [],
            "op_types": ["normalization"],
            "motifs": ["reduction"],
            "dataflow": ["reduce_then_broadcast"],
            "dtypes": ["float16"],
            "layouts": [],
        },
        "workloads": {
            "all": [{"field": "reduction_size", "op": "gte", "value": 4096}],
            "any": [],
            "excludes": [],
        },
        "numerics": {"exact": True},
    }


def _observation():
    return {
        "schema_version": "2.0",
        "id": "kg:observation:run-1:agent-1:round-1",
        "run_id": "run-1",
        "workspace_id": "agent-1",
        "definition_id": "flashinfer:rmsnorm:v1",
        "round_num": 1,
        "target_context_ref": ".kernelgen/knowledge/target_context.json",
        "operator_signature_ref": ".kernelgen/knowledge/operator_signature.json",
        "ledger_locator": {"path": ".ledger.json", "round_num": 1},
        "solution_sha256": _HEX_A,
        "outcome": {
            "status": "passed",
            "geo_mean_speedup": 1.23,
            "workload_results_ref": f"artifact://sha256/{_HEX_A}",
        },
        "experiment_plan": {"strategy": "fixture reduction"},
        "code_changes": "apply the fixture reduction",
        "agent_conclusion": {"expectation_status": "met"},
        "artifact_refs": [],
        "recorded_at": _NOW.isoformat(),
    }


def _concept_payload(kind="experience"):
    evidence = [
        {
            "observation_ref": (
                "kg:observation:run-1:agent-1:round-1"
            ),
            "stance": "supports",
            "rationale": "The measured round directly tested the claim.",
            "confidence": "high",
        }
    ]
    sources = []
    evidence_state = "observed"
    if kind == "reference":
        evidence = []
        evidence_state = "source_supported"
        sources = [
            {
                "id": "triton-docs",
                "resource": "source:triton-docs",
                "title": "Triton documentation",
                "author": "team:triton",
            }
        ]
    return {
        "schema_version": "1.0",
        "id": f"kg:{kind}:reduction-test",
        "kind": kind,
        "title": "Reduction test",
        "summary": "One atomic reduction claim.",
        "claim_key": f"{kind}.reduction.test",
        "domains": ["operator"],
        "status": "stable",
        "verified": [{"by": "process:benchmark-validator", "at": _NOW.isoformat()}],
        "stale_after": None,
        "sources": sources,
        "scope": _scope(),
        "retrieval": {
            "phases": ["initial"],
            "tasks": ["architecture_selection"],
            "symptoms": [],
            "techniques": ["two_stage_reduction"],
            "keywords": ["reduction"],
        },
        "evidence_state": evidence_state,
        "evidence": evidence,
        "relations": [],
        "managed": {
            "content_hash": f"sha256:{_HEX_A}",
            "created_by": "kernelgen/publisher-v1",
            "created_at": _NOW.isoformat(),
            "updated_at": _NOW.isoformat(),
        },
        "body": "# Claim\n\nOne claim.",
    }


def _concept(kind="experience"):
    return Concept.model_validate(_concept_payload(kind))


def test_observation_intent_uses_claim_stance_and_reads_legacy_stance():
    current = ObservationIntent.model_validate(
        {
            "round_num": 2,
            "claim_stance": "supports",
            "rationale": "The regression supports the negative Concept claim.",
        }
    )
    legacy = ObservationIntent.model_validate(
        {
            "round_num": 3,
            "stance": "refutes",
            "rationale": "The observation contradicts the Concept claim.",
        }
    )

    assert current.claim_stance == "supports"
    assert current.stance == "supports"
    assert legacy.claim_stance == "refutes"
    assert legacy.stance == "refutes"
    dumped = current.model_dump(mode="json")
    assert dumped["claim_stance"] == "supports"
    assert "stance" not in dumped


def test_concept_kind_and_support_are_cross_validated():
    assert _concept().kind == "experience"
    wrong = _concept_payload()
    wrong["type"] = "Method"
    with pytest.raises(ValidationError, match="type and kind do not match"):
        Concept.model_validate(wrong)

    unsupported = _concept_payload()
    unsupported["evidence"] = []
    with pytest.raises(ValidationError, match="experience requires"):
        Concept.model_validate(unsupported)


def test_concept_reads_legacy_type_and_generated_without_reemitting_them():
    legacy = _concept_payload()
    legacy["revision"] = 7
    legacy["type"] = "Experience"
    legacy["generated"] = {
        "by": "kernelgen/legacy-publisher",
        "at": _NOW.isoformat(),
    }
    legacy["managed"].pop("created_by")

    concept = Concept.model_validate(legacy)
    emitted = concept.model_dump(mode="json")

    assert concept.kind == "experience"
    assert concept.managed.created_by == "kernelgen/legacy-publisher"
    assert "type" not in emitted
    assert "generated" not in emitted
    assert "revision" not in emitted


def test_legacy_concept_application_ref_normalizes_to_stable_id():
    application = KnowledgeApplication.model_validate(
        {
            "concept_ref": "kg:method:two-stage-reduction@4",
            "role": "implementation",
            "disposition": "adopted",
        }
    )

    assert application.concept_ref == "kg:method:two-stage-reduction"
    assert "@4" not in application.model_dump_json()


def test_source_rule_globstar_is_recursive_and_segment_aware():
    assert source_path_matches("ops/foo/SKILL.md", "ops/**/*.md")
    assert source_path_matches(
        "ops/foo/references/modules/deep/guide.md",
        "ops/**/*.md",
    )
    assert source_path_matches(
        "ops/foo/evals/cases/evals.md",
        "ops/**/evals/**/*.md",
    )
    assert not source_path_matches(
        "plugins-official/foo/SKILL.md",
        "ops/**/*.md",
    )
    assert source_path_matches("deep/nested/file.py", "*")


def test_target_scope_is_advisory_until_runtime_binding():
    incomplete = _concept_payload()
    incomplete["scope"]["target"] = {"level": "backend"}
    assert Concept.model_validate(incomplete).scope.target.level == "backend"
    portable = _concept_payload()
    portable["scope"]["target"] = {"level": "portable"}
    assert Concept.model_validate(portable).scope.target.level == "portable"

    tagged = _concept_payload()
    tagged["scope"]["target"] = {"level": "portable", "backend": "cuda"}
    assert Concept.model_validate(tagged).scope.target.backend == "cuda"


def test_workload_predicates_are_non_executable_and_shape_checked():
    invalid = _concept_payload()
    invalid["scope"]["workloads"]["all"] = [
        {"field": "M", "op": "in", "value": 4096}
    ]
    with pytest.raises(ValidationError, match="requires a non-empty list"):
        Concept.model_validate(invalid)

    invalid = _concept_payload()
    invalid["scope"]["workloads"]["all"] = [
        {"field": "M", "op": "python", "value": "__import__('os')"}
    ]
    with pytest.raises(ValidationError):
        Concept.model_validate(invalid)


def test_observation_identity_round_and_hashes_are_validated():
    observation = ObservationRecord.model_validate(_observation())
    assert observation.round_num == 1
    invalid = _observation()
    invalid["ledger_locator"]["round_num"] = 2
    with pytest.raises(ValidationError, match="must match round_num"):
        ObservationRecord.model_validate(invalid)


def test_observation_v2_rejects_legacy_payloads():
    legacy = _observation()
    legacy["schema_version"] = "1.0"
    with pytest.raises(ValidationError, match="Input should be '2.0'"):
        ObservationRecord.model_validate(legacy)

    legacy = _observation()
    legacy["baseline_sha256"] = _HEX_B
    legacy["evaluation_fingerprint"] = "legacy-eval"
    legacy["profile_fingerprint"] = "legacy-profile"
    legacy["supersedes_observation_id"] = ""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ObservationRecord.model_validate(legacy)


def test_source_reference_reads_legacy_fields_but_emits_exact_shape():
    reference = SourceReference.model_validate(
        {
            "resource": "source:triton-docs",
            "id": "language/load.md",
            "title": "Load semantics",
            "author": "legacy",
            "usage_count": 3,
            "last_modified": "2026-01-01",
        }
    )

    assert reference.locator == "language/load.md"
    assert reference.id == "language/load.md"
    assert reference.revision == ""
    assert reference.model_dump(mode="json") == {
        "resource": "source:triton-docs",
        "revision": "",
        "locator": "language/load.md",
        "title": "Load semantics",
    }


def test_candidate_cannot_smuggle_managed_or_measurement_fields():
    candidate = {
        "schema_version": "1.0",
        "candidate_id": "run-1:agent-1:candidate-1",
        "base_snapshot": f"sha256:{_HEX_A}",
        "proposed_kind": "method",
        "proposed_id": None,
        "claim_key": "method.reduction.two_stage",
        "title": "Two-stage reduction",
        "summary": "Split a reduction into two stages.",
        "domains": ["optimization"],
        "scope": _scope(),
        "retrieval": {},
        "body": "# Claim\n\nSplit reduction.",
        "relations": [],
        "observation_intents": [
            {
                "round_num": 1,
                "stance": "supports",
                "rationale": "The round directly tests the claim.",
            }
        ],
        "source_refs": [],
        "created_by": "agent-1",
    }
    assert CandidateConcept.model_validate(candidate).proposed_kind == "method"
    candidate["managed"] = {"revision": 99}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CandidateConcept.model_validate(candidate)


def test_query_context_accepts_partial_phase_evidence():
    query = {
        "schema_version": "1.0",
        "phase": "post_profile",
        "task": "diagnosis",
        "round_num": 1,
        "question": "What should be tested next?",
        "operator_signature": _signature(),
        "target_context": _target(),
        "workload_summary": {},
        "findings": [],
        "errors": [],
        "max_results": 12,
    }
    parsed = QueryContext.model_validate(query)
    assert parsed.findings == []
    assert "workload_summary" not in parsed.model_dump(mode="json")
    query["findings"] = [
        {
            "category": "memory_bandwidth",
            "label": "low memory throughput",
            "confidence": "medium",
            "workload_uuids": ["u0"],
        }
    ]
    assert QueryContext.model_validate(query).phase == "post_profile"


def test_operator_signature_and_target_context_have_minimum_identity():
    assert OperatorSignature.model_validate(_signature()).motifs == [
        "reduction",
        "elementwise",
    ]
    assert TargetContext.model_validate(_target()).backend == "cuda"
    legacy_target = _target(fingerprint=f"sha256:{_HEX_A}")
    target = TargetContext.model_validate(legacy_target)
    assert "fingerprint" not in target.model_dump(mode="json")
    empty = _signature()
    empty.update(motifs=[], dataflow=[])
    assert OperatorSignature.model_validate(empty).motifs == []

    legacy = _signature()
    legacy.update(
        semantic_names=["rmsnorm"],
        structure_paths=["reduction > broadcast"],
        extensions=[
            {
                "namespace": "legacy",
                "value": "rmsnorm",
                "description": "legacy fallback",
            }
        ],
    )
    emitted = OperatorSignature.model_validate(legacy).model_dump(mode="json")
    assert "semantic_names" not in emitted
    assert "structure_paths" not in emitted
    assert "extensions" not in emitted


def test_target_scope_discards_legacy_fingerprint_and_numerics_can_be_unknown():
    payload = _concept_payload()
    payload["scope"]["target"]["target_fingerprint"] = "legacy:opaque:value"
    payload["scope"]["numerics"] = {}

    concept = Concept.model_validate(payload)

    assert "target_fingerprint" not in concept.scope.target.model_dump()
    assert concept.scope.numerics.exact is None


def test_legacy_exact_false_remains_readable():
    payload = _concept_payload()
    payload["scope"]["numerics"] = {"exact": False}

    concept = Concept.model_validate(payload)

    assert concept.scope.numerics.exact is False


def test_bundle_buckets_replace_legacy_usage_and_summary_evidence():
    result = {
        "concept_ref": "kg:method:two-stage-reduction",
        "kind": "method",
        "summary": "Split reduction.",
        "usage": "direct_action",
        "matched_on": ["motif.reduction"],
        "scope_gaps": [],
        "evidence_state": "corroborated",
        "evidence": [
            {
                "observation_ref": (
                    "kg:observation:run-1:agent-1:round-1"
                ),
                "stance": "supports",
                "rationale": "The round directly tests the claim.",
                "confidence": "high",
            }
        ],
    }
    bundle = {
        "schema_version": "1.0",
        "query_id": "query:test-bundle",
        "snapshot": f"sha256:{_HEX_A}",
        "direct": [result],
        "analogies": [],
        "conflicts": [],
        "coverage": {"matched_routes": ["motif"], "gaps": []},
    }
    parsed = KnowledgeBundle.model_validate(bundle)
    assert len(parsed.direct) == 1
    emitted = parsed.direct[0].model_dump(mode="json")
    assert "usage" not in emitted
    assert "evidence" not in emitted
    bundle["analogies"] = [result]
    assert len(KnowledgeBundle.model_validate(bundle).analogies) == 1


def test_source_package_is_versioned_and_content_addressed():
    package = SourcePackage.model_validate(
        {
            "schema_version": "1.0",
            "id": "source:triton-docs",
            "source_type": "documentation",
            "origin": "https://triton-lang.org/",
            "revision": "3.3.1",
            "content_root": f"artifact://sha256/{_HEX_A}",
            "license": "Apache-2.0",
            "checksum": f"sha256:{_HEX_A}",
            "retrieved_at": _NOW.isoformat(),
            "authority": "official",
            "allowed_hosts": ["triton-lang.org"],
        }
    )
    assert package.authority == "official"
    assert package.search_mode == "static_only"

    invalid_searchable = package.model_dump(mode="json")
    invalid_searchable["search_mode"] = "searchable"
    with pytest.raises(ValidationError, match="requires manifest_root"):
        SourcePackage.model_validate(invalid_searchable)

    restricted = package.model_dump(mode="json")
    restricted["usage_policy"] = "restricted"
    restricted["usage_notes"] = "Ascend-only fixture."
    with pytest.raises(ValidationError, match="requires allowed_backends"):
        SourcePackage.model_validate(restricted)
    restricted["allowed_backends"] = ["ascend"]
    assert SourcePackage.model_validate(restricted).allowed_backends == ["ascend"]


def _write_concept(path: Path, payload: dict):
    provisional = Concept.model_validate(payload)
    payload = provisional.model_dump(mode="json")
    payload["managed"]["content_hash"] = canonical_concept_hash(provisional)
    body = payload.pop("body")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        + "---\n\n"
        + body
        + "\n",
        encoding="utf-8",
    )


def test_filesystem_validator_checks_hash_and_cross_file_references(tmp_path):
    kb = tmp_path / "kb"
    source = {
        "schema_version": "1.0",
        "id": "source:triton-docs",
        "source_type": "documentation",
        "origin": "https://triton-lang.org/",
        "revision": "3.3.1",
        "content_root": f"artifact://sha256/{_HEX_A}",
        "license": "Apache-2.0",
        "checksum": f"sha256:{_HEX_A}",
        "retrieved_at": _NOW.isoformat(),
        "authority": "official",
        "allowed_hosts": ["triton-lang.org"],
    }
    source_path = kb / "sources" / "packages" / "triton-docs.yaml"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")

    observation_path = (
        kb / "observations" / "by_run" / "run-1.jsonl"
    )
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text(
        json.dumps(_observation()) + "\n",
        encoding="utf-8",
    )

    concept_path = kb / "concepts" / "experience--reduction-test.md"
    _write_concept(concept_path, _concept_payload())

    report = validate_knowledge_base(kb)
    assert report.as_dict() == {
        "valid": True,
        "concepts": 1,
        "observations": 1,
        "source_packages": 1,
        "source_entries": 0,
        "knowledge_documents": 0,
        "ingestion_plans": 0,
        "solutions": 0,
        "issues": [],
    }
    assert parse_concept(concept_path).id == "kg:experience:reduction-test"

    observation_path.write_text("", encoding="utf-8")
    report = validate_knowledge_base(kb)
    assert report.valid is False
    assert "observation_ref does not exist" in report.issues[0].message


def test_committed_json_schemas_match_pydantic_models():
    root = Path(__file__).resolve().parents[1]
    assert export_schemas(root / "kb" / "schemas", check=True) == []


def test_committed_seed_catalog_is_valid():
    root = Path(__file__).resolve().parents[1]
    report = validate_knowledge_base(root / "kb")
    assert report.valid, report.as_dict()
    assert report.concepts >= 8
    assert report.source_packages >= 4
    assert report.ingestion_plans >= 1
    assert report.knowledge_documents >= 1


def test_local_source_content_can_be_refreshed_without_checksum_gate(tmp_path):
    kb = tmp_path / "kb"
    content = kb / "sources" / "content" / "example" / "source.html"
    content.parent.mkdir(parents=True)
    content.write_bytes(b"<html>authoritative source</html>")
    checksum = "sha256:" + hashlib.sha256(content.read_bytes()).hexdigest()
    package = kb / "sources" / "packages" / "example.yaml"
    package.parent.mkdir(parents=True)
    package.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "id": "source:example",
                "source_type": "documentation",
                "origin": "https://example.invalid/source",
                "revision": "1",
                "content_root": (
                    "kb://sources/content/example/source.html"
                ),
                "license": "test-only",
                "checksum": checksum,
                "retrieved_at": _NOW.isoformat(),
                "authority": "official",
                "allowed_hosts": ["example.invalid"],
            }
        ),
        encoding="utf-8",
    )
    assert validate_knowledge_base(kb).valid

    content.write_bytes(b"<html>tampered</html>")
    report = validate_knowledge_base(kb)
    assert report.valid


def test_local_source_tree_can_evolve_without_checksum_gate(tmp_path):
    kb = tmp_path / "kb"
    content = kb / "sources" / "content" / "example-tree"
    (content / "docs").mkdir(parents=True)
    (content / "docs" / "a.md").write_text("# A\n", encoding="utf-8")
    (content / "docs" / "b.md").write_text("# B\n", encoding="utf-8")
    package = kb / "sources" / "packages" / "example-tree.yaml"
    package.parent.mkdir(parents=True)
    package.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "id": "source:example-tree.abc123",
                "source_type": "repository",
                "origin": "https://example.invalid/repository.git",
                "revision": "abc123",
                "content_root": "kb://sources/content/example-tree",
                "license": "test-only",
                "checksum": source_content_checksum(content),
                "retrieved_at": _NOW.isoformat(),
                "authority": "official",
                "allowed_hosts": ["example.invalid"],
            }
        ),
        encoding="utf-8",
    )
    assert validate_knowledge_base(kb).valid

    (content / "docs" / "b.md").write_text("# Changed\n", encoding="utf-8")
    report = validate_knowledge_base(kb)
    assert report.valid


def test_git_source_manifest_is_advisory_metadata(tmp_path):
    kb = tmp_path / "kb"
    content = kb / "sources" / "content" / "repo" / "abc123"
    (content / "docs").mkdir(parents=True)
    readme = content / "README.md"
    guide = content / "docs" / "guide.md"
    readme.write_text("# Repository\n", encoding="utf-8")
    guide.write_text("# Guide\n", encoding="utf-8")
    link = content / "docs" / "README.md"
    link.symlink_to("../README.md")

    def entry(path, mode, data):
        return {
            "path": path,
            "mode": mode,
            "object_type": "blob",
            "object_id": "a" * 40,
            "size": len(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        }

    manifest_payload = {
        "schema_version": "1.0",
        "source_package": "source:repo.abc123",
        "revision": "abc123",
        "object_format": "sha1",
        "entries": [
            entry("README.md", "100644", readme.read_bytes()),
            entry("docs/README.md", "120000", b"../README.md"),
            entry("docs/guide.md", "100644", guide.read_bytes()),
            {
                "path": "third_party/dependency",
                "mode": "160000",
                "object_type": "commit",
                "object_id": "b" * 40,
                "size": None,
                "sha256": "",
            },
        ],
    }
    manifest = kb / "sources" / "manifests" / "repo-abc123.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

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
                "retrieved_at": _NOW.isoformat(),
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

    report = validate_knowledge_base(kb)
    assert report.valid, report.as_dict()
    assert report.source_entries == 4

    guide.write_text("# Tampered\n", encoding="utf-8")
    report = validate_knowledge_base(kb)
    assert report.valid


def test_static_ingestion_plan_rejects_experience_and_unselected_entries():
    base = {
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
                "source_path": "docs/api.md",
                "heading": "API",
                "mode": "direct",
                "proposed_kind": "reference",
                "claim_key": "reference.example.api",
                "title": "Example API",
                "summary": "The API has one documented contract.",
                "domains": ["language"],
                "scope": {"target": {"level": "portable"}},
            }
        ],
    }
    assert StaticIngestionPlan.model_validate(base).entries[0].heading == "API"

    invalid = dict(base)
    invalid["rules"] = [
        {
            "include": "docs/*.md",
            "mode": "direct",
            "kinds": ["experience"],
        }
    ]
    with pytest.raises(ValidationError, match="cannot publish Experience"):
        StaticIngestionPlan.model_validate(invalid)

    invalid = dict(base)
    invalid["entries"] = [
        {
            **base["entries"][0],
            "source_path": "other/api.md",
        }
    ]
    with pytest.raises(ValidationError, match="not covered by a rule"):
        StaticIngestionPlan.model_validate(invalid)

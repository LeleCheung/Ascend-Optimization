from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from kernelgen.knowledge.sources import SourceSearcher
from kernelgen.knowledge.publishing.publisher import BatchPublisher
from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.models import (
    CandidateConcept,
    CandidateDraft,
    GitSourceEntry,
    GitSourceManifest,
    RetrievalMetadata,
    Scope,
    SourcePackage,
    SourceReference,
    TargetScope,
)
from kernelgen.knowledge.validation import (
    validate_knowledge_base,
    validate_runtime_candidate,
)
from kernelgen.knowledge.vocabulary import Vocabulary
from kernelgen.scripts.kb.audit_source_vocabulary import (
    audit_source_vocabulary,
)
from kernelgen.tools.snapshot_git_source import snapshot


def _write_yaml(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _source_catalog(tmp_path: Path) -> Path:
    kb = tmp_path / "kb"
    content = kb / "sources" / "content" / "demo"
    content.mkdir(parents=True)
    text = (
        "Use coalesced global access for contiguous data.\n"
        "This guide also contains a novel_token example.\n"
    )
    document = content / "guide.md"
    document.write_text(text, encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(text.encode()).hexdigest()

    package = SourcePackage(
        id="source:demo",
        source_type="repository",
        origin="https://example.invalid/demo.git",
        revision="a" * 40,
        content_root="kb://sources/content/demo",
        license="test",
        search_mode="searchable",
        manifest_root="kb://sources/manifests/demo.json",
        retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    _write_yaml(
        kb / "sources" / "packages" / "demo.yaml",
        package.model_dump(mode="json"),
    )
    manifest = GitSourceManifest(
        source_package=package.id,
        revision=package.revision,
        object_format="sha1",
        entries=[
            GitSourceEntry(
                path="guide.md",
                mode="100644",
                object_type="blob",
                object_id="b" * 40,
                size=len(text.encode()),
                sha256=digest,
            )
        ],
    )
    manifest_path = kb / "sources" / "manifests" / "demo.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json")),
        encoding="utf-8",
    )
    _write_yaml(
        kb / "sources" / "ingestion" / "demo.yaml",
        {
            "schema_version": "1.0",
            "source_package": package.id,
            "rules": [{"include": "*", "mode": "indexed"}],
            "entries": [],
        },
    )
    _write_yaml(
        kb / "aliases.yaml",
        {
            "techniques": {
                "coalesced_access": [
                    "coalesced global access",
                    "coalesced_global_access",
                    "coalescing",
                ]
            },
            "symptoms": {},
        },
    )
    _write_yaml(
        kb / "canonical_techniques.yaml",
        {"schema_version": "1.0", "terms": ["coalesced_access"]},
    )
    _write_yaml(
        kb / "canonical_symptoms.yaml",
        {"schema_version": "1.0", "terms": ["compile_error"]},
    )
    return kb


def test_source_search_expands_exact_aliases(tmp_path):
    kb = _source_catalog(tmp_path)
    canonical = SourceSearcher(kb).search("coalesced_access")
    alias = SourceSearcher(kb).search("coalescing")

    assert set(canonical.expanded_terms) == set(alias.expanded_terms)
    assert [hit.path for hit in canonical.hits] == ["guide.md"]
    assert [hit.path for hit in alias.hits] == ["guide.md"]


def test_source_search_preserves_unknown_terms(tmp_path):
    result = SourceSearcher(_source_catalog(tmp_path)).search("novel_token")

    assert result.expanded_terms == ["novel_token"]
    assert [hit.path for hit in result.hits] == ["guide.md"]


def test_vocabulary_canonicalizes_aliases_and_preserves_unknown_values(tmp_path):
    vocabulary = Vocabulary.from_catalog(_source_catalog(tmp_path))

    assert vocabulary.canonical_terms("techniques") == ["coalesced_access"]
    assert vocabulary.canonicalize("techniques", "coalescing") == "coalesced_access"
    values, unknown = vocabulary.normalize_values(
        "techniques",
        ["coalescing", "novel_technique"],
    )
    assert values == ["coalesced_access", "novel_technique"]
    assert unknown == ["novel_technique"]


def test_vocabulary_rejects_alias_owner_missing_from_canonical_list(tmp_path):
    kb = _source_catalog(tmp_path)
    _write_yaml(
        kb / "aliases.yaml",
        {
            "techniques": {"not_canonical": ["other spelling"]},
            "symptoms": {},
        },
    )

    try:
        Vocabulary.from_catalog(kb)
    except ValueError as exc:
        assert "not listed in canonical_techniques.yaml" in str(exc)
    else:
        raise AssertionError("invalid alias owner was accepted")
    report = validate_knowledge_base(kb)
    assert report.valid is False
    assert "invalid vocabulary" in report.issues[0].message


def test_source_vocabulary_audit_checks_every_family(tmp_path):
    report = audit_source_vocabulary(_source_catalog(tmp_path), max_results=5)

    assert report["canonical_counts"] == {"symptoms": 1, "techniques": 1}
    assert report["inconsistent_families"] == []
    assert report["missing_families"] == ["symptoms.compile_error"]
    coalesced = next(
        item
        for item in report["families"]
        if item["canonical"] == "coalesced_access"
    )
    assert coalesced["consistent"] is True
    assert coalesced["hit_count"] == 1


def _draft(kind: str, body: str) -> CandidateDraft:
    return CandidateDraft(
        proposed_kind=kind,
        claim_key=f"{kind}.demo",
        title="Demo",
        summary="Demo candidate",
        domains=["optimization"],
        retrieval=RetrievalMetadata(
            phases=["post_profile"],
            tasks=["diagnosis" if kind == "diagnostic" else "implementation"],
            symptoms=["compile_error"] if kind == "diagnostic" else [],
            techniques=["coalesced_access"] if kind == "method" else [],
        ),
        body=body,
        source_refs=[
            SourceReference(
                resource="source:demo",
                revision="a" * 40,
                locator="guide.md",
            )
        ],
    )


def _method_body() -> str:
    return "\n\n".join(
        [
            "## Claim\n\nUse coalesced access.",
            "## Evidence\n\nThe retrieved Source documents the pattern.",
            "## Applicability\n\nContiguous tensors.",
            (
                "## Action\n\nApply the layout.\n\n"
                "### Expected Metric Change\n\nLower memory transactions."
            ),
            (
                "## Limits\n\nDo not use for strided layouts.\n\n"
                "### Mechanism Requirements\n\nContiguous adjacent lanes."
            ),
        ]
    )


def test_runtime_candidate_body_contract_depends_on_kind():
    diagnostic = _draft(
        "diagnostic",
        "\n\n".join(
            [
                "## Symptom\n\nCompilation fails.",
                "## Likely Causes\n\nUnsupported lowering.",
                "## Candidate Techniques\n\nInspect the failing stage.",
                "## Diagnosis Checklist\n\n1. Reproduce the error.",
                "## Caveats\n\nConfirm the compiler version.",
            ]
        ),
    )
    validate_runtime_candidate(diagnostic)

    method = _draft(
        "method",
        _method_body(),
    )
    validate_runtime_candidate(method)


def test_publisher_canonicalizes_known_values_and_warns_on_unknown(tmp_path):
    kb = _source_catalog(tmp_path)
    candidate = CandidateConcept(
        candidate_id="candidate:vocabulary",
        proposed_kind="method",
        claim_key="method.vocabulary",
        title="Vocabulary publication",
        summary="Normalize a known alias and retain an unknown term.",
        domains=["optimization"],
        scope=Scope(target=TargetScope(level="portable")),
        retrieval=RetrievalMetadata(
            phases=["initial"],
            tasks=["implementation"],
            techniques=["coalescing", "novel_technique"],
        ),
        body=_method_body(),
        source_refs=[
            SourceReference(
                resource="source:demo",
                revision="a" * 40,
                locator="guide.md",
            )
        ],
        created_by="source-ingest:test",
    )

    result = BatchPublisher(kb, fact_reader=None).publish_static(
        [candidate],
        batch_id="vocabulary-test",
    )

    assert result.audit_warnings == [
        "candidate:vocabulary: unknown technique 'novel_technique'",
    ]
    published = FilesystemCatalog(kb).get_concept(result.created[0])
    assert published.retrieval.techniques == [
        "coalesced_access",
        "novel_technique",
    ]


def test_static_publisher_rejects_source_backend_mismatch(tmp_path):
    kb = _source_catalog(tmp_path)
    package_path = kb / "sources" / "packages" / "demo.yaml"
    package = yaml.safe_load(package_path.read_text(encoding="utf-8"))
    package["allowed_backends"] = ["ascend"]
    _write_yaml(package_path, package)
    candidate = CandidateConcept(
        candidate_id="candidate:wrong-backend",
        proposed_kind="method",
        claim_key="method.wrong_backend",
        title="Backend-specific method",
        summary="This Source only supports Ascend.",
        domains=["optimization"],
        scope=Scope(
            target=TargetScope(level="backend", backend="cuda")
        ),
        retrieval=RetrievalMetadata(
            phases=["initial"],
            tasks=["implementation"],
            techniques=["coalesced_access"],
        ),
        body=_method_body(),
        source_refs=[
            SourceReference(
                resource="source:demo",
                revision="a" * 40,
                locator="guide.md",
            )
        ],
        created_by="source-ingest:test",
    )

    result = BatchPublisher(kb, fact_reader=None).publish_static(
        [candidate],
        batch_id="wrong-backend",
    )

    assert result.status == "noop"
    assert result.rejected == [
        "candidate:wrong-backend: source source:demo only permits backends "
        "['ascend'], but Candidate scope is level=backend, backend='cuda'"
    ]


def test_snapshot_can_exclude_one_git_subtree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )
    (repo / "keep.md").write_text("keep\n", encoding="utf-8")
    excluded = repo / "official" / "CANNBot"
    excluded.mkdir(parents=True)
    (excluded / "duplicate.md").write_text("duplicate\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "fixture"],
        check=True,
    )

    kb = tmp_path / "kb"
    package_path = kb / "sources" / "packages" / "agent-skills.yaml"
    _write_yaml(
        package_path,
        {
            "schema_version": "1.0",
            "id": "source:agent-skills",
            "source_type": "repository",
            "origin": "https://example.invalid/agent-skills.git",
            "revision": "pending",
            "content_root": "pending",
            "license": "test",
            "search_mode": "static_only",
            "retrieved_at": "2026-01-01T00:00:00Z",
        },
    )

    result = snapshot(
        repo=repo,
        kb=kb,
        package_path=package_path,
        name="agent-skills",
        revision="HEAD",
        replace=False,
        exclude_paths=["official/CANNBot"],
    )
    manifest = json.loads(
        (kb / result["manifest_root"].removeprefix("kb://")).read_text()
    )

    assert result["excluded_paths"] == ["official/CANNBot"]
    assert [entry["path"] for entry in manifest["entries"]] == ["keep.md"]

"""Tests for the read-only historical retrieval replay cohort."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_replay_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "kb" / "replay_retrieval_history.py"
    spec = importlib.util.spec_from_file_location("replay_retrieval_history", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(operation: str, query_id: str, **updates):
    event = {
        "operation": operation,
        "origin": "mcp",
        "status": "success",
        "query_id": query_id,
        "request": {},
        "returned_refs": [],
        "returned_sources": [],
        "result_levels": {},
    }
    event.update(updates)
    return event


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def test_published_workspace_cohort_is_frozen_by_query_ids(tmp_path):
    replay = _load_replay_module()
    knowledge = tmp_path / "run-a" / "agent0" / ".kernelgen" / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "publish-result.json").write_text("{}\n", encoding="utf-8")
    missing = tmp_path / "run-b" / "agent1" / ".kernelgen" / "knowledge"
    missing.mkdir(parents=True)
    (missing / "publish-result.json").write_text("{}\n", encoding="utf-8")
    events = [
        _event(
            "query_knowledge",
            "query:concept",
            request={
                "question": "矩阵乘 Triton tile",
                "operator_signature": {"definition_id": "matmul"},
            },
            returned_refs=["kg:method:one", "kg:reference:two"],
            result_levels={
                "kg:method:one": "direct",
                "kg:reference:two": "analogy",
            },
        ),
        _event(
            "get_knowledge",
            "query:concept",
            returned_refs=["kg:method:one"],
        ),
        _event(
            "query_sources",
            "source-query:source",
            request={
                "query": "中文矩阵乘",
                "usage_scope": {"definition_id": "matmul"},
            },
            returned_sources=[
                "source:docs@revision::guide/matmul.md:L1-L8"
            ],
        ),
        _event(
            "get_source",
            "source-query:source",
            returned_sources=[
                "source:docs@revision::guide/matmul.md:L1-L40"
            ],
        ),
    ]
    log = knowledge / "retrieval-log.jsonl"
    _write_jsonl(log, events)

    discovered = replay._collect_cohort(tmp_path, manifest=None)
    counts = replay._cohort_counts(discovered)
    assert counts == {
        "workspace_count": 2,
        "workspace_logs_present": 1,
        "missing_log_count": 1,
        "query_count": 2,
        "concept_query_count": 1,
        "source_query_count": 1,
        "detail_read_query_count": 2,
        "detail_read_rate": 1.0,
    }
    concept, source = discovered["cases"]
    assert concept["historical_top4"] == ["kg:method:one"]
    assert concept["language"] == "mixed"
    assert source["historical_top4"] == ["source:docs::guide/matmul.md"]
    assert source["historical_detail_refs"] == [
        "source:docs::guide/matmul.md"
    ]

    manifest = replay._manifest_payload(discovered, "fixture-v1")
    events.append(
        _event(
            "query_sources",
            "source-query:later",
            request={"query": "later event"},
        )
    )
    _write_jsonl(log, events)
    frozen = replay._collect_cohort(tmp_path, manifest=manifest)
    replay._validate_manifest_counts(manifest, frozen)
    assert [row["query_id"] for row in frozen["cases"]] == manifest["query_ids"]


def test_language_and_source_reference_normalization():
    replay = _load_replay_module()
    assert replay._language("matrix multiplication") == "english"
    assert replay._language("矩阵乘法") == "chinese"
    assert replay._language("Triton 矩阵乘法") == "mixed"
    assert replay._source_key(
        "source:triton-ascend@abc::docs/zh/matmul.md:L10-L30"
    ) == "source:triton-ascend::docs/zh/matmul.md"

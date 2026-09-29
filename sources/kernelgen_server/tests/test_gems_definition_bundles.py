"""Uploaded Gems ABI uses original target pytest, never synthetic Native workloads."""
import hashlib
import json
from pathlib import Path

import pytest

from kernelgen_server import Catalog
from kernelgen_server.operator_bundles import (
    GemsDefinitionSource, OperatorBundleError, OperatorBundleStore, pack_operator_bundle,
)
from kernelgen_server.protocol.schema import EvaluatorBinding
from kernelgen_server.evaluation.adapters.registry import create_adapter
from kernelgen_server.evaluation.adapters.flaggems.discovery import FlagGemsAssets


def bundle(tmp_path):
    root = tmp_path / "upload"
    root.mkdir()
    definition = {"api_version": "v6.0", "name": "negative",
                  "parameters": [{"name": "x", "required": True}], "outputs": ["out"]}
    (root / "definition.json").write_text(json.dumps(definition))
    source = GemsDefinitionSource(source_revision="a" * 40, source_files={
        "tests/test_negative.py": hashlib.sha256(b"correctness").hexdigest(),
        "benchmark/test_negative.py": hashlib.sha256(b"timing").hexdigest(),
    })
    (root / "adapter.json").write_text(source.model_dump_json())
    return root


def install(tmp_path, root):
    archive = tmp_path / "operator.tar"
    with archive.open("w+b") as output:
        digest, _ = pack_operator_bundle(root, output)
    store = OperatorBundleStore(tmp_path / "bundles")
    return store, store.install(archive, digest)[0]


def test_upload_routes_to_gems_adapter_and_keeps_definition_bytes(tmp_path):
    root = bundle(tmp_path)
    store, info = install(tmp_path, root)
    catalog = Catalog(store.catalog_path(info.bundle_id))
    assert catalog.evaluator == "flaggems" and catalog.api_version == "v6.0"
    assert (catalog.root / "definitions/negative.json").read_bytes() == (root / "definition.json").read_bytes()
    assert not (catalog.root / "ops").exists()
    operator = catalog.load("negative")
    assert operator.correctness_workloads == operator.timing_workloads == []
    adapter = create_adapter(EvaluatorBinding(bundle_id=info.bundle_id, definition="negative"), operator_bundle_root=store.root)
    assert adapter.kind == "flaggems"
    assert store.describe()["evaluators"] == ["native", "flaggems"]


@pytest.mark.parametrize("extra", ["oracle.py", "correctness.jsonl", "timing.jsonl", "payload/extra.py"])
def test_adapter_bundle_cannot_inject_code_or_workloads(tmp_path, extra):
    root = bundle(tmp_path)
    path = root / extra
    path.parent.mkdir(exist_ok=True)
    path.write_text("not allowed")
    with pytest.raises(OperatorBundleError, match="only definition.json"):
        install(tmp_path, root)


@pytest.mark.parametrize("name", ["../escape", "a/b", "/absolute", ".", ".."])
def test_definition_name_cannot_escape_staging(tmp_path, name):
    root = bundle(tmp_path)
    path = root / "definition.json"
    value = json.loads(path.read_text());value["name"] = name;path.write_text(json.dumps(value))
    with pytest.raises(OperatorBundleError, match="safe single path"):
        install(tmp_path, root)


@pytest.mark.parametrize("relative", ["../secret.py", "/tmp/secret.py", "a/../b.py", "./test.py", "a\\b.py"])
def test_origin_rejects_foreign_paths(relative):
    with pytest.raises(ValueError, match="repository-relative"):
        GemsDefinitionSource(source_revision="a" * 40, source_files={relative: "b" * 64})


def test_runtime_checks_origin_revision_suite_coverage_and_hashes(tmp_path, monkeypatch):
    from kernelgen_server.evaluation.adapters.flaggems import adapter as module
    root = bundle(tmp_path)
    store, info = install(tmp_path, root)
    gems = tmp_path / "gems"
    for path, text in [(gems / "tests/test_negative.py", "correctness"), (gems / "benchmark/test_negative.py", "timing")]:
        path.parent.mkdir(parents=True, exist_ok=True);path.write_text(text)
    observed = []
    def discover(operator, required_revision=None):
        observed.append((operator, required_revision))
        return FlagGemsAssets(gems, (gems / "tests/test_negative.py",), (gems / "benchmark/test_negative.py",), operator, required_revision, "a" * 40)
    monkeypatch.setattr(module, "discover_assets", discover)
    adapter = create_adapter(EvaluatorBinding(bundle_id=info.bundle_id, definition="negative"), operator_bundle_root=store.root)
    assert adapter._assets().revision == "a" * 40
    assert observed == [("negative", "a" * 40)]
    (gems / "tests/test_negative.py").write_text("changed")
    with pytest.raises(ValueError, match="differs from target"):
        adapter._assets()
    del adapter.catalog.manifest["definition_source"]["source_files"]["tests/test_negative.py"]
    with pytest.raises(ValueError, match="original pytest suites"):
        adapter._assets()

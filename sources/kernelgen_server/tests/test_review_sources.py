"""Source export is bounded, source-only, and tied to the evaluator binding."""

from types import SimpleNamespace

import pytest

from kernelgen_server.catalog import Catalog
from kernelgen_server.evaluation import test_review_sources as sources
from kernelgen_server.protocol import client
from kernelgen_server.protocol.schema import EvaluatorBinding, InspectRequest, OperatorContract, Definition
from test_operator_bundles import _operator_dir, _packed
from kernelgen_server.operator_bundles import OperatorBundleStore


def test_native_export_reads_helpers_without_executing(tmp_path):
    root = _operator_dir(tmp_path / "operator")
    (root / "oracle.py").write_text('REFERENCE_DEVICE="target"\nraise RuntimeError("must not execute")\ndef run(x): return x\n')
    archive = tmp_path / "operator.tar"
    digest, _ = _packed(root, archive)
    store = OperatorBundleStore(tmp_path / "bundles")
    store.install(archive, digest)
    catalog = Catalog(store.catalog_path("sha256:" + digest))
    evidence = sources.export_test_sources(catalog, catalog.load("user_square"))
    assert [f.path for f in evidence.files] == ["oracle.py"]
    assert "must not execute" in evidence.files[0].content


def test_gems_exports_original_suites_and_helpers(tmp_path, monkeypatch):
    from kernelgen_server.evaluation.adapters.flaggems.adapter import FlagGemsEvaluationAdapter

    paths = ["tests/test_op.py", "benchmark/test_op.py", "tests/accuracy_utils.py",
             "benchmark/base.py", "src/flag_gems/testing/__init__.py",
             "benchmark/consts.py", "benchmark/cases.py", "benchmark/utils.py",
             "benchmark/core_shapes.yaml", "benchmark/profile_hook.py"]
    for name in paths:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('raise RuntimeError("source only")\n')
    assets = SimpleNamespace(root=tmp_path, correctness=(tmp_path / paths[0],),
                             performance=(tmp_path / paths[1],), revision="a" * 40)
    calls = []
    monkeypatch.setattr(FlagGemsEvaluationAdapter, "_assets", lambda self: calls.append(True) or assets)
    evidence = sources.export_test_sources(object(), SimpleNamespace(evaluator="flaggems"))
    assert {f.path for f in evidence.files} == set(paths)
    assert all(f.content == 'raise RuntimeError("source only")\n' for f in evidence.files)
    assert evidence.framework_revision == "a" * 40
    assert len(calls) == 2


def test_export_rejects_symlink_and_size_overflow(tmp_path, monkeypatch):
    outside = tmp_path / "outside.py"
    outside.write_text("sensitive")
    root = tmp_path / "root"
    root.mkdir()
    link = root / "helper.py"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        sources._read_sources(root, [link])
    monkeypatch.setattr(sources, "MAX_REVIEW_SOURCE_BYTES", 2)
    with pytest.raises(ValueError, match="byte limit"):
        sources._read_sources(tmp_path, [outside])


@pytest.mark.parametrize("include", [False, True])
def test_sdk_requests_sources_only_explicitly(monkeypatch, include):
    binding = EvaluatorBinding(catalog_name="demo", definition="op")
    contract = OperatorContract(binding=binding, kind="native", catalog_api_version="v6.2",
                                definition=Definition(api_version="v6.2", name="op", parameters=[], outputs=["out"]),
                                correctness_workloads=[], timing_workloads=[])
    raw = contract.model_dump(mode="json", exclude_unset=True)
    if include:
        raw["test_sources"] = {"files": [{"path": "oracle.py", "content": "def run(): return 1"}]}
    def post(url, endpoint, payload, timeout):
        assert endpoint == "/operator-contract"
        assert payload.get("include_test_sources", False) == include
        return raw
    monkeypatch.setattr(client, "_post", post)
    value = client.get_operator_contract(InspectRequest(binding=binding), "http://local", include_test_sources=include)
    assert bool(value.test_sources) == include
    if include:
        raw.pop("test_sources")
        with pytest.raises(client.ServerError, match="test sources"):
            client.get_operator_contract(InspectRequest(binding=binding), "http://local", include_test_sources=True)

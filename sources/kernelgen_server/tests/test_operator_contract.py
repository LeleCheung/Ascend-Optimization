"""Export the target's Catalog bytes, without executing an oracle or pytest."""

import json
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

if find_spec("torch") is not None:
    import kernelgen_server.api.app as server
else:
    server = None
from kernelgen_server.catalog import Catalog, builtin_catalog_path
from kernelgen_server.evaluation.adapters import registry
from kernelgen_server.protocol import client as protocol_client
from kernelgen_server.protocol.schema import EvaluatorBinding, InspectRequest, OperatorContract
from test_operator_bundles import _operator_dir, _packed


@pytest.fixture
def client(tmp_path, monkeypatch):
    if server is None:
        pytest.skip("server application requires torch")
    monkeypatch.setattr(server, "_make_device", lambda _: SimpleNamespace(count_devices_safe=lambda: 1))
    monkeypatch.setattr(server, "configure_device", lambda *a, **k: None)
    monkeypatch.setattr(server, "probe_device", lambda *a, **k: None)
    monkeypatch.setattr(server, "environment_info", lambda *a: {})
    monkeypatch.setattr(server, "create_adapter", lambda *a, **k: pytest.fail("export must not create evaluator"))
    monkeypatch.setattr(server, "run_isolated", lambda *a, **k: pytest.fail("export must not run candidate"))
    monkeypatch.setattr(registry, "_prepare_framework_import", lambda *a: pytest.fail("export must not import Gems"))
    app = server.create_app(backend="cuda", enable_debug_jobs=False,
                            operator_bundle_root=tmp_path / "bundles",
                            profile_artifact_root=tmp_path / "profiles")
    with TestClient(app) as session:
        yield session


def test_status_exposes_pinned_source_policy_separately_from_protocol(client):
    state = client.get('/status').json()
    capability = state['capabilities']['native_source_policy']
    assert capability['enabled'] is True
    assert capability['workload_conditions'] is True
    assert capability['vendor'] == 'nvidia'
    assert capability['flags']['support_fp64'] is True
    assert capability['policy_id'] == 'flaggems/' + capability['source']['revision']
    assert state['api_version'] == 'v6.2'


@pytest.mark.parametrize("catalog_name,definition", [
    ("simple-v6-test", "identity"), ("flaggems-adapter-definitions", "addmm_"),
])
def test_installed_contract_matches_target_and_does_not_claim_readiness(client, catalog_name, definition):
    catalog = Catalog(builtin_catalog_path(catalog_name))
    expected = catalog.load(definition)
    binding = {"catalog_name": catalog_name, "definition": definition}
    result = client.post("/operator-contract", json={"binding": binding})
    assert result.status_code == 200, result.text
    raw = result.json()
    assert raw["binding"] == binding
    assert raw["kind"] == expected.evaluator
    assert raw["definition"] == expected.definition.model_dump(mode="json", exclude_unset=True)
    assert raw["correctness_workloads"] == [w.model_dump(mode="json", exclude_unset=True) for w in expected.correctness_workloads]
    assert raw["timing_workloads"] == [w.model_dump(mode="json", exclude_unset=True) for w in expected.timing_workloads]
    assert "status" not in raw and "benchmark_fingerprint" not in raw
    state = client.get("/status").json()
    assert state["capabilities"]["operator_contract"]["enabled"] is True
    assert state["capabilities"]["operator_contract"]["test_sources"] is True
    assert state["scheduler"]["active"] == state["scheduler"]["waiting"] == 0


def test_bundle_contract_preserves_explicit_null_and_input_factory(client, tmp_path):
    directory = _operator_dir(tmp_path / "operator")
    definition = json.loads((directory / "definition.json").read_text())
    definition["parameters"].append({"name": "bias", "required": False, "default": None})
    (directory / "definition.json").write_text(json.dumps(definition))
    oracle = ('REFERENCE_DEVICE = "target"\n'
              'raise AssertionError("contract export must not execute source")\n'
              'def gen_inputs(case): return {"x": case}\n'
              'def run(x, bias=None): return x * x\n')
    (directory / "oracle.py").write_text(oracle)
    archive = tmp_path / "bundle.tar"
    digest, _ = _packed(directory, archive)
    binding = {"bundle_id": "sha256:" + digest, "definition": "user_square"}
    assert client.post("/operator-contract", json={"binding": binding}).status_code == 404
    assert client.put(f"/operator-bundles/{digest}", content=archive.read_bytes(),
                      headers={"Content-Type": "application/x-tar"}).status_code == 201
    response = client.post("/operator-contract", json={"binding": binding})
    assert response.status_code == 200, response.text
    raw = response.json()
    assert raw["definition"]["reference"] == oracle
    assert "default" not in raw["definition"]["parameters"][0]
    assert raw["definition"]["parameters"][1]["default"] is None
    assert raw["correctness_workloads"][0]["name"] == "user-square-correctness"
    assert raw["timing_workloads"][0]["name"] == "user-square-timing"
    assert client.post("/operator-contract", json={"binding": {**binding, "definition": "other"}}).status_code == 422


def test_uploaded_gems_contract_exports_new_abi_without_running_pytest(client, tmp_path):
    from test_gems_definition_bundles import bundle
    directory = bundle(tmp_path)
    archive = tmp_path / "gems.tar"
    digest, _ = _packed(directory, archive)
    assert client.put(f"/operator-bundles/{digest}", content=archive.read_bytes(),
                      headers={"Content-Type": "application/x-tar"}).status_code == 201
    binding = {"bundle_id": "sha256:" + digest, "definition": "negative"}
    response = client.post("/operator-contract", json={"binding": binding})
    assert response.status_code == 200, response.text
    raw = response.json()
    assert raw["binding"] == binding
    assert raw["kind"] == "flaggems"
    assert raw["definition"] == json.loads((directory / "definition.json").read_text())
    assert raw["correctness_workloads"] == raw["timing_workloads"] == []


@pytest.mark.parametrize("binding,status", [
    ({"catalog_name": "absent", "definition": "identity"}, 404),
    ({"catalog_name": "simple-v6-test", "definition": "absent"}, 404),
    ({"catalog_name": "../secret", "definition": "identity"}, 422),
    ({"catalog_name": "/tmp/catalog", "definition": "identity"}, 422),
    ({"catalog_name": "simple-v6-test", "bundle_id": "sha256:" + "a" * 64, "definition": "identity"}, 422),
])
def test_missing_and_unsafe_bindings_are_rejected(client, binding, status):
    assert client.post("/operator-contract", json={"binding": binding}).status_code == status


def test_optional_native_test_sources(client):
    binding = {"catalog_name": "simple-v6-test", "definition": "identity"}
    response = client.post("/operator-contract", json={"binding": binding, "include_test_sources": True})
    assert response.status_code == 200, response.text
    assert response.json()["test_sources"]["files"]


@pytest.mark.parametrize("too_large", [False, True])
def test_optional_gems_test_sources_are_read_only_and_bounded(client, tmp_path, monkeypatch, too_large):
    from kernelgen_server.evaluation.adapters.flaggems.adapter import FlagGemsEvaluationAdapter
    from kernelgen_server.evaluation import test_review_sources

    source = tmp_path / "test_addmm.py"
    source.write_text('raise RuntimeError("must not execute pytest source")\n')
    assets = SimpleNamespace(root=tmp_path, correctness=(source,), performance=(source,), revision="a" * 40)
    monkeypatch.setattr(FlagGemsEvaluationAdapter, "_assets", lambda self: assets)
    if too_large:
        monkeypatch.setattr(test_review_sources, "MAX_REVIEW_SOURCE_BYTES", 1)
    response = client.post("/operator-contract", json={
        "binding": {"catalog_name": "flaggems-adapter-definitions", "definition": "addmm_"},
        "include_test_sources": True,
    })
    assert response.status_code == (422 if too_large else 200), response.text
    if not too_large:
        evidence = response.json()["test_sources"]
        assert evidence["framework_revision"] == "a" * 40
        assert evidence["files"] == [{"path": "test_addmm.py", "content": source.read_text()}]
    state = client.get("/status").json()["scheduler"]
    assert state["active"] == state["waiting"] == 0


def test_sdk_checks_response_binding(client, monkeypatch):
    request = InspectRequest(binding=EvaluatorBinding(catalog_name="simple-v6-test", definition="identity"))
    def post(url, endpoint, payload, timeout):
        assert endpoint == "/operator-contract"
        return client.post(endpoint, json=payload).json()
    monkeypatch.setattr(protocol_client, "_post", post)
    result = protocol_client.get_operator_contract(request, "http://local")
    assert isinstance(result, OperatorContract)
    raw = result.model_dump(mode="json", exclude_unset=True)
    raw["binding"]["catalog_name"] = "different-catalog"
    monkeypatch.setattr(protocol_client, "_post", lambda *a: raw)
    with pytest.raises(protocol_client.ServerError, match="different operator binding"):
        protocol_client.get_operator_contract(request, "http://local")

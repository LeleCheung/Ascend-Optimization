"""Bundle and built-in bindings share the evaluator, without accepting client paths."""

import pytest

from kernelgen_server.evaluation.adapters import create_adapter
from kernelgen_server.operator_bundles import OperatorBundleStore
from kernelgen_server.protocol.schema import EvaluatorBinding
from test_operator_bundles import _operator_dir, _packed


@pytest.mark.parametrize("fields", [
    {}, {"catalog_name": "simple-v6-test", "bundle_id": "sha256:" + "a" * 64},
    {"bundle_id": "../../catalog"}, {"bundle_id": "sha256:" + "A" * 64},
    {"catalog_name": "/tmp/catalog"},
])
def test_binding_rejects_ambiguous_or_unsafe_source(fields):
    with pytest.raises(ValueError):
        EvaluatorBinding(definition="user_square", **fields)


def test_uploaded_binding_resolves_only_in_explicit_store(tmp_path):
    operator = _operator_dir(tmp_path / "operator")
    archive = tmp_path / "bundle.tar"
    digest, _ = _packed(operator, archive)
    store = OperatorBundleStore(tmp_path / "bundles")
    info, _ = store.install(archive, digest)
    binding = EvaluatorBinding(bundle_id=info.bundle_id, definition=info.definition)
    with pytest.raises(ValueError, match="server-owned"):
        create_adapter(binding)
    with pytest.raises(KeyError, match="not found"):
        create_adapter(binding, operator_bundle_root=tmp_path / "other-server")
    adapter = create_adapter(binding, operator_bundle_root=tmp_path / "bundles")
    assert adapter.catalog.root == store.catalog_path(digest)
    assert adapter.inspect().kind == "native"
    assert adapter.inspect().case_list.cases[0].case_id == "user-square-timing"
    with pytest.raises((KeyError, ValueError, FileNotFoundError)):
        create_adapter(binding.model_copy(update={"definition": "other"}),
                       operator_bundle_root=tmp_path / "bundles")


def test_existing_binding_still_resolves_without_bundle_store():
    binding = EvaluatorBinding(catalog_name="simple-v6-test", definition="square")
    assert binding.model_dump(exclude_unset=True) == {
        "catalog_name": "simple-v6-test", "definition": "square",
    }


def test_http_bundle_binding_reaches_inspect_and_isolated_runner(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import kernelgen_server.api.app as server
    from kernelgen_server.protocol.schema import EvaluateResponse, PreflightResult

    monkeypatch.setattr(server, "_make_device", lambda _: SimpleNamespace(count_devices_safe=lambda: 1))
    monkeypatch.setattr(server, "configure_device", lambda *a, **k: None)
    monkeypatch.setattr(server, "probe_device", lambda *a, **k: None)
    monkeypatch.setattr(server, "environment_info", lambda *a: {})
    root = tmp_path / "bundles"
    observed = []

    def execute(kind, request, backend, device, timing, *, operator_bundle_root):
        adapter = create_adapter(request.binding, operator_bundle_root=operator_bundle_root)
        observed.append((kind, adapter.catalog.root))
        if kind == "preflight":
            return PreflightResult(status="PASSED", stage="complete")
        return EvaluateResponse(status="PASSED", device=device, server_backend=backend, num_workloads=1, num_passed=1)

    monkeypatch.setattr(server, "run_isolated", execute)
    app = server.create_app(backend="cuda", enable_debug_jobs=False,
                            operator_bundle_root=root, profile_artifact_root=tmp_path / "profiles")
    operator = _operator_dir(tmp_path / "operator")
    archive = tmp_path / "bundle.tar"
    digest, _ = _packed(operator, archive)
    binding = {"bundle_id": "sha256:" + digest, "definition": "user_square"}
    implementation = {"name": "candidate", "definition": "user_square", "language": "python",
                      "entrypoint": "main.py::run", "sources": [{"path": "main.py", "content": "def run(x): return x*x"}]}
    with TestClient(app) as client:
        assert client.post("/inspect", json={"binding": binding}).status_code == 404
        assert client.put(f"/operator-bundles/{digest}", content=archive.read_bytes(),
                          headers={"Content-Type": "application/x-tar"}).status_code == 201
        inspected = client.post("/inspect", json={"binding": binding})
        assert inspected.status_code == 200
        assert inspected.json()["kind"] == "native"
        for endpoint in ("preflight", "evaluate"):
            response = client.post("/" + endpoint, json={"binding": binding, "implementation": implementation})
            assert response.status_code == 200, response.text
        assert observed == [(kind, root / digest) for kind in ("preflight", "evaluate")]
        assert client.post("/inspect", json={"binding": {**binding, "definition": "other"}}).status_code == 422
        scheduler = client.get("/status").json()["scheduler"]
        assert scheduler["active"] == scheduler["waiting"] == scheduler["broken"] == 0


def test_profile_uses_uploaded_catalog_and_frozen_fingerprint(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import kernelgen_server.profiling.service as service
    from kernelgen_server.profiling.models import BackendProfileResult, ProfileRequest

    archive = tmp_path / "bundle.tar"
    digest, _ = _packed(_operator_dir(tmp_path / "operator"), archive)
    store = OperatorBundleStore(tmp_path / "bundles")
    info, _ = store.install(archive, digest)
    binding = EvaluatorBinding(bundle_id=info.bundle_id, definition=info.definition)
    manifest = create_adapter(binding, operator_bundle_root=store.root).inspect()
    observed = []
    def profile(command, **kwargs):
        observed.append(command)
        assert Path(command.cwd) == store.catalog_path(digest)
        target = json.loads((kwargs["artifact_dir"] / "target/target.json").read_text())
        assert Path(target["oracle_path"]).is_relative_to(store.catalog_path(digest))
        return BackendProfileResult(status="completed", profiler="fake")
    profiler = SimpleNamespace(name="fake", available_for=lambda opts: True,
        levels_for=lambda opts: ["metrics"], normalize_options=lambda opts: (opts, []), profile_command=profile)
    monkeypatch.setattr(service, "get_profiler", lambda backend: profiler)
    request = ProfileRequest(binding=binding, expected_backend="cuda",
        benchmark_fingerprint=manifest.benchmark_fingerprint, case_id="user-square-timing",
        implementation={"name": "candidate", "definition": info.definition, "language": "python",
            "entrypoint": "main.py::run", "sources": [{"path": "main.py", "content": "def run(x): return x*x"}]})
    runner = service.ProfileService(tmp_path / "profiles", backend="cuda", operator_bundle_root=store.root)
    assert runner.run(request, "cuda:0").status == "completed"
    rejected = runner.run(request.model_copy(update={"benchmark_fingerprint": "stale"}), "cuda:0")
    assert rejected.status == "failed" and "fingerprint" in rejected.error
    assert len(observed) == 1

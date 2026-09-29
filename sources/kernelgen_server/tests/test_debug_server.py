from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import kernelgen_server.api.app as server


class _FakeDevice:
    def count_devices_safe(self) -> int:
        return 1


def test_debug_submit_waits_for_terminal_result(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_resolve_backend", lambda backend: "cuda")
    monkeypatch.setattr(server, "_make_device", lambda backend: _FakeDevice())
    monkeypatch.setattr(server, "configure_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "runtime_device_type", lambda backend: "cuda")
    monkeypatch.setattr(server, "environment_info", lambda *args: {})
    monkeypatch.setattr(server, "probe_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "builtin_catalog_path", lambda name: tmp_path)
    app = server.create_app(
        backend="cuda",
        max_workers=1,
        debug_artifact_root=tmp_path / "debug",
        profile_artifact_root=tmp_path / "profile",
    )

    with TestClient(app) as client:
        status = client.get("/status").json()
        response = client.post(
            "/debug/jobs",
            json={
                "command": ["{python}", "-c", "print('done')"],
                "timeout_seconds": 10,
            },
        )

    assert status["api_version"] == "v6.2"
    assert status["server_version"] == server.KERNELGEN_SERVER_VERSION
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCEEDED"
    assert response.json()["stdout"].strip() == "done"

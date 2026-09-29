from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("torch")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import kernelgen_server.api.app as server
from kernelgen_server.runtime.isolated import IsolatedWorkerCrashed
from kernelgen_server.schema import (
    BoundEvaluateRequest,
    EvaluateResponse,
    EvaluationStatus,
    EvaluatorBinding,
    Implementation,
    SourceFile,
)


class _TwoDevices:
    def count_devices_safe(self) -> int:
        return 2


def _request() -> BoundEvaluateRequest:
    return BoundEvaluateRequest(
        binding=EvaluatorBinding(
            catalog_name="simple-v6-test",
            definition="identity",
        ),
        implementation=Implementation(
            name="slot-isolation-reference",
            definition="identity",
            language="python",
            entrypoint="main.py::run",
            sources=[
                SourceFile(
                    path="main.py",
                    content="def run(x):\n    return x\n",
                )
            ],
        ),
    )


def _passed(device_string: str, backend: str) -> EvaluateResponse:
    return EvaluateResponse(
        status=EvaluationStatus.PASSED,
        device=device_string,
        server_backend=backend,
        geo_mean=1.0,
        min_speedup=1.0,
        latency_ms=1.0,
        num_workloads=1,
        num_passed=1,
    )


def test_max_workers_never_duplicates_an_active_device(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_resolve_backend", lambda backend: "cuda")
    monkeypatch.setattr(server, "_make_device", lambda backend: _TwoDevices())
    monkeypatch.setattr(server, "configure_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "runtime_device_type", lambda backend: "cuda")
    monkeypatch.setattr(server, "environment_info", lambda *args: {})
    monkeypatch.setattr(server, "probe_device", lambda *args, **kwargs: None)

    lock = threading.Lock()
    release = threading.Event()
    first_wave = threading.Event()
    active_by_device: dict[str, int] = {}
    overlaps: list[str] = []
    active_total = 0
    max_active_total = 0

    def fake_run_isolated(
        operation,
        request,
        backend,
        device_string,
        timing,
    ):
        nonlocal active_total, max_active_total
        with lock:
            active_by_device[device_string] = (
                active_by_device.get(device_string, 0) + 1
            )
            if active_by_device[device_string] > 1:
                overlaps.append(device_string)
            active_total += 1
            max_active_total = max(max_active_total, active_total)
            if active_total == 2:
                first_wave.set()
        assert release.wait(timeout=5)
        with lock:
            active_by_device[device_string] -= 1
            active_total -= 1
        return _passed(device_string, backend)

    monkeypatch.setattr(server, "run_isolated", fake_run_isolated)
    app = server.create_app(
        backend="cuda",
        max_workers=4,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
    )
    payload = _request().wire_payload()

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(client.post, "/evaluate", json=payload) for _ in range(4)]
        assert first_wave.wait(timeout=5)
        deadline = time.monotonic() + 5
        while True:
            status = client.get("/status").json()
            if status["scheduler"]["waiting"] == 2:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert status["workers"] == 4
        assert status["scheduler"] == {
            "device_slots": 2,
            "healthy": 2,
            "checking": 0,
            "broken": 0,
            "incidents": 0,
            "recovered": 0,
            "probe_timeout_seconds": 30,
            "max_active": 2,
            "active": 2,
            "waiting": 2,
            "available": 0,
            "slots": [
                {
                    "device": "cuda:0",
                    "state": "active",
                    "reason": "",
                    "last_incident": "",
                    "incidents": 0,
                    "recovered": 0,
                },
                {
                    "device": "cuda:1",
                    "state": "active",
                    "reason": "",
                    "last_incident": "",
                    "incidents": 0,
                    "recovered": 0,
                },
            ],
        }
        release.set()
        responses = [future.result(timeout=5) for future in futures]

    assert all(response.status_code == 200 for response in responses)
    assert max_active_total == 2
    assert overlaps == []


def _patch_server_dependencies(monkeypatch, probe=None):
    monkeypatch.setattr(server, "_resolve_backend", lambda backend: "cuda")
    monkeypatch.setattr(server, "_make_device", lambda backend: _TwoDevices())
    monkeypatch.setattr(server, "configure_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "runtime_device_type", lambda backend: "cuda")
    monkeypatch.setattr(server, "environment_info", lambda *args: {})
    monkeypatch.setattr(
        server,
        "probe_device",
        probe or (lambda *args, **kwargs: None),
    )


def test_startup_probe_excludes_a_broken_slot(tmp_path, monkeypatch):
    def fake_probe(backend, device_string, timing, timeout_seconds):
        if device_string == "cuda:1":
            raise RuntimeError("device is unavailable")

    _patch_server_dependencies(monkeypatch, fake_probe)
    app = server.create_app(
        backend="cuda",
        max_workers=4,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
    )

    with TestClient(app) as client:
        status = client.get("/status").json()

    assert status["status"] == "degraded"
    assert status["scheduler"] == {
        "device_slots": 2,
        "healthy": 1,
        "checking": 0,
        "broken": 1,
        "incidents": 1,
        "recovered": 0,
        "probe_timeout_seconds": 30,
        "max_active": 1,
        "active": 0,
        "waiting": 0,
        "available": 1,
        "slots": [
            {
                "device": "cuda:0",
                "state": "available",
                "reason": "",
                "last_incident": "",
                "incidents": 0,
                "recovered": 0,
            },
            {
                "device": "cuda:1",
                "state": "broken",
                "reason": "RuntimeError: device is unavailable",
                "last_incident": "RuntimeError: device is unavailable",
                "incidents": 1,
                "recovered": 0,
            },
        ],
    }


def test_timeout_marks_slot_broken_but_runtime_error_does_not(
    tmp_path,
    monkeypatch,
):
    _patch_server_dependencies(monkeypatch)
    calls = 0

    def fake_run_isolated(
        operation,
        request,
        backend,
        device_string,
        timing,
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("intentional timeout")
        raise RuntimeError("ordinary operator failure")

    monkeypatch.setattr(server, "run_isolated", fake_run_isolated)
    app = server.create_app(
        backend="cuda",
        max_workers=2,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
    )

    def failed_recovery_probe(*args, **kwargs):
        raise RuntimeError("device queue is unhealthy")

    monkeypatch.setattr(server, "probe_device", failed_recovery_probe)
    payload = _request().wire_payload()

    with TestClient(app) as client:
        timeout = client.post("/evaluate", json=payload)
        failed = client.post("/evaluate", json=payload)
        status = client.get("/status").json()

    assert timeout.status_code == 200
    assert (
        timeout.json()["status"]
        == EvaluationStatus.SUSPECTED_DEVICE_ERROR.value
    )
    assert failed.status_code == 500
    assert calls == 2
    assert status["status"] == "degraded"
    assert status["scheduler"]["healthy"] == 1
    assert status["scheduler"]["checking"] == 0
    assert status["scheduler"]["broken"] == 1
    assert status["scheduler"]["incidents"] == 1
    assert status["scheduler"]["recovered"] == 0
    assert status["scheduler"]["available"] == 1
    assert status["scheduler"]["slots"][0]["state"] == "broken"
    assert status["scheduler"]["slots"][1]["state"] == "available"


def test_abnormal_worker_exit_marks_only_its_slot_broken(
    tmp_path,
    monkeypatch,
):
    _patch_server_dependencies(monkeypatch)

    def fake_run_isolated(*args, **kwargs):
        raise IsolatedWorkerCrashed("exitcode=17")

    monkeypatch.setattr(server, "run_isolated", fake_run_isolated)
    app = server.create_app(
        backend="cuda",
        max_workers=2,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
    )

    def failed_recovery_probe(*args, **kwargs):
        raise RuntimeError("device queue is unhealthy")

    monkeypatch.setattr(server, "probe_device", failed_recovery_probe)
    payload = _request().wire_payload()

    with TestClient(app) as client:
        response = client.post("/evaluate", json=payload)
        status = client.get("/status").json()

    assert response.status_code == 200
    assert (
        response.json()["status"]
        == EvaluationStatus.SUSPECTED_DEVICE_ERROR.value
    )
    assert status["scheduler"]["broken"] == 1
    assert status["scheduler"]["slots"][0]["state"] == "broken"
    assert status["scheduler"]["slots"][1]["state"] == "available"


def test_requests_fail_fast_after_every_slot_is_broken(tmp_path, monkeypatch):
    _patch_server_dependencies(monkeypatch)

    def fake_run_isolated(*args, **kwargs):
        raise IsolatedWorkerCrashed("exitcode=17")

    monkeypatch.setattr(server, "run_isolated", fake_run_isolated)
    app = server.create_app(
        backend="cuda",
        max_workers=4,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
    )

    def failed_recovery_probe(*args, **kwargs):
        raise RuntimeError("device queue is unhealthy")

    monkeypatch.setattr(server, "probe_device", failed_recovery_probe)
    payload = _request().wire_payload()

    with TestClient(app) as client:
        first = client.post("/evaluate", json=payload)
        second = client.post("/evaluate", json=payload)
        unavailable = client.post("/evaluate", json=payload)
        status = client.get("/status").json()

    assert first.status_code == 200
    assert (
        first.json()["status"]
        == EvaluationStatus.SUSPECTED_DEVICE_ERROR.value
    )
    assert second.status_code == 200
    assert (
        second.json()["status"]
        == EvaluationStatus.SUSPECTED_DEVICE_ERROR.value
    )
    assert unavailable.status_code == 503
    assert status["scheduler"]["healthy"] == 0
    assert status["scheduler"]["broken"] == 2
    assert status["scheduler"]["available"] == 0


def test_timeout_with_passing_probe_returns_without_cross_slot_retry(
    tmp_path,
    monkeypatch,
):
    _patch_server_dependencies(monkeypatch)
    devices = []

    def fake_run_isolated(
        operation,
        request,
        backend,
        device_string,
        timing,
    ):
        devices.append(device_string)
        raise TimeoutError("operator exceeded its budget")

    monkeypatch.setattr(server, "run_isolated", fake_run_isolated)
    app = server.create_app(
        backend="cuda",
        max_workers=2,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
    )
    payload = _request().wire_payload()

    with TestClient(app) as client:
        response = client.post("/evaluate", json=payload)
        status = client.get("/status").json()

    assert response.status_code == 200
    assert response.json()["status"] == EvaluationStatus.TIMEOUT.value
    assert devices == ["cuda:0"]
    assert status["status"] == "ok"
    assert status["scheduler"]["healthy"] == 2
    assert status["scheduler"]["checking"] == 0
    assert status["scheduler"]["broken"] == 0
    assert status["scheduler"]["incidents"] == 1
    assert status["scheduler"]["recovered"] == 1
    assert status["scheduler"]["available"] == 2


def test_repeated_timeouts_with_passing_probes_do_not_quarantine_slot(
    tmp_path,
    monkeypatch,
):
    _patch_server_dependencies(monkeypatch)
    devices = []

    def fake_run_isolated(
        operation,
        request,
        backend,
        device_string,
        timing,
    ):
        devices.append(device_string)
        if device_string == "cuda:0":
            raise TimeoutError("polluted runtime")
        return _passed(device_string, backend)

    monkeypatch.setattr(server, "run_isolated", fake_run_isolated)
    app = server.create_app(
        backend="cuda",
        max_workers=2,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
    )
    payload = _request().wire_payload()

    with TestClient(app) as client:
        responses = [
            client.post("/evaluate", json=payload)
            for _ in range(3)
        ]
        status = client.get("/status").json()
        final = client.post("/evaluate", json=payload)

    assert [response.json()["status"] for response in responses] == [
        EvaluationStatus.TIMEOUT.value,
        EvaluationStatus.PASSED.value,
        EvaluationStatus.TIMEOUT.value,
    ]
    assert devices[:3] == ["cuda:0", "cuda:1", "cuda:0"]
    assert status["status"] == "ok"
    assert status["scheduler"]["healthy"] == 2
    assert status["scheduler"]["broken"] == 0
    assert status["scheduler"]["slots"][0]["state"] == "available"
    assert final.json()["status"] == EvaluationStatus.PASSED.value
    assert devices[-1] == "cuda:1"


def test_timeout_does_not_consume_a_second_slot(
    tmp_path,
    monkeypatch,
):
    _patch_server_dependencies(monkeypatch)
    devices = []

    def fake_run_isolated(
        operation,
        request,
        backend,
        device_string,
        timing,
    ):
        devices.append(device_string)
        raise TimeoutError("operator exceeded its budget")

    monkeypatch.setattr(server, "run_isolated", fake_run_isolated)
    app = server.create_app(
        backend="cuda",
        max_workers=2,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
    )
    payload = _request().wire_payload()

    with TestClient(app) as client:
        response = client.post("/evaluate", json=payload)
        status = client.get("/status").json()

    assert response.status_code == 200
    assert response.json()["status"] == EvaluationStatus.TIMEOUT.value
    assert devices == ["cuda:0"]
    assert status["status"] == "ok"
    assert status["scheduler"]["healthy"] == 2
    assert status["scheduler"]["checking"] == 0
    assert status["scheduler"]["broken"] == 0
    assert status["scheduler"]["incidents"] == 1
    assert status["scheduler"]["recovered"] == 1
    assert status["scheduler"]["available"] == 2


def test_abnormal_worker_exit_with_passing_probe_is_not_retried(
    tmp_path,
    monkeypatch,
):
    _patch_server_dependencies(monkeypatch)
    devices = []

    def fake_run_isolated(
        operation,
        request,
        backend,
        device_string,
        timing,
    ):
        devices.append(device_string)
        raise IsolatedWorkerCrashed("exitcode=17")

    monkeypatch.setattr(server, "run_isolated", fake_run_isolated)
    app = server.create_app(
        backend="cuda",
        max_workers=2,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
    )
    payload = _request().wire_payload()

    with TestClient(app) as client:
        response = client.post("/evaluate", json=payload)
        status = client.get("/status").json()

    assert response.status_code == 500
    assert devices == ["cuda:0"]
    assert status["scheduler"]["incidents"] == 1
    assert status["scheduler"]["recovered"] == 1


def test_timeout_is_preserved_when_another_slot_is_already_broken(
    tmp_path,
    monkeypatch,
):
    def fake_probe(backend, device_string, timing, timeout_seconds):
        if device_string == "cuda:1":
            raise RuntimeError("device is unavailable")

    _patch_server_dependencies(monkeypatch, fake_probe)

    def fake_run_isolated(*args, **kwargs):
        raise TimeoutError("operator exceeded its budget")

    monkeypatch.setattr(server, "run_isolated", fake_run_isolated)
    app = server.create_app(
        backend="cuda",
        max_workers=2,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
    )
    payload = _request().wire_payload()

    with TestClient(app) as client:
        response = client.post("/evaluate", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == EvaluationStatus.TIMEOUT.value


def test_slot_is_not_reused_while_recovery_probe_is_running(
    tmp_path,
    monkeypatch,
):
    _patch_server_dependencies(monkeypatch)

    calls = 0

    def fake_run_isolated(
        operation,
        request,
        backend,
        device_string,
        timing,
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("operator exceeded its budget")
        return _passed(device_string, backend)

    monkeypatch.setattr(server, "run_isolated", fake_run_isolated)
    app = server.create_app(
        backend="cuda",
        max_workers=2,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
    )
    probe_started = threading.Event()
    release_probe = threading.Event()

    def blocking_recovery_probe(*args, **kwargs):
        probe_started.set()
        assert release_probe.wait(timeout=5)

    monkeypatch.setattr(server, "probe_device", blocking_recovery_probe)
    payload = _request().wire_payload()

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.post, "/evaluate", json=payload)
        assert probe_started.wait(timeout=5)
        status = client.get("/status").json()
        assert status["status"] == "degraded"
        assert status["scheduler"]["healthy"] == 1
        assert status["scheduler"]["checking"] == 1
        assert status["scheduler"]["broken"] == 0
        assert status["scheduler"]["active"] == 1
        assert status["scheduler"]["available"] == 1
        assert status["scheduler"]["slots"][0]["state"] == "checking"
        release_probe.set()
        response = future.result(timeout=5)

    assert response.status_code == 200
    assert response.json()["status"] == EvaluationStatus.TIMEOUT.value
    assert calls == 1

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

if find_spec("torch") is not None:
    import kernelgen_server.api.app as server
else:
    server = None
from kernelgen_server.profiling.process import ProfileProcessRunner, RequestDeadline
from kernelgen_server.protocol import client as protocol_client
from kernelgen_server.runtime.device_pool import NoHealthyDeviceError
from kernelgen_server.runtime.operations import (
    OperationCancelled,
    OperationRegistry,
    bind_operation,
    current_operation,
)
from kernelgen_server.schema import (
    BoundEvaluateRequest,
    EvaluateResponse,
    EvaluationStatus,
    EvaluatorBinding,
    Implementation,
    SourceFile,
    ReferenceRequest,
)


class _OneDevice:
    def count_devices_safe(self) -> int:
        return 1


def _request() -> BoundEvaluateRequest:
    return BoundEvaluateRequest(
        binding=EvaluatorBinding(
            catalog_name="simple-v6-test",
            definition="identity",
        ),
        implementation=Implementation(
            name="candidate",
            definition="identity",
            language="python",
            entrypoint="main.py::run",
            sources=[SourceFile(path="main.py", content="def run(x): return x")],
        ),
    )


def _passed(device: str, backend: str) -> EvaluateResponse:
    return EvaluateResponse(
        status=EvaluationStatus.PASSED,
        device=device,
        server_backend=backend,
        geo_mean=1.0,
        min_speedup=1.0,
        latency_ms=1.0,
        num_workloads=1,
        num_passed=1,
    )


def _patch_server(monkeypatch) -> None:
    monkeypatch.setattr(server, "_resolve_backend", lambda backend: "cuda")
    monkeypatch.setattr(server, "_make_device", lambda backend: _OneDevice())
    monkeypatch.setattr(server, "configure_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "runtime_device_type", lambda backend: "cuda")
    monkeypatch.setattr(server, "environment_info", lambda *args: {})
    monkeypatch.setattr(server, "probe_device", lambda *args, **kwargs: None)


def _wait_for_state(client: TestClient, operation_id: str, state: str) -> dict:
    deadline = time.monotonic() + 5
    while True:
        response = client.get(f"/operations/{operation_id}")
        if response.status_code == 200 and response.json()["state"] == state:
            return response.json()
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.mark.skipif(server is None, reason="torch is not installed")
def test_status_advertises_cancellable_operations(tmp_path, monkeypatch):
    _patch_server(monkeypatch)
    app = server.create_app(
        backend="cuda",
        max_workers=1,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
        operator_bundle_root=tmp_path / "bundles",
    )

    with TestClient(app) as client:
        capability = client.get("/status").json()["capabilities"][
            "operation_cancel"
        ]

    assert capability == {
        "enabled": True,
        "operations": ["preflight", "evaluate", "profile", "reference"],
    }


@pytest.mark.skipif(server is None, reason="torch is not installed")
@pytest.mark.parametrize("kind", ["preflight", "evaluate", "reference"])
def test_running_evaluation_can_be_cancelled_and_releases_healthy_slot(
    tmp_path,
    monkeypatch,
    kind,
):
    _patch_server(monkeypatch)
    started = threading.Event()

    def cancellable_runner(operation, request, backend, device, timing):
        control = current_operation()
        assert control is not None
        started.set()
        assert control.cancel_event.wait(timeout=5)
        control.raise_if_cancelled()

    monkeypatch.setattr(server, "run_isolated", cancellable_runner)
    app = server.create_app(
        backend="cuda",
        max_workers=1,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
        operator_bundle_root=tmp_path / "bundles",
    )
    operation_id = "running-evaluation"

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        response_future = executor.submit(
            client.post,
            f"/{kind}",
            json=(ReferenceRequest(binding=_request().binding, benchmark_fingerprint="fixture")
                  if kind == "reference" else _request()).wire_payload(),
            headers={"X-KernelGen-Operation-Id": operation_id},
        )
        assert started.wait(timeout=5)
        assert _wait_for_state(client, operation_id, "RUNNING")["device"] == "cuda:0"
        cancelled = client.delete(f"/operations/{operation_id}")
        response = response_future.result(timeout=5)
        terminal = client.get(f"/operations/{operation_id}").json()
        status = client.get("/status").json()

    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "CANCEL_REQUESTED"
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "OPERATION_CANCELLED"
    assert terminal["state"] == "CANCELLED"
    assert status["scheduler"]["active"] == 0
    assert status["scheduler"]["available"] == 1
    assert status["scheduler"]["broken"] == 0
    assert status["scheduler"]["incidents"] == 1
    assert status["scheduler"]["recovered"] == 1


@pytest.mark.skipif(server is None, reason="torch is not installed")
@pytest.mark.parametrize("queued_kind", ["evaluate", "reference"])
def test_queued_evaluation_cancels_without_starting_or_probing(tmp_path, monkeypatch, queued_kind):
    _patch_server(monkeypatch)
    first_started = threading.Event()
    release_first = threading.Event()
    invocations: list[str] = []

    def blocking_runner(operation, request, backend, device, timing):
        control = current_operation()
        assert control is not None
        invocations.append(control.operation_id)
        first_started.set()
        assert release_first.wait(timeout=5)
        return _passed(device, backend)

    monkeypatch.setattr(server, "run_isolated", blocking_runner)
    app = server.create_app(
        backend="cuda",
        max_workers=1,
        enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile",
        operator_bundle_root=tmp_path / "bundles",
    )

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            client.post,
            "/evaluate",
            json=_request().wire_payload(),
            headers={"X-KernelGen-Operation-Id": "first"},
        )
        assert first_started.wait(timeout=5)
        queued = executor.submit(
            client.post,
            f"/{queued_kind}",
            json=(ReferenceRequest(binding=_request().binding, benchmark_fingerprint="fixture")
                  if queued_kind == "reference" else _request()).wire_payload(),
            headers={"X-KernelGen-Operation-Id": "queued"},
        )
        _wait_for_state(client, "queued", "QUEUED")
        client.delete("/operations/queued")
        queued_response = queued.result(timeout=5)
        queued_terminal = client.get("/operations/queued").json()
        status_while_first_runs = client.get("/status").json()
        release_first.set()
        first_response = first.result(timeout=5)

    assert queued_response.status_code == 409
    assert queued_terminal["state"] == "CANCELLED"
    assert invocations == ["first"]
    assert status_while_first_runs["scheduler"]["incidents"] == 0
    assert first_response.status_code == 200


@pytest.mark.skipif(server is None, reason="torch is not installed")
@pytest.mark.parametrize("kind", ["preflight", "evaluate"])
@pytest.mark.parametrize("error,http_status", [(NoHealthyDeviceError, 503), (RuntimeError, 500)])
def test_evaluation_routes_share_error_and_terminal_mapping(tmp_path, monkeypatch, kind, error, http_status):
    _patch_server(monkeypatch)

    async def fail(self, operation_kind, request, operation):
        assert operation_kind == kind
        raise error("test failure")

    monkeypatch.setattr(server.EvaluationExecutor, "run", fail)
    app = server.create_app(
        backend="cuda", max_workers=1, enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profile", operator_bundle_root=tmp_path / "bundles",
    )
    with TestClient(app) as client:
        response = client.post(
            f"/{kind}", json=_request().wire_payload(),
            headers={"X-KernelGen-Operation-Id": "failed-operation"},
        )
        terminal = client.get("/operations/failed-operation").json()
    assert response.status_code == http_status
    assert response.json()["detail"] == "test failure"
    assert terminal["state"] == "FAILED"


def test_profile_process_runner_stops_stage_after_cancellation(tmp_path):
    operation = OperationRegistry().create("profile", "profile-stage")
    marker = tmp_path / "started"
    failures: list[BaseException] = []

    def run_stage() -> None:
        try:
            with bind_operation(operation):
                ProfileProcessRunner(RequestDeadline.from_timeout(30)).run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; import time; "
                            f"Path({str(marker)!r}).write_text('started'); "
                            "time.sleep(30)"
                        ),
                    ],
                    label="cancellable profile stage",
                    cwd=tmp_path,
                    env={},
                )
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=run_stage)
    worker.start()
    deadline = time.monotonic() + 5
    while not marker.is_file():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    operation.request_cancel()
    worker.join(timeout=5)

    assert worker.is_alive() is False
    assert len(failures) == 1
    assert isinstance(failures[0], OperationCancelled)


def test_protocol_client_sends_operation_id_and_decodes_cancellation(monkeypatch):
    class CancelledResponse:
        status_code = 409
        text = "cancelled"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def json():
            return {
                "detail": {
                    "code": "OPERATION_CANCELLED",
                    "operation_id": "op-1",
                    "operation": "evaluate",
                }
            }

    observed = {}

    class Session:
        @staticmethod
        def request(method, url, **kwargs):
            observed.update(method=method, url=url, **kwargs)
            return CancelledResponse()

    monkeypatch.setattr(protocol_client, "_SESSION", Session())

    with pytest.raises(protocol_client.OperationCancelledError) as captured:
        protocol_client._post(
            "http://server:8000",
            "/evaluate",
            {"request": "payload"},
            30,
            operation_id="op-1",
        )

    assert captured.value.operation_id == "op-1"
    assert captured.value.operation == "evaluate"
    assert observed["headers"]["X-KernelGen-Operation-Id"] == "op-1"

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kernelgen_server.protocol import client
from kernelgen_server.debug.jobs import (
    DebugArtifact,
    DebugJob,
    DebugJobRequest,
)


def job_payload(status: str = "QUEUED") -> dict:
    return {
        "api_version": "v6.0",
        "job_id": "job/1",
        "status": status,
        "backend": "thead",
        "created_at": 1.0,
        "timeout_seconds": 30,
    }


def test_debug_client_uses_lifecycle_routes(monkeypatch):
    calls = []

    def fake_request(server_url, path, **kwargs):
        calls.append((server_url, path, kwargs))
        if kwargs["method"] == "DELETE":
            return job_payload("CANCELLED")
        if kwargs["method"] == "POST":
            return job_payload("SUCCEEDED")
        return job_payload("SUCCEEDED")

    monkeypatch.setattr(client, "_request_json", fake_request)
    request = DebugJobRequest(command=["python3", "debug.py"])

    submitted = client.submit_debug_job(request, "http://server")
    fetched = client.get_debug_job(
        submitted.job_id,
        "http://server",
        wait_seconds=12.5,
    )
    cancelled = client.cancel_debug_job(submitted.job_id, "http://server")

    assert submitted.status == "SUCCEEDED"
    assert fetched.status == "SUCCEEDED"
    assert cancelled.status == "CANCELLED"
    assert calls[0][1:] == (
        "/debug/jobs",
        {
            "method": "POST",
            "payload": request.model_dump(mode="json"),
            "timeout": 310,
        },
    )
    assert calls[1][1] == "/debug/jobs/job%2F1"
    assert calls[1][2]["query"] == {"wait_seconds": 12.5}
    assert calls[2][1:] == (
        "/debug/jobs/job%2F1",
        {"method": "DELETE", "timeout": 30},
    )


@pytest.mark.parametrize("service_status", ["ok", "degraded"])
def test_status_accepts_usable_service_states(monkeypatch, service_status):
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *args, **kwargs: {"status": service_status},
    )

    assert client.status("http://server")["status"] == service_status


def test_status_rejects_unknown_service_state(monkeypatch):
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *args, **kwargs: {"status": "stopped"},
    )

    with pytest.raises(client.ServerError, match="invalid status response"):
        client.status("http://server")


def test_download_debug_artifacts_verifies_content(
    tmp_path: Path,
    monkeypatch,
):
    content = b"profiling evidence"
    artifact = DebugArtifact(
        id="a001",
        path="reports/result.txt",
        media_type="text/plain",
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        download_url="/ignored",
    )
    job = DebugJob(
        job_id="job-1",
        status="SUCCEEDED",
        backend="cuda",
        created_at=1,
        timeout_seconds=30,
        artifacts=[artifact],
    )
    requests = []

    class FakeResponse:
        status_code = 200
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def iter_content(self, *, chunk_size):
            assert chunk_size == 1024 * 1024
            yield content

    def fake_get(url, **kwargs):
        requests.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr(client._SESSION, "get", fake_get)

    downloaded = client.download_debug_artifacts(
        job,
        tmp_path,
        "http://server",
    )

    assert downloaded["a001"].read_bytes() == content
    assert requests == [(
        "http://server/debug/jobs/job-1/artifacts/a001",
        {
            "timeout": 600,
            "stream": True,
            "allow_redirects": False,
        },
    )]

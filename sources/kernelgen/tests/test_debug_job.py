from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelgen_client import http as client
from kernelgen_server.debug.jobs import DebugArtifact, DebugJob
import kernelgen.tools.debug_job as debug_job_module
from kernelgen.data.tool_context import ToolContext
from kernelgen.tools.debug_job import (
    WorkspaceDebugJobRequest,
    cancel_workspace_debug_job,
    get_workspace_debug_job,
    submit_workspace_debug_job,
)


_JOB_ID = "a" * 32


def _context() -> ToolContext:
    return ToolContext(
        definition="gelu",
        target_hardware="Ascend910B",
        eval_server_url="http://eval:8002",
        catalog_name="flaggems-v5",
    )


def _request() -> WorkspaceDebugJobRequest:
    return WorkspaceDebugJobRequest(
        purpose="compare row-wise intermediate values",
        command=["{python}", "tmp/debug/check.py"],
        files=[
            {"path": "tmp/debug/check.py"},
            {"path": "tmp/main.py"},
        ],
        timeout_seconds=30,
    )


def _job(
    status: str,
    *,
    stdout: str = "",
    artifacts: list[DebugArtifact] | None = None,
) -> DebugJob:
    return DebugJob(
        job_id=_JOB_ID,
        status=status,
        backend="npu",
        created_at=1.0,
        timeout_seconds=30,
        stdout=stdout,
        artifacts=artifacts or [],
    )


def _write_debug_sources(tmp_path: Path) -> None:
    (tmp_path / "tmp" / "debug").mkdir(parents=True)
    (tmp_path / "tmp" / "debug" / "check.py").write_text(
        "print('debug-ok')\n",
        encoding="utf-8",
    )
    (tmp_path / "tmp" / "main.py").write_text(
        "def run(): pass\n",
        encoding="utf-8",
    )


def test_submit_snapshots_exact_workspace_files(tmp_path: Path, monkeypatch):
    _write_debug_sources(tmp_path)
    captured = {}
    monkeypatch.setattr(
        client,
        "status",
        lambda server_url: {
            "status": "ok",
            "backend": "npu",
            "debug": {"enabled": True},
        },
    )

    def submit(request, server_url, *, timeout):
        captured.update(
            request=request,
            server_url=server_url,
            timeout=timeout,
        )
        return _job("SUCCEEDED")

    monkeypatch.setattr(client, "submit_debug_job", submit)

    result = submit_workspace_debug_job(tmp_path, _context(), _request())

    assert result["job_id"] == _JOB_ID
    assert result["status"] == "SUCCEEDED"
    assert captured["server_url"] == "http://eval:8002"
    assert captured["timeout"] == 1800
    assert captured["request"].command == [
        "{python}",
        "tmp/debug/check.py",
    ]
    assert captured["request"].files[0].path == "tmp/debug/check.py"
    job_dir = tmp_path / ".kernelgen" / "debug-jobs" / _JOB_ID
    request = json.loads((job_dir / "request.json").read_text())
    assert request["purpose"] == "compare row-wise intermediate values"
    assert request["server_backend"] == "npu"
    assert request["files"][0]["sha256"]
    assert (
        job_dir / "sources" / "tmp" / "debug" / "check.py"
    ).read_text() == "print('debug-ok')\n"


def test_submit_downloads_terminal_artifacts(tmp_path: Path, monkeypatch):
    _write_debug_sources(tmp_path)
    artifact = DebugArtifact(
        id="a001",
        path="numerics.json",
        media_type="application/json",
        size_bytes=2,
        sha256="0" * 64,
        download_url=f"/debug/jobs/{_JOB_ID}/artifacts/a001",
    )
    monkeypatch.setattr(
        client,
        "status",
        lambda server_url: {
            "status": "ok",
            "backend": "npu",
            "debug": {"enabled": True},
        },
    )
    monkeypatch.setattr(
        client,
        "submit_debug_job",
        lambda request, server_url, *, timeout: _job(
            "SUCCEEDED",
            artifacts=[artifact],
        ),
    )

    def download(job, output_dir, server_url):
        path = Path(output_dir) / "numerics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return {"a001": path}

    monkeypatch.setattr(client, "download_debug_artifacts", download)

    result = submit_workspace_debug_job(tmp_path, _context(), _request())

    assert result["artifacts"][0]["local_path"].endswith(
        "artifacts/numerics.json"
    )
    assert (
        tmp_path
        / ".kernelgen"
        / "debug-jobs"
        / _JOB_ID
        / "artifacts"
        / "numerics.json"
    ).read_text() == "{}"


def test_submit_requires_enabled_service_capability(tmp_path: Path, monkeypatch):
    _write_debug_sources(tmp_path)
    monkeypatch.setattr(
        client,
        "status",
        lambda server_url: {
            "status": "ok",
            "backend": "npu",
            "debug": {"enabled": False},
        },
    )
    called = False

    def submit(*args, **kwargs):
        nonlocal called
        called = True
        return _job("QUEUED")

    monkeypatch.setattr(client, "submit_debug_job", submit)

    with pytest.raises(RuntimeError, match="--enable-debug-jobs"):
        submit_workspace_debug_job(tmp_path, _context(), _request())
    assert called is False


def test_submit_rejects_wrong_server_backend(tmp_path: Path, monkeypatch):
    _write_debug_sources(tmp_path)
    monkeypatch.setattr(
        client,
        "status",
        lambda server_url: {
            "status": "ok",
            "backend": "cuda",
            "debug": {"enabled": True},
        },
    )

    with pytest.raises(RuntimeError, match="maps to backend 'npu'"):
        submit_workspace_debug_job(tmp_path, _context(), _request())


def test_submit_requires_one_terminal_server_response(
    tmp_path: Path,
    monkeypatch,
):
    _write_debug_sources(tmp_path)
    monkeypatch.setattr(
        client,
        "status",
        lambda server_url: {
            "status": "ok",
            "backend": "npu",
            "debug": {"enabled": True},
        },
    )
    monkeypatch.setattr(
        client,
        "submit_debug_job",
        lambda request, server_url, *, timeout: _job("QUEUED"),
    )

    with pytest.raises(RuntimeError, match="non-terminal Debug Job"):
        submit_workspace_debug_job(tmp_path, _context(), _request())


def test_get_downloads_artifacts_and_bounds_tool_output(
    tmp_path: Path,
    monkeypatch,
):
    job_dir = tmp_path / ".kernelgen" / "debug-jobs" / _JOB_ID
    job_dir.mkdir(parents=True)
    (job_dir / "request.json").write_text("{}\n", encoding="utf-8")
    long_output = "x" * (40 * 1024)
    artifact = DebugArtifact(
        id="a001",
        path="numerics.json",
        media_type="application/json",
        size_bytes=2,
        sha256="0" * 64,
        download_url=f"/debug/jobs/{_JOB_ID}/artifacts/a001",
    )
    monkeypatch.setattr(
        client,
        "get_debug_job",
        lambda job_id, server_url, **kwargs: _job(
            "SUCCEEDED",
            stdout=long_output,
            artifacts=[artifact],
        ),
    )

    def download(job, output_dir, server_url):
        path = Path(output_dir) / "numerics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return {"a001": path}

    monkeypatch.setattr(client, "download_debug_artifacts", download)

    result = get_workspace_debug_job(
        tmp_path,
        _context(),
        _JOB_ID,
        wait_seconds=12,
    )

    assert result["status"] == "SUCCEEDED"
    assert result["client_output_truncated"] is True
    assert len(result["stdout"]) == 32 * 1024
    assert result["artifacts"][0]["local_path"].endswith(
        "artifacts/numerics.json"
    )
    assert (job_dir / "stdout.log").read_text() == long_output


def test_get_waits_across_service_long_poll_windows(
    tmp_path: Path,
    monkeypatch,
):
    job_dir = tmp_path / ".kernelgen" / "debug-jobs" / _JOB_ID
    job_dir.mkdir(parents=True)
    (job_dir / "request.json").write_text("{}\n", encoding="utf-8")
    clock = [100.0]
    waits = []
    statuses = iter(["RUNNING", "RUNNING", "SUCCEEDED"])

    monkeypatch.setattr(
        debug_job_module.time,
        "monotonic",
        lambda: clock[0],
    )

    def get(job_id, server_url, *, wait_seconds):
        waits.append(wait_seconds)
        clock[0] += wait_seconds
        return _job(next(statuses))

    monkeypatch.setattr(client, "get_debug_job", get)

    result = get_workspace_debug_job(
        tmp_path,
        _context(),
        _JOB_ID,
        wait_seconds=75,
    )

    assert result["status"] == "SUCCEEDED"
    assert waits == [30.0, 30.0, 15.0]


def test_get_rejects_wait_beyond_job_limit(tmp_path: Path):
    job_dir = tmp_path / ".kernelgen" / "debug-jobs" / _JOB_ID
    job_dir.mkdir(parents=True)
    (job_dir / "request.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="between 0 and 1800"):
        get_workspace_debug_job(
            tmp_path,
            _context(),
            _JOB_ID,
            wait_seconds=1801,
        )


def test_job_access_is_bound_to_workspace(tmp_path: Path, monkeypatch):
    called = False

    def get(*args, **kwargs):
        nonlocal called
        called = True
        return _job("RUNNING")

    monkeypatch.setattr(client, "get_debug_job", get)
    with pytest.raises(ValueError, match="not submitted by this workspace"):
        get_workspace_debug_job(tmp_path, _context(), _JOB_ID)
    assert called is False


def test_cancel_owned_job(tmp_path: Path, monkeypatch):
    job_dir = tmp_path / ".kernelgen" / "debug-jobs" / _JOB_ID
    job_dir.mkdir(parents=True)
    (job_dir / "request.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        client,
        "cancel_debug_job",
        lambda job_id, server_url: _job("CANCELLED"),
    )

    result = cancel_workspace_debug_job(
        tmp_path,
        _context(),
        _JOB_ID,
    )

    assert result["status"] == "CANCELLED"


def test_request_rejects_files_outside_tmp():
    with pytest.raises(ValueError, match="under tmp"):
        WorkspaceDebugJobRequest(
            purpose="unsafe",
            command=["{python}", "outside.py"],
            files=[{"path": "outside.py"}],
        )


def test_request_rejects_reserved_environment():
    with pytest.raises(ValueError, match="reserved debug environment"):
        WorkspaceDebugJobRequest(
            purpose="unsafe",
            command=["{python}", "tmp/debug/check.py"],
            files=[{"path": "tmp/debug/check.py"}],
            env={"CUDA_VISIBLE_DEVICES": "7"},
        )

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from kernelgen_server.debug.jobs import (
    DebugJobRequest,
    DebugJobStore,
    DebugSourceFile,
    LocalDebugJobRunner,
)


def request(script: str, **overrides) -> DebugJobRequest:
    data = {
        "command": [sys.executable, "debug.py"],
        "files": [{"path": "debug.py", "content": script}],
        "timeout_seconds": 5,
    }
    data.update(overrides)
    return DebugJobRequest.model_validate(data)


def nested_child_script(
    *, parent_sleep_seconds: int, child_pid_path: Path | None = None
) -> str:
    child_script = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print(os.getpid(), flush=True)\n"
        "time.sleep(30)\n"
    )
    script = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_script!r}], "
        "stdout=subprocess.PIPE, text=True)\n"
        "child_pid = child.stdout.readline().strip()\n"
        "print(f'CHILD_PID={child_pid}', flush=True)\n"
    )
    if child_pid_path is not None:
        script += f"Path({str(child_pid_path)!r}).write_text(child_pid)\n"
    return script + f"time.sleep({parent_sleep_seconds})\n"


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if sys.platform == "linux":
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return False
        state = stat[stat.rfind(")") + 2 :].split(maxsplit=1)[0]
        return state != "Z"
    return True


def assert_process_stopped(pid: int) -> None:
    deadline = time.time() + 3
    while time.time() < deadline and process_is_running(pid):
        time.sleep(0.02)
    try:
        assert not process_is_running(pid)
    finally:
        if process_is_running(pid):
            os.kill(pid, signal.SIGKILL)


def test_request_rejects_unsafe_paths_and_reserved_environment():
    with pytest.raises(ValidationError):
        DebugSourceFile(path="../escape.py", content="pass")
    with pytest.raises(ValidationError):
        DebugJobRequest(
            command=["python3", "main.py"],
            env={"KGS_DEVICE": "cuda:7"},
        )
    with pytest.raises(ValidationError):
        DebugJobRequest(
            command=["python3", "main.py"],
            env={"CUDA_VISIBLE_DEVICES": "7"},
        )


def test_store_enforces_total_source_limit(tmp_path: Path):
    store = DebugJobStore(tmp_path, backend="cuda", max_source_bytes=4)
    with pytest.raises(ValueError, match="exceed"):
        store.create(request("print('too large')"))


def test_runner_binds_thead_device_and_collects_artifact(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    job_request = request(
        """
import json
import os
from pathlib import Path

payload = {
    "assigned": os.environ["KGS_ASSIGNED_DEVICE"],
    "backend": os.environ["KGS_BACKEND"],
    "device": os.environ["KGS_DEVICE"],
    "visible": os.environ["CUDA_VISIBLE_DEVICES"],
    "custom": os.environ["EXPERIMENT"],
}
artifact = Path(os.environ["KGS_DEBUG_ARTIFACTS"]) / "result.json"
artifact.write_text(json.dumps(payload))
print("debug-ok")
""",
        env={"EXPERIMENT": "block-size-128"},
    )
    store = DebugJobStore(tmp_path, backend="thead")
    job = store.create(job_request)
    assert store.begin(job.job_id, "cuda:1")
    runner = LocalDebugJobRunner(tmp_path, backend="thead")

    result = runner.run(job.job_id, job_request, device="cuda:1")
    completed = store.finish(job.job_id, result)

    assert completed.status == "SUCCEEDED"
    assert completed.stdout.strip() == "debug-ok"
    assert len(completed.artifacts) == 1
    artifact_path = store.artifact_path(
        job.job_id, completed.artifacts[0].id
    )
    assert json.loads(artifact_path.read_text()) == {
        "assigned": "cuda:1",
        "backend": "thead",
        "device": "cuda:0",
        "visible": "6",
        "custom": "block-size-128",
    }


def test_runner_reports_nonzero_exit(tmp_path: Path):
    job_request = request("raise RuntimeError('broken experiment')")
    store = DebugJobStore(tmp_path, backend="cuda")
    job = store.create(job_request)
    store.begin(job.job_id, "cuda:0")
    runner = LocalDebugJobRunner(tmp_path, backend="cuda")

    result = runner.run(job.job_id, job_request, device="cuda:0")

    assert result.status == "FAILED"
    assert result.exit_code != 0
    assert "broken experiment" in result.stderr


def test_runner_resolves_server_python_placeholder(tmp_path: Path):
    job_request = request(
        "import os, sys\n"
        "assert sys.executable == os.environ['KGS_PYTHON']\n"
        "print(sys.executable)",
        command=["{python}", "debug.py"],
    )
    store = DebugJobStore(tmp_path, backend="cuda")
    job = store.create(job_request)
    store.begin(job.job_id, "cuda:0")
    runner = LocalDebugJobRunner(tmp_path, backend="cuda")

    result = runner.run(job.job_id, job_request, device="cuda:0")

    assert result.status == "SUCCEEDED"
    assert result.stdout.strip() == sys.executable


@pytest.mark.parametrize("override", [None, "/opt/dtk-job"])
@pytest.mark.parametrize("inherited", [None, "/opt/dtk"])
def test_runner_preserves_hygon_device_library_path(tmp_path: Path, monkeypatch, inherited, override):
    if inherited is None:
        monkeypatch.delenv("ROCM_PATH", raising=False)
    else:
        monkeypatch.setenv("ROCM_PATH", inherited)
    monkeypatch.setenv("UNRELATED_DEPLOY_SECRET", "must-not-inherit")
    monkeypatch.setenv("ROCM_UNRELATED_SECRET", "must-not-inherit")
    req = request(
        "import json, os\n"
        "print(json.dumps([os.environ.get('ROCM_PATH'), "
        "any(name in os.environ for name in ['UNRELATED_DEPLOY_SECRET', 'ROCM_UNRELATED_SECRET'])]))\n",
        env={} if override is None else {"ROCM_PATH": override},
    )
    store = DebugJobStore(tmp_path, backend="hygon")
    job = store.create(req)
    store.begin(job.job_id, "cuda:0")
    runner = LocalDebugJobRunner(tmp_path, backend="hygon")
    result = runner.run(job.job_id, req, device="cuda:0")
    assert result.status == "SUCCEEDED"
    assert json.loads(result.stdout) == [override if override is not None else inherited, False]


def test_runner_inherits_metax_vendor_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MACA_PATH", "/opt/maca")
    job_request = request(
        "import os\nprint(os.environ['MACA_PATH'])",
    )
    store = DebugJobStore(tmp_path, backend="metax")
    job = store.create(job_request)
    store.begin(job.job_id, "cuda:0")
    runner = LocalDebugJobRunner(tmp_path, backend="metax")

    result = runner.run(job.job_id, job_request, device="cuda:0")

    assert result.status == "SUCCEEDED"
    assert result.stdout.strip() == "/opt/maca"


def test_runner_times_out_process_group(tmp_path: Path):
    job_request = request(
        nested_child_script(parent_sleep_seconds=30),
        timeout_seconds=1,
    )
    store = DebugJobStore(tmp_path, backend="cuda")
    job = store.create(job_request)
    store.begin(job.job_id, "cuda:0")
    runner = LocalDebugJobRunner(tmp_path, backend="cuda")

    result = runner.run(job.job_id, job_request, device="cuda:0")

    assert result.status == "TIMEOUT"
    assert "exceeded 1 seconds" in result.error
    child_pid = int(result.stdout.strip().removeprefix("CHILD_PID="))
    assert_process_stopped(child_pid)


def test_runner_cleans_process_group_after_parent_exits(tmp_path: Path):
    job_request = request(nested_child_script(parent_sleep_seconds=0))
    store = DebugJobStore(tmp_path, backend="cuda")
    job = store.create(job_request)
    store.begin(job.job_id, "cuda:0")
    runner = LocalDebugJobRunner(tmp_path, backend="cuda")

    result = runner.run(job.job_id, job_request, device="cuda:0")

    assert result.status == "SUCCEEDED"
    child_pid = int(result.stdout.strip().removeprefix("CHILD_PID="))
    assert_process_stopped(child_pid)


def test_running_job_can_be_cancelled(tmp_path: Path):
    child_pid_path = tmp_path / "child.pid"
    job_request = request(
        nested_child_script(
            parent_sleep_seconds=30,
            child_pid_path=child_pid_path,
        ),
        timeout_seconds=10,
    )
    store = DebugJobStore(tmp_path, backend="cuda")
    job = store.create(job_request)
    store.begin(job.job_id, "cuda:0")
    runner = LocalDebugJobRunner(tmp_path, backend="cuda")
    result_holder = {}
    thread = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result",
            runner.run(job.job_id, job_request, device="cuda:0"),
        )
    )
    thread.start()

    deadline = time.time() + 3
    while time.time() < deadline and not child_pid_path.exists():
        time.sleep(0.02)
    runner.cancel(job.job_id)
    thread.join(timeout=3)

    assert child_pid_path.exists()
    assert not thread.is_alive()
    result = result_holder["result"]
    assert result.status == "CANCELLED"
    child_pid = int(child_pid_path.read_text())
    assert_process_stopped(child_pid)

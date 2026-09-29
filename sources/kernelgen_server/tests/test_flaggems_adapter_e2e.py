"""Opt-in local-container E2E for the FlagGems adapter."""

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from kernelgen_server.evaluation.adapters import create_adapter
from kernelgen_server.profiling import ProfileOptions, ProfileRequest
from kernelgen_server.runtime.isolated import run_isolated
from kernelgen_server.schema import (
    BoundEvaluateRequest,
    EvaluateResponse,
    EvaluatorBinding,
    Implementation,
    SourceFile,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("KGS_RUN_FLAGGEMS_E2E") != "1",
    reason="set KGS_RUN_FLAGGEMS_E2E=1 inside the FlagGems GPU container",
)


def _request() -> BoundEvaluateRequest:
    return BoundEvaluateRequest(
        binding=EvaluatorBinding(
            catalog_name="flaggems-adapter-definitions",
            definition="addmm_",
        ),
        implementation=Implementation(
            name="flaggems-reference-candidate",
            definition="addmm_",
            language="python",
            entrypoint="candidate.py::run",
            sources=[
                SourceFile(
                    path="candidate.py",
                    content=(
                        "from flag_gems.ops.addmm_ import addmm_ as _addmm_\n\n"
                        "def run(self, mat1, mat2, *, beta=1, alpha=1):\n"
                        "    return _addmm_(self, mat1, mat2, beta=beta, alpha=alpha)\n"
                    ),
                )
            ],
        ),
        settings={
            "warmup_ms": 5,
            "benchmark_ms": 5,
            "timeout_seconds": 600,
        },
    )


def _acosh_request() -> BoundEvaluateRequest:
    return BoundEvaluateRequest(
        binding=EvaluatorBinding(
            catalog_name="flaggems-adapter-definitions",
            definition="acosh",
        ),
        implementation=Implementation(
            name="flaggems-reference-candidate",
            definition="acosh",
            language="python",
            entrypoint="candidate.py::run",
            sources=[
                SourceFile(
                    path="candidate.py",
                    content=(
                        "from flag_gems.ops.acosh import acosh as _acosh\n\n"
                        "def run(A):\n"
                        "    return _acosh(A)\n"
                    ),
                )
            ],
        ),
        settings={"timeout_seconds": 600},
    )


def test_flaggems_generic_unary_preflight_uses_candidate_override():
    result = run_isolated(
        "preflight", _acosh_request(), "cuda", "cuda:0", "triton"
    )
    assert result["status"] == "PASSED"
    assert result["num_cases"] > 0


def test_flaggems_addmm_profile_command_replays_one_exact_case(tmp_path):
    request = _request()
    adapter = create_adapter(
        request.binding,
        device_string="cuda:0",
        backend="cuda",
    )
    manifest = adapter.inspect()
    profile = ProfileRequest(
        binding=request.binding,
        implementation=request.implementation,
        benchmark_fingerprint=manifest.benchmark_fingerprint,
        case_id=manifest.case_list.cases[0].case_id,
        expected_backend="cuda",
        options=ProfileOptions(warmup=2, iterations=3),
    )
    command = adapter.build_profile_command(
        profile,
        profile.options,
        tmp_path,
        "cuda:0",
    )
    process = subprocess.run(
        command.argv,
        cwd=command.cwd,
        env=command.env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert process.returncode == 0, (process.stdout + process.stderr)[-8000:]
    assert "--candidate-call-count-path" not in command.argv
    assert "--candidate-code-path" not in command.argv
    assert "--override" in command.argv
    assert command.completion_marker_path.endswith("target/runner-completed")
    assert "kernelgen_server.profiling.gems_runner" in command.argv


def test_flaggems_addmm_isolated_preflight_and_evaluate():
    request = _request()
    preflight = run_isolated("preflight", request, "cuda", "cuda:0", "triton")
    assert preflight["status"] == "PASSED"
    assert preflight["num_cases"] == 15

    result = run_isolated("evaluate", request, "cuda", "cuda:0", "triton")
    assert isinstance(result, EvaluateResponse)
    assert set(result.model_dump()) == set(EvaluateResponse.model_fields)
    assert result.status == "PASSED"
    assert result.num_workloads == 51
    assert sum(item.phase == "correctness" for item in result.per_workload) == 36
    assert sum(item.phase == "timing" for item in result.per_workload) == 15


def _addmm_pytest_pids() -> set[int]:
    matches: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"pytest" in command and b"benchmark/test_addmm_.py" in command:
            matches.add(int(entry.name))
    return matches


def test_flaggems_timeout_reaps_pytest_process_group():
    request = _request().model_copy(
        update={
            "settings": _request().settings.model_copy(
                update={"timeout_seconds": 30}
            )
        }
    )
    seen: set[int] = set()
    stop = threading.Event()
    temp_before = {
        path
        for pattern in ("kernelgen-isolated-*", "kernelgen-flaggems-*")
        for path in Path("/tmp").glob(pattern)
    }

    def monitor() -> None:
        while not stop.wait(0.02):
            seen.update(_addmm_pytest_pids())

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with pytest.raises(TimeoutError, match="preflight timed out"):
            run_isolated("preflight", request, "cuda", "cuda:0", "triton")
    finally:
        stop.set()
        thread.join(timeout=2)

    assert seen, "timeout occurred before the FlagGems pytest child was observed"
    deadline = time.monotonic() + 5
    alive = seen & _addmm_pytest_pids()
    while alive and time.monotonic() < deadline:
        time.sleep(0.05)
        alive = seen & _addmm_pytest_pids()
    assert not alive, f"FlagGems pytest processes survived timeout: {sorted(alive)}"
    temp_after = {
        path
        for pattern in ("kernelgen-isolated-*", "kernelgen-flaggems-*")
        for path in Path("/tmp").glob(pattern)
    }
    assert temp_after == temp_before

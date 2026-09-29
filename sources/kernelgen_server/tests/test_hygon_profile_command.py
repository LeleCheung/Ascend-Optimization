"""The vendor wrapper must preserve evaluator-owned relative paths."""
import sys

import pytest

from kernelgen_server.profiling.hygon.dcu import HipprofProfiler
from kernelgen_server.profiling.models import ProfileCommand, ProfileOptions
from kernelgen_server.profiling.process import RequestDeadline


@pytest.mark.parametrize("tool", ["hipprof", "rocprof"])
@pytest.mark.parametrize("relative_input", [True, False], ids=["gems", "native"])
@pytest.mark.parametrize("child_fails", [False, True], ids=["success", "failure"])
def test_hygon_preserves_evaluator_cwd(monkeypatch, tmp_path, tool, relative_input, child_fails):
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    fixture = evaluator / "case.txt"
    fixture.write_text("candidate input")
    unrelated = evaluator / "pmc_results_999999999.csv"
    unrelated.write_text("another job")
    artifacts = tmp_path / "profile"
    marker = artifacts / "runner-completed"
    selected = fixture.name if relative_input else str(fixture)
    command = ProfileCommand(
        argv=(sys.executable, "-c", (
            "from pathlib import Path; "
            f"assert Path({selected!r}).read_text() == 'candidate input'; "
            + ("raise SystemExit(42); " if child_fails else "") +
            f"Path({str(marker)!r}).write_text('completed')"
        )),
        cwd=str(evaluator),
        completion_marker_path=str(marker),
    )
    # Model hipprof's misleading exit code: the wrapper exits zero even if
    # the evaluator failed. The real completion marker must still be required.
    wrapper = tmp_path / tool
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "output = Path(args[args.index('-o') + 1])\n"
        "child = subprocess.run(args[args.index(sys.executable):], check=False)\n"
        "print(f'child-status={child.returncode}')\n"
        "if Path(sys.argv[0]).name == 'hipprof':\n"
        "    print(f\"HIP_PROF:process id '{os.getpid()}'\")\n"
        "    Path(f'pmc_results_{os.getpid()}.csv').write_text(\n"
        "        'kernel-name,dispatch,gpu-id,time,VALUBusy\\n'\n"
        "        'candidate,0,0,0.000009,1\\ncandidate,1,0,0.000005,25\\n')\n"
        "output.write_text('Name,Calls,AverageNs,DurationNs\\ncandidate,2,5000,5000\\n')\n"
    )
    wrapper.chmod(0o755)
    profiler = HipprofProfiler()
    monkeypatch.setattr(profiler, "_detect_profiler_tool", lambda options: (str(wrapper), tool))
    monkeypatch.setattr(profiler, "_find_objdump", lambda options: "")

    result = profiler.profile_command(
        command, ProfileOptions(warmup=1, iterations=1, timeout_sec=10),
        artifacts, "cuda:0", deadline=RequestDeadline.from_timeout(10),
    )

    assert (artifacts / f"{tool}-output" / "results.csv").is_file()
    assert not (evaluator / "results.csv").exists()
    assert unrelated.read_text() == "another job"
    assert list(evaluator.glob("pmc_results_*.csv")) == [unrelated]
    if child_fails:
        assert result.status == "failed"
        assert "completion marker" in result.error
        assert not marker.exists()
        assert "child-status=42" in (artifacts / f"{tool}.log").read_text()
    else:
        assert result.status == "completed", result.error
        assert marker.is_file()
        assert result.metrics["operation_count"] == 1
        if tool == "hipprof":
            pmc = result.metrics["performance_counters"]
            assert pmc["record_count"] == 1
            assert pmc["records"][0]["counters"] == {"VALUBusy": 25}
            assert pmc["records"][0]["duration_us"] == 5


def test_hygon_matches_demangled_pmc_names_without_fuzzy_matching():
    from kernelgen_server.profiling.hygon.dcu_report import select_measured_hipprof_pmc_records

    pmc = {"records": [
        {"kernel_name": "_Z9candidatev", "demangled_name": "candidate()", "dispatch": 0},
        {"kernel_name": "_Z9candidatev", "demangled_name": "candidate()", "dispatch": 1},
        {"kernel_name": "_Z9candidatei", "demangled_name": "candidate(int)", "dispatch": 2},
    ]}
    parsed = {"ops": [{"op_name": "candidate()", "measured_invocations": 1}]}
    selected = select_measured_hipprof_pmc_records(pmc, parsed)
    assert selected["records"] == [pmc["records"][1]]
    assert select_measured_hipprof_pmc_records(pmc, {"ops": [{"op_name": "candidate", "measured_invocations": 1}]}) is None


def test_hygon_recovers_pmc_across_filesystems(monkeypatch, tmp_path):
    import errno
    import os
    from types import SimpleNamespace

    evaluator = tmp_path / "gems"
    evaluator.mkdir()
    output = tmp_path / "artifacts"
    output.mkdir()
    source = evaluator / "pmc_results_123.csv"
    source.write_text("measured counters")

    def cross_device_rename(*args):
        raise OSError(errno.EXDEV, "different mounts")

    monkeypatch.setattr(os, "rename", cross_device_rename)
    context = SimpleNamespace(
        command=SimpleNamespace(cwd=str(evaluator)),
        artifact_dir=output,
        add_artifact=lambda *args: None,
    )
    HipprofProfiler._save_collection_output(
        context, {"tool": "hipprof", "output": output}, "HIP_PROF:process id '123'\n"
    )
    assert not source.exists()
    assert (output / source.name).read_text() == "measured counters"

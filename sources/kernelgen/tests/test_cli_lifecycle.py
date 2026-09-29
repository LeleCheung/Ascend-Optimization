"""The kg lifecycle mode uses the existing foreground dummy orchestration."""

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

import kernelgen.cli.lifecycle as lifecycle_cli
from kernelgen.cli.main import build_parser, main
from kernelgen.cli.models import ProcessState, RunMode
from kernelgen.workflows.operator_development import DummyWorkflowCall, OptimizeConfig, campaign_status


@pytest.fixture(autouse=True)
def isolated_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / ".kernelgen"))

    def forbidden(*args, **kwargs):
        pytest.fail("dummy lifecycle attempted to submit a supervised optimization run")

    monkeypatch.setattr("kernelgen.cli.main.submit_run", forbidden)
    monkeypatch.setattr("kernelgen.cli.main.submit_request", forbidden)


def test_kg_lifecycle_runs_in_current_process_with_default_workspace(monkeypatch, tmp_path, capsys):
    original = lifecycle_cli.run_campaign
    seen = []

    def stage(ctx):
        seen.append((os.getpid(), ctx))
        return DummyWorkflowCall()(ctx)

    def campaign(*args, **kwargs):
        kwargs["workflow_call"] = stage
        return original(*args, **kwargs)

    monkeypatch.setattr(lifecycle_cli, "run_campaign", campaign)
    assert main(["run", "--mode", "lifecycle", "--dummy", "--definition", "add",
                 "--stages", "optimize", "--optimize-mode", "kernelgen"]) == 0
    result = json.loads(capsys.readouterr().out)
    workspace = Path(result["workspace"])
    assert workspace.parent == tmp_path / ".kernelgen/runs"
    assert result["simulated"] is True
    assert result["operators"][0]["state"] == "SUCCEEDED"
    assert len(seen) == 1 and seen[0][0] == os.getpid()
    assert seen[0][1].optimize.mode == RunMode.KERNELGEN
    assert seen[0][1].workspace == workspace / "operators/add/stages/optimize/attempts/01"
    assert not list(tmp_path.rglob("run-request.json"))
    assert not list(tmp_path.rglob("run-process.json"))
    assert not list(tmp_path.rglob(".ledger.json"))
    assert {path.name for path in (tmp_path / ".kernelgen").iterdir()} == {"runs"}


@pytest.mark.parametrize("injection,code,state", [
    ("--fail-stage", 1, "FAILED"), ("--wait-stage", 1, "PENDING"),
    ("--cancel-stage", 130, "CANCELLED"),
])
def test_kg_lifecycle_exit_codes_and_resume(tmp_path, capsys, injection, code, state):
    workspace = tmp_path / "campaign"
    args = ["run", "--mode", "lifecycle", "--dummy", "--operators", "add", "mul",
            "--workspace", str(workspace), "--stages", "optimize", "code_review"]
    assert main(args + [injection, "code_review"]) == code
    result = json.loads(capsys.readouterr().out)
    assert result["operators"][0]["state"] == state
    old = {path: path.read_bytes() for path in workspace.rglob("result.json")}
    assert main(args + ["--resume", "--foreground"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert all(item["state"] == "SUCCEEDED" for item in result["operators"])
    assert all(item["progress"]["progress"]["total_tasks"] == 2 for item in result["operators"])
    assert all(path.read_bytes() == content for path, content in old.items())


@pytest.mark.parametrize("args", [
    ["--mode", "lifecycle", "--definition", "add"],  # explicit dummy required
    ["--mode", "lifecycle", "--dummy", "--batch-file", "missing.yaml"],
    ["--mode", "lifecycle", "--dummy", "--definition", "add", "--eval-server", "http://localhost:8000"],
    ["--mode", "lifecycle", "--dummy", "--definition", "add", "--max-round", "2"],
    ["--mode", "lifecycle", "--dummy", "--definition", "add", "--no-profile"],
    ["--mode", "simple_opt", "--dummy", "--definition", "add"],
    ["--mode", "kernelgen", "--definition", "add", "--stages", "optimize"],
    ["--mode", "simple_opt", "--definition", "add", "--optimize-mode", "kernelgen"],
    ["--mode", "simple_opt", "--definition", "add", "--resume"],
    ["--mode", "kernelgen", "--operators", "add", "mul"],
    ["--batch-file", "missing.yaml", "--wait-stage", "pr_followup"],
    ["--mode", "lifecycle", "--dummy", "--definition", "add", "--stages", "pytest_review",
     "--optimize-mode", "simple_opt"],
])
def test_invalid_mode_specific_flags_have_no_state_side_effects(tmp_path, capsys, args):
    assert main(["run", *args, "--workspace", str(tmp_path / "campaign")]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("kg:")
    assert not list(tmp_path.iterdir())


def test_lifecycle_does_not_expand_the_optimization_mode_contract():
    assert [item.value for item in RunMode] == ["simple_opt", "kernelgen"]
    with pytest.raises(ValueError):
        OptimizeConfig(mode="lifecycle")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--mode", "lifecycle", "--dummy", "--operators", "add",
                                   "--definition", "mul"])


@pytest.mark.parametrize("mode", ["simple_opt", "kernelgen"])
def test_existing_optimization_modes_still_use_the_supervised_submission(mode, tmp_path, monkeypatch, capsys):
    seen = []

    def submit(values, *, definition, workspace, foreground):
        seen.append((values, definition, foreground))
        return SimpleNamespace(workspace=workspace), SimpleNamespace(pid=123, state=ProcessState.SUBMITTED)

    monkeypatch.setattr("kernelgen.cli.main.submit_run", submit)
    assert main(["run", "--mode", mode, "--definition", "add", "--workspace", str(tmp_path / "opt")]) == 0
    assert seen[0][0]["mode"] == mode
    assert seen[0][1:] == ("add", False)
    assert "stages" not in seen[0][0] and "dummy" not in seen[0][0]
    assert "status: SUBMITTED" in capsys.readouterr().out


def test_kg_entrypoint_resume_across_processes(tmp_path):
    prefix = [sys.executable, "-c", "from kernelgen.cli import main; raise SystemExit(main())"]
    workspace = tmp_path / "campaign"
    args = ["run", "--mode", "lifecycle", "--dummy", "--operators", "add", "mul",
            "--workspace", str(workspace), "--stages", "optimize", "code_review",
            "--optimize-mode", "kernelgen"]
    failed = subprocess.run(prefix + args + ["--fail-stage", "code_review"], capture_output=True, text=True, timeout=30)
    assert failed.returncode == 1, failed.stderr
    resumed = subprocess.run(prefix + args + ["--resume"], capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout) == campaign_status(workspace)


def test_kg_foreground_sigint_finishes_dummy_output_before_cancellation(tmp_path):
    script = '''
import sys
from kernelgen.cli import main
import kernelgen.cli.lifecycle as adapter
from kernelgen.workflows.operator_development import WorkflowResult
original = adapter.run_campaign
def stage(ctx):
    print("ready", flush=True)
    sys.stdin.readline()
    return WorkflowResult(simulated=True, output={"output":"fully completed"})
def campaign(*args, **kwargs):
    kwargs["workflow_call"] = stage
    return original(*args, **kwargs)
adapter.run_campaign = campaign
raise SystemExit(main())
'''
    workspace = tmp_path / "campaign"
    proc = subprocess.Popen([sys.executable, "-c", script, "run", "--mode", "lifecycle", "--dummy",
                             "--definition", "add", "--workspace", str(workspace), "--stages", "pytest_generate"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([proc.stdout], [], [], 15)[0], "dummy stage did not become ready"
        assert proc.stdout.readline().strip() == "ready"
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate("finish output\n", timeout=15)
        assert proc.returncode == 130, stderr
        assert json.loads(stdout)["operators"][0]["state"] == "CANCELLED"
        report = workspace / "operators/add/stages/pytest_generate/attempts/01/result.json"
        assert json.loads(report.read_text())["result"]["data"]["output"] == "fully completed"
    finally:
        if proc.poll() is None:
            proc.terminate()  # Only this test-owned dummy process, never an Agent.
            proc.communicate(timeout=15)

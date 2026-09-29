"""CatalogOptimize integration and rejection of the removed combined entry."""

import pytest

from kernelgen.cli.api import build_request
from kernelgen.cli.main import main
from kernelgen.cli.runner import _invoke_workflow, execute_request
from kernelgen.framework.run_options import resolve_run_options
from kernelgen.framework.run_control import RunState, WorkspaceRunControl


@pytest.fixture(autouse=True)
def isolated_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))


def request(tmp_path):
    return build_request(resolve_run_options({
        "mode": "simple_opt", "catalog_path": tmp_path, "runtime": "codex", "model": "test-model",
    }), definition="relu", workspace=tmp_path / "run")


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("exit_code", [0, 1, 130])
def test_catalog_launcher_preserves_runtime_resume_and_exit_code(tmp_path, monkeypatch, resume, exit_code):
    from kernelgen.cli import optimization as run_example

    seen = []
    monkeypatch.setattr(run_example, "run_operator_optimization",
                        lambda inp, **kwargs: seen.append((inp, kwargs)) or exit_code)
    req = request(tmp_path)
    assert _invoke_workflow(req, resume=resume) == exit_code
    assert seen == [(req.workflow_input, dict(workspace=req.workspace, runtime="codex",
                                            model="test-model", base_url=req.base_url, resume=resume))]


@pytest.mark.parametrize("arguments", [
    ["--mode", "extract_and_optimize", "--definition", "relu"],
    ["--mode", "simple_opt", "--definition", "relu", "--input", "old-input.json"],
])
def test_removed_cli_entry_is_rejected_before_submission(tmp_path, arguments):
    with pytest.raises(SystemExit) as exc:
        main(["run", *arguments])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


def test_removed_mode_is_rejected_by_shared_options_and_batch(tmp_path):
    from kernelgen.cli.batch import load_batch_file
    with pytest.raises(ValueError):
        resolve_run_options({"mode": "extract_and_optimize"})
    source = tmp_path / "batch.yaml"
    source.write_text("version: 1\nworkspace: batch\ndefaults:\n  mode: extract_and_optimize\noperators:\n  - definition: relu\n")
    with pytest.raises(ValueError):
        load_batch_file(source)
    assert not (tmp_path / "batch").exists()


def test_catalog_supervisor_keeps_waiting_without_outer_coder_lease(tmp_path, monkeypatch):
    from kernelgen.cli import runner

    monkeypatch.setattr(runner.WorkerLeasePool, "acquire", lambda *a, **k: pytest.fail("outer lease acquired"))
    monkeypatch.setattr(runner.WorkerLeasePool, "release", lambda *a, **k: pytest.fail("outer lease released"))
    req = request(tmp_path)
    control = WorkspaceRunControl(req.workspace)

    def invoke(*args, **kwargs):
        control.update_progress(state=RunState.PENDING, stage="WAITING_REVIEW", message="Needs review")
        return 1

    monkeypatch.setattr(runner, "_invoke_workflow", invoke)
    assert execute_request(req) == 1
    assert control.progress().state == RunState.PENDING
    assert control.progress().stage == "WAITING_REVIEW"


@pytest.mark.parametrize("module", [
    "kernelgen.examples.catalog_optimize",
    "kernelgen.workflows.catalog_optimize",
    "kernelgen.workflows.kernel_gen",
    "kernelgen.workflows.kernel_optimization",
])
def test_removed_public_entries_are_absent(module):
    import importlib.util

    assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize("state,expected", [
    ("SUCCEEDED", 0), ("FAILED", 1), ("PENDING", 1), ("CANCELLED", 130),
])
def test_runtime_entry_preserves_result_and_resume(tmp_path, monkeypatch, capsys, state, expected):
    from types import SimpleNamespace
    from kernelgen.cli import optimization

    seen = []
    inp = {"optimization": {"max_round": 10}}
    monkeypatch.setattr(optimization, "resolve_cli_runtime_options", lambda *a, **k: {})

    class Workflow:
        def __init__(self, *, cwd, runtime_factory):
            assert cwd == tmp_path / "run"

        def run(self, value):
            seen.append(value)
            return SimpleNamespace(state=state, model_dump_json=lambda **kw: '{"state":"' + state + '"}')

    monkeypatch.setattr(optimization, "OperatorOptimizeWorkflow", Workflow)
    assert optimization.run_operator_optimization(inp, workspace=tmp_path / "run", resume=True) == expected
    assert seen == [{**inp, "resume": True}]
    assert "resume" not in inp
    assert capsys.readouterr().out.strip() == '{"state":"' + state + '"}'


@pytest.mark.parametrize("mode,module", [
    ("simple_opt", "kernelgen.cli.legacy_simple_opt"),
    ("kernelgen", "kernelgen.examples.kernel_gen.run_example"),
])
def test_v1_requests_still_dispatch_to_legacy_launcher(tmp_path, monkeypatch, mode, module):
    import importlib
    from kernelgen.cli.models import RunMode

    req = request(tmp_path).model_copy(update={
        "schema_version": "1.0", "workflow_input": None,
        "mode": RunMode(mode), "workflow_args": ["--definition", "relu"],
    })
    seen = []
    monkeypatch.setattr(importlib.import_module(module), "legacy_main", lambda argv: seen.append(argv))
    assert _invoke_workflow(req, resume=True) == 0
    expected = req.workflow_args + (["--start-mode", "resume", "--start-epoch", "1"] if mode == "kernelgen" else [])
    assert seen == [expected]

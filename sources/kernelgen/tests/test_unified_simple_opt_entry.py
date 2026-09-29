"""New single/batch examples submit ordinary kg requests, never legacy runs."""

import importlib.util
from types import SimpleNamespace

import pytest

from kernelgen.cli import api
from kernelgen.cli.batch import load_batch_request
from kernelgen.cli.models import ProcessState
from kernelgen.examples.simple_opt import run_example as single
from kernelgen.examples.batch_simple_opt_definition import run_example as batch

cli = importlib.import_module("kernelgen.cli.main")


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))


@pytest.mark.parametrize("foreground", [False, True])
def test_single_example_uses_cli_submission(tmp_path, monkeypatch, foreground):
    seen = []

    def submit(values, *, definition, workspace, foreground):
        request = api.build_request(values, definition=definition, workspace=workspace)
        seen.append((request, foreground))
        return request, SimpleNamespace(pid=123, state=ProcessState.EXITED, exit_code=0)

    monkeypatch.setattr(cli, "submit_run", submit)
    args = ["--definition", "negative", "--workspace", str(tmp_path / "run"), "--max-round", "2"]
    assert single.main(args + (["--foreground"] if foreground else [])) == 0
    request, observed = seen[0]
    assert observed is foreground
    assert request.schema_version == "2.0" and not request.workflow_args
    assert request.workflow_input["operator"] == "negative"
    assert request.workflow_input["optimization"]["mode"] == "simple_opt"
    assert request.workflow_input["optimization"]["max_round"] == 2


def test_batch_example_uses_same_child_requests_and_index(tmp_path, monkeypatch):
    source = tmp_path / "batch.yaml"
    source.write_text("version: 1\ndefaults:\n  max_round: 2\noperators:\n  - definition: negative\n  - definition: abs\n")
    submitted = []

    def submit(request, *, foreground):
        assert foreground is False
        submitted.append(request)
        return SimpleNamespace(pid=100 + len(submitted))

    monkeypatch.setattr(cli, "submit_request", submit)
    workspace = tmp_path / "batch"
    assert batch.main(["--batch-file", str(source), "--workspace", str(workspace)]) == 0
    index = load_batch_request(workspace)
    assert len(index.children) == len(submitted) == 2
    for request, child in zip(submitted, index.children):
        assert request.schema_version == "2.0" and not request.workflow_args
        assert request.workflow_input["optimization"]["mode"] == "simple_opt"
        assert request.workflow_input["optimization"]["max_round"] == 2
        assert request.workspace == child.workspace == workspace / "definitions" / request.definition
        assert request.batch_workspace == workspace
        assert request.worker_weight == 1
    assert not (workspace / "batch_simple_opt_definition_output.json").exists()


@pytest.mark.parametrize("name", ["simple_opt", "batch_simple_opt_definition"])
def test_old_workflow_paths_are_removed(name):
    assert importlib.util.find_spec("kernelgen.workflows." + name) is None


@pytest.mark.parametrize("relative", [
    ".ledger.json", "optimize_definition_output.json", "kernelgen_output.json",
    "batch_simple_opt_definition_output.json", "1R/agent0/.ledger.json",
])
def test_new_entry_cannot_reinterpret_historical_workspace(tmp_path, monkeypatch, relative):
    from kernelgen.cli import runner
    from kernelgen.cli.state import request_path
    from kernelgen.framework.run_options import resolve_run_options

    workspace = tmp_path / "old"
    evidence = workspace / relative
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_bytes(b"original evidence")
    request = api.build_request(resolve_run_options({"mode": "simple_opt"}),
                                definition="negative", workspace=workspace)
    monkeypatch.setattr(runner, "execute_request", lambda *a, **kw: pytest.fail("must reject before execution"))
    with pytest.raises(RuntimeError, match="legacy Python workspace"):
        runner.submit_request(request, foreground=True)
    assert evidence.read_bytes() == b"original evidence"
    assert not request_path(workspace).exists()


def test_e2e_script_preserves_existing_evidence(tmp_path, monkeypatch):
    from kernelgen.tests import run_simple_opt_e2e as smoke

    evidence = tmp_path / ".ledger.json"
    evidence.write_text("original evidence")
    monkeypatch.setenv("WT", str(tmp_path))
    monkeypatch.delenv("TARGET_HARDWARE", raising=False)
    seen = []
    monkeypatch.setattr(smoke, "main", lambda args: seen.append(args) or 2)
    assert smoke.run() == 2
    assert evidence.read_text() == "original evidence"
    assert seen[0][:4] == ["run", "--mode", "simple_opt", "--foreground"]
    assert "--target-hardware" not in seen[0]

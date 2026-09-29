"""New CLI submissions consume CatalogOptimize without a second argv/lease."""

import pytest

from kernelgen.cli.api import build_request
from kernelgen.cli.runner import _invoke_workflow, execute_request
from kernelgen.framework.run_options import resolve_run_options
from kernelgen.framework.worker_pool import set_max_workers


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    set_max_workers("http://localhost:8000", 4)


@pytest.mark.parametrize("mode", ["simple_opt", "kernelgen"])
@pytest.mark.parametrize("source", ["catalog_name", "catalog_path"])
def test_normalize_and_invoke_once(tmp_path, monkeypatch, mode, source):
    values = resolve_run_options({"mode": mode, source: "kernelgenbench" if source == "catalog_name" else tmp_path,
                                  "runtime": "codex", "model": "test-model", "max_round": 2})
    req = build_request(values, definition="square", workspace=tmp_path / "run")
    assert req.schema_version == "2.0" and req.workflow_args == []
    assert req.workflow_input["optimization"]["mode"] == mode
    assert req.worker_weight == 1
    assert req.workflow_input[source] is not None
    from kernelgen.cli import optimization as run_example
    seen = []
    monkeypatch.setattr(run_example, "run_operator_optimization", lambda inp, **kw: seen.append((inp, kw)) or 0)
    assert _invoke_workflow(req, resume=True) == 0
    assert seen[0][0] == req.workflow_input
    assert seen[0][1]["resume"] and seen[0][1]["runtime"] == "codex"


def test_sources_mutually_exclusive_and_default_only_without_path(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        resolve_run_options({"mode": "simple_opt", "catalog_name": "x", "catalog_path": tmp_path})
    assert resolve_run_options({"mode": "simple_opt", "catalog_path": tmp_path})["catalog_name"] is None
    assert resolve_run_options({"mode": "simple_opt"})["catalog_name"]


def test_kernelgen_parameters_are_not_silently_dropped(tmp_path):
    req = build_request(resolve_run_options({"mode": "kernelgen", "timeout": 85,
        "eval_atol": 0.01, "eval_rtol": 0.02, "eval_tolerance_mode": "fixed",
        "knowledge_catalog_path": tmp_path / "knowledge", "knowledge_mode": "read_only_v1",
        "start_epoch": 2, "n_epoch": 3, "finalize_epoch": 2}),
        definition="square", workspace=tmp_path / "run")
    options = req.workflow_input["optimization"]
    assert options["timeout"] == 85 and options["start_epoch"] == options["finalize_epoch"] == 2
    assert options["knowledge_config"]["mode"] == "read_only_v1"
    assert options["evaluation_contract"]["atol"] == 0.01
    assert options["evaluation_contract"]["rtol"] == 0.02


def test_new_worker_does_not_hold_outer_lease(tmp_path, monkeypatch):
    from kernelgen.cli import runner
    req = build_request(resolve_run_options({"mode": "simple_opt"}), definition="square", workspace=tmp_path / "run")
    monkeypatch.setattr(runner.WorkerLeasePool, "acquire", lambda *a, **k: pytest.fail("duplicate lease"))
    monkeypatch.setattr(runner, "_invoke_workflow", lambda *a, **k: 0)
    assert execute_request(req) == 0


def test_both_code_inputs_are_preserved(tmp_path):
    reference = tmp_path / "source.cu"
    seed = tmp_path / "seed.py"
    reference.write_text("not executed CUDA source")
    seed.write_text("unvalidated Triton context")
    req = build_request(resolve_run_options({"mode": "simple_opt", "reference_code_path": reference,
                         "seed_code_path": seed}), definition="square", workspace=tmp_path / "run")
    assert req.workflow_input["optimization"]["reference_code_path"] == str(reference)
    assert req.workflow_input["optimization"]["seed_code_path"] == str(seed)


@pytest.mark.parametrize("mode", ["simple_opt", "kernelgen"])
def test_history_uses_only_the_nested_optimizer_ledgers(tmp_path, mode):
    from kernelgen.cli.api import run_status
    from kernelgen.cli.state import save_request
    from kernelgen.cli.history import load_run_history
    from kernelgen.data.ledger import Ledger
    from kernelgen.framework.run_control import WorkspaceRunControl, RunState
    from kernelgen.tests.test_cli_history import _record_round

    root = tmp_path / "run"
    req = build_request(resolve_run_options({"mode": mode}), definition="demo", workspace=root)
    save_request(req)
    optimizer = root / "stages/optimize/work"
    paths = [optimizer] if mode == "simple_opt" else [optimizer / "1R/agent0", optimizer / "1R/agent1"]
    control = WorkspaceRunControl(root)
    control.update_progress(state=RunState.RUNNING, progress_kind="tasks")
    for path in paths:
        scope = "stages/optimize" if mode == "simple_opt" else path.relative_to(root).as_posix()
        control.link_workspace(path, scope=scope).update_progress(state=RunState.RUNNING, progress_kind="rounds")
        _record_round(Ledger(path), 1, 1.2)
    _record_round(Ledger(root), 1, 9.9)  # Unbound stray ledger is not a competing source.
    history = load_run_history(root)
    assert history["round_count"] == len(paths) and history["best_geo_mean"] == 1.2
    detail = run_status(root)
    for scope in detail["progress"]["scopes"].values():
        assert scope["progress"]["completed_rounds"] == 1
        assert scope["progress"]["best_geo_mean"] == 1.2
    if mode == "kernelgen":
        control.link_workspace(root / "stages/optimize", scope="stages/optimize").update_progress(state=RunState.RUNNING, progress_kind="epochs")
        parent = run_status(root)["progress"]["scopes"]["stages/optimize"]
        assert parent["progress"]["best_geo_mean"] == 1.2


def test_request_metadata_cannot_drift(tmp_path):
    from kernelgen.cli.models import RunRequest
    req = build_request(resolve_run_options({"mode": "simple_opt"}), definition="square", workspace=tmp_path / "run")
    raw = req.model_dump()
    raw["eval_server"] = "http://other:8000"
    with pytest.raises(ValueError, match="metadata must match"):
        RunRequest.model_validate(raw)


@pytest.mark.parametrize("mode", ["simple_opt", "kernelgen"])
def test_python_launcher_and_cli_build_identical_inputs(tmp_path, monkeypatch, mode):
    from kernelgen.cli import optimization as run_example
    from kernelgen.framework.run_options import optimization_argv
    values = resolve_run_options({"mode": mode, "catalog_path": tmp_path, "max_round": 2, "runtime": "codex"})
    req = build_request(values, definition="square", workspace=tmp_path / "run")
    seen = []
    monkeypatch.setattr(run_example, "run_operator_optimization", lambda inp, **kw: seen.append(inp) or 0)
    assert run_example.run_optimization_example(mode, ["--definition", "square", *optimization_argv(values)]) == 0
    assert seen == [req.workflow_input]


def test_python_launcher_refuses_implicit_legacy_workspace_migration(tmp_path):
    from kernelgen.cli.optimization import run_optimization_example
    (tmp_path / ".ledger.json").write_text('{}')
    with pytest.raises(SystemExit):
        run_optimization_example("simple_opt", ["--definition", "square", "--workspace", str(tmp_path)])


@pytest.mark.parametrize("runtime", ["claude", "codex"])
def test_shared_launcher_limits_mcp_to_optimizer(tmp_path, monkeypatch, runtime):
    from types import SimpleNamespace
    from kernelgen.cli import optimization as launcher

    copied, mcps, runtimes = [], [], []
    monkeypatch.setattr(launcher, "resolve_cli_runtime_options", lambda *a, **k: {"model": "test-model"})
    monkeypatch.setattr(launcher, "copy_claude_directory", lambda source, target, **kw: copied.append((target, kw)))
    monkeypatch.setattr(launcher, "copy_mcp_configuration", lambda source, target, **kw: mcps.append(target))
    monkeypatch.setattr(launcher, "materialize_claude_runtime_config", lambda *a: None)
    monkeypatch.setattr(launcher, "create_cli_runtime", lambda *a, **kw: runtimes.append(kw))

    class Workflow:
        def __init__(self, *, cwd, runtime_factory):
            for relative in ("stages/review_tests/01", "stages/optimize/work", "stages/optimize/work/1R/agent0"):
                runtime_factory(cwd / relative)

        def run(self, inp):
            return SimpleNamespace(state="SUCCEEDED", model_dump_json=lambda **kw: '{}')

    monkeypatch.setattr(launcher, "OperatorOptimizeWorkflow", Workflow)
    assert launcher.run_operator_optimization({"optimization": {}}, workspace=tmp_path, runtime=runtime) == 0
    assert [kw["include_skills"] for _, kw in copied] == [False, True, True]
    assert mcps == [tmp_path / "stages/optimize/work/.mcp.json", tmp_path / "stages/optimize/work/1R/agent0/.mcp.json"]
    assert runtimes[0]["allowed_tools" if runtime == "claude" else "sandbox_mode"] == ("Read" if runtime == "claude" else "read-only")
    assert all("allowed_tools" not in kw and "sandbox_mode" not in kw for kw in runtimes[1:])

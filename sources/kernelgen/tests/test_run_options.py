"""One parameter contract, without a second unmanaged launcher argument path."""

import pytest

from kernelgen.cli.main import build_parser, _explicit_run_options, _new_request_from_options
from kernelgen.cli.state import set_max_workers
from kernelgen.framework.run_options import launcher_parser, optimization_argv, resolve_run_options, validate_options


@pytest.mark.parametrize("mode", ["simple_opt", "kernelgen"])
def test_launcher_and_kg_defaults_and_values_match(mode):
    values = resolve_run_options({"mode": mode})
    direct = vars(launcher_parser(mode).parse_args(["--definition", "demo"]))
    forwarded = vars(launcher_parser(mode).parse_args(["--definition", "demo", *optimization_argv(values)]))
    for key, value in values.items():
        if key != "mode":
            assert direct[key] == forwarded[key] == value


@pytest.mark.parametrize("tail", [["--", "--n-par", "7"], ["--n-par", "7"], ["--clean"], ["--auth-token", "test"]])
def test_managed_run_rejects_unmanaged_or_abbreviated_arguments(tail):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--mode", "kernelgen", "--definition", "demo", *tail])


@pytest.mark.parametrize("values", [
    {"mode": "kernelgen", "warmup_ms": 1},
    {"mode": "simple_opt", "n_epoch": 2},
    {"mode": "kernelgen", "n_parallel": 0},
    {"mode": "simple_opt", "max_round": -1},
    {"mode": "simple_opt", "num_trials": 1.5},
    {"mode": "simple_opt", "profile": "false"},
])
def test_resolved_cli_and_yaml_reject_invalid_values(values):
    with pytest.raises(ValueError):
        resolve_run_options(values)


def test_request_metadata_and_launcher_have_same_endpoint_workspace_weight(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    endpoint = "http://127.0.0.1:18101"
    set_max_workers(endpoint, 7)
    args = build_parser().parse_args([
        "run", "--mode", "kernelgen", "--definition", "demo", "--workspace", str(tmp_path / "run"),
        "--n-parallel", "7", "--eval-server", endpoint,
    ])
    request = _new_request_from_options(resolve_run_options(_explicit_run_options(args)), definition=args.definition, workspace=args.workspace)
    actual = request.workflow_input["optimization"]
    assert request.workflow_args == []
    assert request.eval_server == request.worker_pool == actual["eval_server_url"]
    assert request.workspace == args.workspace
    assert request.worker_weight == actual["n_parallel"] == 7


def test_yaml_paths_use_manifest_parent(tmp_path):
    assert validate_options({"seed_code_path": "seed.py"}, base=tmp_path)["seed_code_path"] == tmp_path / "seed.py"


def test_default_kernelgen_options_satisfy_workflow_evaluation_contract(monkeypatch):
    from kernelgen.framework.models import EvaluationContractModel
    monkeypatch.delenv("KERNELGEN_EVAL_TOLERANCE_MODE", raising=False)
    values = resolve_run_options({"mode": "kernelgen"})
    args = launcher_parser("kernelgen").parse_args(["--definition", "demo", *optimization_argv(values)])
    contract = EvaluationContractModel(
        tolerance_mode=args.eval_tolerance_mode, atol=args.eval_atol, rtol=args.eval_rtol,
        required_matched_ratio=args.eval_required_matched_ratio,
        consider_reduced_precision=not args.no_reduced_precision,
    )
    assert contract.tolerance_mode == ""

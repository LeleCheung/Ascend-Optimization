import json
from pathlib import Path
from types import SimpleNamespace

from kernelgen.agents.pytest_converter import (
    PytestConverterAgent,
    PytestConverterInput,
    PytestConverterOutput,
)
from kernelgen.framework.runtime.base import FakeRuntime
from kernelgen.tools.build_pytest_conversion_inventory import resolve_catalog_id
from kernelgen.examples.pytest_converter.run_one import (
    _audit_reported_commands,
    _extract_bash_commands,
)
from kernelgen.examples.pytest_converter.run_batch import (
    DEFAULT_BATCH_SIZE,
    MAX_BATCH_SIZE,
    _agent_result_is_reviewable,
    _make_patch,
    _pending_batch_operators,
    _select_items,
)
from kernelgen.examples.pytest_converter.run_nvidia_backlog import (
    Candidate,
    _approve,
)
from kernelgen.examples.pytest_converter.validate_one_metax import (
    LocalNvidiaRunner,
    MetaXRunner,
    _benchmark_rows,
    _case_rows,
    _pytest_counts,
    _tar_payload,
)


def _input() -> dict:
    return {
        "source_operator": "gt_.scalar",
        "operator": "gt_scalar_",
        "pytest_mark": "gt_scalar_",
        "accuracy_files": ["tests/test_gt_scalar_.py"],
        "benchmark_files": ["benchmark/test_gt_scalar_.py"],
        "shared_files": {},
    }


def _output() -> str:
    return json.dumps(
        {
            "source_operator": "gt_.scalar",
            "operator": "gt_scalar_",
            "status": "converted",
            "files_modified": [
                "tests/test_gt_scalar_.py",
                "benchmark/test_gt_scalar_.py",
            ],
            "commands": [
                {
                    "command": "python3 -m pytest -q tests/test_gt_scalar_.py",
                    "exit_code": 0,
                    "outcome": "passed",
                    "summary": "passed",
                }
            ],
            "remaining_issues": [],
            "summary": "converted",
        }
    )


def test_pytest_converter_models():
    inp = PytestConverterInput.model_validate(_input())
    assert inp.operator == "gt_scalar_"
    out = PytestConverterOutput.model_validate_json(_output())
    assert out.commands[0].outcome == "passed"


def test_pytest_converter_fallback_prompt_contains_role_and_scope():
    runtime = FakeRuntime([_output()])
    result = PytestConverterAgent().run(_input(), runtime)
    assert result.status == "converted"
    prompt = runtime.calls[0]["prompt"]
    assert "Edit only the paths listed" in prompt
    assert "tests/test_gt_scalar_.py" in prompt


def test_pytest_converter_native_agent_name():
    class NativeFakeRuntime(FakeRuntime):
        supports_native_agents = True

    runtime = NativeFakeRuntime([_output()])
    PytestConverterAgent().run(_input(), runtime)
    assert runtime.calls[0]["agent"] == "kernel-pytest-converter"
    assert "Edit only the paths listed" not in runtime.calls[0]["prompt"]


def test_catalog_resolution_prefers_direct_catalog_ids_and_aten_overloads():
    entries = [
        {"id": "gt_scalar_", "for": ["gt_.Scalar"]},
        {"id": "gt_tensor_", "for": ["gt_.Tensor"]},
        {"id": "special_round_out", "for": ["special.round.out"]},
        {"id": "add_relu", "for": ["_add_relu.Tensor"]},
        {"id": "add_relu_", "for": ["_add_relu_.Tensor"]},
    ]
    assert resolve_catalog_id("gt_.scalar", entries) == "gt_scalar_"
    assert resolve_catalog_id("special_round.out", entries) == "special_round_out"
    assert resolve_catalog_id("_add_relu", entries) == "add_relu"


def test_native_markdown_has_serial_conversion_guards():
    role = (
        Path(__file__).resolve().parents[1]
        / ".kernelgen"
        / "agents"
        / "kernel-pytest-converter.md"
    ).read_text(encoding="utf-8")
    assert "exactly one operator" in role
    assert "Do not run git commit" in role
    assert "resolve_gems_op" in role
    assert "case_fn" in role and "build_inputs_fn" in role
    assert "current direct-call" in role and "return `blocked`" in role


def test_extract_and_audit_runtime_bash_commands():
    log = """[claude] 🔧 Bash: $ Check cwd
[claude] tool input id=one:
{
  "command": "pwd",
  "description": "Check cwd"
}
[claude] tool result metadata: id=one is_error=False
[claude] 🔧 Bash: $ Run test
[claude] tool input id=two:
{
  "command": "python3 -m pytest -q tests/test_op.py",
  "description": "Run test"
}
[claude] tool result metadata: id=two is_error=False
"""
    actual = _extract_bash_commands(log)
    assert actual == ["pwd", "python3 -m pytest -q tests/test_op.py"]
    assert _audit_reported_commands(actual, actual) == []
    issues = _audit_reported_commands(
        actual, ["python3 -m pytest -q tests/test_op.py"]
    )
    assert "do not exactly match" in issues[0]


def test_command_audit_rejects_shell_composition():
    command = "python3 -m pytest -q tests/test_op.py 2>&1 | head -20"
    issues = _audit_reported_commands([command], [command])
    assert "forbidden shell composition" in issues[0]


def test_command_audit_ignores_repeat_count_for_read_only_validation():
    diff_check = "git diff --check -- tests/test_op.py"
    actual = [
        "pwd",
        "pwd",
        diff_check,
        diff_check,
        "git diff -- tests/test_op.py",
    ]
    reported = [
        "pwd",
        diff_check,
        diff_check,
        diff_check,
        "git diff -- tests/test_op.py",
    ]
    assert _audit_reported_commands(actual, reported) == []


def test_batch_selection_skips_active_and_shared_targets(tmp_path):
    assert DEFAULT_BATCH_SIZE == 30
    assert MAX_BATCH_SIZE == 30
    inventory = {
        "operators": [
            {
                "source_operator": "one",
                "operator": "one",
                "accuracy_files": ["tests/shared.py"],
                "benchmark_files": ["benchmark/one.py"],
            },
            {
                "source_operator": "two",
                "operator": "two",
                "accuracy_files": ["tests/shared.py"],
                "benchmark_files": ["benchmark/two.py"],
            },
            {
                "source_operator": "three",
                "operator": "three",
                "accuracy_files": ["tests/three.py"],
                "benchmark_files": ["benchmark/three.py"],
            },
        ]
    }
    state = {
        "active_review": {"source_operator": "three"},
        "history": [],
    }
    selected = _select_items(inventory, state, 10, None)
    assert [item["source_operator"] for item in selected] == ["one"]

    selected = _select_items(inventory, state, 10, None, {"one"})
    assert [item["source_operator"] for item in selected] == ["two"]

    state["protocol_gaps"] = {"one": {"status": "pending_user_confirmation"}}
    selected = _select_items(inventory, state, 10, None)
    assert [item["source_operator"] for item in selected] == ["two"]

    state["validation_blockers"] = {
        "two": {"status": "pending_user_confirmation"}
    }
    selected = _select_items(inventory, state, 10, None)
    assert selected == []

    state["history"] = [
        {"source_operator": "one", "decision": "approved"},
        {"source_operator": "one", "decision": "needs_changes"},
    ]
    state["protocol_gaps"] = {}
    state["validation_blockers"] = {}
    selected = _select_items(inventory, state, 10, None)
    assert [item["source_operator"] for item in selected] == ["one"]

    before = tmp_path / "before"
    after = tmp_path / "after"
    (before / "tests").mkdir(parents=True)
    (after / "tests").mkdir(parents=True)
    (before / "tests/op.py").write_text("value = 1\n")
    (after / "tests/op.py").write_text("value = 2\n")
    patch = _make_patch(before, after, ["tests/op.py"])
    assert "-value = 1" in patch
    assert "+value = 2" in patch


def test_pending_batch_operators_skips_review_backlog(tmp_path):
    batch = tmp_path / "20260827T000000+0800"
    batch.mkdir()
    (batch / "manifest.json").write_text(
        json.dumps(
            {
                "items": [
                    {"source_operator": "ready", "status": "ready_for_review"},
                    {"source_operator": "reviewed", "status": "reviewed"},
                    {"source_operator": "retry", "status": "agent_failed"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _pending_batch_operators(tmp_path) == {"ready", "reviewed"}


def test_already_compliant_agent_result_is_reviewable():
    assert _agent_result_is_reviewable("converted")
    assert _agent_result_is_reviewable("already_compliant")
    assert not _agent_result_is_reviewable("failed")


def test_backlog_approval_forwards_explicit_state_and_repo(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    state = tmp_path / "state.json"
    repo = tmp_path / "FlagGems"
    approved_files = tmp_path / "approved"
    _approve(
        Candidate(tmp_path / "manifest.json", "op", "op", ("tests/op.py",)),
        {"record": "record.json"},
        validation_label="local NVIDIA",
        state_path=state,
        flaggems_repo=repo,
        approved_files=approved_files,
    )

    assert len(calls) == 2
    assert all(str(state) in command for command in calls)
    assert all(str(repo) in command for command in calls)
    assert str(approved_files) in calls[1]


def test_device_validation_helpers():
    assert _pytest_counts("2 passed, 3 skipped in 1.2s") == {
        "passed": 2,
        "skipped": 3,
    }
    benchmark = {
        "op": {"details": [{"result": [{"case_id": "one"}]}]}
    }
    assert _benchmark_rows(benchmark) == [{"case_id": "one"}]
    cases = {"benchmarks": [{"cases": [{"case_id": "one"}]}]}
    assert _case_rows(cases) == [{"case_id": "one"}]
    payload = _tar_payload({"tests/test_op.py": b"value = 1\n"})
    assert b"tests/test_op.py" in payload


def test_local_nvidia_runner_uses_isolated_cuda_container():
    runner = LocalNvidiaRunner(
        SimpleNamespace(
            container="kernelgen-nvidia-cu128",
            device="3",
            remote_root="/tmp/kernelgen-pytest-nvidia-op",
            remote_python="python3",
            command_timeout=30,
        )
    )
    command = runner._docker(["python3", "-V"], workdir=runner.args.remote_root)
    assert command[:2] == ["docker", "exec"]
    assert "CUDA_VISIBLE_DEVICES=3" in command
    assert "PYTHONPATH=/tmp/kernelgen-pytest-nvidia-op/src:/tmp" in command
    assert "kernelgen-nvidia-cu128" in command


def test_remote_runner_uses_configured_device_environment():
    runner = MetaXRunner(
        SimpleNamespace(
            container="codex_fib_thead_20260818",
            device="3",
            device_env="CUDA_VISIBLE_DEVICES",
            remote_root="/workspace/kernelgen-pytest-remote-op",
            remote_site_packages="/usr/local/lib/python3.12/site-packages",
        )
    )
    command = runner._docker(["python3", "-V"], workdir=runner.args.remote_root)
    assert command[:4] == ["sudo", "-n", "docker", "exec"]
    assert "CUDA_VISIBLE_DEVICES=3" in command
    assert (
        "PYTHONPATH=/workspace/kernelgen-pytest-remote-op/src:"
        "/usr/local/lib/python3.12/site-packages:/tmp"
    ) in command
    assert "codex_fib_thead_20260818" in command

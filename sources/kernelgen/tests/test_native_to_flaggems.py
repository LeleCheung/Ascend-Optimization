import json
import subprocess
from pathlib import Path

import pytest

from kernelgen.agents.native_to_flaggems import NativeToFlagGemsOutput
from kernelgen.workflows import native_to_flaggems


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "FlagGems"
    source = root / "src/flag_gems/runtime/backend/_metax/ops/gelu.py"
    accuracy = root / "tests/test_gelu.py"
    benchmark = root / "benchmark/test_gelu.py"
    standard = root / "docs/content/zh-cn/testing/kernelgen-integration.md"
    for path in (source, accuracy, benchmark, standard):
        path.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("def gelu(input):\n    return input\n", encoding="utf-8")
    accuracy.write_text("def test_accuracy():\n    pass\n", encoding="utf-8")
    benchmark.write_text("def test_perf():\n    pass\n", encoding="utf-8")
    standard.write_text("# standard\n", encoding="utf-8")
    (root / "README.md").write_text("clean\n", encoding="utf-8")
    _git(root, "init", "-b", "feature")
    _git(root, "config", "user.name", "KernelGen Test")
    _git(root, "config", "user.email", "kernelgen@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    return root


def _input(tmp_path: Path, root: Path) -> dict:
    native = tmp_path / "native.py"
    native.write_text("def run(input):\n    return input + 1\n", encoding="utf-8")
    definition = tmp_path / "definition.json"
    definition.write_text(
        json.dumps(
            {
                "api_version": "v6.2",
                "name": "gelu",
                "parameters": [{"name": "input", "kind": "positional_or_keyword"}],
            }
        ),
        encoding="utf-8",
    )
    return {
        "operator": "gelu",
        "source_operator": "gelu",
        "vendor": "metax",
        "native_kernel_path": str(native),
        "native_definition_path": str(definition),
        "flaggems_worktree": str(root),
        "accuracy_files": ["tests/test_gelu.py"],
        "benchmark_files": ["benchmark/test_gelu.py"],
        "run_tests": False,
    }


def _workflow(tmp_path: Path, runtime=None):
    return native_to_flaggems.NativeToFlagGemsWorkflow(
        cwd=str(tmp_path / "migration-workspace"),
        runtime_factory=lambda path: runtime or object(),
    )


def test_migration_allows_preexisting_non_python_dirt_and_audits_actual_diff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root = _repo(tmp_path)
    readme = root / "README.md"
    readme.write_text("operator notes\n", encoding="utf-8")
    source_relative = "src/flag_gems/runtime/backend/_metax/ops/gelu.py"

    class Runtime:
        def __init__(self):
            self.directories = []

        def add_directory(self, path):
            self.directories.append(path)

    runtime = Runtime()

    def migrate(self, inp, runtime):
        (root / source_relative).write_text(
            "def gelu(input):\n    return input + 1\n",
            encoding="utf-8",
        )
        return NativeToFlagGemsOutput(
            operator="gelu",
            status="migrated",
            files_modified=[source_relative],
            remaining_checks=["target correctness and timing"],
            summary="prepared vendor implementation",
        )

    monkeypatch.setattr(native_to_flaggems.NativeToFlagGemsAgent, "run", migrate)

    output = _workflow(tmp_path, runtime).run(_input(tmp_path, root))

    assert output.status == "READY_FOR_TARGET_VALIDATION"
    assert output.ready_for_target_validation
    assert output.changed_python_files == [source_relative]
    assert runtime.directories == [root.resolve()]
    assert readme.read_text(encoding="utf-8") == "operator notes\n"
    audit = json.loads(Path(output.audit_path).read_text(encoding="utf-8"))
    assert audit["schema_version"] == native_to_flaggems.AUDIT_SCHEMA
    assert audit["agent_report"]["remaining_checks"] == [
        "target correctness and timing"
    ]


def test_preexisting_dirty_python_blocks_before_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root = _repo(tmp_path)
    (root / "tests/test_gelu.py").write_text(
        "def test_accuracy():\n    assert False\n",
        encoding="utf-8",
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("Agent must not run")

    monkeypatch.setattr(native_to_flaggems.NativeToFlagGemsAgent, "run", unexpected)

    output = _workflow(tmp_path).run(_input(tmp_path, root))

    assert output.status == "AUDIT_FAILED"
    assert "already has dirty Python files" in output.reason
    assert not output.ready_for_target_validation


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("out_of_scope", "out-of-scope files"),
        ("underreported", "report differs from actual Python diff"),
        ("invalid_syntax", "invalid Python after migration"),
        ("staged", "changed the FlagGems staging area"),
    ],
)
def test_deterministic_audit_rejects_unsafe_agent_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    reason: str,
):
    root = _repo(tmp_path)
    source_relative = "src/flag_gems/runtime/backend/_metax/ops/gelu.py"
    tools_relative = "tools/helper.py"
    tools = root / tools_relative
    tools.parent.mkdir()
    tools.write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", tools_relative)
    _git(root, "commit", "-m", "add tool")

    def migrate(self, inp, runtime):
        source = root / source_relative
        source.write_text(
            "def gelu(input):\n    return input + 1\n"
            if mode != "invalid_syntax"
            else "def gelu(:\n",
            encoding="utf-8",
        )
        files = [source_relative]
        if mode == "out_of_scope":
            tools.write_text("VALUE = 2\n", encoding="utf-8")
            files.append(tools_relative)
        elif mode == "underreported":
            files = []
        elif mode == "staged":
            _git(root, "add", source_relative)
        return NativeToFlagGemsOutput(
            operator="gelu",
            status="migrated",
            files_modified=files,
            summary="agent claimed migration",
        )

    monkeypatch.setattr(native_to_flaggems.NativeToFlagGemsAgent, "run", migrate)

    output = _workflow(tmp_path).run(_input(tmp_path, root))

    assert output.status == "AUDIT_FAILED"
    assert reason in output.reason
    assert source_relative in output.changed_python_files
    assert not output.ready_for_target_validation


def test_blocked_agent_must_leave_worktree_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root = _repo(tmp_path)

    def blocked(self, inp, runtime):
        return NativeToFlagGemsOutput(
            operator="gelu",
            status="blocked",
            protocol_gaps=["out alias cannot be represented"],
            summary="public ABI is ambiguous",
        )

    monkeypatch.setattr(native_to_flaggems.NativeToFlagGemsAgent, "run", blocked)

    output = _workflow(tmp_path).run(_input(tmp_path, root))

    assert output.status == "BLOCKED"
    assert not output.ready_for_target_validation
    assert output.changed_python_files == []
    assert output.agent_report.protocol_gaps == ["out alias cannot be represented"]


def test_migrated_report_requires_source_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root = _repo(tmp_path)

    def tests_only(self, inp, runtime):
        relative = "tests/test_gelu.py"
        (root / relative).write_text(
            "def test_accuracy():\n    assert True\n",
            encoding="utf-8",
        )
        return NativeToFlagGemsOutput(
            operator="gelu",
            status="migrated",
            files_modified=[relative],
            summary="tests only",
        )

    monkeypatch.setattr(native_to_flaggems.NativeToFlagGemsAgent, "run", tests_only)

    output = _workflow(tmp_path).run(_input(tmp_path, root))

    assert output.status == "AUDIT_FAILED"
    assert "has no FlagGems source change" in output.reason


def test_definition_identity_is_checked_before_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root = _repo(tmp_path)
    inp = _input(tmp_path, root)
    Path(inp["native_definition_path"]).write_text(
        json.dumps({"api_version": "v6.2", "name": "relu"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        native_to_flaggems.NativeToFlagGemsAgent,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )

    output = _workflow(tmp_path).run(inp)

    assert output.status == "AUDIT_FAILED"
    assert "does not match 'gelu'" in output.reason

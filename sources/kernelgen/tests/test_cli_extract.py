"""The extraction CLI only adapts the existing independent Workflow."""

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from kernelgen.cli import api
from kernelgen.cli.main import main
from kernelgen.framework.run_control import RunCancelled, WorkspaceRunControl
from kernelgen.workflows.catalog_extract import CatalogExtractWorkflow
from kernelgen.workflows.catalog_extract_review import CatalogReviewRequired
from kernelgen.workflows.catalog_extract_review import CatalogExtractionBlocked


def test_cli_reports_blocker_without_success_json(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'attempts/01/extraction.json'
    def blocked(*args, **kwargs):
        raise CatalogExtractionBlocked(path)
    monkeypatch.setattr(api, 'extract_catalog', blocked)
    assert main(['extract', '--operator', 'relu', '--flaggems-repo', str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert 'blocked' in captured.err and str(path) in captured.err


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))


@pytest.mark.parametrize("source", ["checkout", "pr"])
def test_command_passes_source_and_runtime_once(tmp_path, monkeypatch, capsys, source):
    seen = []
    monkeypatch.setattr(api, "extract_catalog", lambda inp, **kwargs: seen.append((inp, kwargs)) or {"catalog_path": "published"})
    flags = ["--flaggems-repo", str(tmp_path)] if source == "checkout" else ["--pr-url", "https://github.com/flagos-ai/FlagGems/pull/5623"]
    assert main(["extract", "--operator", "relu", *flags, "--runtime", "codex", "--model", "test-model",
                 "--max-review-rounds", "2", "--timeout", "123"]) == 0
    inp, options = seen[0]
    assert len(seen) == 1 and inp["operator"] == "relu"
    assert inp["max_review_rounds"] == 2 and "case_list_path" not in inp
    assert bool(inp["flaggems_repo"]) == (source == "checkout")
    assert bool(inp["pr_url"]) == (source == "pr")
    assert options["workspace"].parent == tmp_path / "state/extracts"
    assert options["runtime"] == "codex" and options["model"] == "test-model" and options["timeout"] == 123
    assert json.loads(capsys.readouterr().out) == {"catalog_path": "published"}


@pytest.mark.parametrize("flags", [[], ["--flaggems-repo", ".", "--pr-url", "url"],
    ["--flaggems-repo", ".", "--resume"], ["--flaggems-repo", ".", "--case-list-path", "cases.json"],
    ["--flaggems-repo", ".", "--eval-server", "http://localhost:8000"]])
def test_invalid_command_has_no_side_effects(tmp_path, flags):
    with pytest.raises(SystemExit) as exc:
        main(["extract", "--operator", "relu", *flags])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("flags", [["--timeout", "0"], ["--max-review-rounds", "0"], ["--max-review-rounds", "11"]])
def test_invalid_values_before_workspace(tmp_path, flags):
    assert main(["extract", "--operator", "relu", "--flaggems-repo", str(tmp_path), *flags]) == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_api_materializes_readonly_runtime_and_returns_workflow_paths(tmp_path, monkeypatch, provider):
    import kernelgen.framework.runtime as runtime
    repo = tmp_path / "gems"
    repo.mkdir()
    root = tmp_path / "extract"
    created = []
    monkeypatch.setattr(runtime, "resolve_cli_runtime_options", lambda *a, **kw: {"model": "test", "base_url": None, "auth_token": None})
    monkeypatch.setattr(runtime, "materialize_claude_runtime_config", lambda *a: None)
    monkeypatch.setattr(runtime, "create_cli_runtime", lambda name, **kw: created.append((name, kw)) or object())
    def run(workflow, inp):
        assert workflow._cwd == root and inp["case_list_path"] is None
        for relative in ("case_collection/01/agent", "extractor", "attempts/01/review"):
            path = root / relative
            workflow._runtime_factory(str(path))
            assert (path / ".claude/agents/kernel-flaggems-v62-extractor.md").is_file()
            assert (path / ".claude/agents/kernel-artifact-reviewer.md").is_file()
            assert not (path / ".mcp.json").exists()
            assert WorkspaceRunControl(path).root_workspace == root
        return SimpleNamespace(operator="relu", catalog_path=root/"catalog", operator_dir=root/"catalog/ops/relu",
                               case_list_path="cases.json", review_path=root/"attempts/01/review/review.json")
    monkeypatch.setattr(CatalogExtractWorkflow, "run", run)
    result = api.extract_catalog({"operator": "relu", "flaggems_repo": str(repo)}, workspace=root, runtime=provider)
    assert result["catalog_path"] == str(root / "catalog") and result["target_validation"] == "NOT_RUN"
    assert len(created) == 3
    for name, kw in created:
        assert name == provider and kw["verbose"] is False
        assert kw.get("allowed_tools") == "Read" if provider == "claude" else kw.get("sandbox_mode") == "read-only"
    assert not (root / ".kernelgen/run-request.json").exists()
    assert not (tmp_path / "state").exists()


def test_workspace_cannot_overwrite_or_live_inside_source(tmp_path, monkeypatch):
    repo = tmp_path / "gems"
    repo.mkdir()
    root = tmp_path / "old"
    root.mkdir()
    evidence = root / "evidence.json"
    evidence.write_text("original")
    monkeypatch.setattr(CatalogExtractWorkflow, "run", lambda *a: pytest.fail("must not start"))
    values = {"operator": "relu", "flaggems_repo": str(repo)}
    with pytest.raises(ValueError, match="already exists"):
        api.extract_catalog(values, workspace=root)
    assert evidence.read_text() == "original" and len(list(root.iterdir())) == 1
    with pytest.raises(ValueError, match="outside"):
        api.extract_catalog(values, workspace=repo / "extract")
    assert not list(repo.iterdir())


@pytest.mark.parametrize("outcome,code", [("review", 1), ("cancel", 130), ("failure", 2)])
def test_workflow_failure_exit_codes(tmp_path, monkeypatch, outcome, code):
    repo = tmp_path / "gems"
    repo.mkdir()
    root = tmp_path / "extract"
    def run(workflow, inp):
        if outcome == "review":
            raise CatalogReviewRequired("budget exhausted")
        if outcome == "cancel":
            control = WorkspaceRunControl(root)
            control.request_cancel("test")
            control.checkpoint("AFTER_COMPLETE_OUTPUT")
        raise RuntimeError("model failed")
    monkeypatch.setattr(CatalogExtractWorkflow, "run", run)
    assert main(["extract", "--operator", "relu", "--flaggems-repo", str(repo), "--workspace", str(root)]) == code
    assert root.exists()


def test_sigint_waits_for_complete_output_before_checkpoint(tmp_path, monkeypatch):
    import signal
    import threading
    from kernelgen.framework import cancellation

    repo = tmp_path / "gems"
    repo.mkdir()
    root = tmp_path / "extract"
    requested = threading.Event()
    original = cancellation.request_run_cancellation
    previous_handler = signal.getsignal(signal.SIGINT)

    def request(*args, **kwargs):
        result = original(*args, **kwargs)
        requested.set()
        return result

    def run(workflow, inp):
        signal.raise_signal(signal.SIGINT)
        assert requested.wait(3)
        # SIGINT must not unwind the model call or truncate its last output.
        (root / "complete-output.txt").write_text("complete model response")
        WorkspaceRunControl(root).checkpoint("AFTER_COMPLETE_OUTPUT")

    monkeypatch.setattr(cancellation, "request_run_cancellation", request)
    monkeypatch.setattr(CatalogExtractWorkflow, "run", run)
    assert main(["extract", "--operator", "relu", "--flaggems-repo", str(repo), "--workspace", str(root)]) == 130
    assert (root / "complete-output.txt").read_text() == "complete model response"
    assert signal.getsignal(signal.SIGINT) == previous_handler

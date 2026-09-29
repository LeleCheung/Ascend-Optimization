"""Regression tests for epoch-level KB candidate reduction.

    python tests/test_kb_writeback.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from kernelgen.framework import FakeRuntime  # noqa: E402
from kernelgen.framework.run_control import (  # noqa: E402
    CancellationState,
    RunCancelled,
)
from kernelgen.examples.kernel_gen.run_example import setup_workspace  # noqa: E402
from kernelgen.framework.parallel import Directory  # noqa: E402
from kernelgen.workflows.optimization.kernelgen import KernelGenInput  # noqa: E402
from kernelgen.workflows.optimization.kernelgen.knowledge import merge_epoch_kb


_DEFN = {
    "name": "abl_t3_decode_mla",
    "op_type": "attention",
    "axes": {
        "batch_size": {"type": "const", "value": 16},
        "page_size": {"type": "const", "value": 128},
    },
    "inputs": {"q": {"shape": ["batch_size", 128, 576], "dtype": "float32"}},
    "outputs": {"output": {"shape": ["batch_size", 128, 512], "dtype": "float32"}},
    "reference": "def run(q): return q",
}


class _Report:
    def __init__(self, status="PASSED"):
        self.status = status
        self.summary = ""


def _git_init_kb(kb_path):
    kb_path.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init"], cwd=str(kb_path), capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=str(kb_path), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "base", "--allow-empty"],
        cwd=str(kb_path),
        capture_output=True,
        check=True,
        env=env,
    )


def _entry(root):
    return (
        root
        / "kb"
        / "experience"
        / "by_definition"
        / "attention"
        / "abl_t3_decode_mla"
        / "Ascend910B"
    )


def _write_agent(workspace, name, geo, best_round, experience, detailed):
    path = Path(workspace.allocate(name))
    history = {
        "schema_version": "3.0",
        "definition_name": "abl_t3_decode_mla",
        "target_hardware": "Ascend910B",
        "best_geo_mean": geo,
        "best_round": best_round,
        "best_code": f"def run(): return {geo}",
        "rounds": [],
        "rounds_without_improvement": 0,
    }
    (path / ".ledger.json").write_text(json.dumps(history), encoding="utf-8")
    (path / ".new_experience.md").write_text(experience, encoding="utf-8")
    (path / ".new_detailed.md").write_text(detailed, encoding="utf-8")


def _input():
    return KernelGenInput.model_validate(
        {
            "definition": _DEFN,
            "target_hardware": "Ascend910B",
            "n_parallel": 2,
        }
    )


def test_scored_n_way_merge_writes_both_documents(tmp_path):
    base_entry = _entry(tmp_path)
    base_entry.mkdir(parents=True)
    (base_entry / "experience.md").write_text("## Previous verified lesson")
    _git_init_kb(tmp_path / "kb")

    workspace = Directory(base=tmp_path / "1R")
    _write_agent(
        workspace,
        "agent0",
        6.3311295,
        11,
        "low-score claim: PAGE_SIZE=16",
        "low-score detailed evidence",
    )
    _write_agent(
        workspace,
        "agent1",
        79.7210807,
        12,
        "high-score claim: PAGE_SIZE=128",
        "high-score detailed evidence",
    )
    results = [(_Report(), "agent0"), (_Report(), "agent1")]
    replies = [
        json.dumps({"verdict": "MERGE", "merged_text": "## Canonical experience"}),
        json.dumps({"verdict": "MERGE", "merged_text": "## Canonical detailed"}),
    ]
    runtime = FakeRuntime(replies)

    merge_epoch_kb(
        cwd=tmp_path, inp=_input(),
        results=results, runtime=runtime,
        epoch_num=1,
        epoch_workspace=workspace,
    )

    assert (base_entry / "experience.md").read_text().strip() == "## Canonical experience"
    assert (base_entry / "detailed.md").read_text().strip() == "## Canonical detailed"
    assert not (
        tmp_path
        / "kb"
        / "experience"
        / "by_definition"
        / "elementwise"
        / "abl_t3_decode_mla"
    ).exists()

    prompt = runtime.calls[0]["prompt"]
    assert prompt.index("agent: agent1") < prompt.index("agent: agent0")
    assert "best_geo_mean: 79.7210807" in prompt
    assert '"page_size"' in prompt and "128" in prompt
    assert "conflicting candidate metadata as erroneous" in prompt
    assert "Do not generalize one failed parameter value" in prompt

    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=str(tmp_path / "kb"),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert len([line for line in log.splitlines() if line]) == 2
    assert "epoch 1: reduced 2 agent candidates" in log.splitlines()[0]


def test_single_candidate_is_copied_without_model_merge(tmp_path):
    _git_init_kb(tmp_path / "kb")
    workspace = Directory(base=tmp_path / "1R")
    _write_agent(
        workspace,
        "agent0",
        2.0,
        3,
        "## One experience",
        "## One detailed record",
    )
    runtime = FakeRuntime([])

    merge_epoch_kb(
        cwd=tmp_path, inp=_input(),
        results=[(_Report(), "agent0")], runtime=runtime,
        epoch_num=1,
        epoch_workspace=workspace,
    )

    assert (_entry(tmp_path) / "experience.md").read_text().strip() == "## One experience"
    assert (_entry(tmp_path) / "detailed.md").read_text().strip() == "## One detailed record"
    assert runtime.calls == []


def test_merge_failure_preserves_committed_kb(tmp_path):
    base_entry = _entry(tmp_path)
    base_entry.mkdir(parents=True)
    (base_entry / "experience.md").write_text("## Stable experience")
    (base_entry / "detailed.md").write_text("## Stable details")
    _git_init_kb(tmp_path / "kb")

    workspace = Directory(base=tmp_path / "1R")
    _write_agent(workspace, "agent0", 2.0, 1, "candidate A", "details A")
    _write_agent(workspace, "agent1", 3.0, 2, "candidate B", "details B")
    runtime = FakeRuntime(["invalid", "invalid", "invalid", "invalid"])

    merge_epoch_kb(
        cwd=tmp_path, inp=_input(),
        results=[(_Report(), "agent0"), (_Report(), "agent1")], runtime=runtime,
        epoch_num=1,
        epoch_workspace=workspace,
    )

    assert (base_entry / "experience.md").read_text() == "## Stable experience"
    assert (base_entry / "detailed.md").read_text() == "## Stable details"
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=str(tmp_path / "kb"),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert len([line for line in log.splitlines() if line]) == 1


def test_merge_cancellation_is_not_downgraded_to_fallback(
    tmp_path,
    monkeypatch,
):
    base_entry = _entry(tmp_path)
    base_entry.mkdir(parents=True)
    (base_entry / "experience.md").write_text("## Stable experience")
    _git_init_kb(tmp_path / "kb")

    workspace = Directory(base=tmp_path / "1R")
    _write_agent(workspace, "agent0", 2.0, 1, "candidate A", "details A")
    _write_agent(workspace, "agent1", 3.0, 2, "candidate B", "details B")
    cancellation = RunCancelled(
        CancellationState(requested=True, reason="stop knowledge merge"),
        stage="AFTER_MODEL_INVOCATION",
    )

    def cancel_merge(*args, **kwargs):
        raise cancellation

    monkeypatch.setattr(
        "kernelgen.workflows.knowledge_reducer.MergeAgent.run",
        cancel_merge,
    )

    with pytest.raises(RunCancelled, match="stop knowledge merge"):
        merge_epoch_kb(
            cwd=tmp_path, inp=_input(),
            results=[(_Report(), "agent0"), (_Report(), "agent1")], runtime=FakeRuntime([]),
            epoch_num=1,
            epoch_workspace=workspace,
        )

    assert (base_entry / "experience.md").read_text() == "## Stable experience"


def test_candidate_with_mismatched_ledger_is_ignored(tmp_path):
    _git_init_kb(tmp_path / "kb")
    workspace = Directory(base=tmp_path / "1R")
    _write_agent(workspace, "agent0", 2.0, 1, "wrong scope", "wrong details")
    ledger_path = Path(workspace.path_of("agent0")) / ".ledger.json"
    history = json.loads(ledger_path.read_text())
    history["definition_name"] = "another_definition"
    ledger_path.write_text(json.dumps(history))

    runtime = FakeRuntime([])
    merge_epoch_kb(
        cwd=tmp_path, inp=_input(),
        results=[(_Report(), "agent0")], runtime=runtime,
        epoch_num=1,
        epoch_workspace=workspace,
    )

    assert not _entry(tmp_path).exists()
    assert runtime.calls == []


def test_canonical_documents_are_size_bounded(tmp_path):
    _git_init_kb(tmp_path / "kb")
    workspace = Directory(base=tmp_path / "1R")
    _write_agent(
        workspace,
        "agent0",
        2.0,
        1,
        "e" * 20_000,
        "d" * 40_000,
    )
    runtime = FakeRuntime([])

    merge_epoch_kb(
        cwd=tmp_path, inp=_input(),
        results=[(_Report(), "agent0")], runtime=runtime,
        epoch_num=1,
        epoch_workspace=workspace,
    )

    assert len((_entry(tmp_path) / "experience.md").read_text()) <= 12_001
    assert len((_entry(tmp_path) / "detailed.md").read_text()) <= 30_001


def test_workspace_setup_does_not_create_elementwise_placeholder(tmp_path):
    kernelgen_root = tmp_path / "source"
    source_kb = kernelgen_root / "kb"
    source_kb.mkdir(parents=True)
    (source_kb / "README.md").write_text("# KB")
    (source_kb / ".git").mkdir()
    (source_kb / ".git" / "source-only-sentinel").write_text("history")
    workspace = tmp_path / "run"

    setup_workspace(workspace, kernelgen_root)

    assert (workspace / "kb" / "README.md").is_file()
    assert not (workspace / "kb" / "experience" / "by_definition" / "elementwise").exists()
    assert not (workspace / "kb" / ".git" / "source-only-sentinel").exists()


def test_workspace_setup_skips_legacy_kb_for_shared_catalog(tmp_path):
    kernelgen_root = tmp_path / "source"
    source_kb = kernelgen_root / "kb"
    source_kb.mkdir(parents=True)
    (source_kb / "README.md").write_text("# legacy KB")
    claude = kernelgen_root / ".claude"
    claude.mkdir()
    (claude / "settings.json").write_text("{}")
    mcp_config = kernelgen_root / ".kernelgen" / "mcp.json"
    mcp_config.parent.mkdir(parents=True)
    mcp_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "servers": {
                    "kernelgen": {
                        "transport": "stdio",
                        "command": [
                            "python3",
                            "-m",
                            "kernelgen.mcp_server.server",
                        ],
                        "startup_timeout_seconds": 30,
                        "tool_timeout_seconds": 2100,
                    }
                },
            }
        )
    )
    shared_catalog = tmp_path / "shared-catalog"
    shared_catalog.mkdir()
    workspace = tmp_path / "run"

    setup_workspace(
        workspace,
        kernelgen_root,
        knowledge_catalog_path=shared_catalog,
    )

    assert (workspace / ".claude" / "settings.json").is_file()
    assert (workspace / ".kernelgen" / "mcp.json").is_file()
    assert (workspace / ".mcp.json").is_file()
    assert not (workspace / "kb").exists()


if __name__ == "__main__":
    import inspect
    import traceback

    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            if "tmp_path" in inspect.signature(test).parameters:
                with tempfile.TemporaryDirectory() as directory:
                    test(Path(directory))
            else:
                test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)

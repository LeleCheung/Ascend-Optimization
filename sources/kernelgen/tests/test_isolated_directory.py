"""Unit tests for IsolatedDirectory workspace (ADR-3 #9b).

Verifies: directory creation with role/provider config + tmp/ + kb/ copies, cleanup,
and that agent workspace is benchmark-safe (no repo source visible).

    cd /data/akg_kernel_bench_lite
    python -m pytest tests/test_isolated_directory.py -v
    # or:
    python tests/test_isolated_directory.py
"""

import json
import sys
import tempfile
from pathlib import Path


from kernelgen.framework.parallel import (  # noqa: E402
    Directory,
    IsolatedDirectory,
    copy_claude_directory,
    copy_mcp_configuration,
)


def _setup_sources(tmp_path):
    """Create mock neutral config, Claude settings, MCP config, and KB."""
    claude_src = tmp_path / "repo" / ".claude"
    claude_src.mkdir(parents=True)
    (claude_src / "settings.json").write_text('{"model": "test"}')
    (claude_src / "settings.local.json").write_text(
        '{"env": {"ANTHROPIC_AUTH_TOKEN": "machine-local"}}'
    )
    skills_dir = tmp_path / "repo" / ".kernelgen" / "skills" / "eval-kernel"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\n"
        "name: eval-kernel\n"
        "description: Evaluate one test kernel.\n"
        "---\n"
        "\n"
        "# eval-kernel skill\n"
    )
    agents_dir = tmp_path / "repo" / ".kernelgen" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "kernel-coder.md").write_text(
        "---\n"
        "name: kernel-coder\n"
        "description: test agent\n"
        "capabilities: [read, write, edit, search]\n"
        "mcp_tools: []\n"
        "subagents: []\n"
        "model: inherit\n"
        "---\n"
        "role"
    )

    mcp_src = tmp_path / "repo" / ".kernelgen" / "mcp.json"
    mcp_src.write_text(
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
                        "tool_timeout_seconds": 120,
                    }
                },
            }
        )
    )

    kb_src = tmp_path / "repo" / "kb"
    kb_src.mkdir(parents=True)
    exp_dir = kb_src / "experience" / "by_definition" / "elementwise" / "gelu" / "Ascend910B"
    exp_dir.mkdir(parents=True)
    (exp_dir / "experience.md").write_text("## Prior experience")

    return claude_src, mcp_src, kb_src


def test_creates_workspace_with_tmp(tmp_path):
    ws = IsolatedDirectory(base=tmp_path / "workspaces")
    path = ws.allocate("agent0")
    assert (Path(path) / "tmp").is_dir()


def test_copies_claude_directory(tmp_path):
    claude_src, _, _ = _setup_sources(tmp_path)
    ws = IsolatedDirectory(base=tmp_path / "ws", claude_source=claude_src)
    path = ws.allocate("agent0")
    p = Path(path)
    assert (p / ".claude" / "settings.json").exists()
    assert (p / ".claude" / "skills" / "eval-kernel" / "SKILL.md").exists()
    assert (p / ".agents" / "skills" / "eval-kernel" / "SKILL.md").exists()
    assert (p / ".claude" / "agents" / "kernel-coder.md").exists()
    assert (p / ".codex" / "agents" / "kernel-coder.toml").exists()
    assert (p / ".kernelgen" / "agents" / "kernel-coder.md").exists()
    assert (p / ".kernelgen" / "skills" / "eval-kernel" / "SKILL.md").exists()
    content = (p / ".claude" / "settings.json").read_text()
    assert "test" in content
    assert (p / ".claude" / "settings.local.json").read_text() == (
        claude_src / "settings.local.json"
    ).read_text()


def test_copy_claude_directory_overwrites_stale_local_settings(tmp_path):
    claude_src, _, _ = _setup_sources(tmp_path)
    destination = tmp_path / "workspace" / ".claude"
    destination.mkdir(parents=True)
    (destination / "settings.local.json").write_text('{"model": "stale"}')

    copy_claude_directory(claude_src, destination)

    assert (destination / "settings.json").exists()
    assert (destination / "settings.local.json").read_text() == (
        claude_src / "settings.local.json"
    ).read_text()


def test_copy_claude_directory_can_exclude_all_skills(tmp_path):
    claude_src, _, _ = _setup_sources(tmp_path)
    destination = tmp_path / "workspace" / ".claude"
    stale = destination / "skills" / "stale"
    stale.mkdir(parents=True)
    (stale / "SKILL.md").write_text("# stale")

    copy_claude_directory(
        claude_src,
        destination,
        include_skills=False,
    )

    assert (destination / "agents" / "kernel-coder.md").is_file()
    assert (
        destination.parent / ".codex" / "agents" / "kernel-coder.toml"
    ).is_file()
    assert not (destination / "skills").exists()
    assert not (destination.parent / ".agents" / "skills").exists()


def test_copy_claude_directory_materializes_neutral_sources_without_claude_dir(
    tmp_path,
):
    claude_src, _, _ = _setup_sources(tmp_path)
    for child in claude_src.iterdir():
        child.unlink()
    claude_src.rmdir()
    destination = tmp_path / "workspace" / ".claude"

    copy_claude_directory(claude_src, destination)

    assert (destination / "agents" / "kernel-coder.md").is_file()
    assert (destination / "skills" / "eval-kernel" / "SKILL.md").is_file()
    assert (
        destination.parent / ".agents" / "skills" / "eval-kernel" / "SKILL.md"
    ).is_file()


def test_isolated_directory_can_exclude_provider_skills(tmp_path):
    claude_src, _, _ = _setup_sources(tmp_path)
    workspace = IsolatedDirectory(
        base=tmp_path / "ws",
        claude_source=claude_src,
        include_skills=False,
    )

    path = Path(workspace.allocate("agent0"))

    assert (path / ".claude" / "agents" / "kernel-coder.md").is_file()
    assert (path / ".codex" / "agents" / "kernel-coder.toml").is_file()
    assert not (path / ".claude" / "skills").exists()
    assert not (path / ".agents" / "skills").exists()


def test_isolated_directory_keeps_legacy_include_claude_skills_alias(tmp_path):
    workspace = IsolatedDirectory(
        base=tmp_path / "ws",
        include_claude_skills=False,
    )

    assert workspace.include_skills is False


def test_copies_mcp_config(tmp_path):
    _, mcp_src, _ = _setup_sources(tmp_path)
    ws = IsolatedDirectory(base=tmp_path / "ws", mcp_source=mcp_src)
    path = Path(ws.allocate("agent0"))
    neutral = json.loads((path / ".kernelgen" / "mcp.json").read_text())
    claude = json.loads((path / ".mcp.json").read_text())
    assert neutral["servers"]["kernelgen"]["command"][0] == sys.executable
    assert claude["mcpServers"]["kernelgen"]["command"] == sys.executable
    assert neutral["servers"]["kernelgen"]["tool_timeout_seconds"] == 120
    assert claude["mcpServers"]["kernelgen"]["timeout"] == 120_000


def test_copy_mcp_configuration_sets_kernelgen_timeout(tmp_path):
    _, mcp_src, _ = _setup_sources(tmp_path)
    destination = tmp_path / "workspace" / ".mcp.json"

    copy_mcp_configuration(
        mcp_src,
        destination,
        tool_timeout_seconds=2100,
    )

    config = json.loads(destination.read_text())
    assert config["mcpServers"]["kernelgen"]["timeout"] == 2_100_000
    neutral = json.loads(
        (destination.parent / ".kernelgen" / "mcp.json").read_text()
    )
    assert neutral["servers"]["kernelgen"]["tool_timeout_seconds"] == 2100


def test_copies_kb_directory(tmp_path):
    _, _, kb_src = _setup_sources(tmp_path)
    ws = IsolatedDirectory(base=tmp_path / "ws", kb_source=kb_src)
    path = ws.allocate("agent0")
    p = Path(path)
    exp_file = p / "kb" / "experience" / "by_definition" / "elementwise" / "gelu" / "Ascend910B" / "experience.md"
    assert exp_file.exists()
    assert "Prior experience" in exp_file.read_text()


def test_kb_copy_excludes_source_git_history(tmp_path):
    _, _, kb_src = _setup_sources(tmp_path)
    source_git = kb_src / ".git"
    source_git.mkdir()
    (source_git / "source-only-sentinel").write_text("must not be copied")

    ws = IsolatedDirectory(base=tmp_path / "ws", kb_source=kb_src)
    path = Path(ws.allocate("agent0"))

    assert not (path / "kb" / ".git").exists()


def test_copies_both(tmp_path):
    claude_src, mcp_src, kb_src = _setup_sources(tmp_path)
    ws = IsolatedDirectory(
        base=tmp_path / "ws", claude_source=claude_src,
        mcp_source=mcp_src, kb_source=kb_src,
    )
    path = ws.allocate("agent0")
    p = Path(path)
    assert (p / "tmp").is_dir()
    assert (p / ".claude" / "settings.json").exists()
    assert (p / ".kernelgen" / "mcp.json").exists()
    assert (p / ".mcp.json").exists()
    assert (p / "kb" / "experience").is_dir()


def test_multiple_agents_isolated(tmp_path):
    claude_src, _, kb_src = _setup_sources(tmp_path)
    ws = IsolatedDirectory(base=tmp_path / "ws", claude_source=claude_src, kb_source=kb_src)
    p0 = Path(ws.allocate("agent0"))
    p1 = Path(ws.allocate("agent1"))
    # Each has its own copy (not shared)
    (p0 / "tmp" / "main.py").write_text("# agent 0 code")
    (p1 / "tmp" / "main.py").write_text("# agent 1 code")
    assert (p0 / "tmp" / "main.py").read_text() == "# agent 0 code"
    assert (p1 / "tmp" / "main.py").read_text() == "# agent 1 code"
    # KB is independent (write to one doesn't affect the other)
    kb0 = p0 / "kb" / "new_file.md"
    kb0.write_text("agent0 distilled")
    assert not (p1 / "kb" / "new_file.md").exists()


def test_no_repo_source_visible(tmp_path):
    """Benchmark safety: agent's workspace has ONLY what we put in (tmp/.claude/kb),
    no repo source code / reference implementations / other solutions."""
    claude_src, _, kb_src = _setup_sources(tmp_path)
    # Put some "repo source" next to .claude
    (tmp_path / "repo" / "flashinfer_bench").mkdir()
    (tmp_path / "repo" / "flashinfer_bench" / "secret.py").write_text("answer = 42")
    ws = IsolatedDirectory(base=tmp_path / "ws", claude_source=claude_src, kb_source=kb_src)
    path = ws.allocate("agent0")
    # Agent can NOT see repo source
    all_files = list(Path(path).rglob("*"))
    file_names = [f.name for f in all_files if f.is_file()]
    assert "secret.py" not in file_names  # repo source NOT copied
    assert "settings.json" in file_names  # .claude IS copied
    assert "experience.md" in file_names  # kb IS copied


def test_no_sources_still_works(tmp_path):
    """With no provider config or KB source, just create tmp/."""
    ws = IsolatedDirectory(base=tmp_path / "ws")
    path = ws.allocate("agent0")
    p = Path(path)
    assert (p / "tmp").is_dir()
    assert not (p / ".claude").exists()
    assert not (p / ".codex").exists()
    assert not (p / ".kernelgen").exists()
    assert not (p / "kb").exists()


def test_cleanup_removes_workspace(tmp_path):
    claude_src, _, kb_src = _setup_sources(tmp_path)
    ws = IsolatedDirectory(base=tmp_path / "ws", claude_source=claude_src, kb_source=kb_src)
    path = ws.allocate("agent0")
    assert Path(path).exists()
    ws.cleanup("agent0")
    assert not Path(path).exists()


def test_path_of(tmp_path):
    ws = IsolatedDirectory(base=tmp_path / "ws")
    ws.allocate("task42")
    assert ws.path_of("task42") == str(tmp_path / "ws" / "task42")


if __name__ == "__main__":
    import inspect
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            if "tmp_path" in inspect.signature(t).parameters:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)

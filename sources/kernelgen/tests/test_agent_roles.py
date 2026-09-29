"""Tests for provider-neutral role validation and provider materialization."""

from __future__ import annotations

from pathlib import Path

import yaml

from kernelgen.framework.agent_roles import (
    canonical_agent_roles_dir,
    load_agent_role,
    load_agent_roles,
    materialize_agent_roles,
)


def _write_role(
    root: Path,
    name: str,
    *,
    capabilities: str = "[read, search]",
    mcp_tools: str = "[]",
    subagents: str = "[]",
    approval: str = "default",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: test role {name}\n"
        f"capabilities: {capabilities}\n"
        f"mcp_tools: {mcp_tools}\n"
        f"subagents: {subagents}\n"
        "model: inherit\n"
        f"approval: {approval}\n"
        "---\n"
        f"ROLE {name}\n",
        encoding="utf-8",
    )
    return path


def test_repository_role_catalog_is_provider_neutral_and_complete():
    roles = load_agent_roles(canonical_agent_roles_dir())

    assert "kernel-coder" in roles
    assert "kernel-native-to-flaggems" in roles
    assert roles["kernel-coder"].subagents == ("kernel-profile-analyzer",)
    assert "eval_round" in roles["kernel-coder"].mcp_tools
    assert len(roles) == 21
    assert roles["kernel-gems-case-collector"].capabilities == ("read",)
    assert roles["kernel-artifact-reviewer"].capabilities == ("read",)
    assert roles["kernel-artifact-reviewer"].mcp_tools == ()


def test_materialization_generates_claude_and_codex_from_one_snapshot(tmp_path):
    source = tmp_path / "source"
    _write_role(
        source,
        "kernel-child",
        capabilities="[read]",
        mcp_tools="[get_profile_context]",
        approval="no_prompts",
    )
    _write_role(
        source,
        "kernel-parent",
        capabilities="[shell, read, write, edit, search, skill]",
        subagents="[kernel-child]",
    )
    workspace = tmp_path / "workspace"

    materialize_agent_roles(source, workspace)

    neutral = workspace / ".kernelgen" / "agents" / "kernel-parent.md"
    claude = workspace / ".claude" / "agents" / "kernel-parent.md"
    codex = workspace / ".codex" / "agents" / "kernel-parent.toml"
    assert neutral.read_text(encoding="utf-8") == (
        source / "kernel-parent.md"
    ).read_text(encoding="utf-8")
    claude_metadata = yaml.safe_load(
        claude.read_text(encoding="utf-8").split("---", 2)[1]
    )
    assert claude_metadata["tools"] == (
        "Bash, Read, Write, Edit, Glob, Grep, Skill, Agent(kernel-child)"
    )
    codex_text = codex.read_text(encoding="utf-8")
    assert 'name = "kernel-parent"' in codex_text
    assert "developer_instructions = " in codex_text
    assert "[mcp_servers.kernelgen]" in codex_text
    assert "enabled_tools = []" in codex_text
    child_claude = (
        workspace / ".claude" / "agents" / "kernel-child.md"
    ).read_text(encoding="utf-8")
    assert "permissionMode: dontAsk" in child_claude
    child_codex = (
        workspace / ".codex" / "agents" / "kernel-child.toml"
    ).read_text(encoding="utf-8")
    assert 'enabled_tools = ["get_profile_context"]' in child_codex


def test_role_validation_rejects_provider_specific_frontmatter(tmp_path):
    path = _write_role(tmp_path, "kernel-invalid")
    text = path.read_text(encoding="utf-8").replace(
        "capabilities: [read, search]",
        "tools: Read, Glob, Grep",
    )
    path.write_text(text, encoding="utf-8")

    try:
        load_agent_role(path)
    except RuntimeError as exc:
        assert "unknown provider-neutral agent fields" in str(exc)
    else:
        raise AssertionError("provider-specific role metadata must be rejected")

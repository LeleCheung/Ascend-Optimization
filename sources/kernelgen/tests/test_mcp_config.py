"""Tests for provider-neutral MCP configuration and materialization."""

from __future__ import annotations

import json
import sys

import pytest

from kernelgen.framework.mcp_config import (
    canonical_mcp_configuration_path,
    load_kernelgen_mcp_configuration,
    load_mcp_configuration,
    materialize_mcp_configuration,
)


def test_repository_mcp_configuration_is_provider_neutral():
    path = canonical_mcp_configuration_path()
    server = load_kernelgen_mcp_configuration(path)

    assert server.command == (
        "python3",
        "-m",
        "kernelgen.mcp_server.server",
    )
    assert server.startup_timeout_seconds == 30
    assert server.tool_timeout_seconds == 2100
    assert "mcpServers" not in path.read_text(encoding="utf-8")


def test_materialization_snapshots_neutral_and_generates_claude_config(tmp_path):
    neutral_path, claude_path = materialize_mcp_configuration(
        canonical_mcp_configuration_path(),
        tmp_path,
        tool_timeout_seconds=75,
    )

    neutral = json.loads(neutral_path.read_text(encoding="utf-8"))
    claude = json.loads(claude_path.read_text(encoding="utf-8"))
    assert neutral["servers"]["kernelgen"]["tool_timeout_seconds"] == 75
    assert neutral["servers"]["kernelgen"]["command"][0] == sys.executable
    assert claude["mcpServers"]["kernelgen"] == {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "kernelgen.mcp_server.server"],
        "timeout": 75_000,
    }


def test_materialization_preserves_virtualenv_launcher_path(tmp_path):
    virtualenv_python = tmp_path / "venv" / "bin" / "python"
    virtualenv_python.parent.mkdir(parents=True)
    virtualenv_python.symlink_to(sys.executable)

    neutral_path, claude_path = materialize_mcp_configuration(
        canonical_mcp_configuration_path(),
        tmp_path / "workspace",
        python_executable=virtualenv_python,
    )

    neutral = json.loads(neutral_path.read_text(encoding="utf-8"))
    claude = json.loads(claude_path.read_text(encoding="utf-8"))
    expected = str(virtualenv_python)
    assert neutral["servers"]["kernelgen"]["command"][0] == expected
    assert claude["mcpServers"]["kernelgen"]["command"] == expected


def test_neutral_loader_rejects_provider_specific_schema(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "kernelgen": {
                        "command": "python3",
                        "args": ["-m", "kernelgen.mcp_server.server"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unknown MCP configuration fields"):
        load_mcp_configuration(path)

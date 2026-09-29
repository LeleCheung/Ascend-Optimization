"""Unit tests for the Codex CLI runtime adapter."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from kernelgen.framework import (
    CodexRuntime,
    create_cli_runtime,
    resolve_cli_runtime_options,
)
from kernelgen.framework.runtime import CLIRuntime, PumpState, Runtime


def _event(**payload) -> str:
    return json.dumps(payload)


def _workspace(tmp_path: Path, *, with_mcp: bool = True) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if with_mcp:
        config_path = workspace / ".kernelgen" / "mcp.json"
        config_path.parent.mkdir()
        config_path.write_text(
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
                    }
                }
            ),
            encoding="utf-8",
        )
    return workspace


def _write_agent(
    workspace: Path,
    name: str,
    *,
    capabilities: tuple[str, ...] = ("read",),
    mcp_tools: tuple[str, ...] = (),
    subagents: tuple[str, ...] = (),
    body: str = "ROLE",
) -> None:
    root = workspace / ".kernelgen" / "agents"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: test role\n"
        f"capabilities: {json.dumps(capabilities)}\n"
        f"mcp_tools: {json.dumps(mcp_tools)}\n"
        f"subagents: {json.dumps(subagents)}\n"
        "model: inherit\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )


def _config_values(command: list[str]) -> list[str]:
    return [command[index + 1] for index, value in enumerate(command) if value == "--config"]


def test_public_runtime_contract():
    assert issubclass(CodexRuntime, CLIRuntime)
    assert isinstance(CodexRuntime(), Runtime)
    assert CodexRuntime.supports_native_agents is False
    assert CodexRuntime.supports_agent_roles is True


def test_provider_neutral_factory_constructs_codex(tmp_path):
    runtime = create_cli_runtime(
        "codex",
        workspace=tmp_path,
        model="inherit",
        auth_token="secret",
    )

    assert isinstance(runtime, CodexRuntime)
    assert runtime.api_key == "secret"


def test_provider_neutral_factory_rejects_unknown_runtime(tmp_path):
    with pytest.raises(ValueError, match="unsupported CLI runtime"):
        create_cli_runtime("unknown", workspace=tmp_path)


def test_provider_options_resolve_codex_environment(monkeypatch):
    monkeypatch.setenv("CODEX_MODEL", "gpt-codex-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.invalid/v1")
    monkeypatch.setenv("CODEX_API_KEY", "codex-secret")

    options = resolve_cli_runtime_options("codex")

    assert options == {
        "model": "gpt-codex-test",
        "base_url": "https://api.example.invalid/v1",
        "auth_token": "codex-secret",
    }


def test_provider_options_keep_explicit_values_over_environment(monkeypatch):
    monkeypatch.setenv("CODEX_MODEL", "environment-model")

    options = resolve_cli_runtime_options(
        "codex",
        model="explicit-model",
        base_url="https://explicit.example.invalid/v1",
        auth_token="explicit-secret",
    )

    assert options["model"] == "explicit-model"
    assert options["base_url"] == "https://explicit.example.invalid/v1"
    assert options["auth_token"] == "explicit-secret"


def test_fresh_command_uses_machine_readable_noninteractive_mode(tmp_path):
    runtime = CodexRuntime(workspace=_workspace(tmp_path), model="gpt-test")

    command = runtime._command("prompt", None)

    assert command[:5] == ["codex", "exec", "--color", "never", "--json"]
    assert command[-1] == "-"
    assert "resume" not in command
    assert command[command.index("--model") + 1] == "gpt-test"
    config = _config_values(command)
    assert 'approval_policy="never"' in config
    assert 'sandbox_mode="workspace-write"' in config
    assert 'mcp_servers.kernelgen.command="python3"' in config
    assert 'mcp_servers.kernelgen.args=["-m", "kernelgen.mcp_server.server"]' in config
    assert "mcp_servers.kernelgen.required=true" in config
    assert "mcp_servers.kernelgen.startup_timeout_sec=30.0" in config
    assert "mcp_servers.kernelgen.tool_timeout_sec=2100.0" in config


def test_resume_command_targets_exact_session_and_inherits_model(tmp_path):
    runtime = CodexRuntime(workspace=_workspace(tmp_path), model="inherit")
    session = "0199a213-81c0-7800-8aa1-bbab2a035a53"

    command = runtime._command("continue", session)

    assert command[:6] == ["codex", "exec", "--color", "never", "resume", "--json"]
    assert command[-2:] == [session, "-"]
    assert "--model" not in command


def test_command_can_keep_user_config_and_override_base_url(tmp_path):
    runtime = CodexRuntime(
        workspace=_workspace(tmp_path),
        ignore_user_config=False,
        base_url="https://api.example.invalid/v1",
    )

    command = runtime._command("prompt", None)

    assert "--ignore-user-config" not in command
    assert 'openai_base_url="https://api.example.invalid/v1"' in _config_values(command)


def test_mcp_env_is_forwarded_without_exposing_values_in_argv(tmp_path):
    workspace = _workspace(tmp_path)
    config_path = workspace / ".kernelgen" / "mcp.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["servers"]["kernelgen"]["env"] = {"SECRET_TOKEN": "private-value"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    runtime = CodexRuntime(workspace=workspace)

    command = runtime._command("prompt", None)
    child_env = runtime._env()

    assert "private-value" not in " ".join(command)
    assert child_env["SECRET_TOKEN"] == "private-value"
    assert 'mcp_servers.kernelgen.env_vars=["SECRET_TOKEN"]' in _config_values(command)


def test_invalid_mcp_configuration_fails_before_spawn(tmp_path):
    workspace = _workspace(tmp_path)
    (workspace / ".kernelgen" / "mcp.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="schema_version"):
        CodexRuntime(workspace=workspace)._command("prompt", None)


def test_explicit_agent_role_uses_canonical_definition(tmp_path):
    workspace = _workspace(tmp_path)
    _write_agent(
        workspace,
        "kernel-profile-analyzer",
        mcp_tools=("get_profile_context",),
        body="PROFILE ROLE",
    )
    runtime = CodexRuntime(workspace=workspace)

    runtime._command("analyze round 1", None, agent="kernel-profile-analyzer")

    effective = runtime._effective_prompt("analyze round 1")
    assert "PROFILE ROLE" in effective
    assert "mcp__kernelgen__get_profile_context" in effective
    assert "--- TASK ---\nanalyze round 1" in effective
    assert (
        'mcp_servers.kernelgen.enabled_tools=["get_profile_context"]'
        in _config_values(runtime._command(
            "analyze round 1", None, agent="kernel-profile-analyzer"
        ))
    )


def test_explicit_mcp_agent_requires_workspace_configuration(tmp_path):
    workspace = _workspace(tmp_path, with_mcp=False)
    _write_agent(
        workspace,
        "kernel-profile-analyzer",
        mcp_tools=("get_profile_context",),
    )

    with pytest.raises(RuntimeError, match="requires KernelGen MCP tools"):
        CodexRuntime(workspace=workspace)._command(
            "prompt", None, agent="kernel-profile-analyzer"
        )


def test_explicit_agent_requires_complete_canonical_metadata(tmp_path):
    workspace = _workspace(tmp_path)
    root = workspace / ".kernelgen" / "agents"
    root.mkdir(parents=True)
    (root / "incomplete.md").write_text(
        "---\n"
        "name: incomplete\n"
        "description: missing capabilities\n"
        "mcp_tools: []\n"
        "subagents: []\n"
        "model: inherit\n"
        "---\n"
        "ROLE\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="agent capabilities must be"):
        CodexRuntime(workspace=workspace)._command(
            "prompt", None, agent="incomplete"
        )


def test_resumed_agent_does_not_repeat_role(tmp_path):
    workspace = _workspace(tmp_path)
    _write_agent(workspace, "kernel-profile-analyzer", body="PROFILE ROLE")
    runtime = CodexRuntime(workspace=workspace)

    command = runtime._command(
        "continue",
        "0199a213-81c0-7800-8aa1-bbab2a035a53",
        agent="kernel-profile-analyzer",
    )

    assert runtime._effective_prompt("continue") == "continue"
    assert "mcp_servers.kernelgen.enabled_tools=[]" in _config_values(command)


def test_parse_successful_turn_accumulates_agent_text_and_session():
    runtime = CodexRuntime()
    state = PumpState()

    runtime._parse_line(
        _event(
            type="thread.started",
            thread_id="0199a213-81c0-7800-8aa1-bbab2a035a53",
        ),
        state,
    )
    runtime._parse_line(_event(type="turn.started"), state)
    runtime._parse_line(
        _event(
            type="item.completed",
            item={"id": "item-1", "type": "agent_message", "text": "done"},
        ),
        state,
    )
    progress = runtime._parse_line(
        _event(
            type="turn.completed",
            usage={"input_tokens": 12, "cached_input_tokens": 3, "output_tokens": 4},
        ),
        state,
    )

    assert state.session_id == "0199a213-81c0-7800-8aa1-bbab2a035a53"
    assert state.turn_count == 1
    assert state.text == "done"
    assert state.done_ok is True
    assert "in=12" in progress and "cache=3" in progress


def test_parse_reasoning_command_and_mcp_progress_keeps_full_evidence():
    runtime = CodexRuntime()
    state = PumpState()

    reasoning = runtime._parse_line(
        _event(
            type="item.completed",
            item={"type": "reasoning", "text": "inspect the layout"},
        ),
        state,
    )
    command = runtime._parse_line(
        _event(
            type="item.completed",
            item={
                "type": "command_execution",
                "command": "rg layout",
                "aggregated_output": "main.py:12",
                "exit_code": 0,
            },
        ),
        state,
    )
    mcp = runtime._parse_line(
        _event(
            type="item.completed",
            item={
                "type": "mcp_tool_call",
                "server": "kernelgen",
                "tool": "eval_round",
                "result": {"status": "PASSED"},
            },
        ),
        state,
    )

    assert state.thinking == "inspect the layout"
    assert "inspect the layout" in reasoning
    assert "main.py:12" in command
    assert "PASSED" in mcp
    assert state.text == ""


def test_failed_turn_is_resumable_after_thread_started():
    runtime = CodexRuntime()
    state = PumpState()
    runtime._parse_line(
        _event(
            type="thread.started",
            thread_id="0199a213-81c0-7800-8aa1-bbab2a035a53",
        ),
        state,
    )

    progress = runtime._parse_line(
        _event(type="turn.failed", error={"message": "connection reset"}),
        state,
    )

    assert state.done_ok is False
    assert state.resumable is True
    assert "connection reset" in progress


def test_restore_session_uses_latest_exact_log_record(tmp_path):
    workspace = _workspace(tmp_path)
    log_dir = workspace / ".kernelgen"
    log_dir.mkdir(exist_ok=True)
    first = "0199a213-81c0-7800-8aa1-bbab2a035a53"
    second = "0299a213-81c0-7800-8aa1-bbab2a035a54"
    (log_dir / "codex-runtime.log").write_text(
        f"[codex] session={first}\n"
        "prompt mentions session=not-state\n"
        f"[codex] session={second}\n",
        encoding="utf-8",
    )

    runtime = CodexRuntime(workspace=workspace)

    assert runtime.restore_session() == second
    assert runtime.last_session_id == second


def test_child_environment_preserves_codex_home_but_drops_parent_session(
    tmp_path,
    monkeypatch,
):
    workspace = _workspace(tmp_path)
    monkeypatch.setenv("CODEX_HOME", "/safe/codex-home")
    monkeypatch.setenv("CODEX_SESSION_ID", "parent-session")
    monkeypatch.setenv("CODEX_THREAD_ID", "parent-thread")
    monkeypatch.setenv("CODEX_REMOTE_PAYLOAD", "parent-payload")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-provider-secret")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unrelated-provider-token")
    runtime = CodexRuntime(workspace=workspace, api_key="runtime-key")
    runtime._command("prompt", None)

    env = runtime._env()

    assert env["CODEX_HOME"] == "/safe/codex-home"
    assert env["CODEX_API_KEY"] == "runtime-key"
    assert env["KERNELGEN_WORKSPACE"] == str(workspace.resolve())
    assert "CODEX_SESSION_ID" not in env
    assert "CODEX_THREAD_ID" not in env
    assert "CODEX_REMOTE_PAYLOAD" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env


def test_workspace_human_log_can_disable_console_mirroring(tmp_path):
    workspace = _workspace(tmp_path)
    terminal = io.StringIO()
    runtime = CodexRuntime(
        workspace=workspace,
        log_stream=terminal,
        mirror_to_console=False,
    )
    runtime._prepare()
    try:
        runtime._print("[codex] private workspace detail")
    finally:
        runtime._cleanup()

    human = (workspace / ".kernelgen" / "codex-runtime.log").read_text()
    assert "private workspace detail" in human
    assert terminal.getvalue() == ""


def test_constructor_rejects_unsafe_unknown_policy_values():
    with pytest.raises(ValueError, match="sandbox mode"):
        CodexRuntime(sandbox_mode="unknown")
    with pytest.raises(ValueError, match="approval policy"):
        CodexRuntime(approval_policy="on-failure")

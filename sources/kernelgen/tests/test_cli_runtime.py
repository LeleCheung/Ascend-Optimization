"""Unit tests for the CLIRuntime skeleton + ClaudeRuntime hooks (ADR-3 #9).

Host-testable without spawning a real CLI: we test the pure parts — command
building, env injection, stream-json line parsing (_parse_line), and pump
accumulation over fake and real subprocess streams.

    cd /data/akg_kernel_bench_lite
    python -m pytest tests/test_cli_runtime.py -v
    # or:
    python tests/test_cli_runtime.py
"""

import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

import pytest

import kernelgen.framework.runtime.claude as claude_module
from kernelgen.framework.mcp_config import (
    canonical_mcp_configuration_path,
    materialize_mcp_configuration,
)
from kernelgen.framework.runtime import (  # noqa: E402
    CLIRuntime,
    ClaudeRuntime,
    ClaudeCodeRuntime,
    FakeRuntime,
    IdleTimeout,
    PumpState,
    Runtime,
    materialize_claude_runtime_config,
)


def _ev(**kw):
    return json.dumps(kw)


def _native_workspace(agent: str, tools: str = "Read"):
    temp = tempfile.TemporaryDirectory()
    root = Path(temp.name)
    agents = root / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / f"{agent}.md").write_text(
        f"---\nname: {agent}\ndescription: test\ntools: {tools}\n"
        "model: inherit\n---\nrole"
    )
    materialize_mcp_configuration(canonical_mcp_configuration_path(), root)
    return temp, root


# --- contract / aliases ---------------------------------------------------

def test_hierarchy_and_alias():
    assert issubclass(ClaudeRuntime, CLIRuntime)
    assert ClaudeCodeRuntime is ClaudeRuntime
    assert isinstance(FakeRuntime([]), Runtime)   # protocol
    assert isinstance(ClaudeRuntime(), Runtime)


# --- environment preparation (execution layer owns worktree/env) ----------

def test_env_drops_parent_agent_workspace_boundaries(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("KERNELGEN_WORKSPACE", "/wrong/workspace")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/wrong/project")

    env = ClaudeRuntime(workspace="/tmp/wt", model="m")._env()

    assert "CLAUDECODE" not in env
    assert env["KERNELGEN_WORKSPACE"] == "/tmp/wt"
    assert env["CLAUDE_PROJECT_DIR"] == "/tmp/wt"
    assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"


def test_env_preserves_symlink_runtime_package_parent(tmp_path, monkeypatch):
    real_package = tmp_path / "checkout"
    module_file = real_package / "framework" / "runtime" / "claude.py"
    module_file.parent.mkdir(parents=True)
    module_file.touch()

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "kernelgen").symlink_to(
        real_package,
        target_is_directory=True,
    )
    symlinked_module = (
        runtime_root / "kernelgen" / "framework" / "runtime" / "claude.py"
    )
    monkeypatch.setattr(claude_module, "__file__", str(symlinked_module))
    monkeypatch.setenv("PYTHONPATH", "")

    env = ClaudeRuntime()._env()

    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(runtime_root)


def test_env_without_workspace_does_not_inherit_parent_boundary(monkeypatch):
    monkeypatch.setenv("KERNELGEN_WORKSPACE", "/wrong/workspace")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/wrong/project")

    env = ClaudeRuntime()._env()

    assert "KERNELGEN_WORKSPACE" not in env
    assert "CLAUDE_PROJECT_DIR" not in env


def test_env_injects_runtime_api_config_without_writing_settings(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://ambient.invalid")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-auth-token")
    rt = ClaudeRuntime(
        workspace=tmp_path,
        model="deepseek-test",
        base_url="https://runtime.invalid",
        auth_token="runtime-token",
    )

    env = rt._env()
    rt._prepare()

    assert env["ANTHROPIC_BASE_URL"] == "https://runtime.invalid"
    assert env["ANTHROPIC_API_KEY"] == "runtime-token"
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert not (tmp_path / ".claude" / "settings.local.json").exists()
    assert "--model" in rt._command("hi", None)
    assert "deepseek-test" in rt._command("hi", None)


def test_materialize_campaign_config_for_native_subagent_model(tmp_path):
    config_dir = materialize_claude_runtime_config(tmp_path, "deepseek-test")

    assert config_dir == (tmp_path / ".kernelgen" / "claude-config").resolve()
    settings_path = config_dir / "settings.json"
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "model": "deepseek-test"
    }
    assert stat.S_IMODE(settings_path.stat().st_mode) == 0o600

    # Updating the campaign model must not remove Claude's unrelated session
    # metadata from the shared config directory.
    sentinel = config_dir / "session-sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    assert materialize_claude_runtime_config(tmp_path, "deepseek-next") == config_dir
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "model": "deepseek-next"
    }
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_materialize_campaign_config_skips_inherited_provider_default(tmp_path):
    assert materialize_claude_runtime_config(tmp_path, "inherit") is None
    assert not (tmp_path / ".kernelgen" / "claude-config").exists()


def test_env_uses_explicit_campaign_claude_config(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/ambient/config")
    config_dir = materialize_claude_runtime_config(tmp_path, "deepseek-test")

    runtime = ClaudeRuntime(
        workspace=tmp_path / "agent",
        model="deepseek-test",
        claude_config_dir=config_dir,
    )

    assert runtime._env()["CLAUDE_CONFIG_DIR"] == str(config_dir)
    assert not (tmp_path / "agent" / ".claude" / "settings.local.json").exists()


def test_env_migrates_legacy_shell_auth_token(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://shell.invalid")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "shell-token")

    env = ClaudeRuntime()._env()

    assert env["ANTHROPIC_BASE_URL"] == "https://shell.invalid"
    assert env["ANTHROPIC_API_KEY"] == "shell-token"
    assert "ANTHROPIC_AUTH_TOKEN" not in env


def test_env_prefers_api_key_and_drops_legacy_auth_token(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "shell-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "stale-auth-token")

    env = ClaudeRuntime()._env()

    assert env["ANTHROPIC_API_KEY"] == "shell-api-key"
    assert "ANTHROPIC_AUTH_TOKEN" not in env


def test_command_new_and_resume():
    rt = ClaudeRuntime(model="ep-x")
    cmd = rt._command("hi", None)
    assert cmd[:2] == ["claude", "-p"]
    assert "--resume" not in cmd
    assert "ep-x" in cmd                             # model wired
    cmd_r = rt._command("hi", "sess-42")
    assert "--resume" in cmd_r and "sess-42" in cmd_r


def test_restore_session_uses_latest_exact_init_line(tmp_path):
    first = "6a8ddaca-3b2f-4cab-b3db-a03d6a061555"
    latest = "750ae380-86d0-4326-8524-cc7bac2a760d"
    log_dir = tmp_path / ".kernelgen"
    log_dir.mkdir()
    (log_dir / "claude-runtime.log").write_text(
        "\n".join(
            [
                f"[claude] session={first} tools=12 skills=3",
                "[claude] text: intermediate output",
                # Prompt/tool text must not be accepted as persisted state.
                f"prefix [claude] session={latest} tools=12 skills=3",
                f"[claude] session={latest} tools=12 skills=3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = ClaudeRuntime(workspace=tmp_path)

    assert runtime.restore_session() == latest
    assert runtime.last_session_id == latest
    assert runtime._command("continue", latest)[-2:] == ["--resume", latest]


def test_restore_session_without_workspace_log_returns_none(tmp_path):
    runtime = ClaudeRuntime(workspace=tmp_path)

    assert runtime.restore_session() is None
    assert runtime.last_session_id is None


def test_invoke_captures_session_and_resume_reuses_it(monkeypatch):
    rt = ClaudeRuntime(model="ep-x", verbose=False)
    states = [
        PumpState(
            text_parts=["first result"],
            session_id="sess-42",
            done_ok=True,
        ),
        PumpState(
            text_parts=["continued result"],
            session_id="sess-42",
            done_ok=True,
        ),
    ]
    spawns = []

    def fake_spawn(cmd, prompt):
        spawns.append((cmd, prompt))
        return object()

    monkeypatch.setattr(rt, "_spawn", fake_spawn)
    monkeypatch.setattr(rt, "_pump", lambda proc: states.pop(0))

    assert rt.invoke("first prompt") == "first result"
    assert rt.last_session_id == "sess-42"
    assert rt.resume("continue prompt") == "continued result"

    first_cmd, first_prompt = spawns[0]
    resumed_cmd, resumed_prompt = spawns[1]
    assert "--resume" not in first_cmd
    assert first_prompt == "first prompt"
    assert resumed_cmd[resumed_cmd.index("--resume") + 1] == "sess-42"
    assert resumed_prompt == "continue prompt"


def test_default_permissions_allow_only_the_named_profile_subagent():
    allowed = ClaudeRuntime().allowed_tools
    assert "Agent(kernel-profile-analyzer)" in allowed
    assert "Agent(kernel-knowledge-profile-analyzer)" in allowed
    assert "Agent(*)" not in allowed


def test_command_selects_native_agent_only_for_new_session():
    temp, workspace = _native_workspace("kernel-analyzer")
    try:
        rt = ClaudeRuntime(workspace=workspace)
        cmd = rt._command("hi", None, agent="kernel-analyzer")
        assert "--agent" in cmd and "kernel-analyzer" in cmd

        # Claude Code persists the selected agent in resumed sessions.
        cmd_r = rt._command("hi", "sess-42", agent="kernel-analyzer")
        assert "--resume" in cmd_r and "sess-42" in cmd_r
        assert "--agent" not in cmd_r
    finally:
        temp.cleanup()


def test_command_loads_mcp_config_for_agent_tools():
    temp, workspace = _native_workspace(
        "kernel-coder", "Read, mcp__kernelgen__eval_round"
    )
    try:
        (workspace / ".mcp.json").write_text('{"mcpServers":{}}')
        rt = ClaudeRuntime(workspace=workspace)
        cmd = rt._command(
            "hi", None, agent="kernel-coder"
        )
        assert "--mcp-config" in cmd
        assert str(workspace / ".mcp.json") in cmd
        assert ClaudeRuntime(workspace=workspace)._env().get("MCP_TIMEOUT") is None
        assert rt._env()["MCP_TIMEOUT"] == "30000"
    finally:
        temp.cleanup()


def test_command_uses_absolute_mcp_config_for_relative_workspace(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "relative-workspace"
    agents = workspace / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "kernel-coder.md").write_text(
        "---\nname: kernel-coder\ndescription: test\n"
        "tools: mcp__kernelgen__eval_round\nmodel: inherit\n---\nrole"
    )
    materialize_mcp_configuration(canonical_mcp_configuration_path(), workspace)
    (workspace / ".mcp.json").write_text('{"mcpServers":{}}')
    monkeypatch.chdir(tmp_path)

    cmd = ClaudeRuntime(workspace=Path("relative-workspace"))._command(
        "hi", None, agent="kernel-coder"
    )

    value = Path(cmd[cmd.index("--mcp-config") + 1])
    assert value == (workspace / ".mcp.json").resolve()
    assert value.is_absolute()


def test_command_fails_when_native_agent_is_missing():
    with tempfile.TemporaryDirectory() as directory:
        try:
            ClaudeRuntime(workspace=directory)._command(
                "hi", None, agent="kernel-analyzer"
            )
        except RuntimeError as exc:
            assert "expected" in str(exc)
        else:
            raise AssertionError("missing native definition should fail early")


def test_command_fails_when_referenced_subagent_is_missing():
    temp, workspace = _native_workspace(
        "kernel-coder", "Read, Agent(kernel-profile-analyzer)"
    )
    try:
        try:
            ClaudeRuntime(workspace=workspace)._command(
                "hi", None, agent="kernel-coder"
            )
        except RuntimeError as exc:
            assert "kernel-profile-analyzer" in str(exc)
        else:
            raise AssertionError("missing referenced subagent should fail early")
    finally:
        temp.cleanup()


def test_command_inherit_model_omits_flag():
    rt = ClaudeRuntime(model="inherit")
    assert "--model" not in rt._command("hi", None)  # inherit -> no --model


# --- stream-json parsing (_parse_line) ------------------------------------

def test_parse_session_and_text():
    rt = ClaudeRuntime()
    s = PumpState()
    rt._parse_line(_ev(type="system", subtype="init", session_id="abc-123"), s)
    rt._parse_line(_ev(type="assistant",
                       message={"content": [{"type": "text", "text": "Hello "}]}), s)
    rt._parse_line(_ev(type="assistant",
                       message={"content": [{"type": "text", "text": "world"}]}), s)
    assert s.session_id == "abc-123"
    assert s.text == "Hello world"
    assert s.done_ok is False


def test_parse_result_success():
    rt = ClaudeRuntime()
    s = PumpState()
    rt._parse_line(_ev(type="assistant",
                       message={"content": [{"type": "text", "text": "done"}]}), s)
    rt._parse_line(_ev(type="result", subtype="success", is_error=False), s)
    assert s.done_ok is True
    assert s.resumable is False
    assert s.text == "done"


def test_parse_result_error_is_resumable():
    rt = ClaudeRuntime()
    s = PumpState()
    rt._parse_line(_ev(type="result", subtype="error_max_turns", is_error=True,
                       session_id="s-9"), s)
    assert s.done_ok is False
    assert s.resumable is True
    assert s.session_id == "s-9"


def test_parse_tool_use_is_logged_but_not_returned_as_agent_text():
    rt = ClaudeRuntime()
    s = PumpState()
    progress = rt._parse_line(
        _ev(
            type="assistant",
            message={
                "content": [{
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Bash",
                    "input": {"command": "printf 'full payload'"},
                }]
            },
        ),
        s,
    )
    assert s.text == ""
    assert "Bash" in progress
    assert "printf 'full payload'" in progress


def test_parse_stream_thinking_is_retained_and_logged_in_full():
    rt = ClaudeRuntime()
    s = PumpState()
    started = rt._parse_line(
        _ev(
            type="stream_event",
            event={
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        s,
    )
    rt._parse_line(
        _ev(
            type="stream_event",
            event={
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "first half "},
            },
        ),
        s,
    )
    rt._parse_line(
        _ev(
            type="stream_event",
            event={
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "second half"},
            },
        ),
        s,
    )
    finished = rt._parse_line(
        _ev(
            type="stream_event",
            event={"type": "content_block_stop", "index": 0},
        ),
        s,
    )
    assert "thinking..." in started
    assert s.thinking == "first half second half"
    assert "first half second half" in finished
    assert "thinking done" in finished
    assert s.text == ""


def test_aggregate_thinking_and_usage_are_not_logged_twice():
    rt = ClaudeRuntime()
    s = PumpState()
    thinking = "same complete thinking"
    rt._parse_line(
        _ev(
            type="stream_event",
            event={
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        s,
    )
    rt._parse_line(
        _ev(
            type="stream_event",
            event={
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": thinking},
            },
        ),
        s,
    )
    streamed = rt._parse_line(
        _ev(
            type="stream_event",
            event={"type": "content_block_stop", "index": 0},
        ),
        s,
    )
    aggregate = rt._parse_line(
        _ev(
            type="assistant",
            message={
                "id": "message-1",
                "usage": {"input_tokens": 123},
                "content": [{"type": "thinking", "thinking": thinking}],
            },
        ),
        s,
    )
    tool = rt._parse_line(
        _ev(
            type="assistant",
            message={
                "id": "message-1",
                "usage": {"input_tokens": 123},
                "content": [{
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Read",
                    "input": {"file_path": "main.py"},
                }],
            },
        ),
        s,
    )

    assert thinking in streamed
    assert thinking not in aggregate
    assert "[claude] turn 1" in aggregate
    assert "[claude] turn" not in tool
    assert s.turn_count == 1
    assert s.thinking == thinking


def test_aggregate_thinking_before_stream_stop_is_not_logged_twice():
    rt = ClaudeRuntime()
    s = PumpState()
    thinking = "aggregate arrives immediately before content block stop"
    rt._parse_line(
        _ev(
            type="stream_event",
            event={
                "type": "content_block_start",
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        s,
    )
    rt._parse_line(
        _ev(
            type="stream_event",
            event={
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": thinking},
            },
        ),
        s,
    )
    aggregate = rt._parse_line(
        _ev(
            type="assistant",
            message={
                "id": "message-1",
                "usage": {"input_tokens": 123},
                "content": [{"type": "thinking", "thinking": thinking}],
            },
        ),
        s,
    )
    streamed = rt._parse_line(
        _ev(
            type="stream_event",
            event={"type": "content_block_stop", "index": 0},
        ),
        s,
    )

    assert thinking not in aggregate
    assert thinking in streamed
    assert "[claude] turn 1" in aggregate
    assert s.thinking == thinking


def test_parse_tool_result_is_logged_in_full():
    rt = ClaudeRuntime()
    s = PumpState()
    full_result = "line one\nline two\n" + ("x" * 1000)
    progress = rt._parse_line(
        _ev(
            type="user",
            message={
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": full_result,
                }]
            },
        ),
        s,
    )
    assert full_result in progress
    assert progress.startswith("[claude] 📎")


def test_workspace_keeps_only_human_log():
    import io

    with tempfile.TemporaryDirectory() as directory:
        terminal = io.StringIO()
        rt = ClaudeRuntime(workspace=directory, log_stream=terminal)
        rt._prepare()
        try:
            rt._print("[claude] full human log")
            raw_event = _ev(
                type="assistant",
                message={"content": [{"type": "text", "text": "complete text"}]},
            )
            progress = rt._parse_line(raw_event, PumpState())
            rt._print(progress)
        finally:
            rt._cleanup()

        log_dir = Path(directory) / ".kernelgen"
        human = (log_dir / "claude-runtime.log").read_text()
        assert "full human log" in human
        assert "complete text" in human
        assert not (log_dir / "claude-stream.jsonl").exists()
        assert "complete text" in terminal.getvalue()


def test_workspace_log_can_disable_shared_console_mirroring():
    import io

    with tempfile.TemporaryDirectory() as directory:
        terminal = io.StringIO()
        rt = ClaudeRuntime(
            workspace=directory,
            log_stream=terminal,
            mirror_to_console=False,
        )
        rt._prepare()
        try:
            rt._print("[claude] agent-local detail")
        finally:
            rt._cleanup()

        human = (
            Path(directory) / ".kernelgen" / "claude-runtime.log"
        ).read_text()
        assert "agent-local detail" in human
        assert terminal.getvalue() == ""


# --- pump over a fake stdout (skeleton accumulation) ----------------------

class _FakeProc:
    """Minimal Popen-like: yields prepared stdout lines then EOF, poll()->0."""
    def __init__(self, lines):
        self._lines = list(lines)

    class _Out:
        def __init__(self, lines): self._lines = lines
        def readline(self):
            return self._lines.pop(0) if self._lines else ""
    def __init__(self, lines):
        self.stdout = _FakeProc._Out(list(lines))
        self._done = False
    def poll(self):
        return 0 if not self.stdout._lines else None
    def kill(self): pass
    def wait(self, timeout=None): return 0


class _DelayedPollProc(_FakeProc):
    """Expose EOF one poll before the process exit status becomes visible."""

    def __init__(self, lines):
        super().__init__(lines)
        self._eof_polls = 0

    def poll(self):
        if self.stdout._lines:
            return None
        self._eof_polls += 1
        return None if self._eof_polls == 1 else 0


def test_pump_accumulates_and_stops_on_eof():
    rt = ClaudeRuntime(idle_timeout=1, timeout=5, verbose=False)  # quiet: test accumulation only
    lines = [
        _ev(type="system", subtype="init", session_id="sid") + "\n",
        _ev(type="assistant", message={"content": [{"type": "text", "text": "hi"}]}) + "\n",
        _ev(type="result", subtype="success", is_error=False) + "\n",
    ]
    state = rt._pump(_FakeProc(lines))
    assert state.text == "hi"
    assert state.session_id == "sid"
    assert state.done_ok is True


def test_pump_rechecks_process_after_eof_poll_race():
    rt = ClaudeRuntime(idle_timeout=1, timeout=2, verbose=False)
    started = time.monotonic()
    state = rt._pump(_DelayedPollProc([
        _ev(type="result", subtype="success", is_error=False) + "\n",
    ]))
    assert state.done_ok is True
    assert time.monotonic() - started < 1


def test_pump_idle_timeout_while_stdout_is_silent():
    rt = ClaudeRuntime(idle_timeout=0.15, timeout=2, verbose=False)
    proc = rt._spawn(
        [sys.executable, "-u", "-c", "import time; time.sleep(5)"],
        "",
    )

    with pytest.raises(IdleTimeout, match="no output"):
        rt._pump(proc)

    assert proc.poll() is not None


def test_pump_hard_timeout_while_stdout_is_silent():
    rt = ClaudeRuntime(idle_timeout=5, timeout=0.15, verbose=False)
    proc = rt._spawn(
        [sys.executable, "-u", "-c", "import time; time.sleep(5)"],
        "",
    )

    with pytest.raises(IdleTimeout, match="hard timeout"):
        rt._pump(proc)

    assert proc.poll() is not None


def test_invoke_interrupt_does_not_orphan_cli_process():
    class _InterruptingRuntime(CLIRuntime):
        def _command(self, prompt, session_id, *, agent=None):
            return [
                sys.executable,
                "-u",
                "-c",
                "import time; print('ready', flush=True); time.sleep(30)",
            ]

        def _spawn(self, cmd, prompt):
            self.spawned = super()._spawn(cmd, prompt)
            return self.spawned

        def _parse_line(self, line, state):
            raise KeyboardInterrupt

    rt = _InterruptingRuntime(idle_timeout=5, timeout=5, verbose=False)
    with pytest.raises(KeyboardInterrupt):
        rt.invoke("")

    assert rt.spawned.poll() is not None


def test_stderr_activity_prevents_false_idle_timeout():
    result = _ev(type="result", subtype="success", is_error=False)
    script = (
        "import sys, time\n"
        "for _ in range(6):\n"
        "    sys.stderr.write('active\\n')\n"
        "    sys.stderr.flush()\n"
        "    time.sleep(0.1)\n"
        f"print({result!r}, flush=True)\n"
    )
    rt = ClaudeRuntime(idle_timeout=0.3, timeout=2, verbose=False)
    proc = rt._spawn([sys.executable, "-u", "-c", script], "")

    state = rt._pump(proc)

    assert state.done_ok is True
    assert len(rt._stderr_lines) == 6


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").exists(),
    reason="process-group assertion requires Linux /proc",
)
def test_timeout_kills_cli_descendants():
    with tempfile.TemporaryDirectory() as directory:
        child_pid_file = Path(directory) / "child.pid"
        child_script = "import time; time.sleep(30)"
        parent_script = (
            "import pathlib, subprocess, sys, time\n"
            f"child = subprocess.Popen([sys.executable, '-c', {child_script!r}])\n"
            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid))\n"
            "time.sleep(30)\n"
        )
        rt = ClaudeRuntime(idle_timeout=5, timeout=0.5, verbose=False)
        proc = rt._spawn([sys.executable, "-u", "-c", parent_script], "")

        with pytest.raises(IdleTimeout, match="hard timeout"):
            rt._pump(proc)

        child_pid = int(child_pid_file.read_text())
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stat = Path(f"/proc/{child_pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(0.02)
        else:
            pytest.fail(f"descendant process {child_pid} survived CLI timeout")


def test_parse_returns_progress_strings():
    rt = ClaudeRuntime()
    s = PumpState()
    p1 = rt._parse_line(_ev(type="system", subtype="init", session_id="sid"), s)
    assert p1 and "session=sid" in p1
    p2 = rt._parse_line(_ev(type="assistant",
                            message={"content": [{"type": "text", "text": "Optimizing GELU"}]}), s)
    assert p2 and "[claude] turn 1" in p2 and "[claude] text: Optimizing GELU" in p2
    p3 = rt._parse_line(_ev(type="assistant",
                            message={"content": [{
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "echo complete"},
                            }]}), s)
    assert p3 and "[claude] turn 2" in p3
    assert "[claude] 🔧 Bash: $ echo complete" in p3
    assert '"command": "echo complete"' in p3
    p4 = rt._parse_line(_ev(type="result", subtype="success", is_error=False), s)
    assert p4 and "[claude] result (success)" in p4
    assert "────────────────" in p4 and "[claude] done" in p4


def test_pump_verbose_prints(capsys=None):
    import io
    buf = io.StringIO()
    rt = ClaudeRuntime(idle_timeout=1, timeout=5, verbose=True, log_stream=buf)
    lines = [
        _ev(type="system", subtype="init", session_id="sid") + "\n",
        _ev(type="assistant", message={"content": [{"type": "text", "text": "hi"}]}) + "\n",
        _ev(type="result", subtype="success", is_error=False) + "\n",
    ]
    rt._pump(_FakeProc(lines))
    out = buf.getvalue()
    assert "session=sid" in out and "hi" in out and "result" in out


def test_pump_verbose_off_is_quiet():
    import io
    buf = io.StringIO()
    rt = ClaudeRuntime(idle_timeout=1, timeout=5, verbose=False, log_stream=buf)
    rt._pump(_FakeProc([
        _ev(type="assistant", message={"content": [{"type": "text", "text": "hi"}]}) + "\n",
        _ev(type="result", subtype="success", is_error=False) + "\n",
    ]))
    assert buf.getvalue() == ""   # verbose off -> no progress printed


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)

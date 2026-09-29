"""ClaudeRuntime: CLIRuntime for `claude -p --output-format stream-json`.

Fills the scaffold hooks for command construction, per-process environment, and
stream-json parsing. Custom endpoints are passed through the child environment;
campaign entrypoints may materialize a shared, model-only Claude config, but the
runtime never writes per-workspace ``.claude/settings.local.json`` files.

stream-json events we care about:
- {"type":"system","subtype":"init","session_id":...}   -> session_id
- {"type":"assistant","message":{"content":[{"type":"text","text":...}]}} -> text
- {"type":"result","subtype":"success"|..., "is_error":bool} -> done_ok / resumable
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import yaml

from kernelgen.framework.mcp_config import (
    MCP_CONFIGURATION_PATH,
    load_kernelgen_mcp_configuration,
)
from kernelgen.framework.runtime.base import CLIRuntime, PumpState
from kernelgen.mcp_server.contract import MCP_CLAUDE_TOOL_NAMES


_SESSION_LOG_PATTERN = re.compile(
    r"^\[claude\] session=(?P<session_id>"
    r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}) "
    r"tools=\d+ skills=\d+\s*$"
)

CLAUDE_RUNTIME_CONFIG_DIRECTORY = Path(".kernelgen") / "claude-config"


def materialize_claude_runtime_config(
    campaign_workspace: os.PathLike[str] | str,
    model: str | None,
) -> Path | None:
    """Write the campaign-scoped model inherited by native Claude subagents.

    Claude's top-level ``--model`` flag does not propagate to native subagents
    declared with ``model: inherit``. ``CLAUDE_CONFIG_DIR`` does propagate, so
    one model-only settings file is shared by every Claude process in the
    campaign. Provider endpoints and credentials intentionally remain in the
    child-process environment and are never persisted here.
    """
    if not model or model == "inherit":
        return None

    config_dir = (
        Path(campaign_workspace) / CLAUDE_RUNTIME_CONFIG_DIRECTORY
    ).resolve()
    config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(config_dir, 0o700)
    settings_path = config_dir / "settings.json"
    serialized = json.dumps({"model": model}, indent=2) + "\n"

    try:
        if settings_path.read_text(encoding="utf-8") == serialized:
            os.chmod(settings_path, 0o600)
            return config_dir
    except FileNotFoundError:
        pass

    fd, temporary_name = tempfile.mkstemp(
        prefix=".settings.",
        suffix=".tmp",
        dir=config_dir,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, settings_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return config_dir


class ClaudeRuntime(CLIRuntime):
    """Claude Code CLI runtime."""

    supports_native_agents = True

    def __init__(self, *, base_url: Optional[str] = None, auth_token: Optional[str] = None,
                 claude_config_dir: os.PathLike[str] | str | None = None,
                 allowed_tools: str = "Bash,Read,Write,Edit,Glob,Grep,Skill,Agent(kernel-profile-analyzer),Agent(kernel-knowledge-profile-analyzer),mcp__kernelgen__*",
                 permission_mode: str = "acceptEdits",
                 mirror_to_console: bool = True, **kw):
        super().__init__(**kw)
        self.base_url = base_url
        self.auth_token = auth_token
        self.claude_config_dir = (
            Path(claude_config_dir).resolve()
            if claude_config_dir is not None
            else None
        )
        self.allowed_tools = allowed_tools
        self.permission_mode = permission_mode
        self.mirror_to_console = mirror_to_console
        self._mcp_startup_timeout_ms: int | None = None

    def restore_session(self) -> Optional[str]:
        """Restore this workspace's latest Claude conversation from its run log.

        The log is already written synchronously when Claude emits its ``init``
        event, so it survives a later KernelGen process interruption. Matching the
        exact init-line shape avoids treating prompt or tool text as session state.
        """
        if self.last_session_id:
            return self.last_session_id
        if not self.workspace:
            return None

        log_path = self.workspace / ".kernelgen" / "claude-runtime.log"
        session_id = None
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    match = _SESSION_LOG_PATTERN.match(line)
                    if match:
                        session_id = match.group("session_id")
        except OSError:
            return None

        self.last_session_id = session_id
        return session_id

    # -- hooks -----------------------------------------------------------

    def _command(
        self,
        prompt: str,
        session_id: Optional[str],
        *,
        agent: Optional[str] = None,
    ) -> List[str]:
        self._active_agent = agent
        self._mcp_startup_timeout_ms = None
        model = self.model if self.model and self.model != "inherit" else None
        cmd = [
            "claude", "-p",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--permission-mode", self.permission_mode,
            "--allowedTools", self.allowed_tools,
        ]
        if model:
            cmd += ["--model", model]
        needs_mcp = False
        if agent:
            needs_mcp = self._validate_native_agent(agent)
        if needs_mcp:
            neutral_mcp_config = self._neutral_mcp_config_path()
            if not neutral_mcp_config.is_file():
                raise RuntimeError(
                    f"Claude agent {agent!r} requires KernelGen MCP tools, but "
                    f"{neutral_mcp_config} is missing"
                )
            server = load_kernelgen_mcp_configuration(neutral_mcp_config)
            self._mcp_startup_timeout_ms = int(
                server.startup_timeout_seconds * 1000
            )
            mcp_config = self._mcp_config_path()
            if not mcp_config.is_file():
                raise RuntimeError(
                    f"Claude agent {agent!r} requires KernelGen MCP tools, but "
                    f"{mcp_config} is missing"
                )
            cmd += ["--mcp-config", str(mcp_config)]
        # Claude Code persists the selected agent in a resumed session, so only
        # select it when creating the session.
        if agent and not session_id:
            cmd += ["--agent", agent]
        if session_id:
            cmd += ["--resume", session_id]
        return cmd

    def _validate_native_agent(self, agent: str) -> bool:
        """Fail before spawning Claude and report whether the agent needs MCP."""
        if not self.workspace:
            raise RuntimeError(f"Claude native agent {agent!r} requires a workspace")
        visiting: set[str] = set()

        def validate(name: str) -> bool:
            if name in visiting:
                raise RuntimeError(f"cyclic Claude subagent reference involving {name!r}")
            visiting.add(name)
            path = self.workspace / ".claude" / "agents" / f"{name}.md"
            if not path.is_file():
                raise RuntimeError(
                    f"Claude agent {name!r} is unavailable in workspace "
                    f"{self.workspace}; expected {path}"
                )
            text = path.read_text(encoding="utf-8")
            if not text.startswith("---"):
                raise RuntimeError(f"Claude agent definition has no YAML frontmatter: {path}")
            try:
                _, frontmatter, _ = text.split("---", 2)
                metadata = yaml.safe_load(frontmatter) or {}
            except (ValueError, yaml.YAMLError) as exc:
                raise RuntimeError(f"invalid Claude agent frontmatter in {path}: {exc}") from exc
            if not isinstance(metadata, dict):
                raise RuntimeError(f"Claude agent frontmatter must be a mapping: {path}")
            if metadata.get("name") != name:
                raise RuntimeError(
                    f"Claude agent filename/name mismatch: expected {name!r}, "
                    f"got {metadata.get('name')!r} in {path}"
                )
            if not str(metadata.get("description", "")).strip():
                raise RuntimeError(f"Claude agent description is required: {path}")
            if not str(metadata.get("model", "")).strip():
                raise RuntimeError(f"Claude agent model is required: {path}")
            if "tools" not in metadata:
                raise RuntimeError(f"Claude agent tools must be explicit (use [] for none): {path}")
            tools = metadata.get("tools", [])
            if isinstance(tools, str):
                tools = [item.strip() for item in tools.split(",") if item.strip()]
            if not isinstance(tools, list):
                raise RuntimeError(f"Claude agent tools must be a string or list: {path}")
            mcp_tools = {str(tool) for tool in tools if str(tool).startswith("mcp__kernelgen__")}
            unknown = mcp_tools - MCP_CLAUDE_TOOL_NAMES
            if unknown:
                raise RuntimeError(f"unknown KernelGen MCP tools in {path}: {sorted(unknown)}")
            needs_mcp = bool(mcp_tools)
            for tool in tools:
                match = re.fullmatch(r"Agent\(([^)]+)\)", str(tool))
                if match:
                    needs_mcp = validate(match.group(1)) or needs_mcp
            visiting.remove(name)
            return needs_mcp

        return validate(agent)

    def _mcp_config_path(self) -> Path:
        # The CLI process runs with ``cwd=self.workspace``.  Passing a relative
        # workspace-prefixed path here would therefore resolve the workspace
        # twice (for example ``runs/op/runs/op/.mcp.json``) and Claude would
        # exit before starting the agent.  Keep the provider boundary explicit.
        return (
            (self.workspace / ".mcp.json").resolve()
            if self.workspace
            else Path(".mcp.json").resolve()
        )

    def _neutral_mcp_config_path(self) -> Path:
        return (
            (self.workspace / MCP_CONFIGURATION_PATH).resolve()
            if self.workspace
            else Path(MCP_CONFIGURATION_PATH).resolve()
        )

    @staticmethod
    def _format_tool_input_summary(tool_name: str, tool_input: dict) -> str:
        """Match the legacy generator's compact, human-readable tool headings."""
        if not isinstance(tool_input, dict):
            return str(tool_input)
        if tool_name == "Bash":
            command = str(tool_input.get("command", ""))
            description = str(tool_input.get("description", ""))
            label = description or command
            return f"$ {label}"
        if tool_name == "Read":
            return str(tool_input.get("file_path", "?"))
        if tool_name == "Write":
            file_path = tool_input.get("file_path", "?")
            size = len(str(tool_input.get("content", "")))
            return f"{file_path} ({size} chars)"
        if tool_name == "Edit":
            file_path = tool_input.get("file_path", "?")
            old = str(tool_input.get("old_string", ""))
            return f"{file_path}  old={old!r}"
        if tool_name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "?")
            path = tool_input.get("path", ".")
            return f"{pattern}  in {path}"
        if tool_name == "Skill":
            return str(tool_input.get("skill", "?"))
        return json.dumps(tool_input, ensure_ascii=False)

    def _parse_line(self, line: str, state: PumpState):
        line = line.strip()
        if not line:
            return None
        event = json.loads(line)   # may raise -> caught by _pump
        etype = event.get("type", "")

        if etype == "stream_event":
            inner = event.get("event", {})
            inner_type = inner.get("type", "")
            if inner_type == "content_block_start":
                block = inner.get("content_block", {})
                state.active_content_block = str(block.get("type", ""))
                if state.active_content_block == "thinking":
                    state.current_thinking_parts = []
                    state.thinking_started_at = time.monotonic()
                    return "[claude] 💭 thinking..."
            elif inner_type == "content_block_delta":
                delta = inner.get("delta", {})
                if delta.get("type") == "thinking_delta":
                    thinking = str(delta.get("thinking", ""))
                    state.thinking_parts.append(thinking)
                    state.current_thinking_parts.append(thinking)
            elif inner_type == "content_block_stop":
                if state.active_content_block == "thinking":
                    thinking = "".join(state.current_thinking_parts)
                    already_logged = thinking == state.last_logged_thinking
                    elapsed = max(
                        0,
                        int(time.monotonic() - state.thinking_started_at),
                    )
                    state.current_thinking_parts = []
                    state.active_content_block = ""
                    if not thinking or already_logged:
                        return None
                    state.last_logged_thinking = thinking
                    return (
                        f"[claude] 💭 thinking done "
                        f"({elapsed}s, {len(thinking)} chars)\n"
                        "[claude] 💭 content:\n"
                        f"{thinking}\n"
                        "[claude] 💭 end"
                    )
                state.active_content_block = ""
            return None

        if etype == "system":
            sid = event.get("session_id")
            if sid:
                state.session_id = sid
            if event.get("subtype") == "init":
                tools = event.get("tools", [])
                skills = event.get("skills", [])
                return (
                    f"[claude] session={sid} "
                    f"tools={len(tools)} skills={len(skills)}"
                )

        elif etype == "assistant":
            message = event.get("message", {})
            message_id = str(message.get("id", ""))
            first_message_event = (
                not message_id
                or message_id not in state.seen_assistant_message_ids
            )
            progress = []
            if first_message_event:
                if message_id:
                    state.seen_assistant_message_ids.add(message_id)
                state.turn_count += 1
                usage = message.get("usage") or {}
                progress.append(
                    f"[claude] turn {state.turn_count}  "
                    f"in={usage.get('input_tokens', 0)} "
                    f"out={usage.get('output_tokens', 0)} "
                    f"cache={usage.get('cache_read_input_tokens', 0)}"
                )
            for part in message.get("content", []):
                ptype = part.get("type")
                if ptype == "text":
                    txt = part.get("text", "")
                    state.text_parts.append(txt)
                    if txt:
                        progress.append(f"[claude] text: {txt}")
                elif ptype == "thinking":
                    thinking = str(part.get("thinking", ""))
                    streamed_thinking = "".join(state.current_thinking_parts)
                    if (
                        thinking
                        and thinking != state.last_logged_thinking
                        and thinking != streamed_thinking
                    ):
                        state.thinking_parts.append(thinking)
                        state.last_logged_thinking = thinking
                        progress.append(
                            f"[claude] 💭 thinking done "
                            f"(0s, {len(thinking)} chars)\n"
                            f"[claude] 💭 content:\n{thinking}\n"
                            "[claude] 💭 end"
                        )
                elif ptype == "tool_use":
                    name = part.get("name", "?")
                    tool_id = part.get("id", "")
                    raw_input = part.get("input", {})
                    summary = self._format_tool_input_summary(name, raw_input)
                    tool_input = json.dumps(
                        raw_input,
                        ensure_ascii=False,
                        indent=2,
                    )
                    progress.append(
                        f"[claude] 🔧 {name}: {summary}\n"
                        f"[claude] tool input id={tool_id}:\n{tool_input}"
                    )
            return "\n".join(progress) if progress else None

        elif etype == "user":
            progress = []
            content = event.get("message", {}).get("content", [])
            if isinstance(content, str):
                if content:
                    progress.append(f"[claude] user:\n{content}")
            else:
                for part in content:
                    if not isinstance(part, dict):
                        progress.append(f"[claude] user:\n{part}")
                        continue
                    if part.get("type") != "tool_result":
                        continue
                    result = part.get("content", "")
                    if isinstance(result, str):
                        rendered = result
                    else:
                        rendered = json.dumps(result, ensure_ascii=False, indent=2)
                    progress.append(
                        f"[claude] 📎 {rendered}\n"
                        "[claude] tool result metadata: "
                        f"id={part.get('tool_use_id', '')} "
                        f"is_error={bool(part.get('is_error', False))}"
                    )
            return "\n".join(progress) if progress else None

        elif etype == "result":
            # terminal event: success or error
            if event.get("subtype") == "success" and not event.get("is_error"):
                state.done_ok = True
            else:
                # error results can be resumable (transient API/stream errors)
                state.resumable = True
            sid = event.get("session_id")
            if sid:
                state.session_id = sid
            result_text = str(event.get("result", ""))
            had_assistant_text = bool(state.text_parts)
            if result_text and not had_assistant_text:
                state.text_parts.append(result_text)
            duration_ms = event.get("duration_ms", 0) or 0
            total_cost = event.get("total_cost_usd", 0) or 0
            try:
                duration_seconds = float(duration_ms) / 1000
            except (TypeError, ValueError):
                duration_seconds = 0.0
            try:
                cost_value = float(total_cost)
            except (TypeError, ValueError):
                cost_value = 0.0
            elapsed = max(
                0,
                int(time.monotonic() - getattr(
                    self, "_invocation_started_at", time.monotonic()
                )),
            )
            progress = (
                f"[claude] result ({event.get('subtype', '?')})  "
                f"turns={event.get('num_turns', state.turn_count)}  "
                f"cost=${cost_value:.4f}  "
                f"duration={duration_seconds:.1f}s"
            )
            if result_text and not had_assistant_text:
                progress += f"\n[claude] text: {result_text}"
            progress += (
                "\n" + "─" * 60
                + f"\n[claude] done  ({len(state.text)} chars, {elapsed}s)"
                + "\n" + "━" * 60
            )
            return progress
        return None

    # -- child process environment ---------------------------------------

    def _env(self) -> dict:
        env = super()._env()
        # Profile analysis is a state-machine dependency, not independent work.
        # Claude Code >=2.1.198 otherwise defaults subagents to background tasks,
        # which lets the parent continue before record_profile_analysis finishes.
        env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
        if self.claude_config_dir is not None:
            env["CLAUDE_CONFIG_DIR"] = str(self.claude_config_dir)
        if self._mcp_startup_timeout_ms is not None:
            env["MCP_TIMEOUT"] = str(self._mcp_startup_timeout_ms)
        # MCP/legacy CLI tools execute from isolated workspaces. Ensure the
        # KernelGen package remains importable without relying on installation.
        package_parent = str(Path(__file__).absolute().parents[3])
        current = env.get("PYTHONPATH", "")
        entries = [entry for entry in current.split(os.pathsep) if entry]
        if package_parent not in entries:
            env["PYTHONPATH"] = os.pathsep.join([package_parent, *entries])
        if self.workspace:
            workspace = str(self.workspace.resolve())
            env["KERNELGEN_WORKSPACE"] = workspace
            # Claude Code uses this value when presenting the project root to
            # the model and resolving absolute Read/Write/Edit paths.  Bind it
            # to this isolated workspace so a provider-side virtual default
            # such as /home/user/kernelgen_workspace cannot divert candidate
            # edits away from the file read by the MCP server.
            env["CLAUDE_PROJECT_DIR"] = workspace
        # Explicit runtime values override the shell environment. Claude Code
        # must authenticate to the project provider with ANTHROPIC_API_KEY.
        # Accept the legacy KernelGen variable at this boundary, but never pass
        # both authentication variables to the child: zyapi interprets
        # ANTHROPIC_AUTH_TOKEN as a different authentication mode.
        if self.base_url:
            env["ANTHROPIC_BASE_URL"] = self.base_url
        legacy_auth_token = env.pop("ANTHROPIC_AUTH_TOKEN", None)
        if self.auth_token:
            env["ANTHROPIC_API_KEY"] = self.auth_token
        elif not env.get("ANTHROPIC_API_KEY") and legacy_auth_token:
            env["ANTHROPIC_API_KEY"] = legacy_auth_token
        return env

    def _prepare(self) -> None:
        self._prompt_logged = False
        self._spawn_attempt = 0
        self._invocation_started_at = time.monotonic()
        self._human_log = None
        if self.workspace:
            log_dir = self.workspace / ".kernelgen"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._human_log = (log_dir / "claude-runtime.log").open(
                "a", encoding="utf-8", buffering=1
            )

    def _spawn(self, cmd: List[str], prompt: str):
        self._spawn_attempt = getattr(self, "_spawn_attempt", 0) + 1
        session = "new"
        if "--resume" in cmd:
            session = cmd[cmd.index("--resume") + 1]
        if not getattr(self, "_prompt_logged", False):
            self._print("\n" + "━" * 60)
            self._print(
                f"[claude] model={self.model or 'inherit'}  "
                f"session={session}  "
                f"attempt={self._spawn_attempt}/{self.max_resumes + 1}"
            )
            self._print(
                f"[claude] workspace={self.workspace or '.'}  "
                f"agent={getattr(self, '_active_agent', None) or 'default'}"
            )
            self._print(f"[claude] prompt ({len(prompt)} chars):\n{prompt}")
            self._print("─" * 60)
            self._prompt_logged = True
        else:
            self._print("\n" + "━" * 60)
            self._print(
                f"[claude] model={self.model or 'inherit'}  "
                f"session={session}  "
                f"attempt={self._spawn_attempt}/{self.max_resumes + 1}"
            )
            self._print("[claude] resuming existing session")
            self._print("─" * 60)
        return super()._spawn(cmd, prompt)

    def _print(self, msg: str):
        if self.mirror_to_console:
            super()._print(msg)
        human_log = getattr(self, "_human_log", None)
        if human_log is not None and (
            not self.mirror_to_console or human_log is not self._log
        ):
            print(msg, file=human_log, flush=True)

    def _cleanup(self) -> None:
        stream = getattr(self, "_human_log", None)
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
            self._human_log = None

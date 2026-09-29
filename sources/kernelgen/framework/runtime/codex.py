"""CodexRuntime: CLIRuntime for ``codex exec --json``.

Codex emits one JSON object per stdout line.  This runtime translates those
events into the provider-neutral :class:`PumpState`, preserves the Codex thread
id for durable resume, and configures the workspace-local KernelGen MCP server
without writing into the user's ``CODEX_HOME``.

Agent roles remain single-sourced in ``.kernelgen/agents/*.md``. BaseAgent
inlines those roles for the primary turn, while project-scoped Codex custom
agents are generated under each workspace's ``.codex/agents`` directory.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, List, Optional

from kernelgen.framework.agent_roles import load_agent_role, render_inline_role
from kernelgen.framework.mcp_config import (
    MCP_CONFIGURATION_PATH,
    load_kernelgen_mcp_configuration,
)
from kernelgen.framework.runtime.base import CLIRuntime, PumpState


_SESSION_LOG_PATTERN = re.compile(
    r"^\[codex\] session=(?P<session_id>"
    r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12})\s*$"
)


def _toml_value(value: Any) -> str:
    """Render the scalar/list values accepted by ``codex -c key=value``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (str, list)):
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"unsupported Codex config value: {type(value).__name__}")


class CodexRuntime(CLIRuntime):
    """OpenAI Codex CLI runtime.

    Codex has project custom subagents, but ``codex exec`` does not select one
    as the primary turn.  KernelGen therefore treats ordinary roles as inline
    roles while still supporting the explicit named-role invocation required by
    profile finalization.
    """

    supports_native_agents = False
    supports_agent_roles = True

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        sandbox_mode: str = "workspace-write",
        approval_policy: str = "never",
        ignore_user_config: bool = True,
        mirror_to_console: bool = True,
        **kw,
    ):
        super().__init__(**kw)
        if sandbox_mode not in {"read-only", "workspace-write", "danger-full-access"}:
            raise ValueError(f"unsupported Codex sandbox mode: {sandbox_mode!r}")
        if approval_policy not in {"on-request", "never"}:
            raise ValueError(f"unsupported Codex approval policy: {approval_policy!r}")
        self.base_url = base_url
        self.api_key = api_key
        self.sandbox_mode = sandbox_mode
        self.approval_policy = approval_policy
        self.ignore_user_config = ignore_user_config
        self.mirror_to_console = mirror_to_console
        self._mcp_env: dict[str, str] = {}

    def restore_session(self) -> Optional[str]:
        """Restore the latest Codex thread recorded for this workspace."""
        if self.last_session_id:
            return self.last_session_id
        if not self.workspace:
            return None

        log_path = self.workspace / ".kernelgen" / "codex-runtime.log"
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

    def _command(
        self,
        prompt: str,
        session_id: Optional[str],
        *,
        agent: Optional[str] = None,
    ) -> List[str]:
        self._active_agent = agent
        self._active_agent_role = ""
        self._active_role_definition = None
        if agent:
            role_prompt = self._load_agent_role(agent)
            if not session_id:
                self._active_agent_role = role_prompt

        options = self._common_options()
        model = self.model if self.model and self.model != "inherit" else None
        if model:
            options.extend(["--model", model])

        if session_id:
            return [
                "codex", "exec", "--color", "never", "resume",
                *options, session_id, "-",
            ]
        return ["codex", "exec", "--color", "never", *options, "-"]

    def _common_options(self) -> List[str]:
        options = [
            "--json",
            "--skip-git-repo-check",
        ]
        if self.ignore_user_config:
            options.append("--ignore-user-config")
        options.extend(self._config_option("approval_policy", self.approval_policy))
        options.extend(self._config_option("sandbox_mode", self.sandbox_mode))
        if self.base_url:
            options.extend(self._config_option("openai_base_url", self.base_url))
        options.extend(self._mcp_options())
        return options

    @staticmethod
    def _config_option(key: str, value: Any) -> List[str]:
        return ["--config", f"{key}={_toml_value(value)}"]

    def _mcp_options(self) -> List[str]:
        self._mcp_env = {}
        path = self._mcp_config_path()
        if not path.is_file():
            return []
        server = load_kernelgen_mcp_configuration(path)
        self._mcp_env = server.environment

        pairs: list[tuple[str, Any]] = [
            ("mcp_servers.kernelgen.command", server.executable),
            ("mcp_servers.kernelgen.args", list(server.args)),
            ("mcp_servers.kernelgen.required", True),
            (
                "mcp_servers.kernelgen.startup_timeout_sec",
                server.startup_timeout_seconds,
            ),
            (
                "mcp_servers.kernelgen.tool_timeout_sec",
                server.tool_timeout_seconds,
            ),
            ("mcp_servers.kernelgen.default_tools_approval_mode", "approve"),
        ]
        active_role = getattr(self, "_active_role_definition", None)
        if active_role is not None:
            pairs.append(
                ("mcp_servers.kernelgen.enabled_tools", list(active_role.mcp_tools))
            )
        if self._mcp_env:
            pairs.append(("mcp_servers.kernelgen.env_vars", sorted(self._mcp_env)))
        options: List[str] = []
        for key, value in pairs:
            options.extend(self._config_option(key, value))
        return options

    def _mcp_config_path(self) -> Path:
        return (
            (self.workspace / MCP_CONFIGURATION_PATH).resolve()
            if self.workspace
            else Path(MCP_CONFIGURATION_PATH).resolve()
        )

    def _load_agent_role(self, agent: str) -> str:
        if not self.workspace:
            raise RuntimeError(f"Codex agent role {agent!r} requires a workspace")
        path = self.workspace / ".kernelgen" / "agents" / f"{agent}.md"
        if not path.is_file():
            raise RuntimeError(
                f"Codex agent role {agent!r} is unavailable in workspace "
                f"{self.workspace}; expected {path}"
            )
        role = load_agent_role(path, expected_name=agent)
        self._active_role_definition = role
        if role.mcp_tools and not self._mcp_config_path().is_file():
            raise RuntimeError(
                f"Codex agent role {agent!r} requires KernelGen MCP tools, but "
                f"{self._mcp_config_path()} is missing"
            )
        return render_inline_role(role)

    def _effective_prompt(self, prompt: str) -> str:
        role = getattr(self, "_active_agent_role", "")
        if not role:
            return prompt
        agent = getattr(self, "_active_agent", None) or "unknown"
        return (
            f"--- KERNELGEN AGENT ROLE: {agent} ---\n{role}\n\n"
            f"--- TASK ---\n{prompt}"
        )

    def _parse_line(self, line: str, state: PumpState) -> Optional[str]:
        line = line.strip()
        if not line:
            return None
        event = json.loads(line)
        event_type = str(event.get("type", ""))

        if event_type == "thread.started":
            session_id = event.get("thread_id")
            if session_id:
                state.session_id = str(session_id)
            return f"[codex] session={session_id}"

        if event_type == "turn.started":
            state.turn_count += 1
            return f"[codex] turn {state.turn_count} started"

        if event_type.startswith("item."):
            return self._parse_item(event_type, event.get("item") or {}, state)

        if event_type == "turn.completed":
            state.done_ok = True
            usage = event.get("usage") or {}
            elapsed = max(
                0,
                int(time.monotonic() - getattr(
                    self, "_invocation_started_at", time.monotonic()
                )),
            )
            return (
                f"[codex] turn completed  in={usage.get('input_tokens', 0)} "
                f"out={usage.get('output_tokens', 0)} "
                f"cache={usage.get('cached_input_tokens', 0)}\n"
                + "─" * 60
                + f"\n[codex] done  ({len(state.text)} chars, {elapsed}s)"
                + "\n" + "━" * 60
            )

        if event_type in {"turn.failed", "error"}:
            state.resumable = bool(state.session_id)
            error = event.get("error") or event.get("message") or "unknown error"
            if isinstance(error, dict):
                error = error.get("message") or json.dumps(error, ensure_ascii=False)
            return f"[codex] {event_type}: {error}"
        return None

    def _parse_item(
        self,
        event_type: str,
        item: dict[str, Any],
        state: PumpState,
    ) -> Optional[str]:
        item_type = str(item.get("type", "unknown"))
        completed = event_type == "item.completed"

        if item_type == "agent_message":
            if not completed:
                return None
            text = str(item.get("text", ""))
            if text:
                state.text_parts.append(text)
            return f"[codex] text: {text}" if text else None

        if item_type == "reasoning":
            if not completed:
                return "[codex] thinking..." if event_type == "item.started" else None
            text = str(item.get("text", ""))
            if text:
                state.thinking_parts.append(text)
            return (
                f"[codex] thinking done ({len(text)} chars)\n"
                f"[codex] thinking content:\n{text}\n[codex] thinking end"
                if text else None
            )

        if item_type == "command_execution":
            command = item.get("command", "")
            if not completed:
                return f"[codex] command: {command}" if event_type == "item.started" else None
            output = item.get("aggregated_output", "")
            exit_code = item.get("exit_code")
            rendered = f"[codex] command done exit={exit_code}: {command}"
            if output:
                rendered += f"\n[codex] command output:\n{output}"
            return rendered

        if item_type == "mcp_tool_call":
            server = item.get("server", "?")
            tool = item.get("tool", "?")
            if not completed:
                return (
                    f"[codex] MCP {server}.{tool}: "
                    f"{json.dumps(item.get('arguments', {}), ensure_ascii=False)}"
                    if event_type == "item.started" else None
                )
            result = item.get("result")
            error = item.get("error")
            rendered = f"[codex] MCP {server}.{tool} done"
            payload = error if error is not None else result
            if payload is not None:
                rendered += "\n" + (
                    payload if isinstance(payload, str)
                    else json.dumps(payload, ensure_ascii=False, indent=2)
                )
            return rendered

        if completed:
            return (
                f"[codex] {item_type}: "
                f"{json.dumps(item, ensure_ascii=False, indent=2)}"
            )
        return f"[codex] {item_type} started" if event_type == "item.started" else None

    def _env(self) -> dict:
        env = super()._env()
        for name in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CODEX_SESSION_ID",
            "CODEX_THREAD_ID",
            "CODEX_REMOTE_PAYLOAD",
        ):
            env.pop(name, None)
        package_parent = str(Path(__file__).absolute().parents[3])
        current = env.get("PYTHONPATH", "")
        entries = [entry for entry in current.split(os.pathsep) if entry]
        if package_parent not in entries:
            env["PYTHONPATH"] = os.pathsep.join([package_parent, *entries])
        if self.workspace:
            env["KERNELGEN_WORKSPACE"] = str(self.workspace.resolve())
        if self.api_key:
            env["CODEX_API_KEY"] = self.api_key
        env.update(self._mcp_env)
        return env

    def _prepare(self) -> None:
        self._prompt_logged = False
        self._spawn_attempt = 0
        self._invocation_started_at = time.monotonic()
        self._human_log = None
        if self.workspace:
            log_dir = self.workspace / ".kernelgen"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._human_log = (log_dir / "codex-runtime.log").open(
                "a", encoding="utf-8", buffering=1
            )

    def _spawn(self, cmd: List[str], prompt: str):
        prompt = self._effective_prompt(prompt)
        self._spawn_attempt = getattr(self, "_spawn_attempt", 0) + 1
        session = "new"
        if "resume" in cmd:
            index = cmd.index("resume")
            session = next(
                (item for item in cmd[index + 1:] if re.fullmatch(
                    r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}",
                    item,
                )),
                "existing",
            )
        self._print("\n" + "━" * 60)
        self._print(
            f"[codex] model={self.model or 'inherit'}  session={session}  "
            f"attempt={self._spawn_attempt}/{self.max_resumes + 1}"
        )
        self._print(
            f"[codex] workspace={self.workspace or '.'}  "
            f"agent={getattr(self, '_active_agent', None) or 'default'}"
        )
        if not self._prompt_logged:
            self._print(f"[codex] prompt ({len(prompt)} chars):\n{prompt}")
            self._prompt_logged = True
        else:
            self._print("[codex] resuming existing session")
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

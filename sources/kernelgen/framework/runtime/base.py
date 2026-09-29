"""Runtime base: the Runtime contract, FakeRuntime, and the CLIRuntime skeleton.

Layering (see ADR-3 #9):
- Agent (contract layer): builds prompt, parses/validates output. Knows nothing
  about how the prompt is run.
- Runtime (execution layer): actually runs one prompt -> text. Owns the run
  ENVIRONMENT (worktree/cwd, model, env injection, timeout, resume/retry).
- Orchestrator (coordination layer): constructs the Runtime (injecting worktree +
  env), fans out, combines results.

``CLIRuntime`` is the shared skeleton for CLI-spawning scaffolds (Claude Code,
codex, ...). It owns the non-trivial common machinery — spawn with stdin/stderr
threads, idle-timeout pumping (stderr activity keeps a thinking model alive),
resume loop — so a new scaffold only fills two hooks: ``_command`` (argv) and
``_parse_line`` (scaffold output format -> accumulated text/session_id). This
replaces the old ClaudeCodeRuntime + ~800-line ClaudeCodeClient.

FakeRuntime does NOT extend CLIRuntime (it has no subprocess) — it implements the
Runtime contract directly, so host tests need no real CLI.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, Set, runtime_checkable

from kernelgen.framework.run_control import (
    RunCancelled,
    RunControl,
    RunState,
    WorkspaceRunControl,
)


@runtime_checkable
class Runtime(Protocol):
    """Minimal LLM-invocation contract: prompt in, assistant text out.

    Pure transport. The task prompt is self-contained; cwd/worktree is bound at
    construction; crash/idle resume is the Runtime's own internal robustness.
    Runtimes that support native named agents may also receive an agent name.
    """

    def invoke(
        self,
        prompt: str,
        *,
        model: str = "inherit",
        agent: Optional[str] = None,
    ) -> str:
        ...


class FakeRuntime:
    """Scripted runtime for host tests. Pops one reply per invoke; records calls."""

    supports_native_agents = False

    def __init__(self, replies: List[str]):
        self._replies = list(replies)
        self.calls: List[dict] = []

    def invoke(self, prompt, *, model="inherit", agent=None) -> str:
        self.calls.append({"prompt": prompt, "model": model, "agent": agent})
        if not self._replies:
            raise AssertionError("FakeRuntime: no more scripted replies")
        return self._replies.pop(0)


class IdleTimeout(RuntimeError):
    """Raised inside _pump when a scaffold produces no output for idle_timeout."""

    def __init__(self, message: str, session_id: Optional[str] = None):
        super().__init__(message)
        self.session_id = session_id


@dataclass
class PumpState:
    """Accumulator threaded through _parse_line while pumping a subprocess's stdout.
    Subclass ``_parse_line`` fills these from its own output format."""
    text_parts: List[str] = field(default_factory=list)
    thinking_parts: List[str] = field(default_factory=list)
    current_thinking_parts: List[str] = field(default_factory=list)
    active_content_block: str = ""
    last_logged_thinking: str = ""
    thinking_started_at: float = 0.0
    turn_count: int = 0
    seen_assistant_message_ids: Set[str] = field(default_factory=set)
    session_id: Optional[str] = None
    done_ok: bool = False            # scaffold reported a successful result
    resumable: bool = False          # scaffold reported a resumable error

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    @property
    def thinking(self) -> str:
        """All extended-thinking text emitted by the provider for this invocation."""
        return "".join(self.thinking_parts)


class CLIRuntime(Runtime, ABC):
    """Shared execution skeleton for CLI-spawning scaffolds.

    Owns the run environment (worktree/model/timeout/resume/env) and the common
    machinery (spawn, stdin/stderr threads, idle-timeout pump, resume loop).
    Subclasses fill ``_command`` and ``_parse_line``; optionally ``_prepare`` /
    ``_cleanup`` for other per-invocation setup.
    """

    def __init__(
        self,
        *,
        workspace: Optional[os.PathLike | str] = None,
        model: str = "inherit",
        timeout: float = 1200.0,
        idle_timeout: float = 600.0,
        max_resumes: int = 5,
        log_stream=None,
        verbose: bool = True,
        run_control: Optional[RunControl] = None,
    ):
        self.workspace = Path(workspace) if workspace else None
        self.model = model
        self.timeout = timeout
        self.idle_timeout = idle_timeout
        self.max_resumes = max_resumes
        self._log = log_stream  # optional file-like for progress; None -> stdout
        self.verbose = verbose  # when True, print per-line progress from _parse_line
        self._run_control = run_control
        # Provider conversation ID captured from the latest invocation. It is
        # intentionally scoped to this Runtime/workspace so one Coder can resume
        # its own context without crossing parallel agent boundaries.
        self.last_session_id: Optional[str] = None

    # -- Runtime contract: prompt -> text (template method) --------------

    def invoke(
        self,
        prompt: str,
        *,
        model: str = "inherit",
        agent: Optional[str] = None,
    ) -> str:
        """Start a new provider conversation."""
        return self._invoke(
            prompt,
            initial_session_id=None,
            agent=agent,
        )

    def resume(
        self,
        prompt: str,
        *,
        model: str = "inherit",
        agent: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Append one prompt to a persisted provider conversation."""
        selected = session_id or self.last_session_id
        if not selected:
            raise RuntimeError(
                f"{type(self).__name__}: no provider session is available to resume"
            )
        return self._invoke(
            prompt,
            initial_session_id=selected,
            agent=agent,
        )

    def _invoke(
        self,
        prompt: str,
        *,
        initial_session_id: Optional[str],
        agent: Optional[str],
    ) -> str:
        """Run one new or resumed prompt with in-session transport retries."""
        run_control = self._get_run_control()
        if run_control is not None:
            run_control.checkpoint("BEFORE_MODEL_INVOCATION")
            run_control.update_progress(
                state=RunState.RUNNING,
                stage="MODEL_INVOCATION",
                message=(
                    f"Running {type(self).__name__}"
                    + (f" agent {agent}" if agent else "")
                ),
            )
            run_control.record_event(
                "MODEL_INVOCATION_STARTED",
                stage="MODEL_INVOCATION",
                source=f"runtime:{type(self).__name__}",
                data={
                    "agent": agent,
                    "resumed": initial_session_id is not None,
                    "model": self.model or "inherit",
                },
            )
        self.last_session_id = initial_session_id
        prepared = False
        try:
            self._prepare()
            prepared = True
            session_id = initial_session_id
            for resume_i in range(self.max_resumes + 1):
                if resume_i and run_control is not None:
                    # The previous provider process has ended. Cancellation
                    # must prevent a new attempt, never interrupt its output.
                    run_control.checkpoint("BEFORE_MODEL_RETRY")
                cmd = self._command(prompt, session_id, agent=agent)
                proc = self._spawn(cmd, prompt)
                try:
                    state = self._pump(proc)
                except IdleTimeout as e:
                    session_id = e.session_id
                    if session_id:
                        self.last_session_id = session_id
                    if session_id and resume_i < self.max_resumes:
                        self._print(f"[runtime] idle timeout, resuming ({resume_i + 1}/{self.max_resumes})")
                        if run_control is not None:
                            run_control.record_event(
                                "MODEL_INVOCATION_RETRYING",
                                message=str(e),
                                stage="MODEL_INVOCATION",
                                level="WARNING",
                                source=f"runtime:{type(self).__name__}",
                                data={"retry": resume_i + 1},
                            )
                        continue
                    raise
                except BaseException:
                    # _spawn puts the CLI in its own session, so signals sent to
                    # KernelGen no longer reach it automatically. Ensure an
                    # interrupt or parser failure cannot orphan CLI/MCP children.
                    self._kill_and_wait(proc)
                    raise
                if state.session_id:
                    session_id = state.session_id
                    self.last_session_id = state.session_id
                if state.done_ok:
                    if run_control is not None:
                        run_control.record_event(
                            "MODEL_INVOCATION_COMPLETED",
                            stage="MODEL_INVOCATION",
                            source=f"runtime:{type(self).__name__}",
                            data={
                                "agent": agent,
                                "turn_count": state.turn_count,
                            },
                        )
                        run_control.checkpoint("AFTER_MODEL_INVOCATION")
                    return state.text
                if state.resumable and state.session_id and resume_i < self.max_resumes:
                    self._print(f"[runtime] resumable error, resuming ({resume_i + 1}/{self.max_resumes})")
                    if run_control is not None:
                        run_control.record_event(
                            "MODEL_INVOCATION_RETRYING",
                            message="Provider reported a resumable error",
                            stage="MODEL_INVOCATION",
                            level="WARNING",
                            source=f"runtime:{type(self).__name__}",
                            data={"retry": resume_i + 1},
                        )
                    continue
                if run_control is not None:
                    run_control.record_event(
                        "MODEL_INVOCATION_COMPLETED",
                        message="Provider invocation ended without a success result",
                        stage="MODEL_INVOCATION",
                        level="WARNING",
                        source=f"runtime:{type(self).__name__}",
                        data={
                            "agent": agent,
                            "resumable": state.resumable,
                        },
                    )
                    run_control.checkpoint("AFTER_MODEL_INVOCATION")
                return state.text  # not resumable — return whatever text we got
            raise RuntimeError(f"{type(self).__name__}: exhausted {self.max_resumes} resumes")
        except RunCancelled:
            raise
        except BaseException as exc:
            if run_control is not None:
                run_control.record_event(
                    "MODEL_INVOCATION_FAILED",
                    message=f"{type(exc).__name__}: {exc}",
                    stage="MODEL_INVOCATION",
                    level="ERROR",
                    visibility="DEBUG",
                    source=f"runtime:{type(self).__name__}",
                    data={"agent": agent},
                )
            raise
        finally:
            if prepared:
                self._cleanup()

    # -- common machinery (base provides) --------------------------------

    def _env(self) -> dict:
        # Never inherit a parent agent's workspace boundary. Provider runtimes
        # explicitly bind their own workspace after this sanitized base env.
        blocked = {
            "CLAUDECODE",
            "KERNELGEN_WORKSPACE",
            "CLAUDE_PROJECT_DIR",
        }
        return {k: v for k, v in os.environ.items() if k not in blocked}

    def _spawn(self, cmd: List[str], prompt: str) -> subprocess.Popen:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(self.workspace) if self.workspace else None,
            env=self._env(),
            # Timeout cleanup targets the whole CLI/MCP process group.  Without
            # a separate session, killing the group could also kill KernelGen.
            start_new_session=(os.name == "posix"),
        )
        # stderr drained in a thread (prevent pipe deadlock); its activity timestamp
        # keeps a thinking model alive during idle-timeout checks.
        self._stderr_activity = [time.monotonic()]
        self._stderr_lines: List[str] = []
        threading.Thread(target=self._drain_stderr, args=(proc.stderr,), daemon=True).start()
        # prompt via stdin in a thread (avoid deadlock on large prompts)
        threading.Thread(target=self._write_stdin, args=(proc, prompt), daemon=True).start()
        return proc

    def _write_stdin(self, proc, prompt):
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass

    def _drain_stderr(self, pipe):
        try:
            for line in iter(pipe.readline, ""):
                self._stderr_lines.append(line)
                self._stderr_activity[0] = time.monotonic()
        except (ValueError, OSError):
            pass

    @staticmethod
    def _read_stdout(
        pipe,
        output: "queue.Queue[Optional[str]]",
        activity: List[float],
    ) -> None:
        """Read blocking stdout in a daemon thread and notify the pump at EOF."""
        try:
            for line in iter(pipe.readline, ""):
                activity[0] = time.monotonic()
                output.put(line)
        except (ValueError, OSError):
            pass
        finally:
            output.put(None)

    @staticmethod
    def _kill_and_wait(proc: subprocess.Popen) -> None:
        """Kill the CLI and its normal descendants, then reap the direct child."""
        killed_group = False
        pid = getattr(proc, "pid", None)
        if os.name == "posix" and pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
                killed_group = True
            except ProcessLookupError:
                # The direct process may have exited between the timeout check
                # and cleanup.  wait() below still reaps it.
                pass
            except (PermissionError, OSError):
                # A fake Popen or an unusual launcher may not own a process
                # group. Fall back to killing the direct child.
                pass
        if not killed_group:
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # The group received SIGKILL already; avoid masking the timeout
            # that the caller is about to report.
            pass

    def _pump(self, proc: subprocess.Popen) -> PumpState:
        """Read stdout line by line, delegating each line to the subclass's
        _parse_line, with idle-timeout that respects stderr activity."""
        state = PumpState()
        start = time.monotonic()
        stdout_activity = [start]
        stderr_activity = getattr(self, "_stderr_activity", [start])
        stdout_lines: "queue.Queue[Optional[str]]" = queue.Queue()
        threading.Thread(
            target=self._read_stdout,
            args=(proc.stdout, stdout_lines, stdout_activity),
            daemon=True,
        ).start()
        stdout_eof = False

        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= self.timeout:
                self._kill_and_wait(proc)
                raise IdleTimeout(
                    f"hard timeout {int(self.timeout)}s",
                    session_id=state.session_id,
                )

            stdout_idle = now - stdout_activity[0]
            stderr_idle = now - stderr_activity[0]
            if (
                stdout_idle >= self.idle_timeout
                and stderr_idle >= self.idle_timeout
            ):
                self._kill_and_wait(proc)
                raise IdleTimeout(
                    f"no output for {int(stdout_idle)}s "
                    f"(elapsed {int(elapsed)}s)",
                    session_id=state.session_id,
                )

            if stdout_eof:
                # EOF can race process state publication: poll() may briefly
                # remain None after the reader has observed EOF. Keep polling
                # instead of consuming the one EOF marker and waiting forever.
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue

            try:
                line = stdout_lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                stdout_eof = True
                continue

            try:
                progress = self._parse_line(line, state)
            except (json.JSONDecodeError, ValueError):
                continue  # non-JSON / partial line — ignore
            if progress:
                run_control = self._get_run_control()
                if run_control is not None:
                    run_control.record_event(
                        "RUNTIME_LOG",
                        message=progress,
                        stage="MODEL_INVOCATION",
                        visibility="DEBUG",
                        source=f"runtime:{type(self).__name__}",
                    )
                if self.verbose:
                    self._print(progress)
        return state

    def _print(self, msg: str):
        stream = self._log
        print(msg, file=stream, flush=True) if stream else print(msg, flush=True)

    def _get_run_control(self) -> Optional[RunControl]:
        """Lazily bind observation to the nearest enclosing run workspace."""
        if self._run_control is None and self.workspace is not None:
            self._run_control = WorkspaceRunControl(
                self.workspace,
                source=f"runtime:{type(self).__name__}",
            )
        return self._run_control

    # -- scaffold-specific hooks (subclass fills) ------------------------

    @abstractmethod
    def _command(
        self,
        prompt: str,
        session_id: Optional[str],
        *,
        agent: Optional[str] = None,
    ) -> List[str]:
        """Build the argv for one invocation (+ resume flag if session_id)."""
        ...

    @abstractmethod
    def _parse_line(self, line: str, state: PumpState) -> Optional[str]:
        """Parse one stdout line in this scaffold's format; accumulate into state
        (text_parts / session_id / done_ok / resumable). Optionally return a short
        one-line progress string (printed when verbose=True) — e.g. a tool call or
        a text snippet — for human monitoring."""
        ...

    def _prepare(self) -> None:
        """Optional: set up before spawn (e.g. swap settings for a custom endpoint)."""

    def _cleanup(self) -> None:
        """Optional: undo _prepare after invoke."""

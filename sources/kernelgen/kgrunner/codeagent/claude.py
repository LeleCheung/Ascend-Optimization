"""Claude Code backend implementation."""

import json
import logging
import os
import re
import signal
import subprocess
from pathlib import Path

from .base import CodeAgentBackend, CodeAgentConfig, CodeAgentProcess
from ..platform import PlatformInfo, make_device_env

logger = logging.getLogger(__name__)

_API_ERROR_PATTERNS = [
    "API Error",
    "Unexpected EOF",
    "connection reset",
    "ECONNRESET",
]


class ClaudeBackend(CodeAgentBackend):
    def __init__(self, config: CodeAgentConfig, platform: PlatformInfo | None = None):
        self.config = config
        self.platform = platform

    def spawn(
        self,
        prompt: str,
        *,
        cwd: Path,
        device_id: int | None = None,
        log_dir: Path,
        log_prefix: str = "cc",
    ) -> CodeAgentProcess:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = log_dir / f"{log_prefix}.jsonl"

        env = self._build_env(device_id)
        cmd = self._build_cmd(prompt)

        stdout_file = open(stdout_path, "w")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            stdout_file.close()
            raise

        logger.info("Spawned CC (PID=%d, cwd=%s)", proc.pid, cwd)
        return CodeAgentProcess(
            proc=proc,
            stdout_path=stdout_path,
            stderr_path=None,
            _stdout_file=stdout_file,
            _stderr_file=None,
        )

    def resume(
        self,
        session_id: str,
        *,
        cwd: Path,
        device_id: int | None = None,
        log_dir: Path,
        log_prefix: str = "cc_resume",
    ) -> CodeAgentProcess:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = log_dir / f"{log_prefix}.jsonl"
        stderr_path = log_dir / f"{log_prefix}.log"

        env = self._build_env(device_id)
        cmd = self._build_cmd("Continue the task.", resume_session=session_id)

        stdout_file = open(stdout_path, "w")
        stderr_file = open(stderr_path, "w")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        except Exception:
            stdout_file.close()
            stderr_file.close()
            raise

        logger.info("Resumed CC (PID=%d, session=%s)", proc.pid, session_id)
        return CodeAgentProcess(
            proc=proc,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            _stdout_file=stdout_file,
            _stderr_file=stderr_file,
        )

    def kill(self, agent_proc: CodeAgentProcess, grace_period: float = 10.0) -> None:
        proc = agent_proc.proc
        if proc.poll() is not None:
            agent_proc.close()
            return

        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                proc.terminate()
            except OSError:
                pass

        try:
            proc.wait(timeout=grace_period)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process %d did not exit after SIGKILL", proc.pid)

        agent_proc.close()

    def extract_text(self, agent_proc: CodeAgentProcess) -> str:
        """Layer 1: read JSONL, find result event, return result text."""
        try:
            with open(agent_proc.stdout_path, "r", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "result":
                        return event.get("result", "")
        except Exception:
            pass
        return ""

    def detect_resumable_error(self, agent_proc: CodeAgentProcess) -> bool:
        """Check if result contains API stream error patterns."""
        text = self.extract_text(agent_proc)
        if not text:
            return False
        # Check for auth errors (not resumable)
        try:
            with open(agent_proc.stdout_path, "r", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "result" and event.get("api_error_status") is not None:
                        return False
        except Exception:
            pass
        return any(pat in text for pat in _API_ERROR_PATTERNS)

    def extract_session_id(self, agent_proc: CodeAgentProcess) -> str | None:
        """Extract session_id from init event."""
        try:
            with open(agent_proc.stdout_path, "r", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "system" and event.get("subtype") == "init":
                        return event.get("session_id")
        except Exception:
            pass
        return None

    @property
    def supports_resume(self) -> bool:
        return True

    def _build_env(self, device_id: int | None = None) -> dict:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)

        if self.config.base_url:
            env["ANTHROPIC_BASE_URL"] = self.config.base_url
        if self.config.model:
            env["ANTHROPIC_MODEL"] = self.config.model

        token = os.environ.get(self.config.auth_token_env, "")
        if token:
            env["ANTHROPIC_AUTH_TOKEN"] = token

        if self.config.max_output_tokens:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.config.max_output_tokens)

        env["IS_SANDBOX"] = "1"

        if self.platform and device_id is not None:
            env = make_device_env(self.platform, device_id, env)

        return env

    def _build_cmd(self, prompt: str, resume_session: str | None = None) -> list[str]:
        cmd = [self.config.bin]

        if resume_session:
            cmd.extend(["-p", prompt, "--resume", resume_session])
        else:
            cmd.extend(["-p", prompt])

        cmd.extend(["--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose"])

        if self.config.budget:
            cmd.extend(["--max-budget-usd", str(self.config.budget)])

        cmd.extend(self.config.extra_flags)
        return cmd


# Backward-compatible free functions (delegate to ClaudeBackend)
def spawn_agent(prompt, *, cwd, agent_config, platform=None, device_id=None,
                log_dir, log_prefix="cc", extra_env=None):
    backend = ClaudeBackend(agent_config, platform)
    return backend.spawn(prompt, cwd=cwd, device_id=device_id, log_dir=log_dir, log_prefix=log_prefix)


def resume_agent(session_id, *, cwd, agent_config, platform=None, device_id=None,
                 log_dir, log_prefix="cc_resume", resume_prompt="Continue the task."):
    backend = ClaudeBackend(agent_config, platform)
    return backend.resume(session_id, cwd=cwd, device_id=device_id, log_dir=log_dir, log_prefix=log_prefix)


def kill_agent(agent_proc, grace_period=10.0):
    backend = ClaudeBackend(CodeAgentConfig())
    backend.kill(agent_proc, grace_period)

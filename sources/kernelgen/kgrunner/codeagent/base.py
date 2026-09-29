"""Code agent backend — abstract interface and shared types."""

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class CodeAgentConfig:
    bin: str = "claude"
    model: str | None = None
    base_url: str | None = None
    auth_token_env: str = "ANTHROPIC_AUTH_TOKEN"
    max_output_tokens: int | None = None
    budget: float | None = None
    extra_flags: list[str] = field(default_factory=list)


@dataclass
class CodeAgentProcess:
    proc: subprocess.Popen
    stdout_path: Path
    stderr_path: Path | None = None
    _stdout_file: object = field(repr=False, default=None)
    _stderr_file: object = field(repr=False, default=None)

    def poll(self) -> int | None:
        return self.proc.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self.proc.wait(timeout=timeout)

    def close(self) -> None:
        if self._stdout_file and not self._stdout_file.closed:
            self._stdout_file.close()
        if self._stderr_file and not self._stderr_file.closed:
            self._stderr_file.close()

    @property
    def pid(self) -> int:
        return self.proc.pid


class CodeAgentBackend(ABC):
    """Abstract interface for code agent backends (CC, Codex, etc.)."""

    @abstractmethod
    def spawn(
        self,
        prompt: str,
        *,
        cwd: Path,
        device_id: int | None = None,
        log_dir: Path,
        log_prefix: str = "agent",
    ) -> CodeAgentProcess:
        ...

    @abstractmethod
    def resume(
        self,
        session_id: str,
        *,
        cwd: Path,
        device_id: int | None = None,
        log_dir: Path,
        log_prefix: str = "agent_resume",
    ) -> CodeAgentProcess:
        ...

    @abstractmethod
    def kill(self, agent_proc: CodeAgentProcess) -> None:
        ...

    @abstractmethod
    def extract_text(self, agent_proc: CodeAgentProcess) -> str:
        """Layer 1: extract raw text output from backend-specific format."""
        ...

    @abstractmethod
    def detect_resumable_error(self, agent_proc: CodeAgentProcess) -> bool:
        """Check if the run failed due to a resumable error."""
        ...

    @abstractmethod
    def extract_session_id(self, agent_proc: CodeAgentProcess) -> str | None:
        """Extract session identifier for resume."""
        ...

    @property
    def supports_resume(self) -> bool:
        """Whether this backend supports session resume."""
        return True


# Default Layer 2 output parser: extract JSON from text
def default_output_parser(text: str) -> dict | None:
    """Extract a JSON object from text (code block or raw scan)."""
    import json
    import re

    if not text:
        return None

    # Try code block first
    code_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if code_match:
        block_text = code_match.group(1).strip()
        idx = block_text.find("{")
        if idx >= 0:
            obj_str = _extract_json_object(block_text, idx)
            if obj_str:
                try:
                    return json.loads(obj_str)
                except json.JSONDecodeError:
                    pass

    # Fallback: scan for any JSON object
    idx = text.find("{")
    while idx >= 0:
        obj_str = _extract_json_object(text, idx)
        if obj_str:
            try:
                return json.loads(obj_str)
            except json.JSONDecodeError:
                pass
            idx = text.find("{", idx + 1)
        else:
            idx = text.find("{", idx + 1)

    return None


def _extract_json_object(text: str, start: int) -> str | None:
    """Extract a complete JSON object using brace counting."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

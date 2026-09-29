"""Construction helpers for supported command-line runtimes."""

from __future__ import annotations

import os
from typing import Any

from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.framework.runtime.codex import CodexRuntime


CLI_RUNTIME_NAMES = ("claude", "codex")


def resolve_cli_runtime_options(
    runtime: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    auth_token: str | None = None,
    claude_default_model: str = "inherit",
) -> dict[str, str | None]:
    """Resolve provider-specific environment defaults at the CLI boundary."""
    if runtime == "codex":
        return {
            "model": model or os.environ.get("CODEX_MODEL", "inherit"),
            "base_url": base_url or os.environ.get("OPENAI_BASE_URL"),
            "auth_token": auth_token or os.environ.get("CODEX_API_KEY"),
        }
    if runtime == "claude":
        return {
            "model": model or os.environ.get("MODEL", claude_default_model),
            "base_url": base_url or os.environ.get("ANTHROPIC_BASE_URL"),
            "auth_token": (
                auth_token
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            ),
        }
    raise ValueError(
        f"unsupported CLI runtime {runtime!r}; expected one of {CLI_RUNTIME_NAMES}"
    )


def create_cli_runtime(
    runtime: str,
    *,
    workspace,
    model: str = "inherit",
    base_url: str | None = None,
    auth_token: str | None = None,
    claude_config_dir: os.PathLike[str] | str | None = None,
    **kwargs: Any,
):
    """Create one supported CLI runtime with provider-neutral arguments."""
    if runtime == "claude":
        return ClaudeRuntime(
            workspace=workspace,
            model=model,
            base_url=base_url,
            auth_token=auth_token,
            claude_config_dir=claude_config_dir,
            **kwargs,
        )
    if runtime == "codex":
        return CodexRuntime(
            workspace=workspace,
            model=model,
            base_url=base_url,
            api_key=auth_token,
            **kwargs,
        )
    raise ValueError(
        f"unsupported CLI runtime {runtime!r}; expected one of {CLI_RUNTIME_NAMES}"
    )

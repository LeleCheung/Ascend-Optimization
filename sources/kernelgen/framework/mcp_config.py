"""Provider-neutral MCP configuration and provider materialization."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from kernelgen.mcp_server.contract import MCP_SERVER_NAME


MCP_CONFIGURATION_PATH = ".kernelgen/mcp.json"
CLAUDE_MCP_CONFIGURATION_PATH = ".mcp.json"
MCP_CONFIGURATION_SCHEMA_VERSION = 1
_KERNELGEN_MCP_MODULE = "kernelgen.mcp_server.server"


@dataclass(frozen=True)
class MCPServerConfiguration:
    """One validated provider-neutral MCP server definition."""

    name: str
    transport: str
    command: tuple[str, ...]
    startup_timeout_seconds: float
    tool_timeout_seconds: float
    env: tuple[tuple[str, str], ...]

    @property
    def executable(self) -> str:
        return self.command[0]

    @property
    def args(self) -> tuple[str, ...]:
        return self.command[1:]

    @property
    def environment(self) -> dict[str, str]:
        return dict(self.env)


def canonical_mcp_configuration_path() -> Path:
    """Return the repository/package-owned MCP configuration source."""
    return Path(__file__).resolve().parents[1] / MCP_CONFIGURATION_PATH


def _positive_seconds(value: object, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} must be a positive number: {path}")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(f"{field} must be finite and positive: {path}")
    return seconds


def load_mcp_configuration(path: Path) -> dict[str, MCPServerConfiguration]:
    """Load and validate the complete provider-neutral MCP catalog."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid MCP configuration {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise RuntimeError(f"MCP configuration must be an object: {path}")
    unknown_root = set(document) - {"schema_version", "servers"}
    if unknown_root:
        raise RuntimeError(
            f"unknown MCP configuration fields in {path}: {sorted(unknown_root)}"
        )
    if document.get("schema_version") != MCP_CONFIGURATION_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported MCP configuration schema_version in {path}: "
            f"{document.get('schema_version')!r}"
        )
    raw_servers = document.get("servers")
    if not isinstance(raw_servers, dict) or not raw_servers:
        raise RuntimeError(
            f"MCP configuration servers must be a non-empty object: {path}"
        )

    servers: dict[str, MCPServerConfiguration] = {}
    allowed_server_fields = {
        "transport",
        "command",
        "startup_timeout_seconds",
        "tool_timeout_seconds",
        "env",
    }
    for name, raw_server in raw_servers.items():
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(f"MCP server names must be non-empty strings: {path}")
        if not isinstance(raw_server, dict):
            raise RuntimeError(f"MCP server {name!r} must be an object: {path}")
        unknown_fields = set(raw_server) - allowed_server_fields
        if unknown_fields:
            raise RuntimeError(
                f"unknown MCP server fields for {name!r} in {path}: "
                f"{sorted(unknown_fields)}"
            )
        transport = raw_server.get("transport")
        if transport != "stdio":
            raise RuntimeError(
                f"unsupported MCP transport for {name!r} in {path}: {transport!r}"
            )
        command = raw_server.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise RuntimeError(
                f"MCP server {name!r} command must be a non-empty "
                f"string list: {path}"
            )
        raw_env = raw_server.get("env", {})
        if not isinstance(raw_env, dict) or not all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            for key, value in raw_env.items()
        ):
            raise RuntimeError(
                f"MCP server {name!r} env must map names to strings: {path}"
            )
        servers[name] = MCPServerConfiguration(
            name=name,
            transport=transport,
            command=tuple(command),
            startup_timeout_seconds=_positive_seconds(
                raw_server.get("startup_timeout_seconds"),
                field=f"MCP server {name!r} startup_timeout_seconds",
                path=path,
            ),
            tool_timeout_seconds=_positive_seconds(
                raw_server.get("tool_timeout_seconds"),
                field=f"MCP server {name!r} tool_timeout_seconds",
                path=path,
            ),
            env=tuple(sorted(raw_env.items())),
        )
    return servers


def load_kernelgen_mcp_configuration(path: Path) -> MCPServerConfiguration:
    """Load the required KernelGen server from a neutral MCP catalog."""
    servers = load_mcp_configuration(path)
    try:
        return servers[MCP_SERVER_NAME]
    except KeyError as exc:
        raise RuntimeError(
            f"MCP configuration must define servers.{MCP_SERVER_NAME}: {path}"
        ) from exc


def _seconds_value(seconds: float) -> int | float:
    return int(seconds) if seconds.is_integer() else seconds


def _neutral_document(
    servers: dict[str, MCPServerConfiguration],
) -> dict[str, object]:
    return {
        "schema_version": MCP_CONFIGURATION_SCHEMA_VERSION,
        "servers": {
            name: {
                "transport": server.transport,
                "command": list(server.command),
                "startup_timeout_seconds": _seconds_value(
                    server.startup_timeout_seconds
                ),
                "tool_timeout_seconds": _seconds_value(
                    server.tool_timeout_seconds
                ),
                **({"env": server.environment} if server.env else {}),
            }
            for name, server in servers.items()
        },
    }


def _claude_document(
    servers: dict[str, MCPServerConfiguration],
) -> dict[str, object]:
    return {
        "mcpServers": {
            name: {
                "type": server.transport,
                "command": server.executable,
                "args": list(server.args),
                "timeout": int(server.tool_timeout_seconds * 1000),
                **({"env": server.environment} if server.env else {}),
            }
            for name, server in servers.items()
        }
    }


def _bind_kernelgen_python(
    servers: dict[str, MCPServerConfiguration],
    python_executable: os.PathLike[str] | str,
) -> None:
    """Bind KernelGen's module-based MCP command to the active interpreter."""
    server = servers.get(MCP_SERVER_NAME)
    if server is None or server.args[:2] != ("-m", _KERNELGEN_MCP_MODULE):
        return
    executable = os.fspath(python_executable)
    if not executable:
        raise RuntimeError("Python executable for KernelGen MCP must not be empty")
    if not os.path.isabs(executable):
        executable = os.path.abspath(executable)
    servers[MCP_SERVER_NAME] = replace(
        server,
        command=(executable, *server.args),
    )


def materialize_mcp_configuration(
    source: Path,
    workspace: Path,
    *,
    tool_timeout_seconds: int | float | None = None,
    python_executable: os.PathLike[str] | str | None = None,
) -> tuple[Path, Path]:
    """Snapshot MCP config, binding KernelGen to the active interpreter."""
    servers = load_mcp_configuration(source)
    if tool_timeout_seconds is not None:
        override = _positive_seconds(
            tool_timeout_seconds,
            field="tool_timeout_seconds override",
            path=source,
        )
        if MCP_SERVER_NAME not in servers:
            raise RuntimeError(
                f"MCP configuration must define servers.{MCP_SERVER_NAME}: {source}"
            )
        servers[MCP_SERVER_NAME] = replace(
            servers[MCP_SERVER_NAME],
            tool_timeout_seconds=override,
        )
    _bind_kernelgen_python(
        servers,
        sys.executable if python_executable is None else python_executable,
    )

    neutral_path = workspace / MCP_CONFIGURATION_PATH
    claude_path = workspace / CLAUDE_MCP_CONFIGURATION_PATH
    neutral_path.parent.mkdir(parents=True, exist_ok=True)
    claude_path.parent.mkdir(parents=True, exist_ok=True)
    neutral_path.write_text(
        json.dumps(_neutral_document(servers), indent=2) + "\n",
        encoding="utf-8",
    )
    claude_path.write_text(
        json.dumps(_claude_document(servers), indent=2) + "\n",
        encoding="utf-8",
    )
    return neutral_path, claude_path

"""Provider-neutral KernelGen agent roles and provider materialization."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from kernelgen.mcp_server.contract import MCP_TOOL_NAMES


AGENT_ROLE_DIRECTORY = ".kernelgen/agents"
SUPPORTED_CAPABILITIES = frozenset(
    {"shell", "read", "write", "edit", "search", "skill"}
)
_CLAUDE_CAPABILITY_TOOLS = {
    "shell": ("Bash",),
    "read": ("Read",),
    "write": ("Write",),
    "edit": ("Edit",),
    "search": ("Glob", "Grep"),
    "skill": ("Skill",),
}
_SUPPORTED_APPROVALS = frozenset({"default", "no_prompts"})


@dataclass(frozen=True)
class AgentRole:
    """One provider-neutral role definition."""

    name: str
    description: str
    capabilities: tuple[str, ...]
    mcp_tools: tuple[str, ...]
    subagents: tuple[str, ...]
    model: str
    approval: str
    body: str
    path: Path

    @property
    def qualified_mcp_tools(self) -> tuple[str, ...]:
        return tuple(f"mcp__kernelgen__{name}" for name in self.mcp_tools)


def canonical_agent_roles_dir() -> Path:
    """Return the repository/package-owned role source directory."""
    return Path(__file__).resolve().parents[1] / AGENT_ROLE_DIRECTORY


def _string_list(metadata: dict, key: str, path: Path) -> tuple[str, ...]:
    value = metadata.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"agent {key} must be an explicit string list: {path}")
    if len(value) != len(set(value)):
        raise RuntimeError(f"agent {key} contains duplicates: {path}")
    return tuple(value)


def load_agent_role(path: Path, *, expected_name: str | None = None) -> AgentRole:
    """Parse and validate one canonical KernelGen role."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"unable to read agent role: {path}") from exc
    if not text.startswith("---"):
        raise RuntimeError(f"agent role has no YAML frontmatter: {path}")
    try:
        _, raw_frontmatter, body = text.split("---", 2)
        metadata = yaml.safe_load(raw_frontmatter) or {}
    except (ValueError, yaml.YAMLError) as exc:
        raise RuntimeError(f"invalid agent role frontmatter in {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(f"agent role frontmatter must be a mapping: {path}")

    allowed_keys = {
        "name",
        "description",
        "capabilities",
        "mcp_tools",
        "subagents",
        "model",
        "approval",
    }
    unknown_keys = set(metadata) - allowed_keys
    if unknown_keys:
        raise RuntimeError(
            f"unknown provider-neutral agent fields in {path}: {sorted(unknown_keys)}"
        )

    name = str(metadata.get("name", "")).strip()
    required_name = expected_name or path.stem
    if name != required_name or path.stem != required_name:
        raise RuntimeError(
            f"agent role filename/name mismatch: expected {required_name!r}, "
            f"got {name!r} in {path}"
        )
    description = str(metadata.get("description", "")).strip()
    if not description:
        raise RuntimeError(f"agent role description is required: {path}")
    capabilities = _string_list(metadata, "capabilities", path)
    unknown_capabilities = set(capabilities) - SUPPORTED_CAPABILITIES
    if unknown_capabilities:
        raise RuntimeError(
            f"unknown agent capabilities in {path}: {sorted(unknown_capabilities)}"
        )
    mcp_tools = _string_list(metadata, "mcp_tools", path)
    unknown_mcp_tools = set(mcp_tools) - MCP_TOOL_NAMES
    if unknown_mcp_tools:
        raise RuntimeError(
            f"unknown KernelGen MCP tools in {path}: {sorted(unknown_mcp_tools)}"
        )
    subagents = _string_list(metadata, "subagents", path)
    model = str(metadata.get("model", "")).strip()
    if not model:
        raise RuntimeError(f"agent role model is required: {path}")
    approval = str(metadata.get("approval", "default")).strip()
    if approval not in _SUPPORTED_APPROVALS:
        raise RuntimeError(f"unsupported agent approval mode {approval!r}: {path}")
    body = body.lstrip()
    if not body.strip():
        raise RuntimeError(f"agent role body is empty: {path}")
    return AgentRole(
        name=name,
        description=description,
        capabilities=capabilities,
        mcp_tools=mcp_tools,
        subagents=subagents,
        model=model,
        approval=approval,
        body=body,
        path=path,
    )


def load_agent_roles(directory: Path) -> dict[str, AgentRole]:
    """Load a complete role catalog and validate subagent references."""
    if not directory.is_dir():
        raise RuntimeError(f"KernelGen agent role directory is missing: {directory}")
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise RuntimeError(f"KernelGen agent role directory is empty: {directory}")
    roles = {path.stem: load_agent_role(path) for path in paths}
    referenced = {
        subagent
        for role in roles.values()
        for subagent in role.subagents
    }
    missing = referenced - set(roles)
    if missing:
        raise RuntimeError(f"unknown KernelGen subagent role(s): {sorted(missing)}")
    return roles


def render_inline_role(role: AgentRole) -> str:
    """Render neutral metadata as prompt-visible capability guidance."""
    capabilities = ", ".join(role.capabilities) or "none"
    mcp_tools = ", ".join(role.qualified_mcp_tools) or "none"
    subagents = ", ".join(role.subagents) or "none"
    return (
        "--- KERNELGEN ROLE CAPABILITIES ---\n"
        f"Capabilities: {capabilities}\n"
        f"KernelGen MCP tools: {mcp_tools}\n"
        f"Delegable subagents: {subagents}\n"
        "Use only these capabilities, tools, and subagents.\n\n"
        f"{role.body}"
    )


def _render_claude_role(role: AgentRole) -> str:
    tools: list[str] = []
    for capability in role.capabilities:
        tools.extend(_CLAUDE_CAPABILITY_TOOLS[capability])
    tools.extend(f"Agent({name})" for name in role.subagents)
    tools.extend(role.qualified_mcp_tools)
    rendered_tools = ", ".join(tools) if tools else "[]"
    lines = [
        "---",
        f"name: {role.name}",
        f"description: {json.dumps(role.description, ensure_ascii=False)}",
        f"tools: {rendered_tools}",
        f"model: {role.model}",
    ]
    if role.approval == "no_prompts":
        lines.append("permissionMode: dontAsk")
    lines.extend(["---", "", role.body])
    return "\n".join(lines)


def _render_codex_role(role: AgentRole) -> str:
    instructions = render_inline_role(role)
    lines = [
        f"name = {json.dumps(role.name, ensure_ascii=False)}",
        f"description = {json.dumps(role.description, ensure_ascii=False)}",
        f"developer_instructions = {json.dumps(instructions, ensure_ascii=False)}",
    ]
    if role.model != "inherit":
        lines.append(f"model = {json.dumps(role.model, ensure_ascii=False)}")
    lines.extend(
        [
            "",
            "[mcp_servers.kernelgen]",
            "enabled_tools = "
            + json.dumps(list(role.mcp_tools), ensure_ascii=False),
        ]
    )
    return "\n".join(lines) + "\n"


def materialize_agent_role(source: Path, workspace: Path) -> None:
    """Snapshot and generate provider files for one standalone role."""
    role = load_agent_role(source)
    snapshot = workspace / AGENT_ROLE_DIRECTORY / source.name
    if snapshot.resolve() != source.resolve():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
        role = load_agent_role(snapshot)
    claude_path = workspace / ".claude" / "agents" / f"{role.name}.md"
    codex_path = workspace / ".codex" / "agents" / f"{role.name}.toml"
    claude_path.parent.mkdir(parents=True, exist_ok=True)
    codex_path.parent.mkdir(parents=True, exist_ok=True)
    claude_path.write_text(_render_claude_role(role), encoding="utf-8")
    codex_path.write_text(_render_codex_role(role), encoding="utf-8")


def materialize_agent_roles(source: Path, workspace: Path) -> None:
    """Snapshot neutral roles and generate Claude/Codex provider files."""
    source = source.resolve()
    roles = load_agent_roles(source)
    snapshot = workspace / AGENT_ROLE_DIRECTORY
    if snapshot.resolve() != source:
        if snapshot.exists():
            shutil.rmtree(snapshot)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, snapshot)
        roles = load_agent_roles(snapshot)

    claude_agents = workspace / ".claude" / "agents"
    codex_agents = workspace / ".codex" / "agents"
    for destination in (claude_agents, codex_agents):
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
    for name, role in roles.items():
        (claude_agents / f"{name}.md").write_text(
            _render_claude_role(role),
            encoding="utf-8",
        )
        (codex_agents / f"{name}.toml").write_text(
            _render_codex_role(role),
            encoding="utf-8",
        )

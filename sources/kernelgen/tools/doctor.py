"""Validate provider-neutral roles, skills, and MCP registration."""

from __future__ import annotations

import argparse
from pathlib import Path

from kernelgen.framework.agent_roles import load_agent_roles
from kernelgen.framework.agent_skills import (
    AGENT_SKILLS_DIRECTORY,
    load_agent_skills,
)
from kernelgen.framework.mcp_config import (
    MCP_CONFIGURATION_PATH,
    load_kernelgen_mcp_configuration,
)


_MCP_MODULE = "kernelgen.mcp_server.server"


def validate_workspace(workspace: str | Path) -> list[str]:
    """Return actionable configuration problems; an empty list means healthy."""
    root = Path(workspace).resolve()
    agents_dir = root / ".kernelgen" / "agents"
    errors: list[str] = []
    roles = {}
    try:
        roles = load_agent_roles(agents_dir)
    except RuntimeError as exc:
        errors.append(str(exc))

    skills_dir = root / AGENT_SKILLS_DIRECTORY
    if skills_dir.exists():
        try:
            load_agent_skills(skills_dir)
        except RuntimeError as exc:
            errors.append(str(exc))

    config_path = root / MCP_CONFIGURATION_PATH
    if any(role.mcp_tools for role in roles.values()) or config_path.exists():
        if not config_path.is_file():
            errors.append(
                f"agent roles reference MCP tools but {config_path} is missing"
            )
        else:
            try:
                server = load_kernelgen_mcp_configuration(config_path)
                if _MCP_MODULE not in server.command:
                    errors.append(
                        f"{config_path}: kernelgen must launch {_MCP_MODULE}"
                    )
            except RuntimeError as exc:
                errors.append(str(exc))
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate KernelGen agent-role, Agent Skill, and MCP configuration."
        )
    )
    parser.add_argument(
        "workspace", nargs="?", default=str(Path(__file__).resolve().parents[1])
    )
    args = parser.parse_args(argv)
    problems = validate_workspace(args.workspace)
    if problems:
        print("KernelGen configuration has problems:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(
        "KernelGen agent-role, Agent Skill, and MCP configuration is healthy: "
        f"{Path(args.workspace).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

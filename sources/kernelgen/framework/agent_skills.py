"""Provider-neutral KernelGen skills and provider materialization."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml


AGENT_SKILLS_DIRECTORY = ".kernelgen/skills"
CLAUDE_SKILLS_DIRECTORY = ".claude/skills"
CODEX_SKILLS_DIRECTORY = ".agents/skills"


@dataclass(frozen=True)
class AgentSkill:
    """Validated metadata for one provider-neutral Agent Skill."""

    name: str
    description: str
    path: Path


def canonical_agent_skills_dir() -> Path:
    """Return the repository/package-owned skill source directory."""
    return Path(__file__).resolve().parents[1] / AGENT_SKILLS_DIRECTORY


def load_agent_skill(path: Path, *, expected_name: str | None = None) -> AgentSkill:
    """Parse and validate one canonical Agent Skill directory."""
    skill_file = path / "SKILL.md"
    try:
        text = skill_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"unable to read Agent Skill: {skill_file}") from exc
    if not text.startswith("---"):
        raise RuntimeError(f"Agent Skill has no YAML frontmatter: {skill_file}")
    try:
        _, raw_frontmatter, body = text.split("---", 2)
        metadata = yaml.safe_load(raw_frontmatter) or {}
    except (ValueError, yaml.YAMLError) as exc:
        raise RuntimeError(
            f"invalid Agent Skill frontmatter in {skill_file}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Agent Skill frontmatter must be a mapping: {skill_file}")

    name = str(metadata.get("name", "")).strip()
    required_name = expected_name or path.name
    if name != required_name or path.name != required_name:
        raise RuntimeError(
            f"Agent Skill directory/name mismatch: expected {required_name!r}, "
            f"got {name!r} in {skill_file}"
        )
    description = str(metadata.get("description", "")).strip()
    if not description:
        raise RuntimeError(f"Agent Skill description is required: {skill_file}")
    if not body.strip():
        raise RuntimeError(f"Agent Skill body is empty: {skill_file}")
    return AgentSkill(name=name, description=description, path=path)


def load_agent_skills(directory: Path) -> dict[str, AgentSkill]:
    """Load and validate a complete provider-neutral Agent Skill catalog."""
    if not directory.is_dir():
        raise RuntimeError(f"KernelGen Agent Skill directory is missing: {directory}")
    skill_dirs = sorted(
        path
        for path in directory.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not skill_dirs:
        raise RuntimeError(f"KernelGen Agent Skill directory is empty: {directory}")
    skills = {path.name: load_agent_skill(path) for path in skill_dirs}
    if len(skills) != len(skill_dirs):
        raise RuntimeError(f"duplicate KernelGen Agent Skill names: {directory}")
    return skills


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def remove_materialized_agent_skills(workspace: Path) -> None:
    """Remove generated provider skill directories from a workspace."""
    _remove_path(workspace / CLAUDE_SKILLS_DIRECTORY)
    _remove_path(workspace / CODEX_SKILLS_DIRECTORY)


def materialize_agent_skills(source: Path, workspace: Path) -> None:
    """Snapshot neutral skills and generate Claude/Codex discovery trees."""
    source = source.resolve()
    load_agent_skills(source)
    snapshot = workspace / AGENT_SKILLS_DIRECTORY
    if snapshot.resolve() != source:
        _remove_path(snapshot)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, snapshot)
        load_agent_skills(snapshot)

    remove_materialized_agent_skills(workspace)
    for destination in (
        workspace / CLAUDE_SKILLS_DIRECTORY,
        workspace / CODEX_SKILLS_DIRECTORY,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot, destination)

"""Tests for provider-neutral Agent Skills and provider materialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernelgen.framework.agent_skills import (
    canonical_agent_skills_dir,
    load_agent_skill,
    load_agent_skills,
    materialize_agent_skills,
)


def _write_skill(root: Path, name: str = "test-skill") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: Use {name} for its deterministic test workflow.\n"
        "---\n"
        "\n"
        f"Follow the {name} workflow.\n",
        encoding="utf-8",
    )
    return skill_dir


def test_repository_skill_catalog_is_provider_neutral():
    skills = load_agent_skills(canonical_agent_skills_dir())

    assert set(skills) == {"kernelgen-knowledge", "pytest-review"}
    assert "native knowledge MCP tools" in skills["kernelgen-knowledge"].description


def test_materialization_snapshots_complete_skill_for_claude_and_codex(tmp_path):
    source = tmp_path / "source"
    skill_dir = _write_skill(source)
    references = skill_dir / "references"
    references.mkdir()
    (references / "guide.md").write_text("provider-neutral guide\n")
    workspace = tmp_path / "workspace"

    materialize_agent_skills(source, workspace)

    relative_files = (
        Path("test-skill/SKILL.md"),
        Path("test-skill/references/guide.md"),
    )
    for relative in relative_files:
        expected = (source / relative).read_text(encoding="utf-8")
        assert (
            workspace / ".kernelgen" / "skills" / relative
        ).read_text(encoding="utf-8") == expected
        assert (
            workspace / ".claude" / "skills" / relative
        ).read_text(encoding="utf-8") == expected
        assert (
            workspace / ".agents" / "skills" / relative
        ).read_text(encoding="utf-8") == expected


def test_skill_validation_rejects_directory_name_mismatch(tmp_path):
    skill_dir = _write_skill(tmp_path, "directory-name")
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        skill_file.read_text(encoding="utf-8").replace(
            "name: directory-name",
            "name: different-name",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="directory/name mismatch"):
        load_agent_skill(skill_dir)

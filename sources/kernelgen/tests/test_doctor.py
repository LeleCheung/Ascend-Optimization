"""Tests for provider-neutral role, skill, and MCP validation."""

import json
import sys
import tempfile
from pathlib import Path

import yaml

from kernelgen.tools.doctor import validate_workspace


def _workspace(tmp_path, *, mcp_tool="eval_round", subagents=None):
    agents = tmp_path / ".kernelgen" / "agents"
    agents.mkdir(parents=True)
    (agents / "kernel-coder.md").write_text(
        "---\n"
        "name: kernel-coder\n"
        "description: test\n"
        "capabilities: [read]\n"
        f"mcp_tools: [{mcp_tool}]\n"
        f"subagents: {json.dumps(subagents or [])}\n"
        "model: inherit\n"
        "---\nrole\n"
    )
    config_path = tmp_path / ".kernelgen" / "mcp.json"
    config_path.write_text(json.dumps({
        "schema_version": 1,
        "servers": {
            "kernelgen": {
                "transport": "stdio",
                "command": ["python3", "-m", "kernelgen.mcp_server.server"],
                "startup_timeout_seconds": 30,
                "tool_timeout_seconds": 2100,
            }
        }
    }))


def test_healthy_workspace(tmp_path):
    _workspace(tmp_path)
    assert validate_workspace(tmp_path) == []


def test_unknown_tool_is_reported(tmp_path):
    _workspace(tmp_path, mcp_tool="typo")
    problems = validate_workspace(tmp_path)
    assert any("unknown KernelGen MCP tool" in problem for problem in problems)


def test_missing_mcp_config_is_reported(tmp_path):
    _workspace(tmp_path)
    (tmp_path / ".kernelgen" / "mcp.json").unlink()
    problems = validate_workspace(tmp_path)
    assert any(".kernelgen/mcp.json is missing" in problem for problem in problems)


def test_missing_referenced_subagent_is_reported(tmp_path):
    _workspace(tmp_path, subagents=["kernel-profile-analyzer"])
    problems = validate_workspace(tmp_path)
    assert any("unknown KernelGen subagent role" in problem for problem in problems)


def test_invalid_agent_skill_is_reported(tmp_path):
    _workspace(tmp_path)
    skill_dir = tmp_path / ".kernelgen" / "skills" / "broken-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: broken-skill\n---\nbody\n",
        encoding="utf-8",
    )

    problems = validate_workspace(tmp_path)

    assert any("Agent Skill description is required" in problem for problem in problems)


def test_workflow_role_tools_match_knowledge_mode():
    root = Path(__file__).resolve().parents[1]

    def metadata(name):
        text = (root / ".kernelgen" / "agents" / name).read_text(encoding="utf-8")
        return yaml.safe_load(text.split("---", 2)[1])

    knowledge_tools = {
        "query_knowledge", "get_knowledge", "query_sources", "get_source"
    }
    coder = metadata("kernel-coder.md")
    profile = metadata("kernel-profile-analyzer.md")
    coder_tools = set(coder["mcp_tools"])
    profile_tools = set(profile["mcp_tools"])
    assert coder["subagents"] == ["kernel-profile-analyzer"]
    assert "skill" not in coder["capabilities"]
    assert "preflight_kernel" in coder_tools
    assert not {"get_profile_context", "profile_workloads", "record_profile_analysis"} & coder_tools
    assert {"get_profile_context", "profile_workloads", "record_profile_analysis"} <= profile_tools
    for forbidden in (
        "write",
        "edit",
        "shell",
        "preflight_kernel",
        "eval_round",
        "finalize_round",
        "next",
    ):
        assert forbidden not in profile_tools and forbidden not in profile["capabilities"]
    for tool in knowledge_tools:
        assert tool not in coder_tools and tool not in profile_tools

    knowledge_coder = metadata("kernel-knowledge-coder.md")
    knowledge_profile = metadata("kernel-knowledge-profile-analyzer.md")
    knowledge_coder_tools = set(knowledge_coder["mcp_tools"])
    knowledge_profile_tools = set(knowledge_profile["mcp_tools"])
    assert knowledge_coder["subagents"] == ["kernel-knowledge-profile-analyzer"]
    assert "get_profile_context" in knowledge_profile_tools
    for tool in knowledge_tools:
        assert tool in knowledge_coder_tools
        assert tool in knowledge_profile_tools
    for forbidden in (
        "Write",
        "Edit",
        "Bash",
        "preflight_kernel",
        "eval_round",
        "finalize_round",
        "next",
    ):
        assert forbidden not in knowledge_profile_tools and forbidden not in knowledge_profile["capabilities"]

    distiller = metadata("kernel-distiller.md")
    knowledge_distiller = metadata("kernel-knowledge-distiller.md")
    distiller_tools = set(distiller["mcp_tools"])
    knowledge_distiller_tools = set(knowledge_distiller["mcp_tools"])
    for tool in knowledge_tools:
        assert tool not in distiller_tools
    assert "read" in knowledge_distiller["capabilities"]
    assert "get_knowledge" in knowledge_distiller_tools
    assert "get_source" in knowledge_distiller_tools

    analyzer_tools = set(metadata("kernel-analyzer.md")["mcp_tools"])
    epoch_tools = set(metadata("kernel-epoch-summary.md")["mcp_tools"])
    for tool in knowledge_tools:
        assert tool in analyzer_tools and tool in epoch_tools
    for tools in (analyzer_tools, epoch_tools):
        for forbidden in (
            "write",
            "edit",
            "preflight_kernel",
            "eval_round",
            "finalize_round",
        ):
            assert forbidden not in tools

    for forbidden in (
        "Write",
        "Edit",
        "query_knowledge",
        "query_sources",
        "preflight_kernel",
        "eval_round",
        "finalize_round",
    ):
        assert forbidden not in knowledge_distiller_tools


def test_role_prompts_use_current_knowledge_context_contract():
    root = Path(__file__).resolve().parents[1] / ".kernelgen" / "agents"
    analyzer = (root / "kernel-analyzer.md").read_text(encoding="utf-8")
    epoch = (root / "kernel-epoch-summary.md").read_text(encoding="utf-8")
    profile = (
        root / "kernel-knowledge-profile-analyzer.md"
    ).read_text(encoding="utf-8")

    assert 'phase: "initial"' in analyzer
    assert 'phase: "post_evaluation"' in epoch
    assert 'phase: "post_profile"' in profile
    assert "complete structured draft findings" in profile
    assert "propose a seed" not in epoch.lower()


if __name__ == "__main__":
    import inspect
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            if "tmp_path" in inspect.signature(test).parameters:
                with tempfile.TemporaryDirectory() as directory:
                    test(Path(directory))
            else:
                test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)

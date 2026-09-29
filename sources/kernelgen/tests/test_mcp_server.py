"""Smoke tests for official FastMCP registration."""

import sys
from pathlib import Path

from kernelgen.data.experiment_plan import ExperimentPlan
from kernelgen.mcp_server.contract import MCP_TOOL_NAMES
from kernelgen.mcp_server import server
from kernelgen.mcp_server.server import mcp
from kernelgen.tests.helpers import experiment_plan


def test_server_registers_exact_public_tools():
    assert set(mcp._tool_manager._tools) == MCP_TOOL_NAMES
    assert "finalize_round" in MCP_TOOL_NAMES
    assert "record_round" not in MCP_TOOL_NAMES


def test_profile_tool_schemas_are_explicit():
    profile_schema = mcp._tool_manager._tools["profile_workloads"].parameters
    assert profile_schema["properties"]["level"]["enum"] == [
        "metrics",
        "source",
        "instruction",
    ]
    analysis_schema = mcp._tool_manager._tools["record_profile_analysis"].parameters
    profile_analysis = analysis_schema["$defs"]["ProfileAnalysis"]
    assert profile_analysis["additionalProperties"] is False
    assert profile_analysis["properties"]["status"]["enum"] == [
        "completed",
        "inconclusive",
        "unsupported",
        "failed",
    ]


def test_knowledge_skill_requires_bounded_on_demand_dual_retrieval():
    root = Path(__file__).resolve().parents[1]
    skill = (
        root / ".kernelgen" / "skills" / "kernelgen-knowledge" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "dual retrieval is required" in skill
    assert "A new round by itself is not a reason to query" in skill
    assert "Retrieval is independent of ExperimentPlan kind" in skill
    assert "Carry a prior use forward without a new query" in skill
    assert "max_results=4" in skill
    assert "no more than 200 lines" in skill

    for name in (
        "kernel-knowledge-coder.md",
        "kernel-knowledge-profile-analyzer.md",
    ):
        content = (root / ".kernelgen" / "agents" / name).read_text(
            encoding="utf-8"
        )
        assert "$kernelgen-knowledge" in content
        assert "query_knowledge" in content
        assert "query_sources" in content
        assert "max_results=4" in content

    knowledge_coder = (
        root / ".kernelgen" / "agents" / "kernel-knowledge-coder.md"
    ).read_text(encoding="utf-8")
    assert (
        "Absence from `get_server_status` alone is not a reason"
        in knowledge_coder
    )
    assert "consult `$kernelgen-knowledge` first" in knowledge_coder
    assert "not the default source for documented" in knowledge_coder

    for name in ("kernel-coder.md", "kernel-profile-analyzer.md"):
        content = (root / ".kernelgen" / "agents" / name).read_text(
            encoding="utf-8"
        )
        assert "$kernelgen-knowledge" not in content
        assert "query_knowledge" not in content
        assert "query_sources" not in content


def test_k08_bounds_remain_prompt_guidance_not_mcp_schema_gates():
    source_query = mcp._tool_manager._tools["query_sources"].parameters
    source_read = mcp._tool_manager._tools["get_source"].parameters

    assert source_query["properties"]["max_results"]["default"] == 12
    assert "maximum" not in source_query["properties"]["max_results"]
    assert "maximum" not in source_read["properties"]["line_end"]


def test_preflight_tool_schema_only_accepts_workspace_relative_kernel():
    schema = mcp._tool_manager._tools["preflight_kernel"].parameters
    assert set(schema["properties"]) == {"kernel_path"}
    assert schema["properties"]["kernel_path"]["default"] == "tmp/main.py"


def test_server_status_tool_is_parameterless():
    schema = mcp._tool_manager._tools["get_server_status"].parameters
    assert schema["properties"] == {}
    assert schema.get("required", []) == []


def test_tool_descriptions_explain_probe_only_broken_semantics():
    status_description = mcp._tool_manager._tools[
        "get_server_status"
    ].description
    preflight_description = mcp._tool_manager._tools[
        "preflight_kernel"
    ].description
    eval_description = mcp._tool_manager._tools["eval_round"].description

    assert "scheduler.broken > 0" in status_description
    assert "does not prove either slot is broken" in preflight_description
    assert "not proof of a broken slot" in eval_description


def test_debug_job_schema_uses_workspace_files_and_explicit_request():
    schema = mcp._tool_manager._tools["submit_debug_job"].parameters
    assert set(schema["properties"]) == {"request"}
    request = schema["$defs"]["WorkspaceDebugJobRequest"]
    assert request["additionalProperties"] is False
    assert set(request["required"]) >= {"purpose", "command", "files"}
    source = schema["$defs"]["DebugWorkspaceFile"]
    assert source["additionalProperties"] is False
    assert set(source["required"]) == {"path"}
    assert "get_debug_job" not in mcp._tool_manager._tools
    assert "cancel_debug_job" not in mcp._tool_manager._tools


def test_eval_round_is_a_thin_service_adapter(tmp_path, monkeypatch):
    calls = []
    plan = ExperimentPlan.model_validate(experiment_plan(1))
    monkeypatch.setattr(server, "workspace_from_env", lambda: tmp_path)
    monkeypatch.setattr(
        server,
        "evaluate_round_impl",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "PASSED"},
    )

    result = server.eval_round(plan, "tmp/main.py")

    assert result == {"status": "PASSED"}
    assert calls == [((tmp_path, "tmp/main.py", plan), {"confirm_no_knowledge_applied": False})]


def test_missing_knowledge_lineage_does_not_block_eval(tmp_path, monkeypatch):
    state = tmp_path / ".kernelgen" / "knowledge" / "state.json"
    state.parent.mkdir(parents=True)
    state.write_text('{"mode": "read_write_v1"}', encoding="utf-8")
    plan = ExperimentPlan.model_validate(
        experiment_plan(
            1,
            knowledge_uses=[
                {
                    "concept_ref": "kg:method:test",
                    "query_event_id": "query:missing",
                    "role": "hypothesis",
                    "disposition": "adopted",
                    "application_note": "selected a tiled reduction",
                    "affected_parts": ["strategy"],
                }
            ],
        )
    )
    calls = []
    monkeypatch.setattr(server, "workspace_from_env", lambda: tmp_path)
    monkeypatch.setattr(
        server,
        "evaluate_round_impl",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "PASSED"},
    )

    result = server.eval_round(plan, "tmp/main.py")

    assert result == {"status": "PASSED"}
    assert calls == [((tmp_path, "tmp/main.py", plan), {"confirm_no_knowledge_applied": False})]


def test_preflight_is_a_thin_service_adapter(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(server, "workspace_from_env", lambda: tmp_path)
    monkeypatch.setattr(
        server,
        "preflight_candidate",
        lambda *args: calls.append(args) or {"status": "PASSED"},
    )

    result = server.preflight_kernel("tmp/main.py")

    assert result == {"status": "PASSED"}
    assert calls == [(tmp_path, "tmp/main.py")]


def test_get_server_status_uses_workspace_tool_context(tmp_path, monkeypatch):
    calls = []
    context = type(
        "Context",
        (),
        {"eval_server_url": "http://configured-server:8000"},
    )()
    monkeypatch.setattr(server, "workspace_from_env", lambda: tmp_path)
    monkeypatch.setattr(server, "load_tool_context", lambda workspace: context)
    monkeypatch.setattr(
        server,
        "get_service_status_impl",
        lambda server_url: calls.append(server_url)
        or {
            "status": "ok",
            "backend": "npu",
            "target": {"device": "Ascend910B"},
            "software": {"runtime": "cann"},
        },
    )

    result = server.get_server_status()

    assert result["status"] == "ok"
    assert result["target"]["device"] == "Ascend910B"
    assert calls == ["http://configured-server:8000"]


def test_get_server_status_returns_structured_error(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "workspace_from_env", lambda: tmp_path)
    monkeypatch.setattr(
        server,
        "load_tool_context",
        lambda workspace: (_ for _ in ()).throw(RuntimeError("missing context")),
    )

    result = server.get_server_status()

    assert result == {
        "status": "ERROR",
        "reason": "SERVER_STATUS_FAILED",
        "error": "missing context",
    }


def test_submit_debug_job_is_one_blocking_adapter(tmp_path, monkeypatch):
    calls = []
    context = object()
    request = server.WorkspaceDebugJobRequest(
        purpose="inspect numerics",
        command=["{python}", "tmp/debug/check.py"],
        files=[{"path": "tmp/debug/check.py"}],
    )
    monkeypatch.setattr(server, "workspace_from_env", lambda: tmp_path)
    monkeypatch.setattr(server, "load_tool_context", lambda workspace: context)
    monkeypatch.setattr(
        server,
        "submit_workspace_debug_job",
        lambda *args: calls.append(args) or {"status": "SUCCEEDED"},
    )

    result = server.submit_debug_job(request)

    assert result == {"status": "SUCCEEDED"}
    assert calls == [(tmp_path, context, request)]


if __name__ == "__main__":
    try:
        for test in (
            test_server_registers_exact_public_tools,
            test_profile_tool_schemas_are_explicit,
            test_preflight_tool_schema_only_accepts_workspace_relative_kernel,
        ):
            test()
            print(f"  ✓ {test.__name__}")
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print("\n3/3 passed")

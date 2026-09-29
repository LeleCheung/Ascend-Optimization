"""Real MCP stdio handshake and gate transition test."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from kernelgen.data.ledger import Ledger
from kernelgen.mcp_server.contract import MCP_TOOL_NAMES
from kernelgen.tests.helpers import experiment_plan, round_conclusion


def _eval_result() -> dict:
    return {
        "api_version": "v5.1",
        "status": "PASSED",
        "geo_mean": 1.2,
        "min_speedup": 1.2,
        "worst_workload_uuid": "u0",
        "latency_ms": 0.1,
        "abs_err": 0.0,
        "rel_err": 0.0,
        "num_workloads": 1,
        "num_passed": 1,
        "server_backend": "cuda",
        "per_workload": [
            {
                "uuid": "u0",
                "axes": {"M": 128},
                "status": "PASSED",
                "speedup": 1.2,
                "latency_ms": 0.1,
                "reference_latency_ms": 0.12,
            }
        ],
    }


def _prepare_pending_round(workspace: Path) -> None:
    ledger = Ledger(workspace)
    ledger.record_eval(
        _eval_result(),
        "def run(): pass",
        experiment_plan(1),
        profile_enabled=True,
    )
    snapshot = workspace / ".kernelgen" / "evals" / "round-0001"
    snapshot.mkdir(parents=True)
    payloads = {
        "result.json": _eval_result(),
        "solution.json": {},
        "definition.json": {},
        "workloads.json": [{"uuid": "u0", "axes": {"M": 128}}],
        "identity.json": {
            "round_num": 1,
            "evaluation_fingerprint": "stdio-fingerprint",
            "solution_sha256": "stdio-solution-sha",
            "definition_name": "op",
            "definition_sha256": "definition-sha",
            "workload_uuids": ["u0"],
            "workload_sha256": ["workload-sha"],
            "catalog_name": "fixture",
            "target_hardware": "H100",
            "server_backend": "cuda",
        },
    }
    (snapshot / "main.py").write_text("def run(): pass", encoding="utf-8")
    for name, payload in payloads.items():
        (snapshot / name).write_text(json.dumps(payload), encoding="utf-8")
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint="stdio-fingerprint",
        snapshot_path=str(snapshot.relative_to(workspace)),
    )


def _tool_json(result) -> dict:
    assert result.isError is False
    return json.loads(result.content[0].text)


async def _exercise_stdio(workspace: Path) -> None:
    child_env = os.environ.copy()
    package_parent = str(Path(__file__).resolve().parents[2])
    child_env["PYTHONPATH"] = os.pathsep.join(
        [package_parent, child_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    child_env["KERNELGEN_WORKSPACE"] = str(workspace)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kernelgen.mcp_server.server"],
        env=child_env,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            initialized = await session.initialize()
            assert initialized.serverInfo.name == "kernelgen"
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == MCP_TOOL_NAMES

            conclusion = _tool_json(
                await session.call_tool(
                    "finalize_round",
                    {"conclusion": round_conclusion(1)},
                )
            )
            assert conclusion["recorded"] is True
            assert conclusion["conclusion_recorded"] is True
            assert conclusion["round_finalized"] is True
            assert conclusion["candidate_action"] == "KEEP"
            assert "profile best round 1" in conclusion["recommendations"][0]

            recorded = _tool_json(
                await session.call_tool(
                    "record_profile_analysis",
                    {
                        "round_num": 1,
                        "analysis": {
                            "schema_version": "1.0",
                            "round_num": 1,
                            "evaluation_fingerprint": "stdio-fingerprint",
                            "status": "failed",
                            "solution_sha256": "stdio-solution-sha",
                            "backend": "cuda",
                            "dominant_bound": "unknown",
                            "error": "synthetic profiler failure for stdio gate test",
                        },
                    },
                )
            )
            assert recorded["recorded"] is True
            assert recorded["status"] == "failed"

            unblocked = _tool_json(
                await session.call_tool(
                    "preflight_kernel",
                    {"kernel_path": "tmp/main.py"},
                )
            )
            assert unblocked["status"] != "PROFILE_ANALYSIS_REQUIRED"


def test_real_stdio_handshake_with_advisory_profile(tmp_path):
    _prepare_pending_round(tmp_path)
    asyncio.run(_exercise_stdio(tmp_path))
    record = Ledger(tmp_path).get_round(1)
    assert record.profile.status == "failed"
    assert record.profile.analysis_path == ".kernelgen/profile-analysis/round-0001.json"
    assert record.conclusion is not None


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_real_stdio_handshake_with_advisory_profile(Path(directory))
    print("  ✓ test_real_stdio_handshake_with_advisory_profile")
    print("\n1/1 passed")

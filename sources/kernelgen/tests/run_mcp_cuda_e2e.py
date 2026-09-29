"""Manual CUDA end-to-end test for the KernelGen stdio MCP server.

This test requires a running KernelGen Server with CUDA profiling.
It intentionally keeps the generated workspace so reports can be inspected.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from kernelgen.data.tool_context import ToolContext
from kernelgen.mcp_server.contract import MCP_TOOL_NAMES
from kernelgen.tests.helpers import experiment_plan, round_conclusion


def _tool_json(result) -> dict:
    if result.isError:
        text = result.content[0].text if result.content else "unknown MCP tool error"
        raise RuntimeError(text)
    return json.loads(result.content[0].text)


def _profile_analysis(eval_result: dict, profile_result: dict) -> dict:
    profile = profile_result["profiles"][0]
    workload_uuid = profile["workload_uuid"]
    workload = next(
        item
        for item in eval_result["per_workload"]
        if item["uuid"] == workload_uuid
    )
    artifact = next(
        item
        for item in profile["artifacts"]
        if item["kind"] in {"profile_details", "instruction_listing"}
    )
    return {
        "api_version": "v5.1",
        "round_num": eval_result["round_num"],
        "evaluation_fingerprint": eval_result["evaluation_fingerprint"],
        "status": "completed",
        "solution_sha256": eval_result["solution_sha256"],
        "backend": profile["backend"],
        "profiler": profile["profiler"],
        "capabilities": profile["capabilities"],
        "dominant_bound": "unknown",
        "profiled_workloads": [
            {
                "uuid": workload_uuid,
                "axes": workload.get("axes", {}),
                "eval_speedup": workload.get("speedup"),
                "eval_latency_ms": workload.get("latency_ms"),
                "eval_reference_latency_ms": workload.get("reference_latency_ms"),
                "selection_reason": "Selected from the authoritative profile workload candidates.",
                "profile_ids": [profile["profile_id"]],
                "manifest_paths": [profile["manifest_path"]],
                "status": "completed",
            }
        ],
        "findings": [
            {
                "category": "instruction_efficiency",
                "label": "NCU captured the evaluated kernel and instruction evidence",
                "backend_detail": "Protocol-level E2E evidence, not an optimization diagnosis.",
                "workload_uuids": [workload_uuid],
                "confidence": "high",
                "evidence": [
                    {
                        "metric": "kernel_count",
                        "value": profile.get("metrics", {}).get("kernel_count", 0),
                        "unit": "kernels",
                        "artifact_path": artifact["local_path"],
                        "artifact_kind": artifact["kind"],
                    }
                ],
                "inference": "The backend produced a downloadable report for the exact eval snapshot.",
            }
        ],
        "performance_interpretation": (
            "The test verifies evidence identity and transport only; eval latency remains authoritative."
        ),
        "next_experiment": {
            "action_category": "e2e_validation",
            "action_description": "Run a changed candidate and compare its authoritative eval result.",
            "expected_impact": "Verify that a changed solution receives a new evaluation fingerprint.",
            "risks_and_rollback": "No production state is changed; discard the temporary workspace.",
            "validation_workloads": [workload_uuid],
            "success_criteria": ["The changed candidate is evaluated and profiled under a new fingerprint."],
        },
    }


async def _run(args, workspace: Path) -> dict:
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
            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            expected_tools = set(MCP_TOOL_NAMES)
            if tool_names != expected_tools:
                raise AssertionError(f"unexpected MCP tools: {sorted(tool_names)}")

            preflighted = _tool_json(
                await session.call_tool(
                    "preflight_kernel",
                    {"kernel_path": "tmp/main.py"},
                )
            )
            if preflighted.get("status") != "PASSED":
                raise AssertionError(f"preflight did not pass: {preflighted}")

            evaluated = _tool_json(
                await session.call_tool(
                    "eval_round",
                    {
                        "kernel_path": "tmp/main.py",
                        "experiment_plan": experiment_plan(1),
                    },
                )
            )
            if evaluated.get("status") != "PASSED" or not evaluated.get("profile_required"):
                raise AssertionError(f"eval did not create a pending passing round: {evaluated}")
            round_num = evaluated["round_num"]

            context = _tool_json(
                await session.call_tool("get_profile_context", {"round_num": round_num})
            )
            if context.get("profile_state") != "pending":
                raise AssertionError(f"unexpected profile context: {context}")
            profile_workload_uuids = context.get("profile_workload_uuids", [])
            if not profile_workload_uuids:
                raise AssertionError(f"profile context has no eligible workloads: {context}")
            if args.stop_after_eval:
                return {
                    "server": initialized.serverInfo.name,
                    "tools": sorted(tool_names),
                    "preflight_status": preflighted["status"],
                    "round_num": round_num,
                    "eval_status": evaluated["status"],
                    "evaluation_fingerprint": evaluated["evaluation_fingerprint"],
                    "profile_status": context["profile_state"],
                    "workspace": str(workspace),
                }

            conclusion = round_conclusion(
                round_num,
                root_cause="The stdio MCP server and CUDA profile service are connected.",
                next_suggestion="Change the candidate before another diagnostic profile.",
                architecture_tag="cuda-a100",
                optimization_level="L1_architecture",
            )
            round_recorded = _tool_json(
                await session.call_tool("finalize_round", {"conclusion": conclusion})
            )
            if (
                not round_recorded.get("recorded")
                or not round_recorded.get("round_finalized")
                or "continue" not in round_recorded
            ):
                raise AssertionError(
                    f"pending profile incorrectly blocked finalization: {round_recorded}"
                )

            profile_args = {
                "round_num": round_num,
                "workload_uuids": [profile_workload_uuids[0]],
                "level": args.level,
            }
            profiled = _tool_json(await session.call_tool("profile_workloads", profile_args))
            if profiled.get("status") != "completed":
                raise AssertionError(f"profile did not complete: {profiled}")
            artifacts = {item["kind"] for item in profiled["profiles"][0]["artifacts"]}
            required_artifacts = {"vendor_report", "profile_details"}
            if args.level == "instruction":
                required_artifacts.add("instruction_listing")
            if not required_artifacts.issubset(artifacts):
                raise AssertionError(f"missing profile artifacts: {required_artifacts - artifacts}")

            cached = _tool_json(await session.call_tool("profile_workloads", profile_args))
            if not cached["profiles"][0].get("cached"):
                raise AssertionError("identical profile request did not reuse its manifest")

            analysis = _profile_analysis(evaluated, profiled)
            recorded = _tool_json(
                await session.call_tool(
                    "record_profile_analysis",
                    {"round_num": round_num, "analysis": analysis},
                )
            )
            if not recorded.get("recorded"):
                raise AssertionError(f"profile analysis was not recorded: {recorded}")

            repeated_preflight = _tool_json(
                await session.call_tool(
                    "preflight_kernel",
                    {"kernel_path": "tmp/main.py"},
                )
            )
            if repeated_preflight.get("status") != "PASSED":
                raise AssertionError(
                    f"repeated preflight did not pass: {repeated_preflight}"
                )
            repeated = _tool_json(
                await session.call_tool(
                    "eval_round",
                    {
                        "kernel_path": "tmp/main.py",
                        "experiment_plan": experiment_plan(2),
                    },
                )
            )
            if repeated.get("profile_required"):
                raise AssertionError(
                    f"identical eval retained an unresolved profile request: {repeated}"
                )
            if repeated.get("is_new_best") and not repeated.get("profile_reused"):
                raise AssertionError(
                    f"new-best identical snapshot did not reuse analysis: {repeated}"
                )
            reuse_gate = _tool_json(
                await session.call_tool(
                    "preflight_kernel",
                    {"kernel_path": "tmp/main.py"},
                )
            )
            if reuse_gate.get("status") != "ROUND_CONCLUSION_REQUIRED":
                raise AssertionError(f"reused eval did not require conclusion: {reuse_gate}")
            reused_round_num = repeated["round_num"]
            reused_conclusion = round_conclusion(
                reused_round_num,
                expectation_status="not_met",
                perf_gap_analysis="The identical solution reproduced the prior measurement, so the performance hypothesis had no code change to realize.",
                root_cause="The identical evaluation fingerprint reused prior analysis.",
                next_suggestion="Change the candidate before another evaluation.",
                architecture_tag="cuda-a100",
                optimization_level="L1_architecture",
            )
            reused_recorded = _tool_json(
                await session.call_tool(
                    "finalize_round",
                    {"conclusion": reused_conclusion},
                )
            )
            if not reused_recorded.get("recorded"):
                raise AssertionError(
                    f"reused round conclusion was not recorded: {reused_recorded}"
                )
            if not reused_recorded.get("round_finalized") or "continue" not in reused_recorded:
                raise AssertionError(f"reused round was not finalized: {reused_recorded}")

            return {
                "server": initialized.serverInfo.name,
                "tools": sorted(tool_names),
                "preflight_status": preflighted["status"],
                "round_num": round_num,
                "eval_status": evaluated["status"],
                "profile_status": profiled["status"],
                "profile_id": profiled["profiles"][0]["profile_id"],
                "artifacts": profiled["profiles"][0]["artifacts"],
                "analysis_path": recorded["analysis_path"],
                "gate_cleared": "action_required" not in reused_recorded,
                "profile_cache_reused": cached["profiles"][0]["cached"],
                "eval_analysis_reused": repeated["profile_reused"],
                "reused_round_conclusion_recorded": reused_recorded["recorded"],
                "workspace": str(workspace),
            }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--catalog-name", default="flaggems-v5")
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--definition", default="abl_t1_gelu")
    parser.add_argument("--hardware", default="A100")
    parser.add_argument(
        "--no-dps",
        action="store_true",
        help="Use a value-returning run() instead of destination-passing style.",
    )
    parser.add_argument("--level", choices=("metrics", "source", "instruction"), default="instruction")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--stop-after-eval",
        action="store_true",
        help="Leave a passing round pending so a native profile subagent can finish it.",
    )
    args = parser.parse_args()

    workspace = args.workspace or Path(tempfile.mkdtemp(prefix="kernelgen-mcp-cuda-e2e-"))
    workspace = workspace.resolve()
    if args.workspace and workspace.exists() and any(workspace.iterdir()):
        raise SystemExit(f"workspace must be absent or empty: {workspace}")
    (workspace / "tmp").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.kernel.resolve(), workspace / "tmp" / "main.py")
    ToolContext(
        definition=args.definition,
        target_hardware=args.hardware,
        eval_server_url=args.server,
        catalog_name=args.catalog_name,
        destination_passing_style=not args.no_dps,
        profile_enabled=True,
    ).write(workspace)
    result = asyncio.run(_run(args, workspace))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

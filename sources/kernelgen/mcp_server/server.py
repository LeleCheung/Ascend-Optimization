"""One stdio MCP server exposing KernelGen's deterministic agent tools."""

from __future__ import annotations

from typing import Any, Literal

from kernelgen.data.experiment_plan import ExperimentPlan
from kernelgen.data.profile_analysis import ProfileAnalysis, ProfileFinding
from kernelgen.data.round_conclusion import RoundConclusion
from kernelgen.data.tool_context import load_tool_context, workspace_from_env
from kernelgen.mcp_server.contract import MCP_SERVER_NAME
from kernelgen.mcp_server.knowledge_tools import (
    get_knowledge as get_knowledge_impl,
    get_source as get_source_impl,
    query_knowledge as query_knowledge_impl,
    query_sources as query_sources_impl,
    search_rounds as search_rounds_impl,
)
from kernelgen.mcp_server.stdio import KernelGenFastMCP
from kernelgen.tools.eval_round import evaluate_round as evaluate_round_impl
from kernelgen.tools.eval_only import evaluate_only as evaluate_only_impl
from kernelgen.tools.debug_job import (
    WorkspaceDebugJobRequest,
    submit_workspace_debug_job,
)
from kernelgen.tools.kernelgen_server_adapter import (
    get_service_status as get_service_status_impl,
)
from kernelgen.tools.preflight import preflight_candidate
from kernelgen.tools.finalize_round import finalize_round as finalize_round_impl
from kernelgen.tools.retest import request_retest as request_retest_impl
from kernelgen.tools.profile_round import (
    get_workspace_profile_context as get_profile_context_impl,
    profile_workspace_workloads as profile_workloads_impl,
    record_profile_analysis as record_profile_analysis_impl,
)


mcp = KernelGenFastMCP(
    MCP_SERVER_NAME,
    instructions=(
        "Authoritative tools for one isolated KernelGen workspace. "
        "At the beginning of each Coder session, call get_server_status once "
        "and inspect the actual target and software environment. "
        "After editing a kernel, call preflight_kernel for target compile/smoke "
        "and fix every failure. "
        "SUSPECTED_DEVICE_ERROR is not proof of a broken device; inspect "
        "get_server_status.scheduler and treat only broken>0 as a failed probe. "
        "Use Debug Jobs only for bounded target-side diagnostics when preflight "
        "or evaluation evidence is insufficient; they never replace formal evaluation. "
        "Only submit a frozen experiment plan with eval_round after preflight passes. "
        "When a new best returns profile_required=true, delegate to "
        "kernel-profile-analyzer when profiler evidence would guide the next experiment. "
        "Pending profile analysis is advisory and does not block finalization. "
        "Call finalize_round exactly once with the post-measurement conclusion. "
        "If transport loses the successful response, repeating the identical "
        "conclusion safely replays the persisted result. "
        "finalize_round atomically returns and persists the CONTINUE/STOP verdict; "
        "obey it before editing or evaluating another candidate. "
        "Source paths returned by query_sources are logical identifiers; use "
        "get_source, never filesystem Read, to retrieve their content. "
        "Source package filters use canonical source:<id> values; omit the "
        "filter when the exact ID is unknown. Always inspect knowledge payloads: "
        "status: ERROR is an operation failure even when MCP transport succeeded."
    ),
)


@mcp.tool(name="get_server_status")
def get_server_status() -> dict[str, Any]:
    """Return the configured KernelGen Server's actual environment and capabilities.

    The Server URL is workspace-owned ToolContext state and cannot be supplied
    or redirected by the Agent. The response is the authoritative ``/status``
    payload, including target, software, devices, timing, profile, Debug Job
    capabilities, and scheduler health. Only ``scheduler.broken > 0`` confirms
    a slot whose independent device probe failed.
    """

    workspace = workspace_from_env()
    try:
        context = load_tool_context(workspace)
        return get_service_status_impl(context.eval_server_url)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "reason": "SERVER_STATUS_FAILED",
            "error": str(exc),
        }


@mcp.tool(name="preflight_kernel")
def preflight_kernel(kernel_path: str = "tmp/main.py") -> dict[str, Any]:
    """Check one exact candidate before it is eligible for formal evaluation.

    KernelGen Server validates the backend and compile/smoke launches every
    workload specialization. Failures are diagnostic attempts and do not consume
    an optimization round. ``SUSPECTED_DEVICE_ERROR`` means the request failed
    on two slots; it does not prove either slot is broken. Inspect
    ``get_server_status.scheduler`` before deciding whether operator recovery or
    a lower-concurrency/larger-timeout rerun is appropriate.
    """
    return preflight_candidate(workspace_from_env(), kernel_path)


@mcp.tool(name="submit_debug_job")
def submit_debug_job(request: WorkspaceDebugJobRequest) -> dict[str, Any]:
    """Run a bounded diagnostic script using files under this workspace's tmp/.

    The evaluation endpoint and target configuration come from the server-owned
    tool context. A Debug Job does not create a ledger round and cannot replace
    formal preflight or evaluation. This call waits for a terminal Server status
    and returns the logs and downloaded artifacts directly.
    """

    workspace = workspace_from_env()
    try:
        return submit_workspace_debug_job(
            workspace,
            load_tool_context(workspace),
            request,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "reason": "DEBUG_JOB_SUBMIT_FAILED",
            "error": str(exc),
        }


@mcp.tool(name="eval_round")
def eval_round(
    experiment_plan: ExperimentPlan,
    kernel_path: str = "tmp/main.py",
    confirm_no_knowledge_applied: bool = False,
) -> dict[str, Any]:
    """Evaluate a kernel on the configured target and record the round.

    Args:
        experiment_plan: Focused hypothesis and expected effect, frozen before measurement.
        kernel_path: Kernel source path relative to the current agent workspace.
        confirm_no_knowledge_applied: Confirm that detail-read knowledge did not
            change the submitted solution when knowledge_uses is empty.

    Returns authoritative status, performance, round_num, and is_new_best.
    ``SUSPECTED_DEVICE_ERROR`` is recorded but is not candidate evidence and is
    not proof of a broken slot; inspect ``get_server_status.scheduler`` and
    treat only ``broken > 0`` as a failed device probe. Never compute or replace
    measurements yourself.
    """
    return evaluate_round_impl(
        workspace_from_env(),
        kernel_path,
        experiment_plan,
        confirm_no_knowledge_applied=confirm_no_knowledge_applied,
    )


@mcp.tool(name="eval_only")
def eval_only(
    definition: str,
    kernel_path: str = "tmp/main.py",
    catalog_name: str = "flaggems-adapter-definitions",
    target_hardware: str = "",
    warmup_ms: int = 1000,
    benchmark_ms: int = 100,
    num_trials: int = 1,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Stateless verification eval — no preflight, no ledger, no round recorded.

    Submits ``kernel_path`` against ``definition`` loaded from the built-in
    KernelGen Server Catalog and returns the authoritative flat result
    (``status``, ``geo_mean``,
    ``num_passed``/``num_workloads``, and ``log`` on failure). Use this to
    verify one candidate against the selected Catalog binding. A PASS proves the
    adapter's complete correctness/performance case set accepted that candidate. A
    ``SUSPECTED_DEVICE_ERROR`` result means two slot attempts failed, not that
    either slot is confirmed broken.

    Args:
        definition: definition name to evaluate (must exist in the Catalog).
        kernel_path: solution source path, relative to the agent workspace.
        catalog_name: Server-owned built-in Catalog name.
        target_hardware: e.g. "Ascend910B" (falls back to
            $KERNELGEN_TARGET_HARDWARE).
    """
    return evaluate_only_impl(
        kernel_path=kernel_path,
        definition=definition,
        catalog_name=catalog_name,
        target_hardware=target_hardware,
        warmup_ms=warmup_ms,
        benchmark_ms=benchmark_ms,
        num_trials=num_trials,
        timeout_seconds=timeout_seconds,
    )


@mcp.tool(name="query_knowledge")
def query_knowledge(
    phase: Literal[
        "initial",
        "post_error",
        "post_evaluation",
        "post_profile",
        "plateau",
    ],
    task: Literal[
        "constraint_check",
        "architecture_selection",
        "implementation",
        "diagnosis",
        "next_experiment",
        "portability_analysis",
    ],
    question: str,
    round_num: int | None = None,
    draft_findings: list[ProfileFinding] | None = None,
) -> dict[str, Any]:
    """Return applicable knowledge from the live catalog.

    ``round_num`` and ``draft_findings`` improve ranking when available, but no
    phase requires them. Inspect ``status`` because an ERROR payload is an
    operation failure even when MCP transport succeeded.
    """
    try:
        return query_knowledge_impl(
            workspace_from_env(),
            phase=phase,
            task=task,
            question=question,
            round_num=round_num,
            draft_findings=draft_findings,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "reason": "KNOWLEDGE_QUERY_FAILED",
            "error": str(exc),
        }


@mcp.tool(name="get_knowledge")
def get_knowledge(
    concept_refs: list[str],
    query_id: str = "",
    detail_level: Literal["summary", "full", "sources"] = "full",
) -> dict[str, Any]:
    """Read exact Concept details; ``query_id`` is optional provenance."""
    try:
        return get_knowledge_impl(
            workspace_from_env(),
            query_id=query_id,
            concept_refs=concept_refs,
            detail_level=detail_level,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "reason": "KNOWLEDGE_READ_FAILED",
            "error": str(exc),
        }


@mcp.tool(name="query_sources")
def query_sources(
    query: str,
    package_ids: list[str] | None = None,
    path_globs: list[str] | None = None,
    include_source_only: bool = False,
    max_results: int = 12,
) -> dict[str, Any]:
    """Search Source packages and return provenance-bearing hits.

    ``package_ids`` uses canonical IDs such as ``source:triton-ascend``. A hit
    supplies a ``source-query:`` ID plus the exact source package and logical
    path for ``get_source``. Empty hits should be retried with a simpler query
    or no package filter; never guess a path. Inspect the payload because
    ``status: ERROR`` is an operation failure even when MCP transport succeeded.
    """
    try:
        return query_sources_impl(
            workspace_from_env(),
            query=query,
            package_ids=package_ids,
            path_globs=path_globs,
            include_source_only=include_source_only,
            max_results=max_results,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "reason": "SOURCE_QUERY_FAILED",
            "error": str(exc),
        }


@mcp.tool(name="get_source")
def get_source(
    source_package: str,
    path: str,
    query_id: str = "",
    line_start: int = 1,
    line_end: int | None = None,
) -> dict[str, Any]:
    """Read a bounded range from one logical Source path.

    Copy ``query_id``, ``source_package``, and ``path`` from the same
    ``query_sources`` hit. When supplied, query IDs start with
    ``source-query:``; source packages use canonical ``source:<id>`` form.
    Inspect the payload because ``status: ERROR`` means the read failed even
    when MCP transport succeeded.
    """
    try:
        return get_source_impl(
            workspace_from_env(),
            query_id=query_id,
            source_package=source_package,
            path=path,
            line_start=line_start,
            line_end=line_end,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "reason": "SOURCE_READ_FAILED",
            "error": str(exc),
        }


@mcp.tool(name="search_rounds")
def search_rounds(
    query: str,
    scope: Literal["exact", "definition"] = "exact",
    max_results: int = 12,
) -> dict[str, Any]:
    """Search archived experiments; returned rounds are not KB Concepts."""
    try:
        return search_rounds_impl(
            workspace_from_env(),
            query=query,
            scope=scope,
            max_results=max_results,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "reason": "ROUND_SEARCH_FAILED",
            "error": str(exc),
        }


@mcp.tool(name="request_retest")
def request_retest(round_num: int, reason: str, evidence_workload_uuids: list[str]) -> dict[str, Any]:
    """Remeasure a passing immutable round, including correctness and BOTH timings.

    Supply the observed reason and existing workload UUIDs as evidence anchors.
    All frozen workloads run; these UUIDs do not select or filter measurements.
    No search round is added. At most two Agent requests per candidate SHA and frozen contract.
    Never supply timings, change settings, select favorable repeats or modify the ledger.
    The workflow independently verifies the final best before accepting its output.
    """
    return request_retest_impl(workspace_from_env(), round_num, reason, evidence_workload_uuids)


@mcp.tool(name="finalize_round")
def finalize_round(conclusion: RoundConclusion) -> dict[str, Any]:
    """Attach the conclusion and atomically finalize an existing measured round.

    Strategy, code changes, and expected effect were frozen in the eval plan.
    This records whether that expectation held, the Agent's evidence-based
    assessment of every applied knowledge item, and then returns and persists
    the authoritative CONTINUE/STOP verdict for the optimization loop.
    """
    return finalize_round_impl(
        workspace_from_env(),
        conclusion.model_dump(mode="python"),
    )


@mcp.tool(name="get_profile_context")
def get_profile_context(round_num: int) -> dict[str, Any]:
    """Return trusted eval evidence, profiler capabilities, and cached profiles."""
    return get_profile_context_impl(workspace_from_env(), round_num)


@mcp.tool(name="profile_workloads")
def profile_workloads(
    round_num: int,
    workload_uuids: list[str],
    level: Literal["metrics", "source", "instruction"] = "metrics",
    backend_options: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Profile any selected workloads from one immutable passing eval snapshot."""
    return profile_workloads_impl(
        workspace_from_env(),
        round_num,
        workload_uuids,
        level=level,
        backend_options=backend_options,
    )


@mcp.tool(name="record_profile_analysis")
def record_profile_analysis(round_num: int, analysis: ProfileAnalysis) -> dict[str, Any]:
    """Validate and persist a terminal backend-neutral profile analysis."""
    return record_profile_analysis_impl(
        workspace_from_env(),
        round_num,
        analysis.model_dump(mode="json"),
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

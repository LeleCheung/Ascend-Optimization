"""Final-best profiling support for SingleCoderOptimizationWorkflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from kernelgen.framework.run_control import RunCancelled

if TYPE_CHECKING:
    from kernelgen.workflows.optimization.single_coder.workflow import (
        SingleCoderOptimizationInput,
    )


def ensure_best_profile(
    workspace: Path,
    inp: "SingleCoderOptimizationInput",
    runtime,
    ledger,
) -> None:
    """Best-effort profile of the final best round before distillation."""
    from kernelgen.data.ledger import TERMINAL_PROFILE_STATES, Ledger

    if not inp.profile_enabled or ledger.history.best_round <= 0:
        return
    best_round = ledger.history.best_round
    best = ledger.get_round(best_round)
    if best.evaluation.status != "PASSED":
        return
    if best.profile.status in TERMINAL_PROFILE_STATES:
        return
    if (
        not best.profile.required
        or best.profile.status not in {"pending", "collecting"}
    ):
        return

    print(
        f"[profile] final best is R{best_round}; running backend-native analysis",
        flush=True,
    )
    failure = ""
    try:
        _run_profile_analyzer(
            runtime,
            best_round,
            knowledge_enabled=inp.knowledge_enabled,
        )
    except RunCancelled:
        raise
    except Exception as exc:  # profiler failure must not discard the best kernel
        failure = f"{type(exc).__name__}: {exc}"

    refreshed = Ledger(workspace)
    status = refreshed.get_round(best_round).profile.status
    if status in TERMINAL_PROFILE_STATES:
        return
    if not failure:
        failure = "profile analyzer returned without recording a terminal analysis"
    _record_profile_failure(workspace, refreshed, best_round, failure)


def _run_profile_analyzer(
    runtime,
    round_num: int,
    *,
    knowledge_enabled: bool,
) -> None:
    """Invoke the isolated native analyzer for one immutable round snapshot."""
    if not (
        getattr(runtime, "supports_native_agents", False)
        or getattr(runtime, "supports_agent_roles", False)
    ):
        raise RuntimeError(
            "runtime does not support the profile analyzer role"
        )
    runtime.invoke(
        (
            f"Analyze authoritative eval round {round_num}. Use the "
            "backend-native profiler exposed by the service, persist "
            "ProfileAnalysis with "
            "mcp__kernelgen__record_profile_analysis, and return only after "
            "recorded=true."
        ),
        model="inherit",
        agent=(
            "kernel-knowledge-profile-analyzer"
            if knowledge_enabled
            else "kernel-profile-analyzer"
        ),
    )


def _record_profile_failure(
    workspace: Path,
    ledger,
    round_num: int,
    error: str,
) -> None:
    """Persist a terminal failure if the analyzer cannot record one itself."""
    try:
        record = ledger.get_round(round_num)
        snapshot = (workspace / record.solution.snapshot_path).resolve()
        snapshot.relative_to(workspace.resolve())
        identity = json.loads(
            (snapshot / "identity.json").read_text(encoding="utf-8")
        )
        from kernelgen.tools.profile_round import record_profile_analysis

        record_profile_analysis(
            workspace,
            round_num,
            {
                "round_num": round_num,
                "evaluation_fingerprint": (
                    identity.get("evaluation_fingerprint")
                    or record.evaluation.fingerprint
                ),
                "status": "failed",
                "solution_sha256": identity["solution_sha256"],
                "backend": (
                    identity.get("server_backend")
                    or record.evaluation.server_backend
                    or "unknown"
                ),
                "dominant_bound": "unknown",
                "error": error[:2000],
            },
        )
    except Exception as exc:  # finalization remains best-effort by design
        print(
            f"[profile] unable to persist failure for R{round_num}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

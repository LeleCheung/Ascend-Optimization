"""Recovery of completed KernelGen epochs from durable checkpoints."""

from __future__ import annotations

from pathlib import Path

from kernelgen.agents.analyzer import AnalyzerOutput
from kernelgen.agents.coder import CoderReport
from kernelgen.agents.epoch_summary import EpochSummaryOutput
from kernelgen.data.implementation import ImplementationLanguage
from kernelgen.framework.parallel import Directory
from kernelgen.workflows.optimization.kernelgen.contracts import (
    EpochCompletionManifest,
    EpochResult,
    KernelGenInput,
)
from kernelgen.workflows.optimization.kernelgen.epoch import collect_epoch_result, confirmed_workspace_best, select_best_result


def load_completed_epoch(
    *,
    cwd: Path,
    definition_name: str,
    target_hardware: str,
    implementation_language: ImplementationLanguage,
    epoch_num: int,
    expected_agents: int | None = None,
    allow_empty: bool = False,
) -> tuple[list[tuple[CoderReport, str]], Directory]:
    """Reconstruct one completed epoch from its authoritative ledgers."""
    from kernelgen.data.ledger import Ledger

    epoch_dir = cwd / f"{epoch_num}R"
    manifest_path = epoch_dir / "epoch-completion.json"
    manifest = None
    if manifest_path.is_file():
        try:
            manifest = EpochCompletionManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"invalid epoch completion manifest: {manifest_path}"
            ) from exc
        agent_dirs = [
            epoch_dir / name for name in manifest.successful_agents
        ]
        missing = [
            path
            for path in agent_dirs
            if not path.is_dir() or not (path / ".ledger.json").is_file()
        ]
        if missing:
            raise ValueError(
                "epoch completion manifest references missing successful "
                f"agent ledger(s): {', '.join(path.name for path in missing)}"
            )
        if (
            expected_agents is not None
            and set(manifest.attempted_agents)
            != {f"agent{index}" for index in range(expected_agents)}
        ):
            raise ValueError(
                f"{epoch_num}R attempted agents do not match the expected "
                f"agent0..agent{expected_agents - 1} set"
            )
    else:
        agent_dirs = sorted(
            path
            for path in epoch_dir.glob("agent*")
            if path.is_dir() and (path / ".ledger.json").is_file()
        )
    if not agent_dirs:
        if allow_empty:
            return [], Directory(base=epoch_dir)
        raise ValueError(f"no completed agent ledgers found in {epoch_dir}")
    if (
        manifest is None
        and expected_agents is not None
        and len(agent_dirs) != expected_agents
    ):
        raise ValueError(
            f"{epoch_num}R has {len(agent_dirs)} completed agent ledgers; "
            f"expected {expected_agents}"
        )

    workspace = Directory(base=epoch_dir)
    results: list[tuple[CoderReport, str]] = []
    for agent_dir in agent_dirs:
        workspace.allocate(agent_dir.name)
        ledger = Ledger(agent_dir)
        history = ledger.history
        ledger.validate_identity(
            definition_name=definition_name,
            target_hardware=target_hardware,
            implementation_language=implementation_language.value,
        )
        # A manifest records successful *invocations*, including zero-round
        # failure reports. It never overrides an unfinished measured ledger.
        zero_round_failure = (
            manifest is not None
            and not history.rounds
            and history.best_round == 0
            and history.best_geo_mean <= 0
            and not history.best_code
        )
        if not ledger.coder_completed and not zero_round_failure:
            raise ValueError(
                f"Coder is not complete in {agent_dir}; resume the original "
                "workspace before finalizing this epoch"
            )
        status, _ = confirmed_workspace_best(agent_dir)
        results.append(
            (
                CoderReport(
                    status=status,
                    summary=(
                        f"Recovered from completed {epoch_num}R ledger."
                    ),
                ),
                agent_dir.name,
            )
        )
    return results, workspace


def load_completed_result(
    *,
    cwd: Path,
    definition_name: str,
    target_hardware: str,
    implementation_language: ImplementationLanguage,
    through_epoch: int,
) -> EpochResult:
    """Rebuild the authoritative best from completed per-agent ledgers."""
    overall = EpochResult(definition_name)
    for epoch_num in range(1, through_epoch + 1):
        results, workspace = load_completed_epoch(
            cwd=cwd,
            definition_name=definition_name,
            target_hardware=target_hardware,
            implementation_language=implementation_language,
            epoch_num=epoch_num,
            allow_empty=True,
        )
        if not results:
            continue
        overall = select_best_result(overall, collect_epoch_result(definition_name, results, workspace))
    return overall


def load_resume_state(
    *,
    cwd: Path,
    inp: KernelGenInput,
):
    """Load durable analysis, prior synthesis, and authoritative best."""
    analysis_path = cwd / "1R" / "shared_analysis" / "analysis.json"
    synthesis_path = (
        cwd
        / f"{inp.start_epoch - 1}R"
        / "synthesis"
        / "synthesis.json"
    )
    if not analysis_path.is_file():
        raise ValueError(
            f"resume analysis checkpoint is missing: {analysis_path}"
        )
    if not synthesis_path.is_file():
        raise ValueError(
            f"resume synthesis checkpoint is missing: {synthesis_path}"
        )

    analysis = AnalyzerOutput.model_validate_json(
        analysis_path.read_text(encoding="utf-8")
    )
    synthesis = EpochSummaryOutput.model_validate_json(
        synthesis_path.read_text(encoding="utf-8")
    )
    best_result = load_completed_result(
        cwd=cwd,
        definition_name=inp.definition.name,
        target_hardware=inp.target_hardware,
        implementation_language=inp.implementation_language,
        through_epoch=inp.start_epoch - 1,
    )
    if not best_result.per_agent:
        raise ValueError("resume checkpoint has no completed agent ledgers")
    return analysis, synthesis, best_result


__all__ = [
    "load_completed_epoch",
    "load_completed_result",
    "load_resume_state",
]

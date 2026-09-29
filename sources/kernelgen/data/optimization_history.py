"""Versioned, phase-separated optimization history for kernel generation."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from kernelgen.data.experiment_plan import ExperimentPlan
from kernelgen.data.ledger_migrations import (
    CURRENT_LEDGER_VERSION,
    LedgerOrigin,
    migrate_ledger,
)
from kernelgen.data.round_conclusion import RoundConclusion


LEDGER_SCHEMA_VERSION = CURRENT_LEDGER_VERSION


def _format_ms(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.6f}"


class HistoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SolutionRecord(HistoryModel):
    sha256: str = Field(min_length=1)
    snapshot_path: str = ""
    candidate_path: str = "tmp/main.py"
    code: str = Field(min_length=1)


class WorkloadMeasurement(HistoryModel):
    uuid: str = Field(min_length=1)
    axes: Dict[str, Any] = Field(default_factory=dict)
    phase: str = ""
    status: str = Field(min_length=1)
    skip_reason: str = ""
    latency_ms: Optional[float] = None
    reference_latency_ms: Optional[float] = None
    speedup: Optional[float] = None
    abs_err: Optional[float] = None
    rel_err: Optional[float] = None


class WorkloadComparison(HistoryModel):
    uuid: str = Field(min_length=1)
    axes: Dict[str, Any] = Field(default_factory=dict)
    before_latency_ms: Optional[float] = None
    after_latency_ms: Optional[float] = None
    latency_delta_ms: Optional[float] = None
    latency_delta_pct: Optional[float] = None
    before_speedup: Optional[float] = None
    after_speedup: Optional[float] = None
    speedup_delta: Optional[float] = None
    speedup_delta_pct: Optional[float] = None


class EvaluationComparison(HistoryModel):
    performance_baseline_round_num: Optional[int] = None
    geo_mean_before: Optional[float] = None
    geo_mean_after: Optional[float] = None
    geo_mean_delta_pct: Optional[float] = None
    workloads: List[WorkloadComparison] = Field(default_factory=list)

    @property
    def baseline_round_num(self) -> Optional[int]:
        """Compatibility accessor for current lifecycle consumers."""
        return self.performance_baseline_round_num


class EvaluationRecord(HistoryModel):
    evaluated_at: Optional[datetime] = None
    api_version: str = ""
    status: str = Field(min_length=1)
    is_hack: bool = False
    hack_reason: str = ""
    geo_mean: Optional[float] = None
    min_speedup: Optional[float] = None
    worst_workload_uuid: str = ""
    latency_ms: Optional[float] = None
    max_abs_error: Optional[float] = None
    max_rel_error: Optional[float] = None
    num_workloads: int = 0
    num_passed: int = 0
    timing_skipped: bool = False
    requested_hardware: str = ""
    server_backend: str = ""
    log: str = ""
    workloads: List[WorkloadMeasurement] = Field(default_factory=list)
    comparison: EvaluationComparison = Field(default_factory=EvaluationComparison)
    fingerprint: str = ""


class ProfileRecord(HistoryModel):
    required: bool = False
    status: Literal[
        "not_required",
        "pending",
        "collecting",
        "completed",
        "inconclusive",
        "unsupported",
        "failed",
    ] = "not_required"
    analysis_path: str = ""
    summary: str = ""


class NextVerdictRecord(HistoryModel):
    """Python-owned decision persisted when a measured round is finalized."""

    should_continue: bool
    code: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class RoundRecord(HistoryModel):
    """One measured round plus its later profile, conclusion, and next verdict."""

    round_num: int = Field(gt=0)
    experiment_parent_round_num: Optional[int] = Field(default=None, gt=0)
    plan: ExperimentPlan
    solution: SolutionRecord
    evaluation: EvaluationRecord
    profile: ProfileRecord = Field(default_factory=ProfileRecord)
    conclusion: Optional[RoundConclusion] = None
    next_verdict: Optional[NextVerdictRecord] = None

    def format_for_prompt(self) -> str:
        evaluation = self.evaluation
        conclusion = self.conclusion
        geo_tag = f"geo_mean={evaluation.geo_mean:.4f}x" if evaluation.geo_mean is not None else "geo_mean=N/A"
        min_tag = f" min={evaluation.min_speedup:.4f}x" if evaluation.min_speedup is not None else ""
        lines = [
            f"Round {self.round_num}: {evaluation.status} | {geo_tag}{min_tag} | worst-workload latency={_format_ms(evaluation.latency_ms)}ms",
            f"  Plan: {self.plan.strategy}",
            f"  Hypothesis: {self.plan.hypothesis}",
            f"  Expected: {self.plan.expected_effect.metric} {self.plan.expected_effect.direction} because {self.plan.expected_effect.mechanism}",
            f"  Code changes: {self.plan.code_changes}",
        ]
        if evaluation.timing_skipped:
            lines.append("  Timing: SKIPPED because correctness did not fully pass")
        if evaluation.is_hack:
            lines.append(f"  Anti-hack: REJECTED | {evaluation.hack_reason}")
        if self.plan.key_params:
            params = ", ".join(f"{key}={value}" for key, value in self.plan.key_params.items())
            lines.append(f"  Key params: {params}")
        if self.profile.status != "not_required":
            profile_text = f"  Profile: {self.profile.status}"
            if self.profile.summary:
                profile_text += f" | {self.profile.summary}"
            lines.append(profile_text)
        comparison = evaluation.comparison
        if comparison.baseline_round_num is not None:
            delta = comparison.geo_mean_delta_pct
            delta_text = "N/A" if delta is None else f"{delta:+.4f}%"
            lines.append(f"  Comparison to best R{comparison.baseline_round_num}: geo_mean delta={delta_text}")
        for workload in evaluation.workloads:
            details = [workload.status]
            if workload.phase:
                details.append(f"phase={workload.phase}")
            if workload.skip_reason:
                details.append(f"reason={workload.skip_reason}")
            if workload.latency_ms is not None:
                details.append(f"latency={_format_ms(workload.latency_ms)}ms")
            if workload.reference_latency_ms is not None:
                details.append(f"reference={_format_ms(workload.reference_latency_ms)}ms")
            if workload.speedup is not None:
                details.append(f"speedup={workload.speedup:.6f}x")
            lines.append(f"  Workload {workload.uuid}: {', '.join(details)}")
        if conclusion:
            lines.append(f"  Expectation: {conclusion.expectation_status}")
            lines.append(f"  Root cause: {conclusion.root_cause}")
            if conclusion.perf_gap_analysis:
                lines.append(f"  Perf gap: {conclusion.perf_gap_analysis}")
            lines.append(f"  Suggestion for next round: {conclusion.next_suggestion}")
            if conclusion.debug_lesson:
                lines.append(f"  Debug lesson: {conclusion.debug_lesson}")
            if conclusion.strategy_evolution:
                lines.append(f"  Strategy evolution: {conclusion.strategy_evolution}")
            lines.append(f"  Optimization level: {conclusion.optimization_level}")
        if self.next_verdict:
            action = "CONTINUE" if self.next_verdict.should_continue else "STOP"
            lines.append(f"  Next verdict: {action} | {self.next_verdict.reason}")
        return "\n".join(lines)


class OptimizationHistory(HistoryModel):
    schema_version: Literal["3.0"] = LEDGER_SCHEMA_VERSION
    definition_name: str
    target_hardware: str = ""
    implementation_language: str = ""
    best_geo_mean: float = 0.0
    best_round: int = 0
    best_code: str = ""
    rounds: List[RoundRecord] = Field(default_factory=list)
    rounds_without_improvement: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        origin: LedgerOrigin | None = None,
    ) -> "OptimizationHistory":
        if not path.exists():
            return cls(definition_name="")
        raw = json.loads(path.read_text(encoding="utf-8"))
        data = migrate_ledger(raw, origin=origin)
        history = cls.model_validate(data)
        # This is derived state. Recompute it so ledgers written by older v2
        # builds do not keep treating failed evaluations as a performance plateau.
        history.rounds_without_improvement = sum(
            1
            for record in history.rounds
            if record.round_num > history.best_round
            and record.evaluation.status == "PASSED"
            and record.evaluation.geo_mean is not None
        )
        return history

    def add_round(self, record: RoundRecord) -> None:
        self.rounds.append(record)
        evaluation = record.evaluation
        if (
            evaluation.status == "PASSED"
            and not evaluation.is_hack
            and evaluation.geo_mean is not None
            and evaluation.geo_mean > self.best_geo_mean
        ):
            self.best_geo_mean = evaluation.geo_mean
            self.best_round = record.round_num
            self.best_code = record.solution.code

    def format_for_prompt(self, max_rounds: int = 20) -> str:
        if not self.rounds:
            return ""
        lines = [
            "=" * 60,
            "OPTIMIZATION HISTORY",
            f"Best geo_mean (all workloads): {self.best_geo_mean:.4f}x (Round {self.best_round})",
            "=" * 60,
            "",
            "Round | Status | geo_mean | Min | Level | Planned change",
            "-" * 80,
        ]
        display_rounds = self.rounds[-max_rounds:]
        for record in display_rounds:
            evaluation = record.evaluation
            conclusion = record.conclusion
            geo = f"{evaluation.geo_mean:.4f}x" if evaluation.geo_mean is not None else "N/A"
            minimum = f"{evaluation.min_speedup:.4f}x" if evaluation.min_speedup is not None else "N/A"
            level = conclusion.optimization_level[:5] if conclusion else "?"
            lines.append(f"R{record.round_num:>3} | {evaluation.status:<16} | {geo:>10} | {minimum:>10} | {level:<5} | {record.plan.strategy[:50]}")
        lines.append("")
        if len(self.rounds) > max_rounds:
            older = self.rounds[:-max_rounds]
            passed = sum(1 for record in older if record.evaluation.status == "PASSED")
            lines.append(f"[Rounds {older[0].round_num}-{older[-1].round_num} summarized: {passed} passed, {len(older) - passed} failed]")
            lines.append("")
        for record in display_rounds:
            lines.append(record.format_for_prompt())
            lines.append("")
        lines.extend(self._takeaways())
        return "\n".join(lines)

    def _takeaways(self) -> List[str]:
        lines = ["-" * 60, "KEY TAKEAWAYS:"]
        passed = [record for record in self.rounds if record.evaluation.status == "PASSED" and record.evaluation.geo_mean is not None]
        if passed:
            best = max(passed, key=lambda record: record.evaluation.geo_mean or 0.0)
            minimum = "N/A" if best.evaluation.min_speedup is None else f"{best.evaluation.min_speedup:.4f}x"
            lines.append(f"  BEST: Round {best.round_num} (geo_mean={best.evaluation.geo_mean:.4f}x, min={minimum}) - {best.plan.strategy}")
        for previous, current in zip(self.rounds, self.rounds[1:]):
            before = previous.evaluation.geo_mean
            after = current.evaluation.geo_mean
            if before is not None and after is not None and after < before:
                explanation = current.conclusion.root_cause if current.conclusion else current.plan.strategy
                lines.append(f"  REGRESSION Round {current.round_num}: {before:.4f}x -> {after:.4f}x - {explanation}")
        failed = [record for record in self.rounds if record.evaluation.status != "PASSED"]
        if failed:
            labels = {record.plan.strategy for record in failed}
            lines.append(f"  FAILED strategies (DO NOT repeat): {'; '.join(sorted(labels))}")
        passed_tail = passed[-4:]
        if len(passed_tail) == 4:
            metrics = [record.evaluation.geo_mean or 0.0 for record in passed_tail]
            tail_best = max(metrics)
            tail_worst = min(metrics)
            variation = (tail_best - tail_worst) / tail_best if tail_best else 0.0
            if variation < 0.10:
                lines.append(f"  PLATEAU: Last 4 passing rounds span {tail_worst:.4f}x to {tail_best:.4f}x (<10%). Prefer a different causal mechanism over parameter-only tuning.")
        conclusions = [record.conclusion for record in self.rounds if record.conclusion]
        if conclusions:
            lines.append(f"  SUGGESTED NEXT STEP: {conclusions[-1].next_suggestion}")
        levels = defaultdict(int)
        for conclusion in conclusions:
            levels[conclusion.optimization_level] += 1
        if levels:
            lines.append(f"  LEVEL COVERAGE: {dict(levels)}")
        lines.append("-" * 60)
        return lines

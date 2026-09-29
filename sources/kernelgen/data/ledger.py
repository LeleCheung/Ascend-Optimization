"""Python-owned, worktree-bound ledger for the measured optimization loop."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from kernelgen.data._atomic import atomic_write_json
from kernelgen.data.experiment_plan import ExperimentPlan
from kernelgen.data.optimization_history import (
    EvaluationComparison,
    EvaluationRecord,
    OptimizationHistory,
    NextVerdictRecord,
    ProfileRecord,
    RoundRecord,
    SolutionRecord,
    WorkloadComparison,
    WorkloadMeasurement,
)
from kernelgen.data.round_conclusion import RoundConclusion
from kernelgen.data.stop_policy import StopConfig, Verdict, next_verdict


LEDGER_FILENAME = ".ledger.json"
BEST_KERNEL_FILENAME = ".best_kernel.py"
STOP_CONFIG_FILENAME = ".stop_config.json"
TERMINAL_PROFILE_STATES = {
    "not_required",
    "completed",
    "inconclusive",
    "unsupported",
    "failed",
}


@dataclass
class EvalRecord:
    round_num: int
    status: str
    geo_mean: Optional[float]
    is_new_best: bool
    best_geo_mean: float
    profile_required: bool
    profile_status: str
    conclusion_recorded: bool


@dataclass(frozen=True)
class LifecycleBlocker:
    """One authoritative reason why another candidate cannot advance."""

    kind: Literal["conclusion", "stopped"]
    record: RoundRecord


def _delta_pct(after: Optional[float], before: Optional[float]) -> Optional[float]:
    if after is None or before in (None, 0):
        return None
    return (after - before) / before * 100.0


def _workload_measurements(per_workload: List[dict]) -> List[WorkloadMeasurement]:
    measurements = []
    for index, raw in enumerate(per_workload or []):
        measurements.append(
            WorkloadMeasurement(
                uuid=str(raw.get("uuid") or f"wl_{index}"),
                axes=dict(raw.get("axes") or {}),
                phase=str(raw.get("phase") or ""),
                status=str(raw.get("status") or "UNKNOWN"),
                skip_reason=str(raw.get("skip_reason") or ""),
                latency_ms=raw.get("latency_ms"),
                reference_latency_ms=raw.get("reference_latency_ms"),
                speedup=raw.get("speedup"),
                abs_err=raw.get("abs_err"),
                rel_err=raw.get("rel_err"),
            )
        )
    return measurements


def _comparison(
    current_geo_mean: Optional[float],
    current_workloads: List[WorkloadMeasurement],
    baseline: Optional[RoundRecord],
) -> EvaluationComparison:
    if baseline is None:
        return EvaluationComparison(geo_mean_after=current_geo_mean)
    baseline_workloads = {item.uuid: item for item in baseline.evaluation.workloads}
    comparisons = []
    for current in current_workloads:
        before = baseline_workloads.get(current.uuid)
        before_latency = before.latency_ms if before else None
        before_speedup = before.speedup if before else None
        comparisons.append(
            WorkloadComparison(
                uuid=current.uuid,
                axes=current.axes,
                before_latency_ms=before_latency,
                after_latency_ms=current.latency_ms,
                latency_delta_ms=(
                    current.latency_ms - before_latency
                    if current.latency_ms is not None and before_latency is not None
                    else None
                ),
                latency_delta_pct=_delta_pct(current.latency_ms, before_latency),
                before_speedup=before_speedup,
                after_speedup=current.speedup,
                speedup_delta=(
                    current.speedup - before_speedup
                    if current.speedup is not None and before_speedup is not None
                    else None
                ),
                speedup_delta_pct=_delta_pct(current.speedup, before_speedup),
            )
        )
    before_geo = baseline.evaluation.geo_mean
    return EvaluationComparison(
        performance_baseline_round_num=baseline.round_num,
        geo_mean_before=before_geo,
        geo_mean_after=current_geo_mean,
        geo_mean_delta_pct=_delta_pct(current_geo_mean, before_geo),
        workloads=comparisons,
    )


class Ledger:
    """Authoritative state for one isolated agent workspace."""

    def __init__(self, worktree_dir: os.PathLike | str):
        self.dir = Path(worktree_dir)
        self.path = self.dir / LEDGER_FILENAME
        self.best_kernel_path = self.dir / BEST_KERNEL_FILENAME
        self._stop_config_path = self.dir / STOP_CONFIG_FILENAME
        self.history = OptimizationHistory.load(self.path)
        self._stop_config = self._load_stop_config()

    def set_stop_config(self, cfg: StopConfig) -> None:
        atomic_write_json(self._stop_config_path, asdict(cfg))
        self._stop_config = cfg

    def _load_stop_config(self) -> Optional[StopConfig]:
        try:
            return StopConfig(**json.loads(self._stop_config_path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid stop config: {self._stop_config_path}") from exc

    def validate_experiment_plan(self, plan: ExperimentPlan | Dict[str, Any]) -> ExperimentPlan:
        normalized = ExperimentPlan.model_validate(
            plan
        ).validate_for_submission()
        pending_conclusion = self.pending_conclusion_round()
        if pending_conclusion is not None:
            raise ValueError(
                f"round {pending_conclusion.round_num} requires finalize_round"
            )
        if not self.history.rounds and normalized.kind != "baseline":
            raise ValueError("the first measured round must use plan.kind='baseline'")
        if self.history.rounds and normalized.kind == "baseline":
            raise ValueError("plan.kind='baseline' is only valid for the first measured round")
        if normalized.source.origin == "profile_next_experiment":
            source_round = self.get_round(
                normalized.source.parent_round_num or 0
            )
            if source_round.profile.status not in TERMINAL_PROFILE_STATES - {"not_required"}:
                raise ValueError("plan source profile analysis is not terminal")
            if not source_round.profile.analysis_path:
                raise ValueError(
                    "plan source round has no recorded profile analysis"
                )
            analysis_path = (
                self.dir / source_round.profile.analysis_path
            ).resolve()
            try:
                analysis_path.relative_to(self.dir.resolve())
            except ValueError as exc:
                raise ValueError("plan source analysis_path is outside the workspace") from exc
            if not analysis_path.is_file():
                raise ValueError("plan source profile analysis file is missing")
            try:
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("plan source profile analysis is unreadable") from exc
            if not analysis.get("next_experiment"):
                raise ValueError("profile_next_experiment source has no recorded next_experiment")
        return normalized

    def record_eval(
        self,
        eval_result: Dict[str, Any],
        solution_code: str,
        experiment_plan: ExperimentPlan | Dict[str, Any],
        *,
        profile_enabled: bool = False,
        definition_name: str = "",
        target_hardware: str = "",
        implementation_language: str = "triton",
        candidate_path: str = "tmp/main.py",
        evaluated_at: Optional[datetime] = None,
    ) -> EvalRecord:
        plan = self.validate_experiment_plan(experiment_plan)
        candidate = Path(candidate_path or "tmp/main.py")
        if candidate.is_absolute():
            candidate = candidate.resolve().relative_to(self.dir.resolve())
        else:
            resolved = (self.dir / candidate).resolve()
            resolved.relative_to(self.dir.resolve())
            candidate = resolved.relative_to(self.dir.resolve())
        for field_name, value in (
            ("definition_name", definition_name),
            ("target_hardware", target_hardware),
            ("implementation_language", implementation_language),
        ):
            value = str(value or "").strip()
            current = getattr(self.history, field_name)
            if current and value and current != value:
                raise ValueError(f"ledger {field_name} {current!r} does not match {value!r}")
            if value and not current:
                setattr(self.history, field_name, value)

        previous_best = self.history.best_geo_mean
        baseline = next(
            (
                record
                for record in self.history.rounds
                if record.round_num == self.history.best_round
            ),
            None,
        )
        experiment_parent = (
            baseline
            or (self.history.rounds[-1] if self.history.rounds else None)
        )
        round_num = len(self.history.rounds) + 1
        status = str(eval_result.get("status") or "UNKNOWN")
        geo_mean = eval_result.get("geo_mean")
        workloads = _workload_measurements(eval_result.get("per_workload", []))
        is_new_best = (
            status == "PASSED"
            and not bool(eval_result.get("is_hack", False))
            and geo_mean is not None
            and geo_mean > previous_best
        )
        profile_required = profile_enabled and is_new_best
        record = RoundRecord(
            round_num=round_num,
            experiment_parent_round_num=(
                experiment_parent.round_num
                if experiment_parent is not None
                else None
            ),
            plan=plan,
            solution=SolutionRecord(
                sha256=hashlib.sha256(solution_code.encode("utf-8")).hexdigest(),
                candidate_path=str(candidate),
                code=solution_code,
            ),
            evaluation=EvaluationRecord(
                evaluated_at=evaluated_at or datetime.now(timezone.utc),
                api_version=str(eval_result.get("api_version") or ""),
                status=status,
                is_hack=bool(eval_result.get("is_hack", False)),
                hack_reason=str(eval_result.get("hack_reason") or ""),
                geo_mean=geo_mean,
                min_speedup=eval_result.get("min_speedup"),
                worst_workload_uuid=str(eval_result.get("worst_workload_uuid") or ""),
                latency_ms=eval_result.get("latency_ms"),
                max_abs_error=eval_result.get("abs_err"),
                max_rel_error=eval_result.get("rel_err"),
                num_workloads=int(eval_result.get("num_workloads") or len(workloads)),
                num_passed=int(eval_result.get("num_passed") or 0),
                timing_skipped=bool(eval_result.get("timing_skipped", False)),
                requested_hardware=str(eval_result.get("requested_hardware") or ""),
                server_backend=str(eval_result.get("server_backend") or ""),
                log=str(eval_result.get("log") or ""),
                workloads=workloads,
                comparison=_comparison(geo_mean, workloads, baseline),
            ),
            profile=ProfileRecord(
                required=profile_required,
                status="pending" if profile_required else "not_required",
            ),
        )
        self.history.add_round(record)
        if is_new_best:
            self.history.rounds_without_improvement = 0
            self._write_best_kernel(solution_code)
        elif status == "PASSED" and geo_mean is not None:
            self.history.rounds_without_improvement += 1
        self._atomic_save()
        return EvalRecord(
            round_num=round_num,
            status=status,
            geo_mean=geo_mean,
            is_new_best=is_new_best,
            best_geo_mean=self.history.best_geo_mean,
            profile_required=record.profile.required,
            profile_status=record.profile.status,
            conclusion_recorded=record.conclusion is not None,
        )

    def attach_evaluation_snapshot(
        self,
        round_num: int,
        *,
        evaluation_fingerprint: str,
        snapshot_path: str,
    ) -> Dict[str, Any]:
        record = self.get_round(round_num)
        record.evaluation.fingerprint = evaluation_fingerprint
        record.solution.snapshot_path = snapshot_path
        reused = None
        if record.profile.status == "pending":
            reused = next(
                (
                    prior
                    for prior in self.history.rounds
                    if prior.round_num != round_num
                    and prior.evaluation.fingerprint == evaluation_fingerprint
                    and prior.profile.status in TERMINAL_PROFILE_STATES - {"not_required"}
                ),
                None,
            )
        if reused is not None:
            record.profile = ProfileRecord(
                required=False,
                status=reused.profile.status,
                analysis_path=reused.profile.analysis_path,
                summary=reused.profile.summary,
            )
        self._atomic_save()
        return {
            "profile_reused": reused is not None,
            "profile_required": record.profile.required,
            "profile_status": record.profile.status,
            "profile_analysis_path": record.profile.analysis_path,
        }

    def mark_profile_collecting(self, round_num: int) -> None:
        record = self.get_round(round_num)
        if not record.profile.required or record.profile.status not in {"pending", "collecting"}:
            raise ValueError(f"round {round_num} cannot collect profile data from state {record.profile.status!r}")
        record.profile.status = "collecting"
        self._atomic_save()

    def attach_profile_analysis(
        self,
        round_num: int,
        *,
        status: str,
        analysis_path: str,
        summary: str,
    ) -> None:
        if status not in TERMINAL_PROFILE_STATES - {"not_required"}:
            raise ValueError(f"invalid terminal profile status: {status!r}")
        record = self.get_round(round_num)
        if not record.profile.required or record.profile.status not in {"pending", "collecting"}:
            raise ValueError(f"round {round_num} cannot record profile analysis from state {record.profile.status!r}")
        record.profile.status = status
        record.profile.analysis_path = analysis_path
        record.profile.summary = summary
        self._atomic_save()

    def _validate_conclusion(
        self,
        round_num: int,
        conclusion: RoundConclusion | Dict[str, Any],
    ) -> tuple[RoundRecord, RoundConclusion]:
        record = self.get_round(round_num)
        if record.conclusion is not None:
            raise ValueError(f"round {round_num} conclusion is already recorded")
        normalized = RoundConclusion.model_validate(conclusion)
        if normalized.round_num != round_num:
            raise ValueError("conclusion round_num does not match the target round")
        expected = {
            item.reference_key()
            for item in record.plan.knowledge_uses
            if item.disposition in {"adopted", "adapted"}
        }
        actual = {
            item.reference_key() for item in normalized.knowledge_assessments
        }
        if actual != expected:
            raise ValueError(
                "conclusion knowledge_assessments must match applied "
                "plan knowledge_uses exactly"
            )
        if record.evaluation.status != "SUSPECTED_DEVICE_ERROR":
            if record.plan.kind == "baseline":
                if normalized.expectation_status != "baseline":
                    raise ValueError(
                        "baseline round requires expectation_status='baseline'"
                    )
            else:
                if normalized.expectation_status == "baseline":
                    raise ValueError(
                        "non-baseline round cannot use expectation_status='baseline'"
                    )
                if not normalized.perf_gap_analysis:
                    raise ValueError(
                        "non-baseline round requires non-empty perf_gap_analysis"
                    )
        return record, normalized

    @staticmethod
    def _verdict_record(verdict: Verdict) -> NextVerdictRecord:
        return NextVerdictRecord(
            should_continue=verdict.should_continue,
            code=verdict.code or ("continue" if verdict.should_continue else "stop"),
            reason=verdict.reason,
        )

    def finalize_round(
        self,
        round_num: int,
        conclusion: RoundConclusion | Dict[str, Any],
        cfg: Optional[StopConfig] = None,
    ) -> NextVerdictRecord:
        """Atomically persist the conclusion and authoritative stop verdict."""
        record, normalized = self._validate_conclusion(round_num, conclusion)
        if record is not self.history.rounds[-1]:
            raise ValueError("only the latest round can be finalized")

        record.conclusion = normalized
        if record.evaluation.status == "SUSPECTED_DEVICE_ERROR":
            verdict = Verdict(
                should_continue=False,
                code="suspected_device_error",
                reason=(
                    "evaluation failed on two different device slots; stop this "
                    "optimization run for operator review"
                ),
            )
        else:
            verdict = next_verdict(self.snapshot(), cfg or self.stop_config)
        decision = self._verdict_record(verdict)
        record.next_verdict = decision
        self._atomic_save()
        return decision

    def record_supervisor_stop(
        self,
        round_num: int,
        *,
        code: str,
        reason: str,
    ) -> NextVerdictRecord:
        """Replace a latest CONTINUE verdict with a Python-owned hard stop."""
        record = self.get_round(round_num)
        if record is not self.history.rounds[-1]:
            raise ValueError("only the latest round can receive a supervisor stop")
        if record.conclusion is None or record.next_verdict is None:
            raise ValueError("supervisor stop requires a finalized round")
        if not record.next_verdict.should_continue:
            return record.next_verdict
        decision = NextVerdictRecord(
            should_continue=False,
            code=code,
            reason=reason,
        )
        record.next_verdict = decision
        self._atomic_save()
        return decision

    def clear_supervisor_stop(
        self,
        *,
        code: str = "user_cancelled",
    ) -> NextVerdictRecord:
        """Restore the policy verdict hidden by a resumable supervisor stop.

        Only the latest finalized round can be changed, and only when its stop
        code exactly matches the caller-provided resumable code.  Measured data,
        conclusions, and genuine policy stops remain immutable.
        """
        if not self.history.rounds:
            raise ValueError("ledger has no finalized round to resume")
        record = self.history.rounds[-1]
        if record.conclusion is None or record.next_verdict is None:
            raise ValueError("latest round is not finalized")
        if record.next_verdict.code != code:
            raise ValueError(
                f"latest round stop is not resumable: {record.next_verdict.code}"
            )
        verdict = next_verdict(self.snapshot(), self.stop_config)
        if not verdict.should_continue:
            raise ValueError(
                f"run remains terminal under the stop policy: {verdict.reason}"
            )
        decision = self._verdict_record(verdict)
        record.next_verdict = decision
        self._atomic_save()
        return decision

    def transition_candidate(self, round_num: int) -> Dict[str, Any]:
        """Apply the Python-owned KEEP/REVERT/REPAIR transition after finalization."""
        record = self.get_round(round_num)
        candidate_raw = record.solution.candidate_path or "tmp/main.py"
        candidate_path = (self.dir / candidate_raw).resolve()
        candidate_path.relative_to(self.dir.resolve())

        if self.history.best_round == round_num:
            action = "KEEP"
            candidate_code = record.solution.code
        elif self.history.best_round > 0 and self.history.best_code:
            action = "REVERT"
            candidate_code = self.history.best_code
        else:
            action = "REPAIR"
            candidate_code = record.solution.code

        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=str(candidate_path.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(candidate_code)
            os.replace(temporary, candidate_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

        return {
            "candidate_action": action,
            "candidate_path": str(candidate_path.relative_to(self.dir.resolve())),
            "best_round": self.history.best_round,
        }

    def update_best(self, geo_mean: float, code: str, round_num: int = 0) -> None:
        if geo_mean > self.history.best_geo_mean:
            self.history.best_geo_mean = geo_mean
            self.history.best_round = round_num
            self.history.best_code = code
            if code:
                self._write_best_kernel(code)
            self._atomic_save()

    def snapshot(self) -> Dict[str, Any]:
        history = self.history
        last = history.rounds[-1] if history.rounds else None
        current_vs_best = ""
        if last and history.best_geo_mean > 0 and last.evaluation.geo_mean is not None:
            current_vs_best = f"{(last.evaluation.geo_mean - history.best_geo_mean) / history.best_geo_mean * 100.0:+.4f}%"
        levels = sorted(
            {
                record.conclusion.optimization_level
                for record in history.rounds
                if record.conclusion is not None
            }
        )
        pending_profile = self.pending_profile_round()
        pending_conclusion = self.pending_conclusion_round()
        stopped = self.stopped_round()
        return {
            "schema_version": history.schema_version,
            "best_geo_mean": history.best_geo_mean,
            "best_round": history.best_round,
            "rounds_without_improvement": history.rounds_without_improvement,
            "current_vs_best": current_vs_best,
            "levels_covered": levels,
            "round_count": len(history.rounds),
            "profile_pending_round": pending_profile.round_num if pending_profile else None,
            "conclusion_pending_round": pending_conclusion.round_num if pending_conclusion else None,
            "run_stopped": stopped is not None,
            "stop_reason": (
                stopped.next_verdict.reason
                if stopped is not None and stopped.next_verdict is not None
                else ""
            ),
        }

    def get_round(self, round_num: int) -> RoundRecord:
        for record in self.history.rounds:
            if record.round_num == round_num:
                return record
        raise ValueError(f"no eval round {round_num} in ledger")

    def pending_profile_round(self) -> Optional[RoundRecord]:
        if self.history.best_round <= 0:
            return None
        record = self.get_round(self.history.best_round)
        if (
            record.profile.required
            and record.profile.status in {"pending", "collecting"}
        ):
            return record
        return None

    def pending_conclusion_round(self) -> Optional[RoundRecord]:
        return next(
            (
                record
                for record in self.history.rounds
                if record.conclusion is None
            ),
            None,
        )

    def validate_identity(
        self,
        *,
        definition_name: str,
        target_hardware: str,
        implementation_language: str,
    ) -> None:
        """Measured ledgers must identify the exact definition and target."""
        if not self.history.rounds:
            return
        for field, expected in (
            ("definition_name", definition_name),
            ("target_hardware", target_hardware),
            ("implementation_language", implementation_language),
        ):
            if getattr(self.history, field) != expected:
                raise ValueError(f"ledger {field} mismatch in {self.dir}")

    @property
    def coder_completed(self) -> bool:
        """A measured Coder finishes only at a finalized, non-cancellation STOP."""
        if not self.history.rounds or self.pending_conclusion_round() is not None:
            return False
        verdict = self.history.rounds[-1].next_verdict
        return bool(
            verdict is not None
            and not verdict.should_continue
            and verdict.code != "user_cancelled"
        )

    def lifecycle_blocker(self) -> Optional[LifecycleBlocker]:
        """Return the first unresolved or terminal round lifecycle state."""
        pending_conclusion = self.pending_conclusion_round()
        if pending_conclusion is not None:
            return LifecycleBlocker("conclusion", pending_conclusion)
        stopped = self.stopped_round()
        if stopped is not None:
            return LifecycleBlocker("stopped", stopped)
        return None

    def stopped_round(self) -> Optional[RoundRecord]:
        """Return the terminal round when a persisted next verdict stopped the run."""
        return next(
            (
                record
                for record in reversed(self.history.rounds)
                if record.next_verdict is not None
                and not record.next_verdict.should_continue
            ),
            None,
        )

    @property
    def stop_config(self) -> StopConfig:
        return self._stop_config or StopConfig()

    def should_stop(self, cfg: Optional[StopConfig] = None) -> Verdict:
        return next_verdict(self.snapshot(), cfg or self.stop_config)

    @property
    def best(self) -> Dict[str, Any]:
        """Return one ledger snapshot; the best-kernel file is only an export."""
        return {
            "geo_mean": self.history.best_geo_mean,
            "round": self.history.best_round,
            "code": self.history.best_code,
        }

    def revert_to_best(self, kernel_path: Optional[os.PathLike | str] = None) -> bool:
        code = self.best["code"]
        if not code:
            return False
        target = Path(kernel_path) if kernel_path else self.dir / "kernel.py"
        target.write_text(code, encoding="utf-8")
        return True

    def _write_best_kernel(self, code: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_kernel_path.write_text(code, encoding="utf-8")

    def _atomic_save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.history.model_dump(mode="json"), indent=2, ensure_ascii=False)
        fd, temporary = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

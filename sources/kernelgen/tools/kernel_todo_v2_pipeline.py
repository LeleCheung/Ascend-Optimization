"""Structured state for the Kernel Todo V2 multi-stage optimization pipeline."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "kernelgen.kernel-todo-v2-pipeline/v1"
DEFAULT_THRESHOLD = 0.8


class StageName(str, Enum):
    SIMPLE_OPT = "simple_opt"
    NATIVE_BASELINE = "native_baseline"
    KERNELGEN_NATIVE = "kernelgen_native"
    FINAL_GEMS = "final_gems"


class StageVerdict(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


class PipelineStatus(str, Enum):
    FIRST_STAGE_QUALIFIED = "first_stage_qualified"
    PENDING_SECOND_STAGE = "pending_second_stage"
    SECOND_STAGE_RUNNING = "second_stage_running"
    SECOND_STAGE_QUALIFIED = "second_stage_qualified"
    FAILED = "failed"
    BLOCKED = "blocked"


_STATUS_LABELS = {
    PipelineStatus.FIRST_STAGE_QUALIFIED: "第一阶段达标",
    PipelineStatus.PENDING_SECOND_STAGE: "待二阶段优化",
    PipelineStatus.SECOND_STAGE_RUNNING: "二阶段处理中",
    PipelineStatus.SECOND_STAGE_QUALIFIED: "第二阶段达标",
    PipelineStatus.FAILED: "失败",
    PipelineStatus.BLOCKED: "阻塞",
}
_ROW_OPERATOR = re.compile(r"`([^`]+)`")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    verdict: StageVerdict
    geo_mean: float | None = None
    evidence: list[str] = Field(min_length=1)
    reason: str = ""
    updated_at: str = Field(default_factory=utc_now)

    @field_validator("geo_mean")
    @classmethod
    def _finite_positive_geo_mean(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("geo_mean must be finite and positive")
        return value

    @field_validator("attempt_id")
    @classmethod
    def _non_empty_attempt_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("attempt_id must be non-empty")
        return value

    @field_validator("evidence")
    @classmethod
    def _non_empty_evidence(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("evidence paths must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _validate_verdict_payload(self) -> "StageResult":
        if self.verdict == StageVerdict.PASSED and self.geo_mean is None:
            raise ValueError("passed stage requires geo_mean")
        if self.verdict != StageVerdict.PASSED and not self.reason.strip():
            raise ValueError("failed or blocked stage requires reason")
        return self

    def same_fact(self, other: "StageResult") -> bool:
        return (
            self.attempt_id == other.attempt_id
            and self.verdict == other.verdict
            and self.geo_mean == other.geo_mean
            and self.evidence == other.evidence
            and self.reason == other.reason
        )


class StageHistory(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    attempts: list[StageResult] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_attempt_ids(self) -> "StageHistory":
        identifiers = [attempt.attempt_id for attempt in self.attempts]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("stage attempt_id values must be unique")
        return self

    @property
    def current(self) -> StageResult:
        return self.attempts[-1]

    def append(self, result: StageResult) -> bool:
        existing = next(
            (
                attempt
                for attempt in self.attempts
                if attempt.attempt_id == result.attempt_id
            ),
            None,
        )
        if existing is not None:
            if existing.same_fact(result):
                return False
            raise ValueError(
                f"attempt {result.attempt_id!r} already has different facts"
            )
        self.attempts.append(result)
        return True

    def has_attempt(self, attempt_id: str) -> bool:
        return any(attempt.attempt_id == attempt_id for attempt in self.attempts)


class OperatorPipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source_operator: str
    definition_name: str
    simple_opt: StageHistory
    native_baseline: StageHistory | None = None
    kernelgen_native: StageHistory | None = None
    final_gems: StageHistory | None = None

    @field_validator("source_operator", "definition_name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("operator names must be non-empty")
        return value

    def status(self, threshold: float = DEFAULT_THRESHOLD) -> PipelineStatus:
        if self.final_gems is not None:
            final_gems = self.final_gems.current
            if final_gems.verdict == StageVerdict.BLOCKED:
                return PipelineStatus.BLOCKED
            if final_gems.verdict == StageVerdict.FAILED:
                return PipelineStatus.FAILED
            if final_gems.geo_mean is not None and final_gems.geo_mean >= threshold:
                return PipelineStatus.SECOND_STAGE_QUALIFIED
            return PipelineStatus.FAILED

        for history in (self.kernelgen_native, self.native_baseline):
            if history is None:
                continue
            stage = history.current
            if stage.verdict == StageVerdict.BLOCKED:
                return PipelineStatus.BLOCKED
            if stage.verdict == StageVerdict.FAILED:
                return PipelineStatus.FAILED
            return PipelineStatus.SECOND_STAGE_RUNNING

        simple_opt = self.simple_opt.current
        if simple_opt.verdict == StageVerdict.BLOCKED:
            return PipelineStatus.BLOCKED
        if simple_opt.verdict == StageVerdict.FAILED:
            return PipelineStatus.FAILED
        if simple_opt.geo_mean is not None and simple_opt.geo_mean >= threshold:
            return PipelineStatus.FIRST_STAGE_QUALIFIED
        return PipelineStatus.PENDING_SECOND_STAGE

    def stage(self, name: StageName) -> StageHistory | None:
        return getattr(self, name.value)

    @staticmethod
    def _downstream_names(name: StageName) -> tuple[StageName, ...]:
        order = (
            StageName.SIMPLE_OPT,
            StageName.NATIVE_BASELINE,
            StageName.KERNELGEN_NATIVE,
            StageName.FINAL_GEMS,
        )
        return order[order.index(name) + 1 :]

    def update_stage(
        self,
        name: StageName,
        result: StageResult,
        *,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> bool:
        history = self.stage(name)
        if history is not None:
            if history.has_attempt(result.attempt_id):
                return history.append(result)
            if any(self.stage(later) is not None for later in self._downstream_names(name)):
                raise ValueError(
                    f"{self.source_operator}: cannot append {name.value} after a downstream stage"
                )
            return history.append(result)

        if name == StageName.SIMPLE_OPT:
            raise ValueError("simple_opt can only be set when the operator is created")
        simple_opt = self.simple_opt.current
        if simple_opt.verdict != StageVerdict.PASSED:
            raise ValueError("second-stage results require a passed simple_opt result")
        if name == StageName.NATIVE_BASELINE:
            if simple_opt.geo_mean is None or simple_opt.geo_mean >= threshold:
                raise ValueError("native baseline requires simple_opt geo_mean below threshold")
        elif name == StageName.KERNELGEN_NATIVE:
            if (
                self.native_baseline is None
                or self.native_baseline.current.verdict != StageVerdict.PASSED
            ):
                raise ValueError("kernelgen_native requires a passed native_baseline")
        elif name == StageName.FINAL_GEMS:
            if (
                self.kernelgen_native is None
                or self.kernelgen_native.current.verdict != StageVerdict.PASSED
            ):
                raise ValueError("final_gems requires a passed kernelgen_native result")
        setattr(self, name.value, StageHistory(attempts=[result]))
        return True


class KernelTodoV2PipelineResults(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: str = SCHEMA_VERSION
    chip: str
    threshold: float = DEFAULT_THRESHOLD
    updated_at: str = Field(default_factory=utc_now)
    operators: dict[str, OperatorPipelineResult] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _current_schema(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {value}")
        return value

    @field_validator("chip")
    @classmethod
    def _non_empty_chip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("chip must be non-empty")
        return value

    @field_validator("threshold")
    @classmethod
    def _positive_threshold(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("threshold must be finite and positive")
        return value

    @model_validator(mode="after")
    def _keys_match_operators(self) -> "KernelTodoV2PipelineResults":
        mismatched = [
            key
            for key, result in self.operators.items()
            if key != result.source_operator
        ]
        if mismatched:
            raise ValueError(f"operator keys do not match source_operator: {mismatched}")
        return self

    def upsert_simple_opt(
        self,
        *,
        source_operator: str,
        definition_name: str,
        result: StageResult,
    ) -> bool:
        current = self.operators.get(source_operator)
        if current is not None:
            if current.definition_name != definition_name:
                raise ValueError(
                    f"{source_operator}: definition changed from "
                    f"{current.definition_name} to {definition_name}"
                )
            if current.simple_opt.has_attempt(result.attempt_id):
                return current.simple_opt.append(result)
            if any(
                current.stage(name) is not None
                for name in (
                    StageName.NATIVE_BASELINE,
                    StageName.KERNELGEN_NATIVE,
                    StageName.FINAL_GEMS,
                )
            ):
                raise ValueError(
                    f"{source_operator}: cannot append simple_opt after second stage started"
                )
            changed = current.simple_opt.append(result)
            if changed:
                self.updated_at = utc_now()
            return changed
        self.operators[source_operator] = OperatorPipelineResult(
            source_operator=source_operator,
            definition_name=definition_name,
            simple_opt=StageHistory(attempts=[result]),
        )
        self.updated_at = utc_now()
        return True

    def update_stage(
        self,
        *,
        source_operator: str,
        name: StageName,
        result: StageResult,
    ) -> bool:
        operator = self.operators.get(source_operator)
        if operator is None:
            raise KeyError(f"operator has no simple_opt result: {source_operator}")
        changed = operator.update_stage(name, result, threshold=self.threshold)
        if changed:
            self.updated_at = utc_now()
        return changed

    def status_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in PipelineStatus}
        for result in self.operators.values():
            counts[result.status(self.threshold).value] += 1
        return counts


class PipelineSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    reference_passed: int
    gems_timing_passed: int
    first_stage_qualified: int
    pending_second_stage: int
    second_stage_qualified: int
    failed: int
    blocked: int

    def values(self) -> list[int]:
        return [
            self.total,
            self.reference_passed,
            self.gems_timing_passed,
            self.first_stage_qualified,
            self.pending_second_stage,
            self.second_stage_qualified,
            self.failed,
            self.blocked,
        ]


def markdown_result_rows(lines: list[str]) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in lines:
        if not line.startswith("| `"):
            continue
        columns = [item.strip() for item in line.strip().strip("|").split("|")]
        match = _ROW_OPERATOR.search(columns[0])
        if match and len(columns) >= 9:
            rows[match.group(1)] = columns
    return rows


def _row_speed(columns: list[str]) -> float | None:
    try:
        value = float(columns[4].removesuffix("x"))
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def summarize_markdown_results(
    lines: list[str],
    pipeline: KernelTodoV2PipelineResults | None = None,
) -> PipelineSummary:
    rows = markdown_result_rows(lines)
    threshold = pipeline.threshold if pipeline is not None else DEFAULT_THRESHOLD
    counts = {status: 0 for status in PipelineStatus}
    for source_operator, columns in rows.items():
        record = pipeline.operators.get(source_operator) if pipeline is not None else None
        if record is not None:
            counts[record.status(threshold)] += 1
            continue
        row_status = columns[3]
        speed = _row_speed(columns)
        if row_status == "成功" and speed is not None and speed >= threshold:
            counts[PipelineStatus.FIRST_STAGE_QUALIFIED] += 1
        elif row_status == "待二阶段优化":
            counts[PipelineStatus.PENDING_SECOND_STAGE] += 1
        elif row_status == "二阶段处理中":
            counts[PipelineStatus.SECOND_STAGE_RUNNING] += 1
        elif row_status == "失败":
            counts[PipelineStatus.FAILED] += 1
        elif row_status == "阻塞":
            counts[PipelineStatus.BLOCKED] += 1

    values = list(rows.values())
    return PipelineSummary(
        total=len(values),
        reference_passed=sum(row[1] == "通过" for row in values),
        gems_timing_passed=sum(
            row[1] == "通过" and row[2] == "通过" for row in values
        ),
        first_stage_qualified=counts[PipelineStatus.FIRST_STAGE_QUALIFIED],
        pending_second_stage=(
            counts[PipelineStatus.PENDING_SECOND_STAGE]
            + counts[PipelineStatus.SECOND_STAGE_RUNNING]
        ),
        second_stage_qualified=counts[PipelineStatus.SECOND_STAGE_QUALIFIED],
        failed=counts[PipelineStatus.FAILED],
        blocked=counts[PipelineStatus.BLOCKED],
    )


def refresh_markdown_summary(
    lines: list[str],
    pipeline: KernelTodoV2PipelineResults | None = None,
) -> PipelineSummary:
    summary = summarize_markdown_results(lines, pipeline)
    for index, line in enumerate(lines):
        if not line.startswith("| 总算子数 |"):
            continue
        lines[index] = (
            "| 总算子数 | reference通过数 | Gems 可计时数 | 第一阶段达标数 | "
            "待二阶段优化数 | 第二阶段达标数 | 最终失败数 | 阻塞数 |"
        )
        lines[index + 1] = (
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        lines[index + 2] = "| " + " | ".join(map(str, summary.values())) + " |"
        return summary
    raise RuntimeError("results summary table not found")


def load_pipeline_results(path: Path, *, chip: str) -> KernelTodoV2PipelineResults:
    if not path.is_file():
        return KernelTodoV2PipelineResults(chip=chip)
    value = KernelTodoV2PipelineResults.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if value.chip != chip:
        raise ValueError(f"pipeline chip mismatch: expected {chip}, found {value.chip}")
    return value


def write_pipeline_results(path: Path, value: KernelTodoV2PipelineResults) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _metric(stage: StageHistory | None) -> str:
    if stage is None or stage.current.geo_mean is None:
        return "—"
    return f"{stage.current.geo_mean:.3f}x"


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_second_stage_markdown(value: KernelTodoV2PipelineResults) -> str:
    lines = [
        f"# {value.chip} Kernel Todo V2 二阶段结果",
        "",
        "本页由结构化 pipeline JSON 自动生成，只展示 SimpleOpt 未达到门槛或已经进入 Native KernelGen 的算子。",
        "",
        "| 算子 | SimpleOpt Gems 加速比 | Native baseline 加速比 | KernelGen Native 加速比 | 最终 Gems 加速比 | 最终状态 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    included: Iterable[OperatorPipelineResult] = (
        result
        for _, result in sorted(value.operators.items())
        if result.simple_opt.current.geo_mean is None
        or result.simple_opt.current.geo_mean < value.threshold
        or result.native_baseline is not None
    )
    count = 0
    for result in included:
        count += 1
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_markdown_cell(result.source_operator)}`",
                    _metric(result.simple_opt),
                    _metric(result.native_baseline),
                    _metric(result.kernelgen_native),
                    _metric(result.final_gems),
                    _STATUS_LABELS[result.status(value.threshold)],
                ]
            )
            + " |"
        )
    if count == 0:
        lines.append("| — | — | — | — | — | 暂无二阶段算子 |")
    return "\n".join(lines) + "\n"


def write_second_stage_markdown(path: Path, value: KernelTodoV2PipelineResults) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(render_second_stage_markdown(value), encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "DEFAULT_THRESHOLD",
    "KernelTodoV2PipelineResults",
    "OperatorPipelineResult",
    "PipelineSummary",
    "PipelineStatus",
    "SCHEMA_VERSION",
    "StageName",
    "StageHistory",
    "StageResult",
    "StageVerdict",
    "load_pipeline_results",
    "markdown_result_rows",
    "refresh_markdown_summary",
    "render_second_stage_markdown",
    "summarize_markdown_results",
    "write_pipeline_results",
    "write_second_stage_markdown",
]

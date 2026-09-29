from pathlib import Path

import pytest
from pydantic import ValidationError

from kernelgen.tools.kernel_todo_v2_pipeline import (
    KernelTodoV2PipelineResults,
    PipelineStatus,
    StageName,
    StageResult,
    load_pipeline_results,
    refresh_markdown_summary,
    render_second_stage_markdown,
    write_pipeline_results,
)


def _stage(
    verdict: str = "passed",
    geo_mean: float | None = 0.7,
    *,
    reason: str = "",
    evidence: str = "artifact.json",
    attempt_id: str = "attempt-1",
) -> StageResult:
    return StageResult(
        attempt_id=attempt_id,
        verdict=verdict,
        geo_mean=geo_mean,
        reason=reason,
        evidence=[evidence],
        updated_at="2026-08-30T00:00:00+00:00",
    )


def _pipeline(simple_geo: float = 0.7) -> KernelTodoV2PipelineResults:
    value = KernelTodoV2PipelineResults(chip="muxi")
    value.upsert_simple_opt(
        source_operator="gelu",
        definition_name="gelu",
        result=_stage(geo_mean=simple_geo),
    )
    return value


def test_stage_result_requires_evidence_and_valid_verdict_payload():
    with pytest.raises(ValidationError, match="evidence"):
        StageResult(attempt_id="a", verdict="passed", geo_mean=1.0, evidence=[])
    with pytest.raises(ValidationError, match="requires geo_mean"):
        StageResult(attempt_id="a", verdict="passed", evidence=["result.json"])
    with pytest.raises(ValidationError, match="requires reason"):
        StageResult(attempt_id="a", verdict="failed", evidence=["result.json"])
    with pytest.raises(ValidationError, match="finite and positive"):
        StageResult(
            attempt_id="a",
            verdict="passed",
            geo_mean=float("nan"),
            evidence=["result.json"],
        )


def test_pipeline_derives_first_and_second_stage_statuses():
    qualified = _pipeline(simple_geo=0.9)
    assert qualified.operators["gelu"].status() == PipelineStatus.FIRST_STAGE_QUALIFIED

    value = _pipeline()
    result = value.operators["gelu"]
    assert result.status() == PipelineStatus.PENDING_SECOND_STAGE

    value.update_stage(
        source_operator="gelu",
        name=StageName.NATIVE_BASELINE,
        result=_stage(geo_mean=0.68, evidence="native-baseline.json"),
    )
    assert result.status() == PipelineStatus.SECOND_STAGE_RUNNING

    value.update_stage(
        source_operator="gelu",
        name=StageName.KERNELGEN_NATIVE,
        result=_stage(geo_mean=1.1, evidence="kernelgen.json"),
    )
    assert result.status() == PipelineStatus.SECOND_STAGE_RUNNING

    value.update_stage(
        source_operator="gelu",
        name=StageName.FINAL_GEMS,
        result=_stage(geo_mean=0.95, evidence="final-gems.json"),
    )
    assert result.status() == PipelineStatus.SECOND_STAGE_QUALIFIED
    assert value.status_counts()["second_stage_qualified"] == 1


def test_final_gems_below_threshold_is_terminal_failure():
    value = _pipeline()
    value.update_stage(
        source_operator="gelu",
        name=StageName.NATIVE_BASELINE,
        result=_stage(geo_mean=0.7, evidence="native.json"),
    )
    value.update_stage(
        source_operator="gelu",
        name=StageName.KERNELGEN_NATIVE,
        result=_stage(geo_mean=0.9, evidence="kernelgen.json"),
    )
    value.update_stage(
        source_operator="gelu",
        name=StageName.FINAL_GEMS,
        result=_stage(geo_mean=0.79, evidence="final.json"),
    )
    assert value.operators["gelu"].status() == PipelineStatus.FAILED


def test_stage_dependencies_and_immutable_facts_are_enforced():
    value = _pipeline()
    with pytest.raises(ValueError, match="passed native_baseline"):
        value.update_stage(
            source_operator="gelu",
            name=StageName.KERNELGEN_NATIVE,
            result=_stage(geo_mean=1.0),
        )

    native = _stage(geo_mean=0.7, evidence="native.json")
    assert value.update_stage(
        source_operator="gelu",
        name=StageName.NATIVE_BASELINE,
        result=native,
    )
    assert not value.update_stage(
        source_operator="gelu",
        name=StageName.NATIVE_BASELINE,
        result=native.model_copy(update={"updated_at": "later"}),
    )
    with pytest.raises(ValueError, match="already has different facts"):
        value.update_stage(
            source_operator="gelu",
            name=StageName.NATIVE_BASELINE,
            result=_stage(geo_mean=0.71, evidence="native.json"),
        )

    assert value.update_stage(
        source_operator="gelu",
        name=StageName.NATIVE_BASELINE,
        result=_stage(
            geo_mean=0.71,
            evidence="native-retry.json",
            attempt_id="attempt-2",
        ),
    )
    assert len(value.operators["gelu"].native_baseline.attempts) == 2


def test_custom_threshold_controls_native_entry_gate():
    value = KernelTodoV2PipelineResults(chip="muxi", threshold=1.0)
    value.upsert_simple_opt(
        source_operator="gelu",
        definition_name="gelu",
        result=_stage(geo_mean=0.9),
    )
    value.update_stage(
        source_operator="gelu",
        name=StageName.NATIVE_BASELINE,
        result=_stage(geo_mean=0.91, evidence="native.json"),
    )
    assert value.operators["gelu"].status(value.threshold) == PipelineStatus.SECOND_STAGE_RUNNING


def test_new_attempts_append_without_overwriting_old_evidence():
    value = _pipeline()
    assert value.upsert_simple_opt(
        source_operator="gelu",
        definition_name="gelu",
        result=_stage(
            geo_mean=0.9,
            evidence="simple-retry.json",
            attempt_id="simple-attempt-2",
        ),
    )
    attempts = value.operators["gelu"].simple_opt.attempts
    assert [attempt.geo_mean for attempt in attempts] == [0.7, 0.9]
    assert value.operators["gelu"].status() == PipelineStatus.FIRST_STAGE_QUALIFIED

    second_stage = _pipeline()
    second_stage.update_stage(
        source_operator="gelu",
        name=StageName.NATIVE_BASELINE,
        result=_stage(geo_mean=0.7, evidence="native.json"),
    )
    assert not second_stage.upsert_simple_opt(
        source_operator="gelu",
        definition_name="gelu",
        result=_stage(geo_mean=0.7),
    )
    with pytest.raises(ValueError, match="after second stage started"):
        second_stage.upsert_simple_opt(
            source_operator="gelu",
            definition_name="gelu",
            result=_stage(
                geo_mean=0.9,
                evidence="late-simple-retry.json",
                attempt_id="simple-attempt-2",
            ),
        )


def test_pipeline_json_round_trip_and_chip_guard(tmp_path: Path):
    path = tmp_path / "pipeline.json"
    value = _pipeline()
    write_pipeline_results(path, value)

    loaded = load_pipeline_results(path, chip="muxi")
    assert loaded == value
    with pytest.raises(ValueError, match="chip mismatch"):
        load_pipeline_results(path, chip="huawei")


def test_second_stage_markdown_excludes_first_stage_qualified():
    value = _pipeline()
    value.upsert_simple_opt(
        source_operator="relu",
        definition_name="relu",
        result=_stage(geo_mean=1.2, evidence="relu.json"),
    )

    markdown = render_second_stage_markdown(value)

    assert "`gelu`" in markdown
    assert "0.700x" in markdown
    assert "待二阶段优化" in markdown
    assert "`relu`" not in markdown


def test_markdown_summary_uses_pipeline_stage_authority():
    lines = [
        "| 总算子数 | old | old | old | old | old | old |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| 0 | 0 | 0 | 0 | 0 | 0 | 0 |",
        "| `gelu` | 通过 | 通过 | 成功 | 0.700x | — | — | — | — |",
        "| `relu` | 通过 | 通过 | 成功 | 1.200x | — | — | — | — |",
        "| `blocked` | 通过 | 通过 | 阻塞 | — | — | — | — | — |",
    ]
    pipeline = _pipeline()

    summary = refresh_markdown_summary(lines, pipeline)

    assert summary.values() == [3, 3, 3, 1, 1, 0, 0, 1]
    assert lines[0].startswith("| 总算子数 | reference通过数 | Gems 可计时数 | 第一阶段达标数 |")
    assert lines[2] == "| 3 | 3 | 3 | 1 | 1 | 0 | 0 | 1 |"

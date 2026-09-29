import json
from pathlib import Path

from kernelgen.tools import summarize_batch_simple_opt_campaign as summary
from kernelgen.workflows.optimization.single_coder import (
    KERNEL_OPTIMIZATION_OUTPUT_FILENAME,
)


def _write_ledger(workspace: Path, name: str, best: float) -> Path:
    item = workspace / "definitions" / name
    item.mkdir(parents=True)
    (item / ".ledger.json").write_text(
        json.dumps({
            "best_geo_mean": best,
            "best_round": 2,
            "rounds": [
                {
                    "round_num": 1,
                    "evaluation": {"status": "PASSED"},
                },
                {
                    "round_num": 2,
                    "evaluation": {
                        "status": "PASSED",
                        "latency_ms": 2.0,
                        "workloads": [{
                            "phase": "timing",
                            "latency_ms": 2.0,
                            "reference_latency_ms": 2.0 * best,
                            "speedup": best,
                        }],
                    },
                },
            ],
        }),
        encoding="utf-8",
    )
    return item


def test_collects_final_interrupted_and_running_results(tmp_path):
    completed = tmp_path / "batch_01"
    completed.mkdir()
    (completed / "definitions.txt").write_text(
        "fast_op\ninterrupted_op\n",
        encoding="utf-8",
    )
    fast_item = _write_ledger(
        completed / "workspace",
        "fast_op",
        2.0,
    )
    (fast_item / KERNEL_OPTIMIZATION_OUTPUT_FILENAME).write_text(
        json.dumps({
            "definition_name": "fast_op",
            "status": "PASSED",
            "rounds": 2,
            "best_geo_mean": 2.0,
        }),
        encoding="utf-8",
    )
    _write_ledger(completed / "workspace", "interrupted_op", 1.5)
    (completed / "exit").write_text("143\n", encoding="utf-8")

    running = tmp_path / "batch_02"
    running.mkdir()
    (running / "definitions.txt").write_text("running_op\n", encoding="utf-8")
    _write_ledger(running / "workspace", "running_op", 0.9)

    results = summary.collect_campaign([completed, running])

    assert [result.name for result in results] == [
        "fast_op",
        "interrupted_op",
        "running_op",
    ]
    assert [result.state for result in results] == [
        "PASSED",
        "INTERRUPTED",
        "RUNNING",
    ]
    assert [result.authoritative for result in results] == [
        True,
        False,
        False,
    ]
    assert results[0].reference_latency_ms == 4.0
    assert results[0].candidate_latency_ms == 2.0
    assert results[0].timing_workload_count == 1
    assert summary.summary_counts(results) == {
        "total": 3,
        "finalized": 1,
        "passed": 1,
        "accelerated": 1,
        "qualified": 1,
        "running": 1,
        "interrupted": 1,
    }

    markdown = summary.render_markdown(
        results,
        "2026-08-05T23:00:00+08:00",
        "Demo",
    )
    assert (
        "| `fast_op` | 完成 | 2 | 2 | 4.000000 | 2.000000 | "
        "2.000000x |"
    ) in markdown
    assert "1.500000x（暂定）" in markdown
    assert f"[workspace]({results[0].workspace})" in markdown

    payload = summary.report_json(
        results,
        "2026-08-05T23:00:00+08:00",
        "Demo",
    )
    assert payload["summary"]["total"] == 3
    assert payload["results"][1]["batch_exit_code"] == 143


def test_rejects_duplicate_definitions_across_batches(tmp_path):
    first = tmp_path / "batch_01"
    second = tmp_path / "batch_02"
    for batch in (first, second):
        batch.mkdir()
        (batch / "definitions.txt").write_text("same_op\n", encoding="utf-8")

    try:
        summary.collect_campaign([first, second])
    except ValueError as exc:
        assert "duplicate definition" in str(exc)
    else:
        raise AssertionError("duplicate definition should fail")

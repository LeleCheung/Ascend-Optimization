import json

from kernelgen.tools import summarize_multi_device_batch as summary
from kernelgen.workflows.optimization.single_coder import (
    KERNEL_OPTIMIZATION_OUTPUT_FILENAME,
)


def _write_result(workspace, name, status, best):
    item = workspace / "definitions" / name
    item.mkdir(parents=True)
    (item / KERNEL_OPTIMIZATION_OUTPUT_FILENAME).write_text(
        json.dumps({
            "definition_name": name,
            "status": status,
            "rounds": 3,
            "best_geo_mean": best,
        }),
        encoding="utf-8",
    )


def test_discovers_collected_runs_and_renders_speedup_matrix(tmp_path):
    run_name = "run-1"
    tianshu = tmp_path / "tianshu" / run_name / "workspace"
    hygon = tmp_path / "hygon" / run_name / "workspace"
    _write_result(tianshu, "fast_op", "PASSED", 2.0)
    _write_result(tianshu, "failed_op", "FAILED", None)
    _write_result(hygon, "fast_op", "PASSED", 0.5)

    reports = summary.discover_reports(tmp_path, run_name)
    markdown = summary.render_markdown(reports)
    payload = summary.report_json(reports)

    assert [report.name for report in reports] == ["tianshu", "hygon"]
    assert reports[0].passed == 1
    assert reports[0].failed == 1
    assert reports[0].accelerated == 1
    assert "| `fast_op` | 2x | 0.5x |" in markdown
    assert "| `failed_op` | — | 待测 |" in markdown
    definitions = payload["devices"][0]["definitions"]
    assert {item["name"] for item in definitions} == {"fast_op", "failed_op"}

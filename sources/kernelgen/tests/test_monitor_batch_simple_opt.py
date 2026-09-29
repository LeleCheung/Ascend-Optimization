"""Tests for the read-only BatchSimpleOpt progress monitor."""

import json

from kernelgen.tools import monitor_batch_simple_opt as monitor
from kernelgen.workflows.legacy.batch_simple_opt_definition import (
    BATCH_SIMPLE_OPT_OUTPUT_FILENAME,
)
from kernelgen.workflows.optimization.single_coder import (
    KERNEL_OPTIMIZATION_OUTPUT_FILENAME,
)


def test_collect_progress_reports_terminal_running_starting_and_pending(tmp_path):
    definitions = tmp_path / "definitions"
    passed = definitions / "passed_op"
    running = definitions / "running_op"
    starting = definitions / "starting_op"
    passed.mkdir(parents=True)
    running.mkdir(parents=True)
    starting.mkdir(parents=True)

    (passed / KERNEL_OPTIMIZATION_OUTPUT_FILENAME).write_text(
        json.dumps({
            "definition_name": "passed_op",
            "status": "PASSED",
            "rounds": 3,
            "best_geo_mean": 1.25,
        }),
        encoding="utf-8",
    )
    (running / ".ledger.json").write_text(
        json.dumps({
            "best_geo_mean": 0.75,
            "rounds": [
                {"evaluation": {"status": "RUNTIME_ERROR"}},
                {"evaluation": {"status": "PASSED"}},
            ],
        }),
        encoding="utf-8",
    )

    progress = monitor.collect_progress(
        tmp_path,
        ["passed_op", "running_op", "starting_op", "pending_op"],
    )
    by_name = {item.name: item for item in progress}

    assert by_name["passed_op"].state == "PASSED"
    assert by_name["passed_op"].rounds == 3
    assert by_name["passed_op"].best_geo_mean == 1.25
    assert by_name["running_op"].state == "RUNNING"
    assert by_name["running_op"].rounds == 2
    assert by_name["running_op"].last_eval_status == "PASSED"
    assert by_name["starting_op"].state == "STARTING"
    assert by_name["pending_op"].state == "PENDING"


def test_monitor_discovers_names_from_batch_output_and_renders(tmp_path, monkeypatch):
    (tmp_path / BATCH_SIMPLE_OPT_OUTPUT_FILENAME).write_text(
        json.dumps({
            "summary": "1/1 definitions passed",
            "results": [
                {
                    "definition_name": "done_op",
                    "status": "PASSED",
                    "rounds": 2,
                    "best_geo_mean": 2.0,
                }
            ],
        }),
        encoding="utf-8",
    )
    item_dir = tmp_path / "definitions" / "done_op"
    item_dir.mkdir(parents=True)
    (item_dir / KERNEL_OPTIMIZATION_OUTPUT_FILENAME).write_text(
        json.dumps({
            "definition_name": "done_op",
            "status": "PASSED",
            "rounds": 2,
            "best_geo_mean": 2.0,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(monitor, "_server_status", lambda _: "healthy")

    progress = monitor.collect_progress(tmp_path)
    rendered = monitor.render_snapshot(tmp_path, progress, "http://server")

    assert [item.name for item in progress] == ["done_op"]
    assert "1/1 definitions passed" in rendered
    assert "done_op" in rendered
    assert "healthy" in rendered


def test_format_server_status_supports_kernelgen_server_v5_devices():
    status = monitor._format_server_status({
        "status": "ok",
        "devices": ["cuda:0", "cuda:1"],
        "workers": 2,
        "timing": "triton",
    })

    assert status == "healthy, 2 devices, 2 workers, timing=triton"


def test_format_server_status_preserves_legacy_busy_device_summary():
    status = monitor._format_server_status({
        "devices": [
            {"device": "cuda:0", "status": "busy"},
            {"device": "cuda:1", "status": "idle"},
        ],
    })

    assert status == "healthy, 1/2 devices busy"


def test_collect_progress_marks_finished_nonterminal_coder_as_failed(tmp_path):
    item_dir = tmp_path / "definitions" / "failed_op"
    runtime_dir = item_dir / ".kernelgen"
    runtime_dir.mkdir(parents=True)
    (item_dir / ".ledger.json").write_text(
        json.dumps({
            "best_geo_mean": 0.0,
            "rounds": [{
                "evaluation": {"status": "PARTIAL_PASS"},
                "next_verdict": {"should_continue": True},
            }],
        }),
        encoding="utf-8",
    )
    (runtime_dir / "claude-runtime.log").write_text(
        "[claude] model=deepseek\n[claude] done  (10 chars, 1s)\n",
        encoding="utf-8",
    )

    [progress] = monitor.collect_progress(tmp_path, ["failed_op"])

    assert progress.state == "FAILED"
    assert progress.rounds == 1
    assert progress.last_eval_status == "PARTIAL_PASS"

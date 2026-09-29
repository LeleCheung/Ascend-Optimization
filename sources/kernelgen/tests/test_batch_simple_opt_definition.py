"""Host tests for the thin batch wrapper around SimpleOptWorkflow."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kernelgen.cli.legacy_batch_simple_opt import (
    bind_reference_code_paths,
)
from kernelgen.framework.parallel import (
    ParallelExecutionError,
    ParallelTaskFailure,
)
from kernelgen.workflows.legacy import batch_simple_opt_definition as batch_module
from kernelgen.workflows.legacy.batch_simple_opt_definition import (
    BATCH_SIMPLE_OPT_OUTPUT_FILENAME,
    BatchSimpleOptDefinitionOutput,
    BatchSimpleOptDefinitionWorkflow,
)
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationOutput
from kernelgen.workflows.legacy.simple_opt import SimpleOptWorkflow


BATCH_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "batch_simple_opt_definition"
    / "run_example.py"
)


def test_batch_cli_exposes_codex_runtime_selection():
    completed = subprocess.run(
        [sys.executable, str(BATCH_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--runtime {claude,codex}" in completed.stdout


def test_batch_simple_opt_runs_simple_opt_in_parallel_and_saves_summary(
    tmp_path,
    monkeypatch,
):
    captured = {}
    reference_code = tmp_path / "reference.py"
    reference_code.write_text(
        "@triton.jit\ndef reference_kernel(x):\n    return\n",
        encoding="utf-8",
    )
    reference_code_prompt = tmp_path / "reference.md"
    reference_code_prompt.write_text(
        "Historical Ascend 910B4 native result.",
        encoding="utf-8",
    )
    expected = [
        SingleCoderOptimizationOutput(
            definition_name="op_a",
            status="PASSED",
            best_geo_mean=1.2,
            best_code="def run(x): return x",
            workspace=str(tmp_path / "run" / "definitions" / "op_a"),
        ),
        SingleCoderOptimizationOutput(
            definition_name="op_b",
            status="FAILED",
            workspace=str(tmp_path / "run" / "definitions" / "op_b"),
        ),
    ]

    def fake_run_parallel(
        runnable_cls,
        inputs,
        *,
        workspace,
        runtime_factory,
        max_workers,
        task_name,
        launch_interval_seconds,
        cancellation_token,
    ):
        captured.update(
            runnable_cls=runnable_cls,
            inputs=inputs,
            workspace=workspace,
            runtime_factory=runtime_factory,
            max_workers=max_workers,
            task_name=task_name,
            launch_interval_seconds=launch_interval_seconds,
            cancellation_token=cancellation_token,
        )
        return [
            (result, item.definition_name)
            for result, item in zip(expected, inputs)
        ]

    monkeypatch.setattr(batch_module, "run_parallel", fake_run_parallel)
    runtime_factory = lambda path: path
    run_dir = tmp_path / "run"
    output = BatchSimpleOptDefinitionWorkflow(
        cwd=str(run_dir),
        runtime_factory=runtime_factory,
    ).run({
        "definitions": [
            {
                "definition_name": "op_a",
                "catalog_name": "flaggems-v5",
                "reference_code_path": reference_code,
                "reference_code_prompt_path": reference_code_prompt,
            },
            {
                "definition_name": "op_b",
                "catalog_name": "flaggems-v5",
                "reference_code_path": reference_code,
                "reference_code_prompt_path": reference_code_prompt,
            },
        ],
        "max_workers": 2,
    })

    assert captured["runnable_cls"] is SimpleOptWorkflow
    assert [item.definition_name for item in captured["inputs"]] == ["op_a", "op_b"]
    assert all(
        item.reference_code_path == reference_code
        for item in captured["inputs"]
    )
    assert all(
        item.reference_code_prompt_path == reference_code_prompt
        for item in captured["inputs"]
    )
    assert captured["workspace"].base == run_dir / "definitions"
    assert captured["workspace"].kb_source is None
    assert captured["workspace"].include_skills is False
    assert captured["runtime_factory"] is runtime_factory
    assert captured["max_workers"] == 2
    assert captured["task_name"] == "definition_name"
    assert captured["launch_interval_seconds"] == 1.0
    assert captured["cancellation_token"] is not None
    assert output.summary == "1/2 definitions passed"
    assert output.results == expected

    saved = BatchSimpleOptDefinitionOutput.model_validate_json(
        (run_dir / BATCH_SIMPLE_OPT_OUTPUT_FILENAME).read_text(encoding="utf-8")
    )
    assert saved == output


def test_batch_workflow_does_not_change_simple_opt_contract():
    assert SimpleOptWorkflow.OutputModel is SingleCoderOptimizationOutput


def test_batch_binds_explicit_reference_paths_by_definition_filename(tmp_path):
    addmm = tmp_path / "addmm_.py"
    addmm.write_text("def run(): pass\n", encoding="utf-8")
    softmax = tmp_path / "softmax.py"
    softmax.write_text("def run(): pass\n", encoding="utf-8")

    assert bind_reference_code_paths(
        [softmax, addmm],
        ["addmm_", "softmax", "missing"],
    ) == {
        "addmm_": addmm,
        "softmax": softmax,
    }


def test_batch_rejects_unmatched_or_duplicate_reference_paths(tmp_path):
    addmm = tmp_path / "addmm_.py"
    addmm.write_text("def run(): pass\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        bind_reference_code_paths([addmm], ["softmax"])
    with pytest.raises(ValueError, match="duplicate"):
        bind_reference_code_paths([addmm, addmm], ["addmm_"])


def test_batch_preserves_completed_outputs_when_one_task_raises(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run"

    def fake_run_parallel(
        runnable_cls,
        inputs,
        *,
        workspace,
        runtime_factory,
        max_workers,
        task_name,
        launch_interval_seconds,
        cancellation_token,
    ):
        completed_dir = workspace.base / "op_a"
        completed_dir.mkdir(parents=True)
        (completed_dir / "optimize_definition_output.json").write_text(
            SingleCoderOptimizationOutput(
                definition_name="op_a",
                status="PASSED",
                best_geo_mean=1.5,
                rounds=3,
                workspace=str(completed_dir),
            ).model_dump_json(),
            encoding="utf-8",
        )
        failed_dir = workspace.base / "op_b"
        failed_dir.mkdir(parents=True)
        (failed_dir / ".ledger.json").write_text(
            json.dumps({
                "best_geo_mean": 0.0,
                "rounds": [{"next_verdict": {"should_continue": True}}],
            }),
            encoding="utf-8",
        )
        raise RuntimeError("Coder returned before terminal STOP verdict")

    monkeypatch.setattr(batch_module, "run_parallel", fake_run_parallel)
    output = BatchSimpleOptDefinitionWorkflow(
        cwd=str(run_dir),
        runtime_factory=lambda path: path,
    ).run({
        "definitions": [
            {"definition_name": "op_a", "catalog_name": "flaggems-v5"},
            {"definition_name": "op_b", "catalog_name": "flaggems-v5"},
        ],
        "max_workers": 2,
    })

    assert output.summary == "1/2 definitions passed; 1 task(s) failed"
    assert [result.status for result in output.results] == ["PASSED", "FAILED"]
    assert output.results[1].rounds == 1
    assert "terminal STOP verdict" in output.results[1].summary
    assert (run_dir / BATCH_SIMPLE_OPT_OUTPUT_FILENAME).is_file()


def test_batch_preserves_each_parallel_task_error(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    completed = SingleCoderOptimizationOutput(
        definition_name="op_a",
        status="PASSED",
        best_geo_mean=1.5,
        rounds=3,
        workspace=str(run_dir / "definitions" / "op_a"),
    )

    def fake_run_parallel(*args, **kwargs):
        raise ParallelExecutionError(
            partial_results=[
                (completed, "op_a"),
                None,
                None,
            ],
            failures=[
                ParallelTaskFailure(
                    index=1,
                    name="op_b",
                    error=TimeoutError("eval timed out"),
                ),
                ParallelTaskFailure(
                    index=2,
                    name="op_c",
                    error=RuntimeError("SSH gateway unavailable"),
                ),
            ],
        )

    monkeypatch.setattr(batch_module, "run_parallel", fake_run_parallel)
    output = BatchSimpleOptDefinitionWorkflow(
        cwd=str(run_dir),
        runtime_factory=lambda path: path,
    ).run({
        "definitions": [
            {"definition_name": "op_a", "catalog_name": "flaggems-v5"},
            {"definition_name": "op_b", "catalog_name": "flaggems-v5"},
            {"definition_name": "op_c", "catalog_name": "flaggems-v5"},
        ],
        "max_workers": 3,
    })

    assert output.summary == "1/3 definitions passed; 2 task(s) failed"
    assert output.results[0] == completed
    assert output.results[1].summary == "TimeoutError: eval timed out"
    assert output.results[2].summary == (
        "RuntimeError: SSH gateway unavailable"
    )

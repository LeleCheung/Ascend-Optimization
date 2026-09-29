import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/experiments/run_kernel_todo_v2_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("kernel_todo_v2_baseline", SCRIPT_PATH)
assert SPEC and SPEC.loader
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def _result(tmp_path, record, *, exit_code=0):
    local_dir = tmp_path / "operator"
    local_dir.mkdir(exist_ok=True)
    (local_dir / "benchmark.json").write_text(
        json.dumps({"operator": {"details": [{"result": [record]}]}}),
        encoding="utf-8",
    )
    return {
        "local_dir": str(local_dir),
        "job": {"status": "SUCCEEDED"},
        "summary": {"status": "completed", "exit_code": exit_code},
    }


def test_phase_verdicts_are_independent(tmp_path):
    combined = baseline._verdict(
        _result(tmp_path, {"latency_base": 1.0, "latency": 2.0}), "combined"
    )
    assert combined["reference"] == "通过"
    assert combined["timing"] == "通过"

    (tmp_path / "operator/benchmark.json").unlink()
    reference = baseline._verdict(
        _result(tmp_path, {"latency_base": 1.0}), "reference"
    )
    assert reference["reference"] == "通过"
    assert reference["timing"] is None

    (tmp_path / "operator/benchmark.json").unlink()
    gems = baseline._verdict(_result(tmp_path, {"latency": 2.0}), "gems")
    assert gems["reference"] is None
    assert gems["timing"] == "通过"


def test_phase_target_selection_uses_only_its_result_column(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "REPO_ROOT", tmp_path)
    chip_dir = tmp_path / "kernel_todo_v2/test-chip"
    chip_dir.mkdir(parents=True)
    operators = ["both_failed", "reference_passed", "both_passed", "optimized"]
    (chip_dir / "failed_ops.txt").write_text(
        "\n".join(operators) + "\n", encoding="utf-8"
    )
    inventory = {
        "operators": [
            {
                "source_operator": name,
                "operator": name,
                "chips": ["test-chip"],
            }
            for name in operators
        ]
    }
    (tmp_path / "kernel_todo_v2/pytest_conversion_inventory.json").write_text(
        json.dumps(inventory), encoding="utf-8"
    )
    results = chip_dir / "results.md"
    results.write_text(
        "| `both_failed` | 未通过 | 无法计时 | 未跑 | — | — | — | — | — |\n"
        "| `reference_passed` | 通过 | 无法计时 | 未跑 | — | — | — | — | — |\n"
        "| `both_passed` | 通过 | 通过 | 未跑 | — | — | — | — | — |\n"
        "| `optimized` | 未通过 | 无法计时 | 成功 | — | — | — | — | — |\n",
        encoding="utf-8",
    )

    combined = baseline._load_targets("test-chip", results, set(), "combined")
    reference = baseline._load_targets("test-chip", results, set(), "reference")
    gems = baseline._load_targets("test-chip", results, set(), "gems")

    assert [item["source_operator"] for item in combined] == [
        "both_failed",
        "reference_passed",
    ]
    assert [item["source_operator"] for item in reference] == ["both_failed"]
    assert [item["source_operator"] for item in gems] == [
        "both_failed",
        "reference_passed",
    ]


def test_split_phase_updates_only_one_column_and_keeps_both_evidence(tmp_path):
    results = tmp_path / "results.md"
    results.write_text(
        "| 总算子数 | reference通过数 | Gems 可计时数 | 已完成数 | "
        "达标数（加速比≥0.8） | 未达标数（加速比<0.8） | "
        "失败数（未通过正确性测试） |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        "| 1 | 0 | 0 | 0 | 0 | 0 | 0 |\n\n"
        "| 算子名 | Reference | Gems 可计时 | 优化状态 | 加速比 | "
        "Hack 情况 | 原因 | 后续方向 | code_path |\n"
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |\n"
        "| `op` | 未通过 | 无法计时 | 未跑 | — | 未检查 | 原联合 baseline 失败。 | 保持未跑。 | — |\n",
        encoding="utf-8",
    )
    reference_summary = tmp_path / "reference/operators/op/summary.json"
    gems_summary = tmp_path / "gems/operators/op/summary.json"

    baseline._update_results(
        results,
        "op",
        {
            "reference": "通过",
            "timing": None,
            "reason": "",
        },
        reference_summary,
        "reference",
    )
    after_reference = results.read_text(encoding="utf-8")
    assert "| `op` | 通过 | 无法计时 |" in after_reference
    assert "[Reference 复检证据]" in after_reference

    baseline._update_results(
        results,
        "op",
        {
            "reference": None,
            "timing": "通过",
            "reason": "",
        },
        gems_summary,
        "gems",
    )
    final = results.read_text(encoding="utf-8")
    assert "| `op` | 通过 | 通过 |" in final
    assert "[Reference 复检证据]" in final
    assert "[Gems 可计时复检证据]" in final
    assert "V2 baseline 已通过" in final
    assert "| 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |" in final


def test_summary_counts_gems_timing_as_reference_intersection():
    lines = [
        "| 总算子数 | reference通过数 | Gems 可计时数 | 已完成数 | "
        "达标数（加速比≥0.8） | 未达标数（加速比<0.8） | "
        "失败数（未通过正确性测试） |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| 0 | 0 | 0 | 0 | 0 | 0 | 0 |",
        "| `both_passed` | 通过 | 通过 | 未跑 | — | — | — | — | — |",
        "| `reference_only` | 通过 | 无法计时 | 未跑 | — | — | — | — | — |",
        "| `gems_only` | 未通过 | 通过 | 未跑 | — | — | — | — | — |",
    ]

    baseline._refresh_results_summary(lines)

    assert lines[2] == "| 3 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |"

import importlib.util
import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/experiments"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


simple_update = _load("kernel_todo_simple_update", "update_kernel_todo_v2_results.py")
stage_update = _load(
    "kernel_todo_stage_update", "update_kernel_todo_v2_pipeline_stage.py"
)


def _results(path: Path, operators: list[str]) -> None:
    rows = "\n".join(
        f"| `{operator}` | 通过 | 通过 | 未跑 | — | 未检查 | — | 待优化。 | — |"
        for operator in operators
    )
    path.write_text(
        "| 总算子数 | reference通过数 | Gems 可计时数 | 已完成数 | "
        "达标数（加速比≥0.8） | 未达标数（加速比<0.8） | "
        "失败数（未通过正确性测试） |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| {len(operators)} | {len(operators)} | {len(operators)} | 0 | 0 | 0 | 0 |\n"
        "\n| 算子名 | Reference | Gems 可计时 | 优化状态 | 加速比 | "
        "Hack 情况 | 原因 | 后续方向 | code_path |\n"
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |\n"
        f"{rows}\n",
        encoding="utf-8",
    )


def _inventory(root: Path, operators: list[str]) -> None:
    directory = root / "kernel_todo_v2"
    directory.mkdir()
    (directory / "pytest_conversion_inventory.json").write_text(
        json.dumps(
            {
                "operators": [
                    {
                        "source_operator": operator,
                        "operator": operator,
                        "chips": ["muxi"],
                    }
                    for operator in operators
                ]
            }
        ),
        encoding="utf-8",
    )


def _simple_output(workspace: Path, operator: str, *, status: str, geo: float | None):
    directory = workspace / "definitions" / operator
    directory.mkdir(parents=True)
    output = {"definition_name": operator, "status": status}
    if geo is not None:
        output["best_geo_mean"] = geo
    (directory / "optimize_definition_output.json").write_text(
        json.dumps(output), encoding="utf-8"
    )
    (directory / ".ledger.json").write_text(
        json.dumps(
            {
                "definition_name": operator,
                "best_round": 1 if status == "PASSED" else 0,
                "rounds": [
                    {
                        "round_num": 1,
                        "evaluation": {"is_hack": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if status == "PASSED":
        (directory / ".best_kernel.py").write_text("def run(x): return x\n")


def _run_simple(monkeypatch, tmp_path: Path, workspace: Path, results: Path) -> int:
    pipeline = tmp_path / "pipeline.json"
    second_stage = tmp_path / "kernelgen_results.md"
    monkeypatch.setattr(simple_update, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "update_kernel_todo_v2_results.py",
            "--chip",
            "muxi",
            "--workspace",
            str(workspace),
            "--results-md",
            str(results),
            "--pipeline-json",
            str(pipeline),
            "--kernelgen-results-md",
            str(second_stage),
        ],
    )
    return simple_update.main()


def test_simple_opt_update_creates_pipeline_and_routes_low_speedup(
    tmp_path: Path, monkeypatch
):
    _inventory(tmp_path, ["gelu", "relu"])
    results = tmp_path / "results.md"
    _results(results, ["gelu", "relu"])
    workspace = tmp_path / "workspace"
    _simple_output(workspace, "gelu", status="PASSED", geo=0.7)
    _simple_output(workspace, "relu", status="PASSED", geo=1.2)

    assert _run_simple(monkeypatch, tmp_path, workspace, results) == 0
    assert _run_simple(monkeypatch, tmp_path, workspace, results) == 0

    markdown = results.read_text(encoding="utf-8")
    assert "| `gelu` | 通过 | 通过 | 待二阶段优化 | 0.700x |" in markdown
    assert "| `relu` | 通过 | 通过 | 成功 | 1.200x |" in markdown
    assert "| 2 | 2 | 2 | 1 | 1 | 0 | 0 | 0 |" in markdown
    pipeline = json.loads((tmp_path / "pipeline.json").read_text())
    assert pipeline["operators"]["gelu"]["simple_opt"]["attempts"][-1]["geo_mean"] == 0.7
    assert pipeline["operators"]["relu"]["simple_opt"]["attempts"][-1]["geo_mean"] == 1.2
    second_stage = (tmp_path / "kernelgen_results.md").read_text()
    assert "`gelu`" in second_stage
    assert "`relu`" not in second_stage


def _run_stage(
    monkeypatch,
    tmp_path: Path,
    results: Path,
    *,
    stage: str,
    geo: float,
) -> int:
    monkeypatch.setattr(stage_update, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "update_kernel_todo_v2_pipeline_stage.py",
            "--chip",
            "muxi",
            "--operator",
            "gelu",
            "--stage",
            stage,
            "--verdict",
            "passed",
            "--attempt-id",
            f"{stage}-attempt-1",
            "--geo-mean",
            str(geo),
            "--evidence",
            f"{stage}.json",
            "--results-md",
            str(results),
            "--pipeline-json",
            str(tmp_path / "pipeline.json"),
            "--kernelgen-results-md",
            str(tmp_path / "kernelgen_results.md"),
        ],
    )
    return stage_update.main()


def test_stage_updates_require_order_and_final_gems_is_authoritative(
    tmp_path: Path, monkeypatch
):
    _inventory(tmp_path, ["gelu"])
    results = tmp_path / "results.md"
    _results(results, ["gelu"])
    workspace = tmp_path / "workspace"
    _simple_output(workspace, "gelu", status="PASSED", geo=0.7)
    assert _run_simple(monkeypatch, tmp_path, workspace, results) == 0

    assert _run_stage(
        monkeypatch,
        tmp_path,
        results,
        stage="native_baseline",
        geo=0.69,
    ) == 0
    assert "| `gelu` | 通过 | 通过 | 二阶段处理中 | 0.700x |" in results.read_text()

    assert _run_stage(
        monkeypatch,
        tmp_path,
        results,
        stage="kernelgen_native",
        geo=1.1,
    ) == 0
    assert _run_stage(
        monkeypatch,
        tmp_path,
        results,
        stage="final_gems",
        geo=0.95,
    ) == 0

    markdown = results.read_text()
    assert "| `gelu` | 通过 | 通过 | 成功 | 0.950x |" in markdown
    assert "| 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |" in markdown
    detail = (tmp_path / "kernelgen_results.md").read_text()
    assert "| `gelu` | 0.700x | 0.690x | 1.100x | 0.950x | 第二阶段达标 |" in detail

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from kernelgen.tests.extraction_review_fixtures import stub_extraction_review

from kernelgen.agents.extractor.flaggems.v62_agent import (
    FlagGemsV62ExtractorOutput,
)
from kernelgen.examples.flaggems_v62_extract import batch_extract
from kernelgen.workflows import catalog_extract


def _write_case_report(path: Path, operators: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "flaggems.benchmark-case-list/v2",
                "benchmarks": [
                    {
                        "schema_version": "flaggems.benchmark-case-list/v2",
                        "op_name": operator,
                        "phase": "timing",
                        "level": "core",
                        "cases": [
                            {
                                "case_id": f"benchmark/test_{operator}.py::core::0",
                            }
                        ],
                    }
                    for operator in operators
                ],
            }
        ),
        encoding="utf-8",
    )


def _args(tmp_path: Path, operators: list[str]) -> SimpleNamespace:
    operator_file = tmp_path / "operators.txt"
    operator_file.write_text("\n".join(operators) + "\n", encoding="utf-8")
    case_list_path = tmp_path / "cases.json"
    _write_case_report(case_list_path, operators)
    flaggems_repo = tmp_path / "FlagGems"
    flaggems_repo.mkdir()
    return SimpleNamespace(
        operator_file=operator_file,
        flaggems_repo=flaggems_repo,
        case_list_path=case_list_path,
        catalog_root=tmp_path / "catalog",
        adapter_catalog_root=None,
        workspace_root=tmp_path / "workspaces",
        report=tmp_path / "report.json",
        model="inherit",
        timeout=60,
        max_workers=4,
    )


def _output(operator: str) -> FlagGemsV62ExtractorOutput:
    return FlagGemsV62ExtractorOutput(
        oracle="def run(input):\n    return input\n",
        correctness_workloads=[{"name": "correctness", "inputs": {}}],
        timing_workloads=[{"name": "timing", "inputs": {}}],
    )


def test_read_operators_accepts_plain_list_and_markdown(tmp_path: Path):
    plain = tmp_path / "operators.txt"
    plain.write_text("gelu\n\naddmm_\ngelu\n", encoding="utf-8")
    table = tmp_path / "operators.md"
    table.write_text(
        "| 算子 | 状态 |\n"
        "| --- | --- |\n"
        "| `gelu` | 未跑 |\n"
        "| addmm_ | 未跑 |\n",
        encoding="utf-8",
    )

    assert batch_extract.read_operators(plain) == ["gelu", "addmm_"]
    assert batch_extract.read_operators(table) == ["gelu", "addmm_"]


@pytest.mark.parametrize("value", ["gelu;touch_bad", "linalg.norm", "1bad"])
def test_read_operators_rejects_non_definition_names(tmp_path: Path, value: str):
    path = tmp_path / "operators.txt"
    path.write_text(value + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid operator name"):
        batch_extract.read_operators(path)


def test_catalog_extract_keeps_input_output_contract_and_persisted_name():
    assert catalog_extract.CatalogExtractWorkflow.name == "flaggems_v62_agent_extract"
    assert catalog_extract.CatalogExtractWorkflow.InputModel is catalog_extract.CatalogExtractInput
    assert catalog_extract.CatalogExtractWorkflow.OutputModel is catalog_extract.CatalogExtractOutput
    assert set(catalog_extract.CatalogExtractInput.model_fields) == {"operator", "flaggems_repo", "pr_url", "case_list_path", "max_review_rounds"}
    assert catalog_extract.CatalogExtractInput.model_fields["operator"].is_required()
    assert not catalog_extract.CatalogExtractInput.model_fields["case_list_path"].is_required()
    assert set(catalog_extract.CatalogExtractOutput.model_fields) == {"operator", "extraction", "accuracy_coverage", "case_list_path", "catalog_path", "operator_dir", "source_evidence", "review_path"}
    assert batch_extract.SCHEMA_VERSION == "kernelgen.flaggems-v62-batch-extract/v1"


def test_agent_workflow_requires_runtime_factory(tmp_path: Path):
    workflow = catalog_extract.CatalogExtractWorkflow(
        cwd=str(tmp_path)
    )

    with pytest.raises(RuntimeError, match="requires a runtime_factory"):
        workflow.run(
            {
                "operator": "gelu",
                "flaggems_repo": str(tmp_path),
                "case_list_path": str(tmp_path / "cases.json"),
            }
        )


def test_agent_workflow_returns_persisted_catalog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    extraction = _output("gelu")
    (tmp_path / "cases.json").write_text("fixture")
    (tmp_path / "gelu.py").write_text("source fixture")
    runtime = object()
    seen: dict[str, object] = {}

    def persist(inp, output, catalog):
        assert output == extraction
        root = catalog / "ops/gelu"
        root.mkdir(parents=True)
        return SimpleNamespace(operator_root=root)

    monkeypatch.setattr(catalog_extract, "persist_flaggems_v62_extraction", persist)
    monkeypatch.setattr(catalog_extract, "collect_source_inventory", lambda *_: SimpleNamespace(
        implementation_file=tmp_path / "gelu.py", test_files=[], benchmark_files=[]))

    def fake_run(self, inp, actual_runtime):
        seen["input"] = inp
        seen["runtime"] = actual_runtime
        return extraction

    monkeypatch.setattr(
        catalog_extract.FlagGemsV62ExtractorAgent,
        "run",
        fake_run,
    )
    monkeypatch.setattr(
        catalog_extract,
        "build_flaggems_v62_accuracy_coverage",
        lambda inp, output: {"operator": inp.operator, "covered": True},
    )
    def runtime_factory(path):
        seen["workspace"] = path
        return runtime

    workflow = catalog_extract.CatalogExtractWorkflow.bind(
        str(tmp_path / "workspace"), runtime_factory,
    )

    output = workflow.run(
        {
            "operator": "gelu",
            "flaggems_repo": str(tmp_path),
            "case_list_path": str(tmp_path / "cases.json"),
        }
    )

    assert output.operator == "gelu"
    assert output.extraction == extraction
    assert output.accuracy_coverage == {"operator": "gelu", "covered": True}
    assert seen["input"] == {
        "operator": "gelu", "flaggems_repo": str(tmp_path),
        "case_list_path": str(tmp_path / "cases.json"),
    }
    assert seen["runtime"] is runtime
    assert seen["workspace"] == str(tmp_path / "workspace/extractor")
    assert output.catalog_path == tmp_path / "workspace/catalog"
    assert output.operator_dir == tmp_path / "workspace/catalog/ops/gelu"
    assert output.operator_dir.is_dir()
    assert output.source_evidence == [tmp_path / "gelu.py", tmp_path / "cases.json"]


def test_batch_extracts_in_parallel_but_persists_in_caller_ordered_report(
    tmp_path: Path,
):
    args = _args(tmp_path, ["slow", "fast"])
    caller_thread = threading.get_ident()
    persist_threads: list[int] = []

    def extract_one(operator: str, **kwargs):
        if operator == "slow":
            time.sleep(0.03)
        return SimpleNamespace(operator=operator), {
            "operator": operator,
            "status": "EXTRACTED",
        }

    def persist_one(output, **kwargs):
        persist_threads.append(threading.get_ident())
        return {"operator_root": f"catalog/ops/{output.operator}"}

    exit_code, report = batch_extract.run_batch(
        args,
        extract_one=extract_one,
        persist_one=persist_one,
    )

    assert exit_code == 0
    assert [row["operator"] for row in report["results"]] == ["slow", "fast"]
    assert [row["status"] for row in report["results"]] == [
        "PERSISTED",
        "PERSISTED",
    ]
    assert report["counts"] == {"PERSISTED": 2}
    assert persist_threads == [caller_thread, caller_thread]
    assert json.loads(args.report.read_text(encoding="utf-8")) == report
    assert not args.report.with_suffix(".json.tmp").exists()


def test_batch_isolates_worker_failure_and_keeps_other_results(tmp_path: Path):
    args = _args(tmp_path, ["broken", "gelu"])

    def extract_one(operator: str, **kwargs):
        if operator == "broken":
            raise RuntimeError("worker exited")
        return SimpleNamespace(operator=operator), {
            "operator": operator,
            "status": "EXTRACTED",
        }

    exit_code, report = batch_extract.run_batch(
        args,
        extract_one=extract_one,
        persist_one=lambda output, **kwargs: {},
    )

    assert exit_code == 1
    assert report["results"][0]["status"] == "EXTRACT_FAILED"
    assert report["results"][0]["reason"] == "RuntimeError: worker exited"
    assert report["results"][1]["status"] == "PERSISTED"


def test_batch_records_case_gap_and_existing_package_without_running_agent(
    tmp_path: Path,
):
    args = _args(tmp_path, ["existing", "missing", "partial"])
    _write_case_report(args.case_list_path, ["existing", "partial"])
    existing = args.catalog_root / "ops" / "existing"
    existing.mkdir(parents=True)
    for name in batch_extract._PACKAGE_FILES:
        (existing / name).write_text("present\n", encoding="utf-8")
    partial = args.catalog_root / "ops" / "partial"
    partial.mkdir(parents=True)
    (partial / "definition.json").write_text("{}\n", encoding="utf-8")

    def unexpected_extract(*args, **kwargs):
        raise AssertionError("no operator should reach the Agent")

    exit_code, report = batch_extract.run_batch(
        args,
        extract_one=unexpected_extract,
        persist_one=lambda output, **kwargs: {},
    )

    assert exit_code == 1
    assert [row["status"] for row in report["results"]] == [
        "ALREADY_PRESENT",
        "CASE_LIST_INVALID",
        "TARGET_CONFLICT",
    ]


def test_persist_writes_coverage_before_catalog_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    workspace_root = tmp_path / "workspaces"
    coverage_path = workspace_root / "gelu" / "accuracy_coverage.json"
    output = catalog_extract.CatalogExtractOutput(
        operator="gelu",
        extraction=_output("gelu"),
        accuracy_coverage={"operator": "gelu"},
        catalog_path=workspace_root / "gelu/catalog",
        operator_dir=workspace_root / "gelu/catalog/ops/gelu",
        source_evidence=[],
    )

    def fake_persist(*args, **kwargs):
        assert json.loads(coverage_path.read_text(encoding="utf-8")) == {
            "operator": "gelu"
        }
        return SimpleNamespace(
            operator_root=tmp_path / "catalog" / "ops" / "gelu",
            adapter_definition_path=None,
            num_correctness_workloads=1,
            num_timing_workloads=1,
        )

    monkeypatch.setattr(
        batch_extract,
        "persist_flaggems_v62_extraction",
        fake_persist,
    )

    result = batch_extract._persist_one(
        output,
        flaggems_repo=tmp_path / "FlagGems",
        case_list_path=tmp_path / "cases.json",
        catalog_root=tmp_path / "catalog",
        adapter_catalog_root=None,
        workspace_root=workspace_root,
    )

    assert result["accuracy_coverage"] == str(coverage_path)

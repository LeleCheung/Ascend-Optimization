"""Opt-in structural E2E for every FlagGems Definition in kernel_todo."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from kernelgen_server import Catalog, builtin_catalog_path
from kernelgen_server.evaluation.adapters import create_adapter
from kernelgen_server.evaluation.adapters.flaggems.discovery import discover_assets
from kernelgen_server.schema import EvaluatorBinding


pytestmark = pytest.mark.skipif(
    os.environ.get("KGS_RUN_FLAGGEMS_CATALOG_E2E") != "1",
    reason="set KGS_RUN_FLAGGEMS_CATALOG_E2E=1 in a FlagGems source container",
)


def test_all_kernel_todo_definitions_have_native_timing_cases(tmp_path, monkeypatch):
    """Collect native cases once, then exercise KGS normalization for every op."""

    catalog = Catalog(builtin_catalog_path("flaggems-adapter-definitions"))
    assets = {
        operator: discover_assets(operator)
        for operator in catalog.operator_names
    }
    benchmark_files = sorted(
        {
            path.relative_to(asset.root).as_posix()
            for asset in assets.values()
            for path in asset.performance
        }
    )
    markers = sorted({asset.native_operator for asset in assets.values()})
    output = tmp_path / "all-cases.json"
    root = next(iter(assets.values())).root
    env = dict(os.environ)
    python_path = [str(root / "src")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *benchmark_files,
            "-m",
            " or ".join(markers),
            "--level",
            "core",
            "--list-cases",
            "--output",
            str(output),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert process.returncode == 0, (process.stdout + process.stderr)[-8000:]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "flaggems.benchmark-case-list/v2"
    benchmarks = report["benchmarks"]
    assert all("nodeid" not in benchmark for benchmark in benchmarks)
    native_case_ids = [
        case["case_id"] for benchmark in benchmarks for case in benchmark["cases"]
    ]
    assert len(native_case_ids) == len(set(native_case_ids))

    total_cases = 0
    for operator, asset in assets.items():
        native_report = {
            "schema_version": report["schema_version"],
            "benchmarks": [
                item
                for item in benchmarks
                if item.get("op_name") == asset.native_operator
            ],
        }
        assert native_report["benchmarks"], operator
        adapter = create_adapter(
            EvaluatorBinding(
                catalog_name="flaggems-adapter-definitions",
                definition=operator,
            )
        )
        monkeypatch.setattr(
            adapter,
            "_native_case_report",
            lambda report=native_report: report,
        )
        manifest = adapter.inspect()
        assert manifest.kind == "flaggems"
        assert manifest.case_list.operator == operator
        assert manifest.case_list.cases
        assert [case.case_id for case in manifest.case_list.cases] == [
            case["case_id"]
            for benchmark in native_report["benchmarks"]
            for case in benchmark["cases"]
        ]
        total_cases += len(manifest.case_list.cases)

    assert len(catalog.operator_names) == 170
    assert total_cases == 2644

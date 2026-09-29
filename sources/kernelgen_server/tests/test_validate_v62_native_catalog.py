import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tools.validate_v62_native_catalog import (
    _module_belongs_to,
    _random_tensor,
)


SCRIPT = Path(__file__).parents[1] / "tools" / "validate_v62_native_catalog.py"


def _write_catalog(
    root: Path,
    *,
    oracle: str = (
        'REFERENCE_DEVICE = "target"\n'
        "def run(x): return x\n"
    ),
    timing: bool = True,
) -> Path:
    operator = root / "ops" / "identity"
    operator.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "api_version": "v6.2",
                "name": root.name,
                "evaluator": "native",
                "layout": "per-operator",
            }
        ),
        encoding="utf-8",
    )
    (operator / "definition.json").write_text(
        json.dumps(
            {
                "api_version": "v6.2",
                "name": "identity",
                "description": "Identity test operator.",
                "parameters": [
                    {
                        "name": "x",
                        "kind": "positional_or_keyword",
                        "required": True,
                        "type_hint": "int",
                    }
                ],
                "outputs": ["out"],
                "effects": {"mutates": [], "returns_alias_of": {}},
            }
        ),
        encoding="utf-8",
    )
    (operator / "oracle.py").write_text(oracle, encoding="utf-8")
    (operator / "correctness.jsonl").write_text(
        json.dumps(
            {
                "name": "identity-correctness",
                "inputs": {"x": {"type": "scalar", "value": 1}},
                "seed": 42,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if timing:
        (operator / "timing.jsonl").write_text(
            json.dumps(
                {
                    "name": "identity-timing",
                    "inputs": {"x": {"type": "scalar", "value": 2}},
                    "seed": 42,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return root


def test_standalone_cleanup_tolerates_noniterable_module_path(tmp_path):
    module = ModuleType("nonstandard_namespace")
    module.__path__ = object()

    assert _module_belongs_to(module, tmp_path) is False


def _run(script: Path, catalog: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, str(script), str(catalog), *args],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )


def _framework_checkout(root: Path) -> str:
    package = root / "src" / "flag_gems"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("OFFSET = 3\n", encoding="utf-8")
    commands = [
        ["git", "init", "-q"],
        ["git", "config", "user.name", "Catalog Validator Test"],
        ["git", "config", "user.email", "validator@example.invalid"],
        ["git", "add", "src/flag_gems/__init__.py"],
        ["git", "commit", "-q", "-m", "fixture"],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def test_standalone_validator_has_no_kernelgen_server_import_and_passes(tmp_path):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert all(not name.startswith("kernelgen_server") for name in imports)

    copied = tmp_path / "validator.py"
    shutil.copyfile(SCRIPT, copied)
    catalog = _write_catalog(tmp_path / "vendor-catalog")
    report_path = tmp_path / "report.json"
    result = _run(
        copied,
        catalog,
        "--format",
        "json",
        "--output",
        str(report_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["passed"] is True
    assert report["summary"] == {
        "operators": 1,
        "correctness_workloads": 1,
        "timing_workloads": 1,
        "asset_files": 0,
        "asset_bytes": 0,
        "errors": 0,
        "warnings": 0,
    }


def test_delivery_mode_accepts_operator_or_bundle_without_manifest(tmp_path):
    catalog = _write_catalog(tmp_path / "source-catalog")
    operator = catalog / "ops" / "identity"

    single = _run(SCRIPT, operator, "--delivery", "--format", "json")
    assert single.returncode == 0, single.stdout + single.stderr
    assert json.loads(single.stdout)["summary"]["operators"] == 1

    bundle = tmp_path / "delivery"
    shutil.copytree(operator, bundle / "identity")
    bundled = _run(SCRIPT, bundle, "--delivery", "--format", "json")
    assert bundled.returncode == 0, bundled.stdout + bundled.stderr
    assert json.loads(bundled.stdout)["summary"]["operators"] == 1

    catalog_only = _run(SCRIPT, operator, "--format", "json")
    assert catalog_only.returncode == 1
    codes = {issue["code"] for issue in json.loads(catalog_only.stdout)["issues"]}
    assert "file.missing" in codes


def test_schema_mode_accepts_one_phase_but_standard_requires_both(tmp_path):
    catalog = _write_catalog(tmp_path / "one-phase", timing=False)

    schema = _run(SCRIPT, catalog, "--mode", "schema", "--format", "json")
    standard = _run(SCRIPT, catalog, "--mode", "standard", "--format", "json")

    assert schema.returncode == 0, schema.stdout + schema.stderr
    assert standard.returncode == 1
    codes = {issue["code"] for issue in json.loads(standard.stdout)["issues"]}
    assert "workload.timing_required" in codes


def test_validator_accepts_v62_file_backed_workload_fields(tmp_path):
    catalog = _write_catalog(tmp_path / "file-backed")
    operator = catalog / "ops" / "identity"
    correctness = json.loads((operator / "correctness.jsonl").read_text())
    correctness.update(
        {
            "input_path": str((tmp_path / "inputs.safetensors").resolve()),
            "output_path": str((tmp_path / "outputs.safetensors").resolve()),
            "inputs": {"x": {"type": "safetensor"}},
        }
    )
    (operator / "correctness.jsonl").write_text(
        json.dumps(correctness) + "\n", encoding="utf-8"
    )

    result = _run(SCRIPT, catalog, "--format", "json")

    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_rejects_relative_tensor_path_and_timing_output_path(tmp_path):
    catalog = _write_catalog(tmp_path / "invalid-file-backed")
    operator = catalog / "ops" / "identity"
    timing = json.loads((operator / "timing.jsonl").read_text())
    timing.update(
        {
            "input_path": "relative-inputs.safetensors",
            "output_path": str((tmp_path / "golden.safetensors").resolve()),
            "inputs": {"x": {"type": "safetensor"}},
        }
    )
    (operator / "timing.jsonl").write_text(
        json.dumps(timing) + "\n", encoding="utf-8"
    )

    result = _run(SCRIPT, catalog, "--format", "json")

    assert result.returncode == 1
    codes = {issue["code"] for issue in json.loads(result.stdout)["issues"]}
    assert {
        "workload.input_path_absolute",
        "workload.timing_output_path",
    }.issubset(codes)


def test_validator_accepts_full_archives_and_requires_active_rows_to_be_present(
    tmp_path,
):
    catalog = _write_catalog(tmp_path / "sampled")
    operator = catalog / "ops" / "identity"
    for phase in ("correctness", "timing"):
        (operator / f"{phase}_full.jsonl").write_text(
            (operator / f"{phase}.jsonl").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    passing = _run(SCRIPT, catalog, "--format", "json")
    assert passing.returncode == 0, passing.stdout + passing.stderr

    active = json.loads((operator / "timing.jsonl").read_text(encoding="utf-8"))
    active["name"] = "identity-timing-not-archived"
    (operator / "timing.jsonl").write_text(
        json.dumps(active) + "\n", encoding="utf-8"
    )
    failing = _run(SCRIPT, catalog, "--format", "json")

    assert failing.returncode == 1
    codes = {issue["code"] for issue in json.loads(failing.stdout)["issues"]}
    assert "workload.archive_missing_active" in codes


def test_validator_rejects_abi_custom_input_and_files_outside_assets(tmp_path):
    catalog = _write_catalog(
        tmp_path / "invalid",
        oracle=(
            'REFERENCE_DEVICE = "target"\n'
            "def run(y): return y\n"
        ),
    )
    operator = catalog / "ops" / "identity"
    workload = json.loads((operator / "correctness.jsonl").read_text())
    workload["inputs"]["x"] = {
        "type": "custom",
        "shape": [2, 2],
        "dtype": "float32",
    }
    (operator / "correctness.jsonl").write_text(
        json.dumps(workload) + "\n", encoding="utf-8"
    )
    (operator / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = _run(SCRIPT, catalog, "--format", "json")

    assert result.returncode == 1
    codes = {issue["code"] for issue in json.loads(result.stdout)["issues"]}
    assert {
        "assets.outside_directory",
        "oracle.phase_abi",
        "workload.required_not_materialized",
    }.issubset(codes)


def test_runtime_smokes_assets_and_all_workloads_without_kgs(tmp_path):
    catalog = _write_catalog(
        tmp_path / "runtime-assets",
        oracle=(
            'REFERENCE_DEVICE = "target"\n'
            "from helper import offset\n"
            "def run(x): return x + offset()\n"
        ),
    )
    assets = catalog / "ops" / "identity" / "assets"
    assets.mkdir()
    (assets / "helper.py").write_text("def offset(): return 1\n", encoding="utf-8")

    copied = tmp_path / "standalone.py"
    shutil.copyfile(SCRIPT, copied)
    result = _run(
        copied,
        catalog,
        "--runtime",
        "--target-device",
        "cpu",
        "--format",
        "json",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["runtime"]["passed"] is True
    assert report["runtime"]["operators"]["identity"] == {
        "passed": True,
        "source": "primary",
        "correctness": 1,
        "timing": 1,
    }
    assert report["summary"]["asset_files"] == 1


def test_runtime_loads_file_backed_inputs_and_golden_outputs(tmp_path):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    catalog = _write_catalog(
        tmp_path / "runtime-file-backed",
        oracle=(
            'REFERENCE_DEVICE = "target"\n'
            "import torch\n"
            "def correctness_run(x): raise RuntimeError('must not run')\n"
            "def timing_run(x):\n"
            "    assert torch.equal(x, torch.arange(4, device=x.device))\n"
            "    return x\n"
        ),
    )
    operator = catalog / "ops" / "identity"
    input_path = tmp_path / "inputs.safetensors"
    output_path = tmp_path / "outputs.safetensors"
    tensor = torch.arange(4)
    save_file({"x": tensor}, input_path)
    save_file({"out": tensor}, output_path)
    for phase in ("correctness", "timing"):
        path = operator / f"{phase}.jsonl"
        workload = json.loads(path.read_text())
        workload["inputs"] = {"x": {"type": "safetensor"}}
        workload["input_path"] = str(input_path)
        if phase == "correctness":
            workload["output_path"] = str(output_path)
        path.write_text(json.dumps(workload) + "\n", encoding="utf-8")

    result = _run(
        SCRIPT,
        catalog,
        "--runtime",
        "--target-device",
        "cpu",
        "--format",
        "json",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["runtime"]["operators"]["identity"]["source"] == "primary"


def test_runtime_binds_pinned_clean_framework_checkout(tmp_path):
    catalog = _write_catalog(
        tmp_path / "framework-catalog",
        oracle=(
            'REFERENCE_DEVICE = "target"\n'
            "import flag_gems\n"
            "def run(x): return x + flag_gems.OFFSET\n"
        ),
    )
    framework = tmp_path / "FlagGems"
    revision = _framework_checkout(framework)
    manifest_path = catalog / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "framework": "flaggems",
            "framework_repository": "https://example.com/FlagGems.git",
            "framework_branch": "feat/kernelgen",
            "framework_revision": revision,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    missing = _run(
        SCRIPT,
        catalog,
        "--runtime",
        "--target-device",
        "cpu",
        "--format",
        "json",
    )
    assert missing.returncode == 1
    assert "runtime.framework_root_required" in {
        issue["code"] for issue in json.loads(missing.stdout)["issues"]
    }

    result = _run(
        SCRIPT,
        catalog,
        "--runtime",
        "--target-device",
        "cpu",
        "--framework-root",
        str(framework),
        "--format",
        "json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["runtime"]["passed"] is True
    assert report["runtime"]["framework"]["head"] == revision

    (framework / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    dirty = _run(
        SCRIPT,
        catalog,
        "--runtime",
        "--target-device",
        "cpu",
        "--framework-root",
        str(framework),
        "--format",
        "json",
    )
    assert dirty.returncode == 1
    assert "runtime.framework_dirty" in {
        issue["code"] for issue in json.loads(dirty.stdout)["issues"]
    }


def test_runtime_uses_whole_operator_torch_fallback(tmp_path):
    catalog = _write_catalog(
        tmp_path / "runtime-fallback",
        oracle=(
            'REFERENCE_DEVICE = "target"\n'
            "def run(x): raise RuntimeError('primary unavailable')\n"
            "def torch_run(x): return x\n"
        ),
    )

    result = _run(
        SCRIPT,
        catalog,
        "--runtime",
        "--target-device",
        "cpu",
        "--format",
        "json",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    runtime = report["runtime"]["operators"]["identity"]
    assert runtime["passed"] is True
    assert runtime["source"] == "torch_fallback"
    assert runtime["correctness"] == runtime["timing"] == 1
    assert {issue["code"] for issue in report["issues"]} == {
        "runtime.primary_fallback"
    }


def test_runtime_without_fallback_reports_primary_failure(tmp_path):
    catalog = _write_catalog(
        tmp_path / "runtime-primary-failure",
        oracle=(
            'REFERENCE_DEVICE = "target"\n'
            "def run(x): raise RuntimeError('primary unavailable')\n"
        ),
    )

    result = _run(
        SCRIPT,
        catalog,
        "--runtime",
        "--target-device",
        "cpu",
        "--format",
        "json",
    )

    assert result.returncode == 1
    runtime = json.loads(result.stdout)["runtime"]["operators"]["identity"]
    assert "primary unavailable" in runtime["error"]
    assert "torch_run fallback" not in runtime["error"]


def test_standalone_runtime_random_recipe_preserves_integer_bounds():
    torch = pytest.importorskip("torch")
    generator = torch.Generator(device="cpu").manual_seed(123)

    actual = _random_tensor(
        torch,
        {
            "type": "random",
            "shape": [64],
            "dtype": "int64",
            "distribution": "integer",
            "low": 3,
            "high": 7,
        },
        generator,
    )

    assert actual.dtype is torch.int64
    assert actual.min().item() >= 3
    assert actual.max().item() < 7


def test_runtime_reports_import_failure(tmp_path):
    catalog = _write_catalog(
        tmp_path / "runtime-failure",
        oracle=(
            'REFERENCE_DEVICE = "target"\n'
            "import dependency_that_does_not_exist\n"
            "def run(x): return x\n"
        ),
    )

    result = _run(
        SCRIPT,
        catalog,
        "--runtime",
        "--target-device",
        "cpu",
        "--format",
        "json",
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["runtime"]["passed"] is False
    assert "runtime.oracle_failed" in {issue["code"] for issue in report["issues"]}


def test_runtime_reports_hard_oracle_exit(tmp_path):
    catalog = _write_catalog(
        tmp_path / "runtime-hard-exit",
        oracle=(
            'REFERENCE_DEVICE = "target"\n'
            "import os\n"
            "def run(x): os._exit(17)\n"
        ),
    )

    result = _run(
        SCRIPT,
        catalog,
        "--runtime",
        "--target-device",
        "cpu",
        "--format",
        "json",
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    runtime = report["runtime"]["operators"]["identity"]
    assert runtime["passed"] is False
    assert "exitcode=17" in runtime["error"]

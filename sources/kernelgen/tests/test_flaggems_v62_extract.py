import json
import subprocess
from pathlib import Path

import pytest

from kernelgen.agents.extractor.flaggems.v62 import (
    extract_flaggems_addmm_v62,
    package_flaggems_v62_extraction,
)
from kernelgen.workflows.flaggems_v62_extract import DEFAULT_CATALOG_ROOT


_HIGH_PRECISION_REFERENCE_TEST = (
    "def test_addmm_():\n"
    "    ref_mat1 = utils.to_reference(mat1, True)\n"
    "    ref_mat2 = utils.to_reference(mat2, True)\n"
    "    ref_inp1 = utils.to_reference(inp1, True)\n"
    "    ref_inp1.addmm_(ref_mat1, ref_mat2)\n"
    "\n"
)


def test_default_catalog_root_is_kernelgen_server_data():
    expected = (
        Path(__file__).resolve().parents[2]
        / "kernelgen_server"
        / "data"
        / "flaggems-native"
    )
    assert Path(DEFAULT_CATALOG_ROOT) == expected


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "FlagGems"
    ops = repo / "src" / "flag_gems" / "ops"
    fused = repo / "src" / "flag_gems" / "fused"
    tests = repo / "tests"
    benchmark = repo / "benchmark"
    conf = repo / "conf"
    for directory in (ops, fused, tests, benchmark, conf):
        directory.mkdir(parents=True, exist_ok=True)
    (ops / "__init__.py").write_text(
        "from flag_gems.ops.addmm_ import addmm_\n__all__ = ['addmm_']\n",
        encoding="utf-8",
    )
    (fused / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    (ops / "addmm_.py").write_text(
        "def addmm_(self, mat1, mat2, *, beta=1, alpha=1): return self\n",
        encoding="utf-8",
    )
    (tests / "test_addmm_.py").write_text(
        _HIGH_PRECISION_REFERENCE_TEST
        + "if QUICK_MODE:\n"
        "    MNK_SHAPES = [(1, 1, 32)]\n"
        "else:\n"
        "    MNK_SHAPES = [(1, 1, 32), (15, 160, 1024)]\n",
        encoding="utf-8",
    )
    (tests / "accuracy_utils.py").write_text(
        "SCALARS = [0.001, -0.999]\n",
        encoding="utf-8",
    )
    (benchmark / "test_addmm_.py").write_text("# addmm_ timing\n", encoding="utf-8")
    (conf / "operators.yaml").write_text(
        "ops:\n  - id: addmm_\n    for: [addmm_]\n",
        encoding="utf-8",
    )
    return repo


def _case_report(
    *,
    dtypes=("torch.float16", "torch.float32"),
    shapes=((2, 384, 384, 384), (16, 1024, 1024, 1024)),
) -> dict:
    cases = []
    for dtype in dtypes:
        for ordinal, (b, m, n, k) in enumerate(shapes):
            cases.append(
                {
                    "case_id": (
                        "benchmark/test_addmm_.py::test_addmm_::core::"
                        f"{dtype.removeprefix('torch.')}::{ordinal}"
                    ),
                    "ordinal": ordinal,
                    "dtype": dtype,
                    "shape": {"b": b, "m": m, "n": n, "k": k},
                    "params": {"b_column_major": False},
                }
            )
    return {
        "schema_version": "flaggems.benchmark-case-list/v2",
        "benchmarks": [
            {
                "schema_version": "flaggems.benchmark-case-list/v2",
                "op_name": "addmm_",
                "phase": "timing",
                "level": "core",
                "cases": cases,
            }
        ],
    }


def _write_report(path: Path) -> None:
    path.write_text(json.dumps(_case_report()), encoding="utf-8")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_extract_addmm_writes_v62_per_operator_package(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _write_report(report)
    result = extract_flaggems_addmm_v62(
        repo,
        tmp_path / "catalog",
        case_list_path=report,
    )

    assert json.loads((result.catalog_root / "manifest.json").read_text()) == {
        "api_version": "v6.2",
        "evaluator": "native",
        "layout": "per-operator",
    }
    definition = json.loads(result.definition_path.read_text())
    assert definition["api_version"] == "v6.2"
    assert definition["name"] == "addmm_"
    assert [parameter["name"] for parameter in definition["parameters"]] == [
        "self",
        "mat1",
        "mat2",
        "beta",
        "alpha",
    ]
    assert "reference" not in definition
    assert "correctness_reference" not in definition
    assert "reference_device" not in definition
    oracle = result.oracle_path.read_text()
    assert result.oracle_path.name == "oracle.py"
    assert 'REFERENCE_DEVICE = "target"' in oracle
    assert "def timing_run(self, mat1, mat2, *, beta=1, alpha=1):" in oracle
    assert "def correctness_run(self, mat1, mat2, *, beta=1, alpha=1):" in oracle
    assert 'torch.set_float32_matmul_precision("highest")' in oracle
    assert "torch.set_float32_matmul_precision(precision)" in oracle
    assert "def gen_inputs" not in oracle
    assert "self.to(torch.float64)" not in oracle
    assert "self.to(torch.float32)" not in oracle
    assert "self.copy_(torch.addmm(self, mat1, mat2, beta=beta, alpha=alpha))" in oracle
    assert sorted(path.name for path in result.operator_root.iterdir()) == [
        "correctness.jsonl",
        "definition.json",
        "oracle.py",
        "timing.jsonl",
    ]

    correctness = _jsonl(result.correctness_path)
    assert len(correctness) == 8
    assert all("call" not in workload for workload in correctness)
    assert correctness[-1]["tolerance"]["atol_scale"] == 1024.0
    timing = _jsonl(result.timing_path)
    assert len(timing) == 4
    assert timing[0]["name"].endswith("core::float16::0")
    assert timing[0]["inputs"]["mat2"]["shape"] == [384, 384]
    assert all("call" not in workload for workload in timing)


def test_extract_addmm_invokes_native_core_case_listing(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    observed = {}

    def fake_run(command, *, cwd, text, capture_output, check):
        observed["command"] = command
        observed["cwd"] = cwd
        output = Path(command[command.index("--output") + 1])
        _write_report(output)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    extract_flaggems_addmm_v62(
        repo,
        tmp_path / "catalog",
        python_executable="target-python",
    )

    assert observed["command"][:4] == [
        "target-python",
        "-m",
        "pytest",
        "benchmark/test_addmm_.py",
    ]
    assert observed["command"][4:8] == ["--level", "core", "--list-cases", "--output"]
    assert observed["cwd"] == repo.resolve()


def test_extract_addmm_preserves_real_full_and_core_cardinality(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tests" / "test_addmm_.py").write_text(
        _HIGH_PRECISION_REFERENCE_TEST
        + "if QUICK_MODE:\n"
        "    MNK_SHAPES = [(1, 1, 32)]\n"
        "else:\n"
        "    MNK_SHAPES = [(1, 1, 32), (15, 160, 1024), (495, 5333, 71)]\n",
        encoding="utf-8",
    )
    (repo / "tests" / "accuracy_utils.py").write_text(
        "SCALARS = [0.001, -0.999, 100.001, -111.999]\n",
        encoding="utf-8",
    )
    report = tmp_path / "cases.json"
    report.write_text(
        json.dumps(
            _case_report(
                dtypes=("torch.float16", "torch.float32", "torch.bfloat16"),
                shapes=(
                    (2, 384, 384, 384),
                    (2, 4096, 4096, 4096),
                    (16, 1024, 1024, 1024),
                    (16, 2048, 2048, 2048),
                    (16, 4096, 4096, 4096),
                ),
            )
        ),
        encoding="utf-8",
    )
    result = extract_flaggems_addmm_v62(
        repo,
        tmp_path / "catalog",
        case_list_path=report,
    )

    assert result.num_correctness_workloads == 36
    assert result.num_timing_workloads == 15
    assert [workload["name"] for workload in _jsonl(result.timing_path)] == [
        case["case_id"] for case in _case_report(
            dtypes=("torch.float16", "torch.float32", "torch.bfloat16"),
            shapes=(
                (2, 384, 384, 384),
                (2, 4096, 4096, 4096),
                (16, 1024, 1024, 1024),
                (16, 2048, 2048, 2048),
                (16, 4096, 4096, 4096),
            ),
        )["benchmarks"][0]["cases"]
    ]


def test_extract_addmm_rejects_non_native_case_schema(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    value = _case_report()
    value["schema_version"] = "legacy/v1"
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark-case-list/v2"):
        extract_flaggems_addmm_v62(
            repo,
            tmp_path / "catalog",
            case_list_path=report,
        )


def test_extract_addmm_refuses_to_overwrite_existing_operator(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _write_report(report)
    root = tmp_path / "catalog"
    extract_flaggems_addmm_v62(repo, root, case_list_path=report)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        extract_flaggems_addmm_v62(repo, root, case_list_path=report)


def test_extract_addmm_rejects_changed_correctness_run_semantics(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tests" / "test_addmm_.py").write_text(
        "def test_addmm_():\n"
        "    ref_inp1.addmm_(mat1, mat2)\n"
        "MNK_SHAPES = [(1, 1, 32)]\n",
        encoding="utf-8",
    )
    report = tmp_path / "cases.json"
    _write_report(report)

    with pytest.raises(ValueError, match="recognized high-precision"):
        extract_flaggems_addmm_v62(repo, tmp_path / "catalog", case_list_path=report)


def test_extract_addmm_generates_layout_hook_only_for_listed_case(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    value = _case_report(dtypes=("torch.float16",), shapes=((2, 384, 384, 384),))
    value["benchmarks"][0]["cases"][0]["params"]["b_column_major"] = True
    report.write_text(json.dumps(value), encoding="utf-8")

    result = extract_flaggems_addmm_v62(
        repo,
        tmp_path / "catalog",
        case_list_path=report,
    )

    assert "def gen_inputs" in result.oracle_path.read_text()
    timing = _jsonl(result.timing_path)
    assert timing[0]["inputs"]["mat2"] == {
        "type": "custom",
        "shape": [384, 384],
        "dtype": "float16",
        "generator_params": {"kind": "column_major"},
    }


def _alpha_staging_catalog(tmp_path: Path) -> Path:
    root = tmp_path / "staging"
    definition_path = root / "definitions" / "pointwise" / "alpha_dropout.json"
    correctness_path = (
        root / "workloads" / "pointwise" / "alpha_dropout.correctness.jsonl"
    )
    timing_path = root / "workloads" / "pointwise" / "alpha_dropout.timing.jsonl"
    definition_path.parent.mkdir(parents=True)
    correctness_path.parent.mkdir(parents=True)
    definition = {
        "api_version": "v6.0",
        "name": "alpha_dropout",
        "description": "stochastic dropout",
        "parameters": [
            {
                "name": "input",
                "type": "Tensor",
                "kind": "positional_or_keyword",
                "required": True,
            },
            {
                "name": "p",
                "type": "float",
                "kind": "positional_or_keyword",
                "required": False,
                "default": 0.5,
            },
            {
                "name": "train",
                "type": "bool",
                "kind": "positional_or_keyword",
                "required": False,
                "default": True,
            },
        ],
        "outputs": ["out"],
        "effects": {"mutates": [], "returns_alias_of": {}, "cases": []},
        "reference": (
            "import torch\n"
            "def gen_inputs(ctx, device):\n"
            "    spec = ctx['inputs']['x']\n"
            "    return {'input': torch.randn(spec['shape'], device=device)}\n"
            "def run(input, p=0.5, train=True):\n"
            "    return torch.alpha_dropout(input, p, train)\n"
            "def valid(ref_outputs, sol_outputs, inputs, ctx):\n"
            "    return sol_outputs[0].shape == ref_outputs[0].shape\n"
        ),
        "correctness_reference": (
            "import torch\n"
            "def run(input, p=0.5, train=True):\n"
            "    return torch.alpha_dropout(input, p, train)\n"
        ),
        "reference_device": "target",
        "custom_valid_entrypoint": "valid",
    }
    definition_path.write_text(json.dumps(definition), encoding="utf-8")
    workloads = [
        {
            "name": "generated-name",
            "inputs": {
                "x": {
                    "type": "custom",
                    "shape": [64, 64],
                    "dtype": "float16",
                    "generator_params": {"kind": "ordinary"},
                },
                "probability": {"type": "literal", "value": 0.5},
                "training": {"type": "literal", "value": True},
            },
            "call": "alpha_dropout(x, p=probability, train=training)",
            "seed": 0,
        },
        {
            "name": "generated-name-2",
            "inputs": {
                "x": {
                    "type": "custom",
                    "shape": [128, 128],
                    "dtype": "float16",
                    "generator_params": {"kind": "ordinary"},
                },
                "probability": {"type": "literal", "value": 0.5},
                "training": {"type": "literal", "value": True},
            },
            "call": "alpha_dropout(x, p=probability, train=training)",
            "seed": 0,
        },
    ]
    correctness_path.write_text(json.dumps(workloads[0]) + "\n", encoding="utf-8")
    timing_path.write_text(
        "\n".join(json.dumps(value) for value in workloads) + "\n",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "api_version": "v6.0",
                "operators": [
                    {
                        "id": "alpha_dropout",
                        "name": "alpha_dropout",
                        "definition": "definitions/pointwise/alpha_dropout.json",
                        "correctness_workloads": (
                            "workloads/pointwise/alpha_dropout.correctness.jsonl"
                        ),
                        "timing_workloads": (
                            "workloads/pointwise/alpha_dropout.timing.jsonl"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def _alpha_case_report() -> dict:
    return {
        "schema_version": "flaggems.benchmark-case-list/v2",
        "benchmarks": [
            {
                "schema_version": "flaggems.benchmark-case-list/v2",
                "op_name": "alpha_dropout",
                "phase": "timing",
                "level": "core",
                "cases": [
                    {
                        "case_id": "benchmark/test_alpha_dropout.py::core::float16::0",
                        "dtype": "torch.float16",
                        "shape": {"input": [64, 64]},
                        "params": {"p": 0.5, "train": True},
                    },
                    {
                        "case_id": "benchmark/test_alpha_dropout.py::core::float16::1",
                        "dtype": "torch.float16",
                        "shape": {"input": [128, 128]},
                        "params": {"p": 0.5, "train": True},
                    },
                ],
            }
        ],
    }


def test_package_agent_extraction_as_v62_and_bind_list_case_ids(tmp_path):
    staging = _alpha_staging_catalog(tmp_path)
    report = tmp_path / "alpha-cases.json"
    report.write_text(json.dumps(_alpha_case_report()), encoding="utf-8")

    result = package_flaggems_v62_extraction(
        staging,
        tmp_path / "catalog",
        "alpha_dropout",
        case_list_path=report,
    )

    definition = json.loads(result.definition_path.read_text())
    assert definition["api_version"] == "v6.2"
    assert definition["parameters"][0]["type_hint"] == "Tensor"
    assert "reference" not in definition
    oracle = result.oracle_path.read_text()
    assert oracle.count("REFERENCE_DEVICE") == 1
    assert "timing_run = _timing_oracle.run" in oracle
    assert "correctness_run = _correctness_oracle.run" in oracle
    assert "valid = _timing_oracle.valid" in oracle
    assert "aliases.get(name, name)" in oracle
    timing = _jsonl(result.timing_path)
    assert [workload["name"] for workload in timing] == [
        case["case_id"] for case in _alpha_case_report()["benchmarks"][0]["cases"]
    ]
    assert timing[0]["inputs"]["input"]["type"] == "custom"
    assert timing[0]["inputs"]["_kgs_input_aliases"] == {
        "input": "x",
        "p": "probability",
        "train": "training",
    }
    assert all("call" not in workload for workload in timing)


def test_package_agent_extraction_rejects_timing_cardinality_drift(tmp_path):
    staging = _alpha_staging_catalog(tmp_path)
    report = tmp_path / "alpha-cases.json"
    value = _alpha_case_report()
    value["benchmarks"][0]["cases"].pop()
    report.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="--list-cases returned 1"):
        package_flaggems_v62_extraction(
            staging,
            tmp_path / "catalog",
            "alpha_dropout",
            case_list_path=report,
        )

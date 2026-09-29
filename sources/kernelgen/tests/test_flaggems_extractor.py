"""Unit tests for the FlagGems v6 extractor contract and hard gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelgen.agents.extractor.flaggems import (
    FlagGemsExtractorAgent,
    FlagGemsExtractorInput,
    FlagGemsExtractorOutput,
    OperatorNotImplementedError,
    OperatorSourceNotFoundError,
)
from kernelgen.agents.extractor.flaggems.source_inventory import (
    collect_source_inventory,
)
from kernelgen.agents.extractor.flaggems.source_validation import (
    pytest_requires_equal_nan,
    validate_equal_nan_coverage,
)
from kernelgen.examples.flaggems_extract.run_example import prepare_native_agent
from kernelgen.framework.runtime.base import FakeRuntime
from kernelgen.workflows.flaggems_extract import FlagGemsExtractWorkflow


def _parameter(
    name: str,
    type_: str = "Tensor",
    *,
    kind: str = "positional_or_keyword",
    required: bool | None = True,
    default=...,
) -> dict:
    result = {"name": name, "type": type_, "kind": kind}
    if kind not in {"var_positional", "var_keyword"}:
        result["required"] = required
        if default is not ...:
            result["default"] = default
    return result


def _definition(
    *,
    name: str = "gelu",
    parameters: list[dict] | None = None,
    outputs: list[str] | None = None,
    effects: dict | None = None,
    reference: str = "import torch\n\ndef run(x):\n    return torch.nn.functional.gelu(x)\n",
) -> dict:
    return {
        "api_version": "v6.0",
        "name": name,
        "description": f"Extracted from FlagGems {name} pytest and benchmark.",
        "parameters": parameters or [_parameter("x")],
        "outputs": outputs or ["output"],
        "effects": effects or {"mutates": [], "returns_alias_of": {}},
        "reference": reference,
        "reference_device": "target",
    }


def _workload(
    name: str = "gelu-corr-0",
    inputs: dict | None = None,
    call: str = "gelu(x)",
) -> dict:
    return {
        "name": name,
        "inputs": inputs
        or {"x": {"type": "random", "shape": [64, 64], "dtype": "float32"}},
        "call": call,
        "seed": 0,
    }


def _output() -> dict:
    return {
        "results": [
            {
                "group": "pointwise",
                "definition": _definition(),
                "correctness_workloads": [_workload()],
                "timing_workloads": [_workload("gelu-time-0")],
            }
        ]
    }


def _fake_repo(tmp_path: Path) -> Path:
    ops = tmp_path / "src" / "flag_gems" / "ops"
    tests = tmp_path / "tests"
    benchmark = tmp_path / "benchmark"
    conf = tmp_path / "conf"
    for directory in (ops, tests, benchmark, conf):
        directory.mkdir(parents=True, exist_ok=True)
    (ops / "__init__.py").write_text(
        "from flag_gems.ops.unbind_copy import unbind_copy\n"
        "__all__ = ['unbind_copy']\n",
        encoding="utf-8",
    )
    (ops / "unbind_copy.py").write_text(
        "def unbind_copy(input, dim=0): return input\n",
        encoding="utf-8",
    )
    (tests / "test_unbind_copy.py").write_text("# pytest\n", encoding="utf-8")
    (tests / "test_einsum.py").write_text("# pytest\n", encoding="utf-8")
    (benchmark / "test_einsum.py").write_text("# benchmark\n", encoding="utf-8")
    (conf / "operators.yaml").write_text(
        "ops:\n"
        "  - id: clone\n"
        "    for: [clone]\n"
        "  - id: einsum\n"
        "    for: [einsum]\n"
        "  - id: unbind_copy\n"
        "    for: [unbind_copy]\n",
        encoding="utf-8",
    )
    return tmp_path


def _addmm_output() -> dict:
    definition = _definition(
        name="addmm_",
        parameters=[
            _parameter("self"),
            _parameter("mat1"),
            _parameter("mat2"),
            _parameter(
                "beta",
                "Scalar",
                kind="keyword_only",
                required=False,
                default=1,
            ),
            _parameter(
                "alpha",
                "Scalar",
                kind="keyword_only",
                required=False,
                default=1,
            ),
        ],
        effects={
            "mutates": ["self"],
            "returns_alias_of": {"out": "self"},
        },
        outputs=["out"],
        reference=(
            "import torch\n\n"
            "def run(self, mat1, mat2, *, beta=1, alpha=1):\n"
            "    return self.addmm_(mat1, mat2, beta=beta, alpha=alpha)\n"
        ),
    )
    tensors = {
        "self": {"type": "random", "shape": [2, 3], "dtype": "float32"},
        "mat1": {"type": "random", "shape": [2, 4], "dtype": "float32"},
        "mat2": {"type": "random", "shape": [4, 3], "dtype": "float32"},
    }
    core_shapes = [
        (2, 384, 384, 384),
        (2, 4096, 4096, 4096),
        (16, 1024, 1024, 1024),
        (16, 2048, 2048, 2048),
        (16, 4096, 4096, 4096),
    ]
    timing_workloads = []
    for dtype in ("float16", "float32", "bfloat16"):
        for index, (_, m, n, k) in enumerate(core_shapes):
            timing_workloads.append(
                _workload(
                    f"addmm_-time-{dtype}-{index}",
                    {
                        "self": {
                            "type": "random",
                            "shape": [m, n],
                            "dtype": dtype,
                        },
                        "mat1": {
                            "type": "random",
                            "shape": [m, k],
                            "dtype": dtype,
                        },
                        "mat2": {
                            "type": "random",
                            "shape": [k, n],
                            "dtype": dtype,
                        },
                    },
                    "addmm_(self, mat1, mat2)",
                )
            )
    return {
        "results": [
            {
                "group": "gemm",
                "definition": definition,
                "correctness_workloads": [
                    _workload(
                        "addmm_-corr-explicit",
                        {
                            **tensors,
                            "beta": {"type": "scalar", "value": 0.001},
                            "alpha": {"type": "scalar", "value": 0.001},
                        },
                        "addmm_(self, mat1, mat2, beta=beta, alpha=alpha)",
                    )
                ],
                "timing_workloads": timing_workloads,
            }
        ]
    }


def _fake_addmm_repo(tmp_path: Path) -> Path:
    repo = _fake_repo(tmp_path)
    ops = repo / "src" / "flag_gems" / "ops"
    (ops / "__init__.py").write_text(
        "from flag_gems.ops.addmm_ import addmm_\n"
        "__all__ = ['addmm_']\n",
        encoding="utf-8",
    )
    (ops / "addmm_.py").write_text(
        "def addmm_(self, mat1, mat2, *, beta=1, alpha=1): return self\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_addmm_.py").write_text(
        "# pytest-backed addmm_ correctness\n",
        encoding="utf-8",
    )
    (repo / "benchmark" / "test_addmm_.py").write_text(
        "from . import base, consts\n\n"
        "def _input_fn(b, m, n, k, dtype, device, b_column_major):\n"
        "    yield None\n\n"
        "bench = base.BlasBenchmark(\n"
        "    op_name='addmm_', input_fn=_input_fn, dtypes=consts.FLOAT_DTYPES\n"
        ")\n",
        encoding="utf-8",
    )
    (repo / "benchmark" / "consts.py").write_text(
        "import torch\n\n"
        "FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]\n",
        encoding="utf-8",
    )
    (repo / "benchmark" / "core_shapes.yaml").write_text(
        "BlasBenchmark:\n"
        "  shapes:\n"
        "    - [2, 384, 384, 384]\n"
        "    - [2, 4096, 4096, 4096]\n"
        "    - [16, 1024, 1024, 1024]\n"
        "    - [16, 2048, 2048, 2048]\n"
        "    - [16, 4096, 4096, 4096]\n"
        "  shape_desc: B, M, N, K\n",
        encoding="utf-8",
    )
    with (repo / "conf" / "operators.yaml").open("a", encoding="utf-8") as file:
        file.write("  - id: addmm_\n    for: [addmm_]\n")
    return repo


def test_blas_timing_requires_exact_core_sequence_and_rejects_column_major(
    tmp_path,
):
    repo = _fake_addmm_repo(tmp_path)
    agent = FlagGemsExtractorAgent()
    prompt = agent.preprocess(
        FlagGemsExtractorInput(operator="addmm_", flaggems_repo=str(repo)),
        FakeRuntime([]),
    )
    assert "Mandatory core timing profile" in prompt
    assert "Deterministically resolved addmm_ core sequence" in prompt
    assert '"expected_workloads": 15' in prompt
    assert "b_column_major=False" in prompt

    value = _addmm_output()
    agent.postprocess(json.dumps(value), FakeRuntime([]))

    missing_duplicate = json.loads(json.dumps(value))
    missing_duplicate["results"][0]["timing_workloads"].pop(4)
    with pytest.raises(ValueError, match="requires exactly 15 timing workloads"):
        agent.postprocess(json.dumps(missing_duplicate), FakeRuntime([]))

    wrong_order = json.loads(json.dumps(value))
    timing = wrong_order["results"][0]["timing_workloads"]
    timing[0], timing[1] = timing[1], timing[0]
    with pytest.raises(ValueError, match="core timing index 0 input"):
        agent.postprocess(json.dumps(wrong_order), FakeRuntime([]))

    value["results"][0]["definition"]["reference"] = (
        "import torch\n\n"
        "def gen_inputs(ctx, device):\n"
        "    values = dict(ctx['values'])\n"
        "    spec = ctx['specs'].get('mat2')\n"
        "    if spec and spec.get('generator_params', {}).get('column_major'):\n"
        "        generator = torch.Generator(device='cpu')\n"
        "        generator.manual_seed(ctx['seed'])\n"
        "        base = torch.randn((3, 4), device='cpu', generator=generator)\n"
        "        values['mat2'] = base.t().to(device)\n"
        "    return values\n\n"
        "def run(self, mat1, mat2, *, beta=1, alpha=1):\n"
        "    return self.addmm_(mat1, mat2, beta=beta, alpha=alpha)\n"
    )
    value["results"][0]["timing_workloads"][0]["inputs"]["mat2"] = {
        "type": "custom",
        "shape": [384, 384],
        "dtype": "float16",
        "generator_params": {"column_major": True},
    }
    with pytest.raises(ValueError, match="does not enter b_column_major=True"):
        agent.postprocess(json.dumps(value), FakeRuntime([]))


def test_target_list_cases_drives_timing_names_and_cardinality(tmp_path):
    repo = _fake_addmm_repo(tmp_path)
    value = _addmm_output()
    cases = []
    for ordinal, workload in enumerate(value["results"][0]["timing_workloads"]):
        dtype = workload["inputs"]["self"]["dtype"]
        cases.append(
            {
                "case_id": f"benchmark/test_addmm_.py::core::{dtype}::{ordinal}",
                "ordinal": ordinal,
                "dtype": f"torch.{dtype}",
                "shape": {},
                "params": {"b_column_major": False},
            }
        )
    report = tmp_path / "cases.json"
    report.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    agent = FlagGemsExtractorAgent()
    prompt = agent.preprocess(
        FlagGemsExtractorInput(
            operator="addmm_",
            flaggems_repo=str(repo),
            case_list_path=str(report),
        ),
        FakeRuntime([]),
    )
    assert "Authoritative target `--list-cases` sequence" in prompt

    output = agent.postprocess(json.dumps(value), FakeRuntime([]))

    assert [
        workload.name for workload in output.results[0].timing_workloads
    ] == [case["case_id"] for case in cases]

    cases.pop()
    report.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    agent.preprocess(
        FlagGemsExtractorInput(
            operator="addmm_",
            flaggems_repo=str(repo),
            case_list_path=str(report),
        ),
        FakeRuntime([]),
    )
    with pytest.raises(ValueError, match="--list-cases returned 14"):
        agent.postprocess(json.dumps(value), FakeRuntime([]))


def test_pytest_keyword_scalar_values_are_mandatory(tmp_path):
    repo = _fake_addmm_repo(tmp_path)
    (repo / "tests" / "test_addmm_.py").write_text(
        "import pytest\n\n"
        "SCALARS = [0.001, -0.999]\n\n"
        "@pytest.mark.parametrize('scalar', SCALARS)\n"
        "def test_addmm_(self, mat1, mat2, scalar):\n"
        "    alpha = beta = scalar\n"
        "    self.addmm_(mat1, mat2, alpha=alpha, beta=beta)\n",
        encoding="utf-8",
    )
    agent = FlagGemsExtractorAgent()
    prompt = agent.preprocess(
        FlagGemsExtractorInput(operator="addmm_", flaggems_repo=str(repo)),
        FakeRuntime([]),
    )
    assert "Mandatory pytest keyword-value coverage" in prompt
    assert "-0.999" in prompt
    with pytest.raises(ValueError, match="omit source pytest parametrized keyword"):
        agent.postprocess(json.dumps(_addmm_output()), FakeRuntime([]))


def test_source_inventory_is_independent_of_prompt_and_schema(tmp_path):
    repo = _fake_repo(tmp_path)
    inventory = collect_source_inventory(str(repo), "einsum")
    assert inventory.operator == "einsum"
    assert inventory.test_file == repo / "tests" / "test_einsum.py"
    assert inventory.benchmark_files == (repo / "benchmark" / "test_einsum.py",)
    assert inventory.implementation_status.startswith("pytest baseline only")
    assert inventory.public_signatures == ()

    exported = collect_source_inventory(str(repo), "unbind_copy")
    assert len(exported.public_signatures) == 1
    assert exported.public_signatures[0].public_name == "unbind_copy"
    assert "def unbind_copy(input, dim=0)" in exported.public_signatures[0].source


def test_source_equal_nan_assertion_requires_workload_tolerance(tmp_path):
    test_file = tmp_path / "test_softmax.py"
    test_file.write_text(
        "def test_softmax(x):\n"
        "    ref = torch.softmax(x, 0)\n"
        "    out = torch.softmax(x, 0)\n"
        "    gems_assert_close(out, ref, equal_nan=True)\n",
        encoding="utf-8",
    )
    assert pytest_requires_equal_nan("softmax", [test_file])

    value = _output()
    result = FlagGemsExtractorOutput.model_validate(value).results[0]
    with pytest.raises(ValueError, match="equal_nan=True"):
        validate_equal_nan_coverage(result, True)

    value["results"][0]["correctness_workloads"][0]["tolerance"] = {
        "equal_nan": True
    }
    result = FlagGemsExtractorOutput.model_validate(value).results[0]
    validate_equal_nan_coverage(result, True)


def test_source_inventory_resolves_export_alias_signature(tmp_path):
    repo = _fake_repo(tmp_path)
    ops = repo / "src" / "flag_gems" / "ops"
    (ops / "__init__.py").write_text(
        "from flag_gems.ops.factor import ldl_factor as linalg_ldl_factor\n"
        "__all__ = ['linalg_ldl_factor']\n",
        encoding="utf-8",
    )
    (ops / "factor.py").write_text(
        "def ldl_factor(input, hermitian=False): return input\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_linalg_ldl_factor.py").write_text(
        "# pytest\n",
        encoding="utf-8",
    )
    with (repo / "conf" / "operators.yaml").open("a", encoding="utf-8") as file:
        file.write(
            "  - id: linalg_ldl_factor\n"
            "    for: [linalg_ldl_factor]\n"
        )

    inventory = collect_source_inventory(str(repo), "linalg_ldl_factor")

    assert inventory.implementation_file == ops / "factor.py"
    assert len(inventory.public_signatures) == 1
    signature = inventory.public_signatures[0]
    assert signature.public_name == "linalg_ldl_factor"
    assert signature.implementation_name == "ldl_factor"
    assert "def linalg_ldl_factor(input, hermitian=False)" in signature.source


def test_source_inventory_resolves_top_level_registered_alias(tmp_path):
    repo = _fake_repo(tmp_path)
    package = repo / "src" / "flag_gems"
    ops = package / "ops"
    (ops / "__init__.py").write_text(
        "from flag_gems.ops.ge import ge, ge_scalar\n"
        "from flag_gems.ops.greater_equal import greater_equal_\n"
        "__all__ = ['ge', 'ge_scalar', 'greater_equal_']\n",
        encoding="utf-8",
    )
    (ops / "ge.py").write_text(
        "def ge(A, B): return A\n\ndef ge_scalar(A, B): return A\n",
        encoding="utf-8",
    )
    (ops / "greater_equal.py").write_text(
        "def greater_equal_(A, B): return A\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        "from flag_gems.ops import *\n"
        "_FULL_CONFIG = (\n"
        "    ('greater_equal.Scalar', ge_scalar),\n"
        "    ('greater_equal.Tensor', ge),\n"
        "    ('greater_equal_.Tensor', greater_equal_),\n"
        ")\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_greater_equal.py").write_text(
        "# pytest\n", encoding="utf-8"
    )
    with (repo / "conf" / "operators.yaml").open("a", encoding="utf-8") as file:
        file.write(
            "  - id: greater_equal_\n"
            "    for: [greater_equal_.Tensor]\n"
        )

    inventory = collect_source_inventory(str(repo), "greater_equal")

    assert inventory.implementation_file == ops / "ge.py"
    assert inventory.implemented is True
    assert len(inventory.public_signatures) == 1
    signature = inventory.public_signatures[0]
    assert signature.public_name == "greater_equal"
    assert signature.implementation_name == "ge"
    assert "def greater_equal(A, B)" in signature.source


def test_source_inventory_prefers_public_export_over_same_named_file(tmp_path):
    repo = _fake_repo(tmp_path)
    ops = repo / "src" / "flag_gems" / "ops"
    (ops / "__init__.py").write_text(
        "from flag_gems.ops.nextafter import nextafter_\n"
        "__all__ = ['nextafter_']\n",
        encoding="utf-8",
    )
    (ops / "nextafter.py").write_text(
        "def nextafter_(input, other): return input\n",
        encoding="utf-8",
    )
    (ops / "nextafter_.py").write_text(
        "def nextafter_(A, B): return A\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_nextafter_.py").write_text(
        "# pytest\n", encoding="utf-8"
    )
    with (repo / "conf" / "operators.yaml").open("a", encoding="utf-8") as file:
        file.write("  - id: nextafter_\n    for: [nextafter_]\n")

    inventory = collect_source_inventory(str(repo), "nextafter_")

    assert inventory.implementation_file == ops / "nextafter.py"
    assert len(inventory.public_signatures) == 1
    assert "def nextafter_(input, other)" in inventory.public_signatures[0].source


def test_source_inventory_limits_signatures_to_requested_public_callable(tmp_path):
    repo = _fake_repo(tmp_path)
    ops = repo / "src" / "flag_gems" / "ops"
    (ops / "__init__.py").write_text(
        "from flag_gems.ops.shared import first, second\n"
        "__all__ = ['first', 'second']\n",
        encoding="utf-8",
    )
    (ops / "shared.py").write_text(
        "def first(x): return x\n\ndef second(x, dim=0): return x\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_first.py").write_text("# pytest\n", encoding="utf-8")
    with (repo / "conf" / "operators.yaml").open("a", encoding="utf-8") as file:
        file.write("  - id: first\n    for: [first]\n")

    inventory = collect_source_inventory(str(repo), "first")

    assert [signature.public_name for signature in inventory.public_signatures] == [
        "first"
    ]


def test_source_inventory_resolves_inherited_benchmark_shapes(tmp_path):
    repo = _fake_repo(tmp_path)
    (repo / "benchmark" / "test_einsum.py").write_text(
        "from . import base\n\nclass EinsumBenchmark(base.BlasBenchmark): pass\n\n"
        "def test_einsum():\n"
        "    EinsumBenchmark(op_name='einsum', input_fn=lambda: None)\n",
        encoding="utf-8",
    )
    (repo / "benchmark" / "core_shapes.yaml").write_text(
        "BlasBenchmark:\n"
        "  shapes:\n"
        "    - [2, 384, 384, 384]\n"
        "  shape_desc: B, M, N, K\n",
        encoding="utf-8",
    )

    inventory = collect_source_inventory(str(repo), "einsum")
    shapes = json.loads(inventory.benchmark_shapes)

    assert shapes["resolved_key"] == "BlasBenchmark"
    assert shapes["shapes"] == [[2, 384, 384, 384]]


def test_source_inventory_prefers_benchmark_set_shapes_override(tmp_path):
    repo = _fake_repo(tmp_path)
    (repo / "benchmark" / "test_einsum.py").write_text(
        "from . import base\n\n"
        "EINSUM_SHAPES = [(1024, 256), (4096, 64)]\n\n"
        "class EinsumBenchmark(base.GenericBenchmarkExcluse1D):\n"
        "    def set_shapes(self, shape_file_path=None):\n"
        "        self.shapes = EINSUM_SHAPES\n\n"
        "def test_einsum():\n"
        "    EinsumBenchmark(op_name='einsum', input_fn=lambda: None)\n",
        encoding="utf-8",
    )
    (repo / "benchmark" / "core_shapes.yaml").write_text(
        "GenericBenchmarkExcluse1D:\n"
        "  shapes:\n"
        "    - [64, 64]\n",
        encoding="utf-8",
    )

    inventory = collect_source_inventory(str(repo), "einsum")
    shapes = json.loads(inventory.benchmark_shapes)

    assert shapes["resolved_key"] == "EinsumBenchmark"
    assert shapes["resolved_from"] == "benchmark set_shapes override"
    assert shapes["shapes"] == [[1024, 256], [4096, 64]]


def test_source_inventory_resolves_direct_generic_benchmark_mro(tmp_path):
    repo = _fake_repo(tmp_path)
    (repo / "benchmark" / "base.py").write_text(
        "class Benchmark: pass\n\n"
        "class GenericBenchmark(Benchmark): pass\n",
        encoding="utf-8",
    )
    (repo / "benchmark" / "test_einsum.py").write_text(
        "from . import base\n\n"
        "def test_einsum():\n"
        "    base.GenericBenchmark(op_name='einsum', input_fn=lambda: None)\n",
        encoding="utf-8",
    )
    (repo / "benchmark" / "core_shapes.yaml").write_text(
        "Benchmark:\n"
        "  shapes:\n"
        "    - [64, 64]\n",
        encoding="utf-8",
    )

    inventory = collect_source_inventory(str(repo), "einsum")
    shapes = json.loads(inventory.benchmark_shapes)

    assert shapes["resolved_key"] == "Benchmark"
    assert shapes["shapes"] == [[64, 64]]


def test_v6_output_validates():
    output = FlagGemsExtractorOutput.model_validate(_output())
    result = output.results[0]
    assert result.definition.api_version == "v6.0"
    assert result.definition.name == "gelu"
    assert [parameter.name for parameter in result.definition.parameters] == ["x"]
    assert result.correctness_workloads[0].call == "gelu(x)"


@pytest.mark.parametrize("field", ["reference", "correctness_reference"])
def test_reference_sources_cannot_depend_on_flaggems(field):
    value = _output()
    value["results"][0]["definition"][field] = (
        "import flag_gems\n\ndef run(x):\n    return x\n"
    )

    with pytest.raises(ValueError, match="standalone.*cannot import flag_gems"):
        FlagGemsExtractorOutput.model_validate(value)


def test_addmm_preserves_keyword_only_abi_and_tests_omitted_defaults():
    output = FlagGemsExtractorOutput.model_validate(_addmm_output())
    definition = output.results[0].definition
    assert [(p.name, p.kind, p.required, p.default) for p in definition.parameters] == [
        ("self", "positional_or_keyword", True, None),
        ("mat1", "positional_or_keyword", True, None),
        ("mat2", "positional_or_keyword", True, None),
        ("beta", "keyword_only", False, 1),
        ("alpha", "keyword_only", False, 1),
    ]
    assert set(output.results[0].timing_workloads[0].inputs) == {
        "self",
        "mat1",
        "mat2",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: d["parameters"][3].update(kind="positional_or_keyword"), "ABI differs"),
        (lambda d: d["parameters"][3].update(default=2), "default differs"),
        (lambda d: d["parameters"][3].update(name="scale"), "ABI differs"),
        (lambda d: d["parameters"].reverse(), "ABI order"),
    ],
)
def test_run_and_parameters_abi_must_match(mutation, message):
    value = _addmm_output()
    mutation(value["results"][0]["definition"])
    with pytest.raises(ValueError, match=message):
        FlagGemsExtractorOutput.model_validate(value)


def test_old_prefixed_name_and_v5_fields_are_rejected():
    value = _output()
    value["results"][0]["definition"]["name"] = "flaggems_gelu"
    value["results"][0]["correctness_workloads"][0]["call"] = "flaggems_gelu(x)"
    value["results"][0]["timing_workloads"][0]["call"] = "flaggems_gelu(x)"
    with pytest.raises(ValueError, match="real public callable"):
        FlagGemsExtractorOutput.model_validate(value)

    value = _output()
    value["results"][0]["definition"]["custom_inputs_entrypoint"] = "gen_inputs"
    with pytest.raises(ValueError, match="extra"):
        FlagGemsExtractorOutput.model_validate(value)


def test_record_id_distinguishes_specializations_without_renaming_callable():
    value = _output()
    first = value["results"][0]
    first["record_id"] = "gelu_tensor"
    second = json.loads(json.dumps(first))
    second["record_id"] = "gelu_scalar"
    second["correctness_workloads"][0]["name"] = "gelu-scalar-corr"
    second["timing_workloads"][0]["name"] = "gelu-scalar-time"
    value["results"].append(second)

    output = FlagGemsExtractorOutput.model_validate(value)

    assert [result.catalog_id for result in output.results] == [
        "gelu_tensor",
        "gelu_scalar",
    ]
    assert {result.definition.name for result in output.results} == {"gelu"}

    value["results"][1]["record_id"] = "gelu_tensor"
    with pytest.raises(ValueError, match="duplicate catalog record_id"):
        FlagGemsExtractorOutput.model_validate(value)


def test_parameter_required_default_and_type_rules_are_strict():
    value = _output()
    parameter = value["results"][0]["definition"]["parameters"][0]
    del parameter["type"]
    with pytest.raises(ValueError, match="ordinary parameters require an ABI type"):
        FlagGemsExtractorOutput.model_validate(value)

    value = _output()
    parameter = value["results"][0]["definition"]["parameters"][0]
    parameter.update(required=False, default=None)
    with pytest.raises(ValueError, match=r"default null requires Optional"):
        FlagGemsExtractorOutput.model_validate(value)

    value = _output()
    parameter = value["results"][0]["definition"]["parameters"][0]
    parameter["type"] = "Any"
    with pytest.raises(ValueError):
        FlagGemsExtractorOutput.model_validate(value)


def test_non_json_framework_default_is_normalized_and_checked():
    value = _output()
    definition = value["results"][0]["definition"]
    definition["parameters"].append(
        _parameter(
            "layout",
            "layout",
            kind="keyword_only",
            required=False,
            default="strided",
        )
    )
    definition["reference"] = (
        "import torch\n\n"
        "def run(x, *, layout=torch.strided):\n"
        "    return torch.nn.functional.gelu(x)\n"
    )
    FlagGemsExtractorOutput.model_validate(value)
    definition["parameters"][-1]["default"] = "sparse_coo"
    with pytest.raises(ValueError, match="default differs"):
        FlagGemsExtractorOutput.model_validate(value)


def test_custom_recipe_has_optional_top_level_shape_dtype_and_fixed_hook():
    value = _output()
    definition = value["results"][0]["definition"]
    definition["parameters"] = [_parameter("tensors", "List[Tensor]")]
    definition["reference"] = (
        "import torch\n\n"
        "def gen_inputs(ctx, device):\n"
        "    specs = ctx['specs']['tensors']['generator_params']['items']\n"
        "    return {'tensors': [torch.empty(tuple(s['shape']), device=device) for s in specs]}\n\n"
        "def run(tensors):\n"
        "    return torch.cat(tensors)\n"
    )
    custom = {
        "type": "custom",
        "generator_params": {
            "items": [
                {"shape": [1, 4], "dtype": "float32"},
                {"shape": [2, 4], "dtype": "float32"},
            ]
        },
    }
    for workload in (
        value["results"][0]["correctness_workloads"][0],
        value["results"][0]["timing_workloads"][0],
    ):
        workload["inputs"] = {"tensors": custom}
        workload["call"] = "gelu(tensors)"
    definition["name"] = "gelu"
    output = FlagGemsExtractorOutput.model_validate(value)
    assert output.results[0].correctness_workloads[0].inputs["tensors"].shape is None

    del definition["reference"]
    definition["reference"] = "def run(tensors):\n    return tensors\n"
    with pytest.raises(ValueError, match="require fixed gen_inputs"):
        FlagGemsExtractorOutput.model_validate(value)


def test_custom_requires_generator_params_and_exact_gen_inputs_signature():
    value = _output()
    spec = value["results"][0]["correctness_workloads"][0]["inputs"]["x"]
    spec["type"] = "custom"
    with pytest.raises(ValueError, match="generator_params"):
        FlagGemsExtractorOutput.model_validate(value)

    spec["generator_params"] = {}
    definition = value["results"][0]["definition"]
    definition["reference"] = (
        "def gen_inputs(ctx):\n    return {}\n\n"
        "def run(x):\n    return x\n"
    )
    with pytest.raises(ValueError, match="exact signature"):
        FlagGemsExtractorOutput.model_validate(value)


def test_custom_random_base_must_be_generated_on_cpu_before_device_transfer():
    value = _output()
    spec = value["results"][0]["correctness_workloads"][0]["inputs"]["x"]
    spec["type"] = "custom"
    spec["generator_params"] = {"shape": [64, 64]}
    definition = value["results"][0]["definition"]
    definition["reference"] = (
        "import torch\n\n"
        "def gen_inputs(ctx, device):\n"
        "    gen = torch.Generator(device='cpu').manual_seed(ctx['seed'])\n"
        "    return {'x': torch.randn((64, 64), device=device, generator=gen)}\n\n"
        "def run(x):\n"
        "    return torch.nn.functional.gelu(x)\n"
    )
    with pytest.raises(ValueError, match="generate its base value on CPU"):
        FlagGemsExtractorOutput.model_validate(value)

    definition["reference"] = (
        "import torch\n\n"
        "def gen_inputs(ctx, device):\n"
        "    gen = torch.Generator(device='cpu').manual_seed(ctx['seed'])\n"
        "    base = torch.randn((64, 64), device='cpu', generator=gen)\n"
        "    return {'x': base.to(device)}\n\n"
        "def run(x):\n"
        "    return torch.nn.functional.gelu(x)\n"
    )
    FlagGemsExtractorOutput.model_validate(value)


def test_arbitrary_tensor_tuple_is_one_var_positional_parameter_and_is_expanded():
    reference = (
        "import torch\n\n"
        "def gen_inputs(ctx, device):\n"
        "    items = ctx['specs']['tensors']['generator_params']['items']\n"
        "    return {'tensors': tuple(torch.empty(tuple(x['shape']), device=device) for x in items)}\n\n"
        "def run(*tensors):\n"
        "    return torch.broadcast_tensors(*tensors)\n"
    )
    custom = {
        "type": "custom",
        "generator_params": {
            "items": [
                {"shape": [1, 7], "dtype": "float32"},
                {"shape": [2, 1, 7], "dtype": "float32"},
            ]
        },
    }
    value = {
        "results": [
            {
                "group": "shape",
                "definition": _definition(
                    name="broadcast_tensors",
                    parameters=[
                        _parameter(
                            "tensors",
                            "Tuple[Tensor,...]",
                            kind="var_positional",
                            required=None,
                        )
                    ],
                    reference=reference,
                ),
                "correctness_workloads": [
                    _workload(
                        "broadcast-corr",
                        {"tensors": custom},
                        "broadcast_tensors(*tensors)",
                    )
                ],
                "timing_workloads": [],
            }
        ]
    }
    output = FlagGemsExtractorOutput.model_validate(value)
    assert output.results[0].definition.parameters[0].kind == "var_positional"

    value["results"][0]["correctness_workloads"][0]["call"] = (
        "broadcast_tensors(tensors)"
    )
    with pytest.raises(ValueError, match="must be expanded"):
        FlagGemsExtractorOutput.model_validate(value)


def test_framework_values_and_scalar_tuple_are_json_recipes_materialized_by_hook():
    parameters = [
        _parameter("x"),
        _parameter("size", "Tuple[int,...]"),
        _parameter(
            "dtype",
            "Optional[dtype]",
            kind="keyword_only",
            required=False,
            default=None,
        ),
        _parameter(
            "device",
            "Optional[device]",
            kind="keyword_only",
            required=False,
            default=None,
        ),
    ]
    reference = (
        "import torch\n\n"
        "def gen_inputs(ctx, device):\n"
        "    values = ctx['values']\n"
        "    return {\n"
        "        'size': tuple(values['size']),\n"
        "        'dtype_arg': getattr(torch, values['dtype_arg']),\n"
        "        'device_arg': torch.device(device),\n"
        "    }\n\n"
        "def run(x, size, *, dtype=None, device=None):\n"
        "    return x.new_ones(size, dtype=dtype, device=device)\n"
    )
    inputs = {
        "x": {"type": "random", "shape": [2], "dtype": "float32"},
        "size": {"type": "literal", "value": [2, 3, 4, 5]},
        "dtype_arg": {"type": "literal", "value": "float32"},
        "device_arg": {"type": "literal", "value": "target"},
    }
    value = {
        "results": [
            {
                "group": "shape",
                "definition": _definition(
                    name="new_ones",
                    parameters=parameters,
                    reference=reference,
                ),
                "correctness_workloads": [
                    _workload(
                        "new-ones-corr",
                        inputs,
                        "new_ones(x, size, dtype=dtype_arg, device=device_arg)",
                    )
                ],
                "timing_workloads": [],
            }
        ]
    }
    output = FlagGemsExtractorOutput.model_validate(value)
    assert output.results[0].correctness_workloads[0].inputs["size"].value == [
        2,
        3,
        4,
        5,
    ]

    value["results"][0]["definition"]["reference"] = (
        "def run(x, size, *, dtype=None, device=None):\n    return x\n"
    )
    with pytest.raises(ValueError, match="require fixed gen_inputs"):
        FlagGemsExtractorOutput.model_validate(value)


def test_literal_value_must_match_declared_abi_type():
    value = _output()
    parameter = value["results"][0]["definition"]["parameters"][0]
    parameter["type"] = "int"
    value["results"][0]["definition"]["reference"] = "def run(x):\n    return x\n"
    for workload in (
        value["results"][0]["correctness_workloads"][0],
        value["results"][0]["timing_workloads"][0],
    ):
        workload["inputs"]["x"] = {"type": "scalar", "value": True}
    with pytest.raises(ValueError, match="does not match ABI type int"):
        FlagGemsExtractorOutput.model_validate(value)


@pytest.mark.parametrize(
    ("call", "inputs", "message"),
    [
        ("torch.gelu(x)", None, "public symbol"),
        ("gelu(torch.abs(x))", None, "input names"),
        ("gelu(x[0])", None, "input names"),
        ("gelu(1)", None, "input names"),
        ("gelu(x, **options)", {"options": {"type": "literal", "value": {}}}, "kwargs"),
        ("gelu(x)", {"unused": {"type": "scalar", "value": 1}}, "unused"),
    ],
)
def test_call_grammar_is_restricted(call, inputs, message):
    value = _output()
    workload = value["results"][0]["correctness_workloads"][0]
    workload["call"] = call
    if inputs:
        workload["inputs"].update(inputs)
    with pytest.raises(ValueError, match=message):
        FlagGemsExtractorOutput.model_validate(value)


def test_call_must_bind_required_parameters_but_may_omit_defaults():
    value = _addmm_output()
    workload = value["results"][0]["timing_workloads"][0]
    del workload["inputs"]["mat2"]
    workload["call"] = "addmm_(self, mat1)"
    with pytest.raises(ValueError, match="does not bind"):
        FlagGemsExtractorOutput.model_validate(value)


def test_effects_targets_and_conditional_null_are_checked():
    value = _output()
    definition = value["results"][0]["definition"]
    definition["parameters"].append(
        _parameter(
            "out",
            "Optional[Tensor]",
            kind="keyword_only",
            required=False,
            default=None,
        )
    )
    definition["reference"] = (
        "def run(x, *, out=None):\n    return x if out is None else out.copy_(x)\n"
    )
    definition["effects"] = {
        "cases": [
            {"when": {"out": {"is": None}}, "mutates": []},
            {
                "when": {"out": {"is_not": None}},
                "mutates": ["out"],
                "returns_alias_of": {"output": "out"},
            },
        ]
    }
    FlagGemsExtractorOutput.model_validate(value)
    definition["effects"]["cases"][1]["mutates"] = ["missing"]
    with pytest.raises(ValueError, match="unknown parameters"):
        FlagGemsExtractorOutput.model_validate(value)


def test_correctness_reference_has_fixed_run_and_matching_abi():
    value = _output()
    definition = value["results"][0]["definition"]
    definition["correctness_reference"] = (
        "import torch\n\ndef run(x):\n    return torch.nn.functional.gelu(x.float()).to(x.dtype)\n"
    )
    FlagGemsExtractorOutput.model_validate(value)
    definition["correctness_reference"] = definition["correctness_reference"].replace(
        "def run(x):", "def run(x, extra):"
    )
    with pytest.raises(ValueError, match="correctness_reference run"):
        FlagGemsExtractorOutput.model_validate(value)


@pytest.mark.parametrize("mapping", ["specs", "values"])
def test_hooks_reject_obsolete_nested_v6_input_context(mapping):
    value = _output()
    definition = value["results"][0]["definition"]
    definition["reference"] = (
        "def run(x):\n"
        "    return x\n\n"
        "def valid(ref_outputs, sol_outputs, inputs, ctx):\n"
        f"    ctx[{mapping!r}]['inputs']\n"
        "    return {'passed': True, 'message': '', 'metrics': {}}\n"
    )
    definition["custom_valid_entrypoint"] = "valid"

    with pytest.raises(ValueError, match=r"flat ctx.*mapping"):
        FlagGemsExtractorOutput.model_validate(value)


def test_timing_baseline_rejects_pytest_only_fixed_float_cast():
    value = _output()
    value["results"][0]["definition"]["reference"] = (
        "import torch\n\ndef run(x):\n    return torch.log2(x.to(torch.float64))\n"
    )
    with pytest.raises(ValueError, match=r"benchmark baseline run\(\).*fixed"):
        FlagGemsExtractorOutput.model_validate(value)


def test_reference_cannot_define_conflicting_public_symbol():
    value = _output()
    value["results"][0]["definition"]["reference"] += (
        "\ndef gelu(x):\n    return x\n"
    )
    with pytest.raises(ValueError, match="only internal run"):
        FlagGemsExtractorOutput.model_validate(value)


def test_expected_exception_is_correctness_only():
    value = _output()
    value["results"][0]["timing_workloads"][0]["expect"] = {
        "raises": "RuntimeError",
        "message_regex": "out of range",
    }
    with pytest.raises(ValueError, match="correctness-only"):
        FlagGemsExtractorOutput.model_validate(value)


def test_cpu_reference_rejects_timing_workloads():
    value = _output()
    value["results"][0]["definition"]["reference_device"] = "cpu"
    with pytest.raises(ValueError, match="correctness-only"):
        FlagGemsExtractorOutput.model_validate(value)


def test_gate_accepts_exported_and_pytest_only_but_rejects_catalog_only(tmp_path):
    repo = _fake_repo(tmp_path)
    runtime = FakeRuntime([])
    agent = FlagGemsExtractorAgent()
    for operator in ("unbind_copy", "einsum"):
        prompt = agent.preprocess(
            FlagGemsExtractorInput(operator=operator, flaggems_repo=str(repo)),
            runtime,
        )
        assert "Protocol**: v6.0 Definition + Workloads" in prompt
        assert "Definition `parameters` is the single public ABI" in prompt
        assert "v4 only" not in prompt
        if operator == "unbind_copy":
            assert "Deterministically resolved public Python signature" in prompt
            assert "def unbind_copy(input, dim=0)" in prompt

    with pytest.raises(OperatorNotImplementedError, match="neither"):
        agent.preprocess(
            FlagGemsExtractorInput(operator="clone", flaggems_repo=str(repo)),
            runtime,
        )


def test_gate_finds_overload_test_by_pytest_marker(tmp_path):
    repo = _fake_repo(tmp_path)
    (repo / "tests" / "test_dunder_ior.py").write_text(
        "import pytest\n\n@pytest.mark.dunder_ior_scalar\ndef test_scalar(): pass\n",
        encoding="utf-8",
    )
    with (repo / "conf" / "operators.yaml").open("a", encoding="utf-8") as file:
        file.write("  - id: dunder_ior_scalar\n    for: [__ior__.Scalar]\n")
    prompt = FlagGemsExtractorAgent().preprocess(
        FlagGemsExtractorInput(
            operator="dunder_ior_scalar",
            flaggems_repo=str(repo),
        ),
        FakeRuntime([]),
    )
    assert "test_dunder_ior.py" in prompt


def test_source_inventory_does_not_prefix_match_pytest_markers(tmp_path):
    repo = _fake_repo(tmp_path)
    (repo / "tests" / "test_einsum_dtype.py").write_text(
        "import pytest\n\n@pytest.mark.einsum_dtype\ndef test_dtype(): pass\n",
        encoding="utf-8",
    )

    inventory = collect_source_inventory(str(repo), "einsum")

    assert inventory.test_files == (repo / "tests" / "test_einsum.py",)


def test_gate_resolves_exported_fused_implementation_and_v6_rules(tmp_path):
    repo = _fake_repo(tmp_path)
    fused = repo / "src" / "flag_gems" / "fused"
    fused.mkdir(parents=True)
    (fused / "__init__.py").write_text(
        "from flag_gems.fused.fused_moe import fused_experts_impl\n"
        "__all__ = ['fused_experts_impl']\n",
        encoding="utf-8",
    )
    (fused / "fused_moe.py").write_text(
        "def fused_experts_impl(x, use_int8=False, scale=None): return x\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_fused_experts_impl.py").write_text(
        "# default and quantized pytest modes\n",
        encoding="utf-8",
    )
    (repo / "benchmark" / "test_fused_experts_impl.py").write_text(
        "# default benchmark mode\n",
        encoding="utf-8",
    )
    (repo / "benchmark" / "test_fused_moe_int8.py").write_text(
        "import pytest\n\n@pytest.mark.fused_experts_impl\ndef test_int8_mode(): pass\n",
        encoding="utf-8",
    )
    with (repo / "conf" / "operators.yaml").open("a", encoding="utf-8") as file:
        file.write(
            "  - id: fused_experts_impl\n"
            "    for: [vllm._custom_ops.fused_experts]\n"
        )
    prompt = FlagGemsExtractorAgent().preprocess(
        FlagGemsExtractorInput(
            operator="fused_experts_impl",
            flaggems_repo=str(repo),
        ),
        FakeRuntime([]),
    )
    assert "Implementation status**: exported implementation" in prompt
    assert "/src/flag_gems/fused/fused_moe.py` ✓ exists" in prompt
    assert "Additional benchmark files marked for this operator" in prompt
    assert "inventory each semantically distinct" in prompt
    assert "parameter's name" in prompt
    assert "omitted defaults" in prompt


def test_gate_finds_internal_aten_suite_without_leading_underscore(tmp_path):
    repo = _fake_repo(tmp_path)
    symbol = "_thnn_fused_lstm_cell_backward_impl"
    (repo / "src" / "flag_gems" / "ops" / "__init__.py").write_text(
        "from flag_gems.ops.unbind_copy import unbind_copy\n"
        f"from flag_gems.ops.{symbol} import {symbol}\n"
        f"__all__ = ['unbind_copy', '{symbol}']\n",
        encoding="utf-8",
    )
    (repo / "src" / "flag_gems" / "ops" / f"{symbol}.py").write_text(
        f"def {symbol}(*args): return args\n",
        encoding="utf-8",
    )
    for suite in ("tests", "benchmark"):
        (repo / suite / "test_thnn_fused_lstm_cell_backward_impl.py").write_text(
            "import pytest\n\n@pytest.mark.thnn_fused_lstm_cell_backward_impl\n"
            "def test_backward(): pass\n",
            encoding="utf-8",
        )
    prompt = FlagGemsExtractorAgent().preprocess(
        FlagGemsExtractorInput(operator=symbol, flaggems_repo=str(repo)),
        FakeRuntime([]),
    )
    expected = "test_thnn_fused_lstm_cell_backward_impl.py"
    assert f"/tests/{expected}` ✓ exists" in prompt
    assert f"/benchmark/{expected}` ✓ exists" in prompt
    assert "same source data flow" in prompt
    assert "Keep has_bias=True" in prompt


def test_special_modified_bessel_k0_guidance_preserves_capability_upcast(tmp_path):
    repo = _fake_repo(tmp_path)
    for suite in ("tests", "benchmark"):
        (repo / suite / "test_special_modified_bessel_k0.py").write_text(
            "# source-backed special_modified_bessel_k0 suite\n",
            encoding="utf-8",
        )
    with (repo / "conf" / "operators.yaml").open("a", encoding="utf-8") as file:
        file.write(
            "  - id: special_modified_bessel_k0\n"
            "    for: [special_modified_bessel_k0]\n"
        )
    prompt = FlagGemsExtractorAgent().preprocess(
        FlagGemsExtractorInput(
            operator="special_modified_bessel_k0",
            flaggems_repo=str(repo),
        ),
        FakeRuntime([]),
    )
    assert "upcast=True is not necessarily float64" in prompt
    assert "unconditional float64" in prompt
    assert "portable float32 subset" in prompt


def test_gate_rejects_exported_operator_without_pytest(tmp_path):
    repo = _fake_repo(tmp_path)
    (repo / "src" / "flag_gems" / "ops" / "__init__.py").write_text(
        "from flag_gems.ops.orphan import orphan\n__all__ = ['orphan']\n",
        encoding="utf-8",
    )
    (repo / "src" / "flag_gems" / "ops" / "orphan.py").write_text(
        "def orphan(x): return x\n",
        encoding="utf-8",
    )
    with (repo / "conf" / "operators.yaml").open("a", encoding="utf-8") as file:
        file.write("  - id: orphan\n    for: [orphan]\n")
    with pytest.raises(OperatorSourceNotFoundError, match="no operator pytest"):
        FlagGemsExtractorAgent().preprocess(
            FlagGemsExtractorInput(operator="orphan", flaggems_repo=str(repo)),
            FakeRuntime([]),
        )


def test_postprocess_rejects_timing_without_benchmark(tmp_path):
    repo = _fake_repo(tmp_path)
    agent = FlagGemsExtractorAgent()
    agent.preprocess(
        FlagGemsExtractorInput(operator="unbind_copy", flaggems_repo=str(repo)),
        FakeRuntime([]),
    )
    value = {
        "results": [
            {
                "group": "shape",
                "definition": _definition(
                    name="unbind_copy",
                    parameters=[
                        _parameter("input"),
                        _parameter("dim", "int", required=False, default=0),
                    ],
                    reference=(
                        "import torch\n\n"
                        "def run(input, dim=0):\n"
                        "    return torch.unbind_copy(input, dim)\n"
                    ),
                ),
                "correctness_workloads": [
                    _workload(
                        "unbind-corr",
                        {
                            "input": {
                                "type": "random",
                                "shape": [3, 4],
                                "dtype": "float32",
                            }
                        },
                        "unbind_copy(input)",
                    )
                ],
                "timing_workloads": [
                    _workload(
                        "unbind-time",
                        {
                            "input": {
                                "type": "random",
                                "shape": [3, 4],
                                "dtype": "float32",
                            }
                        },
                        "unbind_copy(input)",
                    )
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="timing workloads require"):
        agent.postprocess(json.dumps(value), FakeRuntime([]))
    value["results"][0]["timing_workloads"] = []
    agent.postprocess(json.dumps(value), FakeRuntime([]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("group", "addmm_", "unsupported v6 group"),
        ("reference_device", "some_accelerator", "reference_device"),
    ],
)
def test_postprocess_does_not_silently_rewrite_unknown_vocabulary(
    tmp_path,
    field,
    value,
    message,
):
    repo = _fake_repo(tmp_path)
    agent = FlagGemsExtractorAgent()
    agent.preprocess(
        FlagGemsExtractorInput(operator="einsum", flaggems_repo=str(repo)),
        FakeRuntime([]),
    )
    output = _output()
    if field == "group":
        output["results"][0][field] = value
    else:
        output["results"][0]["definition"][field] = value

    with pytest.raises(ValueError, match=message):
        agent.postprocess(json.dumps(output), FakeRuntime([]))


def test_postprocess_rejects_self_consistent_abi_that_differs_from_public_source(
    tmp_path,
):
    repo = _fake_repo(tmp_path)
    agent = FlagGemsExtractorAgent()
    agent.preprocess(
        FlagGemsExtractorInput(operator="unbind_copy", flaggems_repo=str(repo)),
        FakeRuntime([]),
    )
    value = {
        "results": [
            {
                "group": "shape",
                "definition": _definition(
                    name="unbind_copy",
                    parameters=[_parameter("input"), _parameter("dim", "int")],
                    reference=(
                        "import torch\n\n"
                        "def run(input, dim):\n"
                        "    return torch.unbind_copy(input, dim)\n"
                    ),
                ),
                "correctness_workloads": [
                    _workload(
                        "unbind-corr",
                        {
                            "input": {
                                "type": "random",
                                "shape": [3, 4],
                                "dtype": "float32",
                            },
                            "dim": {"type": "scalar", "value": 0},
                        },
                        "unbind_copy(input, dim)",
                    )
                ],
                "timing_workloads": [],
            }
        ]
    }
    with pytest.raises(ValueError, match="FlagGems public source.*required/default"):
        agent.postprocess(json.dumps(value), FakeRuntime([]))


def test_postprocess_rejects_wrong_public_callable_name(tmp_path):
    repo = _fake_repo(tmp_path)
    agent = FlagGemsExtractorAgent()
    agent.preprocess(
        FlagGemsExtractorInput(operator="unbind_copy", flaggems_repo=str(repo)),
        FakeRuntime([]),
    )
    value = _output()
    value["results"][0]["timing_workloads"] = []

    with pytest.raises(ValueError, match="does not match.*public callable"):
        agent.postprocess(json.dumps(value), FakeRuntime([]))


def test_postprocess_requires_target_timing_when_benchmark_exists(tmp_path):
    repo = _fake_repo(tmp_path)
    agent = FlagGemsExtractorAgent()
    agent.preprocess(
        FlagGemsExtractorInput(operator="einsum", flaggems_repo=str(repo)),
        FakeRuntime([]),
    )
    value = _output()
    value["results"][0]["timing_workloads"] = []
    with pytest.raises(ValueError, match="requires at least one timing"):
        agent.postprocess(json.dumps(value), FakeRuntime([]))
    value["results"][0]["definition"]["reference_device"] = "cpu"
    agent.postprocess(json.dumps(value), FakeRuntime([]))


def test_postprocess_rejects_description_that_denies_resolved_source(tmp_path):
    repo = _fake_repo(tmp_path)
    agent = FlagGemsExtractorAgent()
    agent.preprocess(
        FlagGemsExtractorInput(operator="einsum", flaggems_repo=str(repo)),
        FakeRuntime([]),
    )
    value = _output()
    value["results"][0]["definition"]["description"] = (
        "No pytest or benchmark exists; inferred from schema."
    )
    with pytest.raises(ValueError, match="contradicts resolved pytest"):
        agent.postprocess(json.dumps(value), FakeRuntime([]))


def test_agent_uses_native_role_and_v6_contract(tmp_path):
    repo = _fake_repo(tmp_path)
    runtime = FakeRuntime([json.dumps(_output())])
    output = FlagGemsExtractorAgent().run(
        {"operator": "einsum", "flaggems_repo": str(repo)},
        runtime,
    )
    assert output.results[0].definition.api_version == "v6.0"
    prompt = runtime.calls[0]["prompt"]
    assert "V6 contract" in prompt
    assert "General extraction principles (mandatory)" in prompt
    assert "reference-as-solution" in prompt
    assert "generator_params" in prompt
    assert "correctness_reference" in prompt
    assert "Definition `parameters` is the single public ABI" in prompt


def test_cli_installs_native_agent_in_isolated_workspace(tmp_path):
    installed = prepare_native_agent(tmp_path)
    assert installed == tmp_path / ".claude" / "agents" / "kernel-flaggems-extractor.md"
    text = installed.read_text(encoding="utf-8")
    assert "name: kernel-flaggems-extractor" in text
    assert "## V6 contract" in text


def test_workflow_writes_v6_catalog(tmp_path):
    output = FlagGemsExtractorOutput.model_validate(_output())
    root = FlagGemsExtractWorkflow._dump_catalog(
        output.results,
        str(tmp_path / "flaggems-v6"),
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["api_version"] == "v6.0"
    assert manifest["source_format"] == "flaggems-extractor-v6"
    definition = json.loads(
        (root / "definitions" / "pointwise" / "gelu.json").read_text(
            encoding="utf-8"
        )
    )
    assert definition["parameters"][0]["name"] == "x"
    workload = json.loads(
        (root / "workloads" / "pointwise" / "gelu.correctness.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert workload["call"] == "gelu(x)"


def test_workflow_uses_record_id_as_filename_and_keeps_public_name(tmp_path):
    value = _output()
    value["results"][0]["record_id"] = "gelu_tensor_specialization"
    output = FlagGemsExtractorOutput.model_validate(value)
    root = FlagGemsExtractWorkflow._dump_catalog(
        output.results,
        str(tmp_path / "catalog"),
    )

    definition_path = (
        root
        / "definitions"
        / "pointwise"
        / "gelu_tensor_specialization.json"
    )
    assert json.loads(definition_path.read_text(encoding="utf-8"))["name"] == "gelu"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["operators"][0]["id"] == "gelu_tensor_specialization"
    assert manifest["operators"][0]["name"] == "gelu"


def test_explicit_nulls_survive_catalog_dump(tmp_path):
    value = _output()
    definition = value["results"][0]["definition"]
    definition["parameters"].append(
        _parameter(
            "optional",
            "Optional[Tensor]",
            kind="keyword_only",
            required=False,
            default=None,
        )
    )
    definition["reference"] = (
        "def run(x, *, optional=None):\n    return x if optional is None else optional\n"
    )
    definition["effects"] = {
        "cases": [
            {"when": {"optional": {"is": None}}, "mutates": []},
        ]
    }
    output = FlagGemsExtractorOutput.model_validate(value)
    root = FlagGemsExtractWorkflow._dump_catalog(
        output.results,
        str(tmp_path / "catalog"),
    )
    dumped = json.loads(
        (root / "definitions" / "pointwise" / "gelu.json").read_text(
            encoding="utf-8"
        )
    )
    assert dumped["parameters"][-1]["default"] is None
    assert dumped["effects"]["cases"][0]["when"]["optional"]["is"] is None

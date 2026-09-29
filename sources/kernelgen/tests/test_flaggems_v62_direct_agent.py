import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.flaggems_v62_extract import run_direct_agent
from kernelgen.agents.extractor.flaggems.v62_agent import (
    FlagGemsV62ExtractorAgent,
    FlagGemsV62DtypeExpansion,
    FlagGemsV62ExtractorInput,
    FlagGemsV62Workload,
    _accuracy_coverage_plan,
    _accuracy_coverage_report,
    _accuracy_manual_seeds,
    _accuracy_sequence_comparison_owns_return_contract,
    _accuracy_test_identities,
    _duplicate_accuracy_test_identities,
    _dynamic_float_dtype_identities,
    _accuracy_dtype_tolerances,
    _accuracy_test_scope,
    _benchmark_uses_exported_operator_as_baseline,
    _expand_dynamic_float_dtypes,
    _drop_redundant_strict_tolerances,
    _requires_full_range_integer_timing,
    _requires_implementation_float32_timing_promotion,
    _requires_primary_float_dtype_pair,
    _requires_bfloat16_accuracy,
    _resolved_target_float_dtypes,
    _requires_source_torch_fallback,
    _normalize_full_range_integer_timing_contexts,
    _normalize_implicit_softmax_timing_dims,
    _normalize_accuracy_manual_seeds,
    _normalize_accuracy_dtype_tolerances,
    _validate_oracle,
    _validate_recipe_types,
    _validate_broadcast_tensors_shapes,
    _validate_broadcast_tensors_run,
    _validate_conj_workloads,
    _validate_dynamic_float_dtypes,
    _validate_accuracy_workload_scope,
    _validate_full_range_integer_timing_oracle,
    _validate_full_range_integer_timing_workloads,
    ensure_flaggems_v6_adapter_definition,
    persist_flaggems_v62_extraction,
)
from kernelgen.agents.extractor.flaggems.source_inventory import (
    _resolved_helper_context,
    collect_source_inventory,
)


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
        "from flag_gems.ops.adaptive_max_pool3d_backward import "
        "adaptive_max_pool3d_backward\n"
        "__all__ = ['adaptive_max_pool3d_backward']\n",
        encoding="utf-8",
    )
    (fused / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    (ops / "adaptive_max_pool3d_backward.py").write_text(
        "def adaptive_max_pool3d_backward(grad_output, self, indices):\n"
        "    return grad_output\n",
        encoding="utf-8",
    )
    (tests / "test_adaptive_max_pool3d_backward.py").write_text(
        "# accuracy source\n",
        encoding="utf-8",
    )
    (benchmark / "base.py").write_text(
        "if device == 'cuda':\n"
        "    torch.backends.cuda.matmul.allow_tf32 = False\n\n"
        "class GenericBenchmark:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (benchmark / "test_adaptive_max_pool3d_backward.py").write_text(
        "from . import base\n"
        "bench = base.GenericBenchmark()\n",
        encoding="utf-8",
    )
    (conf / "operators.yaml").write_text(
        "ops:\n"
        "  - id: adaptive_max_pool3d_backward\n"
        "    for: [adaptive_max_pool3d_backward]\n",
        encoding="utf-8",
    )
    return repo


@pytest.mark.parametrize('statement', [
    'from kernelgen_server.runtime.source_policy import source_flag',
    'from kernelgen_server.runtime.source_policy import source_vendor',
    'import kernelgen_server.runtime.source_policy as policy',
])
def test_source_policy_oracle_requires_identity_without_conditional_rows(statement):
    from kernelgen.agents.extractor.flaggems.v62_agent import FlagGemsV62ExtractorOutput
    values = dict(oracle=statement + '\ndef run(x): return x\n',
                  correctness_workloads=[{'name': 'c', 'inputs': {}}],
                  timing_workloads=[{'name': 't', 'inputs': {}}])
    with pytest.raises(ValueError, match='source_policy_id'):
        FlagGemsV62ExtractorOutput(**values)
    assert FlagGemsV62ExtractorOutput(**values, source_policy_id='pinned').source_policy_id == 'pinned'


def _case_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "flaggems.benchmark-case-list/v2",
                "benchmarks": [
                    {
                        "schema_version": "flaggems.benchmark-case-list/v2",
                        "op_name": "adaptive_max_pool3d_backward",
                        "phase": "timing",
                        "level": "core",
                        "cases": [
                            {
                                "case_id": "benchmark/test.py::core::float16::0",
                                "ordinal": 0,
                                "dtype": "torch.float16",
                                "shape": {
                                    "grad_output": [1, 1, 4, 4, 4],
                                    "input": [1, 1, 8, 8, 8],
                                    "indices": [1, 1, 4, 4, 4],
                                },
                                "params": {"output_size": [4, 4, 4]},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _oracle(*, dynamic: bool = False) -> str:
    prefix = (
        "import torch\n\n"
        "REFERENCE_DEVICE = 'target'\n\n"
        "def gen_inputs(ctx, device):\n"
        "    return {'grad_output': 1, 'self': 2, 'indices': 3}\n\n"
        "def run(grad_output, self, indices):\n"
        "    return grad_output\n"
    )
    return prefix + ("\nexec('x = 1')\n" if dynamic else "")


def _proposal(*, dynamic: bool = False) -> str:
    return json.dumps(
        {
            "oracle": _oracle(dynamic=dynamic),
            "correctness_workloads": [
                {
                    "name": "accuracy::float16::0",
                    "inputs": {
                        "case": {
                            "phase": "correctness",
                            "dtype": "float16",
                            "shape": {"input": [1, 1, 8, 8, 8]},
                            "params": {"output_size": [4, 4, 4]},
                        }
                    },
                    "seed": 0,
                }
            ],
            "timing_workloads": [
                {
                    "name": "benchmark/test.py::core::float16::0",
                    "inputs": {
                        "case": {
                            "phase": "timing",
                            "dtype": "float16",
                            "shape": {"input": [1, 1, 8, 8, 8]},
                            "params": {"output_size": [4, 4, 4]},
                        }
                    },
                    "seed": 0,
                }
            ],
        }
    )


class _Runtime:
    supports_native_agents = False

    def __init__(self, response: str):
        self.response = response
        self.prompt = ""

    def invoke(self, prompt: str, model: str = "inherit") -> str:
        self.prompt = prompt
        return self.response


def test_direct_agent_runtime_only_allows_read(monkeypatch, tmp_path):
    captured = {}

    class _ClaudeRuntime:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(run_direct_agent, "ClaudeRuntime", _ClaudeRuntime)

    run_direct_agent._runtime(tmp_path, "inherit", 900)

    assert captured["allowed_tools"] == "Read"


def test_direct_agent_inlines_directly_imported_accuracy_tolerance(tmp_path):
    repo = tmp_path / "FlagGems"
    tests = repo / "tests"
    benchmark = repo / "benchmark"
    testing = repo / "src" / "flag_gems" / "testing"
    tests.mkdir(parents=True)
    benchmark.mkdir(parents=True)
    testing.mkdir(parents=True)
    accuracy = tests / "test_conv.py"
    accuracy.write_text(
        "from .accuracy_utils import gems_assert_close\n"
        "from .conftest import QUICK_MODE\n"
        "gems_assert_close(result, reference, dtype)\n",
        encoding="utf-8",
    )
    (tests / "conftest.py").write_text(
        "QUICK_MODE = False\n",
        encoding="utf-8",
    )
    (tests / "accuracy_utils.py").write_text(
        "def gems_assert_close(res, ref, dtype, atol=1e-4):\n"
        "    flag_gems.testing.assert_close(res, ref, dtype, atol=atol)\n",
        encoding="utf-8",
    )
    (testing / "__init__.py").write_text(
        "RESOLUTION = {'float16': 1e-3}\n"
        "def assert_close(res, ref, dtype, atol=1e-4):\n"
        "    return torch.testing.assert_close(\n"
        "        res, ref, rtol=RESOLUTION[dtype], atol=atol\n"
        "    )\n",
        encoding="utf-8",
    )
    timing = benchmark / "test_conv.py"
    timing.write_text("# no benchmark helper\n", encoding="utf-8")

    context = _resolved_helper_context(repo, accuracy, timing)

    assert "def gems_assert_close" in context
    assert "RESOLUTION = {'float16': 1e-3}" in context
    assert "def assert_close" in context
    assert "QUICK_MODE = False" in context


def test_direct_agent_detects_source_vendor_torch_fallback(tmp_path):
    accuracy = tmp_path / "test_op.py"
    accuracy.write_text(
        "def reference(x):\n"
        "    if flag_gems.vendor_name == 'ascend':\n"
        "        return torch.conv2d(x, weight)\n"
        "    return torch.cudnn_convolution(x, weight)\n",
        encoding="utf-8",
    )
    inventory = SimpleNamespace(test_files=(accuracy,), repo=_repo(tmp_path))

    assert _requires_source_torch_fallback(inventory)


def test_direct_agent_does_not_treat_vendor_skip_as_torch_fallback(tmp_path):
    accuracy = tmp_path / "test_linalg_cholesky.py"
    accuracy.write_text(
        "def test_linalg_cholesky(shape, dtype):\n"
        "    if flag_gems.vendor_name == 'ascend' and dtype == torch.float64:\n"
        "        pytest.skip('not supported')\n"
        "    matrix = torch.randn(shape)\n"
        "    value = matrix @ matrix.T + torch.eye(shape[-1])\n"
        "    return torch.linalg.cholesky(value)\n",
        encoding="utf-8",
    )
    inventory = SimpleNamespace(test_files=(accuracy,), repo=_repo(tmp_path))

    assert not _requires_source_torch_fallback(inventory)


def test_direct_agent_drops_only_strict_default_tolerance():
    inputs = {
        "case": {
            "phase": "correctness",
            "dtype": "bfloat16",
            "shape": [8],
        }
    }
    strict = FlagGemsV62Workload(
        name="strict",
        inputs=inputs,
        tolerance={"rtol": 0.016, "atol": 1e-4},
    )
    custom = FlagGemsV62Workload(
        name="custom",
        inputs=inputs,
        tolerance={"rtol": 0.02, "atol": 1e-4},
    )

    normalized = _drop_redundant_strict_tolerances([strict, custom])

    assert normalized[0].tolerance is None
    assert normalized[1].tolerance is not None
    assert normalized[1].tolerance.rtol == 0.02


def test_direct_agent_accepts_source_style_indirect_torch_fallback():
    definition = SimpleNamespace(
        parameters=[
            SimpleNamespace(
                name=name,
                kind="positional_or_keyword",
                required=True,
                default=None,
                type_hint="Tensor",
            )
            for name in ("input", "weight")
        ]
    )
    oracle = (
        "import torch\n\n"
        "REFERENCE_DEVICE = 'target'\n\n"
        "def run(input, weight):\n"
        "    return torch.cudnn_convolution(input, weight)\n\n"
        "def torch_run(input, weight):\n"
        "    convolution = {3: torch.conv1d, 4: torch.conv2d, "
        "5: torch.conv3d}[input.dim()]\n"
        "    return convolution(input, weight)\n"
    )

    _validate_oracle(
        "cudnn_convolution",
        definition,
        oracle,
        has_correctness=True,
        has_timing=True,
        needs_gen_inputs=False,
        mixes_generated_and_direct_inputs=False,
        needs_torch_fallback=True,
    )


def test_direct_agent_rejects_unsourced_torch_fallback():
    definition = SimpleNamespace(
        parameters=[
            SimpleNamespace(
                name="input",
                kind="positional_or_keyword",
                required=True,
                default=None,
                type_hint="Tensor",
            )
        ]
    )
    oracle = (
        "import torch\n"
        "REFERENCE_DEVICE = 'target'\n"
        "def run(input): return torch.softmax(input, 0)\n"
        "def torch_run(input): return torch.softmax(input.float(), 0)\n"
    )

    with pytest.raises(ValueError, match="without a source-backed"):
        _validate_oracle(
            "softmax",
            definition,
            oracle,
            has_correctness=True,
            has_timing=True,
            needs_gen_inputs=False,
            mixes_generated_and_direct_inputs=False,
            needs_torch_fallback=False,
        )


def test_normalizes_implicit_one_dimensional_softmax_dim():
    workload = FlagGemsV62Workload(
        name="benchmark/test_softmax.py::test_softmax::core::float32::0",
        inputs={
            "self": {
                "type": "random",
                "shape": [1048576],
                "dtype": "float32",
                "device": "target",
            },
            "dim": {"type": "scalar", "value": None},
        },
    )
    cases = [
        {
            "shape": {"input": [1048576]},
            "params": {},
            "dtype": "torch.float32",
        }
    ]

    normalized = _normalize_implicit_softmax_timing_dims(
        "softmax",
        [workload],
        cases,
    )

    assert normalized[0].inputs["dim"] == {"type": "scalar", "value": 0}


def test_direct_agent_detects_len_zip_return_contract(tmp_path):
    source = tmp_path / "test_nonzero_numpy.py"
    source.write_text(
        "import pytest\n"
        "@pytest.mark.nonzero_numpy\n"
        "def test_nonzero_numpy():\n"
        "    ref_out = reference()\n"
        "    res_out = candidate()\n"
        "    assert len(res_out) == len(ref_out)\n"
        "    for res, ref in zip(res_out, ref_out):\n"
        "        assert_equal(res, ref)\n",
        encoding="utf-8",
    )
    inventory = SimpleNamespace(test_files=(source,))

    assert _accuracy_sequence_comparison_owns_return_contract(
        inventory, "nonzero_numpy"
    )


def test_direct_agent_requires_explicit_valid_owned_return_contract():
    definition = SimpleNamespace(
        parameters=[
            SimpleNamespace(
                name="inp",
                kind="positional_or_keyword",
                required=True,
                default=None,
                type_hint="Tensor",
            )
        ]
    )
    missing = (
        "import torch\n"
        "REFERENCE_DEVICE = 'target'\n"
        "def run(inp): return torch.ops.aten.nonzero_numpy(inp)\n"
    )
    with pytest.raises(ValueError, match="VALID_OWNS_RETURN_CONTRACT"):
        _validate_oracle(
            "nonzero_numpy",
            definition,
            missing,
            has_correctness=True,
            has_timing=True,
            needs_gen_inputs=False,
            mixes_generated_and_direct_inputs=False,
            needs_torch_fallback=False,
            requires_valid_owned_return_contract=True,
        )

    valid = missing + (
        "VALID_OWNS_RETURN_CONTRACT = True\n"
        "def valid(ref_outputs, sol_outputs, inputs, ctx):\n"
        "    return len(ref_outputs) == len(sol_outputs)\n"
    )
    _validate_oracle(
        "nonzero_numpy",
        definition,
        valid,
        has_correctness=True,
        has_timing=True,
        needs_gen_inputs=False,
        mixes_generated_and_direct_inputs=False,
        needs_torch_fallback=False,
        requires_valid_owned_return_contract=True,
    )


def test_direct_agent_rejects_tensor_ops_on_output_lists():
    definition = SimpleNamespace(
        parameters=[
            SimpleNamespace(
                name="self",
                kind="positional_or_keyword",
                required=True,
                default=None,
                type_hint="Tensor",
            ),
            SimpleNamespace(
                name="high",
                kind="positional_or_keyword",
                required=True,
                default=None,
                type_hint="int",
            ),
        ]
    )
    invalid = (
        "import torch\n"
        "REFERENCE_DEVICE = 'target'\n"
        "def run(self, high): return torch.randint_like(self, high)\n"
        "def valid(ref_outputs, sol_outputs, inputs, ctx):\n"
        "    return (sol_outputs >= 0).all()\n"
    )

    with pytest.raises(ValueError, match="top-level output lists"):
        _validate_oracle(
            "randint_like",
            definition,
            invalid,
            has_correctness=True,
            has_timing=True,
            needs_gen_inputs=False,
            mixes_generated_and_direct_inputs=False,
            needs_torch_fallback=False,
        )


def test_direct_agent_detects_self_baselined_float32_timing_promotion(tmp_path):
    repo = _repo(tmp_path)
    (repo / "src/flag_gems/ops/adaptive_max_pool3d_backward.py").write_text(
        "import torch\n"
        "def adaptive_max_pool3d_backward(grad_output, self, indices):\n"
        "    if grad_output.dtype == torch.float16:\n"
        "        return grad_output.to(torch.float32).to(grad_output.dtype)\n"
        "    return grad_output\n",
        encoding="utf-8",
    )
    (repo / "benchmark/test_adaptive_max_pool3d_backward.py").write_text(
        "import flag_gems\n"
        "from . import base\n"
        "bench = base.GenericBenchmark(\n"
        "    torch_op=flag_gems.adaptive_max_pool3d_backward\n"
        ")\n",
        encoding="utf-8",
    )

    inventory = collect_source_inventory(
        str(repo), "adaptive_max_pool3d_backward"
    )

    assert _requires_implementation_float32_timing_promotion(inventory)
    assert _benchmark_uses_exported_operator_as_baseline(inventory)


def test_direct_agent_rejects_dropped_self_baseline_dtype_promotion():
    definition = SimpleNamespace(
        parameters=[
            SimpleNamespace(
                name=name,
                kind="positional_or_keyword",
                required=True,
                default=None,
                type_hint="Tensor",
            )
            for name in ("input", "grad_output", "weight", "output_mask")
        ]
    )
    oracle = (
        "import torch\n\n"
        "REFERENCE_DEVICE = 'target'\n\n"
        "def run(input, grad_output, weight, output_mask):\n"
        "    return grad_output @ weight, grad_output.t() @ input, "
        "grad_output.sum(0)\n"
    )

    with pytest.raises(Exception, match="float16-to-float32"):
        _validate_oracle(
            "linear_backward",
            definition,
            oracle,
            has_correctness=True,
            has_timing=True,
            needs_gen_inputs=False,
            mixes_generated_and_direct_inputs=False,
            needs_torch_fallback=False,
            requires_float32_timing_promotion=True,
        )


def test_direct_agent_rejects_unknown_recipe_types():
    workloads = [
        FlagGemsV62Workload(
            name="timing",
            inputs={
                "self": {
                    "type": "randn",
                    "shape": [8],
                    "dtype": "float16",
                    "device": "target",
                }
            },
        )
    ]

    with pytest.raises(ValueError, match="type='random'.*type='randn'"):
        _validate_recipe_types("log_normal_", workloads)


def test_direct_agent_allows_exact_flaggems_timing_reference_only():
    definition = SimpleNamespace(
        parameters=[
            SimpleNamespace(
                name=name,
                kind="positional_or_keyword",
                required=True,
                default=None,
                type_hint="Tensor",
            )
            for name in ("input", "grad_output", "weight", "output_mask")
        ]
    )
    oracle = (
        "import torch\n"
        "import flag_gems\n\n"
        "REFERENCE_DEVICE = 'target'\n\n"
        "def correctness_run(input, grad_output, weight, output_mask):\n"
        "    return grad_output @ weight, grad_output.t() @ input, grad_output.sum(0)\n\n"
        "def timing_run(input, grad_output, weight, output_mask):\n"
        "    return flag_gems.linear_backward(input, grad_output, weight, output_mask)\n"
    )

    _validate_oracle(
        "linear_backward",
        definition,
        oracle,
        has_correctness=True,
        has_timing=True,
        needs_gen_inputs=False,
        mixes_generated_and_direct_inputs=False,
        needs_torch_fallback=False,
        requires_float32_timing_promotion=True,
        allows_flaggems_timing_reference=True,
    )

    invalid = oracle.replace(
        "return grad_output @ weight, grad_output.t() @ input, grad_output.sum(0)",
        "return flag_gems.linear_backward(input, grad_output, weight, output_mask)",
        1,
    )
    with pytest.raises(ValueError, match="correctness reference"):
        _validate_oracle(
            "linear_backward",
            definition,
            invalid,
            has_correctness=True,
            has_timing=True,
            needs_gen_inputs=False,
            mixes_generated_and_direct_inputs=False,
            needs_torch_fallback=False,
            allows_flaggems_timing_reference=True,
        )


@pytest.mark.parametrize('conditional', [False, True])
def test_direct_agent_uses_flaggems_source_and_persists_v62(tmp_path, conditional):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    if conditional:
        proposal['source_policy_id'] = 'flaggems/d64794e63b502cb836bc015a92a62c42de4be05a'
        for workload in proposal['correctness_workloads']:
            workload['source_condition'] = {'flags': {'support_fp64': True}}
    runtime = _Runtime(json.dumps(proposal))
    output = FlagGemsV62ExtractorAgent().run(inp.model_dump(), runtime)
    adapter_root = tmp_path / "flaggems-adapter-definitions"
    adapter_root.mkdir()
    (adapter_root / "manifest.json").write_text(
        json.dumps({"api_version": "v6.0", "evaluator": "flaggems"}),
        encoding="utf-8",
    )
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir()
    (catalog_root / "manifest.json").write_text(
        json.dumps(
            {
                "api_version": "v6.2",
                "evaluator": "native",
                "layout": "per-operator",
                "framework": "flaggems",
                "framework_repository": "https://example.com/FlagGems.git",
                "framework_branch": "feat/kernelgen",
                "framework_revision": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    result = persist_flaggems_v62_extraction(
        inp,
        output,
        catalog_root,
        adapter_catalog_root=adapter_root,
    )
    if conditional:
        assert json.loads(result.definition_path.read_text())['source_policy_id'] == proposal['source_policy_id']
        rows = [json.loads(line) for line in result.correctness_path.read_text().splitlines()]
        assert all(row['source_condition'] == {'flags': {'support_fp64': True}} for row in rows)

    assert str(repo / "tests" / "test_adaptive_max_pool3d_backward.py") in runtime.prompt
    assert "do not search or read any KernelGen/KernelGen Server catalog" in runtime.prompt
    assert "Do not probe the Agent host's Torch installation" in runtime.prompt
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in runtime.prompt
    assert "torch.Generator(device=device)" in runtime.prompt
    assert "large timing tensors" in runtime.prompt
    assert "invokes a defined `gen_inputs` hook for every workload" in runtime.prompt
    assert "pytest Torch reference is normally authoritative" in runtime.prompt
    assert "KGS checks PyTree structure" in runtime.prompt
    assert "v6.2 Workload schema has no expected-exception contract" in runtime.prompt
    assert "mixed parametrization omit only the combinations" in runtime.prompt
    definition = json.loads(result.definition_path.read_text())
    assert definition["api_version"] == "v6.2"
    assert "reference" not in definition
    adapter_definition = json.loads(result.adapter_definition_path.read_text())
    assert adapter_definition["api_version"] == "v6.0"
    assert adapter_definition["parameters"] == definition["parameters"]
    assert "reference" not in adapter_definition
    oracle = result.oracle_path.read_text()
    assert "exec(" not in oracle
    assert "def run" in oracle
    assert "def correctness_run" not in oracle
    assert "def timing_run" not in oracle
    timing = [json.loads(line) for line in result.timing_path.read_text().splitlines()]
    assert [workload["name"] for workload in timing] == [
        "benchmark/test.py::core::float16::0"
    ]
    assert timing[0]["inputs"]["case"]["dtype"] == "float16"
    assert timing[0]["inputs"]["case"]["phase"] == "timing"


def test_source_inventory_excludes_gated_secondary_variant_suite(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tests" / "test_adaptive_max_pool3d_backward.py").write_text(
        "import pytest\n"
        "@pytest.mark.adaptive_max_pool3d_backward\n"
        "def test_primary(): pass\n",
        encoding="utf-8",
    )
    gated = repo / "tests" / "test_arch_adaptive_max_pool3d_backward.py"
    gated.write_text(
        "import pytest\n"
        "INSTALLED = False\n"
        "pytestmark = pytest.mark.skipif(not INSTALLED, reason='variant absent')\n"
        "@pytest.mark.adaptive_max_pool3d_backward\n"
        "def test_arch_variant(): pass\n",
        encoding="utf-8",
    )

    inventory = collect_source_inventory(
        str(repo),
        "adaptive_max_pool3d_backward",
    )

    assert inventory.test_files == (
        repo / "tests" / "test_adaptive_max_pool3d_backward.py",
    )


def test_adapter_definition_rejects_existing_abi_drift(tmp_path):
    repo = _repo(tmp_path)
    adapter_root = tmp_path / "flaggems-adapter-definitions"
    definitions = adapter_root / "definitions"
    definitions.mkdir(parents=True)
    (adapter_root / "manifest.json").write_text(
        json.dumps({"api_version": "v6.0", "evaluator": "flaggems"}),
        encoding="utf-8",
    )
    (definitions / "adaptive_max_pool3d_backward.json").write_text(
        json.dumps({"api_version": "v6.0", "name": "wrong"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differs from the current FlagGems export"):
        ensure_flaggems_v6_adapter_definition(
            repo,
            "adaptive_max_pool3d_backward",
            adapter_root,
        )


def test_direct_agent_rejects_dynamic_oracle_source(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    with pytest.raises(Exception, match="static source"):
        FlagGemsV62ExtractorAgent().run(inp.model_dump(), _Runtime(_proposal(dynamic=True)))


def test_direct_agent_rejects_forward_autograd_in_backward_timing(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["oracle"] = (
        "import torch\n\n"
        "REFERENCE_DEVICE = 'target'\n\n"
        "def gen_inputs(ctx, device):\n"
        "    return {'grad_output': 1, 'self': 2, 'indices': 3}\n\n"
        "def correctness_run(grad_output, self, indices):\n"
        "    return grad_output\n\n"
        "def timing_run(grad_output, self, indices):\n"
        "    output = self + 1\n"
        "    return torch.autograd.grad(output, self, grad_output)\n"
    )

    with pytest.raises(Exception, match="outside the timed region"):
        FlagGemsV62ExtractorAgent().run(
            inp.model_dump(), _Runtime(json.dumps(proposal))
        )


def test_direct_agent_rejects_benchmark_setup_inside_timed_run(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["oracle"] = proposal["oracle"].replace(
        "    return grad_output\n",
        "    torch.backends.cudnn.allow_tf32 = False\n"
        "    return grad_output\n",
    )

    with pytest.raises(Exception, match="benchmark setup must stay outside"):
        FlagGemsV62ExtractorAgent().run(
            inp.model_dump(), _Runtime(json.dumps(proposal))
        )


def test_direct_agent_allows_source_guarded_backend_setup(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["oracle"] = proposal["oracle"].replace(
        "def gen_inputs(ctx, device):\n",
        "def gen_inputs(ctx, device):\n"
        "    try:\n"
        "        torch.backends.cuda.matmul.allow_tf32 = False\n"
        "    except Exception:\n"
        "        pass\n",
    )

    output = FlagGemsV62ExtractorAgent().run(
        inp.model_dump(), _Runtime(json.dumps(proposal))
    )

    assert "except Exception" in output.oracle


def test_direct_agent_rejects_raw_indexed_device_backend_guard(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["oracle"] = proposal["oracle"].replace(
        "def gen_inputs(ctx, device):\n",
        "def gen_inputs(ctx, device):\n"
        "    device_str = str(device)\n"
        "    if device_str == 'npu':\n"
        "        torch.backends.cuda.matmul.allow_tf32 = False\n",
    )

    with pytest.raises(Exception, match="normalized device type"):
        FlagGemsV62ExtractorAgent().run(
            inp.model_dump(), _Runtime(json.dumps(proposal))
        )


def test_direct_agent_rejects_hard_coded_generic_backend_fallback(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["oracle"] = proposal["oracle"].replace(
        "def gen_inputs(ctx, device):\n",
        "def gen_inputs(ctx, device):\n"
        "    device_type = torch.device(device).type\n"
        "    if device_type == 'musa':\n"
        "        torch.backends.mudnn.allow_tf32 = False\n"
        "    elif device_type == 'npu':\n"
        "        torch.backends.cudnn.allow_tf32 = False\n"
        "    else:\n"
        "        torch.backends.cuda.matmul.allow_tf32 = False\n",
    )

    with pytest.raises(Exception, match="generic device fallback"):
        FlagGemsV62ExtractorAgent().run(
            inp.model_dump(), _Runtime(json.dumps(proposal))
        )


def test_direct_agent_rejects_timing_correctness_tolerance(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["timing_workloads"][0]["tolerance"] = {
        "rtol": 0.1,
        "atol": 0.1,
    }

    with pytest.raises(Exception, match="performance-only"):
        FlagGemsV62ExtractorAgent().run(
            inp.model_dump(), _Runtime(json.dumps(proposal))
        )


def test_direct_agent_rejects_timing_recipe_shape_drift(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    direct_inputs = {
        "grad_output": {"type": "random", "shape": [99], "dtype": "float16"},
        "self": {"type": "scalar", "value": 2},
        "indices": {"type": "scalar", "value": 3},
    }
    proposal["correctness_workloads"][0]["inputs"] = direct_inputs
    proposal["timing_workloads"][0]["inputs"] = direct_inputs
    proposal["oracle"] = (
        "REFERENCE_DEVICE = 'target'\n\n"
        "def run(grad_output, self, indices): return grad_output\n"
    )

    with pytest.raises(Exception, match="listed shape.*extracted recipe"):
        FlagGemsV62ExtractorAgent().run(
            inp.model_dump(), _Runtime(json.dumps(proposal))
        )


def test_direct_agent_does_not_require_gen_inputs_for_direct_recipes(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    direct_inputs = {
        "grad_output": {"type": "scalar", "value": 1},
        "self": {"type": "scalar", "value": 2},
        "indices": {"type": "scalar", "value": 3},
    }
    proposal = json.dumps(
        {
            "oracle": (
                "REFERENCE_DEVICE = 'target'\n\n"
                "def run(grad_output, self, indices): return grad_output\n"
            ),
            "correctness_workloads": [
                {"name": "accuracy", "inputs": direct_inputs, "seed": 0}
            ],
            "timing_workloads": [
                {
                    "name": "benchmark/test.py::core::float16::0",
                    "inputs": direct_inputs,
                    "seed": 0,
                }
            ],
        }
    )

    output = FlagGemsV62ExtractorAgent().run(inp.model_dump(), _Runtime(proposal))

    assert "gen_inputs" not in output.oracle


def test_direct_agent_requires_target_device_for_flaggems_random_recipe(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    inputs = {
        "grad_output": {
            "type": "random",
            "shape": [1, 1, 4, 4, 4],
            "dtype": "float16",
        },
        "self": {"type": "scalar", "value": 2},
        "indices": {"type": "scalar", "value": 3},
    }
    proposal = json.dumps(
        {
            "oracle": (
                "REFERENCE_DEVICE = 'target'\n\n"
                "def run(grad_output, self, indices): return grad_output\n"
            ),
            "correctness_workloads": [
                {"name": "accuracy", "inputs": inputs, "seed": 0}
            ],
            "timing_workloads": [
                {
                    "name": "benchmark/test.py::core::float16::0",
                    "inputs": inputs,
                    "seed": 0,
                }
            ],
        }
    )

    with pytest.raises(Exception, match="device='target'"):
        FlagGemsV62ExtractorAgent().run(inp.model_dump(), _Runtime(proposal))


def test_full_range_integer_pointwise_timing_requires_generated_context(tmp_path):
    benchmark = tmp_path / "test_gcd_.py"
    benchmark.write_text(
        "bench = base.BinaryPointwiseBenchmark(op_name='gcd_')\n",
        encoding="utf-8",
    )
    inventory = SimpleNamespace(
        benchmark_files=(benchmark,),
        helper_context=(
            "def generate_tensor_input(shape, dtype, device):\n"
            "    return torch.randint(torch.iinfo(dtype).min, "
            "torch.iinfo(dtype).max, shape, dtype=dtype, "
            "device='cpu').to(device)\n"
        ),
    )
    cases = [
        {
            "case_id": "benchmark/test_gcd_.py::core::int16::0",
            "dtype": "torch.int16",
            "shape": {"inputs": [[64], [64]]},
            "params": {},
        }
    ]
    assert _requires_full_range_integer_timing(inventory, cases)

    direct = FlagGemsV62Workload(
        name=cases[0]["case_id"],
        inputs={
            "A": {
                "type": "random",
                "shape": [64],
                "dtype": "int16",
                "device": "target",
            }
        },
    )
    with pytest.raises(ValueError, match="cannot represent"):
        _validate_full_range_integer_timing_workloads(
            "gcd_", [direct], cases, required=True
        )

    generated = FlagGemsV62Workload(
        name=cases[0]["case_id"],
        inputs={
            "case": {
                "phase": "timing",
                "dtype": "int16",
                "shape": [[64], [64]],
                "params": {},
            }
        },
    )
    _validate_full_range_integer_timing_workloads(
        "gcd_", [generated], cases, required=True
    )

    incomplete = generated.model_copy(
        update={
            "inputs": {
                "case": {
                    "phase": "timing",
                    "dtype": "torch.int16",
                    "shape": [64],
                    "shapes": [[1], [2]],
                    "params": {"wrong": True},
                }
            }
        }
    )
    normalized = _normalize_full_range_integer_timing_contexts(
        [incomplete], cases, required=True
    )
    assert normalized[0].inputs["case"] == {
        "phase": "timing",
        "dtype": "int16",
        "shape": [[64], [64]],
        "params": {},
    }


def test_accuracy_scope_excludes_sibling_operator_markers(tmp_path):
    accuracy = tmp_path / "test_gcd.py"
    accuracy.write_text(
        "import pytest\n"
        "@pytest.mark.gcd\n"
        "def test_gcd(): pass\n"
        "@pytest.mark.gcd_out\n"
        "def test_gcd_out(): pass\n",
        encoding="utf-8",
    )
    allowed, all_tests = _accuracy_test_scope(
        SimpleNamespace(test_files=(accuracy,)), "gcd"
    )
    assert allowed == {"test_gcd"}
    workloads = [
        FlagGemsV62Workload(
            name="tests/test_gcd.py::test_gcd_out[int16]",
            inputs={"case": {"test": "test_gcd_out"}},
        )
    ]
    with pytest.raises(ValueError, match="sibling tests"):
        _validate_accuracy_workload_scope(
            "gcd", workloads, allowed=allowed, all_tests=all_tests
        )


def test_accuracy_scope_parses_prefix_conflicting_nodeids_exactly():
    allowed = {"test_greater_equal_"}
    all_tests = {"test_greater_equal", "test_greater_equal_"}
    valid = [
        FlagGemsV62Workload(
            name=(
                "tests/test_greater_equal.py::"
                "test_greater_equal_[float16-scalar]"
            ),
            inputs={},
        )
    ]

    _validate_accuracy_workload_scope(
        "greater_equal_", valid, allowed=allowed, all_tests=all_tests
    )

    invalid = [
        FlagGemsV62Workload(
            name=(
                "tests/test_greater_equal.py::"
                "test_greater_equal[float16-scalar]"
            ),
            inputs={},
        )
    ]
    with pytest.raises(ValueError, match="test_greater_equal"):
        _validate_accuracy_workload_scope(
            "greater_equal_", invalid, allowed=allowed, all_tests=all_tests
        )


def test_accuracy_scope_keeps_same_named_marked_tests_distinct_by_path(tmp_path):
    repo = tmp_path / "FlagGems"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    first = tests / "test_renorm_.py"
    second = tests / "test_renorm.py"
    source = (
        "import pytest\n"
        "@pytest.mark.renorm_\n"
        "def test_renorm_(): pass\n"
    )
    first.write_text(source, encoding="utf-8")
    second.write_text(source, encoding="utf-8")
    inventory = SimpleNamespace(repo=repo, test_files=(first, second))
    identities = _accuracy_test_identities(inventory, "renorm_")
    required = _duplicate_accuracy_test_identities(identities)

    assert identities == {
        "tests/test_renorm_.py::test_renorm_",
        "tests/test_renorm.py::test_renorm_",
    }
    incomplete = [
        FlagGemsV62Workload(
            name="tests/test_renorm_.py::test_renorm_[float16]",
            inputs={},
        )
    ]
    with pytest.raises(ValueError, match="tests/test_renorm.py::test_renorm_"):
        _validate_accuracy_workload_scope(
            "renorm_",
            incomplete,
            allowed={"test_renorm_"},
            all_tests={"test_renorm_"},
            required_identities=required,
        )

    complete = incomplete + [
        FlagGemsV62Workload(
            name="tests/test_renorm.py::test_renorm_[float16]",
            inputs={},
        )
    ]
    _validate_accuracy_workload_scope(
        "renorm_",
        complete,
        allowed={"test_renorm_"},
        all_tests={"test_renorm_"},
        required_identities=required,
    )


def test_accuracy_coverage_classifies_autograd_only_and_partial_sources(tmp_path):
    repo = tmp_path / "FlagGems"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    accuracy = tests / "test_demo.py"
    accuracy.write_text(
        "import pytest\n"
        "import torch\n"
        "@pytest.mark.demo\n"
        "def test_grad_only():\n"
        "    out, hidden = demo(x)\n"
        "    (out.sum() + hidden.sum()).backward()\n"
        "    assert_close(x.grad, ref_x.grad)\n"
        "@pytest.mark.demo\n"
        "def test_forward_and_grad():\n"
        "    out = demo(x)\n"
        "    assert_close(out, ref_out)\n"
        "    (x_grad,) = torch.autograd.grad(out, (x,))\n"
        "    assert_close(x_grad, ref_x_grad)\n",
        encoding="utf-8",
    )
    inventory = SimpleNamespace(repo=repo, test_files=(accuracy,))

    plan = _accuracy_coverage_plan(inventory, "demo")

    assert plan.required_identities == {
        "tests/test_demo.py::test_forward_and_grad"
    }
    assert plan.excluded_identities == {
        "tests/test_demo.py::test_grad_only": (
            "unrepresentable: autograd contract"
        )
    }
    assert plan.partial_identities == {
        "tests/test_demo.py::test_forward_and_grad": ("autograd contract",)
    }


def test_accuracy_coverage_keeps_public_backward_sources(tmp_path):
    repo = tmp_path / "FlagGems"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    accuracy = tests / "test_demo_backward.py"
    accuracy.write_text(
        "import pytest\n"
        "import torch\n"
        "@pytest.mark.demo_backward\n"
        "def test_demo_backward():\n"
        "    result = demo_backward(grad, x)\n"
        "    (ref_grad,) = torch.autograd.grad(ref_out, (ref_x,), grad)\n"
        "    assert_close(result, ref_grad)\n",
        encoding="utf-8",
    )

    plan = _accuracy_coverage_plan(
        SimpleNamespace(repo=repo, test_files=(accuracy,)), "demo_backward"
    )

    assert plan.required_identities == {
        "tests/test_demo_backward.py::test_demo_backward"
    }
    assert plan.excluded_identities == {}
    assert plan.partial_identities == {}


def test_accuracy_scope_requires_every_representable_marked_source(tmp_path):
    repo = tmp_path / "FlagGems"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    accuracy = tests / "test_demo.py"
    accuracy.write_text(
        "import pytest\n"
        "@pytest.mark.demo\n"
        "def test_first(): assert output == reference\n"
        "@pytest.mark.demo\n"
        "def test_second(): assert output == reference\n",
        encoding="utf-8",
    )
    inventory = SimpleNamespace(repo=repo, test_files=(accuracy,))
    plan = _accuracy_coverage_plan(inventory, "demo")
    incomplete = [
        FlagGemsV62Workload(
            name="tests/test_demo.py::test_first[case]",
            inputs={},
        )
    ]

    with pytest.raises(ValueError, match="test_second"):
        _validate_accuracy_workload_scope(
            "demo",
            incomplete,
            allowed=set(plan.marked_functions),
            all_tests=set(plan.all_test_functions),
            required_identities=set(plan.required_identities),
            excluded_identities=plan.excluded_identities,
        )


def test_accuracy_scope_accepts_unambiguous_legacy_operator_names(tmp_path):
    repo = tmp_path / "FlagGems"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    accuracy = tests / "test_demo.py"
    accuracy.write_text(
        "import pytest\n"
        "@pytest.mark.demo\n"
        "def test_demo(): assert output == reference\n",
        encoding="utf-8",
    )
    plan = _accuracy_coverage_plan(
        SimpleNamespace(repo=repo, test_files=(accuracy,)), "demo"
    )

    _validate_accuracy_workload_scope(
        "demo",
        [FlagGemsV62Workload(name="demo_correctness_case_0", inputs={})],
        allowed=set(plan.marked_functions),
        all_tests=set(plan.all_test_functions),
        required_identities=set(plan.required_identities),
        excluded_identities=plan.excluded_identities,
    )


def test_accuracy_coverage_report_records_excluded_and_partial_sources(tmp_path):
    repo = tmp_path / "FlagGems"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    accuracy = tests / "test_demo.py"
    accuracy.write_text(
        "import pytest\n"
        "@pytest.mark.demo\n"
        "def test_forward(): assert output == reference\n"
        "@pytest.mark.demo\n"
        "def test_grad_only():\n"
        "    output.backward()\n"
        "    assert input.grad == reference.grad\n",
        encoding="utf-8",
    )
    plan = _accuracy_coverage_plan(
        SimpleNamespace(repo=repo, test_files=(accuracy,)), "demo"
    )

    report = _accuracy_coverage_report(
        "demo",
        plan,
        [
            FlagGemsV62Workload(
                name="tests/test_demo.py::test_forward[case]",
                inputs={},
            )
        ],
    )

    assert report["represented_sources"] == [
        "tests/test_demo.py::test_forward"
    ]
    assert report["excluded_sources"] == [
        {
            "identity": "tests/test_demo.py::test_grad_only",
            "reason": "unrepresentable: autograd contract",
        }
    ]


def test_full_range_integer_timing_oracle_requires_cpu_iinfo_randint():
    invalid = (
        "import torch\n"
        "def gen_inputs(ctx, device):\n"
        "    return {'A': torch.randint(-1024, 1024, (8,), device=device)}\n"
    )
    with pytest.raises(ValueError, match="iinfo"):
        _validate_full_range_integer_timing_oracle(
            "gcd_", invalid, required=True
        )

    valid = (
        "import torch\n"
        "def gen_inputs(ctx, device):\n"
        "    dtype = torch.int16\n"
        "    value = torch.randint(\n"
        "        torch.iinfo(dtype).min, torch.iinfo(dtype).max, (8,),\n"
        "        dtype=dtype, device='cpu'\n"
        "    )\n"
        "    return {'A': value.to(device)}\n"
    )
    _validate_full_range_integer_timing_oracle("gcd_", valid, required=True)

    alias = (
        "import torch\n"
        "def gen_inputs(ctx, device):\n"
        "    dtype = torch.int16\n"
        "    info = torch.iinfo(dtype)\n"
        "    value = torch.randint(\n"
        "        info.min, info.max, (8,), dtype=dtype, device='cpu'\n"
        "    )\n"
        "    return {'A': value.to(device)}\n"
    )
    _validate_full_range_integer_timing_oracle("gcd_", alias, required=True)

    bound_alias = (
        "import torch\n"
        "def gen_inputs(ctx, device):\n"
        "    dtype = torch.int16\n"
        "    info = torch.iinfo(dtype)\n"
        "    low, high = 1, 100\n"
        "    if ctx['inputs']['case']['phase'] == 'timing':\n"
        "        low, high = info.min, info.max\n"
        "    value = torch.randint(\n"
        "        low, high, (8,), dtype=dtype, device='cpu'\n"
        "    )\n"
        "    return {'A': value.to(device)}\n"
    )
    _validate_full_range_integer_timing_oracle(
        "gcd_", bound_alias, required=True
    )

    duplicate_shape_key = alias.replace(
        "dtype = torch.int16", "dtype = torch.int16\n    shapes = ctx['inputs']['case']['shapes']"
    )
    with pytest.raises(ValueError, match="parallel 'shapes'"):
        _validate_full_range_integer_timing_oracle(
            "gcd_", duplicate_shape_key, required=True
        )

    wrong_timing_branch = (
        "import torch\n"
        "def gen_inputs(ctx, device):\n"
        "    dtype = torch.int16\n"
        "    if ctx['inputs']['case']['phase'] == 'timing':\n"
        "        value = torch.randint(\n"
        "            torch.iinfo(dtype).min, torch.iinfo(dtype).max, (8,),\n"
        "            dtype=dtype, device=device\n"
        "        )\n"
        "    else:\n"
        "        value = torch.randint(\n"
        "            torch.iinfo(dtype).min, torch.iinfo(dtype).max, (8,),\n"
        "            dtype=dtype, device='cpu'\n"
        "        ).to(device)\n"
        "    return {'A': value}\n"
    )
    with pytest.raises(ValueError, match="device='cpu'"):
        _validate_full_range_integer_timing_oracle(
            "gcd_", wrong_timing_branch, required=True
        )


def test_full_range_integer_timing_oracle_uses_each_ordered_input_shape():
    whole_shape_list = (
        "import torch\n"
        "def gen_inputs(ctx, device):\n"
        "    case = ctx['inputs']['case']\n"
        "    dtype = getattr(torch, case['dtype'])\n"
        "    shape = tuple(case['shape'])\n"
        "    info = torch.iinfo(dtype)\n"
        "    lhs = torch.randint(info.min, info.max, shape, dtype=dtype, device='cpu')\n"
        "    rhs = torch.randint(info.min, info.max, shape, dtype=dtype, device='cpu')\n"
        "    return {'self': lhs.to(device), 'other': rhs.to(device)}\n"
    )
    with pytest.raises(ValueError, match="ordered input shapes"):
        _validate_full_range_integer_timing_oracle(
            "lcm", whole_shape_list, required=True, input_shape_count=2
        )

    indexed_shapes = (
        "import torch\n"
        "def gen_inputs(ctx, device):\n"
        "    case = ctx['inputs']['case']\n"
        "    shapes = case['shape']\n"
        "    lhs_shape = tuple(shapes[0])\n"
        "    rhs_shape = tuple(shapes[1])\n"
        "    dtype = getattr(torch, case['dtype'])\n"
        "    info = torch.iinfo(dtype)\n"
        "    lhs = torch.randint(info.min, info.max, lhs_shape, dtype=dtype, device='cpu')\n"
        "    rhs = torch.randint(info.min, info.max, rhs_shape, dtype=dtype, device='cpu')\n"
        "    return {'self': lhs.to(device), 'other': rhs.to(device)}\n"
    )
    _validate_full_range_integer_timing_oracle(
        "lcm", indexed_shapes, required=True, input_shape_count=2
    )


def test_direct_agent_rejects_nested_recipes_without_gen_inputs(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    nested_inputs = {
        "grad_output": [
            {"type": "random", "shape": [1], "dtype": "float16"},
            {"type": "random", "shape": [1], "dtype": "float16"},
        ],
        "self": {"type": "scalar", "value": 2},
        "indices": {"type": "scalar", "value": 3},
    }
    proposal = json.dumps(
        {
            "oracle": (
                "REFERENCE_DEVICE = 'target'\n\n"
                "def run(grad_output, self, indices): return grad_output\n"
            ),
            "correctness_workloads": [
                {"name": "accuracy", "inputs": nested_inputs, "seed": 0}
            ],
            "timing_workloads": [
                {
                    "name": "benchmark/test.py::core::float16::0",
                    "inputs": nested_inputs,
                    "seed": 0,
                }
            ],
        }
    )

    with pytest.raises(Exception, match="require gen_inputs"):
        FlagGemsV62ExtractorAgent().run(inp.model_dump(), _Runtime(proposal))


def test_direct_agent_requires_none_path_for_mixed_generated_inputs(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    direct_inputs = {
        "grad_output": {"type": "scalar", "value": 1},
        "self": {"type": "scalar", "value": 2},
        "indices": {"type": "scalar", "value": 3},
    }
    proposal = {
        "oracle": (
            "REFERENCE_DEVICE = 'target'\n\n"
            "def gen_inputs(ctx, device):\n"
            "    return {'grad_output': 1, 'self': 2, 'indices': 3}\n\n"
            "def run(grad_output, self, indices): return grad_output\n"
        ),
        "correctness_workloads": [
            {"name": "generated", "inputs": {"case": {}}, "seed": 0}
        ],
        "timing_workloads": [
            {
                "name": "benchmark/test.py::core::float16::0",
                "inputs": direct_inputs,
                "seed": 0,
            }
        ],
    }

    with pytest.raises(Exception, match="explicitly return None"):
        FlagGemsV62ExtractorAgent().run(
            inp.model_dump(), _Runtime(json.dumps(proposal))
        )

    proposal["oracle"] = (
        "REFERENCE_DEVICE = 'target'\n\n"
        "def gen_inputs(ctx, device):\n"
        "    if 'case' not in ctx['inputs']:\n"
        "        return None\n"
        "    return {'grad_output': 1, 'self': 2, 'indices': 3}\n\n"
        "def run(grad_output, self, indices): return grad_output\n"
    )
    output = FlagGemsV62ExtractorAgent().run(
        inp.model_dump(), _Runtime(json.dumps(proposal))
    )

    assert "return None" in output.oracle


def test_direct_agent_rejects_inconsistent_broadcast_case_shapes():
    workload = FlagGemsV62Workload(
        name="bad-broadcast",
        inputs={
            "case": {
                "dtype": "float16",
                "shape": {
                    "inputs": [[1, 3, 4], [3, 1, 4]],
                    "output": [3, 1, 4],
                },
            }
        },
    )

    with pytest.raises(ValueError, match="broadcast to"):
        _validate_broadcast_tensors_shapes("broadcast_tensors", [workload])


def test_direct_agent_reports_all_broadcast_workloads_missing_shape_metadata():
    workloads = [
        FlagGemsV62Workload(
            name=f"missing-{phase}",
            inputs={"case": {"phase": phase, "shapes": [[1], [1]]}},
        )
        for phase in ("correctness", "timing")
    ]

    with pytest.raises(
        ValueError,
        match=r"2 workload\(s\).*correctness.*timing",
    ):
        _validate_broadcast_tensors_shapes("broadcast_tensors", workloads)


def test_direct_agent_rejects_broadcast_implementation_input_normalization():
    module = ast.parse(
        "def run(*tensors):\n"
        "    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):\n"
        "        tensors = tuple(tensors[0])\n"
        "    return torch.broadcast_tensors(*tensors)\n"
    )

    with pytest.raises(ValueError, match="must directly return"):
        _validate_broadcast_tensors_run("broadcast_tensors", module.body[0])


def test_direct_agent_accepts_raw_broadcast_torch_reference():
    module = ast.parse(
        "def run(*tensors):\n"
        "    return torch.broadcast_tensors(*tensors)\n"
    )

    _validate_broadcast_tensors_run("broadcast_tensors", module.body[0])


def test_direct_agent_rejects_conj_correctness_distribution_substitution():
    correctness = [
        FlagGemsV62Workload(
            name="direct-complex-random",
            inputs={
                "input": {
                    "type": "random",
                    "shape": [256],
                    "dtype": "complex32",
                }
            },
        )
    ]
    timing = [
        FlagGemsV62Workload(
            name="timing",
            inputs={
                "input": {
                    "type": "random",
                    "shape": [64],
                    "dtype": "complex64",
                }
            },
        )
    ]

    with pytest.raises(ValueError, match="different distribution"):
        _validate_conj_workloads("conj", correctness, timing)

    faithful = [
        FlagGemsV62Workload(
            name="source-builder",
            inputs={
                "case": {
                    "phase": "correctness",
                    "dtype": "complex32",
                    "shape": {"input": [256]},
                }
            },
        )
    ]
    _validate_conj_workloads("conj", faithful, timing)


def test_direct_agent_requires_bfloat16_dynamic_dtype_counterpart():
    def workload(name, dtype):
        return FlagGemsV62Workload(
            name=name,
            inputs={
                "input": {
                    "type": "random",
                    "shape": [8],
                    "dtype": dtype,
                    "device": "target",
                }
            },
        )

    primary = [workload("fp16", "float16"), workload("fp32", "float32")]
    with pytest.raises(ValueError, match="bfloat16 counterpart"):
        _validate_dynamic_float_dtypes(
            "dynamic_float", primary, require_bfloat16=True
        )

    _validate_dynamic_float_dtypes(
        "dynamic_float",
        [*primary, workload("bf16", "bfloat16")],
        require_bfloat16=True,
    )


def test_direct_agent_compactly_expands_bfloat16_correctness_matrix():
    def workload(name, shape, dtype, tolerance=None):
        return FlagGemsV62Workload(
            name=name,
            inputs={
                "input": {
                    "type": "random",
                    "shape": shape,
                    "dtype": dtype,
                    "device": "target",
                },
                "params": {"dtype": dtype},
            },
            tolerance=tolerance,
        )

    primary = [
        workload("case::float16::0", [8], "float16", {"rtol": 0.001}),
        workload("case::float16::1", [16], "float16", {"rtol": 0.001}),
        workload("case::float32::0", [8], "float32", {"rtol": 1.3e-6}),
        workload("case::float32::1", [16], "float32", {"rtol": 1.3e-6}),
    ]
    expanded = _expand_dynamic_float_dtypes(
        "dynamic_float",
        primary,
        [
            FlagGemsV62DtypeExpansion(
                source_dtype="float16",
                target_dtype="bfloat16",
                tolerance={"rtol": 0.016, "atol": 0.0001},
            )
        ],
        require_bfloat16=True,
    )

    assert len(expanded) == 6
    bfloat16 = [
        workload
        for workload in expanded
        if _nested_dtype(workload.inputs) == "bfloat16"
    ]
    assert [workload.name for workload in bfloat16] == [
        "case::bfloat16::0",
        "case::bfloat16::1",
    ]
    assert all(
        workload.inputs["params"]["dtype"] == "bfloat16"
        for workload in bfloat16
    )
    assert all(workload.tolerance.rtol == 0.016 for workload in bfloat16)


def test_direct_agent_compactly_expands_primary_float_dtype_matrix():
    source = FlagGemsV62Workload(
        name="case::float32::0",
        inputs={
            "input": {
                "type": "random",
                "shape": [8],
                "dtype": "float32",
                "device": "target",
            }
        },
    )

    expanded = _expand_dynamic_float_dtypes(
        "dynamic_float",
        [source],
        [
            FlagGemsV62DtypeExpansion(
                source_dtype="float32",
                target_dtype="float16",
            )
        ],
        require_bfloat16=False,
        require_primary_pair=True,
    )

    assert [workload.name for workload in expanded] == [
        "case::float32::0",
        "case::float16::0",
    ]
    assert _expand_dynamic_float_dtypes(
        "dynamic_float",
        expanded,
        [
            FlagGemsV62DtypeExpansion(
                source_dtype="float32",
                target_dtype="float16",
            )
        ],
        require_bfloat16=False,
        require_primary_pair=True,
    ) == expanded
    _validate_dynamic_float_dtypes(
        "dynamic_float",
        expanded,
        require_bfloat16=False,
        require_primary_pair=True,
    )


def test_direct_agent_dtype_expansion_inherits_case_dependent_tolerance():
    source = FlagGemsV62Workload(
        name="case::float16::0",
        inputs={
            "input": {
                "type": "random",
                "shape": [8],
                "dtype": "float16",
                "device": "target",
            }
        },
        tolerance={"atol": 0.005, "atol_scale": 8.0},
    )

    expanded = _expand_dynamic_float_dtypes(
        "dynamic_float",
        [source],
        [
            FlagGemsV62DtypeExpansion(
                source_dtype="float16",
                target_dtype="float32",
            )
        ],
        require_bfloat16=False,
        require_primary_pair=True,
    )

    assert expanded[1].tolerance == source.tolerance


def test_direct_agent_dtype_expansion_merges_explicit_tolerance_fields():
    source = FlagGemsV62Workload(
        name="case::float16::0",
        inputs={
            "input": {
                "type": "random",
                "shape": [8],
                "dtype": "float16",
                "device": "target",
            }
        },
        tolerance={"rtol": 0.001, "atol": 0.005, "atol_scale": 8.0},
    )

    expanded = _expand_dynamic_float_dtypes(
        "dynamic_float",
        [source],
        [
            FlagGemsV62DtypeExpansion(
                source_dtype="float16",
                target_dtype="float32",
                tolerance={"rtol": 1.3e-6},
            )
        ],
        require_bfloat16=False,
        require_primary_pair=True,
    )

    assert expanded[1].tolerance.rtol == 1.3e-6
    assert expanded[1].tolerance.atol == 0.005
    assert expanded[1].tolerance.atol_scale == 8.0


def test_direct_agent_allows_filtered_subset_expansion_for_target_dtype():
    source = FlagGemsV62Workload(
        name="case::float16::0",
        inputs={
            "input": {
                "type": "random",
                "shape": [1],
                "dtype": "float16",
                "device": "target",
            }
        },
    )
    rule = FlagGemsV62DtypeExpansion(
        source_dtype="float16",
        target_dtype="bfloat16",
    )

    expanded = _expand_dynamic_float_dtypes(
        "filtered",
        [source],
        [rule],
        require_bfloat16=False,
        require_primary_pair=False,
        target_dtypes={"float16", "float32", "bfloat16"},
    )

    assert [_nested_dtype(workload.inputs) for workload in expanded] == [
        "float16",
        "bfloat16",
    ]
    with pytest.raises(ValueError, match="unsupported correctness dtype expansion"):
        _expand_dynamic_float_dtypes(
            "filtered",
            [source],
            [rule],
            require_bfloat16=False,
            require_primary_pair=False,
            target_dtypes={"float16", "float32"},
        )


def test_resolved_target_float_dtypes_keeps_report_inferred_bfloat16():
    cases = [{"dtype": "torch.float32"}]

    assert _resolved_target_float_dtypes(
        cases,
        require_bfloat16=True,
    ) == {"float32", "bfloat16"}

    source = FlagGemsV62Workload(
        name="case::float32::0",
        inputs={"input": {"dtype": "float32", "shape": [1]}},
    )
    expanded = _expand_dynamic_float_dtypes(
        "report-inferred-bfloat16",
        [source],
        [
            FlagGemsV62DtypeExpansion(
                source_dtype="float32",
                target_dtype="bfloat16",
            )
        ],
        require_bfloat16=True,
        target_dtypes=_resolved_target_float_dtypes(
            cases,
            require_bfloat16=True,
        ),
    )

    assert [_nested_dtype(workload.inputs) for workload in expanded] == [
        "float32",
        "bfloat16",
    ]


def test_direct_agent_detects_primary_float_dtype_pair(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inventory = collect_source_inventory(
        str(repo), "adaptive_max_pool3d_backward"
    )
    cases = [
        {"dtype": "torch.float16"},
        {"dtype": "torch.float32"},
    ]

    assert not _requires_primary_float_dtype_pair(inventory, cases)
    (repo / "tests/test_adaptive_max_pool3d_backward.py").write_text(
        "@pytest.mark.adaptive_max_pool3d_backward\n"
        "@pytest.mark.parametrize('dtype', utils.FLOAT_DTYPES)\n"
        "def test_adaptive_max_pool3d_backward(dtype):\n"
        "    pass\n",
        encoding="utf-8",
    )
    inventory = collect_source_inventory(
        str(repo), "adaptive_max_pool3d_backward"
    )
    assert _requires_primary_float_dtype_pair(inventory, cases)


@pytest.mark.parametrize('declaration', [
    'FLOAT_DTYPES = [torch.float32, torch.float16]\n',
    'if QUICK_MODE:\n    FLOAT_DTYPES = [torch.float32]\nelse:\n    FLOAT_DTYPES = [torch.float32, torch.float16]\n',
    'from accuracy_utils import FLOAT_DTYPES\nFLOAT_DTYPES = [torch.float32]\n',
])
def test_direct_agent_does_not_expand_source_local_dtype_list(tmp_path, declaration):
    repo = _repo(tmp_path)
    (repo / 'tests/test_adaptive_max_pool3d_backward.py').write_text(
        declaration
        + '@pytest.mark.adaptive_max_pool3d_backward\n'
        + "@pytest.mark.parametrize('dtype', FLOAT_DTYPES)\n"
        + 'def test_adaptive_max_pool3d_backward(dtype):\n    pass\n',
        encoding='utf-8',
    )
    inventory = collect_source_inventory(str(repo), 'adaptive_max_pool3d_backward')
    cases = [{'dtype': f'torch.{dtype}'} for dtype in ('float16', 'float32', 'bfloat16')]
    assert not _requires_primary_float_dtype_pair(inventory, cases)
    assert not _requires_bfloat16_accuracy(inventory, cases)


def test_dtype_expansion_preserves_fixed_dtype_sibling_tests(tmp_path):
    repo = _repo(tmp_path)
    path = repo / 'tests/test_adaptive_max_pool3d_backward.py'
    path.write_text(
        '@pytest.mark.adaptive_max_pool3d_backward\n'
        "@pytest.mark.parametrize('dtype', utils.FLOAT_DTYPES)\n"
        'def test_matrix(dtype):\n    pass\n'
        '@pytest.mark.adaptive_max_pool3d_backward\n'
        'def test_empty():\n    pass\n', encoding='utf-8',
    )
    inventory = collect_source_inventory(str(repo), 'adaptive_max_pool3d_backward')
    prefix = 'tests/test_adaptive_max_pool3d_backward.py::'
    assert _dynamic_float_dtype_identities(inventory) == {prefix + 'test_matrix'}
    fixed = {prefix + 'test_empty', 'tests/another.py::test_matrix'}
    def row(test, dtype, shape):
        return FlagGemsV62Workload(name=prefix + test + '::' + dtype,
            inputs={'input': {'type': 'random', 'dtype': dtype, 'shape': shape}})
    original = [row('test_matrix', 'float32', [8]), row('test_empty', 'float32', [0])]
    expanded = _expand_dynamic_float_dtypes('adaptive_max_pool3d_backward', original,
        [FlagGemsV62DtypeExpansion(source_dtype='float32', target_dtype=d)
         for d in ('float16', 'bfloat16')],
        require_bfloat16=True, require_primary_pair=True, fixed_sources=fixed)
    assert len(expanded) == 4
    assert [r for r in expanded if 'test_empty' in r.name] == [original[1]]
    _validate_dynamic_float_dtypes('adaptive_max_pool3d_backward', expanded,
        require_bfloat16=True, require_primary_pair=True, fixed_sources=fixed)
    with pytest.raises(ValueError, match='incomplete'):
        _validate_dynamic_float_dtypes('adaptive_max_pool3d_backward', original,
            require_bfloat16=True, require_primary_pair=True, fixed_sources=fixed)


def test_direct_agent_does_not_expand_dtype_filtered_combined_cases(tmp_path):
    repo = _repo(tmp_path)
    cases = [
        {"dtype": "torch.float16"},
        {"dtype": "torch.float32"},
        {"dtype": "torch.bfloat16"},
    ]
    (repo / "tests/test_adaptive_max_pool3d_backward.py").write_text(
        "FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]\n"
        "FILTERED_CASES = [(shape, dtype) for shape in SHAPES "
        "for dtype in FLOAT_DTYPES if allowed(shape, dtype)]\n"
        "@pytest.mark.adaptive_max_pool3d_backward\n"
        "@pytest.mark.parametrize('shape,dtype', FILTERED_CASES)\n"
        "def test_adaptive_max_pool3d_backward(shape, dtype):\n"
        "    pass\n",
        encoding="utf-8",
    )
    inventory = collect_source_inventory(
        str(repo), "adaptive_max_pool3d_backward"
    )

    assert not _requires_primary_float_dtype_pair(inventory, cases)
    assert not _requires_bfloat16_accuracy(inventory, cases)


def test_direct_agent_normalizes_literal_accuracy_manual_seed(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tests/test_adaptive_max_pool3d_backward.py").write_text(
        "@pytest.mark.adaptive_max_pool3d_backward\n"
        "def test_adaptive_max_pool3d_backward():\n"
        "    torch.manual_seed(42)\n",
        encoding="utf-8",
    )
    inventory = collect_source_inventory(
        str(repo), "adaptive_max_pool3d_backward"
    )
    seeds = _accuracy_manual_seeds(
        inventory, "adaptive_max_pool3d_backward"
    )
    workload = FlagGemsV62Workload(
        name="tests/test.py::test_adaptive_max_pool3d_backward[case]",
        inputs={},
        seed=0,
    )

    normalized = _normalize_accuracy_manual_seeds([workload], seeds)

    assert seeds == {"test_adaptive_max_pool3d_backward": 42}
    assert normalized[0].seed == 42


def test_direct_agent_normalizes_literal_dtype_conditional_tolerance(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tests/test_adaptive_max_pool3d_backward.py").write_text(
        "@pytest.mark.adaptive_max_pool3d_backward\n"
        "def test_adaptive_max_pool3d_backward(dtype):\n"
        "    atol = 1e-2 if dtype == torch.float16 else 1e-4\n"
        "    utils.gems_assert_close(out, ref, dtype, atol=atol)\n",
        encoding="utf-8",
    )
    inventory = collect_source_inventory(
        str(repo), "adaptive_max_pool3d_backward"
    )
    tolerances = _accuracy_dtype_tolerances(
        inventory, "adaptive_max_pool3d_backward"
    )
    workloads = [
        FlagGemsV62Workload(
            name=f"test_adaptive_max_pool3d_backward::{dtype}",
            inputs={"case": {"dtype": dtype}},
            tolerance={"atol": 0.01},
        )
        for dtype in ("float16", "float32", "bfloat16")
    ]

    normalized = _normalize_accuracy_dtype_tolerances(workloads, tolerances)

    assert [workload.tolerance.atol for workload in normalized] == [
        0.01,
        0.0001,
        0.0001,
    ]


def test_direct_agent_uses_global_target_bfloat16_evidence(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tests/test_adaptive_max_pool3d_backward.py").write_text(
        "@pytest.mark.adaptive_max_pool3d_backward\n"
        "@pytest.mark.parametrize('dtype', utils.FLOAT_DTYPES)\n"
        "def test_adaptive_max_pool3d_backward(dtype):\n"
        "    pass\n",
        encoding="utf-8",
    )
    inventory = collect_source_inventory(
        str(repo), "adaptive_max_pool3d_backward"
    )
    report = tmp_path / "all-cases.json"
    report.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "op_name": "adaptive_max_pool3d_backward",
                        "cases": [{"dtype": "torch.float16"}],
                    },
                    {
                        "op_name": "sibling",
                        "cases": [{"dtype": "torch.bfloat16"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _requires_bfloat16_accuracy(
        inventory,
        [{"dtype": "torch.float16"}],
        report,
    )


def _nested_dtype(inputs):
    return inputs["input"]["dtype"]


def test_direct_agent_allows_target_device_random_generation(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["oracle"] = (
        "import torch\n\n"
        "REFERENCE_DEVICE = 'target'\n\n"
        "def gen_inputs(ctx, device):\n"
        "    generator = torch.Generator(device=device)\n"
        "    generator.manual_seed(ctx.get('seed', 0))\n"
        "    value = torch.randn((8,), device=device, generator=generator)\n"
        "    return {'grad_output': value, 'self': value, 'indices': value}\n\n"
        "def run(grad_output, self, indices):\n"
        "    return grad_output\n"
    )

    output = FlagGemsV62ExtractorAgent().run(
        inp.model_dump(), _Runtime(json.dumps(proposal))
    )

    assert "device=device" in output.oracle


def test_direct_agent_allows_guarded_target_cpu_device_alias(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["oracle"] = (
        "import torch\n\n"
        "REFERENCE_DEVICE = 'target'\n\n"
        "def gen_inputs(ctx, device):\n"
        "    try:\n"
        "        generator = torch.Generator(device=device)\n"
        "        gen_device = device\n"
        "    except Exception:\n"
        "        generator = torch.Generator(device='cpu')\n"
        "        gen_device = 'cpu'\n"
        "    generator.manual_seed(ctx.get('seed', 0))\n"
        "    value = torch.randn(\n"
        "        (8,), device=gen_device, generator=generator\n"
        "    ).to(device)\n"
        "    return {'grad_output': value, 'self': value, 'indices': value}\n\n"
        "def run(grad_output, self, indices):\n"
        "    return grad_output\n"
    )

    output = FlagGemsV62ExtractorAgent().run(
        inp.model_dump(), _Runtime(json.dumps(proposal))
    )

    assert "device=gen_device" in output.oracle


def test_direct_agent_allows_generator_device_factory_alias(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["oracle"] = (
        "import torch\n\n"
        "REFERENCE_DEVICE = 'target'\n\n"
        "def gen_inputs(ctx, device):\n"
        "    try:\n"
        "        generator = torch.Generator(device=device)\n"
        "    except Exception:\n"
        "        generator = torch.Generator(device='cpu')\n"
        "    generator.manual_seed(ctx.get('seed', 0))\n"
        "    factory_device = generator.device\n"
        "    value = torch.randn(\n"
        "        (8,), device=factory_device, generator=generator\n"
        "    ).to(device)\n"
        "    return {'grad_output': value, 'self': value, 'indices': value}\n\n"
        "def run(grad_output, self, indices):\n"
        "    return grad_output\n"
    )

    output = FlagGemsV62ExtractorAgent().run(
        inp.model_dump(), _Runtime(json.dumps(proposal))
    )

    assert "factory_device = generator.device" in output.oracle


def test_direct_agent_rejects_cpu_generator_with_target_factory(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["oracle"] = (
        "import torch\n\n"
        "REFERENCE_DEVICE = 'target'\n\n"
        "def gen_inputs(ctx, device):\n"
        "    try:\n"
        "        generator = torch.Generator(device=device)\n"
        "    except Exception:\n"
        "        generator = torch.Generator(device='cpu')\n"
        "    generator.manual_seed(ctx.get('seed', 0))\n"
        "    value = torch.empty((8,), device=device).uniform_(\n"
        "        generator=generator\n"
        "    )\n"
        "    return {'grad_output': value, 'self': value, 'indices': value}\n\n"
        "def run(grad_output, self, indices):\n"
        "    return grad_output\n"
    )

    with pytest.raises(Exception, match="generator and factory devices"):
        FlagGemsV62ExtractorAgent().run(
            inp.model_dump(), _Runtime(json.dumps(proposal))
        )


def test_direct_agent_rejects_hard_coded_random_device(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["oracle"] = (
        "import torch\n\n"
        "REFERENCE_DEVICE = 'target'\n\n"
        "def gen_inputs(ctx, device):\n"
        "    value = torch.randn((8,), device='cuda:0')\n"
        "    return {'grad_output': value, 'self': value, 'indices': value}\n\n"
        "def run(grad_output, self, indices):\n"
        "    return grad_output\n"
    )

    with pytest.raises(Exception, match="hard-coded target devices"):
        FlagGemsV62ExtractorAgent().run(
            inp.model_dump(), _Runtime(json.dumps(proposal))
        )


def test_direct_agent_checks_random_generation_helpers(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["oracle"] = (
        "import torch\n\n"
        "REFERENCE_DEVICE = 'target'\n\n"
        "def _make():\n"
        "    return torch.randn((8,), device='cuda:0')\n\n"
        "def gen_inputs(ctx, device):\n"
        "    value = _make()\n"
        "    return {'grad_output': value, 'self': value, 'indices': value}\n\n"
        "def run(grad_output, self, indices):\n"
        "    return grad_output\n"
    )

    with pytest.raises(Exception, match="hard-coded target devices"):
        FlagGemsV62ExtractorAgent().run(
            inp.model_dump(), _Runtime(json.dumps(proposal))
        )


def test_direct_agent_rejects_cpu_retry_for_target_allocation(tmp_path):
    repo = _repo(tmp_path)
    report = tmp_path / "cases.json"
    _case_report(report)
    inp = FlagGemsV62ExtractorInput(
        operator="adaptive_max_pool3d_backward",
        flaggems_repo=str(repo),
        case_list_path=str(report),
    )
    proposal = json.loads(_proposal())
    proposal["oracle"] = (
        "import torch\n\n"
        "REFERENCE_DEVICE = 'target'\n\n"
        "def _make(device):\n"
        "    try:\n"
        "        value = torch.randn((8,), device=device)\n"
        "    except Exception:\n"
        "        value = torch.randn((8,), device='cpu').to(device)\n"
        "    return value\n\n"
        "def gen_inputs(ctx, device):\n"
        "    value = _make(device)\n"
        "    return {'grad_output': value, 'self': value, 'indices': value}\n\n"
        "def run(grad_output, self, indices):\n"
        "    return grad_output\n"
    )

    with pytest.raises(Exception, match="guard only target Generator construction"):
        FlagGemsV62ExtractorAgent().run(
            inp.model_dump(), _Runtime(json.dumps(proposal))
        )

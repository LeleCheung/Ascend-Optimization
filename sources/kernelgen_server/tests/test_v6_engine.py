import pytest

torch = pytest.importorskip("torch")

from kernelgen_server.evaluation.engine import EvaluationEngine
from kernelgen_server.runtime.device import Device
from kernelgen_server.schema import (
    Definition,
    EvaluateRequest,
    EvaluationSettings,
    EvaluationStatus,
    Implementation,
    SourceFile,
    Workload,
    WorkloadStatus,
)


class CpuTestDevice(Device):
    backend = "cpu"

    def __init__(self):
        self.time_calls = 0

    def set_device(self, device: str) -> None:
        return None

    def synchronize(self, device: str) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def is_available(self) -> bool:
        return True

    def list_devices(self) -> list[str]:
        return ["cpu"]

    def device_name(self, device: str) -> str:
        return "CPU"

    def time(self, fn, args, warmup, iters, device, grad_to_none=None):
        del warmup, iters, device, grad_to_none
        self.time_calls += 1
        fn(*args)
        return 1.0


def _inplace_definition() -> Definition:
    return Definition(
        api_version="v6.0",
        name="scale_",
        parameters=[
            {"name": "self", "required": True},
            {
                "name": "scale",
                "kind": "keyword_only",
                "required": False,
                "default": 1,
            },
        ],
        outputs=["out"],
        effects={
            "mutates": ["self"],
            "returns_alias_of": {"out": "self"},
        },
        reference="def run(self, *, scale=1): return self.mul_(scale)",
        reference_device="cpu",
    )


def _implementation(definition: Definition, source: str) -> Implementation:
    return Implementation(
        name="candidate",
        definition=definition.name,
        language="python",
        entrypoint="main.py::run",
        sources=[SourceFile(path="main.py", content=source)],
    )


def _scale_workload(name="scale", *, scale=True) -> Workload:
    inputs = {"self": {"type": "random", "shape": [8], "dtype": "float32"}}
    if scale:
        inputs["scale"] = {"type": "scalar", "value": 2}
    return Workload(name=name, inputs=inputs)


def test_mutation_alias_and_keyword_only_parameter_pass():
    definition = _inplace_definition()
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(
            definition,
            "def run(self, *, scale=1): return self.mul_(scale)",
        ),
        correctness_workloads=[_scale_workload()],
    )
    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)
    assert result.status == EvaluationStatus.PASSED


def test_missing_mutation_and_alias_are_rejected():
    definition = _inplace_definition()
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(
            definition,
            "def run(self, *, scale=1): return self * scale",
        ),
        correctness_workloads=[_scale_workload()],
    )
    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)
    assert result.status == EvaluationStatus.INCORRECT_NUMERICAL
    assert "does not alias" in result.log


def test_undeclared_mutation_is_rejected():
    definition = Definition(
        api_version="v6.0",
        name="identity",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference="def run(x): return x + 1",
        reference_device="cpu",
    )
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(
            definition,
            "def run(x):\n    x.add_(1)\n    return x",
        ),
        correctness_workloads=[
            Workload(
                name="mutation",
                inputs={"x": {"type": "random", "shape": [8], "dtype": "float32"}},
            )
        ],
    )
    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)
    assert result.status == EvaluationStatus.INCORRECT_NUMERICAL
    assert "unexpectedly mutated" in result.log


def test_timing_runs_correctness_gate_before_device_timer():
    definition = _inplace_definition().model_copy(update={"reference_device": "target"})
    device = CpuTestDevice()
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(
            definition,
            "def run(self, *, scale=1): return self + scale",
        ),
        timing_workloads=[_scale_workload(name="timing")],
        settings=EvaluationSettings(warmup_ms=0, benchmark_ms=1),
    )
    result = EvaluationEngine(device, "cpu").evaluate(request)
    assert result.per_workload[0].status == WorkloadStatus.INCORRECT_NUMERICAL
    assert "correctness/effects gate failed" in result.per_workload[0].log
    assert device.time_calls == 0


def test_v62_timing_skips_correctness_and_effects_gate():
    definition = _inplace_definition().model_copy(
        update={
            "api_version": "v6.2",
            "reference": (
                "def correctness_run(self, *, scale=1): "
                "return self.mul_(scale)\n"
                "def timing_run(self, *, scale=1): "
                "return self.mul_(scale)\n"
            ),
            "reference_device": "target",
        }
    )
    device = CpuTestDevice()
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(
            definition,
            # Numerically and semantically wrong, but executable and therefore
            # valid for the performance-only timing phase.
            "def run(self, *, scale=1): return self + scale",
        ),
        timing_workloads=[_scale_workload(name="timing")],
        settings=EvaluationSettings(warmup_ms=0, benchmark_ms=1),
    )
    result = EvaluationEngine(device, "cpu").evaluate(request)
    assert result.status == EvaluationStatus.PASSED
    assert result.per_workload[0].status == WorkloadStatus.PASSED
    assert result.per_workload[0].speedup == 1.0
    assert device.time_calls == 2


def test_v62_custom_valid_replaces_return_value_comparison_and_gets_named_inputs():
    definition = Definition(
        api_version="v6.2",
        name="random_like",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference=(
            "def correctness_run(x): return x + 1\n"
            "def valid(ref_outputs, sol_outputs, inputs, ctx):\n"
            "    assert set(inputs) == {'x'}\n"
            "    assert inputs['x'].shape == sol_outputs[0].shape\n"
            "    return {\n"
            "        'passed': sol_outputs[0].dtype == ref_outputs[0].dtype,\n"
            "        'metrics': {'named_inputs': True},\n"
            "    }\n"
        ),
        reference_device="cpu",
    )
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(definition, "def run(x): return x + 999"),
        correctness_workloads=[
            Workload(
                name="custom",
                inputs={"x": {"type": "random", "shape": [8], "dtype": "float32"}},
            )
        ],
    )

    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)

    assert result.status == EvaluationStatus.PASSED
    assert result.per_workload[0].metrics == {"named_inputs": True}


def test_v62_custom_valid_replaces_declared_mutation_value_comparison():
    definition = _inplace_definition().model_copy(
        update={
            "api_version": "v6.2",
            "reference": (
                "def correctness_run(self, *, scale=1): return self.mul_(scale)\n"
                "def valid(ref_outputs, sol_outputs, inputs, ctx):\n"
                "    assert sol_outputs[0] is not ref_outputs[0]\n"
                "    assert sol_outputs[0].shape == ref_outputs[0].shape\n"
                "    return True\n"
            ),
        }
    )
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(
            definition,
            "def run(self, *, scale=1): return self.add_(scale)",
        ),
        correctness_workloads=[_scale_workload()],
    )

    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)

    assert result.status == EvaluationStatus.PASSED


def test_v62_custom_valid_cannot_override_return_structure_mismatch():
    definition = Definition(
        api_version="v6.2",
        name="wrong_shape",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference=(
            "def correctness_run(x): return x\n"
            "def valid(ref_outputs, sol_outputs, inputs, ctx): return True\n"
        ),
        reference_device="cpu",
    )
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(definition, "def run(x): return x[:1]"),
        correctness_workloads=[
            Workload(
                name="shape",
                inputs={"x": {"type": "random", "shape": [8], "dtype": "float32"}},
            )
        ],
    )

    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)

    assert result.status == EvaluationStatus.INCORRECT_NUMERICAL
    assert "tensor shape differs" in result.log


def test_v62_custom_valid_can_explicitly_own_the_return_contract():
    definition = Definition(
        api_version="v6.2",
        name="sequence_kind_agnostic",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference=(
            "import torch\n"
            "VALID_OWNS_RETURN_CONTRACT = True\n"
            "def correctness_run(x): return [x, x + 1]\n"
            "def valid(ref_outputs, sol_outputs, inputs, ctx):\n"
            "    if len(ref_outputs) != len(sol_outputs): return False\n"
            "    return all(\n"
            "        isinstance(ref, torch.Tensor)\n"
            "        and isinstance(sol, torch.Tensor)\n"
            "        and ref.shape == sol.shape\n"
            "        and ref.dtype == sol.dtype\n"
            "        and torch.equal(ref, sol)\n"
            "        for ref, sol in zip(ref_outputs, sol_outputs)\n"
            "    )\n"
        ),
        reference_device="cpu",
    )
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(
            definition,
            "def run(x): return (x, x + 1)",
        ),
        correctness_workloads=[
            Workload(
                name="container",
                inputs={"x": {"type": "random", "shape": [8], "dtype": "float32"}},
            )
        ],
    )

    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)

    assert result.status == EvaluationStatus.PASSED


def test_v62_return_contract_ownership_requires_valid_hook():
    definition = Definition(
        api_version="v6.2",
        name="missing_valid",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference=(
            "VALID_OWNS_RETURN_CONTRACT = True\n"
            "def correctness_run(x): return x\n"
        ),
        reference_device="cpu",
    )
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(definition, "def run(x): return x"),
        correctness_workloads=[
            Workload(
                name="missing-valid",
                inputs={"x": {"type": "random", "shape": [8], "dtype": "float32"}},
            )
        ],
    )

    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)

    assert result.status == EvaluationStatus.RUNTIME_ERROR
    assert "requires a valid hook" in result.log


def test_v60_custom_valid_does_not_replace_return_value_comparison():
    definition = Definition(
        api_version="v6.0",
        name="legacy_valid",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference=(
            "def run(x): return x + 1\n"
            "def valid(ref_outputs, sol_outputs, inputs, ctx): return True\n"
        ),
        reference_device="cpu",
    )
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(definition, "def run(x): return x + 999"),
        correctness_workloads=[
            Workload(
                name="legacy",
                inputs={"x": {"type": "random", "shape": [8], "dtype": "float32"}},
            )
        ],
    )

    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)

    assert result.status == EvaluationStatus.INCORRECT_NUMERICAL


def test_gen_inputs_handles_framework_specific_shapes_without_schema_types():
    definition = Definition(
        api_version="v6.0",
        name="sum_list",
        parameters=[{"name": "values", "required": True}],
        outputs=["out"],
        reference=(
            "import torch\n"
            "def gen_inputs(ctx, device):\n"
            "    n = ctx['inputs']['count']\n"
            "    return {'values': tuple(torch.arange(4, device=device) for _ in range(n))}\n"
            "def run(values): return sum(values)\n"
        ),
        reference_device="cpu",
    )
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(
            definition,
            "def run(values): return sum(values)",
        ),
        correctness_workloads=[Workload(name="tuple", inputs={"count": 3})],
    )
    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)
    assert result.status == EvaluationStatus.PASSED


def test_gen_inputs_none_keeps_ordinary_recipe_values():
    definition = Definition(
        api_version="v6.0",
        name="mixed_inputs",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference=(
            "import torch\n"
            "def gen_inputs(ctx, device):\n"
            "    case = ctx['inputs'].get('case')\n"
            "    if case is None:\n"
            "        return None\n"
            "    return {'x': torch.arange(case['n'], device=device)}\n"
            "def run(x): return x\n"
        ),
        reference_device="cpu",
    )
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(definition, "def run(x): return x"),
        correctness_workloads=[
            Workload(
                name="direct",
                inputs={
                    "x": {"type": "random", "shape": [4], "dtype": "float32"}
                },
            ),
            Workload(name="generated", inputs={"case": {"n": 4}}),
        ],
    )

    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)

    assert result.status == EvaluationStatus.PASSED


def test_v62_output_path_supplies_golden_without_running_correctness_reference(
    tmp_path,
):
    from safetensors.torch import save_file

    output_path = tmp_path / "golden.safetensors"
    save_file({"out": torch.arange(4, dtype=torch.float32) + 1}, output_path)
    definition = Definition(
        api_version="v6.2",
        name="plus_one",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference="def correctness_run(x): raise RuntimeError('must not run')",
        reference_device="cpu",
    )
    input_path = tmp_path / "inputs.safetensors"
    save_file({"x": torch.arange(4, dtype=torch.float32)}, input_path)
    request = EvaluateRequest(
        api_version="v6.2",
        definition=definition,
        implementation=_implementation(definition, "def run(x): return x + 1"),
        correctness_workloads=[
            Workload(
                name="golden",
                inputs={"x": {"type": "safetensor"}},
                input_path=str(input_path),
                output_path=str(output_path),
            )
        ],
    )

    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)

    assert result.status == EvaluationStatus.PASSED


def test_v62_output_path_preserves_inplace_alias_and_mutation_checks(tmp_path):
    from safetensors.torch import save_file

    input_path = tmp_path / "inputs.safetensors"
    output_path = tmp_path / "golden.safetensors"
    save_file({"self": torch.arange(4, dtype=torch.float32)}, input_path)
    save_file({"out": torch.arange(4, dtype=torch.float32) + 2}, output_path)
    definition = Definition(
        api_version="v6.2",
        name="add_",
        parameters=[
            {"name": "self", "required": True},
            {"name": "value", "required": True},
        ],
        outputs=["out"],
        effects={
            "mutates": ["self"],
            "returns_alias_of": {"out": "self"},
        },
        reference="def correctness_run(self, value): raise RuntimeError('must not run')",
        reference_device="cpu",
    )
    request = EvaluateRequest(
        api_version="v6.2",
        definition=definition,
        implementation=_implementation(
            definition, "def run(self, value): return self.add_(value)"
        ),
        correctness_workloads=[
            Workload(
                name="golden-inplace",
                inputs={
                    "self": {"type": "safetensor"},
                    "value": {"type": "scalar", "value": 2},
                },
                input_path=str(input_path),
                output_path=str(output_path),
            )
        ],
    )

    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)

    assert result.status == EvaluationStatus.PASSED


def test_v62_timing_rejects_output_path_but_allows_input_path(tmp_path):
    from safetensors.torch import save_file

    tensor_path = tmp_path / "values.safetensors"
    save_file({"x": torch.ones(1)}, tensor_path)
    path = str(tensor_path)
    definition = Definition(
        api_version="v6.2",
        name="identity",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference="def run(x): return x",
    )
    implementation = _implementation(definition, "def run(x): return x")

    request = EvaluateRequest(
        api_version="v6.2",
        definition=definition,
        implementation=implementation,
        timing_workloads=[
            Workload(
                name="timing-input",
                inputs={"x": {"type": "safetensor"}},
                input_path=path,
            )
        ],
        settings=EvaluationSettings(warmup_ms=0, benchmark_ms=1),
    )
    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(request)
    assert result.status == EvaluationStatus.PASSED

    with pytest.raises(ValueError, match="KGS must time the reference"):
        EvaluateRequest(
            api_version="v6.2",
            definition=definition,
            implementation=implementation,
            timing_workloads=[
                Workload(
                    name="timing",
                    inputs={"x": {"type": "safetensor"}},
                    input_path=path,
                    output_path=path,
                )
            ],
        )


def test_preflight_smokes_only_timing_workloads():
    definition = Definition(
        api_version="v6.0",
        name="identity",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference="def run(x): return x",
    )
    request = EvaluateRequest(
        api_version=definition.api_version,
        definition=definition,
        implementation=_implementation(definition, "def run(x): return x"),
        correctness_workloads=[
            Workload(
                name="correctness",
                inputs={"x": {"type": "random", "shape": [4], "dtype": "float32"}},
            )
        ],
        timing_workloads=[
            Workload(
                name="timing",
                inputs={"x": {"type": "random", "shape": [8], "dtype": "float32"}},
            )
        ],
    )

    result = EvaluationEngine(CpuTestDevice(), "cpu").preflight(request)

    assert result["status"] == "PASSED"
    assert list(result["per_workload"]) == ["timing"]

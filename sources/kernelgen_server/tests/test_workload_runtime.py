from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kernelgen_server.evaluation import workload_runtime
from kernelgen_server.protocol.schema import Definition, Workload


def test_target_random_recipe_materializes_on_call_device(monkeypatch):
    definition = Definition.model_validate(
        {
            "api_version": "v6.2",
            "name": "identity",
            "parameters": [{"name": "x", "required": True}],
            "outputs": ["out"],
        }
    )
    workload = Workload.model_validate(
        {
            "name": "target-random",
            "inputs": {
                "x": {
                    "type": "random",
                    "shape": [4],
                    "dtype": "float32",
                    "device": "target",
                }
            },
        }
    )
    calls = []

    def fake_random(spec, generator, *, device="cpu"):
        calls.append((spec, generator, device))
        return "target-value"

    monkeypatch.setattr(workload_runtime, "_random_tensor", fake_random)
    cpu_bases = workload_runtime.make_cpu_bases(workload)
    values = workload_runtime.materialize_values(
        definition,
        SimpleNamespace(gen_inputs=None),
        workload,
        cpu_bases,
        "cuda:7",
    )

    assert cpu_bases == {}
    assert values == {"x": "target-value"}
    assert calls == [
        (
            {
                "type": "random",
                "shape": [4],
                "dtype": "float32",
                "device": "target",
            },
            None,
            "cuda:7",
        )
    ]


def test_random_recipe_rejects_physical_device_id():
    workload = Workload.model_validate(
        {
            "name": "physical-device",
            "inputs": {
                "x": {
                    "type": "random",
                    "shape": [4],
                    "dtype": "float32",
                    "device": "cuda:0",
                }
            },
        }
    )

    with pytest.raises(ValueError, match="device must be cpu or target"):
        workload_runtime.make_cpu_bases(workload)


def test_random_recipe_preserves_distribution_and_transform_order():
    generator = torch.Generator(device="cpu").manual_seed(123)
    expected_generator = torch.Generator(device="cpu").manual_seed(123)
    expected_scale = torch.rand((), generator=expected_generator)
    expected = torch.rand((3, 4), generator=expected_generator)
    expected = (expected * expected_scale).softmax(dim=-1)

    actual = workload_runtime._random_tensor(
        {
            "type": "random",
            "shape": [3, 4],
            "dtype": "float32",
            "distribution": "uniform",
            "transforms": [
                {
                    "op": "multiply_random",
                    "input": {
                        "type": "random",
                        "shape": [],
                        "dtype": "float32",
                        "distribution": "uniform",
                    },
                    "before": True,
                },
                {"op": "softmax", "dim": -1},
            ],
        },
        generator,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_integer_recipe_uses_source_dtype_before_cast_and_arithmetic():
    generator = torch.Generator(device="cpu").manual_seed(321)
    expected_generator = torch.Generator(device="cpu").manual_seed(321)
    expected = torch.randint(0, 2, (8,), generator=expected_generator)
    expected = expected.to(torch.float32) * 2 - 1

    actual = workload_runtime._random_tensor(
        {
            "type": "random",
            "shape": [8],
            "dtype": "float32",
            "source_dtype": "int64",
            "distribution": "integer",
            "low": 0,
            "high": 2,
            "transforms": [
                {"op": "cast", "dtype": "float32"},
                {"op": "multiply", "value": 2},
                {"op": "subtract", "value": 1},
            ],
        },
        generator,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_safetensor_recipe_loads_named_tensor_from_input_path(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "inputs.safetensors"
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    save_file({"x": expected, "unused": torch.ones(1)}, path)
    workload = Workload.model_validate(
        {
            "name": "file-backed",
            "input_path": str(path),
            "inputs": {
                "x": {
                    "type": "safetensor",
                    "shape": [999],
                    "dtype": "float16",
                }
            },
        }
    )

    cpu_bases = workload_runtime.make_cpu_bases(workload)

    assert torch.equal(cpu_bases["x"], expected)
    assert torch.equal(cpu_bases["unused"], torch.ones(1))
    assert workload.context()["input_path"] == str(path)


def test_safetensor_recipe_requires_input_path():
    workload = Workload.model_validate(
        {
            "name": "missing-path",
            "inputs": {"x": {"type": "safetensor"}},
        }
    )

    with pytest.raises(ValueError, match="require workload.input_path"):
        workload_runtime.make_cpu_bases(workload)

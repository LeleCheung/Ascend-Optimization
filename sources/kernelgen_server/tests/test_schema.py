import pytest
from pydantic import ValidationError

from kernelgen_server.schema import (
    BoundEvaluateRequest,
    Definition,
    EvaluateRequest,
    EvaluateResponse,
    EvaluatorBinding,
    Implementation,
    SourceFile,
    Workload,
)
from kernelgen_server.evaluation.effects import named_outputs


def _definition(**updates) -> Definition:
    values = {
        "name": "addmm_",
        "parameters": [
            {"name": "self", "required": True},
            {"name": "mat1", "required": True},
            {"name": "mat2", "required": True},
            {
                "name": "beta",
                "kind": "keyword_only",
                "required": False,
                "default": 1,
            },
            {
                "name": "alpha",
                "kind": "keyword_only",
                "required": False,
                "default": 1,
            },
        ],
        "outputs": ["out"],
        "effects": {
            "mutates": ["self"],
            "returns_alias_of": {"out": "self"},
        },
        "reference": "def run(self, mat1, mat2, *, beta=1, alpha=1): return self",
    }
    values.update(updates)
    return Definition(**values)


def _implementation(source: str | None = None) -> Implementation:
    return Implementation(
        name="candidate",
        definition="addmm_",
        language="python",
        entrypoint="main.py::run",
        sources=[
            SourceFile(
                path="main.py",
                content=source or _definition().reference,
            )
        ],
    )


def _workload(name: str = "small") -> Workload:
    return Workload(
        name=name,
        inputs={
            "self": {"type": "random", "shape": [2, 2], "dtype": "float32"},
            "mat1": {"type": "random", "shape": [2, 2], "dtype": "float32"},
            "mat2": {"type": "random", "shape": [2, 2], "dtype": "float32"},
        },
    )


def test_simplified_definition_preserves_exact_abi_defaults_and_effects():
    definition = _definition()
    assert [parameter.name for parameter in definition.parameters] == [
        "self",
        "mat1",
        "mat2",
        "beta",
        "alpha",
    ]
    assert definition.parameters[-1].kind.value == "keyword_only"
    assert definition.parameters[-1].default == 1
    assert definition.effects.mutates == ["self"]


def test_optional_parameter_must_explicitly_store_null_default():
    with pytest.raises(ValidationError, match="must declare a default"):
        Definition(
            name="bad",
            parameters=[
                {"name": "out", "kind": "keyword_only", "required": False}
            ],
            outputs=["output"],
            reference="def run(*, out=None): return out",
        )


def test_void_operator_accepts_empty_outputs_and_none_return():
    definition = _definition(
        outputs=[],
        effects={"mutates": ["self"], "returns_alias_of": {}},
        reference="def run(self, mat1, mat2, *, beta=1, alpha=1): return None",
    )
    assert definition.outputs == []
    assert named_outputs(definition, None) == {}
    with pytest.raises(TypeError, match="declares no outputs"):
        named_outputs(definition, object())


def test_var_positional_is_a_small_abi_kind_not_a_type_dsl():
    definition = Definition(
        name="broadcast_tensors",
        parameters=[
            {
                "name": "tensors",
                "kind": "var_positional",
                "required": True,
                "type_hint": "Tensor",
            }
        ],
        outputs=["out"],
        reference="def run(*tensors): return tensors",
    )
    assert str(definition.parameters[0].kind.value) == "var_positional"

    with pytest.raises(ValidationError, match="var_positional"):
        Definition(
            name="bad",
            parameters=[
                {
                    "name": "items",
                    "kind": "var_positional",
                    "required": False,
                    "default": None,
                }
            ],
            outputs=["out"],
            reference="def run(*items): return items",
        )


def test_generic_schema_intentionally_rejects_type_and_call_dsl():
    with pytest.raises(ValidationError, match="type"):
        Definition(
            name="identity",
            parameters=[{"name": "x", "type": "Tensor", "required": True}],
            outputs=["out"],
            reference="def run(x): return x",
        )
    with pytest.raises(ValidationError, match="call"):
        Workload(name="bad", inputs={"x": 1}, call="identity(x)")


def test_workload_inputs_are_plain_json_generator_context():
    workload = Workload(
        name="context",
        inputs={
            "shape": [2, 4],
            "dtype": "float16",
            "nested": {"page_size": 16, "enabled": True},
        },
    )
    assert workload.context()["inputs"]["nested"]["page_size"] == 16


def test_wire_payload_preserves_absent_fields_and_explicit_null_defaults():
    definition_data = _definition().model_dump(mode="json", exclude_unset=True)
    definition_data["parameters"].append(
        {
            "name": "out",
            "kind": "keyword_only",
            "required": False,
            "default": None,
        }
    )
    definition_data["reference"] = (
        "def run(self, mat1, mat2, *, beta=1, alpha=1, out=None): return self"
    )
    definition = Definition.model_validate(definition_data)
    request = EvaluateRequest(
        definition=definition,
        implementation=_implementation(definition.reference),
        correctness_workloads=[_workload()],
    )

    payload = request.wire_payload()
    assert payload["api_version"] == "v6.2"
    assert "default" not in payload["definition"]["parameters"][0]
    assert payload["definition"]["parameters"][-1]["default"] is None
    EvaluateRequest.model_validate(payload)


def test_cpu_reference_rejects_timing_workloads():
    with pytest.raises(ValidationError, match="correctness-only"):
        EvaluateRequest(
            definition=_definition(reference_device="cpu"),
            implementation=_implementation(),
            timing_workloads=[_workload()],
        )


def test_workload_names_are_unique_across_phases():
    duplicate = _workload(name="same")
    with pytest.raises(ValidationError, match="globally unique"):
        EvaluateRequest(
            definition=_definition(),
            implementation=_implementation(),
            correctness_workloads=[duplicate],
            timing_workloads=[duplicate],
        )


def test_bound_request_has_no_translated_workloads_or_adapter_fields():
    request = BoundEvaluateRequest(
        binding=EvaluatorBinding(
            catalog_name="flaggems-adapter-definitions", definition="addmm_"
        ),
        implementation=_implementation(),
    )
    payload = request.wire_payload()
    assert payload["binding"] == {
        "catalog_name": "flaggems-adapter-definitions",
        "definition": "addmm_",
    }
    assert "correctness_workloads" not in payload
    assert "timing_workloads" not in payload
    assert "definition" not in payload
    assert "adapter" not in payload


def test_both_v6_runners_share_exactly_one_response_model():
    assert "binding" in BoundEvaluateRequest.model_fields
    assert EvaluateResponse.model_json_schema()["title"] == "EvaluateResponse"


def test_source_path_cannot_escape():
    with pytest.raises(ValidationError):
        SourceFile(path="../main.py", content="")


def test_v5_definition_is_explicitly_rejected():
    with pytest.raises(ValidationError):
        Definition.model_validate(
            {
                "api_version": "v5.1",
                "name": "identity",
                "inputs": ["x"],
                "outputs": ["output"],
                "reference": "def run(x): return x",
            }
        )


def test_v60_request_remains_parseable_and_preserves_wire_version():
    definition = _definition(api_version="v6.0")
    request = EvaluateRequest(
        api_version="v6.0",
        definition=definition,
        implementation=_implementation(),
        correctness_workloads=[_workload()],
    )

    payload = request.wire_payload()
    assert payload["api_version"] == "v6.0"
    assert payload["definition"]["api_version"] == "v6.0"
    assert EvaluateRequest.model_validate(payload).api_version == "v6.0"


def test_request_and_definition_versions_must_match():
    with pytest.raises(ValidationError, match="must match"):
        EvaluateRequest(
            api_version="v6.0",
            definition=_definition(api_version="v6.2"),
            implementation=_implementation(),
            correctness_workloads=[_workload()],
        )

import pytest
from pydantic import ValidationError

from kernelgen_server.profiling.models import ProfileTarget
from kernelgen_server.schema import Definition, Implementation, SourceFile, Workload


def _definition() -> Definition:
    return Definition(
        name="identity",
        parameters=[{"name": "x", "required": True}],
        outputs=["output"],
        reference="def run(x): return x",
    )


def _implementation(definition: str = "identity") -> Implementation:
    return Implementation(
        name="candidate",
        definition=definition,
        language="python",
        entrypoint="main.py::run",
        sources=[SourceFile(path="main.py", content="def run(x): return x")],
    )


def test_profile_target_uses_simplified_workload():
    definition = _definition()
    workload = Workload(
        name="default",
        inputs={"x": {"type": "scalar", "value": 1}},
    )
    target = ProfileTarget(
        implementation=_implementation(),
        definition=definition,
        workload=workload,
        expected_backend="cuda",
    )
    assert target.definition.name == target.implementation.definition
    assert target.workload.name == "default"


def test_profile_target_rejects_definition_mismatch():
    with pytest.raises(ValidationError, match="must match"):
        ProfileTarget(
            implementation=_implementation("other"),
            definition=_definition(),
            workload=Workload(name="bad", inputs={}),
            expected_backend="cuda",
        )

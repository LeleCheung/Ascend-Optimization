import pytest

from kernelgen_server.evaluation.hack_detection import detect_obvious_hack
from kernelgen_server.evaluation.metadata_policy import METADATA_OPERATORS, permits_no_jit
from kernelgen_server.schema import Implementation, SourceFile


def implementation(code, *, name="candidate"):
    return Implementation(name=name, definition="candidate_claim", language="triton",
                          entrypoint="main.py::run", sources=[SourceFile(path="main.py", content=code)])


@pytest.mark.parametrize("operator,code", [
    ("lift_fresh", "def run(x):\n    return x\n"),
    ("unsqueeze_", "def run(x, dim):\n    return x.unsqueeze_(dim)\n"),
    ("unsqueeze_", "import torch\ndef run(x: torch.Tensor, dim: int):\n    x.unsqueeze_(dim)\n    return x\n"),
])
def test_reviewed_metadata_recipes_do_not_require_dummy_kernels(operator, code):
    impl = implementation(code)
    assert permits_no_jit(impl, operator)
    assert not detect_obvious_hack(impl, operator_name=operator).is_hack
    # The same candidate cannot grant its own exception.
    assert detect_obvious_hack(impl).is_hack
    assert detect_obvious_hack(impl, operator_name="matmul").is_hack


@pytest.mark.parametrize("operator,code", [
    ("resize_output_", "def run(x):\n    return x\n"),
    ("unbind_copy", "def run(x):\n    return x\n"),
    ("custom_lift_fresh", "def run(x):\n    return x\n"),
    ("lift_fresh", "def run(x):\n    return x.clone()\n"),
    ("lift_fresh", "import torch\ndef run(x):\n    return torch.matmul(x,x)\n"),
    ("lift_fresh", "import torch\ntorch.add = lambda x: x\ndef run(x):\n    return x\n"),
    ("lift_fresh", "import torch\ndef run(x: torch.set_grad_enabled(False)):\n    return x\n"),
    ("lift_fresh", "def run(x):\n    x.zero_()\n    return x\n"),
    ("unsqueeze_", "def run(x, dim):\n    return x.unsqueeze(dim)\n"),
    ("unsqueeze_", "def run(x, dim):\n    return x.unsqueeze_(0)\n"),
    ("unsqueeze_", "def run(x, dim):\n    return x.clone().unsqueeze_(dim)\n"),
])
def test_operator_whitelist_does_not_approve_other_code(operator, code):
    impl = implementation(code, name=operator)
    assert not permits_no_jit(impl, operator)
    assert detect_obvious_hack(impl, operator_name=operator).is_hack


def test_whitelist_is_maintainer_owned_and_auxiliary_sources_do_not_bypass_it():
    assert set(METADATA_OPERATORS) == {"lift_fresh", "unsqueeze_"}
    with pytest.raises(TypeError):
        METADATA_OPERATORS["matmul"] = "candidate declaration"
    impl = implementation("def run(x):\n    return x\n")
    impl.sources.append(SourceFile(path="helper.py", content="import torch\ntorch.add = None\n"))
    assert not permits_no_jit(impl, "lift_fresh")

"""Host checks for native reference dtype preservation, not hardware capability."""

from types import SimpleNamespace

import pytest

from kernelgen.agents.extractor.flaggems.v62 import _CORRECTNESS_RUN
from kernelgen.agents.extractor.flaggems.v62_agent import _validate_oracle


def test_addmm_reference_keeps_inputs_scalars_and_mutation():
    copied = []
    target = SimpleNamespace(copy_=copied.append)
    left, right, output = object(), object(), object()
    calls = []
    def addmm(*args, **kwargs):
        calls.append((args, kwargs))
        return output
    namespace = {"torch": SimpleNamespace(addmm=addmm)}
    exec(_CORRECTNESS_RUN, namespace)
    assert namespace["correctness_run"](target, left, right, beta=2, alpha=3) is target
    assert calls == [((target, left, right), {"beta": 2, "alpha": 3})]
    assert copied == [output]


@pytest.mark.parametrize("body", [
    "return x.to(torch.float64)",
    "return x.to(dtype=torch.float64)",
    "return x.double()",
    "return x if x.dtype == torch.float64 else x",
    "return torch.relu(x)",
])
def test_oracle_structure_does_not_decide_source_reference_precision(body):
    definition = SimpleNamespace(parameters=[SimpleNamespace(name="x", kind="positional_or_keyword",
                                  required=True, default=None, type_hint="Tensor")])
    source = "import torch\nREFERENCE_DEVICE = 'target'\ndef correctness_run(x):\n    " + body + "\n"
    def validate():
        _validate_oracle("relu", definition, source, has_correctness=True, has_timing=False,
                         needs_gen_inputs=False, mixes_generated_and_direct_inputs=False,
                         needs_torch_fallback=False)
    validate()

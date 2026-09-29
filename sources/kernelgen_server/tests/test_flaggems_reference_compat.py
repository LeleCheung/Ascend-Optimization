import sys
from types import ModuleType

from kernelgen_server.evaluation.adapters.flaggems import reference_compat
from kernelgen_server.evaluation.adapters.flaggems.reference_compat import (
    prepare_flaggems_reference,
)


def test_metax_linear_backward_reference_compat_is_scoped_and_idempotent(
    monkeypatch,
):
    calls = []

    class JITFunction:
        def _pack_args(self, backend, kwargs, *rest):
            calls.append((backend, kwargs, rest))
            return kwargs

    flag_gems = ModuleType("flag_gems")
    flag_gems.vendor_name = "metax"
    jit = ModuleType("triton.runtime.jit")
    jit.JITFunction = JITFunction
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    monkeypatch.setitem(sys.modules, "triton.runtime.jit", jit)

    prepare_flaggems_reference("linear_backward")
    first_pack = JITFunction()._pack_args("metax", {"SPLIT_K": 1, "BLOCK": 32})
    prepare_flaggems_reference("linear_backward")

    assert first_pack == {"BLOCK": 32}
    assert JITFunction._kg_splitk_compat is True

    monkeypatch.setattr(flag_gems, "vendor_name", "nvidia")
    prepare_flaggems_reference("linear_backward")


def test_reference_compat_does_not_import_framework_for_other_operators(
    monkeypatch,
):
    def unexpected_import(name):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(reference_compat.importlib, "import_module", unexpected_import)

    prepare_flaggems_reference("addmm_")

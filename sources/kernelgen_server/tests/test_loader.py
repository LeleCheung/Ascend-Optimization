from pathlib import Path
from types import ModuleType

import pytest

from kernelgen_server.evaluation.loader import (
    BuildError,
    _module_belongs_to,
    load_implementation,
    load_operator_adapter,
    materialize_implementation,
)
from kernelgen_server.schema import Definition, Implementation, SourceFile


def _definition(**updates):
    values = {
        "api_version": "v6.0",
        "name": "add_one",
        "parameters": [
            {"name": "x", "required": True},
            {
                "name": "scale",
                "kind": "keyword_only",
                "required": False,
                "default": 1,
            },
        ],
        "outputs": ["output"],
        "reference": "def run(x, *, scale=1): return x + scale",
    }
    values.update(updates)
    return Definition(**values)


def test_oracle_module_cleanup_tolerates_noniterable_module_path(tmp_path):
    module = ModuleType("nonstandard_namespace")
    module.__path__ = object()

    assert _module_belongs_to(module, Path(tmp_path)) is False


def test_multifile_python_implementation_and_public_alias():
    definition = _definition()
    implementation = Implementation(
        name="candidate",
        definition="add_one",
        language="python",
        entrypoint="main.py::run",
        sources=[
            SourceFile(
                path="main.py",
                content=(
                    "from helper import add_one_impl\n\n"
                    "def run(x, *, scale=1): return add_one_impl(x, scale)"
                ),
            ),
            SourceFile(
                path="helper.py",
                content="def add_one_impl(x, scale): return x + scale",
            ),
        ],
    )
    candidate = load_implementation(implementation, definition)
    assert candidate(3, scale=2) == 5


def test_operator_adapter_loads_optional_fixed_correctness_hook():
    adapter = load_operator_adapter(
        _definition(
            reference=(
                "def run(x, *, scale=1): return x + scale\n"
                "def correctness(x, *, scale=1): return x + scale + 1\n"
            )
        )
    )

    assert adapter.reference(3, scale=2) == 6
    assert adapter.timing_reference(3, scale=2) == 5


def test_operator_adapter_rejects_correctness_hook_abi_mismatch():
    with pytest.raises(BuildError, match="ABI"):
        load_operator_adapter(
            _definition(
                reference=(
                    "def run(x, *, scale=1): return x + scale\n"
                    "def correctness(x, scale=1): return x + scale\n"
                )
            )
        )


def test_v62_operator_adapter_uses_named_oracle_entrypoints():
    definition = _definition(
        api_version="v6.2",
        reference=(
            "def correctness_run(x, *, scale=1): return x + scale + 1\n"
            "def timing_run(x, *, scale=1): return x + scale\n"
            "def torch_run(x, *, scale=1): return x - scale\n"
        ),
    )

    primary = load_operator_adapter(definition)
    fallback = load_operator_adapter(definition, "torch_fallback")

    assert primary.reference(3, scale=2) == 6
    assert primary.timing_reference(3, scale=2) == 5
    assert primary.has_torch_fallback is True
    assert fallback.reference(3, scale=2) == 1
    assert fallback.timing_reference(3, scale=2) == 1
    assert fallback.reference_source == "torch_fallback"


def test_v62_operator_adapter_uses_shared_run_with_phase_override():
    definition = _definition(
        api_version="v6.2",
        reference=(
            "def run(x, *, scale=1): return x + scale\n"
            "def correctness_run(x, *, scale=1): return x + scale + 1\n"
        ),
    )

    adapter = load_operator_adapter(definition)

    assert adapter.reference(3, scale=2) == 6
    assert adapter.timing_reference(3, scale=2) == 5


def test_v62_operator_adapter_uses_shared_run_for_both_phases():
    adapter = load_operator_adapter(
        _definition(
            api_version="v6.2",
            reference="def run(x, *, scale=1): return x + scale\n",
        )
    )

    assert adapter.reference is adapter.timing_reference
    assert adapter.reference(3, scale=2) == 5


def test_v62_operator_adapter_rejects_named_entrypoint_abi_mismatch():
    with pytest.raises(BuildError, match="timing_run ABI"):
        load_operator_adapter(
            _definition(
                api_version="v6.2",
                reference=(
                    "def correctness_run(x, *, scale=1): return x\n"
                    "def timing_run(x, scale=1): return x\n"
                ),
            )
        )


def test_candidate_rejects_keyword_only_mismatch():
    definition = _definition()
    implementation = Implementation(
        name="bad",
        definition=definition.name,
        language="python",
        entrypoint="main.py::run",
        sources=[SourceFile(path="main.py", content="def run(x, scale=1): return x")],
    )
    with pytest.raises(BuildError, match="ABI differs"):
        load_implementation(implementation, definition)


def test_candidate_accepts_exact_var_positional_abi():
    definition = Definition(
        name="broadcast_tensors",
        parameters=[
            {
                "name": "tensors",
                "kind": "var_positional",
                "required": True,
            }
        ],
        outputs=["out"],
    )
    candidate = load_implementation(
        Implementation(
            name="candidate",
            definition=definition.name,
            language="python",
            entrypoint="main.py::run",
            sources=[
                SourceFile(
                    path="main.py",
                    content="def run(*tensors): return tensors",
                )
            ],
        ),
        definition,
    )
    assert candidate(1, 2) == (1, 2)


def test_candidate_rejects_conflicting_public_symbol():
    definition = _definition()
    implementation = Implementation(
        name="bad",
        definition=definition.name,
        language="python",
        entrypoint="main.py::run",
        sources=[
            SourceFile(
                path="main.py",
                content=(
                    "def run(x, *, scale=1): return x\n"
                    "def add_one(x, *, scale=1): return x + scale\n"
                ),
            )
        ],
    )
    with pytest.raises(BuildError, match="conflicting public symbol"):
        load_implementation(implementation, definition)


def test_explicit_source_root_preserves_keyword_only_abi(tmp_path):
    definition = _definition()
    implementation = Implementation(
        name="candidate",
        definition=definition.name,
        language="python",
        entrypoint="main.py::run",
        sources=[
            SourceFile(
                path="main.py",
                content="def run(x, *, scale=1): return x + scale",
            )
        ],
    )
    source_root = materialize_implementation(
        implementation, tmp_path / "candidate"
    )

    candidate = load_implementation(
        implementation,
        definition,
        source_root=source_root,
    )

    assert candidate(3, scale=2) == 5


def test_explicit_source_root_preserves_var_positional_abi(tmp_path):
    definition = Definition(
        name="broadcast_tensors",
        parameters=[
            {"name": "tensors", "kind": "var_positional", "required": True}
        ],
        outputs=["out"],
    )
    implementation = Implementation(
        name="candidate",
        definition=definition.name,
        language="python",
        entrypoint="main.py::run",
        sources=[
            SourceFile(path="main.py", content="def run(*tensors): return tensors")
        ],
    )
    source_root = materialize_implementation(
        implementation, tmp_path / "candidate"
    )

    candidate = load_implementation(
        implementation,
        definition,
        source_root=source_root,
    )

    assert candidate(1, 2) == (1, 2)


def test_explicit_source_root_rejects_missing_or_changed_source(tmp_path):
    definition = _definition()
    implementation = Implementation(
        name="candidate",
        definition=definition.name,
        language="python",
        entrypoint="main.py::run",
        sources=[
            SourceFile(
                path="main.py",
                content="def run(x, *, scale=1): return x + scale",
            )
        ],
    )
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    with pytest.raises(BuildError, match="source is missing"):
        load_implementation(
            implementation,
            definition,
            source_root=missing_root,
        )

    changed_root = materialize_implementation(
        implementation, tmp_path / "changed"
    )
    (changed_root / "main.py").write_text(
        "def run(x, *, scale=1): return x - scale",
        encoding="utf-8",
    )
    with pytest.raises(BuildError, match="source differs"):
        load_implementation(
            implementation,
            definition,
            source_root=changed_root,
        )


def test_explicit_source_root_rejects_conflicting_public_symbol(tmp_path):
    definition = _definition()
    implementation = Implementation(
        name="candidate",
        definition=definition.name,
        language="python",
        entrypoint="main.py::run",
        sources=[
            SourceFile(
                path="main.py",
                content=(
                    "def run(x, *, scale=1): return x\n"
                    "def add_one(x, *, scale=1): return x + scale\n"
                ),
            )
        ],
    )
    source_root = materialize_implementation(
        implementation, tmp_path / "candidate"
    )

    with pytest.raises(BuildError, match="conflicting public symbol"):
        load_implementation(
            implementation,
            definition,
            source_root=source_root,
        )

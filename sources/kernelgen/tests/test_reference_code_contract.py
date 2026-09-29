"""Language-neutral evidence is data; legacy names normalize at input boundaries."""

from pathlib import Path

import pytest

from kernelgen.agents.coder import CoderInput
from kernelgen.cli.legacy_batch_simple_opt import bind_reference_code_paths
from kernelgen.framework.run_options import launcher_parser, resolve_run_options, validate_options
from kernelgen.workflows.optimization.kernelgen.contracts import KernelGenInput
from kernelgen.workflows.optimization.single_coder.reference import load_reference_code
from kernelgen.workflows.legacy.simple_opt import SimpleOptInput


DEFINITION = {
    "name": "identity", "op_type": "elementwise", "axes": {},
    "inputs": {"x": {"shape": [16], "dtype": "float32"}},
    "outputs": {"out": {"shape": [16], "dtype": "float32"}},
    "reference": "def run(x): return x\n",
}


@pytest.mark.parametrize("filename,source", [
    ("identity.cu", '__global__ void kernel(float* x) { x[0] = 1; }'),
    ("identity.cpp", 'extern "C" __global__ __aicore__ void kernel(GM_ADDR x) {}'),
    ("identity.py", 'raise AssertionError("reference must never execute")'),
])
def test_reference_accepts_language_neutral_text_without_execution(tmp_path, filename, source):
    path = tmp_path / filename
    path.write_text(source)
    loaded = load_reference_code(path, None)
    assert loaded == {"reference_code_source": source, "reference_code_prompt": ""}
    assert bind_reference_code_paths([path], ["identity"]) == {"identity": path}
    inp = KernelGenInput(definition=DEFINITION, target_hardware="target", **loaded)
    assert inp.definition.reference == DEFINITION["reference"]
    assert inp.initial_seed_code == ""
    assert not inp.initial_seed_is_validated_baseline


@pytest.mark.parametrize("mode", ["simple_opt", "kernelgen"])
def test_cli_old_and_new_reference_names_share_one_destination(mode):
    parser = launcher_parser(mode)
    common = ["--definition", "identity"]
    new = parser.parse_args([*common, "--reference-code-path", "source.cu"])
    old = parser.parse_args([*common, "--reference-triton-path", "source.cu"])
    assert vars(old) == vars(new)
    assert new.reference_code_path == Path("source.cu")
    assert "reference_triton_path" not in vars(new)


def test_yaml_aliases_canonicalize_and_conflicts_are_not_silently_ignored(tmp_path):
    result = validate_options({"reference_triton_path": "source.cu"}, base=tmp_path)
    assert result == {"reference_code_path": tmp_path / "source.cu"}
    resolved = resolve_run_options({"mode": "kernelgen", **result})
    assert resolved["reference_code_path"] == tmp_path / "source.cu"
    with pytest.raises(ValueError, match="conflicting values"):
        validate_options({"reference_triton_path": "a.cu", "reference_code_path": "b.cu"})


def test_persisted_old_field_names_load_but_only_canonical_names_are_written():
    inp = SimpleOptInput(definition_name="identity", reference_triton_path="source.cu",
                         reference_triton_prompt_path="guidance.md")
    assert inp.reference_code_path == Path("source.cu")
    assert inp.reference_code_prompt_path == Path("guidance.md")
    assert not any("reference_triton" in key for key in inp.model_dump())
    for model in (KernelGenInput, CoderInput):
        item = model(definition=DEFINITION, target_hardware="target",
                     reference_triton_source="CUDA source", reference_triton_prompt="read only")
        assert item.reference_code_source == "CUDA source"
        assert item.reference_code_prompt == "read only"
        assert not any("reference_triton" in key for key in item.model_dump())

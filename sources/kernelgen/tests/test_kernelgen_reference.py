"""Unvalidated reference evidence travels through CLI/launcher/epochs, not seeds."""

from types import SimpleNamespace

import pytest

from kernelgen.cli.api import build_request
from kernelgen.cli.batch import load_batch_file
from kernelgen.cli.state import set_max_workers
from kernelgen.framework.run_options import launcher_parser, resolve_run_options
from kernelgen.workflows.optimization.kernelgen.contracts import KernelGenInput
from kernelgen.workflows.optimization.kernelgen import epoch
from kernelgen.workflows.optimization.single_coder.reference import load_reference_code


DEFINITION = {
    "name": "identity", "op_type": "elementwise", "axes": {},
    "inputs": {"x": {"shape": [16], "dtype": "float32"}},
    "outputs": {"out": {"shape": [16], "dtype": "float32"}},
    "reference": "def run(x): return x\n",
}


@pytest.mark.parametrize("with_seed", [False, True])
def test_reference_reaches_every_epoch_without_becoming_seed(tmp_path, monkeypatch, with_seed):
    source = tmp_path / "failed.py"
    source.write_text("raise RuntimeError('must never execute reference')\n")
    prompt = tmp_path / "context.md"
    prompt.write_text("Prior result: FAILED; preserve the original correctness oracle.")
    inp = KernelGenInput(
        definition=DEFINITION, target_hardware="target", n_parallel=3, n_epoch=2,
        **load_reference_code(source, prompt),
        initial_seed_code="validated seed" if with_seed else "",
        initial_seed_is_validated_baseline=with_seed,
    )
    monkeypatch.setattr(epoch, "resolve_builtin_catalog_path", lambda _: tmp_path)
    monkeypatch.setattr(epoch, "load_catalog_optimization_context", lambda *_: (inp.definition, []))
    for number in (1, 2):
        seed = inp.initial_seed_code if number == 1 else "measured epoch winner"
        inputs = epoch.build_epoch_inputs(inp, {}, [], seed, number)
        assert len(inputs) == 3
        for index, item in enumerate(inputs):
            assert item["reference_code_source"] == source.read_text()
            assert item["reference_code_prompt"] == prompt.read_text()
            assert item["seed_code"] == seed
            assert item["seed_is_validated_baseline"] == (with_seed and number == 1 and index == 0)
            assert item["definition"]["reference"] == DEFINITION["reference"]


def test_reference_guidance_requires_source():
    with pytest.raises(ValueError, match="requires reference_code_source"):
        KernelGenInput(definition=DEFINITION, target_hardware="target", reference_code_prompt="failed")


def test_batch_request_and_launcher_pass_reference_as_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    source = tmp_path / "failed.py"
    source.write_text("invalid python is permitted as evidence")
    prompt = tmp_path / "context.md"
    prompt.write_text("Failed in first-stage optimization.")
    batch = tmp_path / "batch.yaml"
    batch.write_text("""version: 1
defaults:
  mode: kernelgen
  n_parallel: 2
operators:
  - definition: identity
    reference_code_path: failed.py
    reference_code_prompt_path: context.md
""")
    _, defaults, items, _ = load_batch_file(batch)
    item = dict(items[0])
    definition = item.pop("definition")
    values = resolve_run_options(defaults, item)
    set_max_workers(values["eval_server"], 2)
    request = build_request(values, definition=definition, workspace=tmp_path / "run")
    args = launcher_parser("kernelgen").parse_args(request.workflow_args)
    assert args.reference_code_path == source
    assert args.reference_code_prompt_path == prompt
    assert args.seed_code_path is None
    assert request.worker_weight == 2

    from kernelgen.examples.kernel_gen import run_example
    captured = []
    monkeypatch.setattr(run_example, "setup_workspace", lambda *a, **k: None)
    monkeypatch.setattr(run_example, "load_definition", lambda *a: DEFINITION)
    monkeypatch.setattr(run_example, "print_results", lambda *a: None)
    monkeypatch.setattr(run_example, "resolve_cli_runtime_options", lambda *a, **k: dict(model="test", base_url=None, auth_token=None))
    monkeypatch.setenv("FIB_EVAL_SERVER", "http://127.0.0.1:8000")
    class Workflow:
        def __init__(self, **kwargs):
            pass

        def run(self, inp):
            captured.append(KernelGenInput.model_validate(inp))
            return SimpleNamespace(status="PASSED")
    monkeypatch.setattr(run_example, "KernelGenWorkflow", Workflow)
    with pytest.raises(SystemExit) as exc:
        run_example.main([*request.workflow_args, "--runtime", "codex"])
    assert exc.value.code == 0
    assert captured[0].reference_code_source == source.read_text()
    assert captured[0].reference_code_prompt == prompt.read_text()
    assert captured[0].initial_seed_code == ""
    assert not captured[0].initial_seed_is_validated_baseline


@pytest.mark.parametrize("kind,message", [
    ("missing", "does not exist"), ("directory", "regular file"),
    ("large", "prompt limit"), ("encoding", "UTF-8"), ("empty", "is empty"),
    ("orphan_prompt", "requires reference_code_path"),
])
def test_launcher_rejects_bad_reference_before_runtime(tmp_path, monkeypatch, capsys, kind, message):
    from kernelgen.examples.kernel_gen import run_example
    source = tmp_path / "source.py"
    if kind == "directory":
        source.mkdir()
    elif kind == "large":
        source.write_bytes(b"x" * 1_000_001)
    elif kind == "encoding":
        source.write_bytes(b"\xff")
    elif kind == "empty":
        source.write_text(" \n")
    monkeypatch.setattr(run_example, "resolve_cli_runtime_options", lambda *a, **k: pytest.fail("must reject first"))
    option = "--reference-code-prompt-path" if kind == "orphan_prompt" else "--reference-code-path"
    with pytest.raises(SystemExit) as exc:
        run_example.main(["--definition", "identity", option, str(source)])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err

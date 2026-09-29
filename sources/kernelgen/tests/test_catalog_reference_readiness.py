from types import SimpleNamespace

import pytest

from kernelgen.workflows.optimization.reference_readiness import reference_implementation
from kernelgen_server.evaluation.candidate_admission import check_candidate_admission
from kernelgen_server.evaluation.loader import load_implementation
from kernelgen_server import Definition


def build(source):
    operator = SimpleNamespace(definition=SimpleNamespace(name="test_op", reference=source))
    candidate = reference_implementation(operator)
    assert operator.definition.reference == source
    return candidate


def test_evaluator_hooks_not_copied_into_candidate():
    candidate = build('''
import torch
def run(x):
    return x + 1
def gen_inputs(ctx, device):
    torch.backends.cuda.matmul.allow_tf32 = False
def valid(ref, sol, inputs, ctx):
    return True
''')
    source = candidate.sources[0].content
    assert "gen_inputs" not in source and "valid" not in source
    assert candidate.entrypoint == "main.py::run"
    assert not check_candidate_admission(candidate, operator_name="test_op").is_hack


def test_correctness_entrypoint_does_not_rebind_run():
    candidate = build('''
def helper(x):
    return x + 1
def run(x):
    return helper(x)
def correctness_run(x):
    return (run(x),)
def timing_run(x):
    return run(x)
''')
    definition = Definition(api_version="v6.2", name="test_op",
                            parameters=[{"name": "x", "kind": "positional_or_keyword", "required": True}],
                            outputs=["out"])
    assert candidate.entrypoint == "main.py::run"
    assert load_implementation(candidate, definition)(2) == (3,)


def test_candidate_owned_mutation_is_still_rejected():
    candidate = build('''
import torch
def run(x):
    torch.backends.cuda.matmul.allow_tf32 = False
    return x
''')
    assert check_candidate_admission(candidate, operator_name="test_op").is_hack


def test_hook_dependency_fails_explicitly():
    with pytest.raises(ValueError, match="depends on evaluator hooks: gen_inputs"):
        build('''
def gen_inputs(ctx, device):
    return ctx
def run(x):
    return gen_inputs(x, None)
''')


def test_missing_callable_fails_explicitly():
    with pytest.raises(ValueError, match="requires correctness_run or run"):
        build("def timing_run(x): return x")

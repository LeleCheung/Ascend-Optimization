"""Pre-import policy checks require no Torch installation or device."""
from pathlib import Path

import pytest

from kernelgen_server.evaluation.candidate_admission import (
    CandidateAdmissionError, admission_capability, check_candidate_admission, require_candidate_admission,
)
from kernelgen_server.schema import Implementation, SourceFile


KERNEL = '''
import triton
import triton.language as tl
@triton.jit
def kernel(x):
    pass
def run(x):
    kernel[(1,)](x)
'''


def impl(source, language="triton"):
    return Implementation(name="candidate", definition="op", language=language,
                          entrypoint="main.py::run", sources=[SourceFile(path="main.py", content=source)])


@pytest.mark.parametrize("source", [
    "import torch as t\nt.add = lambda x: x",
    "from torch import ops as o\no.aten.add = lambda x: x",
    "from triton.runtime.jit import JITFunction as J\nJ.run = lambda *a: None",
    "import pytest\ndel pytest.fail",
    "import torch\ntorch.library.Library('aten', 'IMPL')",
    "import torch\ntorch.manual_seed(42)",
])
def test_protected_state_is_rejected_for_native_and_gems_even_in_python_mode(source):
    for evaluator in ("native", "flaggems"):
        decision = check_candidate_admission(impl(source + '\ndef run(x): return x\n', "python"), operator_name="op", evaluator_kind=evaluator)
        assert decision.is_hack
        assert 'protected candidate state' in decision.hack_reason


def test_explicit_fallback_rejects_but_unknown_receiver_is_only_review_signal():
    direct = KERNEL.replace('    kernel[(1,)](x)', '    import torch\n    kernel[(1,)](x)\n    return torch.matmul(x, x)')
    assert check_candidate_admission(impl(direct), operator_name="op").is_hack
    receiver = KERNEL.replace('    kernel[(1,)](x)', '    kernel[(1,)](x)\n    return x.softmax(-1)')
    decision = check_candidate_admission(impl(receiver), operator_name="op")
    assert not decision.is_hack
    assert decision.review_reasons and 'ADMISSION_REVIEW' in decision.log


def test_zero_grid_is_not_a_kernel_and_empty_branch_does_not_hide_real_launch():
    zero = KERNEL.replace('kernel[(1,)](x)', 'kernel[(0,)](x)')
    assert 'no nonzero launch' in check_candidate_admission(impl(zero), operator_name="op").hack_reason
    both = zero + '\ndef other(x):\n    kernel[(1,)](x)\n'
    assert 'no nonzero launch' not in check_candidate_admission(impl(both), operator_name="op").hack_reason


def test_metadata_exemption_retains_shared_gate_and_requires_bound_operator():
    metadata = impl('def run(x): return x')
    assert not require_candidate_admission(metadata, operator_name='lift_fresh').is_hack
    with pytest.raises(CandidateAdmissionError, match='no @triton.jit'):
        require_candidate_admission(metadata, operator_name='matmul')


def test_policy_inspection_does_not_execute_top_level_code(tmp_path):
    marker = tmp_path / 'executed'
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nimport torch\ntorch.add = None\ndef run(x): return x\n"
    with pytest.raises(CandidateAdmissionError):
        require_candidate_admission(impl(source), operator_name='op')
    assert not marker.exists()


def test_capability_fingerprints_the_actual_rule_files():
    capability = admission_capability()
    assert capability['version'] == 1
    assert set(capability['stages']) == {'preflight'}
    assert len(capability['policy_sha256']) == 64
    assert capability == admission_capability()


def test_python_callable_list_index_zero_is_not_a_zero_grid():
    source = 'def run(x):\n    calls = [lambda y: y]\n    return calls[0](x)\n'
    assert not check_candidate_admission(impl(source, 'python'), operator_name='op').is_hack


@pytest.mark.parametrize("source", [
    "import flag_gems as g\nsetattr(g, 'register', lambda *a: None)",
    "import flag_gems\nflag_gems.use_gems()",
    "from flag_gems import runtime as r\nr.register = None",
])
def test_gems_rules_only_apply_to_the_trusted_flaggems_adapter(source):
    candidate = impl(source + '\ndef run(x): return x\n', 'python')
    assert not check_candidate_admission(candidate, operator_name='flaggems_named_native_op').is_hack
    with pytest.raises(CandidateAdmissionError, match='protected candidate state'):
        require_candidate_admission(candidate, operator_name='op', evaluator_kind='flaggems')


@pytest.mark.parametrize('source', [
    'import torch\ntorch.cuda.empty_cache()',
    'from torch.cuda import empty_cache as clear\nclear()',
    'import torch\nclear = torch.cuda.empty_cache\nclear()',
    'import torch_npu\ntorch_npu.npu.empty_cache()',
    'import torch\ntorch.cuda.memory.change_current_allocator(None)',
    'from torch.cuda.memory import _set_allocator_settings as configure\nconfigure("max_split_size_mb:128")',
    'import torch\ntorch.cuda.set_per_process_memory_fraction(0.5)',
    'import os\nos.environ["TRITON_PPU_LLC_PATH"] = "/tmp/shim"',
    'import os\ndel os.environ["TRITON_CACHE_DIR"]',
    'from os import environ as env\nenv.update(TRITON_CACHE_DIR="/tmp")',
    'import os\nenv = os.environ\nenv.setdefault("MODE", "1")',
    'import os\nos.environ |= {"MODE": "1"}',
    'import os\nos.putenv("MODE", "1")',
    'import os\nsetattr(os, "environ", {})',
    'with open("/tmp/shim.py", "w") as f:\n    f.write("shim")',
    'from pathlib import Path as P\np = P("/tmp/shim.py")\np.write_text("shim")',
    'from pathlib import Path\nPath("/tmp/shim.py").open(mode="a")',
    'import os\nos.chmod("/tmp/shim.py", 0o755)',
    'import subprocess as sp\nsp.run(["patch", "compiler.py"])',
])
def test_candidate_environment_mutations_reject_in_both_modes(source):
    for evaluator in ('native', 'flaggems'):
        decision = check_candidate_admission(impl(source + '\ndef run(x): return x\n', 'python'), operator_name='op', evaluator_kind=evaluator)
        assert decision.is_hack
        assert 'candidate environment mutation:' in decision.hack_reason


@pytest.mark.parametrize('source', [
    'import os\nx = os.environ.get("MODE")\ny = os.getenv("MODE")',
    'import os\nlocal = os.environ.copy()\nlocal.update(MODE="1")',
    'from pathlib import Path\nx = Path("/tmp/info").read_text()',
    'with open("/tmp/info", "r") as f:\n    x = f.read()',
    'import torch\ndef helper(x):\n    return torch.empty_like(x), x.shape, x.stride(), x.dtype',
    'def helper(obj):\n    obj.empty_cache()\n    obj.write_text("not a known file")',
])
def test_read_only_environment_metadata_and_unknown_receivers_are_not_rejected(source):
    for evaluator in ('native', 'flaggems'):
        decision = check_candidate_admission(impl(source + '\ndef run(x): return x\n', 'python'), operator_name='op', evaluator_kind=evaluator)
        assert not decision.is_hack, decision.hack_reason


def test_registration_context_is_gems_specific_and_rejected_before_entering():
    source = 'from flag_gems.testing import override_registered_op as override\nctx = override("op", None)\nctx.__enter__()\ndef run(x): return x\n'
    candidate = impl(source, 'python')
    assert not check_candidate_admission(candidate, operator_name='op').is_hack
    assert check_candidate_admission(candidate, operator_name='op', evaluator_kind='flaggems').is_hack


@pytest.mark.parametrize('evaluator', ['native', 'flaggems'])
def test_triton_workspace_allocator_is_allowed_but_callback_cannot_clear_cache(evaluator):
    workspace = '''
import torch
def alloc_fn(size, alignment, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")
triton.set_allocator(alloc_fn)
'''
    source = KERNEL + workspace
    decision = check_candidate_admission(impl(source), operator_name='op', evaluator_kind=evaluator)
    assert not decision.is_hack, decision.hack_reason
    unsafe = source.replace('    return torch.empty(', '    torch.cuda.empty_cache()\n    return torch.empty(')
    decision = check_candidate_admission(impl(unsafe), operator_name='op', evaluator_kind=evaluator)
    assert decision.is_hack
    assert 'torch.cuda.empty_cache' in decision.hack_reason

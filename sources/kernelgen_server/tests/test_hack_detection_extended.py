"""Extended anti-hack tests for this repo's ``evaluation/hack_detection.py``.

Run with the repo helper (handles the hyphenated-directory import quirk)::

    ./run_tests.sh tests/test_hack_detection_extended.py -v

This repo's ``detect_obvious_hack`` is a SINGLE static-AST layer with a
BLACKLIST policy. It flags a triton implementation when ANY of:

* it calls a forbidden torch API (exact set: torch.bmm/einsum/matmul/mm, or a
  prefix in torch.ops./torch.nn./torch.linalg./torch.fft.), OR
* it has no ``@triton.jit`` kernel, OR
* it never launches a discovered kernel (``kernel[grid](...)``).

The tests below split into two groups:

1. Behaviours the detector HANDLES (must pass).
2. KNOWN GAPS of the blacklist approach (recorded as xfail): a fake kernel that
   is launched but whose real compute uses a torch op OUTSIDE the small
   blacklist (e.g. torch.softmax) slips through. A whitelist policy -- like the
   one in KernelGenBench's anti_hack.py -- would catch these.
"""

from kernelgen_server.evaluation.hack_detection import detect_obvious_hack
from kernelgen_server.protocol.schema import Implementation, SourceFile


def _impl(source: str, *, language: str = "triton", path: str = "main.py") -> Implementation:
    return Implementation(
        name="candidate",
        definition="op",
        language=language,
        entrypoint=f"{path}::run",
        sources=[SourceFile(path=path, content=source)],
    )


# A legitimate triton implementation: real @triton.jit kernel, launched, and
# only uses torch for allocation.
PURE_TRITON = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_one_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

def run(x):
    out = torch.empty_like(x)
    n = x.numel()
    add_one_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    return out
"""


# ==========================================================================
# Group 1: behaviours the detector correctly HANDLES
# ==========================================================================

def test_pure_triton_not_flagged():
    result = detect_obvious_hack(_impl(PURE_TRITON))
    assert result.is_hack is False, f"honest triton wrongly flagged: {result.hack_reason}"
    assert result.hack_reason == ""


def test_forbidden_matmul_flagged():
    """torch.matmul is on the exact blacklist."""
    result = detect_obvious_hack(_impl("import torch\ndef run(x): return torch.matmul(x, x)"))
    assert result.is_hack is True
    assert "torch.matmul" in result.hack_reason


def test_forbidden_bmm_mm_einsum_flagged():
    for op in ("bmm", "mm", "einsum"):
        result = detect_obvious_hack(_impl(f"import torch\ndef run(x): return torch.{op}(x, x)"))
        assert result.is_hack is True, f"torch.{op} not flagged"
        assert f"torch.{op}" in result.hack_reason


def test_torch_nn_prefix_flagged():
    """torch.nn.* prefix is blacklisted (e.g. F.linear via functional)."""
    code = "import torch.nn.functional as F\ndef run(x, w): return F.linear(x, w)"
    result = detect_obvious_hack(_impl(code))
    assert result.is_hack is True
    assert "linear" in result.hack_reason


def test_torch_linalg_prefix_flagged():
    code = "import torch\ndef run(x): return torch.linalg.svd(x)"
    result = detect_obvious_hack(_impl(code))
    assert result.is_hack is True
    assert "torch.linalg.svd" in result.hack_reason


def test_torch_ops_prefix_flagged():
    code = "import torch\ndef run(x): return torch.ops.aten.sum(x)"
    result = detect_obvious_hack(_impl(code))
    assert result.is_hack is True
    assert "torch.ops.aten.sum" in result.hack_reason


def test_missing_triton_kernel_flagged():
    """A 'triton' impl with no @triton.jit kernel is flagged."""
    result = detect_obvious_hack(_impl("import torch\ndef run(x): return x + 1"))
    assert result.is_hack is True
    assert "no @triton.jit kernel found" in result.hack_reason
    assert "no Triton kernel launch found" in result.hack_reason


def test_kernel_defined_but_not_launched_flagged():
    """Defining a kernel but never launching it is flagged."""
    code = """
import triton
import triton.language as tl

@triton.jit
def k(x, out):
    tl.store(out, tl.load(x))

def run(x):
    return x + 1
"""
    result = detect_obvious_hack(_impl(code))
    assert result.is_hack is True
    assert "no Triton kernel launch found" in result.hack_reason
    assert "no @triton.jit" not in result.hack_reason  # kernel exists, just unused


def test_import_alias_resolved():
    """`import torch as t; t.matmul(...)` resolves back to torch.matmul."""
    code = """
import torch as t
from torch import matmul as mm
import triton as tri

@tri.jit
def k(x, out):
    tl.store(out, tl.load(x))

def run(x, out):
    k[(1,)](x, out)
    return mm(x, x) + t.ops.aten.sum(x)
"""
    result = detect_obvious_hack(_impl(code))
    assert result.is_hack is True
    assert "torch.matmul" in result.hack_reason
    assert "torch.ops.aten.sum" in result.hack_reason


def test_fallback_branch_flagged_even_with_launch():
    """A blacklisted op in a fallback branch is flagged even if a kernel launches."""
    code = """
import torch
import triton
import triton.language as tl

@triton.jit
def k(x, out):
    tl.store(out, tl.load(x))

def run(equation, x, out):
    if equation == "i->i":
        k[(1,)](x, out)
        return out
    return torch.einsum(equation, x)
"""
    result = detect_obvious_hack(_impl(code))
    assert result.is_hack is True
    assert result.hack_reason.endswith("torch.einsum")


def test_comments_and_strings_do_not_trigger():
    code = """
import triton
import triton.language as tl

@triton.jit
def k(x, out):
    tl.store(out, tl.load(x))

def run(x, out):
    note = "torch.matmul(x, x)"
    # torch.einsum("i->i", x)
    k[(1,)](x, out)
    return out
"""
    result = detect_obvious_hack(_impl(code))
    assert result.is_hack is False, f"string/comment wrongly flagged: {result.hack_reason}"


def test_python_language_is_exempt():
    """The triton purity policy does not apply to language='python'."""
    result = detect_obvious_hack(
        _impl("import torch\ndef run(x): return torch.matmul(x, x)", language="python")
    )
    assert result.is_hack is False


def test_non_python_source_files_ignored():
    """Non-.py sources are skipped by the AST scan."""
    impl = Implementation(
        name="c",
        definition="op",
        language="triton",
        entrypoint="main.py::run",
        sources=[
            SourceFile(path="main.py", content=PURE_TRITON),
            SourceFile(path="notes.txt", content="torch.matmul(x, x)"),
        ],
    )
    result = detect_obvious_hack(impl)
    assert result.is_hack is False, f"non-py file wrongly scanned: {result.hack_reason}"


# ==========================================================================
# Group 2: MIXED cheating -- now COVERED by the whitelist scanner
#
# These were the blacklist policy's blind spot: a decoy kernel is launched (so
# the kernel/launch structural checks pass) while the real compute runs through
# a torch op the old blacklist never listed. With the whitelist scanner wired
# in, the non-allowed torch op is flagged regardless of the decoy launch.
# ==========================================================================

def _launched_fake_kernel_using(op_expr: str) -> str:
    return f"""
import torch
import triton
import triton.language as tl

@triton.jit
def _decoy(x_ptr, out_ptr):
    tl.store(out_ptr, tl.load(x_ptr))

def run(x):
    out = torch.empty_like(x)
    _decoy[(1,)](x, out)     # decoy launch to satisfy the launch check
    return {op_expr}          # real compute via a non-whitelisted torch op
"""


def test_decoy_kernel_plus_softmax_flagged():
    result = detect_obvious_hack(_impl(_launched_fake_kernel_using("torch.softmax(x, dim=-1)")))
    assert result.is_hack is True
    assert "torch.softmax" in result.hack_reason
    # decoy satisfied the structural checks; the flag comes from the whitelist
    assert "no @triton.jit" not in result.hack_reason
    assert "no Triton kernel launch" not in result.hack_reason


def test_decoy_kernel_plus_cumsum_flagged():
    result = detect_obvious_hack(_impl(_launched_fake_kernel_using("torch.cumsum(x, dim=-1)")))
    assert result.is_hack is True
    assert "torch.cumsum" in result.hack_reason


def test_getattr_obfuscation_with_launch_flagged():
    code = """
import torch
import triton
import triton.language as tl

@triton.jit
def _decoy(x_ptr, out_ptr):
    tl.store(out_ptr, tl.load(x_ptr))

def run(x):
    out = torch.empty_like(x)
    _decoy[(1,)](x, out)
    op = getattr(torch, "matmul")
    return op(x, x)
"""
    result = detect_obvious_hack(_impl(code))
    assert result.is_hack is True
    assert "torch.matmul" in result.hack_reason

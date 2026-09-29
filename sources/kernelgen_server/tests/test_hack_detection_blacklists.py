"""Anti-hack coverage for the five torch namespaces, across call spellings.

Run with the repo helper (handles the hyphenated-directory import quirk)::

    ./run_tests.sh tests/test_hack_detection_blacklists.py -v

The blacklist YAMLs under ``kernelgen_server/evaluation/blacklists/`` enumerate
forbidden compute calls in five namespaces:

    torch.*            (torch.yaml)
    torch.Tensor       (tensor_methods.yaml)   <- the x.softmax() method form
    torch.nn.functional(nn_functional.yaml)
    torch.linalg       (linalg.yaml)
    torch.fft          (fft.yaml)

Each namespace is exercised in the spellings a submission could realistically
use to reach the op:

    1. direct:      torch.softmax(x)
    2. module alias:  import torch as t;  t.softmax(x)
    3. from-import:   from torch import softmax;  softmax(x)
    4. method form:   x.softmax()            (Tensor namespace only)

Two groups:

* Group 1 -- spellings the current detector HANDLES (function-style calls whose
  receiver statically resolves to torch). Must PASS.
* Group 2 -- the method-form gap (``x.softmax()``): the receiver is a runtime
  variable, so a static ``torch.``-prefix scan cannot see it. Recorded as
  ``xfail(strict=True)``. When the Tensor-method blacklist (tensor_methods.yaml)
  is wired in, these flip to XPASS and fail -- the signal to drop the marker.
"""

import pytest

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


def _detect(source: str):
    return detect_obvious_hack(_impl(source))


# ==========================================================================
# Group 1a: torch.*  (torch.yaml) -- direct / alias / from-import
# ==========================================================================

def test_torch_softmax_direct():
    result = _detect("import torch\ndef run(x): return torch.softmax(x, dim=-1)")
    assert result.is_hack is True
    assert "torch.softmax" in result.hack_reason


def test_torch_softmax_module_alias():
    """`import torch as t; t.softmax(x)` resolves back to torch.softmax."""
    result = _detect("import torch as t\ndef run(x): return t.softmax(x, dim=-1)")
    assert result.is_hack is True
    assert "torch.softmax" in result.hack_reason


def test_torch_softmax_from_import():
    """`from torch import softmax; softmax(x)` resolves back to torch.softmax."""
    result = _detect("from torch import softmax\ndef run(x): return softmax(x, dim=-1)")
    assert result.is_hack is True
    assert "torch.softmax" in result.hack_reason


def test_torch_from_import_aliased():
    """`from torch import cumsum as cs; cs(x)` resolves back to torch.cumsum."""
    result = _detect("from torch import cumsum as cs\ndef run(x): return cs(x, dim=-1)")
    assert result.is_hack is True
    assert "torch.cumsum" in result.hack_reason


# ==========================================================================
# Group 1b: torch.nn.functional  (nn_functional.yaml)
# ==========================================================================

def test_functional_relu_alias():
    """`import torch.nn.functional as F; F.relu(x)`."""
    result = _detect("import torch.nn.functional as F\ndef run(x): return F.relu(x)")
    assert result.is_hack is True
    assert "torch.nn.functional.relu" in result.hack_reason


def test_functional_relu_from_import():
    """`from torch.nn.functional import relu; relu(x)`."""
    result = _detect("from torch.nn.functional import relu\ndef run(x): return relu(x)")
    assert result.is_hack is True
    assert "torch.nn.functional.relu" in result.hack_reason


def test_functional_softmax_from_import_aliased():
    result = _detect(
        "from torch.nn.functional import softmax as smax\ndef run(x): return smax(x, dim=-1)"
    )
    assert result.is_hack is True
    assert "torch.nn.functional.softmax" in result.hack_reason


# ==========================================================================
# Group 1c: torch.linalg  (linalg.yaml)
# ==========================================================================

def test_linalg_svd_direct():
    result = _detect("import torch\ndef run(x): return torch.linalg.svd(x)")
    assert result.is_hack is True
    assert "torch.linalg.svd" in result.hack_reason


def test_linalg_svd_from_import():
    result = _detect("from torch.linalg import svd\ndef run(x): return svd(x)")
    assert result.is_hack is True
    assert "torch.linalg.svd" in result.hack_reason


def test_linalg_alias():
    """`import torch.linalg as la; la.inv(x)`."""
    result = _detect("import torch.linalg as la\ndef run(x): return la.inv(x)")
    assert result.is_hack is True
    assert "torch.linalg.inv" in result.hack_reason


# ==========================================================================
# Group 1d: torch.fft  (fft.yaml)
# ==========================================================================

def test_fft_direct():
    result = _detect("import torch\ndef run(x): return torch.fft.fft(x)")
    assert result.is_hack is True
    assert "torch.fft.fft" in result.hack_reason


def test_fft_from_import():
    result = _detect("from torch.fft import rfft\ndef run(x): return rfft(x)")
    assert result.is_hack is True
    assert "torch.fft.rfft" in result.hack_reason


# ==========================================================================
# Group 1e: allocation / dtype helpers must NOT be flagged (allow-set)
# ==========================================================================

def test_allocation_helpers_not_flagged():
    """torch.empty / zeros_like / dtype are legitimate and excluded from the blacklist."""
    code = """
import torch
import triton
import triton.language as tl

@triton.jit
def _k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

def run(x):
    out = torch.empty_like(x)
    scratch = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    n = x.numel()
    _k[(1,)](x, out, n, BLOCK=1024)
    return out
"""
    result = _detect(code)
    assert result.is_hack is False, f"allocation helper wrongly flagged: {result.hack_reason}"


# ==========================================================================
# Group 2: torch.Tensor method form (tensor_methods.yaml) -- KNOWN GAP
#
# `x.softmax()` has a runtime-variable receiver, invisible to a torch.-prefix
# static scan. Each sample embeds a REAL @triton.jit kernel that is launched,
# so the "must have/launch a kernel" structural rules are satisfied and cannot
# mask the result -- the flag (if any) must come from recognizing the method
# call itself. This mirrors the "decoy kernel + method-form real compute" cheat.
#
# Recorded as xfail(strict=True): wiring in tensor_methods.yaml will flip these
# to XPASS and fail, prompting removal of the marker.
# ==========================================================================

def _launched_kernel_returning(expr: str) -> str:
    return f"""
import torch
import triton
import triton.language as tl

@triton.jit
def _decoy(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

def run(x):
    out = torch.empty_like(x)
    n = x.numel()
    _decoy[(1,)](x, out, n, BLOCK=1024)   # decoy launch satisfies structural checks
    return {expr}                          # real compute via a Tensor method
"""


def test_tensor_method_softmax():
    result = _detect(_launched_kernel_returning("x.softmax(dim=-1)"))
    assert result.is_hack is True


def test_tensor_method_matmul():
    result = _detect(_launched_kernel_returning("x.matmul(x)"))
    assert result.is_hack is True


def test_tensor_method_chained():
    result = _detect(_launched_kernel_returning("x.relu().softmax(dim=-1)"))
    assert result.is_hack is True


def test_tensor_method_via_intermediate():
    result = _detect(_launched_kernel_returning("(x + 1).cumsum(dim=-1)"))
    assert result.is_hack is True

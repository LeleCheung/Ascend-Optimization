from kernelgen_server.evaluation.hack_detection import detect_obvious_hack
from kernelgen_server.schema import Implementation, SourceFile


def _implementation(source: str, *, language: str = "triton") -> Implementation:
    return Implementation(
        name="candidate",
        definition="op",
        language=language,
        entrypoint="main.py::run",
        sources=[SourceFile(path="main.py", content=source)],
    )


def test_valid_triton_wrapper_is_not_marked_as_hack():
    result = detect_obvious_hack(
        _implementation(
            """
import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(x, out, n: tl.constexpr):
    offset = tl.program_id(0)
    tl.store(out + offset, tl.load(x + offset))

def run(x):
    out = x.clone()
    copy_kernel[(x.numel(),)](x, out, x.numel())
    return out.to(x.dtype)
"""
        )
    )

    assert result.is_hack is False
    assert result.hack_reason == ""


def test_torch_ops_fallback_and_missing_triton_are_marked():
    result = detect_obvious_hack(
        _implementation(
            """
import torch

def run(x):
    return torch.ops.aten.nonzero_numpy(x)
"""
        )
    )

    assert result.is_hack is True
    assert "torch.ops.aten.nonzero_numpy" in result.hack_reason
    assert "no @triton.jit kernel found" in result.hack_reason
    assert "no Triton kernel launch found" in result.hack_reason


def test_fallback_branch_is_marked_even_when_triton_is_launched():
    result = detect_obvious_hack(
        _implementation(
            """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x, out):
    tl.store(out, tl.load(x))

def run(equation, x, out):
    if equation == "i->i":
        kernel[(1,)](x, out)
        return out
    return torch.einsum(equation, x)
"""
        )
    )

    assert result.is_hack is True
    assert result.hack_reason.endswith("torch.einsum")
    assert "no @triton.jit" not in result.hack_reason
    assert "no Triton kernel launch" not in result.hack_reason


def test_torch_import_aliases_are_resolved():
    result = detect_obvious_hack(
        _implementation(
            """
import torch as t
from torch import matmul as framework_matmul
import triton as tri

@tri.jit
def kernel(x, out):
    pass

def run(x, y, out):
    kernel[(1,)](x, out)
    return framework_matmul(x, y) + t.ops.aten.sum(x)
"""
        )
    )

    assert result.is_hack is True
    assert "torch.matmul" in result.hack_reason
    assert "torch.ops.aten.sum" in result.hack_reason


def test_comments_and_strings_do_not_trigger_detection():
    result = detect_obvious_hack(
        _implementation(
            """
import triton

@triton.jit
def kernel(x, out):
    pass

def run(x, out):
    note = "torch.ops.aten.nonzero_numpy(x)"
    # torch.einsum("i->i", x)
    kernel[(1,)](x, out)
    return out
"""
        )
    )

    assert result.is_hack is False


def test_python_implementation_is_outside_triton_hack_policy():
    result = detect_obvious_hack(
        _implementation(
            "import torch\ndef run(x): return torch.ops.aten.sum(x)",
            language="python",
        )
    )

    assert result.is_hack is False

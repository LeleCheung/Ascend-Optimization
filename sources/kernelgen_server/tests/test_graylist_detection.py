"""Tests for graylist_detection.py -- legitimate plumbing usage detection.

Run with::

    python -m pytest tests/test_graylist_detection.py -v
"""

import pytest

from kernelgen_server.evaluation.graylist_detection import (
    detect_graylist_usage,
    _gray_full_names,
    _gray_tensor_methods,
)
from kernelgen_server.protocol.schema import Implementation, SourceFile, Language


def _impl(source: str, *, language: str = "triton", path: str = "main.py") -> Implementation:
    return Implementation(
        name="candidate",
        definition="op",
        language=language,
        entrypoint=f"{path}::run",
        sources=[SourceFile(path=path, content=source)],
    )


def test_graylist_loads_successfully():
    """Sanity check: graylists load and contain expected entries."""
    full = _gray_full_names()
    methods = _gray_tensor_methods()
    assert len(full) > 0, "gray_full_names should not be empty"
    assert len(methods) > 0, "gray_tensor_methods should not be empty"
    # View/layout ops should be present (NOT tensor creation like empty/zeros)
    assert "torch.reshape" in full
    assert "torch.transpose" in full
    assert "torch.clone" in full
    assert "reshape" in methods
    assert "contiguous" in methods


def test_non_triton_language_returns_empty():
    """Graylist detection only applies to language='triton'."""
    result = detect_graylist_usage(_impl("def run(): pass", language="python"))
    assert not result.has_hits()
    assert result.summary() == ""


def test_pure_triton_with_no_torch_plumbing():
    """A kernel that never calls torch plumbing has no graylist hits."""
    code = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x, y, out, n: tl.constexpr):
    idx = tl.program_id(0) * tl.constexpr(128) + tl.arange(0, 128)
    mask = idx < n
    tl.store(out + idx, tl.load(x + idx, mask=mask) + tl.load(y + idx, mask=mask), mask=mask)

def run(x, y):
    out_ptr = 0  # assume pre-allocated
    add_kernel[(1,)](x, y, out_ptr, x.numel())
    return out_ptr
"""
    result = detect_graylist_usage(_impl(code))
    # .numel() is a Tensor method in the graylist
    assert result.has_hits()
    assert len(result.gray_methods) == 1
    assert result.gray_methods[0][2] == "numel"


def test_torch_clone_is_graylisted():
    """torch.clone is legitimate view/copy operation."""
    code = """
import torch
import triton

@triton.jit
def dummy(): pass

def run(x):
    out = torch.clone(x)
    dummy[(1,)]()
    return out
"""
    result = detect_graylist_usage(_impl(code))
    assert result.has_hits()
    assert len(result.gray_calls) == 1
    path, line, name = result.gray_calls[0]
    assert name == "torch.clone"
    assert "graylist Torch API" in result.summary()
    assert "torch.clone" in result.summary()


def test_tensor_reshape_method_is_graylisted():
    """x.reshape(...) is legitimate layout plumbing."""
    code = """
import torch
import triton

@triton.jit
def dummy(): pass

def run(x):
    y = x.reshape(-1)
    dummy[(1,)]()
    return y
"""
    result = detect_graylist_usage(_impl(code))
    assert result.has_hits()
    assert len(result.gray_methods) == 1
    path, line, name = result.gray_methods[0]
    assert name == "reshape"
    assert "graylist Tensor method" in result.summary()
    assert ".reshape" in result.summary()


def test_multiple_graylist_hits_are_all_recorded():
    """Multiple distinct graylist calls are all collected."""
    code = """
import torch
import triton

@triton.jit
def k(): pass

def run(x):
    a = torch.clone(x)
    b = torch.transpose(x, 0, 1)
    c = x.reshape(-1).contiguous()
    k[(1,)]()
    return a, b, c
"""
    result = detect_graylist_usage(_impl(code))
    assert result.has_hits()
    # torch.clone, torch.transpose
    assert len(result.gray_calls) == 2
    call_names = {name for _, _, name in result.gray_calls}
    assert "torch.clone" in call_names
    assert "torch.transpose" in call_names
    # .reshape, .contiguous
    assert len(result.gray_methods) == 2
    method_names = {name for _, _, name in result.gray_methods}
    assert "reshape" in method_names
    assert "contiguous" in method_names


def test_torch_alias_is_resolved():
    """import torch as tr; tr.clone(...) canonicalises to torch.clone."""
    code = """
import torch as tr
import triton

@triton.jit
def k(): pass

def run(x):
    y = tr.clone(x)
    k[(1,)]()
    return y
"""
    result = detect_graylist_usage(_impl(code))
    assert result.has_hits()
    assert len(result.gray_calls) == 1
    assert result.gray_calls[0][2] == "torch.clone"


def test_from_import_is_resolved():
    """from torch import transpose as t; t(...) canonicalises to torch.transpose."""
    code = """
from torch import transpose as t
import triton

@triton.jit
def k(): pass

def run(x):
    y = t(x, 0, 1)
    k[(1,)]()
    return y
"""
    result = detect_graylist_usage(_impl(code))
    assert result.has_hits()
    assert len(result.gray_calls) == 1
    assert result.gray_calls[0][2] == "torch.transpose"


def test_namespace_method_not_double_counted():
    """torch.reshape(x, ...) is a namespace call, not also a method call."""
    code = """
import torch
import triton

@triton.jit
def k(): pass

def run(x):
    y = torch.reshape(x, (-1,))
    k[(1,)]()
    return y
"""
    result = detect_graylist_usage(_impl(code))
    assert result.has_hits()
    # Should be 1 namespace call, not also 1 method
    assert len(result.gray_calls) == 1
    assert result.gray_calls[0][2] == "torch.reshape"
    assert len(result.gray_methods) == 0


def test_triton_namespace_method_not_flagged():
    """tl.load(...) / tl.store(...) are Triton namespace calls, not Tensor methods."""
    code = """
import triton
import triton.language as tl

@triton.jit
def k(x, out):
    tl.store(out, tl.load(x))

def run(x, out):
    k[(1,)](x, out)
"""
    result = detect_graylist_usage(_impl(code))
    # tl.load / tl.store should NOT be flagged as Tensor methods
    assert len(result.gray_methods) == 0


def test_f_pad_is_graylisted():
    """F.pad is in nn.functional graylist."""
    code = """
import torch.nn.functional as F
import triton

@triton.jit
def k(): pass

def run(x):
    y = F.pad(x, (1, 1))
    k[(1,)]()
    return y
"""
    result = detect_graylist_usage(_impl(code))
    assert result.has_hits()
    assert len(result.gray_calls) == 1
    assert result.gray_calls[0][2] == "torch.nn.functional.pad"


def test_linalg_diagonal_is_graylisted():
    """torch.linalg.diagonal is the only graylist entry in linalg."""
    code = """
import torch.linalg
import triton

@triton.jit
def k(): pass

def run(x):
    d = torch.linalg.diagonal(x)
    k[(1,)]()
    return d
"""
    result = detect_graylist_usage(_impl(code))
    assert result.has_hits()
    assert len(result.gray_calls) == 1
    assert result.gray_calls[0][2] == "torch.linalg.diagonal"


def test_syntax_error_is_silently_skipped():
    """A source with a syntax error does not crash detection."""
    code = """
import torch
def run(x)
    return torch.empty_like(x  # missing close paren
"""
    result = detect_graylist_usage(_impl(code))
    # Should not raise, should return empty
    assert not result.has_hits()


def test_non_py_source_is_skipped():
    """Non-.py sources (e.g. .cu) are not parsed."""
    impl = Implementation(
        name="c",
        definition="op",
        language="triton",
        entrypoint="main.py::run",
        sources=[
            SourceFile(path="main.py", content="def run(): pass"),
            SourceFile(path="kernel.cu", content="__global__ void k() {}"),
        ],
    )
    result = detect_graylist_usage(impl)
    # Only main.py is parsed; kernel.cu is skipped
    assert not result.has_hits()

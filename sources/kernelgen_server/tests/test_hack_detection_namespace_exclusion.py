"""Tests for namespace exclusion in hack_detection.py.

Verifies that tl.*, libdevice.*, and math.* calls are not flagged as Tensor
method fallbacks.

Run with::

    python -m pytest tests/test_hack_detection_namespace_exclusion.py -v
"""

import pytest

from kernelgen_server.evaluation.hack_detection import detect_obvious_hack
from kernelgen_server.schema import Implementation, SourceFile


def _impl(source: str) -> Implementation:
    return Implementation(
        name="candidate",
        definition="op",
        language="triton",
        entrypoint="main.py::run",
        sources=[SourceFile(path="main.py", content=source)],
    )


def test_tl_math_methods_not_flagged_as_tensor_methods():
    """tl.math.tanh, tl.math.exp should not match Tensor.tanh/exp."""
    code = """
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, out_ptr, n: tl.constexpr):
    idx = tl.program_id(0)
    x = tl.load(x_ptr + idx)
    # These are tl.math namespace calls, NOT Tensor methods
    y = tl.math.tanh(x)
    z = tl.math.exp(y)
    tl.store(out_ptr + idx, z)

def run(x, out):
    kernel[(x.numel(),)](x, out, x.numel())
"""
    result = detect_obvious_hack(_impl(code))
    assert not result.is_hack
    assert "tanh" not in result.hack_reason
    assert "exp" not in result.hack_reason


def test_libdevice_methods_not_flagged():
    """libdevice.tanh, libdevice.exp are Triton namespace calls."""
    code = """
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def kernel(x_ptr, out_ptr):
    x = tl.load(x_ptr)
    # libdevice namespace
    y = libdevice.tanh(x)
    z = libdevice.exp(y)
    tl.store(out_ptr, z)

def run(x, out):
    kernel[(1,)](x, out)
"""
    result = detect_obvious_hack(_impl(code))
    assert not result.is_hack
    assert "tanh" not in result.hack_reason
    assert "exp" not in result.hack_reason


def test_python_math_module_not_flagged():
    """math.log, math.sqrt are stdlib, not Tensor methods."""
    code = """
import math
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, out_ptr, scale: tl.constexpr):
    x = tl.load(x_ptr)
    tl.store(out_ptr, x * scale)

def run(x, out):
    # math module usage in host code
    scale = math.sqrt(2.0) * math.log(10.0)
    kernel[(1,)](x, out, scale)
"""
    result = detect_obvious_hack(_impl(code))
    assert not result.is_hack
    # 'log' and 'sqrt' should not appear as forbidden method hits
    assert "log" not in result.hack_reason
    assert "sqrt" not in result.hack_reason


def test_tl_load_store_not_flagged():
    """tl.load, tl.store are Triton primitives."""
    code = """
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, out_ptr):
    x = tl.load(x_ptr)
    tl.store(out_ptr, x)

def run(x, out):
    kernel[(1,)](x, out)
"""
    result = detect_obvious_hack(_impl(code))
    assert not result.is_hack
    assert "load" not in result.hack_reason
    assert "store" not in result.hack_reason


def test_tensor_method_still_flagged_when_not_namespace():
    """x.sum() where x is a variable SHOULD be flagged."""
    code = """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, out_ptr):
    tl.store(out_ptr, tl.load(x_ptr))

def run(x):
    out = torch.empty_like(x)
    kernel[(1,)](x, out)
    # This is a genuine Tensor method on a runtime value
    return out.sum()
"""
    result = detect_obvious_hack(_impl(code))
    assert result.is_hack
    assert "sum" in result.hack_reason


def test_chained_namespace_call_not_flagged():
    """tl.math.abs(tl.math.log(...)) should not flag .abs or .log."""
    code = """
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, out_ptr):
    x = tl.load(x_ptr)
    y = tl.math.abs(tl.math.log(x))
    tl.store(out_ptr, y)

def run(x, out):
    kernel[(1,)](x, out)
"""
    result = detect_obvious_hack(_impl(code))
    assert not result.is_hack
    assert "abs" not in result.hack_reason
    assert "log" not in result.hack_reason

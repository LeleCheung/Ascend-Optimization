"""Test graylist persistence to disk."""

import json
import tempfile
from pathlib import Path

import pytest

from kernelgen_server.evaluation.hack_detection import log_graylist_hits, _GRAYLIST_LOG_DIR
from kernelgen_server.protocol.schema import Implementation, SourceFile


def _impl(source: str, *, name: str = "test_op", vendor: str = "test_vendor") -> Implementation:
    return Implementation(
        name=name,
        definition="op",
        language="triton",
        entrypoint="main.py::run",
        sources=[SourceFile(path="main.py", content=source)],
    )


def test_graylist_persistence_creates_json_file():
    """log_graylist_hits with persist_dir creates a JSON file."""
    code = """
import torch
import triton

@triton.jit
def k(): pass

def run(x):
    y = torch.clone(x)
    z = y.reshape(-1).contiguous()
    k[(1,)]()
    return z
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        impl = _impl(code, name="test_op", vendor="test_vendor")
        log_graylist_hits(impl, vendor="test_vendor", persist_dir=tmpdir)

        # Should create test_vendor_test_op.json
        target = Path(tmpdir) / "test_vendor_test_op.json"
        assert target.exists(), f"Expected {target} to be created"

        record = json.loads(target.read_text(encoding="utf-8"))
        assert record["operator"] == "test_op"
        assert record["vendor"] == "test_vendor"
        assert record["language"] == "triton"
        assert record["hit_count"] == 3  # clone, reshape, contiguous
        assert len(record["gray_calls"]) == 1  # torch.clone
        assert len(record["gray_methods"]) == 2  # .reshape, .contiguous


def test_graylist_persistence_handles_slashed_name():
    """log_graylist_hits extracts operator name from 'vendor/op' format."""
    code = """
import torch
import triton

@triton.jit
def k(): pass

def run(x):
    return torch.clone(x)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        impl = _impl(code, name="muxi/conj", vendor="muxi")
        log_graylist_hits(impl, vendor="muxi", persist_dir=tmpdir)

        # Should create muxi_conj.json (not muxi_muxi/conj.json)
        target = Path(tmpdir) / "muxi_conj.json"
        assert target.exists(), f"Expected {target} to be created"

        record = json.loads(target.read_text(encoding="utf-8"))
        assert record["operator"] == "conj"
        assert record["vendor"] == "muxi"


def test_graylist_persistence_clean_implementation():
    """Clean implementation creates no file."""
    code = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x, y, out, n: tl.constexpr):
    idx = tl.program_id(0) * 128 + tl.arange(0, 128)
    mask = idx < n
    tl.store(out + idx, tl.load(x + idx, mask=mask) + tl.load(y + idx, mask=mask), mask=mask)

def run(x, y, out):
    add_kernel[(1,)](x, y, out, 1024)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        impl = _impl(code, name="clean_op", vendor="test")
        log_graylist_hits(impl, vendor="test", persist_dir=tmpdir)

        # Should NOT create any file
        target = Path(tmpdir) / "test_clean_op.json"
        assert not target.exists(), f"Expected no file for clean implementation"


def test_graylist_persistence_without_persist_dir():
    """log_graylist_hits without persist_dir only logs, no file."""
    code = """
import torch
import triton

@triton.jit
def k(): pass

def run(x):
    return torch.clone(x)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        impl = _impl(code, name="test_op", vendor="test")
        # Call without persist_dir
        log_graylist_hits(impl, vendor="test")

        # Should NOT create any file in tmpdir
        files = list(Path(tmpdir).glob("*.json"))
        assert len(files) == 0, f"Expected no files, found {files}"


def test_graylist_persistence_default_path():
    """log_graylist_hits uses graylists/log/ by default."""
    code = """
import torch
import triton

@triton.jit
def k(): pass

def run(x):
    return x.reshape(-1)
"""
    impl = _impl(code, name="test_default", vendor="test")
    # Call with default persist_dir (should use _GRAYLIST_LOG_DIR)
    log_graylist_hits(impl, vendor="test")

    # Should create file in graylists/log/
    target = _GRAYLIST_LOG_DIR / "test_test_default.json"
    assert target.exists(), f"Expected {target} to be created in default location"

    # Clean up
    target.unlink()


def test_to_method_excluded_from_graylist():
    """.to() method is excluded from graylist (it's a conversion, not a view)."""
    code = """
import torch
import triton

@triton.jit
def k(): pass

def run(x):
    # .to() returns a COPY when dtype/device differs, not a view
    y = x.to(torch.float32)
    z = x.to(device='cuda')
    k[(1,)]()
    return y + z
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        impl = _impl(code, name="test_to_excluded", vendor="test")
        log_graylist_hits(impl, vendor="test", persist_dir=tmpdir)

        # Should NOT create any file (no graylist hits)
        target = Path(tmpdir) / "test_test_to_excluded.json"
        assert not target.exists(), f"Expected no file since .to() is excluded from graylist"


def test_graylist_persistence_explicit_none():
    """log_graylist_hits with persist_dir=None disables persistence."""
    code = """
import torch
import triton

@triton.jit
def k(): pass

def run(x):
    return x.reshape(-1)
"""
    impl = _impl(code, name="test_no_persist", vendor="test")
    # Explicitly disable persistence
    log_graylist_hits(impl, vendor="test", persist_dir=None)

    # Should NOT create file in default location
    target = _GRAYLIST_LOG_DIR / "test_test_no_persist.json"
    assert not target.exists(), f"Expected no file when persist_dir=None"


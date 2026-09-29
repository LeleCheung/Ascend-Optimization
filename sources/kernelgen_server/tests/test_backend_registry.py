"""Backend declarations remain importable and have one authoritative registry."""

import ast
from pathlib import Path

import pytest

from kernelgen_server.runtime import backend


def test_device_module_parses_without_accelerator_dependencies():
    path = Path(backend.__file__).with_name("device.py")
    ast.parse(path.read_text(encoding="utf-8"))
    assert {"enflame", "txda", "npu"} <= backend.KNOWN_BACKENDS
    assert backend.runtime_device_type("txda") == "txda"


def test_device_reuses_shared_backend_registry():
    pytest.importorskip("torch")
    from kernelgen_server.runtime import device

    assert device.KNOWN_BACKENDS is backend.KNOWN_BACKENDS
    assert device._make_device("txda").backend == "txda"

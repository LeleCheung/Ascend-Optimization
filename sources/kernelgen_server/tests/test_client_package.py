"""The local Agent and target Server must use one set of wire classes."""

import os
from pathlib import Path
import subprocess
import sys
import tomllib

import kernelgen_client
import kernelgen_server
from kernelgen_client import http
from kernelgen_client.debug.models import DebugJobRequest
from kernelgen_client.operator_bundles import pack_operator_bundle
from kernelgen_client.profiling.models import ProfileRequest
from kernelgen_server.debug.jobs import DebugJobRequest as ServerDebugJobRequest
from kernelgen_server.operator_bundles import pack_operator_bundle as server_pack_operator_bundle
from kernelgen_server.profiling.models import ProfileRequest as ServerProfileRequest
from kernelgen_server.protocol import client as server_client


def test_server_reuses_client_contract_classes_and_bundle_packer():
    metadata = tomllib.loads((Path(__file__).resolve().parents[1] / "client/pyproject.toml").read_text())
    assert metadata["project"]["version"] == kernelgen_client.__version__
    assert kernelgen_server.Definition is kernelgen_client.Definition
    assert ServerDebugJobRequest is DebugJobRequest
    assert ServerProfileRequest is ProfileRequest
    assert server_pack_operator_bundle is pack_operator_bundle
    assert server_client is http


def test_client_import_does_not_load_server_execution_modules():
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
import kernelgen_client
import kernelgen_client.http
import kernelgen_client.catalog
import kernelgen_client.flaggems_discovery
assert not any(name.startswith('kernelgen_server') for name in sys.modules)
"""
    environment = {**os.environ, "PYTHONPATH": str(root / "client"), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-c", script], env=environment, cwd=root / "client", check=True)

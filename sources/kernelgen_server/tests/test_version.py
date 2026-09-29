from __future__ import annotations

import json
import tomllib
from pathlib import Path

import kernelgen_server
from kernelgen_server import Definition, EvaluateRequest
from kernelgen_server.debug.jobs import DebugJob
from kernelgen_server.profiling.models import ProfileResult
from kernelgen_server.protocol.version import (
    KERNELGEN_API_VERSION,
    KERNELGEN_SERVER_VERSION,
    KERNELGEN_SUPPORTED_API_VERSIONS,
)


def test_public_version_fields_are_v6():
    assert KERNELGEN_SERVER_VERSION == "v6.5.0"
    assert KERNELGEN_API_VERSION == "v6.2"
    assert KERNELGEN_SUPPORTED_API_VERSIONS == ("v6.0", "v6.2")
    assert kernelgen_server.__version__ == "v6.5.0"
    for model in (Definition, EvaluateRequest, DebugJob, ProfileResult):
        assert model.model_fields["api_version"].default == "v6.2"


def test_package_metadata_uses_equivalent_pep440_version():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert project["version"] == "6.5.0"


def test_compatibility_manifest_matches_release_contract():
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "compatibility.yaml").read_text(encoding="utf-8")
    framework_binding = {
        "framework_repository": "https://github.com/YaooXu/FlagGems.git",
        "framework_branch": "feat/new-api-for-kernelgen-server",
        "framework_revision": "d64794e63b502cb836bc015a92a62c42de4be05a",
    }

    assert "version: v6.5.0" in manifest
    assert "api_version: v6.2" in manifest
    assert "v6.0:\n    default_mode: adapter" in manifest
    assert "v6.2:\n    default_mode: native" in manifest
    assert "v6.1.2:" in manifest
    assert (
        "v6.2.0:\n    required_api_version: v6.2\n"
        "    latest_validated_kgs: v6.3.0\n    validation_scope: host"
    ) in manifest
    assert "latest_validated_kgs: v6.3.1" in manifest
    assert "repository: https://github.com/flagos-ai/FlagGems.git" in manifest
    assert "branch: kernelgen-dev" in manifest
    assert "revision_policy: branch" in manifest
    native_manifest = json.loads(
        (root / "data/flaggems-native/manifest.json").read_text(encoding="utf-8")
    )
    for key, value in framework_binding.items():
        assert native_manifest[key] == value
    adapter_manifest = json.loads(
        (root / "data/flaggems-adapter-definitions/manifest.json").read_text(encoding="utf-8")
    )
    assert not framework_binding.keys() & adapter_manifest.keys()

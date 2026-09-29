from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


def test_package_metadata_matches_release_version():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert project["version"] == "6.7.0"
    lock = yaml.safe_load((root / "deployment/kgs.lock.yaml").read_text(encoding="utf-8"))
    assert lock["kg_release"] == "v" + project["version"]
    assert lock["kgs"]["release"] == "v6.5.0"
    assert lock["validated_protocol"] == "v6.2"
    client_dependencies = [
        item for item in project["dependencies"]
        if item.startswith("kernelgen-server-client @ ")
    ]
    assert len(client_dependencies) == 1
    assert client_dependencies[0].endswith(
        f"@{lock['kgs']['commit']}#subdirectory=client"
    )

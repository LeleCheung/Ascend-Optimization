"""Tests for Server-owned Catalog names and Solution benchmark identity."""

from __future__ import annotations

import json

import pytest

from kernelgen.data import catalog as catalog_module


def _write_manifest(root, *, name="flaggems-v5", api_version="v5.1"):
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"name": name, "api_version": api_version}),
        encoding="utf-8",
    )


def test_catalog_benchmark_id_uses_manifest_name_and_api_version(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "flaggems-v5"
    _write_manifest(root)
    monkeypatch.setattr(
        catalog_module,
        "builtin_catalog_path",
        lambda name: root,
    )

    assert (
        catalog_module.catalog_benchmark_id("flaggems-v5")
        == "flaggems-v5-v5.1"
    )


def test_catalog_name_must_match_manifest(tmp_path, monkeypatch):
    root = tmp_path / "flaggems-v5"
    _write_manifest(root, name="different")
    monkeypatch.setattr(
        catalog_module,
        "builtin_catalog_path",
        lambda name: root,
    )

    with pytest.raises(ValueError, match="does not match catalog_name"):
        catalog_module.resolve_builtin_catalog_path("flaggems-v5")


def test_catalog_benchmark_id_requires_api_version(tmp_path, monkeypatch):
    root = tmp_path / "flaggems-v5"
    _write_manifest(root, api_version="")
    monkeypatch.setattr(
        catalog_module,
        "builtin_catalog_path",
        lambda name: root,
    )

    with pytest.raises(ValueError, match="api_version is required"):
        catalog_module.catalog_benchmark_id("flaggems-v5")

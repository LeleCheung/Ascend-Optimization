"""Canonical KernelGen Server Catalog identity and loading helpers."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Any

from kernelgen.data.constants import DEFAULT_CATALOG_NAME


def builtin_catalog_path(name: str) -> Path:
    """Legacy local-Catalog lookup; current runs use the KGS operator contract."""

    if not name or Path(name).name != name:
        raise ValueError(f"invalid built-in catalog name: {name!r}")
    spec = importlib.util.find_spec("kernelgen_server")
    if spec is not None and spec.origin is not None:
        package_root = Path(spec.origin).resolve().parent
        for root in (package_root / "data" / name, package_root.parent / "data" / name):
            if (root / "manifest.json").is_file():
                return root
    raise FileNotFoundError(
        f"legacy built-in Catalog {name!r} requires a local KGS checkout; "
        "new runs should use the target KGS operator contract"
    )


def resolve_builtin_catalog_path(
    catalog_name: str = DEFAULT_CATALOG_NAME,
) -> Path:
    """Resolve one Server-owned built-in Catalog and validate its name."""

    name = catalog_name.strip()
    if not name:
        raise ValueError("catalog_name must not be empty")
    path = Path(builtin_catalog_path(name)).resolve()
    manifest = load_catalog_manifest(name, path=path)
    manifest_name = manifest.get("name")
    if manifest_name is not None and manifest_name != name:
        raise ValueError(
            "Catalog manifest name does not match catalog_name: "
            f"{manifest.get('name')!r} != {name!r}"
        )
    return path


def load_catalog_manifest(
    catalog_name: str = DEFAULT_CATALOG_NAME,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Load the manifest used as the stable Catalog identity source."""

    catalog_path = (
        Path(path)
        if path is not None
        else Path(builtin_catalog_path(catalog_name.strip()))
    )
    manifest_path = catalog_path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"KernelGen Server Catalog manifest not found: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Catalog manifest must be an object: {manifest_path}")
    return manifest


def catalog_benchmark_id(
    catalog_name: str = DEFAULT_CATALOG_NAME,
) -> str:
    """Return ``<manifest name>-<API version>`` for Solution Registry slots."""

    path = resolve_builtin_catalog_path(catalog_name)
    manifest = load_catalog_manifest(catalog_name, path=path)
    manifest_name = str(manifest.get("name") or catalog_name).strip()
    api_version = str(manifest.get("api_version") or "").strip()
    if not api_version:
        raise ValueError(
            f"Catalog manifest api_version is required: {path / 'manifest.json'}"
        )
    return f"{manifest_name}-{api_version}"

"""Static and derived hints for the Web console: device list + operator catalog.

Device presets are UI hints, not hardware discovery. Connections and images
are maintained in ``tests/hosts.md``; actual device facts come from KGS status.
The port numbers here match the defaults used by
``scripts/proxy/start_all.sh`` (local_port = 18000 + last two digits of the
remote port), so a single ``kg-service`` on the dev machine can talk to any
vendor's KernelGen Server through the persistent SSH proxies.

The definition list is scanned from the built-in catalog on disk each call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DevicePreset:
    name: str
    target_hardware: str
    eval_server: str  # what the local kg-service should talk to
    label: str        # human-readable UI label


# Ordered for UI presentation. First entry is the sensible default.
DEVICE_PRESETS: tuple[DevicePreset, ...] = (
    DevicePreset("ascend", "Ascend910B", "http://localhost:18006",
                 "ascend (Ascend910B via :18006)"),
    DevicePreset("nvidia", "H20", "http://localhost:18008",
                 "nvidia (H20 via :18008)"),
    DevicePreset("ascend-8", "Ascend910B", "http://localhost:18016",
                 "ascend-8 (Ascend910B via :18016)"),
    DevicePreset("ascend-31", "Ascend910B", "http://localhost:18026",
                 "ascend-31 (Ascend910B via :18026)"),
    DevicePreset("ascend-32", "Ascend910B", "http://localhost:18036",
                 "ascend-32 (Ascend910B via :18036)"),
    DevicePreset("tianshu", "BI-V150", "http://localhost:18001",
                 "tianshu (BI-V150 via :18001)"),
    DevicePreset("hygon", "BW1000", "http://localhost:18002",
                 "hygon (BW1000 via :18002)"),
    DevicePreset("musa", "S5000", "http://localhost:18003",
                 "musa (S5000 via :18003)"),
    DevicePreset("metax", "C550", "http://localhost:18004",
                 "metax (C550 via :18004)"),
    DevicePreset("kunlun", "P800", "http://localhost:18005",
                 "kunlun (P800 via :18005)"),
    DevicePreset("ppu", "ZW810E", "http://localhost:18007",
                 "ppu (ZW810E via :18007)"),
    DevicePreset("suiyuan", "S60", "http://localhost:18009",
                 "suiyuan (S60 via :18009)"),
)


def list_devices() -> list[dict]:
    """Return device presets as plain dicts, ready for JSON serialization."""
    return [
        {
            "name": preset.name,
            "target_hardware": preset.target_hardware,
            "eval_server": preset.eval_server,
            "label": preset.label,
        }
        for preset in DEVICE_PRESETS
    ]


def list_definitions(catalog_name: str = "") -> dict:
    """Return operator names in one built-in catalog.

    Returns ``{catalog, path, definitions}``. ``definitions`` is empty if the
    catalog cannot be resolved, so the UI can degrade gracefully.
    """
    from kernelgen.data.catalog import (
        load_catalog_manifest,
        resolve_builtin_catalog_path,
    )
    from kernelgen.data.constants import DEFAULT_CATALOG_NAME

    name = catalog_name.strip() or DEFAULT_CATALOG_NAME
    try:
        catalog_dir = resolve_builtin_catalog_path(name)
        manifest = load_catalog_manifest(name, path=catalog_dir)
    except (FileNotFoundError, OSError, ValueError):
        return {"catalog": name, "path": None, "definitions": []}

    layout = manifest.get("layout")
    try:
        if layout is None:
            definitions_dir = catalog_dir / "definitions"
            names = (
                sorted(
                    entry.stem
                    for entry in definitions_dir.iterdir()
                    if entry.is_file() and entry.suffix == ".json"
                )
                if definitions_dir.is_dir()
                else []
            )
        elif layout == "per-operator":
            operators_dir = catalog_dir / "ops"
            names = (
                sorted(
                    entry.name
                    for entry in operators_dir.iterdir()
                    if entry.is_dir() and (entry / "definition.json").is_file()
                )
                if operators_dir.is_dir()
                else []
            )
        else:
            names = []
    except OSError:
        names = []
    return {
        "catalog": name,
        "path": str(catalog_dir),
        "definitions": names,
    }

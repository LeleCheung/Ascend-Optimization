"""Compatibility imports for pure FlagGems suite discovery."""

from kernelgen_client.flaggems_discovery import (
    FlagGemsAssets,
    _discover_suite,
    _suite_marker_index,
    benchmark_fingerprint,
    discover_assets,
    flag_gems_root,
)

__all__ = ["FlagGemsAssets", "benchmark_fingerprint", "discover_assets", "flag_gems_root"]

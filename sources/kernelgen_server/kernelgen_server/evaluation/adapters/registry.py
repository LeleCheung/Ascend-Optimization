"""Resolve a trusted evaluator binding to exactly one adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from kernelgen_server.catalog import Catalog, resolve_catalog
from kernelgen_server.protocol.schema import EvaluatorBinding

from .base import EvaluatorAdapter


def _prepare_framework_import(catalog, definition: str) -> None:
    """Bind a native oracle to the catalog's validated framework checkout."""

    if catalog.framework != "flaggems":
        return
    from .flaggems.discovery import discover_assets

    assets = discover_assets(definition, catalog.framework_revision)
    source = str(assets.root / "src")
    loaded = sys.modules.get("flag_gems")
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if not loaded_path.is_relative_to(Path(source).resolve()):
            raise RuntimeError(
                "flag_gems was imported before the native catalog framework "
                f"binding from a different checkout: {loaded_path}"
            )
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)
    catalog.framework_root = assets.root
    from .flaggems.reference_compat import prepare_flaggems_reference

    prepare_flaggems_reference(definition)


def resolve_binding_catalog(
    binding: EvaluatorBinding,
    *,
    operator_bundle_root: str | Path | None = None,
) -> Catalog:
    """Resolve server-owned content without importing an oracle or framework."""
    if binding.bundle_id is not None:
        from kernelgen_server.operator_bundles import OperatorBundleStore

        if operator_bundle_root is None:
            raise ValueError("operator bundle execution requires a server-owned bundle root")
        store = OperatorBundleStore(operator_bundle_root)
        return Catalog(store.catalog_path(binding.bundle_id))
    return resolve_catalog(binding.catalog_name)


def create_adapter(
    binding: EvaluatorBinding,
    *,
    device: Any | None = None,
    device_string: str = "",
    backend: str = "",
    operator_bundle_root: str | Path | None = None,
) -> EvaluatorAdapter:
    catalog = resolve_binding_catalog(binding, operator_bundle_root=operator_bundle_root)
    _prepare_framework_import(catalog, binding.definition)
    operator = catalog.load(binding.definition)
    if catalog.evaluator == "native":
        from .native import NativeEvaluationAdapter

        adapter_type = NativeEvaluationAdapter
    elif catalog.evaluator == "flaggems":
        from .flaggems.adapter import FlagGemsEvaluationAdapter

        adapter_type = FlagGemsEvaluationAdapter
    else:  # catalog validation makes this unreachable
        raise ValueError(f"unknown evaluator: {catalog.evaluator}")
    return adapter_type(
        catalog=catalog,
        operator=operator,
        device=device,
        device_string=device_string,
        backend=backend,
    )


__all__ = ["create_adapter"]

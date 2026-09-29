"""Server-owned built-in Catalog location over shared Catalog parsing."""

from pathlib import Path

from kernelgen_client.catalog import Catalog, EvaluatorKind, FrameworkKind, OperatorData


def builtin_catalog_path(name: str = "simple-v6-test") -> Path:
    if not name or Path(name).name != name:
        raise ValueError(f"invalid built-in catalog name: {name!r}")
    roots = (
        Path(__file__).resolve().parent / "data" / name,
        Path(__file__).resolve().parents[1] / "data" / name,
    )
    for root in roots:
        if (root / "manifest.json").is_file():
            return root.resolve()
    raise FileNotFoundError(f"built-in catalog is unavailable: {roots[0]}")


def resolve_catalog(name: str) -> Catalog:
    return Catalog(builtin_catalog_path(name))


__all__ = ["Catalog", "EvaluatorKind", "FrameworkKind", "OperatorData", "builtin_catalog_path", "resolve_catalog"]

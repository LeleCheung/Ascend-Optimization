"""Setuptools hooks for bundling the active built-in catalogs."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2, copytree, rmtree

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


_PROJECT_ROOT = Path(__file__).parent
_CATALOG_SOURCE_ROOT = _PROJECT_ROOT / "data"
_PACKAGE_DATA_ROOT = _PROJECT_ROOT / "kernelgen_server" / "data"


def _ignored_catalog_artifacts(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name == "__pycache__" or Path(name).suffix in {".pyc", ".pyo"}
    }


def _active_catalogs() -> list[Path]:
    return sorted(
        path
        for path in _CATALOG_SOURCE_ROOT.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and (path / "manifest.json").is_file()
    )


class build_py(_build_py):
    """Copy repository-level catalogs into the importable wheel package."""

    def run(self) -> None:
        super().run()
        target_root = Path(self.build_lib) / "kernelgen_server" / "data"
        target_root.mkdir(parents=True, exist_ok=True)
        package_catalogs = {
            path.name
            for path in _PACKAGE_DATA_ROOT.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        }
        for target in target_root.iterdir():
            if (
                target.is_dir()
                and (target / "manifest.json").is_file()
                and target.name not in package_catalogs
            ):
                rmtree(target)
        readme = _CATALOG_SOURCE_ROOT / "README.md"
        if readme.is_file():
            copy2(readme, target_root / readme.name)
        for source in _active_catalogs():
            target = target_root / source.name
            if target.exists():
                rmtree(target)
            copytree(
                source,
                target,
                ignore=_ignored_catalog_artifacts,
            )


setup(cmdclass={"build_py": build_py})

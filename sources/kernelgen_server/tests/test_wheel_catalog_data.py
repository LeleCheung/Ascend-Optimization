from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_ROOT = PROJECT_ROOT / "data"
WHEEL_DATA_ROOT = "kernelgen_server/data"
PACKAGE_CATALOGS = ("simple-v6-test",)


def _active_catalogs() -> list[Path]:
    return sorted(
        path
        for path in SOURCE_DATA_ROOT.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and (path / "manifest.json").is_file()
    )


def _catalog_files(catalogs: list[Path]) -> set[str]:
    return {
        f"{WHEEL_DATA_ROOT}/{path.relative_to(SOURCE_DATA_ROOT).as_posix()}"
        for catalog in catalogs
        for path in catalog.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def test_wheel_bundles_every_active_catalog(tmp_path: Path):
    wheel_dir = tmp_path / "wheel"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(PROJECT_ROOT / "client"),
            str(PROJECT_ROOT),
        ],
        cwd=tmp_path,
        check=True,
    )
    wheels = list(wheel_dir.glob("kernelgen_server-*.whl"))
    assert len(wheels) == 1
    client_wheels = list(wheel_dir.glob("kernelgen_server_client-*.whl"))
    assert len(client_wheels) == 1

    catalogs = _active_catalogs()
    assert catalogs
    expected = _catalog_files(catalogs)
    with ZipFile(wheels[0]) as archive:
        packaged = set(archive.namelist())
        archive.extractall(tmp_path / "installed")
    with ZipFile(client_wheels[0]) as archive:
        client_packaged = set(archive.namelist())
        archive.extractall(tmp_path / "installed")
    assert "kernelgen_client/http.py" in client_packaged
    assert not any(name.startswith("kernelgen_server/") for name in client_packaged)

    assert expected <= packaged
    for name in PACKAGE_CATALOGS:
        assert f"{WHEEL_DATA_ROOT}/{name}/manifest.json" in packaged
    assert not any(name.startswith(f"{WHEEL_DATA_ROOT}/.old/") for name in packaged)

    installed = tmp_path / "installed"
    script = """
import sys
from pathlib import Path

installed = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installed))
import kernelgen_server
from kernelgen_server import Catalog, builtin_catalog_path

assert Path(kernelgen_server.__file__).resolve().is_relative_to(installed)
for name in sys.argv[2:]:
    assert builtin_catalog_path(name).is_dir()
catalog = Catalog(builtin_catalog_path("kernelgenbench"))
assert "kernelgenbench_square" in catalog.operator_names
adapter = Catalog(builtin_catalog_path("flaggems-adapter-definitions"))
assert adapter.framework_revision is None
assert "framework_revision" not in adapter.manifest
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(installed),
            *PACKAGE_CATALOGS,
            *(path.name for path in catalogs),
        ],
        cwd=tmp_path,
        check=True,
    )

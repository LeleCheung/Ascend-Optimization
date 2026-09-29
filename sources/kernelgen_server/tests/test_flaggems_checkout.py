import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelgen_server.evaluation.adapters.flaggems.discovery import discover_assets
from kernelgen_server import Catalog, builtin_catalog_path
from kernelgen_server.evaluation.adapters.flaggems.adapter import FlagGemsEvaluationAdapter
from kernelgen_server.profiling import ProfileOptions


def _git(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout.strip()


def _checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "FlagGems"
    (root / "src/flag_gems").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "benchmark").mkdir()
    (root / "src/flag_gems/__init__.py").write_text("", encoding="utf-8")
    (root / "tests/test_addmm_.py").write_text(
        "import pytest\n\n@pytest.mark.addmm_\ndef test_addmm_(): pass\n",
        encoding="utf-8",
    )
    (root / "benchmark/test_addmm_.py").write_text(
        "import pytest\n\n@pytest.mark.addmm_\ndef test_addmm_(): pass\n",
        encoding="utf-8",
    )
    (root / "benchmark/base.py").write_text("", encoding="utf-8")
    (root / "benchmark/conftest.py").write_text("", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "KernelGen Test")
    _git(root, "config", "user.email", "kernelgen-test@localhost")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    return root, _git(root, "rev-parse", "HEAD")


def test_flaggems_checkout_accepts_only_exact_clean_revision(tmp_path, monkeypatch):
    root, base = _checkout(tmp_path)
    monkeypatch.setenv("KGS_FLAGGEMS_ROOT", str(root))

    assets = discover_assets("addmm_", base)
    assert assets.required_revision == base
    assert assets.revision == base

    (root / "vendor_compat.py").write_text("ENABLED = True\n", encoding="utf-8")
    _git(root, "add", "vendor_compat.py")
    _git(root, "commit", "-q", "-m", "vendor compatibility")
    with pytest.raises(RuntimeError, match="required fixed revision"):
        discover_assets("addmm_", base)


@pytest.mark.parametrize("pinned", [False, True])
def test_flaggems_checkout_rejects_uncommitted_changes(tmp_path, monkeypatch, pinned):
    root, base = _checkout(tmp_path)
    monkeypatch.setenv("KGS_FLAGGEMS_ROOT", str(root))
    (root / "vendor_compat.py").write_text("ENABLED = True\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        discover_assets("addmm_", base if pinned else None)


def test_flaggems_checkout_accepts_untracked_runtime_artifacts(
    tmp_path, monkeypatch
):
    root, base = _checkout(tmp_path)
    monkeypatch.setenv("KGS_FLAGGEMS_ROOT", str(root))
    (root / "worker.mudmp").write_bytes(b"runtime dump")
    (root / "pytest.log").write_text("runtime log\n", encoding="utf-8")
    (root / "report.json").write_text("{}\n", encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="ignored non-source changes"):
        assets = discover_assets("addmm_", base)

    assert assets.revision == base


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("config_name", ["pyproject.toml", "pytest.ini", "setup.cfg"])
def test_flaggems_checkout_rejects_uncommitted_pytest_configuration(
    tmp_path, monkeypatch, config_name, pinned
):
    root, base = _checkout(tmp_path)
    monkeypatch.setenv("KGS_FLAGGEMS_ROOT", str(root))
    (root / config_name).write_text("# local pytest configuration\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="pytest configuration"):
        discover_assets("addmm_", base if pinned else None)


def test_flaggems_checkout_rejects_deleted_or_renamed_python_source(
    tmp_path, monkeypatch
):
    root, base = _checkout(tmp_path)
    monkeypatch.setenv("KGS_FLAGGEMS_ROOT", str(root))
    helper = root / "tests/helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "tests/helper.py")
    _git(root, "commit", "-q", "-m", "add helper")
    revision = _git(root, "rev-parse", "HEAD")

    helper.rename(root / "tests/helper.txt")

    with pytest.raises(RuntimeError, match="tests/helper.py"):
        discover_assets("addmm_", revision)


def test_flaggems_checkout_rejects_missing_fixed_revision(tmp_path, monkeypatch):
    root, _ = _checkout(tmp_path)
    monkeypatch.setenv("KGS_FLAGGEMS_ROOT", str(root))

    with pytest.raises(RuntimeError, match="does not contain the required fixed"):
        discover_assets("addmm_", "0" * 40)


@pytest.mark.parametrize("legacy_fields", [False, True])
def test_adapter_uses_runtime_revision_and_rejects_stale_profile_fingerprint(
    tmp_path, monkeypatch, legacy_fields,
):
    root, base = _checkout(tmp_path)
    monkeypatch.setenv("KGS_FLAGGEMS_ROOT", str(root))
    catalog_root = tmp_path / "catalog"
    definitions = catalog_root / "definitions"
    definitions.mkdir(parents=True)
    source = builtin_catalog_path("flaggems-adapter-definitions") / "definitions/addmm_.json"
    (definitions / source.name).write_bytes(source.read_bytes())
    manifest = {"api_version": "v6.0", "evaluator": "flaggems", "benchmark_level": "core"}
    if legacy_fields:
        manifest.update(framework_repository="https://example.com/old.git",
                        framework_branch="old-branch", framework_revision=base)
    (catalog_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    catalog = Catalog(catalog_root)
    assert catalog.framework_revision is None
    adapter = FlagGemsEvaluationAdapter(catalog=catalog, operator=catalog.load("addmm_"))
    report = {
        "schema_version": "flaggems.benchmark-case-list/v2",
        "benchmarks": [{"schema_version": "flaggems.benchmark-case-list/v2",
                        "op_name": "addmm_", "phase": "timing", "level": "core",
                        "cases": [{"case_id": "case-0", "ordinal": 0}]}],
    }
    monkeypatch.setattr(adapter, "_native_case_report", lambda assets: report)
    before = adapter.inspect()
    assert adapter._assets().revision == base
    assert adapter._assets().required_revision is None

    # Even an empty commit (unchanged assets) changes execution identity.
    _git(root, "commit", "--allow-empty", "-q", "-m", "new runtime revision")
    actual = _git(root, "rev-parse", "HEAD")
    after = adapter.inspect()
    assert adapter._assets().revision == actual
    assert after.benchmark_fingerprint != before.benchmark_fingerprint
    assert after.case_list.cases == before.case_list.cases
    with pytest.raises(ValueError, match="fingerprint changed after inspect"):
        adapter.build_profile_command(
            SimpleNamespace(expected_backend="", benchmark_fingerprint=before.benchmark_fingerprint),
            ProfileOptions(), tmp_path / "profile", "cpu",
        )
    # The native oracle path still supplies and enforces its frozen revision.
    with pytest.raises(RuntimeError, match="HEAD does not match"):
        discover_assets("addmm_", base)


def test_unpinned_adapter_still_requires_test_assets(tmp_path, monkeypatch):
    root, _ = _checkout(tmp_path)
    monkeypatch.setenv("KGS_FLAGGEMS_ROOT", str(root))
    with pytest.raises(RuntimeError, match="could not discover pytest assets"):
        discover_assets("missing_operator")

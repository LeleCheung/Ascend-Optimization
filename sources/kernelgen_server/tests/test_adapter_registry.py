import sys
from pathlib import Path
from types import SimpleNamespace

from kernelgen_server.evaluation.adapters import registry
from kernelgen_server.evaluation.adapters.flaggems import discovery


def test_native_framework_import_uses_validated_flaggems_checkout(
    tmp_path, monkeypatch
):
    root = tmp_path / "FlagGems"
    source = root / "src"
    source.mkdir(parents=True)
    catalog = SimpleNamespace(
        framework="flaggems",
        framework_revision="a" * 40,
        framework_root=None,
    )
    calls = []

    def discover(operator, revision):
        calls.append((operator, revision))
        return SimpleNamespace(root=root)

    monkeypatch.setattr(discovery, "discover_assets", discover)
    monkeypatch.setattr(sys, "path", list(sys.path))

    registry._prepare_framework_import(catalog, "addmm_")

    assert calls == [("addmm_", "a" * 40)]
    assert sys.path[0] == str(source)
    assert catalog.framework_root == Path(root)


def test_native_framework_import_rejects_preloaded_different_checkout(
    tmp_path, monkeypatch
):
    root = tmp_path / "FlagGems"
    (root / "src").mkdir(parents=True)
    installed = tmp_path / "site-packages" / "flag_gems" / "__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    catalog = SimpleNamespace(
        framework="flaggems",
        framework_revision="a" * 40,
        framework_root=None,
    )

    monkeypatch.setattr(
        discovery,
        "discover_assets",
        lambda *_: SimpleNamespace(root=root),
    )
    monkeypatch.setitem(
        sys.modules,
        "flag_gems",
        SimpleNamespace(__file__=str(installed)),
    )

    try:
        registry._prepare_framework_import(catalog, "linear_backward")
    except RuntimeError as exc:
        assert "different checkout" in str(exc)
    else:
        raise AssertionError("preloaded different FlagGems checkout was accepted")

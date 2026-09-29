from pathlib import Path

import pytest

from kernelgen_server.profiling.hygon import hygon_trace_compat


def _fake_rocprof(tmp_path: Path) -> Path:
    root = tmp_path / "dtk" / "rocprofiler"
    executable = root / "bin" / "rocprof"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    table = root / "libexec" / "rocprofiler" / "tblextr.py"
    table.parent.mkdir(parents=True)
    table.write_text("", encoding="utf-8")
    return executable


def _write_fake_compat(build_dir: Path) -> None:
    for name in ("libroctracer_tool.so", "libfile_plugin.so"):
        (build_dir / name).write_text("binary", encoding="utf-8")
    (build_dir / "libroctracer64.so.4").symlink_to("libroctracer_tool.so")


def test_hygon_trace_compat_is_published_atomically(monkeypatch, tmp_path: Path):
    executable = _fake_rocprof(tmp_path)
    cache = tmp_path / "cache"

    def fake_build(build_dir, dtk_root, *, command_runner=None):
        del dtk_root, command_runner
        _write_fake_compat(build_dir)

    monkeypatch.setattr(hygon_trace_compat, "_build_libraries", fake_build)

    result = hygon_trace_compat.prepare_hygon_rocprof(executable, cache)

    assert result.library_dir.is_dir()
    assert result.launcher.is_file()
    assert str(result.library_dir) in result.launcher.read_text(encoding="utf-8")
    assert not list(cache.glob(".*.build-*"))


def test_hygon_trace_compat_failure_does_not_publish_partial_cache(
    monkeypatch, tmp_path: Path
):
    executable = _fake_rocprof(tmp_path)
    cache = tmp_path / "cache"

    def failing_build(build_dir, dtk_root, *, command_runner=None):
        del dtk_root, command_runner
        (build_dir / "partial.o").write_text("partial", encoding="utf-8")
        raise RuntimeError("compile failed")

    monkeypatch.setattr(hygon_trace_compat, "_build_libraries", failing_build)

    with pytest.raises(RuntimeError, match="compile failed"):
        hygon_trace_compat.prepare_hygon_rocprof(executable, cache)

    assert not list(cache.glob("rocm61-*"))
    assert not list(cache.glob(".*.build-*"))

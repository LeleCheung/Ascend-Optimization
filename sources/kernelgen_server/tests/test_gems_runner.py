"""Host tests for the KGS-owned pytest profile runner."""

from kernelgen_server.profiling.gems_runner import ProfilePlugin
from kernelgen_server.profiling import gems_runner
from contextlib import contextmanager
import pytest


@pytest.mark.parametrize(
    "backend,vendor",
    [
        ("cuda", "nvidia"),
        ("npu", "ascend"),
        ("musa", "mthreads"),
        ("mlu", "cambricon"),
        ("metax", "metax"),
        ("thead", "thead"),
        ("hygon", "hygon"),
        ("iluvatar", "iluvatar"),
        ("enflame", "enflame"),
        ("kunlunxin", "kunlunxin"),
    ],
)
def test_capture_uses_kgs_backend_not_gems_vendor(monkeypatch, backend, vendor):
    events = []

    @contextmanager
    def scope(actual):
        events.append(("start", actual))
        try:
            yield
        finally:
            events.append(("stop", actual))

    monkeypatch.setattr(gems_runner, "capture_scope", scope)
    hook = ProfilePlugin(backend, "case")
    with hook.pytest_flaggems_profile_scope(backend=vendor, case_id="case"):
        events.append(("candidate", backend))
    assert hook.completed == 1
    assert events == [("start", backend), ("candidate", backend), ("stop", backend)]


def test_candidate_failure_closes_capture_without_completion(monkeypatch):
    events = []

    @contextmanager
    def scope(backend):
        try:
            yield
        finally:
            events.append("stopped")

    monkeypatch.setattr(gems_runner, "capture_scope", scope)
    hook = ProfilePlugin("cuda", "case")
    with pytest.raises(ValueError):
        with hook.pytest_flaggems_profile_scope(backend="nvidia", case_id="case"):
            raise ValueError("candidate")
    assert events == ["stopped"]
    assert hook.completed == 0


def test_capture_hook_requires_the_selected_case():
    hook = ProfilePlugin("cuda", "case-1")
    assert hook.case_id == "case-1"
    assert hook.completed == 0


def test_capture_hook_rejects_a_different_case():
    hook = ProfilePlugin("cuda", "case-1")
    try:
        with hook.pytest_flaggems_profile_scope(backend="nvidia", case_id="case-2"):
            pass
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched profile case was accepted")


def test_compiler_preparation_precedes_pytest_imports(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        gems_runner,
        "prepare_profile_compiler",
        lambda backend: events.append(("prepare", backend)),
    )

    def run_pytest(args, *, plugins):
        events.append(("pytest", args))
        assert len(plugins) == 1
        assert isinstance(plugins[0], ProfilePlugin)
        plugins[0].completed = 1
        return 0

    monkeypatch.setattr(gems_runner.pytest, "main", run_pytest)
    marker = tmp_path / "completed"
    assert gems_runner.run("enflame", "case", marker, ["--profile-only"]) == 0
    assert events == [("prepare", "enflame"), ("pytest", ["--profile-only"])]
    assert marker.read_text() == "completed\n"


@pytest.mark.parametrize("completed", [0, 2])
def test_success_without_exact_capture_rejects_stale_marker(
    monkeypatch, tmp_path, completed
):
    marker = tmp_path / "completed"
    marker.write_text("stale\n")

    def run_pytest(args, *, plugins):
        plugins[0].completed = completed
        return 0

    monkeypatch.setattr(gems_runner.pytest, "main", run_pytest)
    with pytest.raises(RuntimeError, match="exactly one capture"):
        gems_runner.run("cuda", "case", marker, [])
    assert not marker.exists()


def test_failed_pytest_does_not_write_completion(monkeypatch, tmp_path):
    marker = tmp_path / "completed"
    marker.write_text("stale\n")
    monkeypatch.setattr(gems_runner.pytest, "main", lambda args, plugins: 1)
    assert gems_runner.run("cuda", "case", marker, []) == 1
    assert not marker.exists()


def test_runs_have_independent_plugin_instances(monkeypatch, tmp_path):
    instances = []

    @contextmanager
    def capture(backend):
        yield

    monkeypatch.setattr(gems_runner, "capture_scope", capture)

    def run_pytest(args, *, plugins):
        plugin = plugins[0]
        instances.append(plugin)
        assert plugin.completed == 0
        with plugin.pytest_flaggems_profile_scope("nvidia", "case"):
            pass
        return 0

    monkeypatch.setattr(gems_runner.pytest, "main", run_pytest)
    for index in range(2):
        assert gems_runner.run("cuda", "case", tmp_path / str(index), []) == 0
    assert instances[0] is not instances[1]
    assert [plugin.completed for plugin in instances] == [1, 1]


def test_plugin_rejects_wrong_backend_and_second_capture(monkeypatch):
    @contextmanager
    def capture(backend):
        yield

    monkeypatch.setattr(gems_runner, "capture_scope", capture)
    plugin = ProfilePlugin("cuda", "case")
    with pytest.raises(RuntimeError, match="does not match"):
        with plugin.pytest_flaggems_profile_scope("ascend", "case"):
            pass
    with plugin.pytest_flaggems_profile_scope("nvidia", "case"):
        pass
    with pytest.raises(RuntimeError, match="does not match"):
        with plugin.pytest_flaggems_profile_scope("nvidia", "case"):
            pass


@pytest.mark.parametrize(
    "backend,line_info,expected",
    [
        ("enflame", "0", True),
        ("enflame", "false", True),
        ("enflame", "1", False),
        ("cuda", "0", False),
    ],
)
def test_shared_profile_compiler_preparation(monkeypatch, backend, line_info, expected):
    from kernelgen_server.profiling.capture import prepare_profile_compiler
    from kernelgen_server.profiling.enflame import enflame_line_info

    events = []
    monkeypatch.setenv("TRITON_DISABLE_LINE_INFO", line_info)
    monkeypatch.setattr(
        enflame_line_info,
        "enable_enflame_line_info_compat",
        lambda: events.append("enabled"),
    )
    prepare_profile_compiler(backend)
    assert events == (["enabled"] if expected else [])

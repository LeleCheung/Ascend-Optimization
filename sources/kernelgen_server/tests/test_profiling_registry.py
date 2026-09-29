import pytest

from kernelgen_server.profiling import registry
from kernelgen_server.profiling.capture import capture_scope
from kernelgen_server.profiling.iluvatar.iluvatar import IxknProfiler
from kernelgen_server.profiling.models import ProfileOptions


@pytest.mark.parametrize(
    ("backend", "class_name"),
    [
        ("cuda", "NcuProfiler"),
        ("npu", "MsprofProfiler"),
        ("mlu", "CnperfProfiler"),
        ("metax", "McTracerProfiler"),
        ("kunlunxin", "XProfilerProfiler"),
        ("musa", "McuProfiler"),
        ("enflame", "TopsProfProfiler"),
        ("hygon", "HipprofProfiler"),
        ("iluvatar", "IxknProfiler"),
        ("thead", "THeadACUProfiler"),
    ],
)
def test_registry_lazily_resolves_every_supported_backend(backend, class_name):
    registry._INSTANCES.clear()

    profiler = registry.get_profiler(backend)

    assert type(profiler).__name__ == class_name
    assert profiler.backend == backend


def test_registry_returns_explicit_unsupported_profiler():
    registry._INSTANCES.clear()

    profiler = registry.get_profiler("future-device")

    assert profiler.available() is False
    assert profiler.levels() == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [("bad", 10), (False, 10), (0, 1), ("3", 3), (999, 10)],
)
def test_iluvatar_normalizes_ixkn_kernel_limit(value, expected):
    profiler = IxknProfiler()
    options = ProfileOptions(
        backend_options={"iluvatar": {"ixkn_max_kernels": value}}
    )

    normalized, warnings = profiler.normalize_options(options)

    assert normalized.backend_options["iluvatar"]["ixkn_max_kernels"] == expected
    assert bool(warnings) is (value not in {"3"})


def test_capture_scope_uses_fixed_backend_control_and_stops_on_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "kernelgen_server.profiling.capture._runtime_control",
        lambda library, function: calls.append((library, function)),
    )

    with pytest.raises(RuntimeError, match="candidate failed"):
        with capture_scope("mlu"):
            calls.append("candidate")
            raise RuntimeError("candidate failed")

    assert calls == [
        ("libcnrt.so", "cnrtProfilerStart"),
        "candidate",
        ("libcnrt.so", "cnrtProfilerStop"),
    ]


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("cuda", [("cuda", True), "candidate", ("cuda", False)]),
        ("iluvatar", [("cuda", True), "candidate", ("cuda", False)]),
        (
            "mlu",
            [
                ("libcnrt.so", "cnrtProfilerStart"),
                "candidate",
                ("libcnrt.so", "cnrtProfilerStop"),
            ],
        ),
        (
            "metax",
            [
                ("libmcruntime.so", "mcProfilerStart"),
                "candidate",
                ("libmcruntime.so", "mcProfilerStop"),
            ],
        ),
        (
            "enflame",
            [
                ("libtopsrt.so", "topsProfilerStart"),
                "candidate",
                ("libtopsrt.so", "topsProfilerStop"),
            ],
        ),
        (
            "thead",
            [
                ("libhggcrt1.so", "hggcProfilerStart"),
                "candidate",
                ("libhggcrt1.so", "hggcProfilerStop"),
            ],
        ),
        ("npu", ["candidate"]),
        ("hygon", ["candidate"]),
        ("musa", ["candidate"]),
    ],
)
def test_capture_scope_has_one_fixed_control_for_each_runtime_backend(
    monkeypatch, backend, expected
):
    calls = []
    monkeypatch.setattr(
        "kernelgen_server.profiling.capture._cuda_control",
        lambda start: calls.append(("cuda", start)),
    )
    monkeypatch.setattr(
        "kernelgen_server.profiling.capture._runtime_control",
        lambda library, function: calls.append((library, function)),
    )

    with capture_scope(backend):
        calls.append("candidate")

    assert calls == expected


def test_capture_scope_emits_kunlunxin_monotonic_window(monkeypatch, capsys):
    values = iter((1_000, 6_000))
    monkeypatch.setattr(
        "kernelgen_server.profiling.capture.time.clock_gettime_ns",
        lambda clock: next(values),
    )

    with capture_scope("kunlunxin"):
        pass

    output = capsys.readouterr().out
    assert "KGS_PROFILE_START_NS=1000" in output
    assert "KGS_PROFILE_END_NS=6000" in output
    assert "KGS_PROFILE_WALL_TIME_US=5.000" in output

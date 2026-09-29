import sys
import time
from pathlib import Path

from kernelgen_server.profiling.base import (
    ManagedProfiler,
    ProfileExecutionContext,
)
from kernelgen_server.profiling.models import (
    BackendProfileResult,
    ProfileCommand,
    ProfileOptions,
    ProfileRequest,
)
from kernelgen_server.profiling.process import RequestDeadline
from kernelgen_server.profiling.service import ProfileService
from kernelgen_server.schema import (
    EvaluatorBinding,
    Implementation,
    SourceFile,
)


def _request(options: ProfileOptions) -> ProfileRequest:
    return ProfileRequest(
        binding=EvaluatorBinding(catalog_name="test", definition="identity"),
        implementation=Implementation(
            name="candidate",
            definition="identity",
            language="python",
            entrypoint="main.py::run",
            sources=[SourceFile(path="main.py", content="def run(x): return x")],
        ),
        benchmark_fingerprint="sha256:test",
        case_id="case-0",
        expected_backend="cuda",
        options=options,
    )


def test_legacy_source_level_is_normalized_at_input_boundary():
    options = ProfileOptions(level="source")

    assert options.level == "instruction"
    assert options.model_dump()["level"] == "instruction"


def test_profile_service_rejects_level_before_building_command(
    monkeypatch, tmp_path: Path
):
    class MetricsOnlyProfiler:
        name = "metrics-only"

        @staticmethod
        def available():
            return True

        @staticmethod
        def available_for(options):
            return MetricsOnlyProfiler.available()

        @staticmethod
        def levels():
            return ["metrics"]

        @staticmethod
        def levels_for(options):
            return MetricsOnlyProfiler.levels()

        @staticmethod
        def normalize_options(options):
            raise AssertionError(f"normalized unsupported options: {options}")

    monkeypatch.setattr(
        "kernelgen_server.profiling.service.get_profiler",
        lambda backend: MetricsOnlyProfiler(),
    )
    monkeypatch.setattr(
        "kernelgen_server.evaluation.adapters.create_adapter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("built a command for an unsupported level")
        ),
    )

    result = ProfileService(tmp_path, backend="cuda").run(
        _request(ProfileOptions(level="instruction")),
        "cuda:0",
    )

    assert result.status == "unsupported"
    assert result.options.level == "instruction"


def test_profile_service_reports_startup_hardware_when_backend_has_no_override(
    monkeypatch, tmp_path: Path
):
    class UnavailableProfiler:
        name = "unavailable"

        @staticmethod
        def available_for(options):
            return False

    monkeypatch.setattr(
        "kernelgen_server.profiling.service.get_profiler",
        lambda backend: UnavailableProfiler(),
    )
    hardware = {"target": {"device": "observed-model", "architecture": "isa"}}

    result = ProfileService(
        tmp_path,
        backend="cuda",
        hardware=hardware,
    ).run(_request(ProfileOptions()), "cuda:0")

    assert result.status == "unsupported"
    assert result.hardware == hardware


def test_profile_service_normalizes_before_adapter_and_propagates_hardware(
    monkeypatch, tmp_path: Path
):
    observed = {}

    class FakeProfiler:
        name = "fake-profiler"

        @staticmethod
        def available():
            return True

        @staticmethod
        def available_for(options):
            observed["availability_options"] = options
            return FakeProfiler.available()

        @staticmethod
        def levels():
            return ["metrics"]

        @staticmethod
        def levels_for(options):
            observed["level_options"] = options
            return FakeProfiler.levels()

        @staticmethod
        def normalize_options(options):
            return (
                options.model_copy(update={"iterations": 1, "timeout_sec": 7}),
                ["iterations normalized"],
            )

        @staticmethod
        def profile_command(command, options, artifact_dir, device, *, deadline):
            observed["profile"] = (command, options, artifact_dir, device, deadline)
            return BackendProfileResult(
                status="completed",
                profiler="fake-profiler",
                hardware={"device": "observed-model"},
            )

    class FakeAdapter:
        @staticmethod
        def build_profile_command(request, options, artifact_dir, device):
            observed["adapter_options"] = options
            return ProfileCommand(
                argv=(sys.executable, "-c", "pass"),
                cwd=str(tmp_path),
            )

    monkeypatch.setattr(
        "kernelgen_server.profiling.service.get_profiler",
        lambda backend: FakeProfiler(),
    )
    monkeypatch.setattr(
        "kernelgen_server.evaluation.adapters.create_adapter",
        lambda *args, **kwargs: FakeAdapter(),
    )

    started = time.monotonic()
    result = ProfileService(tmp_path / "profiles", backend="cuda").run(
        _request(ProfileOptions(iterations=50, timeout_sec=30)),
        "cuda:0",
    )

    assert observed["adapter_options"].iterations == 1
    assert observed["availability_options"].iterations == 50
    assert observed["level_options"].iterations == 50
    assert observed["profile"][1].iterations == 1
    assert observed["profile"][4].expires_at <= started + 7.1
    assert result.options.iterations == 1
    assert result.hardware == {"device": "observed-model"}
    assert result.warnings == ["iterations normalized"]


class _MarkerProfiler(ManagedProfiler):
    backend = "test"
    name = "marker-profiler"

    def __init__(self, *, write_marker: bool):
        self.write_marker = write_marker
        self.cleaned = False

    @staticmethod
    def available():
        return True

    @staticmethod
    def capabilities():
        return ["kernel_profile"]

    def collect(self, context: ProfileExecutionContext, prepared):
        del prepared
        script = "from pathlib import Path; "
        if self.write_marker:
            script += f"Path({str(context.completion_marker)!r}).write_text('done')"
        else:
            script += "pass"
        return context.run_profile_stage(
            [sys.executable, "-c", script],
            label="marker collection",
        )

    @staticmethod
    def validate_evidence(context: ProfileExecutionContext):
        context.add_capability("kernel_profile")

    def cleanup(self, context: ProfileExecutionContext, prepared):
        del context, prepared
        self.cleaned = True


class _SlowPostprocessProfiler(_MarkerProfiler):
    def postprocess(self, context, prepared, collected):
        del context, prepared, collected
        time.sleep(0.03)


class _MultiStageMarkerProfiler(_MarkerProfiler):
    def collect(self, context: ProfileExecutionContext, prepared):
        del prepared
        for stage in range(2):
            context.run_profile_stage(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"Path({str(context.completion_marker)!r}).write_text('done')"
                    ),
                ],
                label=f"marker collection {stage}",
            )


def _run_marker_profiler(
    profiler: _MarkerProfiler,
    artifact_dir: Path,
    *,
    source_roots: tuple[str, ...] = (),
):
    marker = artifact_dir / "runner-completed"
    return profiler.profile_command(
        ProfileCommand(
            argv=(sys.executable, "-c", "pass"),
            cwd=str(artifact_dir.parent),
            source_roots=source_roots,
            completion_marker_path=str(marker),
        ),
        ProfileOptions(timeout_sec=10),
        artifact_dir,
        "cuda:0",
        deadline=RequestDeadline.from_timeout(10),
    )


def test_managed_profiler_requires_native_completion_marker(tmp_path: Path):
    profiler = _MarkerProfiler(write_marker=False)

    result = _run_marker_profiler(profiler, tmp_path / "profile")

    assert result.status == "failed"
    assert "completion marker" in (result.error or "")
    assert profiler.cleaned is True


def test_managed_profiler_accepts_completion_marker(tmp_path: Path):
    profiler = _MarkerProfiler(write_marker=True)

    result = _run_marker_profiler(profiler, tmp_path / "profile")

    assert result.status == "completed"
    assert result.capabilities == ["kernel_profile"]
    assert profiler.cleaned is True


def test_managed_profiler_reuses_one_command_across_completed_stages(tmp_path: Path):
    profiler = _MultiStageMarkerProfiler(write_marker=True)

    result = _run_marker_profiler(profiler, tmp_path / "profile")

    assert result.status == "completed"
    assert result.summary["stage_count"] == 2


def test_managed_profiler_rejects_source_root_outside_artifacts(tmp_path: Path):
    profiler = _MarkerProfiler(write_marker=True)
    outside = tmp_path / "outside"
    outside.mkdir()

    result = _run_marker_profiler(
        profiler,
        tmp_path / "profile",
        source_roots=(str(outside),),
    )

    assert result.status == "failed"
    assert "unsafe or missing ProfileCommand source_root" in (result.error or "")
    assert profiler.cleaned is True


def test_managed_profiler_cannot_complete_after_request_deadline(tmp_path: Path):
    profiler = _SlowPostprocessProfiler(write_marker=True)
    artifact_dir = tmp_path / "profile"
    marker = artifact_dir / "runner-completed"

    result = profiler.profile_command(
        ProfileCommand(
            argv=(sys.executable, "-c", "pass"),
            cwd=str(tmp_path),
            completion_marker_path=str(marker),
        ),
        ProfileOptions(timeout_sec=10),
        artifact_dir,
        "cuda:0",
        deadline=RequestDeadline.from_timeout(0.01),
    )

    assert result.status == "failed"
    assert "deadline expired" in (result.error or "")

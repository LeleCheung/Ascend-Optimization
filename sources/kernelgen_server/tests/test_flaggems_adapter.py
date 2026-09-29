import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from kernelgen_server import Catalog, builtin_catalog_path
from kernelgen_server.evaluation.adapters import create_adapter
from kernelgen_server.evaluation.adapters.flaggems.adapter import _accuracy_results
from kernelgen_server.evaluation.adapters.flaggems import adapter as adapter_module
from kernelgen_server.evaluation.audit import RequestAudit
from kernelgen_server.evaluation.loader import BuildError
from kernelgen_server.evaluation.result import aggregate_workload_results
from kernelgen_server.protocol import client
from kernelgen_server.profiling import ProfileOptions, ProfileRequest
from kernelgen_server.schema import (
    AdapterCapabilities,
    AdapterCase,
    AdapterManifest,
    BoundEvaluateRequest,
    CandidateContract,
    CaseList,
    EvaluateResponse,
    EvaluatorBinding,
    Implementation,
    InspectRequest,
    SourceFile,
    WorkloadStatus,
)


_CASE_0 = "benchmark/test_addmm_.py::test_addmm_::core::float16::0"
_CASE_1 = "benchmark/test_addmm_.py::test_addmm_::core::float16::1"


@pytest.mark.parametrize(
    "reason",
    [
        "ModuleNotFoundError: No module named 'shmem'",
        "triton.compiler.errors.CompilationError: at 12:0: invalid operand",
        "RuntimeError: CUDA error: an illegal memory access was encountered",
        "ValueError: unexpected candidate shape",
        "E   RuntimeError: failed to launch",
        "  builtins.TypeError: invalid argument",
        "Exception: candidate execution failed",
    ],
)
def test_pytest_call_exceptions_are_execution_errors(reason):
    result = _accuracy_results({"test.py::test_op": {"result": "failed", "reason": reason}})[0]
    assert result.status == WorkloadStatus.RUNTIME_ERROR
    assert result.log == reason
    assert result.metrics["native_outcome"] == "failed"


@pytest.mark.parametrize(
    "reason",
    [
        "AssertionError: Tensor-likes are not close!",
        "mismatch",
        "AssertionError: RuntimeError was not raised",
        "E   AssertionError: mismatched elements",
        "builtins.AssertionError: expected equal tensors",
        "mismatch: RuntimeError appears only in diagnostic text",
        "",
        None,
    ],
)
def test_pytest_assertions_keep_the_existing_incorrect_result(reason):
    result = _accuracy_results({"test.py::test_op": {"result": "failed", "reason": reason}})[0]
    assert result.status == WorkloadStatus.INCORRECT_NUMERICAL


@pytest.mark.parametrize("reason, expected", [
    ("RuntimeError: compilation failed", "RUNTIME_ERROR"),
    ("AssertionError: Tensor-likes are not close!", "INCORRECT_NUMERICAL"),
])
def test_accuracy_failure_classification_survives_response_aggregation(reason, expected):
    results = _accuracy_results({"test.py::test_op": {"result": "failed", "reason": reason}})
    response = aggregate_workload_results(results, device="cuda:0", backend="cuda")
    decoded = EvaluateResponse.model_validate_json(response.model_dump_json())
    assert decoded.status == expected
    assert decoded.per_workload[0].status == expected
    assert decoded.log == decoded.per_workload[0].log == reason
    assert decoded.num_passed == 0
    assert decoded.geo_mean is None


def _binding() -> EvaluatorBinding:
    return EvaluatorBinding(
        catalog_name="flaggems-adapter-definitions",
        definition="addmm_",
    )


def _request(source: str | None = None) -> BoundEvaluateRequest:
    return BoundEvaluateRequest(
        binding=_binding(),
        implementation=Implementation(
            name="candidate",
            definition="addmm_",
            language="python",
            entrypoint="candidate.py::run",
            sources=[
                SourceFile(
                    path="candidate.py",
                    content=source
                    or "def run(self, mat1, mat2, *, beta=1, alpha=1): return self",
                )
            ],
        ),
    )


def _native_case_report(_assets=None):
    return {
        "schema_version": "flaggems.benchmark-case-list/v2",
        "benchmarks": [
            {
                "schema_version": "flaggems.benchmark-case-list/v2",
                "op_name": "addmm_",
                "phase": "timing",
                "level": "core",
                "cases": [
                    {
                        "case_id": _CASE_0,
                        "ordinal": 0,
                        "dtype": "torch.float16",
                        "shape": {"b": 2, "m": 384, "n": 384, "k": 384},
                        "params": {"b_column_major": False},
                    },
                    {
                        "case_id": _CASE_1,
                        "ordinal": 1,
                        "dtype": "torch.float16",
                        "shape": {"b": 2, "m": 4096, "n": 4096, "k": 4096},
                        "params": {"b_column_major": False},
                    },
                ],
            }
        ],
    }


def _manifest() -> AdapterManifest:
    return AdapterManifest(
        kind="flaggems",
        benchmark_fingerprint="test-fingerprint",
        candidate_contract=CandidateContract(
            signature="(self, mat1, mat2, *, beta=1, alpha=1)"
        ),
        capabilities=AdapterCapabilities(preflight=True, profile=True),
        case_list=CaseList(
            adapter_kind="flaggems",
            operator="addmm_",
            benchmark_fingerprint="test-fingerprint",
            cases=[AdapterCase(case_id=_CASE_0, ordinal=0)],
        ),
    )


def test_flaggems_catalog_owns_pure_definition_only():
    catalog = Catalog(builtin_catalog_path("flaggems-adapter-definitions"))
    operator = catalog.load("addmm_")

    assert catalog.evaluator == "flaggems"
    assert catalog.framework_repository is None
    assert catalog.framework_branch is None
    assert catalog.framework_revision is None
    assert not any(key.startswith("framework_") for key in catalog.manifest)
    assert operator.definition.reference is None
    assert [parameter.name for parameter in operator.definition.parameters] == [
        "self",
        "mat1",
        "mat2",
        "beta",
        "alpha",
    ]
    assert operator.definition.parameters[-1].kind.value == "keyword_only"


def test_inspect_exposes_only_ordered_timing_cases(monkeypatch):
    adapter = create_adapter(_binding())
    monkeypatch.setattr(adapter, "_native_case_report", _native_case_report)

    manifest = adapter.inspect()

    assert manifest.kind == "flaggems"
    assert manifest.candidate_contract.signature == (
        "(self, mat1, mat2, *, beta=1, alpha=1)"
    )
    assert manifest.capabilities.profile is True
    assert [case.case_id for case in manifest.case_list.cases] == [
        _CASE_0,
        _CASE_1,
    ]
    assert all(case.phase == "timing" for case in manifest.case_list.cases)
    assert manifest.case_list.cases[0].params == {"b_column_major": False}


def test_preflight_leaves_accelerator_initialization_to_pytest_child(
    monkeypatch, tmp_path
):
    class ParentDevice:
        def set_device(self, device):
            raise AssertionError(f"parent initialized accelerator {device}")

    # This is a mocked two-device binding test, independent of the host mask.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    adapter = create_adapter(
        _binding(),
        device=ParentDevice(),
        device_string="cuda:1",
        backend="metax",
    )
    monkeypatch.setattr(adapter, "_native_case_report", _native_case_report)
    manifest = adapter.inspect()
    assets = SimpleNamespace(
        root=tmp_path,
        performance=[tmp_path / "benchmark/test_addmm_.py"],
        native_operator="addmm_",
    )
    monkeypatch.setattr(adapter, "_manifest", lambda: manifest)
    monkeypatch.setattr(adapter, "_assets", lambda: assets)
    monkeypatch.setattr(
        adapter, "_validate_candidate_abi", lambda request, assets: None
    )
    monkeypatch.setattr(
        adapter,
        "_read_preflight_coverage",
        lambda path, process, expected_operator: [
            {"case_id": case.case_id, "count": 1}
            for case in manifest.case_list.cases
        ],
    )
    pytest_calls = []

    def fake_run_pytest(args, **kwargs):
        pytest_calls.append((args, kwargs))
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(
        adapter_module,
        "_run_pytest",
        fake_run_pytest,
    )

    result = adapter.preflight(_request())

    assert result.status == "PASSED"
    args, kwargs = pytest_calls[0]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert args[args.index("--override") + 1].endswith(
        "candidate/candidate.py:run"
    )
    assert args[args.index("--output") + 1].endswith(
        "preflight.json"
    )
    assert "--candidate-code-path" not in args
    assert "--candidate-report-path" not in args
    assert args[args.index("--record") + 1] == "json"
    assert "-p" not in args


def test_flaggems_pytest_loads_system_site_but_ignores_user_site(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(adapter_module.subprocess, "run", fake_run)
    adapter_module._run_pytest(
        ["-q", "tests/test_addmm_.py"],
        env={},
        root=tmp_path,
        timeout=30,
    )

    assert calls[0][0][1:3] == ["-s", "-c"]
    assert calls[0][0][3] == adapter_module._PYTEST_BOOTSTRAP


def test_flaggems_pytest_env_defers_site_packages_until_after_stdlib(
    monkeypatch, tmp_path
):
    adapter = create_adapter(_binding())
    monkeypatch.setattr(
        adapter_module.site,
        "getsitepackages",
        lambda: ["/opt/test/site-packages"],
    )
    monkeypatch.setenv(
        "PYTHONPATH",
        f"/opt/test/site-packages{adapter_module.os.pathsep}/opt/custom",
    )

    python_paths = adapter._base_env(tmp_path)["PYTHONPATH"].split(
        adapter_module.os.pathsep
    )

    assert python_paths == [
        str(tmp_path / "src"),
        str(adapter_module.Path(adapter_module.__file__).resolve().parents[4]),
        "/opt/custom",
    ]


@pytest.mark.parametrize("backend,vendor", [
    ("cuda", "nvidia"), ("npu", "ascend"), ("musa", "mthreads"),
    ("mlu", "cambricon"), ("metax", "metax"), ("hygon", "hygon"),
    ("iluvatar", "iluvatar"), ("thead", "thead"),
    ("enflame", "enflame"), ("kunlunxin", "kunlunxin"), ("txda", "txda"),
])
@pytest.mark.parametrize("inherited", [None, "wrong-inherited-vendor"])
def test_flaggems_vendor_comes_from_server_backend(monkeypatch, tmp_path, backend, vendor, inherited):
    if inherited is None:
        monkeypatch.delenv("GEMS_VENDOR", raising=False)
    else:
        monkeypatch.setenv("GEMS_VENDOR", inherited)
    adapter = create_adapter(_binding(), backend=backend)
    assert adapter._base_env(tmp_path)["GEMS_VENDOR"] == vendor
    assert adapter_module.os.environ.get("GEMS_VENDOR") == inherited


def test_native_case_report_runs_concurrently_and_uses_extended_timeout(
    monkeypatch, tmp_path
):
    adapter = create_adapter(_binding())
    assets = SimpleNamespace(
        root=tmp_path,
        performance=[tmp_path / "benchmark/test_addmm_.py"],
        native_operator="addmm_",
    )
    both_entered = threading.Event()
    release = threading.Event()
    calls_lock = threading.Lock()
    timeouts = []
    active = 0
    max_active = 0

    def fake_run_pytest(args, **kwargs):
        nonlocal active, max_active
        with calls_lock:
            active += 1
            max_active = max(max_active, active)
            timeouts.append(kwargs["timeout"])
            if active == 2:
                both_entered.set()
        assert release.wait(timeout=5)
        output = adapter_module.Path(args[args.index("--output") + 1])
        output.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
        with calls_lock:
            active -= 1
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(adapter_module, "_run_pytest", fake_run_pytest)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(adapter._native_case_report, assets)
        second = executor.submit(adapter._native_case_report, assets)
        try:
            assert both_entered.wait(timeout=5)
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert max_active == 2
    assert timeouts == [600.0, 600.0]


@pytest.mark.parametrize("calls", [{}, {"flag_gems.other": 1}, {"flag_gems.addmm_": 0}, {"flag_gems.addmm_": True}])
def test_candidate_coverage_rejects_missing_or_wrong_operator(calls):
    with pytest.raises(RuntimeError, match="bypassed"):
        adapter_module.FlagGemsEvaluationAdapter._assert_accuracy_call_coverage(
            {"test_op": {"result": "passed", "candidate_calls": calls}}, "addmm_"
        )


def test_preflight_maps_candidate_import_failure_to_failed_result(monkeypatch):
    adapter = create_adapter(
        _binding(), device=object(), device_string="musa:0", backend="musa"
    )
    monkeypatch.setattr(
        adapter,
        "_prepare_workdir",
        lambda request, work: (_ for _ in ()).throw(
            BuildError("failed importing candidate.py: CUDA backend is unavailable")
        ),
    )
    monkeypatch.setattr(adapter, "_manifest", _manifest)

    result = adapter.preflight(_request())

    assert result.status == "FAILED"
    assert result.stage == "candidate_import"
    assert result.benchmark_fingerprint == "test-fingerprint"
    assert result.num_cases == 1
    assert "CUDA backend is unavailable" in result.log


def test_evaluate_maps_candidate_import_failure_to_runtime_result(monkeypatch):
    adapter = create_adapter(
        _binding(), device=object(), device_string="musa:0", backend="musa"
    )
    monkeypatch.setattr(
        adapter,
        "_prepare_workdir",
        lambda request, work: (_ for _ in ()).throw(
            BuildError("failed importing candidate.py: torch.cuda.Stream failed")
        ),
    )

    result = adapter.evaluate(_request())

    assert result.status == "RUNTIME_ERROR"
    assert result.device == "musa:0"
    assert result.server_backend == "musa"
    assert result.num_workloads == 0
    assert "torch.cuda.Stream failed" in result.log


def test_preflight_maps_wrong_public_operator_to_failed_result(
    monkeypatch, tmp_path
):
    adapter = create_adapter(
        _binding(), device=object(), device_string="musa:0", backend="musa"
    )
    manifest = _manifest()
    assets = SimpleNamespace(
        root=tmp_path,
        performance=[tmp_path / "benchmark/test_addmm_.py"],
        native_operator="addmm_",
    )
    monkeypatch.setattr(
        adapter,
        "_prepare_workdir",
        lambda request, work: (manifest, assets, work, {}),
    )

    def fake_run_pytest(args, **kwargs):
        report = adapter_module.Path(args[args.index("--output") + 1])
        report.write_text(
            json.dumps({"schema_version": "flaggems.preflight/v1", "records": [
                {"operator": "different_op", "case_id": _CASE_0, "override": True, "count": 1, "status": "passed"}
            ]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(adapter_module, "_run_pytest", fake_run_pytest)

    result = adapter.preflight(_request())

    assert result.status == "FAILED"
    assert result.stage == "candidate_contract"
    assert result.benchmark_fingerprint == "test-fingerprint"
    assert "wrong public operator" in result.log


def test_evaluate_maps_wrong_public_operator_to_runtime_result(
    monkeypatch, tmp_path
):
    adapter = create_adapter(
        _binding(), device=object(), device_string="musa:0", backend="musa"
    )
    manifest = _manifest()
    assets = SimpleNamespace(
        root=tmp_path,
        correctness=[tmp_path / "tests/test_addmm_.py"],
        performance=[tmp_path / "benchmark/test_addmm_.py"],
        native_operator="addmm_",
    )
    monkeypatch.setattr(
        adapter,
        "_prepare_workdir",
        lambda request, work: (manifest, assets, work, {}),
    )

    def fake_run_pytest(args, **kwargs):
        accuracy_path = adapter_module.Path(args[args.index("--output") + 1])
        accuracy_path.write_text(
            json.dumps(
                {"tests/test_addmm_.py::test_addmm_": {"result": "passed"}}
            ),
            encoding="utf-8",
        )
        assert "--override" in args
        assert "--candidate-code-path" not in args
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(adapter_module, "_run_pytest", fake_run_pytest)

    result = adapter.evaluate(_request())

    assert result.status == "RUNTIME_ERROR"
    assert result.num_workloads == 0
    assert "bypassed" in result.log


def test_native_accuracy_failure_is_not_hidden():
    results = _accuracy_results(
        {
            "tests/test_addmm_.py::test_addmm_[bad]": {
                "params": {},
                "result": "failed",
                "reason": "values differ",
            }
        }
    )
    assert results[0].status == WorkloadStatus.INCORRECT_NUMERICAL
    assert results[0].log == "values differ"


@pytest.mark.parametrize("status", ["PASSED", "ALL_SKIP"])
def test_bound_client_uses_common_evaluate_and_inspect_routes(monkeypatch, status):
    calls = []

    def fake_post(server_url, path, payload, timeout, *, operation_id=None):
        calls.append((server_url, path, payload, timeout))
        if path == "/inspect":
            adapter = create_adapter(_binding())
            monkeypatch.setattr(adapter, "_native_case_report", _native_case_report)
            return adapter.inspect().model_dump(mode="json")
        return {
            "api_version": "v6.0",
            "status": status,
            "device": "cuda:0",
            "server_backend": "cuda",
            "num_workloads": 2,
            "num_passed": 2,
        }

    monkeypatch.setattr(client, "_post", fake_post)
    manifest = client.inspect(InspectRequest(binding=_binding()), "http://server")
    response = client.evaluate(_request(), "http://server", timeout=10)

    assert manifest.kind == "flaggems"
    assert isinstance(response, EvaluateResponse)
    assert response.status == status
    assert [call[1] for call in calls] == ["/inspect", "/evaluate"]
    assert "suite" not in calls[1][2]


def test_bound_request_audit_records_catalog_definition(tmp_path):
    audit = RequestAudit(tmp_path, manifest={"backend": "cuda"})
    handle = audit.begin("evaluate", _request())
    record = json.loads(
        (handle.request_dir / "request.json").read_text(encoding="utf-8")
    )
    assert record["metadata"]["definition"] == "addmm_"
    assert record["payload"]["binding"]["catalog_name"] == "flaggems-adapter-definitions"
    assert "suite" not in record["payload"]


def test_profile_command_replays_exact_timing_selector(monkeypatch, tmp_path):
    adapter = create_adapter(
        _binding(), device_string="cuda:0", backend="cuda"
    )
    monkeypatch.setattr(adapter, "_native_case_report", _native_case_report)
    manifest = adapter.inspect()
    request = ProfileRequest(
        binding=_binding(),
        implementation=_request().implementation,
        benchmark_fingerprint=manifest.benchmark_fingerprint,
        case_id=_CASE_1,
        expected_backend="cuda",
        options=ProfileOptions(warmup=2, iterations=3),
    )

    command = adapter.build_profile_command(
        request, request.options, tmp_path, "cuda:0"
    )

    assert "benchmark/test_addmm_.py" in command.argv
    assert command.argv[command.argv.index("--case-id") + 1] == _CASE_1
    assert command.argv[command.argv.index("--profile-warmup") + 1] == "2"
    assert command.argv[command.argv.index("--profile-iterations") + 1] == "3"
    override = command.argv[command.argv.index("--override") + 1]
    assert "addmm_:" in override
    assert override.endswith(":run")
    assert "target/candidate/candidate.py" in override
    assert "--candidate-code-path" not in command.argv
    assert "kernelgen_server.evaluation.adapters.flaggems.pytest_plugin" not in (
        command.argv
    )
    assert "--candidate-call-count-path" not in command.argv
    assert command.completion_marker_path.endswith("target/runner-completed")
    assert "kernelgen_server.profiling.gems_runner" in command.argv
    assert command.source_roots == (str(tmp_path / "target/candidate"),)
    assert command.env["GEMS_VENDOR"] == "nvidia"
    assert "KGS_FLAGGEMS_OPERATOR" not in command.env
    assert "KGS_FLAGGEMS_CALL_COUNT" not in command.env
    assert (tmp_path / "target/candidate/candidate.py").is_file()


def test_accuracy_candidate_coverage_requires_every_non_skipped_nodeid():
    report = {
        "tests/test_addmm_.py::test_addmm_[first]": {"result": "passed"},
        "tests/test_addmm_.py::test_addmm_[second]": {"result": "failed"},
        "tests/test_addmm_.py::test_addmm_[skipped]": {"result": "skipped"},
    }
    for nodeid in ("first", "second"):
        report[f"tests/test_addmm_.py::test_addmm_[{nodeid}]"]["candidate_calls"] = {"flag_gems.addmm_": 1}

    adapter = create_adapter(_binding())
    adapter._assert_accuracy_call_coverage(report, "addmm_")


def test_timing_candidate_coverage_requires_each_case_id_exactly_once():
    adapter = create_adapter(_binding())
    adapter._assert_timing_call_coverage(
        {_CASE_0},
        [
            {
                "nodeid": "benchmark/test_addmm_.py::test_addmm_",
                "case_id": _CASE_0,
                "count": 1,
            }
        ],
    )

    with pytest.raises(RuntimeError, match="not_called_once"):
        adapter._assert_timing_call_coverage(
            {_CASE_0},
            [
                {
                    "nodeid": "benchmark/test_addmm_.py::test_addmm_",
                    "case_id": _CASE_0,
                    "count": 2,
                }
            ],
        )


@pytest.mark.parametrize("records", [
    [],
    [{"case_id": _CASE_0, "count": 1}, {"case_id": _CASE_0, "count": 1}],
    [{"case_id": _CASE_0, "count": True}],
    [{"case_id": _CASE_0, "count": 0}],
    [{"case_id": "unexpected", "count": 1}],
])
def test_preflight_coverage_rejects_missing_duplicate_or_invalid_cases(records):
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        adapter_module.FlagGemsEvaluationAdapter._assert_timing_call_coverage({_CASE_0}, records)


@pytest.mark.parametrize("patch", [{"status": "failed"}, {"override": False}, {"override": 1}])
def test_preflight_report_requires_successful_override(tmp_path, patch):
    path = tmp_path / "preflight.json"
    record = {"operator": "addmm_", "case_id": _CASE_0, "status": "passed", "override": True, "count": 1}
    record.update(patch)
    path.write_text(json.dumps({"schema_version": "flaggems.preflight/v1", "records": [record]}))
    with pytest.raises(RuntimeError, match="injected candidate"):
        adapter_module.FlagGemsEvaluationAdapter._read_preflight_coverage(path, subprocess.CompletedProcess([], 0), "addmm_")


def test_preflight_report_accepts_current_schema(tmp_path):
    path = tmp_path / "preflight.json"
    records = [{"operator": "addmm_", "case_id": _CASE_0, "status": "passed", "override": True, "count": 1}]
    path.write_text(json.dumps({"schema_version": "flaggems.preflight/v1", "records": records}))
    assert adapter_module.FlagGemsEvaluationAdapter._read_preflight_coverage(path, subprocess.CompletedProcess([], 0), "addmm_") == records


@pytest.mark.parametrize("report", [{}, {"schema_version": "old", "records": []}, {"schema_version": "flaggems.preflight/v1", "records": None}])
def test_preflight_report_rejects_incompatible_schema(tmp_path, report):
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError):
        adapter_module.FlagGemsEvaluationAdapter._read_preflight_coverage(path, subprocess.CompletedProcess([], 0), "addmm_")


def test_empty_accuracy_report_is_not_all_skip():
    with pytest.raises(RuntimeError, match="correctness report contains no cases"):
        adapter_module.FlagGemsEvaluationAdapter._assert_accuracy_call_coverage({}, "addmm_")


@pytest.fixture
def evaluation_reports(monkeypatch, tmp_path):
    adapter = create_adapter(_binding(), device=object(), device_string="cuda:0", backend="cuda")
    assets = SimpleNamespace(
        root=tmp_path,
        correctness=[tmp_path / "tests/test_addmm_.py"],
        performance=[tmp_path / "benchmark/test_addmm_.py"],
        native_operator="addmm_",
    )
    reports = SimpleNamespace(
        adapter=adapter,
        accuracy={"tests/test_addmm_.py::test_op": {"result": "skipped", "reason": "unsupported dtype"}},
        timing_error=None,
        returncode=0,
        write_coverage=True,
        missing_timing=False,
        phases=[],
    )
    monkeypatch.setattr(adapter, "_prepare_workdir", lambda request, work: (_manifest(), assets, work, {}))
    monkeypatch.setattr(adapter, "_candidate_env", lambda *args: {})

    def run_pytest(args, **kwargs):
        output = adapter_module.Path(args[args.index("--output") + 1])
        reports.phases.append(output.stem)
        if output.stem == "accuracy":
            data = reports.accuracy
            if reports.write_coverage:
                for item in data.values():
                    if item["result"] != "skipped":
                        item["candidate_calls"] = {"flag_gems.addmm_": 1}
        else:
            data = {"addmm_": {"details": [{"dtype": "float16", "result": [{
                "case_id": _CASE_0, "candidate_source": "override",
                "latency": 1.0, "latency_base": 2.0, "speedup": 2.0,
                "error_msg": reports.timing_error,
            }]}]}}
            if reports.missing_timing:
                data = {"addmm_": {"result": "failed", "reason": "candidate failed"}}
        output.write_text(json.dumps(data), encoding="utf-8")
        return subprocess.CompletedProcess([], reports.returncode, "", "")

    monkeypatch.setattr(adapter_module, "_run_pytest", run_pytest)
    return reports


@pytest.mark.parametrize("other_outcome, expected_status", [
    ("skipped", "ALL_SKIP"), ("passed", "PASSED"), ("failed", "PARTIAL_PASS"),
])
def test_accuracy_skips_keep_benchmark_execution(evaluation_reports, other_outcome, expected_status):
    reports = evaluation_reports
    reports.accuracy["tests/test_addmm_.py::test_other"] = {"result": other_outcome}
    result = reports.adapter.evaluate(_request())

    assert reports.phases == ["accuracy", "benchmark"]
    assert result.status == expected_status
    assert len(result.per_workload) == 3
    assert result.per_workload[0].metrics["skipped"] is True
    assert result.per_workload[0].log == "unsupported dtype"
    assert result.per_workload[-1].phase == "timing"
    assert result.per_workload[-1].speedup == 2.0
    if expected_status in {"ALL_SKIP", "PASSED"}:
        assert result.geo_mean == 2.0


def test_all_skip_does_not_hide_benchmark_failure(evaluation_reports):
    reports = evaluation_reports
    reports.timing_error = "benchmark failed"
    result = reports.adapter.evaluate(_request())

    assert reports.phases == ["accuracy", "benchmark"]
    assert result.status == "PARTIAL_PASS"
    assert result.per_workload[-1].status == "RUNTIME_ERROR"
    assert result.log == "benchmark failed"


def test_failed_pytest_without_timing_returns_runtime_error(evaluation_reports):
    reports = evaluation_reports
    reports.returncode = 1
    reports.missing_timing = True
    result = reports.adapter.evaluate(_request())
    assert result.status == "RUNTIME_ERROR"
    assert "benchmark" in result.log
    assert result.geo_mean is None


def test_successful_pytest_without_timing_remains_a_contract_error(evaluation_reports):
    evaluation_reports.missing_timing = True
    with pytest.raises(RuntimeError, match="omitted case_id"):
        evaluation_reports.adapter.evaluate(_request())


def test_all_skip_without_candidate_coverage_still_benchmarks(evaluation_reports):
    reports = evaluation_reports
    reports.write_coverage = False
    result = reports.adapter.evaluate(_request())

    assert result.status == "ALL_SKIP"
    assert result.geo_mean == 2.0
    assert reports.phases == ["accuracy", "benchmark"]


def test_all_skip_does_not_hide_pytest_process_failure(evaluation_reports):
    reports = evaluation_reports
    reports.returncode = 1
    with pytest.raises(RuntimeError, match="pytest exited non-zero"):
        reports.adapter.evaluate(_request())


def test_all_skip_is_http_200_and_releases_device(evaluation_reports, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import kernelgen_server.api.app as server

    monkeypatch.setattr(server, "_resolve_backend", lambda backend: "cuda")
    monkeypatch.setattr(server, "_make_device", lambda backend: SimpleNamespace(count_devices_safe=lambda: 1))
    monkeypatch.setattr(server, "configure_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "runtime_device_type", lambda backend: "cuda")
    monkeypatch.setattr(server, "environment_info", lambda *args: {})
    monkeypatch.setattr(server, "probe_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "run_isolated", lambda operation, request, *args: evaluation_reports.adapter.evaluate(request))
    app = server.create_app(
        backend="cuda", enable_debug_jobs=False,
        profile_artifact_root=tmp_path / "profiles", operator_bundle_root=tmp_path / "bundles",
    )
    with TestClient(app) as http:
        response = http.post("/evaluate", json=_request().wire_payload(), headers={"X-KernelGen-Operation-Id": "all-skip-eval"})
        assert response.status_code == 200
        parsed = EvaluateResponse.model_validate(response.json())
        assert parsed.status == "ALL_SKIP" and parsed.geo_mean == 2.0
        assert evaluation_reports.phases == ["accuracy", "benchmark"]
        scheduler = http.get("/status").json()["scheduler"]
        assert scheduler["active"] == scheduler["waiting"] == scheduler["broken"] == 0
        assert scheduler["available"] == scheduler["healthy"] == 1
        assert http.get("/operations/all-skip-eval").json()["state"] == "SUCCEEDED"


@pytest.mark.parametrize('patch', [
    'import torch\ntorch.add = None', 'import flag_gems\nflag_gems.register = None',
    'import torch\ntorch.cuda.empty_cache()',
    'import os\nos.environ["TRITON_PPU_LLC_PATH"] = "/tmp/shim"',
    'from pathlib import Path\nPath("/tmp/shim.py").write_text("shim")',
    'from flag_gems.testing import override_registered_op\nctx = override_registered_op("op", None)\nctx.__enter__()',
])
def test_preflight_admission_rejects_before_candidate_import_or_pytest(monkeypatch, patch):
    adapter = create_adapter(_binding(), device=SimpleNamespace(), device_string='cuda:0', backend='nvidia')
    def forbidden(*args, **kwargs):
        raise AssertionError('rejected candidate must not be imported or executed')
    monkeypatch.setattr(adapter, '_manifest', forbidden)
    monkeypatch.setattr(adapter, '_assets', forbidden)
    monkeypatch.setattr(adapter, '_validate_candidate_abi', forbidden)
    monkeypatch.setattr(adapter_module, '_run_pytest', forbidden)
    result = adapter.preflight(_request(patch + '\ndef run(self, mat1, mat2, *, beta=1, alpha=1): return self'))
    assert result.status in {'FAILED', 'RUNTIME_ERROR'}
    assert result.is_hack
    assert 'candidate_admission' in result.log


@pytest.mark.parametrize('phase', ['evaluate', 'profile'])
def test_evaluate_and_profile_do_not_repeat_admission(monkeypatch, tmp_path, phase):
    adapter = create_adapter(_binding(), device=object(), backend='nvidia')
    def unexpected(*args, **kwargs):
        raise AssertionError('admission belongs only to preflight')
    monkeypatch.setattr(adapter_module, 'require_candidate_admission', unexpected)
    req = _request('import flag_gems\nflag_gems.register = None\ndef run(self, mat1, mat2, *, beta=1, alpha=1): return self')
    if phase == 'evaluate':
        monkeypatch.setattr(adapter, '_assets', lambda: None)
        def import_error(*args):
            raise BuildError('ordinary candidate import failure')
        monkeypatch.setattr(adapter, '_validate_candidate_abi', import_error)
        result = adapter.evaluate(req)
        assert result.status == 'RUNTIME_ERROR' and not result.is_hack
        assert 'ordinary candidate import failure' in result.log
    else:
        class ReachedProfilePreparation(Exception):
            pass
        def preparation():
            raise ReachedProfilePreparation()
        monkeypatch.setattr(adapter, '_manifest', preparation)
        with pytest.raises(ReachedProfilePreparation):
            adapter.build_profile_command(SimpleNamespace(implementation=req.implementation, expected_backend='nvidia'), None, tmp_path, 'cuda:0')

"""Host-only profile adapter tests using V6 binding objects."""

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from kernelgen_server import Definition, EvaluatorBinding, Implementation, SourceFile, Workload
from kernelgen_client.profiling import ProfileResult
from kernelgen_server.schema import (
    AdapterCapabilities,
    AdapterCase,
    AdapterManifest,
    CandidateContract,
    CaseList,
)

from kernelgen.data.ledger import Ledger
from kernelgen.tests.helpers import experiment_plan
from kernelgen.data.tool_context import ToolContext
from kernelgen.framework.run_control import WorkspaceRunControl
from kernelgen.tools.profile_round import (
    EvaluationBundle,
    _find_cached_profile,
    get_profile_context,
    prepare_evaluation_bundle,
    profile_workspace_workloads,
    profile_workloads,
    write_evaluation_snapshot,
)


def test_profile_honors_cancellation_before_configuration(tmp_path):
    WorkspaceRunControl(tmp_path).request_cancel("stop before profile")

    result = profile_workspace_workloads(tmp_path, 1, ["w0"])

    assert result["status"] == "RUN_CANCELLED"
    assert result["stage"] == "BEFORE_PROFILE"


def test_profile_returns_server_cancellation(tmp_path, monkeypatch):
    from kernelgen_client.http import OperationCancelledError
    from kernelgen.tools import profile_round as profile_module

    ToolContext(
        definition="op",
        target_hardware="H100",
        eval_server_url="http://eval:8000",
    ).write(tmp_path)

    def cancelled(*args, run_control, **kwargs):
        run_control.request_cancel("stop active profile")
        raise OperationCancelledError("profile-1", "profile")

    monkeypatch.setattr(profile_module, "profile_workloads", cancelled)

    result = profile_workspace_workloads(tmp_path, 1, ["w0"])

    assert result["status"] == "RUN_CANCELLED"
    assert result["stage"] == "DURING_PROFILE"


def _write_catalog(
    root: Path,
    *,
    correctness_names: list[str],
    timing_names: list[str],
) -> None:
    (root / "definitions").mkdir(parents=True)
    (root / "references").mkdir()
    (root / "workloads").mkdir()
    definition = Definition(
        api_version="v6.0",
        name="op",
        parameters=[{"name": "x", "required": True}],
        outputs=["output"],
    )
    (root / "definitions" / "op.json").write_text(
        definition.model_dump_json(indent=2, exclude_unset=True),
        encoding="utf-8",
    )
    (root / "references" / "op.py").write_text(
        'REFERENCE_DEVICE = "target"\n\ndef run(x):\n    return x\n',
        encoding="utf-8",
    )

    def write_workloads(path: Path, names: list[str]) -> None:
        path.write_text(
            "\n".join(
                Workload(
                    name=name,
                    inputs={"x": {"type": "scalar", "value": 1}},
                ).model_dump_json()
                for name in names
            )
            + ("\n" if names else ""),
            encoding="utf-8",
        )

    write_workloads(
        root / "workloads" / "op.correctness.jsonl", correctness_names
    )
    write_workloads(root / "workloads" / "op.timing.jsonl", timing_names)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "api_version": "v6.0",
                "evaluator": "native",
            }
        ),
        encoding="utf-8",
    )


def _patch_inspect(monkeypatch, timing_names: list[str]) -> None:
    from kernelgen_client import http as client

    monkeypatch.setattr(
        client,
        "inspect",
        lambda request, server_url: AdapterManifest(
            kind="native",
            benchmark_fingerprint="benchmark-fingerprint",
            candidate_contract=CandidateContract(signature="(x)"),
            capabilities=AdapterCapabilities(preflight=True, profile=True),
            case_list=CaseList(
                adapter_kind="native",
                operator="op",
                benchmark_fingerprint="benchmark-fingerprint",
                cases=[
                    AdapterCase(case_id=name, ordinal=index)
                    for index, name in enumerate(timing_names)
                ],
            ),
        ),
    )


@contextmanager
def _fake_profile_modules(call_log, *, device="NVIDIA H100", requests=None):
    from kernelgen.tools import kernelgen_server_adapter
    from kernelgen_client import http as client

    def get_service_status(server_url=None, **kwargs):
        return {
            "status": "ok",
            "api_version": "v6.0",
            "backend": "cuda",
            "timing": "triton",
            "target": {
                "backend": "cuda",
                "vendor": "nvidia",
                "architecture": "hopper",
                "device": device,
            },
            "profile": {
                "supported": True,
                "profiler": "ncu",
                "levels": ["metrics", "source", "instruction"],
                "capabilities": ["performance_counters", "instruction_listing"],
            },
        }

    def profile(request, server_url, *, operation_id=None):
        workload_name = request.case_id
        call_log.append(workload_name)
        if requests is not None:
            requests.append(request)
        return ProfileResult(
            profile_id=f"profile-{workload_name}",
            status="completed",
            backend="cuda",
            profiler="ncu",
            device="cuda:0",
            evaluation_id=request.evaluation_id,
            workload_name=workload_name,
            options=request.options,
            capabilities=["performance_counters"],
            summary={"kernel_count": 1},
            metrics={"sm_throughput_pct": 50},
        )

    old_status = kernelgen_server_adapter.get_service_status
    old_profile = client.profile
    kernelgen_server_adapter.get_service_status = get_service_status
    client.profile = profile
    try:
        yield
    finally:
        kernelgen_server_adapter.get_service_status = old_status
        client.profile = old_profile


def _eval(workload_uuids):
    return {
        "api_version": "v6.0",
        "status": "PASSED",
        "geo_mean": 1.1,
        "min_speedup": 0.9,
        "worst_workload_uuid": workload_uuids[0],
        "latency_ms": 0.2,
        "abs_err": 0.0,
        "rel_err": 0.0,
        "num_workloads": len(workload_uuids),
        "num_passed": len(workload_uuids),
        "server_backend": "cuda",
        "per_workload": [
            {
                "uuid": item,
                "axes": {"M": index + 1},
                "status": "PASSED",
                "speedup": 0.9 + index * 0.1,
                "latency_ms": 0.2,
                "reference_latency_ms": 0.2 * (0.9 + index * 0.1),
            }
            for index, item in enumerate(workload_uuids)
        ],
    }


def _write_snapshot_fixture(
    root: Path,
    workload_uuids,
    *,
    profile_workload_uuids=None,
):
    snapshot = root / ".kernelgen" / "evals" / "round-0001"
    snapshot.mkdir(parents=True)
    identity = {
        "round_num": 1,
        "evaluation_fingerprint": "fingerprint",
        "solution_sha256": "solution-sha",
        "definition_name": "op",
        "definition_sha256": "definition-sha",
        "workload_uuids": workload_uuids,
        "workload_sha256": [f"sha-{item}" for item in workload_uuids],
        "catalog_name": "fixture",
        "adapter_kind": "native",
        "benchmark_fingerprint": "benchmark-fingerprint",
        "target_hardware": "H100",
        "server_backend": "cuda",
    }
    if profile_workload_uuids is not None:
        identity["workload_mode"] = "phased"
        identity["profile_workload_uuids"] = profile_workload_uuids
    payloads = {
        "result.json": _eval(workload_uuids),
        "solution.json": Implementation(
            name="candidate",
            definition="op",
            language="triton",
            entrypoint="main.py::run",
            sources=[
                SourceFile(path="main.py", content="def run(x): return x")
            ],
        ).model_dump(mode="json"),
        "definition.json": Definition(
            name="op",
            parameters=[{"name": "x", "required": True}],
            outputs=["output"],
        ).model_dump(mode="json"),
        "workloads.json": [
            Workload(
                name=item,
                inputs={"x": {"type": "scalar", "value": index + 1}},
            ).model_dump(mode="json")
            for index, item in enumerate(workload_uuids)
        ],
        "identity.json": identity,
    }
    (snapshot / "main.py").write_text("evaluated code", encoding="utf-8")
    for name, payload in payloads.items():
        (snapshot / name).write_text(json.dumps(payload), encoding="utf-8")
    return snapshot


def test_prepare_bundle_keeps_correctness_only_workload(tmp_path, monkeypatch):
    _write_catalog(
        tmp_path,
        correctness_names=["correctness-0"],
        timing_names=[],
    )
    kernel = tmp_path / "main.py"
    kernel.write_text("def run(x): return x", encoding="utf-8")
    monkeypatch.setattr(
        "kernelgen.data.catalog.resolve_builtin_catalog_path",
        lambda catalog_name: tmp_path,
    )
    _patch_inspect(monkeypatch, [])
    context = ToolContext(
        definition="op",
        target_hardware="H100",
        catalog_name="fixture",
        destination_passing_style=False,
    )

    bundle = prepare_evaluation_bundle(kernel, context)

    assert [item.name for item in bundle.workloads] == ["correctness-0"]
    assert bundle.profile_workload_uuids == []
    assert bundle.is_phased is True


def test_prepare_bundle_profiles_only_timing_workloads_in_phased_mode(
    tmp_path,
    monkeypatch,
):
    _write_catalog(
        tmp_path,
        correctness_names=["correctness-0"],
        timing_names=["timing-0"],
    )
    kernel = tmp_path / "main.py"
    kernel.write_text("def run(x): return x", encoding="utf-8")
    monkeypatch.setattr(
        "kernelgen.data.catalog.resolve_builtin_catalog_path",
        lambda catalog_name: tmp_path,
    )
    _patch_inspect(monkeypatch, ["timing-0"])
    context = ToolContext(
        definition="op",
        target_hardware="H100",
        catalog_name="fixture",
        destination_passing_style=False,
    )

    bundle = prepare_evaluation_bundle(kernel, context)

    assert [item.name for item in bundle.workloads] == [
        "correctness-0",
        "timing-0",
    ]
    assert bundle.profile_workload_uuids == ["timing-0"]
    assert bundle.is_phased is True


def test_profile_workloads_has_no_count_cap_and_reuses_manifests(tmp_path):
    workload_uuids = [
        "benchmark/test_op.py::test_op::core::float16::0",
        *[f"u{index}" for index in range(6)],
    ]
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(workload_uuids),
        "evaluated code",
        experiment_plan(1),
        profile_enabled=True,
    )
    snapshot = _write_snapshot_fixture(tmp_path, workload_uuids)
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint="fingerprint",
        snapshot_path=str(snapshot.relative_to(tmp_path)),
    )
    context = ToolContext(
        definition="op",
        target_hardware="H100",
        eval_server_url="http://fake",
        catalog_name="fixture",
    )
    calls = []
    with _fake_profile_modules(calls):
        profile_context = get_profile_context(tmp_path, context, 1)
        first = profile_workloads(tmp_path, context, 1, workload_uuids, level="metrics")
        second = profile_workloads(tmp_path, context, 1, workload_uuids, level="metrics")

    assert profile_context["experiment_plan"]["kind"] == "baseline"
    assert profile_context["per_workload"][0]["reference_latency_ms"] == 0.18000000000000002
    assert profile_context["profile_workload_uuids"] == workload_uuids
    assert first["status"] == "completed"
    assert len(first["profiles"]) == len(workload_uuids)
    assert calls == workload_uuids
    assert all(item["cached"] for item in second["profiles"])
    assert all(not Path(item["manifest_path"]).is_absolute() for item in first["profiles"])
    assert all(not item["artifacts"] for item in first["profiles"])
    assert first["profiles"][0]["workload_uuid"] == workload_uuids[0]
    assert "benchmark/test_op.py" not in first["profiles"][0]["manifest_path"]
    assert Ledger(tmp_path).get_round(1).profile.status == "collecting"


def test_find_cached_profile_skips_failed_results(tmp_path):
    workload_root = (
        tmp_path
        / ".kernelgen"
        / "profiles"
        / "round-0001"
        / "u0"
    )
    options = {"level": "metrics"}

    def write_manifest(directory: str, profile_id: str, status: str) -> None:
        path = workload_root / directory / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "request": {
                        "target": {
                            "solution_sha256": "solution-sha",
                            "workload_sha256": "sha-u0",
                        },
                        "options": options,
                    },
                    "result": {
                        "profile_id": profile_id,
                        "status": status,
                    },
                }
            ),
            encoding="utf-8",
        )

    write_manifest("z-failed", "failed-profile", "failed")
    assert (
        _find_cached_profile(
            tmp_path,
            1,
            "u0",
            "solution-sha",
            "sha-u0",
            options,
        )
        is None
    )

    write_manifest("a-completed", "completed-profile", "completed")
    cached = _find_cached_profile(
        tmp_path,
        1,
        "u0",
        "solution-sha",
        "sha-u0",
        options,
    )

    assert cached is not None
    assert cached["profile_id"] == "completed-profile"
    assert cached["cached"] is True


def test_profile_context_rejects_a_different_server_device(tmp_path):
    from kernelgen.data.target_context import TargetContextError

    workload_uuids = ["u0"]
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(workload_uuids),
        "evaluated code",
        experiment_plan(1),
        profile_enabled=True,
    )
    snapshot = _write_snapshot_fixture(tmp_path, workload_uuids)
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint="fingerprint",
        snapshot_path=str(snapshot.relative_to(tmp_path)),
    )
    context = ToolContext(
        definition="op",
        target_hardware="H100",
        eval_server_url="http://fake",
        catalog_name="fixture",
    )

    with _fake_profile_modules([], device="NVIDIA A100"):
        with pytest.raises(TargetContextError, match="TARGET_HARDWARE_MISMATCH"):
            get_profile_context(tmp_path, context, 1)


def test_phased_profile_rejects_correctness_workload_and_profiles_timing(tmp_path):
    workload_uuids = ["correctness-0", "timing-0"]
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(workload_uuids),
        "evaluated code",
        experiment_plan(1),
        profile_enabled=True,
    )
    snapshot = _write_snapshot_fixture(
        tmp_path,
        workload_uuids,
        profile_workload_uuids=["timing-0"],
    )
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint="fingerprint",
        snapshot_path=str(snapshot.relative_to(tmp_path)),
    )
    context = ToolContext(
        definition="op",
        target_hardware="H100",
        eval_server_url="http://fake",
        catalog_name="fixture",
    )
    calls = []
    with _fake_profile_modules(calls):
        profile_context = get_profile_context(tmp_path, context, 1)
        with pytest.raises(ValueError, match="not timing/profile candidates"):
            profile_workloads(tmp_path, context, 1, ["correctness-0"])
        result = profile_workloads(tmp_path, context, 1, ["timing-0"])

    assert profile_context["profile_workload_uuids"] == ["timing-0"]
    assert result["status"] == "completed"
    assert calls == ["timing-0"]


@pytest.mark.parametrize("source", [None, {"catalog_name": "actual-catalog"}, {"bundle_id": "sha256:" + "a" * 64}])
def test_snapshot_keeps_evaluated_code_and_full_identity(tmp_path, source):
    workload_uuids = ["u0", "u1"]
    definition = Definition(
        name="op",
        parameters=[{"name": "x", "required": True}],
        outputs=["output"],
    )
    solution = Implementation(
        name="candidate",
        definition="op",
        language="triton",
        entrypoint="main.py::run",
        sources=[SourceFile(path="main.py", content="evaluated code")],
    )
    workloads = [
        Workload(
            name=item,
            inputs={"x": {"type": "scalar", "value": index + 1}},
        )
        for index, item in enumerate(workload_uuids)
    ]
    bundle = EvaluationBundle(
        kernel_code="evaluated code",
        solution=solution,
        definition=definition,
        workloads=workloads,
        is_phased=True,
        profile_workload_uuids=["u1"],
        adapter_kind="native",
        benchmark_fingerprint="benchmark-fingerprint",
        binding=EvaluatorBinding(definition="op", **source) if source else None,
    )
    context = ToolContext(
        definition="op",
        target_hardware="H100",
        catalog_name="fixture",
    )
    with _fake_profile_modules([]):
        written = write_evaluation_snapshot(tmp_path, 1, bundle, _eval(workload_uuids), context)

    candidate = tmp_path / "tmp" / "main.py"
    candidate.parent.mkdir()
    candidate.write_text("later code", encoding="utf-8")
    snapshot = tmp_path / written["snapshot_path"]
    identity = json.loads((snapshot / "identity.json").read_text(encoding="utf-8"))
    assert (snapshot / "main.py").read_text(encoding="utf-8") == "evaluated code"
    assert identity["workload_uuids"] == workload_uuids
    assert identity["workload_mode"] == "phased"
    assert identity["profile_workload_uuids"] == ["u1"]
    assert identity["server_backend"] == "cuda"
    assert identity["binding"] == {"definition": "op", **(source or {"catalog_name": "fixture"})}
    assert written["evaluation_fingerprint"] == identity["evaluation_fingerprint"]

    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(workload_uuids), "evaluated code", experiment_plan(1), profile_enabled=True)
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint=written["evaluation_fingerprint"],
        snapshot_path=written["snapshot_path"],
    )
    requests = []
    with _fake_profile_modules([], requests=requests):
        profile_workloads(tmp_path, context, 1, ["u1"])
    assert requests[0].binding.model_dump(exclude_none=True) == identity["binding"]
    assert requests[0].implementation == solution
    assert requests[0].case_id == "u1"
    assert requests[0].benchmark_fingerprint == "benchmark-fingerprint"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            with tempfile.TemporaryDirectory() as directory:
                test(Path(directory))
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception:
            import traceback

            print(f"  ✗ {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    raise SystemExit(0 if passed == len(tests) else 1)

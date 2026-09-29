import json

from kernelgen_server import (
    Definition,
    Implementation,
    SourceFile,
    Workload,
)
from kernelgen_server.schema import (
    AdapterCapabilities,
    AdapterCase,
    AdapterManifest,
    CandidateContract,
    CaseList,
    EvaluateResponse,
    EvaluationStatus,
    WorkloadResult,
    WorkloadStatus,
)

from kernelgen.data.tool_context import ToolContext
from kernelgen.data.trace import load_catalog_optimization_context
from kernelgen.framework.run_control import WorkspaceRunControl
from kernelgen.tools.eval_only import evaluate_only
from kernelgen.tools.kernelgen_server_adapter import (
    EvaluationBundle,
    _transport_timeout_from_context,
    build_evaluate_request,
    hardware_family,
    tracked_server_operation,
)


def test_c500_maps_to_metax_hardware_family():
    assert hardware_family("C500") == "metax"
    assert hardware_family("MetaX C500") == "metax"


def test_tracked_server_operation_uses_advertised_capability(tmp_path):
    control = WorkspaceRunControl(tmp_path)
    status = {
        "capabilities": {
            "operation_cancel": {
                "enabled": True,
                "operations": ["evaluate"],
            }
        }
    }

    with tracked_server_operation(
        control,
        status,
        "http://127.0.0.1:18000",
        "evaluate",
    ) as operation_id:
        active = control.active_server_operations()
        assert operation_id is not None
        assert [operation.operation_id for operation in active] == [operation_id]

    assert control.active_server_operations() == []


def test_tracked_server_operation_is_disabled_without_capability(tmp_path):
    control = WorkspaceRunControl(tmp_path)

    with tracked_server_operation(
        control,
        {},
        "http://127.0.0.1:18000",
        "evaluate",
    ) as operation_id:
        assert operation_id is None

    assert control.active_server_operations() == []


def _catalog(root):
    (root / "definitions").mkdir()
    (root / "references").mkdir()
    (root / "workloads").mkdir()
    definition = Definition(
        api_version="v6.0",
        name="identity",
        parameters=[{"name": "x", "required": True}],
        outputs=["output"],
    )
    (root / "definitions" / "identity.json").write_text(
        definition.model_dump_json(exclude_unset=True),
        encoding="utf-8",
    )
    (root / "references" / "identity.py").write_text(
        'REFERENCE_DEVICE = "target"\n\ndef run(x):\n    return x\n',
        encoding="utf-8",
    )
    correctness = Workload(
        name="corr-0",
        inputs={"x": {"type": "scalar", "value": 1}},
    )
    timing = Workload(
        name="time-0",
        inputs={"x": {"type": "scalar", "value": 2}},
    )
    (root / "workloads" / "identity.correctness.jsonl").write_text(
        correctness.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (root / "workloads" / "identity.timing.jsonl").write_text(
        timing.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "api_version": "v6.0",
                "evaluator": "native",
            }
        ),
        encoding="utf-8",
    )


def test_eval_only_submits_bound_v6_request_and_returns_server_result_directly(
    tmp_path,
    monkeypatch,
):
    from kernelgen_client import http as client
    from kernelgen.data import catalog as catalog_module

    _catalog(tmp_path)
    monkeypatch.setattr(
        catalog_module,
        "resolve_builtin_catalog_path",
        lambda catalog_name: tmp_path,
    )
    kernel = tmp_path / "main.py"
    kernel.write_text("def run(x): return x", encoding="utf-8")
    requests = []
    monkeypatch.setattr(
        client,
        "status",
        lambda server_url, timeout=10: {
            "status": "ok",
            "api_version": "v6.0",
            "backend": "cuda",
            "timing": "triton",
            "target": {
                "backend": "cuda",
                "vendor": "nvidia",
                "architecture": "ampere",
                "device": "NVIDIA A100-SXM4-40GB",
            },
        },
    )
    monkeypatch.setattr(
        client,
        "inspect",
        lambda request, server_url: AdapterManifest(
            kind="native",
            benchmark_fingerprint="fixture-fingerprint",
            candidate_contract=CandidateContract(signature="(x)"),
            capabilities=AdapterCapabilities(preflight=True, profile=True),
            case_list=CaseList(
                adapter_kind="native",
                operator="identity",
                benchmark_fingerprint="fixture-fingerprint",
                cases=[AdapterCase(case_id="time-0", ordinal=0)],
            ),
        ),
    )

    def evaluate(request, server_url, timeout=None, *, operation_id=None):
        requests.append((request, server_url, timeout))
        return EvaluateResponse(
            status=EvaluationStatus.PASSED,
            device="cuda:0",
            server_backend="cuda",
            geo_mean=2.0,
            min_speedup=2.0,
            worst_workload_uuid="time-0",
            latency_ms=1.0,
            abs_err=0.0,
            rel_err=0.0,
            num_workloads=2,
            num_passed=2,
            per_workload=[
                WorkloadResult(
                    uuid="corr-0",
                    phase="correctness",
                    status=WorkloadStatus.PASSED,
                    abs_err=0.0,
                    rel_err=0.0,
                ),
                WorkloadResult(
                    uuid="time-0",
                    phase="timing",
                    status=WorkloadStatus.PASSED,
                    speedup=2.0,
                    latency_ms=1.0,
                    reference_latency_ms=2.0,
                ),
            ],
        )

    monkeypatch.setattr(client, "evaluate", evaluate)
    result = evaluate_only(
        kernel_path=str(kernel),
        definition="identity",
        catalog_name="fixture",
        target_hardware="A100",
        server_url="http://server:8000",
        warmup_ms=5,
        benchmark_ms=7,
    )

    assert result["status"] == "PASSED"
    assert result["is_hack"] is False
    assert result["hack_reason"] == ""
    assert result["geo_mean"] == 2.0
    assert result["per_workload"][1]["uuid"] == "time-0"
    assert len(requests) == 1
    request, server_url, transport_timeout = requests[0]
    assert server_url == "http://server:8000"
    assert transport_timeout == 690
    assert request.binding.catalog_name == "fixture"
    assert request.binding.definition == "identity"
    assert request.implementation.entrypoint == "main.py::run"
    assert request.settings.warmup_ms == 5
    assert request.settings.benchmark_ms == 7


def test_bound_evaluate_request_uses_effective_tolerance_settings():
    context = ToolContext(
        definition="identity",
        target_hardware="A100",
        catalog_name="fixture",
        eval_tolerance_mode="fixed",
        eval_atol=0.123,
        eval_rtol=0.456,
    )
    bundle = EvaluationBundle(
        kernel_code="def run(x): return x",
        solution=Implementation(
            name="candidate",
            definition="identity",
            language="triton",
            entrypoint="main.py::run",
            sources=[SourceFile(path="main.py", content="def run(x): return x")],
        ),
        definition=Definition(
            name="identity",
            parameters=[{"name": "x", "required": True}],
            outputs=["output"],
            reference="def run(x): return x",
        ),
        workloads=[],
    )

    request = build_evaluate_request(bundle, context)

    assert request.settings.tolerance_mode == "fixed"
    assert request.settings.atol == 0.123
    assert request.settings.rtol == 0.456
    assert request.binding.catalog_name == "fixture"
    assert request.binding.definition == "identity"


def test_v6_native_catalog_builds_coder_prompt_context(tmp_path):
    _catalog(tmp_path)

    definition, workloads = load_catalog_optimization_context(
        tmp_path,
        "identity",
    )

    assert definition.name == "identity"
    assert definition.inputs == {
        "x": {"shape": "dynamic", "dtype": "dynamic"}
    }
    assert definition.outputs == {
        "output": {"shape": "dynamic", "dtype": "dynamic"}
    }
    assert definition.run_signature == "(x)"
    assert [item["uuid"] for item in workloads] == ["corr-0", "time-0"]
    assert [item["phase"] for item in workloads] == [
        "correctness",
        "timing",
    ]


def test_v6_catalog_preserves_stable_workload_input_dtype(tmp_path):
    _catalog(tmp_path)
    correctness = Workload(
        name="corr-0",
        inputs={
            "x": {
                "type": "random",
                "shape": [8],
                "dtype": "float16",
            }
        },
    )
    timing = Workload(
        name="time-0",
        inputs={
            "x": {
                "type": "random",
                "shape": [64],
                "dtype": "float16",
            }
        },
    )
    (tmp_path / "workloads" / "identity.correctness.jsonl").write_text(
        correctness.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (tmp_path / "workloads" / "identity.timing.jsonl").write_text(
        timing.model_dump_json() + "\n",
        encoding="utf-8",
    )

    definition, _ = load_catalog_optimization_context(
        tmp_path,
        "identity",
    )

    assert definition.inputs == {
        "x": {"shape": "dynamic", "dtype": "float16"}
    }
    assert definition.outputs == {
        "output": {"shape": "dynamic", "dtype": "dynamic"}
    }


def test_catalog_manifest_group_is_preserved_for_per_operator_layout(tmp_path):
    _catalog(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["operators"] = [{"name": "identity", "group": "pointwise"}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    definition, _ = load_catalog_optimization_context(tmp_path, "identity")

    assert definition.op_type == "pointwise"


def test_v6_catalog_preserves_keyword_only_run_signature(tmp_path):
    _catalog(tmp_path)
    definition_path = tmp_path / "definitions" / "identity.json"
    payload = json.loads(definition_path.read_text(encoding="utf-8"))
    payload["parameters"] = [
        {"name": "self", "required": True},
        {"name": "mat1", "required": True},
        {"name": "mat2", "required": True},
        {
            "name": "beta",
            "kind": "keyword_only",
            "required": False,
            "default": 1,
        },
        {
            "name": "alpha",
            "kind": "keyword_only",
            "required": False,
            "default": 1,
        },
    ]
    definition_path.write_text(json.dumps(payload), encoding="utf-8")

    definition, _ = load_catalog_optimization_context(tmp_path, "identity")

    assert definition.run_signature == "(self, mat1, mat2, *, beta=1, alpha=1)"


def test_simple_opt_transport_timeout_uses_explicit_single_request_budget():
    class Context:
        eval_timeout_seconds = 600
        eval_transport_timeout_seconds = 1200

    assert _transport_timeout_from_context(Context()) == 1200

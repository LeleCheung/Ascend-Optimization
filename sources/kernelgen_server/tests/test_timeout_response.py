from __future__ import annotations

from kernelgen_server import (
    BoundEvaluateRequest,
    EvaluatorBinding,
    Implementation,
    SourceFile,
)
from kernelgen_server.schema import (
    AdapterCapabilities,
    AdapterManifest,
    CandidateContract,
    CaseList,
    EvaluationStatus,
    InspectRequest,
)
from kernelgen_server.protocol import client
from kernelgen_server.protocol.responses import (
    suspected_device_error_response,
    timeout_response,
)


def _request() -> BoundEvaluateRequest:
    return BoundEvaluateRequest(
        binding=EvaluatorBinding(
            catalog_name="simple-v6-test",
            definition="identity",
        ),
        implementation=Implementation(
            name="candidate",
            definition="identity",
            language="python",
            entrypoint="main.py::run",
            sources=[
                SourceFile(
                    path="main.py",
                    content="def run(x): return x",
                )
            ],
        ),
    )


def test_evaluate_timeout_is_a_structured_response():
    result = timeout_response(
        "evaluate",
        _request(),
        backend="cuda",
        device_string="cuda:0",
        message="evaluate timed out after 600s on cuda:0",
    )

    assert result.status == EvaluationStatus.TIMEOUT
    assert result.device == "cuda:0"
    assert result.server_backend == "cuda"
    assert result.num_workloads == 0
    assert result.num_passed == 0
    assert result.geo_mean is None
    assert "timed out after 600s" in result.log


def test_preflight_timeout_is_a_structured_response():
    result = timeout_response(
        "preflight",
        _request(),
        backend="npu",
        device_string="npu:0",
        message="preflight timed out after 600s on npu:0",
    )

    assert result.status == "TIMEOUT"
    assert result.stage == "isolated_execution"
    assert result.log == "preflight timed out after 600s on npu:0"


def test_evaluate_suspected_device_error_is_a_structured_response():
    result = suspected_device_error_response(
        "evaluate",
        _request(),
        backend="musa",
        device_string="musa:1",
        message="strong device probe failed; slot quarantined",
    )

    assert result.status == EvaluationStatus.SUSPECTED_DEVICE_ERROR
    assert result.device == "musa:1"
    assert result.server_backend == "musa"
    assert result.num_workloads == 0
    assert result.num_passed == 0
    assert "slot quarantined" in result.log


def test_preflight_suspected_device_error_is_a_structured_response():
    result = suspected_device_error_response(
        "preflight",
        _request(),
        backend="musa",
        device_string="musa:1",
        message="strong device probe failed; slot quarantined",
    )

    assert result.status == "SUSPECTED_DEVICE_ERROR"
    assert result.stage == "device_probe"
    assert result.log == "strong device probe failed; slot quarantined"


def test_evaluate_client_accepts_structured_timeout(monkeypatch):
    payload = timeout_response(
        "evaluate",
        _request(),
        backend="cuda",
        device_string="cuda:0",
        message="evaluate timed out after 600s on cuda:0",
    ).model_dump(mode="json")
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *args, **kwargs: payload,
    )

    result = client.evaluate(
        _request(),
        "http://server",
        timeout=1200,
    )

    assert result.status == EvaluationStatus.TIMEOUT
    assert result.log == "evaluate timed out after 600s on cuda:0"


def test_evaluate_client_leaves_time_for_the_recovery_probe(monkeypatch):
    payload = timeout_response(
        "evaluate",
        _request(),
        backend="cuda",
        device_string="cuda:0",
        message="evaluate timed out",
    ).model_dump(mode="json")
    observed = {}

    def fake_request(*args, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        return payload

    monkeypatch.setattr(client, "_request_json", fake_request)
    request = _request()

    client.evaluate(request, "http://server")

    assert observed["timeout"] == (
        client.EVALUATION_MAX_ATTEMPTS * request.settings.timeout_seconds
        + client.EVALUATION_TRANSPORT_GRACE_SECONDS
    )


def test_inspect_client_allows_slow_case_listing(monkeypatch):
    observed = {}

    def fake_post(server_url, path, payload, timeout):
        observed.update({
            "server_url": server_url,
            "path": path,
            "timeout": timeout,
        })
        return AdapterManifest(
            kind="flaggems",
            benchmark_fingerprint="fixture-fingerprint",
            candidate_contract=CandidateContract(signature="(x)"),
            capabilities=AdapterCapabilities(),
            case_list=CaseList(
                adapter_kind="flaggems",
                operator="identity",
                benchmark_fingerprint="fixture-fingerprint",
            ),
        ).model_dump(mode="json")

    monkeypatch.setattr(client, "_post", fake_post)
    request = InspectRequest(
        binding=EvaluatorBinding(
            catalog_name="fixture",
            definition="identity",
        )
    )

    manifest = client.inspect(request, "http://server")

    assert manifest.kind == "flaggems"
    assert observed == {
        "server_url": "http://server",
        "path": "/inspect",
        "timeout": 600,
    }

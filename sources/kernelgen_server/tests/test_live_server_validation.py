import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelgen_server import Definition, EvaluationSettings, Workload
from kernelgen_server.protocol.workload_call import definition_signature
from kernelgen_server.client import ServerError
from kernelgen_server.schema import EvaluationStatus


MODULE_PATH = Path(__file__).with_name("live_server_validation.py")
SPEC = importlib.util.spec_from_file_location("live_server_validation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
LIVE_VALIDATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIVE_VALIDATION)


class _Evaluation:
    def __init__(self, status=EvaluationStatus.PASSED, device="cuda:0"):
        self.status = status
        self.device = device

    def model_dump(self, *, mode):
        assert mode == "json"
        return {"status": self.status.value, "device": self.device}


def _operator(definition, *, correctness=True, timing=True):
    return SimpleNamespace(
        definition=definition,
        correctness_workloads=(
            [Workload(name="correctness", inputs={"x": 1})]
            if correctness
            else []
        ),
        timing_workloads=(
            [Workload(name="timing", inputs={"x": 1})] if timing else []
        ),
    )


def test_reference_check_sends_one_bound_request_without_client_workloads(monkeypatch):
    definition = Definition(
        api_version="v6.0",
        name="identity",
        parameters=[{"name": "x", "required": True}],
        outputs=["output"],
        reference="def run(x): return x\n",
    )
    catalog = SimpleNamespace(load=lambda name: _operator(definition))
    requests = []

    def fake_preflight(request, server):
        requests.append(("preflight", request, server))
        return {"status": EvaluationStatus.PASSED.value}

    def fake_evaluate(request, server):
        requests.append(("evaluate", request, server))
        return _Evaluation()

    monkeypatch.setattr(LIVE_VALIDATION, "preflight", fake_preflight)
    monkeypatch.setattr(LIVE_VALIDATION, "evaluate", fake_evaluate)

    _, result = LIVE_VALIDATION._reference_check(
        catalog,
        "kernelgenbench",
        definition.name,
        EvaluationSettings(),
        "http://server",
        True,
    )

    assert result["passed"] is True
    assert result["selected_source"] == "primary"
    assert [operation for operation, _, _ in requests] == [
        "preflight",
        "evaluate",
    ]
    for _, request, server in requests:
        assert server == "http://server"
        assert request.binding.catalog_name == "kernelgenbench"
        assert request.binding.definition == definition.name
        assert request.implementation.sources[0].content == definition.reference
        assert not hasattr(request, "correctness_workloads")
        assert not hasattr(request, "timing_workloads")


def test_v62_reference_candidate_aliases_the_requested_oracle_role():
    definition = Definition(
        api_version="v6.2",
        name="identity",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference=(
            "def correctness_run(x): return x\n"
            "def timing_run(x): return x\n"
        ),
    )

    correctness = LIVE_VALIDATION._reference_candidate_source(
        definition, "correctness"
    )
    timing = LIVE_VALIDATION._reference_candidate_source(definition, "timing")

    assert correctness.endswith("run = correctness_run\n")
    assert timing.endswith("run = timing_run\n")


def test_v62_reference_candidate_keeps_shared_run_without_alias():
    definition = Definition(
        api_version="v6.2",
        name="identity",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference="def run(x): return x\n",
    )

    correctness = LIVE_VALIDATION._reference_candidate_source(
        definition, "correctness"
    )
    timing = LIVE_VALIDATION._reference_candidate_source(definition, "timing")

    assert correctness == definition.reference
    assert timing == definition.reference


def test_reference_check_retries_whole_request_with_torch_fallback(monkeypatch):
    definition = Definition(
        api_version="v6.2",
        name="identity",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference=(
            "def correctness_run(x): return x\n"
            "def timing_run(x): return x\n"
            "def torch_run(x): return x\n"
        ),
    )
    catalog = SimpleNamespace(load=lambda name: _operator(definition))
    seen_sources = []

    def fake_preflight(request, server):
        source = request.implementation.sources[0].content
        seen_sources.append(("preflight", source))
        return {
            "status": (
                EvaluationStatus.RUNTIME_ERROR.value
                if source.endswith("run = correctness_run\n")
                else EvaluationStatus.PASSED.value
            )
        }

    def fake_evaluate(request, server):
        seen_sources.append(("evaluate", request.implementation.sources[0].content))
        return _Evaluation()

    monkeypatch.setattr(LIVE_VALIDATION, "preflight", fake_preflight)
    monkeypatch.setattr(LIVE_VALIDATION, "evaluate", fake_evaluate)

    _, result = LIVE_VALIDATION._reference_check(
        catalog,
        "catalog",
        definition.name,
        EvaluationSettings(),
        "http://server",
        True,
    )

    assert result["passed"] is True
    assert result["selected_source"] == "torch_fallback"
    assert [operation for operation, _ in seen_sources] == [
        "preflight",
        "preflight",
        "evaluate",
    ]
    assert seen_sources[-1][1].endswith("run = torch_run\n")


def test_failure_source_preserves_varargs_and_keyword_only_abi():
    definition = Definition(
        api_version="v6.2",
        name="variadic",
        parameters=[
            {"name": "x", "kind": "positional_only", "required": True},
            {"name": "tensors", "kind": "var_positional", "required": True},
            {
                "name": "alpha",
                "kind": "keyword_only",
                "required": False,
                "default": 1,
            },
        ],
        outputs=["out"],
        reference="def run(x, /, *tensors, alpha=1): return x\n",
    )
    namespace = {}

    exec(
        LIVE_VALIDATION._failure_source(definition, "raise RuntimeError('boom')"),
        namespace,
    )

    assert inspect.signature(namespace["run"]) == definition_signature(definition)
    with pytest.raises(RuntimeError, match="boom"):
        namespace["run"](1, 2, alpha=3)


def test_idle_scheduler_requires_every_slot_available_and_healthy():
    scheduler = {
        "device_slots": 2,
        "healthy": 2,
        "available": 2,
        "active": 0,
        "waiting": 0,
        "checking": 0,
        "broken": 0,
        "max_active": 2,
    }

    assert LIVE_VALIDATION._idle_scheduler(scheduler, 2) is True
    assert (
        LIVE_VALIDATION._idle_scheduler({**scheduler, "checking": 1}, 2)
        is False
    )
    assert (
        LIVE_VALIDATION._idle_scheduler({**scheduler, "available": 1}, 2)
        is False
    )


def test_resilience_check_uses_bound_requests_and_requires_probe_recovery(
    monkeypatch,
):
    definition = Definition(
        api_version="v6.2",
        name="identity",
        parameters=[{"name": "x", "required": True}],
        outputs=["out"],
        reference="def run(x): return x\n",
    )
    catalog = SimpleNamespace(load=lambda name: _operator(definition))
    requests = []

    def fake_evaluate(request, server):
        requests.append(request)
        name = request.implementation.name
        if name.endswith("concurrent-0"):
            return _Evaluation(EvaluationStatus.RUNTIME_ERROR, "cuda:0")
        if name.endswith("concurrent-1"):
            return _Evaluation(EvaluationStatus.PASSED, "cuda:1")
        if name.endswith("hard-crash"):
            raise ServerError("HTTP 500: isolated worker exited with code 17")
        assert name.endswith("recovery")
        return _Evaluation(EvaluationStatus.PASSED, "cuda:0")

    def scheduler(*, incidents, recovered):
        return {
            "device_slots": 2,
            "healthy": 2,
            "available": 2,
            "active": 0,
            "waiting": 0,
            "checking": 0,
            "broken": 0,
            "max_active": 2,
            "incidents": incidents,
            "recovered": recovered,
        }

    statuses = iter(
        [
            {"scheduler": scheduler(incidents=3, recovered=2)},
            {"scheduler": scheduler(incidents=4, recovered=3)},
            {"scheduler": scheduler(incidents=4, recovered=3)},
        ]
    )
    monkeypatch.setattr(LIVE_VALIDATION, "evaluate", fake_evaluate)
    monkeypatch.setattr(LIVE_VALIDATION, "_status", lambda server: next(statuses))

    result = LIVE_VALIDATION._concurrency_and_failure_checks(
        catalog,
        "catalog",
        definition.name,
        "primary",
        EvaluationSettings(),
        "http://server",
        2,
        2,
    )

    assert result["passed"] is True
    assert result["hard_crash_slot_recovered"] is True
    assert result["final_scheduler_idle"] is True
    assert result["unique_devices"] == ["cuda:0", "cuda:1"]
    assert len(requests) == 4
    assert all(request.binding.catalog_name == "catalog" for request in requests)
    assert all(request.binding.definition == definition.name for request in requests)

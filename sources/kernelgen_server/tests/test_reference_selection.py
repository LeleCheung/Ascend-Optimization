from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import kernelgen_server.evaluation.engine as engine_module
from kernelgen_server.evaluation.engine import EvaluationEngine
from kernelgen_server.evaluation.loader import BuildError
from kernelgen_server.schema import (
    Definition,
    EvaluateRequest,
    EvaluationStatus,
    Implementation,
    SourceFile,
    Workload,
)


class _Device:
    def set_device(self, device: str) -> None:
        return None

    def synchronize(self, device: str) -> None:
        return None


def _request() -> EvaluateRequest:
    return EvaluateRequest(
        definition=Definition(
            api_version="v6.2",
            name="identity",
            parameters=[{"name": "x", "required": True}],
            outputs=["out"],
            reference=(
                "def correctness_run(x): return x\n"
                "def timing_run(x): return x\n"
                "def torch_run(x): return x\n"
            ),
        ),
        implementation=Implementation(
            name="candidate",
            definition="identity",
            language="python",
            entrypoint="main.py::run",
            sources=[SourceFile(path="main.py", content="def run(x): return x")],
        ),
        correctness_workloads=[
            Workload(name="correctness-0", inputs={"x": 1}),
            Workload(name="correctness-1", inputs={"x": 2}),
        ],
        timing_workloads=[Workload(name="timing-0", inputs={"x": 3})],
    )


def test_primary_readiness_failure_restarts_whole_request_on_fallback(monkeypatch):
    primary = SimpleNamespace(reference_source="primary")
    fallback = SimpleNamespace(reference_source="torch_fallback")
    loads: list[str] = []
    smokes: list[str] = []

    def fake_load(definition, reference_source="primary", oracle_path=None):
        del definition, oracle_path
        loads.append(reference_source)
        return primary if reference_source == "primary" else fallback

    def fake_smoke(self, operator, request):
        smokes.append(operator.reference_source)
        if operator is primary:
            raise RuntimeError("late primary dtype failure")

    monkeypatch.setattr(engine_module, "load_operator_adapter", fake_load)
    monkeypatch.setattr(EvaluationEngine, "_smoke_reference", fake_smoke)

    selected = EvaluationEngine(_Device(), "target:0")._select_reference(_request())

    assert selected is fallback
    assert loads == ["primary", "torch_fallback", "torch_fallback"]
    assert smokes == ["primary", "torch_fallback"]


def test_v60_reference_failure_never_attempts_fallback(monkeypatch):
    request = _request().model_copy(
        update={
            "api_version": "v6.0",
            "definition": _request().definition.model_copy(
                update={"api_version": "v6.0"}
            ),
        }
    )
    loads: list[str] = []

    def fake_load(definition, reference_source="primary", oracle_path=None):
        del definition, oracle_path
        loads.append(reference_source)
        raise BuildError("bad v6.0 reference")

    monkeypatch.setattr(engine_module, "load_operator_adapter", fake_load)

    with pytest.raises(BuildError, match="bad v6.0 reference"):
        EvaluationEngine(_Device(), "target:0")._select_reference(request)
    assert loads == ["primary"]


def test_candidate_load_error_does_not_trigger_fallback(monkeypatch):
    primary = SimpleNamespace(reference_source="primary")
    loads: list[str] = []

    def fake_load(definition, reference_source="primary", oracle_path=None):
        del definition, oracle_path
        loads.append(reference_source)
        return primary

    monkeypatch.setattr(engine_module, "load_operator_adapter", fake_load)
    monkeypatch.setattr(EvaluationEngine, "_smoke_reference", lambda *args: None)
    monkeypatch.setattr(
        engine_module,
        "load_implementation",
        lambda *args: (_ for _ in ()).throw(BuildError("candidate ABI failure")),
    )

    result = EvaluationEngine(_Device(), "target:0").evaluate(_request())

    assert result.status == EvaluationStatus.RUNTIME_ERROR
    assert result.reference_source == "primary"
    assert loads == ["primary", "primary"]
    assert "candidate ABI failure" in result.log


def test_failed_primary_and_fallback_readiness_returns_runtime_error(monkeypatch):
    loads: list[str] = []

    def fake_load(definition, reference_source="primary", oracle_path=None):
        del definition, oracle_path
        loads.append(reference_source)
        raise BuildError(f"{reference_source} unavailable")

    monkeypatch.setattr(engine_module, "load_operator_adapter", fake_load)

    result = EvaluationEngine(_Device(), "target:0").evaluate(_request())

    assert result.status == EvaluationStatus.RUNTIME_ERROR
    assert result.reference_source == "primary"
    assert loads == ["primary", "torch_fallback"]
    assert "Torch fallback readiness failed" in result.log

import json

import pytest

pytest.importorskip("torch")

from kernelgen_server.evaluation.engine import EvaluationEngine
from kernelgen_server.runtime.device import Device
from kernelgen_server.schema import (
    Definition,
    EvaluateRequest,
    EvaluationStatus,
    Implementation,
    SourceFile,
    Workload,
    WorkloadStatus,
)


class CpuTestDevice(Device):
    def set_device(self, device: str) -> None:
        return None

    def synchronize(self, device: str) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def is_available(self) -> bool:
        return True

    def list_devices(self) -> list[str]:
        return ["cpu"]

    def device_name(self, device: str) -> str:
        return "CPU"


REFERENCE = """
import torch

def run(x):
    return {"values": [x + 1, (x + 2,)]}
"""


def request(candidate_source: str) -> EvaluateRequest:
    return EvaluateRequest(
        api_version="v6.0",
        definition=Definition(
            api_version="v6.0",
            name="nested",
            parameters=[
                {
                    "name": "x",
                    "kind": "positional_or_keyword",
                    "required": True,
                }
            ],
            outputs=["values"],
            reference=REFERENCE,
            reference_device="cpu",
        ),
        implementation=Implementation(
            name="candidate",
            definition="nested",
            language="python",
            entrypoint="main.py::run",
            sources=[SourceFile(path="main.py", content=candidate_source)],
        ),
        correctness_workloads=[
            Workload(
                name="default",
                inputs={
                    "x": {
                        "type": "random",
                        "shape": [4],
                        "dtype": "float32",
                    }
                },
            )
        ],
    )


def test_nested_pytree_passes():
    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(
        request('def run(x): return {"values": [x + 1, (x + 2,)]}')
    )
    assert result.status == EvaluationStatus.PASSED


def test_nested_pytree_detects_error():
    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(
        request('def run(x): return {"values": [x + 1, (x + 3,)]}')
    )
    assert result.status == EvaluationStatus.INCORRECT_NUMERICAL


def test_non_finite_error_is_json_safe():
    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(
        request(
            'def run(x):\n'
            '    import torch\n'
            '    bad = torch.full_like(x, float("nan"))\n'
            '    return {"values": [bad, (bad,)]}\n'
        )
    )

    item = result.per_workload[0]
    assert item.status == WorkloadStatus.INCORRECT_NUMERICAL
    assert item.abs_err is None
    assert item.rel_err is None
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_evaluate_does_not_repeat_preflight_source_checks(monkeypatch):
    import kernelgen_server.evaluation.engine as module
    def unexpected(*args, **kwargs):
        raise AssertionError("source checks belong only to preflight")
    monkeypatch.setattr(module, "check_candidate_admission", unexpected)
    monkeypatch.setattr(module, "log_graylist_hits", unexpected)
    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(
        request('def run(x): return {"values": [x + 1, (x + 2,)]}')
    )
    assert result.status == EvaluationStatus.PASSED


def test_preflight_still_rejects_before_candidate_import(monkeypatch):
    import kernelgen_server.evaluation.engine as module
    def unexpected(*args, **kwargs):
        raise AssertionError("rejected candidate must not be imported")
    monkeypatch.setattr(module, "load_implementation", unexpected)
    result = EvaluationEngine(CpuTestDevice(), "cpu").preflight(
        request('import torch\ntorch.add = None\ndef run(x): return x')
    )
    assert result['status'] == 'FAILED' and result['stage'] == 'candidate_admission'

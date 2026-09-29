import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from kernelgen_server.evaluation.engine import (
    EvaluationEngine,
    _deterministic_correctness,
)
from kernelgen_server.evaluation.loader import load_operator_adapter
from kernelgen_server.runtime.device import Device
from kernelgen_server.schema import (
    Definition,
    EvaluateRequest,
    EvaluationSettings,
    EvaluationStatus,
    Implementation,
    SourceFile,
    Workload,
)


class CpuTestDevice(Device):
    def __init__(self):
        self.empty_cache_count = 0

    def set_device(self, device: str) -> None:
        del device

    def synchronize(self, device: str) -> None:
        del device

    def empty_cache(self) -> None:
        self.empty_cache_count += 1

    def is_available(self) -> bool:
        return True

    def list_devices(self) -> list[str]:
        return ["cpu"]

    def device_name(self, device: str) -> str:
        del device
        return "CPU"

    def time(self, fn, args, warmup, iters, device, grad_to_none=None):
        del warmup, iters, device, grad_to_none
        fn(*args)
        return 1.0


def _random_request() -> EvaluateRequest:
    source = (
        "import random\n"
        "import numpy\n"
        "import torch\n\n"
        "def run(x):\n"
        "    return x + torch.rand_like(x) + random.random() + numpy.random.random()\n"
    )
    return EvaluateRequest(
        api_version="v6.2",
        definition=Definition(
            api_version="v6.2",
            name="randomized",
            parameters=[{"name": "x", "required": True}],
            outputs=["output"],
            reference=source,
        ),
        implementation=Implementation(
            name="candidate",
            definition="randomized",
            language="python",
            entrypoint="main.py::run",
            sources=[SourceFile(path="main.py", content=source)],
        ),
        correctness_workloads=[
            Workload(
                name="random-correctness",
                inputs={
                    "x": {"type": "random", "shape": [16], "dtype": "float32"}
                },
                seed=1234,
            )
        ],
    )


def test_reference_and_candidate_replay_python_numpy_and_torch_rng():
    random.seed(9876)

    result = EvaluationEngine(CpuTestDevice(), "cpu").evaluate(
        _random_request()
    )

    assert result.status == EvaluationStatus.PASSED


def test_deterministic_correctness_scopes_and_restores_cudnn_flags(monkeypatch):
    cudnn = torch.backends.cudnn
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cudnn, "is_available", lambda: True)
    monkeypatch.setattr(cudnn, "deterministic", False)
    monkeypatch.setattr(cudnn, "benchmark", True)

    with pytest.raises(RuntimeError, match="body failed"):
        with _deterministic_correctness():
            assert cudnn.deterministic is True
            assert cudnn.benchmark is False
            raise RuntimeError("body failed")

    assert cudnn.deterministic is False
    assert cudnn.benchmark is True


def test_evaluate_releases_memory_after_each_workload_and_timing_trial():
    source = "def run(x): return x + 1\n"
    request = EvaluateRequest(
        api_version="v6.2",
        definition=Definition(
            api_version="v6.2",
            name="lifecycle",
            parameters=[{"name": "x", "required": True}],
            outputs=["output"],
            reference=source,
        ),
        implementation=Implementation(
            name="candidate",
            definition="lifecycle",
            language="python",
            entrypoint="main.py::run",
            sources=[SourceFile(path="main.py", content=source)],
        ),
        correctness_workloads=[
            Workload(
                name="correctness",
                inputs={"x": {"type": "random", "shape": [4], "dtype": "float32"}},
            )
        ],
        timing_workloads=[
            Workload(
                name="timing",
                inputs={"x": {"type": "random", "shape": [4], "dtype": "float32"}},
            )
        ],
        settings=EvaluationSettings(
            warmup_ms=0,
            benchmark_ms=1,
            num_trials=2,
        ),
    )

    class FormalOnlyEngine(EvaluationEngine):
        def _select_reference(self, selected_request):
            return load_operator_adapter(selected_request.definition)

    device = CpuTestDevice()
    result = FormalOnlyEngine(device, "cpu").evaluate(request)

    assert result.status == EvaluationStatus.PASSED
    assert device.empty_cache_count == 3

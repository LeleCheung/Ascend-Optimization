import pytest

torch = pytest.importorskip("torch")

from kernelgen_server.evaluation.compare import compare
from kernelgen_server.schema import EvaluationSettings, Tolerance, Workload


def _workload(tolerance: Tolerance | None = None) -> Workload:
    return Workload(
        name="case",
        inputs={"x": {"type": "scalar", "value": 1}},
        tolerance=tolerance,
    )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
def test_strict_default_atol_matches_flaggems(dtype):
    reference = torch.zeros(1, dtype=dtype)
    candidate = torch.full((1,), 5e-5, dtype=dtype)

    result = compare(
        reference,
        candidate,
        EvaluationSettings(tolerance_mode="strict"),
        _workload(),
    )

    assert result.passed


def test_workload_atol_scale_matches_flaggems_reduce_dim():
    reference = torch.zeros(1, dtype=torch.float32)
    candidate = torch.full((1,), 0.01, dtype=torch.float32)

    unscaled = compare(
        reference,
        candidate,
        EvaluationSettings(tolerance_mode="strict"),
        _workload(),
    )
    scaled = compare(
        reference,
        candidate,
        EvaluationSettings(tolerance_mode="strict"),
        _workload(Tolerance(atol_scale=128)),
    )

    assert not unscaled.passed
    assert scaled.passed


def test_explicit_workload_atol_overrides_flaggems_default():
    reference = torch.zeros(1, dtype=torch.float32)
    candidate = torch.full((1,), 1.0, dtype=torch.float32)

    result = compare(
        reference,
        candidate,
        EvaluationSettings(tolerance_mode="strict"),
        _workload(Tolerance(atol=2.0)),
    )

    assert result.passed


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (
            torch.tensor([float("inf"), float("-inf")]),
            torch.tensor([float("inf"), float("-inf")]),
        ),
        (float("inf"), float("inf")),
        (float("-inf"), float("-inf")),
    ],
)
def test_matching_infinities_pass(reference, candidate):
    result = compare(
        reference,
        candidate,
        EvaluationSettings(tolerance_mode="strict"),
        _workload(),
    )

    assert result.passed
    assert result.matched_ratio == 1.0
    assert result.max_absolute_error == 0.0
    assert result.max_relative_error == 0.0


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (torch.tensor([float("inf")]), torch.tensor([float("-inf")])),
        (torch.tensor([0.0]), torch.tensor([float("inf")])),
        (torch.tensor([float("inf")]), torch.tensor([0.0])),
        (float("inf"), float("-inf")),
        (0.0, float("inf")),
    ],
)
def test_mismatched_infinities_fail(reference, candidate):
    result = compare(
        reference,
        candidate,
        EvaluationSettings(tolerance_mode="strict"),
        _workload(),
    )

    assert not result.passed
    assert result.matched_ratio == 0.0
    assert result.max_absolute_error is None
    assert result.max_relative_error is None
    assert "non-finite" in result.message


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (torch.tensor([float("nan")]), torch.tensor([float("nan")])),
        (float("nan"), float("nan")),
    ],
)
def test_matching_nan_still_fails_by_default(reference, candidate):
    result = compare(
        reference,
        candidate,
        EvaluationSettings(tolerance_mode="strict"),
        _workload(),
    )

    assert not result.passed
    assert "non-finite" in result.message


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (torch.tensor([float("nan")]), torch.tensor([float("nan")])),
        (float("nan"), float("nan")),
    ],
)
def test_matching_nan_passes_when_workload_requests_equal_nan(reference, candidate):
    result = compare(
        reference,
        candidate,
        EvaluationSettings(tolerance_mode="strict"),
        _workload(Tolerance(equal_nan=True)),
    )

    assert result.passed
    assert result.matched_ratio == 1.0
    assert result.max_absolute_error == 0.0
    assert result.max_relative_error == 0.0

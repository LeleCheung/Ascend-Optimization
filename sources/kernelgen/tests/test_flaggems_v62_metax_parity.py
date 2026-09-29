import inspect
import math

import pytest

from examples.flaggems_v62_extract.validate_metax_parity import (
    _adapt_saved_candidate_abi,
    _reference_solution_source,
    _timing_geo_mean,
)


def test_reference_solution_can_select_whole_round_torch_fallback():
    source = (
        "def run(x):\n"
        "    return x + 1\n\n"
        "def torch_run(x):\n"
        "    return x + 2\n"
    )

    selected = _reference_solution_source(source, use_torch_fallback=True)

    assert selected.endswith("\nrun = torch_run\n")


def test_timing_geo_mean_ignores_correctness_status():
    response = {
        "status": "PARTIAL_PASS",
        "per_workload": [
            {"phase": "correctness", "status": "INCORRECT_NUMERICAL"},
            {"phase": "timing", "status": "PASSED", "speedup": 0.5},
            {"phase": "timing", "status": "PASSED", "speedup": 2.0},
        ],
    }

    assert _timing_geo_mean(response) == pytest.approx(math.sqrt(1.0))


def test_archived_candidate_argument_rename_gets_equal_parity_adapter():
    candidate = (
        "def run(input, k, dim=-1, keepdim=False):\n"
        "    return input + k + dim + keepdim\n"
    )
    oracle = (
        "def run(inp, k, dim=-1, keepdim=False):\n"
        "    return inp\n"
    )

    adapted, changed = _adapt_saved_candidate_abi(candidate, oracle)
    namespace = {}
    exec(adapted, namespace)

    assert changed
    assert namespace["run"](inp=3, k=2, dim=1, keepdim=True) == 7
    assert namespace["run"](input=3, k=2, dim=1, keepdim=True) == 7
    assert "__kgs_parity_saved_run = run" in adapted
    assert inspect.signature(namespace["run"]) == inspect.signature(
        namespace["__kgs_parity_saved_run"]
    )
    with pytest.raises(TypeError, match="received both inp and input"):
        namespace["run"](inp=3, input=3, k=2)


def test_archived_candidate_structural_abi_mismatch_is_not_hidden():
    with pytest.raises(ValueError, match="differs structurally"):
        _adapt_saved_candidate_abi(
            "def run(input, k):\n    return input\n",
            "def run(inp, k, dim=-1):\n    return inp\n",
        )


def test_archived_candidate_same_names_defer_runtime_signature_validation():
    candidate = (
        "def run(self, p=5):\n"
        "    return self\n\n"
        "run.__signature__ = object()\n"
    )
    oracle = "def run(self, p):\n    return self\n"

    adapted, changed = _adapt_saved_candidate_abi(candidate, oracle)

    assert adapted == candidate
    assert not changed

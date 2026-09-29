import ast
import json
from collections import Counter
from pathlib import Path

import pytest

from kernelgen_server import Catalog
from tools.convert_pr17_model_catalogs_v62 import SOURCE_REVISION


ROOT = Path(__file__).parents[1] / "data"
EXPECTED = {
    "kernelbench": {
        "counts": {
            "operators": 250,
            "correctness_workloads": 1_250,
            "timing_workloads": 250,
        },
        "groups": {"level1": 100, "level2": 100, "level3": 50},
    },
    "kernelswift": {
        "counts": {
            "operators": 10,
            "correctness_workloads": 10,
            "timing_workloads": 10,
        },
        "groups": {"attention": 2, "moe": 2, "special": 6},
    },
}


@pytest.mark.parametrize("catalog_name", tuple(EXPECTED))
def test_pr17_model_catalog_is_native_v62(catalog_name):
    root = ROOT / catalog_name
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = EXPECTED[catalog_name]
    catalog = Catalog(root)

    assert manifest["api_version"] == "v6.2"
    assert manifest["evaluator"] == "native"
    assert manifest["layout"] == "per-operator"
    assert manifest["source_api_version"] == "v5.1"
    assert manifest["source_revision"] == SOURCE_REVISION
    assert manifest["counts"] == expected["counts"]
    assert Counter(item["group"] for item in manifest["operators"]) == expected[
        "groups"
    ]
    assert len(catalog.operator_names) == expected["counts"]["operators"]

    correctness = timing = 0
    workload_names: set[str] = set()
    for name in catalog.operator_names:
        operator = catalog.load(name)
        assert operator.oracle_path is not None
        operator_root = operator.oracle_path.parent
        public = json.loads(
            (operator_root / "definition.json").read_text(encoding="utf-8")
        )
        oracle = operator.oracle_path.read_text(encoding="utf-8")
        ast.parse(oracle)

        assert public["api_version"] == "v6.2"
        assert "reference" not in public
        assert "reference_device" not in public
        assert oracle.count("REFERENCE_DEVICE =") == 1
        assert "def run(" in oracle
        correctness += len(operator.correctness_workloads)
        timing += len(operator.timing_workloads)
        for workload in (
            *operator.correctness_workloads,
            *operator.timing_workloads,
        ):
            assert workload.name not in workload_names
            workload_names.add(workload.name)

    assert correctness == expected["counts"]["correctness_workloads"]
    assert timing == expected["counts"]["timing_workloads"]


def test_kernelswift_sinkhorn_custom_validator_is_in_oracle():
    operator = Catalog(ROOT / "kernelswift").load(
        "ks_hc_split_sinkhorn_hc4_iter20"
    )
    oracle = operator.definition.reference

    assert oracle is not None
    assert "VALID_OWNS_RETURN_CONTRACT = True" in oracle
    assert "def valid(ref_outputs, sol_outputs, inputs, ctx):" in oracle


def test_kernelbench_preserves_upstream_license():
    assert (ROOT / "kernelbench" / "LICENSE").read_text(encoding="utf-8") == (
        Path(__file__).parents[1] / "licenses" / "KernelBench-MIT.txt"
    ).read_text(encoding="utf-8")

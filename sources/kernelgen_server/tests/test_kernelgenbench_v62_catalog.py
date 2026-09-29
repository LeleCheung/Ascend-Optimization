import ast
import json
from collections import Counter
from pathlib import Path

from kernelgen_server import Catalog, builtin_catalog_path
from tools.convert_kernelgenbench_v5 import (
    EXPECTED_COUNTS,
    EXPECTED_FULL_COUNTS,
    WORKLOAD_LIMIT,
    _generic_custom_inputs,
)
from tools.sample_v62_catalog_workloads import ALGORITHM


def test_kernelgenbench_is_a_complete_v62_native_catalog():
    root = builtin_catalog_path("kernelgenbench")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    catalog = Catalog(root)

    assert manifest["api_version"] == "v6.2"
    assert manifest["evaluator"] == "native"
    assert manifest["layout"] == "per-operator"
    assert manifest["counts"] == EXPECTED_COUNTS
    assert manifest["full_counts"] == EXPECTED_FULL_COUNTS
    assert manifest["sampling"] == {
        "algorithm": ALGORITHM,
        "max_workloads_per_phase": WORKLOAD_LIMIT,
        "source_files": ["correctness_full.jsonl", "timing_full.jsonl"],
    }
    assert len(catalog.operator_names) == EXPECTED_COUNTS["operators"]
    assert not (root / "definitions").exists()
    assert not (root / "workloads").exists()
    groups = {entry["name"]: entry["group"] for entry in manifest["operators"]}
    assert Counter(groups.values()) == {
        "pointwise": 110,
        "vllm": 50,
        "cublas": 50,
    }
    assert {path.name for path in (root / "ops").iterdir()} == {
        "pointwise",
        "vllm",
        "cublas",
    }

    correctness = 0
    timing = 0
    full_correctness = 0
    full_timing = 0
    names = set()
    for name in catalog.operator_names:
        operator = catalog.load(name)
        correctness += len(operator.correctness_workloads)
        timing += len(operator.timing_workloads)
        operator_root = root / "ops" / groups[name] / name
        archived_correctness = (
            operator_root / "correctness_full.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        archived_timing = (
            operator_root / "timing_full.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        full_correctness += len(archived_correctness)
        full_timing += len(archived_timing)
        assert len(operator.correctness_workloads) <= WORKLOAD_LIMIT
        assert len(operator.timing_workloads) <= WORKLOAD_LIMIT
        assert {
            workload.name for workload in operator.correctness_workloads
        }.issubset({json.loads(line)["name"] for line in archived_correctness})
        assert {
            workload.name for workload in operator.timing_workloads
        }.issubset({json.loads(line)["name"] for line in archived_timing})
        for workload in operator.correctness_workloads + operator.timing_workloads:
            assert workload.name not in names
        names.add(workload.name)
        public = json.loads(
            (
                root / "ops" / groups[name] / name / "definition.json"
            ).read_text(encoding="utf-8")
        )
        assert public["api_version"] == "v6.2"
        assert "reference" not in public
        assert "reference_device" not in public
        oracle = operator.definition.reference
        assert oracle is not None
        tree = ast.parse(oracle)
        assignments = [
            node
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "REFERENCE_DEVICE"
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
        ]
        assert len(assignments) == 1

    assert correctness == EXPECTED_COUNTS["correctness_workloads"]
    assert timing == EXPECTED_COUNTS["timing_workloads"]
    assert full_correctness == EXPECTED_FULL_COUNTS["correctness_workloads"]
    assert full_timing == EXPECTED_FULL_COUNTS["timing_workloads"]


def test_kernelgenbench_migration_preserves_phase_hooks_and_effects():
    catalog = Catalog(builtin_catalog_path("kernelgenbench"))

    empty = catalog.load("kernelgenbench_empty_strided")
    assert "correctness_run = run_correctness" in empty.definition.reference
    assert "VALID_OWNS_RETURN_CONTRACT = True" in empty.definition.reference

    contiguous = catalog.load("kernelgenbench_contiguous")
    assert "def _legacy_context(ctx):" in contiguous.definition.reference
    assert "def gen_inputs(ctx, device):" in contiguous.definition.reference

    for name in ("kernelgenbench_clone", "kernelgenbench__to_copy"):
        strided = catalog.load(name)
        assert "def gen_inputs(ctx, device):" in strided.definition.reference
        assert "torch.as_strided" in strided.definition.reference

    add_inplace = catalog.load("kernelgenbench_add_").definition
    assert add_inplace.effects.mutates == ["x"]
    assert add_inplace.effects.returns_alias_of == {"output": "x"}

    cublas_copy = catalog.load("cublas_cublasCcopy_v2").definition
    assert cublas_copy.effects.mutates == ["y"]
    assert cublas_copy.effects.returns_alias_of == {"y": "y"}

    fused_norm = catalog.load("vllm_fused_add_rms_norm").definition
    assert fused_norm.effects.mutates == ["input", "residual"]
    assert fused_norm.effects.returns_alias_of == {"output": "input"}


def test_kernelgenbench_converter_materializes_generic_strided_custom_inputs():
    source = _generic_custom_inputs(
        {"name": "clone"},
        [
            {
                "inputs": {
                    "x": {
                        "type": "custom",
                        "shape": [16, 8],
                        "dtype": "float32",
                        "stride": [1, 16],
                        "storage_offset": 0,
                    }
                }
            }
        ],
    )

    assert "def gen_inputs(ctx, device):" in source
    assert "torch.as_strided" in source

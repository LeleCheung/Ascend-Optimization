import json
from pathlib import Path

from kernelgen_server import Catalog
from tools.sample_v62_catalog_workloads import (
    ALGORITHM,
    WorkloadRow,
    sample_catalog,
    sample_rows,
)


def _row(index: int, **inputs):
    value = {
        "name": f"case-{index}",
        "inputs": {
            name: {"type": "scalar", "value": value}
            for name, value in inputs.items()
        },
        "seed": 42,
    }
    line = json.dumps(value, sort_keys=True)
    return WorkloadRow(line, value, f"{index:08d}")


def _write_catalog(root: Path) -> Path:
    operator = root / "ops" / "group" / "identity"
    operator.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "api_version": "v6.2",
                "name": root.name,
                "evaluator": "native",
                "layout": "per-operator",
                "counts": {
                    "operators": 1,
                    "correctness_workloads": 8,
                    "timing_workloads": 8,
                },
                "operators": [{"name": "identity", "group": "group"}],
            }
        ),
        encoding="utf-8",
    )
    (operator / "definition.json").write_text(
        json.dumps(
            {
                "api_version": "v6.2",
                "name": "identity",
                "parameters": [
                    {
                        "name": "x",
                        "kind": "positional_or_keyword",
                        "required": True,
                    },
                    {
                        "name": "mode",
                        "kind": "positional_or_keyword",
                        "required": True,
                    },
                ],
                "outputs": ["output"],
                "effects": {"mutates": [], "returns_alias_of": {}},
            }
        ),
        encoding="utf-8",
    )
    (operator / "oracle.py").write_text(
        "REFERENCE_DEVICE = 'target'\n\ndef run(x, mode):\n    return x\n",
        encoding="utf-8",
    )
    for phase in ("correctness", "timing"):
        phase_rows = [
            json.dumps(
                {
                    "name": f"identity-{phase}-{index}",
                    "inputs": {
                        "x": {"type": "scalar", "value": index % 4},
                        "mode": {"type": "scalar", "value": index // 4},
                    },
                    "seed": 42,
                }
            )
            for index in range(8)
        ]
        (operator / f"{phase}.jsonl").write_text(
            "\n".join(phase_rows) + "\n", encoding="utf-8"
        )
    return operator


def test_sample_rows_is_deterministic_and_covers_every_input_value():
    rows = tuple(
        _row(index, a=a, b=b, c=c)
        for index, (a, b, c) in enumerate(
            (a, b, c) for a in range(8) for b in range(8) for c in range(8)
        )
    )

    first = sample_rows(rows, 24)
    second = sample_rows(rows, 24)

    assert [row.value["name"] for row in first.rows] == [
        row.value["name"] for row in second.rows
    ]
    assert len(first.rows) == 24
    for name in ("a", "b", "c"):
        assert {
            row.value["inputs"][name]["value"] for row in first.rows
        } == set(range(8))
    assert first.covered_unary_features == first.unary_features


def test_sample_catalog_renames_full_files_and_is_idempotent(tmp_path):
    operator = _write_catalog(tmp_path / "catalog")
    originals = {
        phase: (operator / f"{phase}.jsonl").read_bytes()
        for phase in ("correctness", "timing")
    }

    manifest = sample_catalog(tmp_path / "catalog", limit=3)
    repeated = sample_catalog(tmp_path / "catalog", limit=3)

    assert manifest == repeated
    assert manifest["counts"] == {
        "operators": 1,
        "correctness_workloads": 3,
        "timing_workloads": 3,
    }
    assert manifest["full_counts"] == {
        "operators": 1,
        "correctness_workloads": 8,
        "timing_workloads": 8,
    }
    assert manifest["sampling"] == {
        "algorithm": ALGORITHM,
        "max_workloads_per_phase": 3,
        "source_files": ["correctness_full.jsonl", "timing_full.jsonl"],
    }
    for phase in ("correctness", "timing"):
        assert (operator / f"{phase}_full.jsonl").read_bytes() == originals[phase]
        assert len((operator / f"{phase}.jsonl").read_text().splitlines()) == 3
    loaded = Catalog(tmp_path / "catalog").load("identity")
    assert len(loaded.correctness_workloads) == 3
    assert len(loaded.timing_workloads) == 3

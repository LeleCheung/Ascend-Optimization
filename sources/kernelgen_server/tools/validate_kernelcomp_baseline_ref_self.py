#!/usr/bin/env python3
"""Reference-as-solution end-to-end smoke for a native v6.2 catalog.

Runs the production evaluation chain (``EvaluationEngine``) with the
oracle's own ``run`` bound as BOTH the reference and the candidate
implementation.  This exercises the complete pipeline a real submission
would go through — CPU-base materialization, target-device inputs, reference
execution, candidate execution, PyTree structure + tolerance comparison,
mutation and return-alias gates — and verifies the declared contract
(tolerance / outputs / effects) is consistent with what the oracle actually
produces.

Any failure here means the operator's declared contract is internally
inconsistent (a configuration bug that ``validate_v62_native_catalog.py
--runtime`` cannot catch, because it only re-feeds the reference output to
``valid``).

Usage:
    python3 tools/validate_kernelcomp_baseline_ref_self.py \
        data/kernelcomp-baseline --target-device cuda:6 [--operator NAME ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelgen_server.evaluation.engine import EvaluationEngine  # noqa: E402
from kernelgen_server.evaluation.loader import (  # noqa: E402
    load_implementation,
    load_operator_adapter,
)
from kernelgen_server.protocol.schema import (  # noqa: E402
    Definition,
    EvaluateRequest,
    EvaluationSettings,
    Implementation,
    Language,
    SourceFile,
    Workload,
)
from kernelgen_server.runtime.device import get_device  # noqa: E402


def _load_definition(path: Path) -> Definition:
    return Definition(**json.loads(path.read_text(encoding="utf-8")))


def _load_workloads(path: Path) -> list[Workload]:
    workloads: list[Workload] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            workloads.append(Workload(**json.loads(line)))
    return workloads


def _oracle_implementation(operator_root: Path) -> Implementation:
    """Package the oracle's own ``run`` as a candidate implementation so the
    full evaluation chain treats it exactly like a submission."""
    oracle_source = (operator_root / "oracle.py").read_text(encoding="utf-8")
    return Implementation(
        name=f"{operator_root.name}-as-solution",
        definition=operator_root.name,
        language=Language.PYTHON,
        entrypoint="oracle.py::run",
        sources=[SourceFile(path="oracle.py", content=oracle_source)],
    )


def run_operator(
    operator_root: Path,
    *,
    target_device: str,
) -> tuple[int, list[str]]:
    definition = _load_definition(operator_root / "definition.json")
    correctness = _load_workloads(operator_root / "correctness.jsonl")
    # The engine loads the trusted oracle source from Definition.reference;
    # attach it so ``load_operator_adapter`` finds it.
    definition = definition.model_copy(
        update={
            "reference": (operator_root / "oracle.py").read_text(encoding="utf-8")
        }
    )
    request = EvaluateRequest(
        definition=definition,
        implementation=_oracle_implementation(operator_root),
        correctness_workloads=correctness,
        timing_workloads=[],
        settings=EvaluationSettings(),
    )
    engine = EvaluationEngine(
        get_device(target_device),
        target_device,
        backend="native",
        oracle_path=operator_root / "oracle.py",
    )
    response = engine.evaluate(request)
    failures: list[str] = []
    for result in response.per_workload:
        if result.status.value != "PASSED":
            failures.append(f"{result.uuid}: {result.status.value}: {result.log}")
    return len(response.per_workload), failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--target-device", required=True)
    parser.add_argument("--operator", action="append", default=[])
    args = parser.parse_args()

    catalog = args.catalog.resolve()
    operators = sorted(
        path.parent for path in (catalog / "ops").rglob("definition.json")
    )
    if args.operator:
        selected = set(args.operator)
        operators = [path for path in operators if path.name in selected]

    failed = 0
    total_ops = 0
    total_wl = 0
    for operator_root in operators:
        count, failures = run_operator(
            operator_root, target_device=args.target_device
        )
        total_ops += 1
        total_wl += count
        if failures:
            failed += len(failures)
            print(f"FAIL {operator_root.relative_to(catalog)} ({len(failures)}/{count})", flush=True)
            for message in failures[:5]:
                print(f"  {message}", flush=True)
        else:
            print(f"PASS {operator_root.relative_to(catalog)} ({count} workloads)", flush=True)
    print(f"\nref-as-solution: operators={total_ops} workloads={total_wl} failed={failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()


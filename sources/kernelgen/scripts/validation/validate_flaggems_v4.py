#!/usr/bin/env python3
"""Validate a trace set through eval_cli, one isolated operator at a time.

Each eval_cli process receives a temporary trace set containing exactly one
Definition and its phased workloads.  This prevents a malformed workload for
one operator from making TraceSet.from_path reject every other operator.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _copy_operator_trace(
    trace_set: Path,
    definition_path: Path,
    isolated_root: Path,
) -> tuple[dict[str, Any], int]:
    definition = json.loads(definition_path.read_text())
    name = definition["name"]
    op_type = definition["op_type"]

    output_definition = (
        isolated_root / "definitions" / op_type / definition_path.name
    )
    output_definition.parent.mkdir(parents=True)
    shutil.copy2(definition_path, output_definition)

    workload_count = 0
    for phase in ("correctness", "timing"):
        workload_path = (
            trace_set
            / "phased_workloads"
            / op_type
            / f"{name}.{phase}.jsonl"
        )
        if not workload_path.is_file():
            raise FileNotFoundError(f"Missing {phase} workloads: {workload_path}")
        output_workload = (
            isolated_root
            / "phased_workloads"
            / op_type
            / workload_path.name
        )
        output_workload.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workload_path, output_workload)
        workload_count += sum(
            bool(line.strip()) for line in workload_path.read_text().splitlines()
        )

    return definition, workload_count


def _run_operator(
    args: argparse.Namespace,
    definition_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f"fib-v4-{definition_path.stem}-",
        dir=args.temp_root,
    ) as temp_dir:
        temp_root = Path(temp_dir)
        isolated_trace = temp_root / "trace"
        definition, expected_workloads = _copy_operator_trace(
            args.trace_set, definition_path, isolated_trace
        )
        kernel_path = temp_root / "kernel.py"
        kernel_path.write_text(definition["reference"])

        command = [
            args.python,
            "-m",
            "eval_service.eval_cli",
            "--kernel",
            str(kernel_path),
            "--definition",
            definition["name"],
            "--trace-set",
            str(isolated_trace),
            "--key",
            args.key,
            "--target-hardware",
            args.target_hardware,
            "--language",
            "python",
            "--entry-point",
            "main.py::run",
            "--no-dps",
            "--server",
            args.server,
        ]
        if args.num_workloads:
            command.extend(["--num-workloads", str(args.num_workloads)])

        try:
            process = subprocess.run(
                command,
                cwd=args.flashinfer_bench,
                env={**args.environment, "PYTHONPATH": str(args.flashinfer_bench)},
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return {
                "definition": definition["name"],
                "op_type": definition["op_type"],
                "status": "TIMEOUT",
                "returncode": None,
                "expected_workloads": expected_workloads,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "stderr": (error.stderr or "")[-4000:],
                "result": None,
            }

        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError:
            result = {
                "status": "INVALID_CLI_OUTPUT",
                "log": process.stdout[-4000:],
            }
        return {
            "definition": definition["name"],
            "op_type": definition["op_type"],
            "status": result.get("status", "INVALID_CLI_OUTPUT"),
            "returncode": process.returncode,
            "expected_workloads": expected_workloads,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "stderr": process.stderr[-4000:],
            "result": result,
        }


def _run_operator_safe(
    args: argparse.Namespace,
    definition_path: Path,
) -> dict[str, Any]:
    try:
        return _run_operator(args, definition_path)
    except Exception as error:
        return {
            "definition": definition_path.stem,
            "op_type": definition_path.parent.name,
            "status": "VALIDATION_ERROR",
            "returncode": None,
            "expected_workloads": None,
            "elapsed_sec": None,
            "stderr": f"{type(error).__name__}: {error}",
            "result": None,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-set", type=Path, required=True)
    parser.add_argument("--flashinfer-bench", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--target-hardware", default="Ascend910B")
    parser.add_argument("--key", default="unified-trace-flaggems-v4")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num-workloads", type=int, default=0)
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp"))
    args = parser.parse_args()

    import os

    args.environment = os.environ.copy()
    args.trace_set = args.trace_set.resolve()
    args.flashinfer_bench = args.flashinfer_bench.resolve()
    args.temp_root.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)

    definitions = sorted((args.trace_set / "definitions").rglob("*.json"))
    if args.operator:
        selected = set(args.operator)
        definitions = [path for path in definitions if path.stem in selected]
        missing = selected - {path.stem for path in definitions}
        if missing:
            parser.error("Unknown operators: " + ", ".join(sorted(missing)))
    if not definitions:
        parser.error("No definitions selected")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    passed = 0
    with args.results.open("w") as results_file:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(_run_operator_safe, args, path): path
                for path in definitions
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                completed += 1
                ok = record["returncode"] == 0 and record["status"] == "PASSED"
                passed += ok
                print(
                    f"[{completed}/{len(definitions)}] {record['definition']} "
                    f"{record['status']} rc={record['returncode']} "
                    f"elapsed={record['elapsed_sec']}s",
                    flush=True,
                )
                results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                results_file.flush()

    summary = {
        "selected": len(definitions),
        "passed": passed,
        "failed": len(definitions) - passed,
        "results": str(args.results),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if passed == len(definitions) else 1


if __name__ == "__main__":
    raise SystemExit(main())

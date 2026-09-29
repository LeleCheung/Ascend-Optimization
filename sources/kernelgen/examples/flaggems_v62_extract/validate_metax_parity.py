#!/usr/bin/env python3
"""Validate one extracted v6.2 operator against the FlagGems adapter."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import requests


ADAPTER_CATALOG = "flaggems-adapter-definitions"
NATIVE_CATALOG = "flaggems-native"


def _parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:19904")
    parser.add_argument(
        "--server-catalog-root",
        type=Path,
        default=repo.parent / "kernelgen_server/data/flaggems-native",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=repo / "kernel_todo_v1/muxi-flaggems-adapter/codes",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("auto", "oracle-timing"),
        default="auto",
        help=(
            "auto uses a saved candidate when present and otherwise the oracle; "
            "oracle-timing always submits the oracle timing entry to both evaluators"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo / "runs/v62_muxi_native_parity_20260826",
    )
    parser.add_argument("--server-commit", default="")
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--benchmark-ms", type=int, default=100)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


def _save(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class Client:
    def __init__(self, server_url: str, output: Path):
        self.server_url = server_url.rstrip("/")
        self.output = output
        self.session = requests.Session()
        self.session.trust_env = False

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: int = 1200,
    ) -> dict[str, Any]:
        response = self.session.request(
            method,
            self.server_url + path,
            json=payload,
            timeout=timeout,
        )
        try:
            value = response.json()
        except ValueError:
            value = {"raw": response.text}
        if response.status_code >= 400:
            raise RuntimeError(f"{path} HTTP {response.status_code}: {value}")
        return value

    def post(
        self,
        path: str,
        payload: dict[str, Any],
        artifact: str,
        *,
        timeout: int = 1200,
    ) -> dict[str, Any]:
        started = time.monotonic()
        value = self.request("POST", path, payload, timeout=timeout)
        _save(
            self.output / artifact,
            {"elapsed_seconds": time.monotonic() - started, "response": value},
        )
        print(artifact, value.get("status"), value.get("geo_mean"), flush=True)
        return value


def _implementation(operator: str, source: str, *, name: str, language: str) -> dict:
    return {
        "name": name,
        "definition": operator,
        "language": language,
        "entrypoint": "main.py::run",
        "sources": [{"path": "main.py", "content": source}],
    }


def _bound(
    operator: str,
    catalog: str,
    candidate: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "api_version": "v6.2",
        "binding": {"catalog_name": catalog, "definition": operator},
        "implementation": candidate,
        "settings": settings,
    }


def _case_ids(manifest: dict[str, Any]) -> list[str]:
    return [case["case_id"] for case in manifest["case_list"]["cases"]]


def _timing_rows(response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["uuid"]: row
        for row in response["per_workload"]
        if row["phase"] == "timing"
    }


def _public_bindings(source: str) -> set[str]:
    bindings: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bindings.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings.add(target.id)
    return bindings


def _reference_solution_source(
    source: str, *, use_torch_fallback: bool = False
) -> str:
    bindings = _public_bindings(source)
    if use_torch_fallback:
        if "torch_run" not in bindings:
            raise ValueError("oracle does not define torch_run")
        return source + "\nrun = torch_run\n"
    if "correctness_run" in bindings:
        return source + "\nrun = correctness_run\n"
    if "run" in bindings:
        return source
    if "timing_run" in bindings:
        return source + "\nrun = timing_run\n"
    raise ValueError("oracle defines none of run/correctness_run/timing_run")


def _timing_candidate_source(
    source: str, *, use_torch_fallback: bool = False
) -> str:
    bindings = _public_bindings(source)
    if use_torch_fallback:
        if "torch_run" not in bindings:
            raise ValueError("oracle does not define torch_run")
        return source + "\nrun = torch_run\n"
    if "timing_run" in bindings:
        return source + "\nrun = timing_run\n"
    if "run" in bindings:
        return source
    if "correctness_run" in bindings:
        return source + "\nrun = correctness_run\n"
    raise ValueError("oracle defines none of run/correctness_run/timing_run")


def _public_run_function(source: str) -> ast.FunctionDef:
    functions = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    for name in ("run", "correctness_run", "timing_run"):
        function = functions.get(name)
        if function is not None:
            return function
    raise ValueError("oracle defines none of run/correctness_run/timing_run")


def _argument_shape(arguments: ast.arguments) -> tuple[Any, ...]:
    return (
        len(arguments.posonlyargs),
        len(arguments.args),
        arguments.vararg is not None,
        len(arguments.kwonlyargs),
        tuple(value is not None for value in arguments.kw_defaults),
        arguments.kwarg is not None,
        tuple(ast.dump(value, include_attributes=False) for value in arguments.defaults),
        tuple(
            None if value is None else ast.dump(value, include_attributes=False)
            for value in arguments.kw_defaults
        ),
    )


def _argument_names(arguments: ast.arguments) -> tuple[str, ...]:
    values = [argument.arg for argument in arguments.posonlyargs]
    values.extend(argument.arg for argument in arguments.args)
    values.extend(argument.arg for argument in arguments.kwonlyargs)
    return tuple(values)


def _adapt_saved_candidate_abi(
    candidate_source: str,
    oracle_source: str,
) -> tuple[str, bool]:
    """Add an equal-cost adapter for archived candidates with renamed arguments.

    Old adapter evaluations invoked positional-or-keyword parameters positionally,
    so a Coder could return ``run(input, ...)`` for a Definition whose public name
    was ``inp``. Native v6.2 correctly materializes that ABI by name. Keep the
    archived source immutable and append the same thin wrapper for both parity
    evaluations, but only when the signatures differ by names alone.
    """

    candidate_module = ast.parse(candidate_source)
    candidate_run = next(
        (
            node
            for node in candidate_module.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        ),
        None,
    )
    if candidate_run is None:
        raise ValueError("saved candidate does not define a top-level run")
    public_run = _public_run_function(oracle_source)
    if _argument_names(candidate_run.args) == _argument_names(public_run.args):
        # Do not second-guess an archived candidate's runtime __signature__.
        # Both evaluators use inspect.signature and will apply the same ABI gate;
        # this helper only needs to synthesize a wrapper for renamed arguments.
        return candidate_source, False
    if _argument_shape(candidate_run.args) != _argument_shape(public_run.args):
        raise ValueError(
            "saved candidate ABI differs structurally from the v6.2 Definition"
        )

    candidate_fixed = [
        *candidate_run.args.posonlyargs,
        *candidate_run.args.args,
        *candidate_run.args.kwonlyargs,
    ]
    public_fixed = [
        *public_run.args.posonlyargs,
        *public_run.args.args,
        *public_run.args.kwonlyargs,
    ]
    renamed = [
        (candidate.arg, public.arg)
        for candidate, public in zip(candidate_fixed, public_fixed)
        if candidate.arg != public.arg
    ]
    rows = [
        "",
        "# Parity-only dual-name adapter for an archived positional candidate.",
        "import inspect as __kgs_parity_inspect",
        "__kgs_parity_saved_run = run",
        "__kgs_parity_missing = object()",
        "def run(*args, **kwargs):",
    ]
    for index, (candidate_name, public_name) in enumerate(renamed):
        public_value = f"__kgs_public_{index}"
        candidate_value = f"__kgs_candidate_{index}"
        rows.extend(
            [
                f"    {public_value} = kwargs.pop({public_name!r}, __kgs_parity_missing)",
                f"    {candidate_value} = kwargs.pop({candidate_name!r}, __kgs_parity_missing)",
                f"    if {public_value} is not __kgs_parity_missing and {candidate_value} is not __kgs_parity_missing:",
                f"        raise TypeError(\"run() received both {public_name} and {candidate_name}\")",
                f"    if {public_value} is not __kgs_parity_missing:",
                f"        kwargs[{candidate_name!r}] = {public_value}",
                f"    elif {candidate_value} is not __kgs_parity_missing:",
                f"        kwargs[{candidate_name!r}] = {candidate_value}",
            ]
        )
    rows.append("    return __kgs_parity_saved_run(*args, **kwargs)")
    rows.append(
        "run.__signature__ = __kgs_parity_inspect.signature(__kgs_parity_saved_run)"
    )
    adapter = "\n".join(rows) + "\n"
    return candidate_source.rstrip() + adapter, True


def _timing_geo_mean(response: dict[str, Any]) -> float:
    rows = [
        row for row in response["per_workload"] if row["phase"] == "timing"
    ]
    if not rows or any(row["status"] != "PASSED" for row in rows):
        raise AssertionError("not every timing workload passed")
    speedups = [row["speedup"] for row in rows]
    if any(not isinstance(value, (int, float)) or value <= 0 for value in speedups):
        raise AssertionError("every timing workload needs a positive speedup")
    return math.exp(sum(math.log(value) for value in speedups) / len(speedups))


def _ratio_summary(values: list[float]) -> dict[str, float]:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise AssertionError("latency ratios must be finite and positive")
    geometric_mean = math.exp(
        sum(math.log(value) for value in values) / len(values)
    )
    return {
        "geometric_mean": geometric_mean,
        "geometric_mean_percent_delta": (geometric_mean - 1.0) * 100.0,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _assert_idle(status: dict[str, Any]) -> None:
    scheduler = status["scheduler"]
    assert scheduler["active"] == scheduler["waiting"] == 0, scheduler
    assert scheduler["checking"] == scheduler["broken"] == 0, scheduler
    assert scheduler["healthy"] == scheduler["device_slots"], scheduler
    assert scheduler["available"] == scheduler["device_slots"], scheduler


def main() -> None:
    args = _parse_args()
    output = args.output_root.resolve() / args.operator
    output.mkdir(parents=True, exist_ok=True)
    client = Client(args.server_url, output)
    settings = {
        "warmup_ms": args.warmup_ms,
        "benchmark_ms": args.benchmark_ms,
        "num_trials": args.num_trials,
        "timeout_seconds": args.timeout_seconds,
    }

    initial_status = client.request("GET", "/status", timeout=60)
    assert initial_status["api_version"] == "v6.2", initial_status
    assert initial_status["backend"] == "metax", initial_status
    assert initial_status["target"]["device"] == "MetaX C550", initial_status
    _assert_idle(initial_status)
    _save(output / "status_initial.json", initial_status)

    manifests = {
        catalog: client.request(
            "POST",
            "/inspect",
            {
                "api_version": "v6.2",
                "binding": {
                    "catalog_name": catalog,
                    "definition": args.operator,
                },
            },
            timeout=300,
        )
        for catalog in (ADAPTER_CATALOG, NATIVE_CATALOG)
    }
    _save(output / "inspect.json", manifests)
    adapter_ids = _case_ids(manifests[ADAPTER_CATALOG])
    native_ids = _case_ids(manifests[NATIVE_CATALOG])
    assert native_ids == adapter_ids

    oracle_path = (
        args.server_catalog_root.resolve()
        / "ops"
        / args.operator
        / "oracle.py"
    )
    oracle_source = oracle_path.read_text(encoding="utf-8")
    reference_solution = _implementation(
        args.operator,
        _reference_solution_source(oracle_source),
        name=f"{args.operator}-reference-as-solution",
        language="python",
    )
    reference_result = client.post(
        "/evaluate",
        _bound(args.operator, NATIVE_CATALOG, reference_solution, settings),
        "reference_as_solution.json",
        timeout=args.timeout_seconds + 120,
    )
    if (
        reference_result["status"] != "PASSED"
        and reference_result.get("reference_source") == "torch_fallback"
    ):
        fallback_solution = _implementation(
            args.operator,
            _reference_solution_source(
                oracle_source, use_torch_fallback=True
            ),
            name=f"{args.operator}-torch-fallback-as-solution",
            language="python",
        )
        reference_result = client.post(
            "/evaluate",
            _bound(args.operator, NATIVE_CATALOG, fallback_solution, settings),
            "reference_as_solution_fallback.json",
            timeout=args.timeout_seconds + 120,
        )
    assert reference_result["status"] == "PASSED", reference_result
    assert reference_result["num_passed"] == reference_result["num_workloads"]

    candidate_path = args.candidate_root.resolve() / f"{args.operator}.py"
    if args.candidate_mode == "oracle-timing":
        candidate_source = _timing_candidate_source(
            oracle_source,
            use_torch_fallback=(
                reference_result.get("reference_source") == "torch_fallback"
            ),
        )
        original_candidate_source = candidate_source
        candidate_abi_adapted = False
        candidate_source_kind = "oracle_timing_candidate"
        candidate = _implementation(
            args.operator,
            candidate_source,
            name=f"{args.operator}-oracle-timing-candidate",
            language="python",
        )
    elif candidate_path.is_file():
        original_candidate_source = candidate_path.read_text(encoding="utf-8")
        candidate_source, candidate_abi_adapted = _adapt_saved_candidate_abi(
            original_candidate_source,
            oracle_source,
        )
        candidate_source_kind = "saved_muxi_candidate"
        candidate = _implementation(
            args.operator,
            candidate_source,
            name=f"muxi-{args.operator}-saved-best",
            language="triton",
        )
    else:
        candidate_source = _reference_solution_source(oracle_source)
        original_candidate_source = candidate_source
        candidate_abi_adapted = False
        candidate_source_kind = "reference_as_candidate_no_saved_muxi_candidate"
        candidate = reference_solution
    for catalog in (ADAPTER_CATALOG, NATIVE_CATALOG):
        result = client.post(
            "/preflight",
            _bound(args.operator, catalog, candidate, settings),
            f"preflight_{catalog}.json",
            timeout=args.timeout_seconds + 120,
        )
        assert result["status"] == "PASSED", result
        assert result["num_cases"] == len(native_ids), result

    results = {}
    catalogs_to_evaluate = (ADAPTER_CATALOG, NATIVE_CATALOG)
    if candidate_source_kind == "reference_as_candidate_no_saved_muxi_candidate":
        results[NATIVE_CATALOG] = reference_result
        catalogs_to_evaluate = (ADAPTER_CATALOG,)
    for catalog in catalogs_to_evaluate:
        results[catalog] = client.post(
            "/evaluate",
            _bound(args.operator, catalog, candidate, settings),
            f"evaluate_{catalog}.json",
            timeout=args.timeout_seconds + 120,
        )
        _timing_geo_mean(results[catalog])

    adapter_timing = _timing_rows(results[ADAPTER_CATALOG])
    native_timing = _timing_rows(results[NATIVE_CATALOG])
    assert list(adapter_timing) == list(native_timing) == native_ids
    per_case = []
    for case_id in native_ids:
        adapter_row = adapter_timing[case_id]
        native_row = native_timing[case_id]
        candidate_latency_ratio = (
            native_row["latency_ms"] / adapter_row["latency_ms"]
        )
        reference_latency_ratio = (
            native_row["reference_latency_ms"]
            / adapter_row["reference_latency_ms"]
        )
        per_case.append(
            {
                "case_id": case_id,
                "adapter_status": adapter_row["status"],
                "native_status": native_row["status"],
                "adapter_speedup": adapter_row["speedup"],
                "native_speedup": native_row["speedup"],
                "native_over_adapter_speedup": (
                    native_row["speedup"] / adapter_row["speedup"]
                ),
                "adapter_candidate_latency_ms": adapter_row["latency_ms"],
                "native_candidate_latency_ms": native_row["latency_ms"],
                "native_over_adapter_candidate_latency": candidate_latency_ratio,
                "candidate_latency_percent_delta": (
                    candidate_latency_ratio - 1.0
                )
                * 100.0,
                "adapter_reference_latency_ms": adapter_row["reference_latency_ms"],
                "native_reference_latency_ms": native_row["reference_latency_ms"],
                "native_over_adapter_reference_latency": reference_latency_ratio,
            }
        )
    assert all(
        row["adapter_status"] == row["native_status"] == "PASSED"
        for row in per_case
    )
    timing_geo_mean = {
        catalog: _timing_geo_mean(results[catalog])
        for catalog in (ADAPTER_CATALOG, NATIVE_CATALOG)
    }
    candidate_latency_parity = _ratio_summary(
        [row["native_over_adapter_candidate_latency"] for row in per_case]
    )
    reference_latency_parity = _ratio_summary(
        [row["native_over_adapter_reference_latency"] for row in per_case]
    )

    final_status = client.request("GET", "/status", timeout=60)
    _assert_idle(final_status)
    _save(output / "status_final.json", final_status)
    _save(
        output / "summary.json",
        {
            "operator": args.operator,
            "server_commit": args.server_commit,
            "flaggems_commit": manifests[ADAPTER_CATALOG].get(
                "framework_revision", ""
            ),
            "candidate_sha256": hashlib.sha256(
                candidate_source.encode()
            ).hexdigest(),
            "candidate_original_sha256": hashlib.sha256(
                original_candidate_source.encode()
            ).hexdigest(),
            "candidate_source_kind": candidate_source_kind,
            "candidate_mode": args.candidate_mode,
            "candidate_abi_adapted": candidate_abi_adapted,
            "settings": settings,
            "case_ids_equal": True,
            "num_timing_cases": len(native_ids),
            "reference_as_solution": reference_result["status"],
            "statuses": {
                catalog: results[catalog]["status"]
                for catalog in (ADAPTER_CATALOG, NATIVE_CATALOG)
            },
            "reference_sources": {
                catalog: results[catalog]["reference_source"]
                for catalog in (ADAPTER_CATALOG, NATIVE_CATALOG)
            },
            "timing_statuses": {
                catalog: "PASSED"
                for catalog in (ADAPTER_CATALOG, NATIVE_CATALOG)
            },
            "timing_geo_mean": timing_geo_mean,
            "candidate_latency_native_over_adapter": candidate_latency_parity,
            "reference_latency_native_over_adapter": reference_latency_parity,
            "native_over_adapter_geo_mean": (
                timing_geo_mean[NATIVE_CATALOG]
                / timing_geo_mean[ADAPTER_CATALOG]
            ),
            "per_case": per_case,
            "final_scheduler": final_status["scheduler"],
        },
    )
    print(
        f"{args.operator}: native/adapter candidate latency ratio "
        f"geo={candidate_latency_parity['geometric_mean']:.6f} "
        f"delta={candidate_latency_parity['geometric_mean_percent_delta']:+.3f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()

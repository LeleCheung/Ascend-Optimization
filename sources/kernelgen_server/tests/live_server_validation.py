#!/usr/bin/env python3
"""Reusable functional and resilience validation for a live KernelGen Server."""

from __future__ import annotations

import argparse
import ast
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from kernelgen_server import (
    BoundEvaluateRequest,
    Catalog,
    EvaluationSettings,
    EvaluatorBinding,
    Implementation,
    SourceFile,
)
from kernelgen_server.client import ServerError, evaluate, preflight
from kernelgen_server.protocol.workload_call import definition_signature
from kernelgen_server.schema import EvaluationStatus


PORTABLE_TRITON_CANDIDATES = {
    "kernelgenbench_square": '''\
import torch
import triton
import triton.language as tl

BLOCK_SIZE = 256
MAX_GRID_SIZE = 65535


@triton.jit
def _square_kernel(x, output, n_elements, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_start in range(
        program_id * BLOCK_SIZE,
        n_elements,
        num_programs * BLOCK_SIZE,
    ):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x + offsets, mask=offsets < n_elements)
        tl.store(output + offsets, values * values, mask=offsets < n_elements)


def run(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = (max(1, min(triton.cdiv(n_elements, BLOCK_SIZE), MAX_GRID_SIZE)),)
    _square_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
''',
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Local copy of the native catalog used by the live server.",
    )
    parser.add_argument(
        "--catalog-name",
        help=(
            "Built-in catalog name sent in EvaluatorBinding; defaults to the "
            "manifest name or the local catalog directory name."
        ),
    )
    parser.add_argument("--expected-backend", required=True)
    parser.add_argument("--expected-device-count", type=int, required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    parser.add_argument("--expected-timing", default="triton")
    parser.add_argument("--definition", action="append", default=[])
    parser.add_argument("--all-definitions", action="store_true")
    parser.add_argument("--skip-definition", action="append", default=[])
    parser.add_argument(
        "--smoke-definition",
        help=(
            "Definition used by Triton and resilience checks; defaults to "
            "kernelgenbench_square when available, otherwise the first selected "
            "Definition."
        ),
    )
    parser.add_argument("--parallel-definitions", type=int, default=1)
    parser.add_argument("--skip-reference-preflight", action="store_true")
    parser.add_argument("--skip-triton", action="store_true")
    parser.add_argument("--skip-resilience", action="store_true")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--benchmark-ms", type=int, default=100)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_device_count < 1 or args.expected_workers < 1:
        parser.error("expected device and worker counts must be positive")
    if args.expected_workers < args.expected_device_count:
        parser.error("expected workers must be at least the expected device count")
    if args.concurrency < args.expected_device_count:
        parser.error("concurrency must be at least the expected device count")
    if args.parallel_definitions < 1:
        parser.error("definition parallelism must be positive")
    if args.all_definitions and args.definition:
        parser.error("--all-definitions and --definition are mutually exclusive")
    return args


def _status(server: str) -> dict[str, Any]:
    with urlopen(server.rstrip("/") + "/status", timeout=10) as response:
        return json.loads(response.read())


def _catalog_name(catalog: Catalog, explicit: str | None) -> str:
    if explicit:
        return explicit
    manifest_name = catalog.manifest.get("name")
    if isinstance(manifest_name, str) and manifest_name:
        return manifest_name
    return catalog.root.name


def _definition_names(
    catalog: Catalog,
    requested: list[str],
    *,
    all_definitions: bool,
    skipped: set[str],
) -> list[str]:
    available = set(catalog.operator_names)
    unknown_requested = set(requested).difference(available)
    unknown_skips = skipped.difference(available)
    if unknown_requested:
        raise ValueError(f"unknown definitions: {sorted(unknown_requested)}")
    if unknown_skips:
        raise ValueError(f"unknown skipped definitions: {sorted(unknown_skips)}")

    if all_definitions:
        names = list(catalog.operator_names)
    elif requested:
        names = list(requested)
    elif "kernelgenbench_square" in available:
        names = ["kernelgenbench_square"]
    else:
        names = [catalog.operator_names[0]]
    names = [name for name in names if name not in skipped]
    if not names:
        raise ValueError("no definitions remain after applying skips")
    return names


def _smoke_definition(
    catalog: Catalog,
    names: list[str],
    explicit: str | None,
) -> str:
    available = set(catalog.operator_names)
    if explicit:
        if explicit not in available:
            raise ValueError(f"unknown smoke definition: {explicit!r}")
        return explicit
    if "kernelgenbench_square" in available:
        return "kernelgenbench_square"
    return names[0]


def _binding(catalog_name: str, definition: str) -> EvaluatorBinding:
    return EvaluatorBinding(
        catalog_name=catalog_name,
        definition=definition,
    )


def _implementation(
    definition: Any,
    source: str,
    *,
    language: str = "python",
    suffix: str,
) -> Implementation:
    return Implementation(
        name=f"live-validation-{definition.name}-{suffix}",
        definition=definition.name,
        language=language,
        entrypoint="main.py::run",
        sources=[SourceFile(path="main.py", content=source)],
    )


def _request(
    catalog_name: str,
    definition: Any,
    implementation: Implementation,
    settings: EvaluationSettings,
) -> BoundEvaluateRequest:
    return BoundEvaluateRequest(
        binding=_binding(catalog_name, definition.name),
        implementation=implementation,
        settings=settings,
    )


def _oracle_functions(definition: Any) -> set[str]:
    source = definition.reference
    if not isinstance(source, str):
        raise ValueError("native Definition is missing its trusted oracle source")
    module = ast.parse(source, filename="<oracle.py>")
    return {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _reference_candidate_source(
    definition: Any,
    phase: str,
    reference_source: str = "primary",
) -> str:
    source = definition.reference
    if not isinstance(source, str):
        raise ValueError("native Definition is missing its trusted oracle source")
    functions = _oracle_functions(definition)
    if reference_source == "torch_fallback":
        symbol = "torch_run"
    elif definition.api_version == "v6.0":
        symbol = "run"
    else:
        preferred = "correctness_run" if phase == "correctness" else "timing_run"
        symbol = preferred if preferred in functions else "run"
    if symbol not in functions:
        raise ValueError(f"oracle.py does not define {symbol}()")
    if symbol == "run":
        return source
    return source.rstrip() + f"\n\nrun = {symbol}\n"


def _reference_variants(operator: Any) -> list[tuple[str, str]]:
    phase = "correctness" if operator.correctness_workloads else "timing"
    variants = [
        (
            "primary",
            _reference_candidate_source(operator.definition, phase, "primary"),
        )
    ]
    if (
        operator.definition.api_version == "v6.2"
        and "torch_run" in _oracle_functions(operator.definition)
    ):
        variants.append(
            (
                "torch_fallback",
                _reference_candidate_source(
                    operator.definition,
                    phase,
                    "torch_fallback",
                ),
            )
        )
    return variants


def _reference_check(
    catalog: Catalog,
    catalog_name: str,
    name: str,
    settings: EvaluationSettings,
    server: str,
    run_preflight: bool,
) -> tuple[str, dict[str, Any]]:
    operator = catalog.load(name)
    attempts = []
    for variant, source in _reference_variants(operator):
        implementation = _implementation(
            operator.definition,
            source,
            suffix=f"reference-{variant}",
        )
        request = _request(catalog_name, operator.definition, implementation, settings)
        attempt: dict[str, Any] = {"source": variant}
        try:
            preflight_result = (
                preflight(request, server) if run_preflight else {"status": "SKIPPED"}
            )
            attempt["preflight"] = preflight_result
            preflight_passed = (
                not run_preflight
                or preflight_result.get("status") == EvaluationStatus.PASSED.value
            )
            if not preflight_passed:
                attempt["passed"] = False
                attempts.append(attempt)
                continue

            evaluate_started = time.perf_counter()
            evaluation = evaluate(request, server)
            attempt["evaluate_elapsed_seconds"] = (
                time.perf_counter() - evaluate_started
            )
            attempt["evaluation"] = evaluation.model_dump(mode="json")
            attempt["passed"] = evaluation.status == EvaluationStatus.PASSED
        except Exception:
            attempt["passed"] = False
            attempt["request_error"] = traceback.format_exc()
        attempts.append(attempt)
        if attempt["passed"]:
            return name, {
                "passed": True,
                "selected_source": variant,
                "attempts": attempts,
            }
    return name, {"passed": False, "selected_source": None, "attempts": attempts}


def _reference_checks(
    catalog: Catalog,
    catalog_name: str,
    names: list[str],
    settings: EvaluationSettings,
    server: str,
    parallel_definitions: int,
    run_preflight: bool,
) -> dict[str, Any]:
    results = {}
    with ThreadPoolExecutor(max_workers=parallel_definitions) as executor:
        futures = {
            executor.submit(
                _reference_check,
                catalog,
                catalog_name,
                name,
                settings,
                server,
                run_preflight,
            ): name
            for name in names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, result = future.result()
            except Exception:
                result = {
                    "passed": False,
                    "request_error": traceback.format_exc(),
                }
            results[name] = result
    return {name: results[name] for name in names}


def _triton_check(
    catalog: Catalog,
    catalog_name: str,
    definition_name: str,
    settings: EvaluationSettings,
    server: str,
) -> dict[str, Any]:
    try:
        source = PORTABLE_TRITON_CANDIDATES[definition_name]
    except KeyError as exc:
        raise ValueError(
            f"no portable Triton smoke is registered for {definition_name!r}; "
            "choose kernelgenbench_square or pass --skip-triton"
        ) from exc
    operator = catalog.load(definition_name)
    implementation = _implementation(
        operator.definition,
        source,
        language="triton",
        suffix="triton",
    )
    request = _request(catalog_name, operator.definition, implementation, settings)
    preflight_result = preflight(request, server)
    evaluation = evaluate(request, server)
    return {
        "passed": (
            preflight_result.get("status") == EvaluationStatus.PASSED.value
            and evaluation.status == EvaluationStatus.PASSED
        ),
        "definition": definition_name,
        "preflight": preflight_result,
        "evaluation": evaluation.model_dump(mode="json"),
    }


def _failure_source(definition: Any, statement: str) -> str:
    signature = definition_signature(definition)
    return f"def run{signature}:\n    {statement}\n"


def _delayed_source(source: str) -> str:
    return source.rstrip() + "\n\nimport time as _time\n_time.sleep(1.0)\n"


def _idle_scheduler(scheduler: dict[str, Any], expected_device_count: int) -> bool:
    return (
        scheduler.get("device_slots") == expected_device_count
        and scheduler.get("healthy") == expected_device_count
        and scheduler.get("available") == expected_device_count
        and scheduler.get("active") == 0
        and scheduler.get("waiting") == 0
        and scheduler.get("checking") == 0
        and scheduler.get("broken") == 0
        and scheduler.get("max_active", expected_device_count)
        <= expected_device_count
    )


def _concurrency_and_failure_checks(
    catalog: Catalog,
    catalog_name: str,
    definition_name: str,
    reference_source: str,
    settings: EvaluationSettings,
    server: str,
    concurrency: int,
    expected_device_count: int,
) -> dict[str, Any]:
    operator = catalog.load(definition_name)
    phase = "correctness" if operator.correctness_workloads else "timing"
    passing_source = _reference_candidate_source(
        operator.definition,
        phase,
        reference_source,
    )
    runtime_error_source = _failure_source(
        operator.definition,
        'raise RuntimeError("intentional live-server validation failure")',
    )
    requests = []
    for index in range(concurrency):
        source = runtime_error_source if index == 0 else _delayed_source(passing_source)
        implementation = _implementation(
            operator.definition,
            source,
            suffix=f"concurrent-{index}",
        )
        requests.append(
            _request(catalog_name, operator.definition, implementation, settings)
        )

    responses = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(evaluate, request, server): index
            for index, request in enumerate(requests)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                response = future.result()
                responses.append(
                    {
                        "index": index,
                        "status": response.status.value,
                        "device": response.device,
                    }
                )
            except Exception:
                responses.append(
                    {
                        "index": index,
                        "status": "REQUEST_ERROR",
                        "device": None,
                        "request_error": traceback.format_exc(),
                    }
                )
    responses.sort(key=lambda item: item["index"])
    runtime_errors = sum(
        item["status"] == EvaluationStatus.RUNTIME_ERROR.value for item in responses
    )
    passed = sum(
        item["status"] == EvaluationStatus.PASSED.value for item in responses
    )
    devices = sorted(
        {item["device"] for item in responses if item["device"] is not None}
    )

    pre_crash_status = _status(server)
    pre_crash_scheduler = pre_crash_status.get("scheduler", {})
    crash_request = _request(
        catalog_name,
        operator.definition,
        _implementation(
            operator.definition,
            _failure_source(
                operator.definition,
                'import os; os._exit(17)',
            ),
            suffix="hard-crash",
        ),
        settings,
    )
    hard_crash_isolated = False
    hard_crash_error = ""
    try:
        evaluate(crash_request, server)
    except ServerError as exc:
        hard_crash_isolated = True
        hard_crash_error = str(exc)
    post_crash_status = _status(server)
    post_crash_scheduler = post_crash_status.get("scheduler", {})
    hard_crash_slot_recovered = (
        post_crash_scheduler.get("incidents", 0)
        >= pre_crash_scheduler.get("incidents", 0) + 1
        and post_crash_scheduler.get("recovered", 0)
        >= pre_crash_scheduler.get("recovered", 0) + 1
        and _idle_scheduler(post_crash_scheduler, expected_device_count)
    )

    recovery_request = _request(
        catalog_name,
        operator.definition,
        _implementation(
            operator.definition,
            passing_source,
            suffix="recovery",
        ),
        settings,
    )
    recovery_payload: dict[str, Any]
    try:
        recovery = evaluate(recovery_request, server)
        recovery_passed = recovery.status == EvaluationStatus.PASSED
        recovery_payload = recovery.model_dump(mode="json")
    except Exception:
        recovery_passed = False
        recovery_payload = {"request_error": traceback.format_exc()}
    final_status = _status(server)
    final_scheduler_idle = _idle_scheduler(
        final_status.get("scheduler", {}), expected_device_count
    )

    return {
        "passed": (
            runtime_errors == 1
            and passed == concurrency - 1
            and len(devices) >= expected_device_count
            and hard_crash_isolated
            and hard_crash_slot_recovered
            and recovery_passed
            and final_scheduler_idle
        ),
        "definition": definition_name,
        "reference_source": reference_source,
        "responses": responses,
        "unique_devices": devices,
        "pre_crash_status": pre_crash_status,
        "hard_crash_isolated": hard_crash_isolated,
        "hard_crash_error": hard_crash_error,
        "hard_crash_slot_recovered": hard_crash_slot_recovered,
        "post_crash_status": post_crash_status,
        "recovery": recovery_payload,
        "final_status": final_status,
        "final_scheduler_idle": final_scheduler_idle,
    }


def _run_check(function: Any, *args: Any) -> dict[str, Any]:
    try:
        return function(*args)
    except Exception:
        return {"passed": False, "request_error": traceback.format_exc()}


def main() -> int:
    args = _parse_args()
    catalog = Catalog(args.catalog)
    if catalog.evaluator != "native":
        raise ValueError(
            "live_server_validation.py requires a native catalog; framework "
            "adapter catalogs have separate adapter E2E validation"
        )
    catalog_name = _catalog_name(catalog, args.catalog_name)
    names = _definition_names(
        catalog,
        args.definition,
        all_definitions=args.all_definitions,
        skipped=set(args.skip_definition),
    )
    smoke_definition = _smoke_definition(catalog, names, args.smoke_definition)

    status = _status(args.server)
    scheduler = status.get("scheduler", {})
    status_passed = (
        status.get("status") == "ok"
        and status.get("api_version") == "v6.2"
        and status.get("backend") == args.expected_backend
        and len(status.get("devices", [])) == args.expected_device_count
        and status.get("workers") == args.expected_workers
        and status.get("timing") == args.expected_timing
        and _idle_scheduler(scheduler, args.expected_device_count)
    )
    settings = EvaluationSettings(
        warmup_ms=args.warmup_ms,
        benchmark_ms=args.benchmark_ms,
        timeout_seconds=args.request_timeout,
    )
    references = _reference_checks(
        catalog,
        catalog_name,
        names,
        settings,
        args.server,
        args.parallel_definitions,
        not args.skip_reference_preflight,
    )
    smoke_reference_source = references.get(smoke_definition, {}).get(
        "selected_source"
    ) or "primary"
    result = {
        "catalog_name": catalog_name,
        "smoke_definition": smoke_definition,
        "status": {"passed": status_passed, "value": status},
        "skipped_definitions": sorted(set(args.skip_definition)),
        "references": references,
        "triton": (
            {"passed": True, "status": "SKIPPED"}
            if args.skip_triton
            else _run_check(
                _triton_check,
                catalog,
                catalog_name,
                smoke_definition,
                settings,
                args.server,
            )
        ),
        "resilience": (
            {"passed": True, "status": "SKIPPED"}
            if args.skip_resilience
            else _run_check(
                _concurrency_and_failure_checks,
                catalog,
                catalog_name,
                smoke_definition,
                smoke_reference_source,
                settings,
                args.server,
                args.concurrency,
                args.expected_device_count,
            )
        ),
    }
    result["passed"] = (
        result["status"]["passed"]
        and all(item["passed"] for item in result["references"].values())
        and result["triton"]["passed"]
        and result["resilience"]["passed"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

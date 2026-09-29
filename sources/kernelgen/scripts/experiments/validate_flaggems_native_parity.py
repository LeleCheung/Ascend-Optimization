#!/usr/bin/env python3
"""Validate the same candidate through FlagGems and Native v6.2 bindings."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import urllib.error
import urllib.request
from typing import Any, Callable


SCHEMA_VERSION = "kernelgen.flaggems-native-parity/v1"
TERMINAL_VERDICTS = {
    "PASSED",
    "CANDIDATE_FAILED",
    "PARITY_FAILED",
    "BLOCKED",
}
_SAFE_OPERATOR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class GateError(RuntimeError):
    verdict = "BLOCKED"


class CandidateError(GateError):
    verdict = "CANDIDATE_FAILED"


class ParityError(GateError):
    verdict = "PARITY_FAILED"


class JsonClient:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.server_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GateError(f"{path} HTTP {exc.code}: {detail}") from exc
        except (OSError, ValueError) as exc:
            raise GateError(f"{path} transport/JSON error: {exc}") from exc
        if not isinstance(value, dict):
            raise GateError(f"{path} response root is not an object")
        return value


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_operators(path: pathlib.Path) -> list[str]:
    operators: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value.startswith("|") and value.count("|") >= 2:
            value = value.split("|", 2)[1].strip().strip("`")
        if (
            not value
            or value in {"算子", "算子名称", "operator", "Operator"}
            or set(value) <= {"-", ":", " "}
        ):
            continue
        if not _SAFE_OPERATOR.fullmatch(value):
            raise GateError(f"invalid operator name in {path}: {value!r}")
        if value not in seen:
            seen.add(value)
            operators.append(value)
    if not operators:
        raise GateError(f"operator file is empty: {path}")
    return operators


def _scheduler_gate(status: dict[str, Any], *, require_idle: bool) -> None:
    if status.get("api_version") != "v6.2":
        raise GateError(
            f"server api_version={status.get('api_version')!r}, expected 'v6.2'"
        )
    scheduler = status.get("scheduler")
    if not isinstance(scheduler, dict):
        raise GateError("/status has no scheduler")
    slots = scheduler.get("device_slots")
    healthy = scheduler.get("healthy")
    if (
        not isinstance(slots, int)
        or slots < 1
        or healthy != slots
        or scheduler.get("checking") != 0
        or scheduler.get("broken") != 0
    ):
        raise GateError(f"scheduler is unhealthy: {scheduler}")
    if require_idle and (
        scheduler.get("active") != 0
        or scheduler.get("waiting") != 0
        or scheduler.get("available") != slots
    ):
        raise GateError(f"scheduler is not idle: {scheduler}")


def _inspect_contract(payload: dict[str, Any], catalog: str) -> dict[str, Any]:
    case_list = payload.get("case_list")
    cases = case_list.get("cases") if isinstance(case_list, dict) else None
    if not isinstance(cases, list) or not cases:
        raise GateError(f"{catalog} /inspect returned no cases")
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if (
        len(case_ids) != len(cases)
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(set(case_ids)) != len(case_ids)
    ):
        raise GateError(f"{catalog} /inspect returned invalid case_id values")
    contract = payload.get("candidate_contract")
    if (
        not isinstance(contract, dict)
        or not contract.get("entrypoint")
        or not contract.get("signature")
    ):
        raise GateError(f"{catalog} /inspect returned no candidate contract")
    fingerprint = payload.get("benchmark_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise GateError(f"{catalog} /inspect returned no benchmark fingerprint")
    case_fingerprint = case_list.get("benchmark_fingerprint")
    if case_fingerprint not in {None, fingerprint}:
        raise GateError(f"{catalog} /inspect has inconsistent fingerprints")
    return {
        "case_ids": case_ids,
        "contract": contract,
        "fingerprint": fingerprint,
    }


def _validate_evaluation(
    payload: dict[str, Any],
    *,
    catalog: str,
    timing_case_ids: list[str],
) -> dict[str, Any]:
    rows = payload.get("per_workload")
    if not isinstance(rows, list) or not rows:
        raise GateError(f"{catalog} PASSED evaluation returned no workloads")
    if payload.get("num_workloads") != len(rows):
        raise GateError(f"{catalog} num_workloads differs from per_workload")
    if payload.get("num_passed") != len(rows):
        raise GateError(f"{catalog} num_passed differs from per_workload")
    if any(row.get("status") != "PASSED" for row in rows):
        raise GateError(f"{catalog} PASSED evaluation contains failed workloads")
    identities = [[row.get("uuid"), row.get("phase")] for row in rows]
    if any(
        not isinstance(identity, str) or not identity
        for pair in identities
        for identity in pair
    ) or len({tuple(pair) for pair in identities}) != len(identities):
        raise GateError(f"{catalog} evaluation returned invalid workload identities")
    correctness = [row for row in rows if row.get("phase") == "correctness"]
    timing = [row for row in rows if row.get("phase") == "timing"]
    if not correctness:
        raise GateError(f"{catalog} evaluation returned no correctness workloads")
    timing_ids = [row.get("uuid") for row in timing]
    if timing_ids != timing_case_ids:
        raise ParityError(
            f"{catalog} evaluate timing case order differs from /inspect"
        )
    for row in timing:
        for key in ("speedup", "latency_ms", "reference_latency_ms"):
            value = row.get(key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise GateError(
                    f"{catalog} invalid {key} for timing case {row.get('uuid')}"
                )
    geo_mean = payload.get("geo_mean")
    if (
        not isinstance(geo_mean, (int, float))
        or not math.isfinite(geo_mean)
        or geo_mean <= 0
    ):
        raise GateError(f"{catalog} PASSED evaluation has no positive geo_mean")
    if payload.get("is_hack"):
        raise CandidateError(
            f"{catalog} marked candidate as hack: "
            f"{payload.get('hack_reason') or 'unspecified'}"
        )
    return {
        "geo_mean": float(geo_mean),
        "num_workloads": len(rows),
        "num_correctness": len(correctness),
        "num_timing": len(timing),
        "workload_identities": identities,
        "timing_case_ids": timing_ids,
    }


def _implementation(operator: str, source: str, language: str) -> dict[str, Any]:
    return {
        "name": f"{operator}-flaggems-native-parity",
        "definition": operator,
        "language": language,
        "entrypoint": "main.py::run",
        "sources": [{"path": "main.py", "content": source}],
    }


def _request(
    operator: str,
    catalog: str,
    implementation: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "api_version": "v6.2",
        "binding": {"catalog_name": catalog, "definition": operator},
        "implementation": implementation,
        "settings": settings,
    }


def _classify_statuses(
    responses: dict[str, dict[str, Any]],
    *,
    stage: str,
) -> None:
    statuses = {catalog: value.get("status") for catalog, value in responses.items()}
    if all(status == "PASSED" for status in statuses.values()):
        return
    if any(
        status in {"TIMEOUT", "SUSPECTED_DEVICE_ERROR"}
        for status in statuses.values()
    ):
        raise GateError(f"{stage} blocked: {statuses}")
    if len(set(statuses.values())) == 1:
        raise CandidateError(f"candidate failed {stage} in both catalogs: {statuses}")
    raise ParityError(f"catalogs disagree at {stage}: {statuses}")


def validate_one(
    *,
    operator: str,
    candidate_path: pathlib.Path,
    server_url: str,
    output_root: pathlib.Path,
    adapter_catalog: str,
    native_catalog: str,
    settings: dict[str, Any],
    language: str,
    transport_timeout: float,
    force: bool,
    client_factory: Callable[[str], Any] = JsonClient,
) -> dict[str, Any]:
    output = output_root / "operators" / operator
    summary_path = output / "summary.json"
    candidate_sha = _sha256(candidate_path)
    identity = {
        "operator": operator,
        "candidate_sha256": candidate_sha,
        "server_url": server_url.rstrip("/"),
        "adapter_catalog": adapter_catalog,
        "native_catalog": native_catalog,
    }
    if summary_path.is_file() and not force:
        existing = _read_json(summary_path)
        if all(existing.get(key) == value for key, value in identity.items()) and (
            existing.get("verdict") in TERMINAL_VERDICTS
        ):
            return existing
        raise GateError(
            f"existing summary identity differs for {operator}; use a new output root"
        )
    output.mkdir(parents=True, exist_ok=True)
    source = candidate_path.read_text(encoding="utf-8")
    implementation = _implementation(operator, source, language)
    client = client_factory(server_url)
    catalogs = (adapter_catalog, native_catalog)
    stage = "inspect"
    try:
        inspections: dict[str, dict[str, Any]] = {}
        contracts: dict[str, dict[str, Any]] = {}
        for catalog in catalogs:
            payload = client.request(
                "POST",
                "/inspect",
                {
                    "api_version": "v6.2",
                    "binding": {
                        "catalog_name": catalog,
                        "definition": operator,
                    },
                },
                timeout=transport_timeout,
            )
            inspections[catalog] = payload
            contracts[catalog] = _inspect_contract(payload, catalog)
            _atomic_json(output / f"inspect_{catalog}.json", payload)
        if contracts[adapter_catalog]["case_ids"] != contracts[native_catalog][
            "case_ids"
        ]:
            raise ParityError("adapter and native /inspect case_id order differs")
        adapter_contract = contracts[adapter_catalog]["contract"]
        native_contract = contracts[native_catalog]["contract"]
        for key in ("entrypoint", "signature"):
            if adapter_contract.get(key) != native_contract.get(key):
                raise ParityError(
                    f"adapter and native candidate contract {key} differs"
                )

        stage = "preflight"
        preflights = {
            catalog: client.request(
                "POST",
                "/preflight",
                _request(operator, catalog, implementation, settings),
                timeout=transport_timeout,
            )
            for catalog in catalogs
        }
        for catalog, payload in preflights.items():
            _atomic_json(output / f"preflight_{catalog}.json", payload)
            if payload.get("api_version") != "v6.2":
                raise GateError(f"{catalog} preflight returned unexpected api_version")
            if payload.get("benchmark_fingerprint") != contracts[catalog][
                "fingerprint"
            ]:
                raise GateError(f"{catalog} fingerprint changed at preflight")
            if payload.get("status") == "PASSED" and payload.get(
                "num_cases"
            ) != len(contracts[catalog]["case_ids"]):
                raise GateError(f"{catalog} preflight case count changed")
        _classify_statuses(preflights, stage=stage)

        stage = "evaluate"
        evaluations = {
            catalog: client.request(
                "POST",
                "/evaluate",
                _request(operator, catalog, implementation, settings),
                timeout=transport_timeout,
            )
            for catalog in catalogs
        }
        for catalog, payload in evaluations.items():
            _atomic_json(output / f"evaluate_{catalog}.json", payload)
            if payload.get("api_version") != "v6.2":
                raise GateError(f"{catalog} evaluate returned unexpected api_version")
        _classify_statuses(evaluations, stage=stage)
        metrics = {
            catalog: _validate_evaluation(
                evaluations[catalog],
                catalog=catalog,
                timing_case_ids=contracts[catalog]["case_ids"],
            )
            for catalog in catalogs
        }
        if metrics[adapter_catalog]["num_correctness"] != metrics[native_catalog][
            "num_correctness"
        ]:
            raise ParityError("adapter and native correctness workload counts differ")
        if metrics[adapter_catalog]["timing_case_ids"] != metrics[native_catalog][
            "timing_case_ids"
        ]:
            raise ParityError("adapter and native timing workload identities differ")
        summary = {
            "schema_version": SCHEMA_VERSION,
            **identity,
            "verdict": "PASSED",
            "stage": "complete",
            "reason": (
                "same candidate passed matching timing cases and equal correctness "
                "coverage counts in both catalogs"
            ),
            "correctness_identity_policy": (
                "catalog-local names; semantic coverage is audited by the extraction "
                "accuracy_coverage.json"
            ),
            "case_ids": contracts[adapter_catalog]["case_ids"],
            "metrics": metrics,
            "completed_at": _now(),
        }
    except GateError as exc:
        summary = {
            "schema_version": SCHEMA_VERSION,
            **identity,
            "verdict": exc.verdict,
            "stage": stage,
            "reason": str(exc),
            "completed_at": _now(),
        }
    _atomic_json(summary_path, summary)
    return summary


def run(
    args: argparse.Namespace,
    *,
    client_factory: Callable[[str], Any] = JsonClient,
) -> tuple[int, dict[str, Any]]:
    operators = list(args.operator)
    if args.operator_file is not None:
        operators.extend(read_operators(args.operator_file.resolve()))
    operators = list(dict.fromkeys(operators))
    if not operators:
        raise GateError("provide --operator or --operator-file")
    invalid = [operator for operator in operators if not _SAFE_OPERATOR.fullmatch(operator)]
    if invalid:
        raise GateError(f"invalid operator names: {invalid}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    client = client_factory(args.server_url)
    status_before = client.request("GET", "/status", timeout=60)
    _atomic_json(output_root / "status_before.json", status_before)
    _scheduler_gate(status_before, require_idle=True)
    if args.expected_backend and status_before.get("backend") != args.expected_backend:
        raise GateError(
            f"server backend={status_before.get('backend')!r}, "
            f"expected {args.expected_backend!r}"
        )

    settings = {
        "warmup_ms": args.warmup_ms,
        "benchmark_ms": args.benchmark_ms,
        "num_trials": args.num_trials,
        "timeout_seconds": args.timeout_seconds,
    }
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.max_workers, len(operators))
    ) as pool:
        futures = {}
        for operator in operators:
            candidate_path = args.candidate_root.resolve() / f"{operator}.py"
            if not candidate_path.is_file():
                results[operator] = {
                    "schema_version": SCHEMA_VERSION,
                    "operator": operator,
                    "candidate_sha256": None,
                    "server_url": args.server_url.rstrip("/"),
                    "adapter_catalog": args.adapter_catalog,
                    "native_catalog": args.native_catalog,
                    "verdict": "BLOCKED",
                    "stage": "input",
                    "reason": f"candidate file is missing: {candidate_path}",
                    "completed_at": _now(),
                }
                _atomic_json(
                    output_root / "operators" / operator / "summary.json",
                    results[operator],
                )
                continue
            future = pool.submit(
                validate_one,
                operator=operator,
                candidate_path=candidate_path,
                server_url=args.server_url,
                output_root=output_root,
                adapter_catalog=args.adapter_catalog,
                native_catalog=args.native_catalog,
                settings=settings,
                language=args.language,
                transport_timeout=args.timeout_seconds + 120,
                force=args.force,
                client_factory=client_factory,
            )
            futures[future] = operator
        for future in concurrent.futures.as_completed(futures):
            operator = futures[future]
            try:
                results[operator] = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate one operator
                results[operator] = {
                    "schema_version": SCHEMA_VERSION,
                    "operator": operator,
                    "candidate_sha256": None,
                    "server_url": args.server_url.rstrip("/"),
                    "adapter_catalog": args.adapter_catalog,
                    "native_catalog": args.native_catalog,
                    "verdict": "BLOCKED",
                    "stage": "worker",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "completed_at": _now(),
                }
                _atomic_json(
                    output_root / "operators" / operator / "summary.json",
                    results[operator],
                )
            print(f"{operator}: {results[operator]['verdict']}", flush=True)

    status_after = client.request("GET", "/status", timeout=60)
    _atomic_json(output_root / "status_after.json", status_after)
    _scheduler_gate(status_after, require_idle=True)
    ordered = [results[operator] for operator in operators]
    counts: dict[str, int] = {}
    for result in ordered:
        verdict = str(result["verdict"])
        counts[verdict] = counts.get(verdict, 0) + 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "server_url": args.server_url.rstrip("/"),
        "backend": status_before.get("backend"),
        "adapter_catalog": args.adapter_catalog,
        "native_catalog": args.native_catalog,
        "settings": settings,
        "operator_count": len(operators),
        "counts": counts,
        "results": ordered,
        "completed_at": _now(),
    }
    _atomic_json(output_root / "report.json", report)
    return (0 if counts.get("PASSED") == len(operators) else 1), report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--operator-file", type=pathlib.Path)
    parser.add_argument("--candidate-root", type=pathlib.Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--adapter-catalog", default="flaggems-adapter-definitions"
    )
    parser.add_argument("--native-catalog", default="flaggems-native")
    parser.add_argument("--expected-backend")
    parser.add_argument("--language", default="triton")
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--benchmark-ms", type=int, default=100)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1500)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.warmup_ms,
        args.benchmark_ms,
        args.num_trials,
        args.timeout_seconds,
        args.max_workers,
    ) < 1:
        parser.error("timing and worker values must be positive")
    return args


def main() -> int:
    try:
        exit_code, report = run(_parse_args())
    except GateError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print(json.dumps({"counts": report["counts"]}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

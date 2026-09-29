#!/usr/bin/env python3
"""Validate one NVIDIA Triton candidate on multiple KGS targets.

The script freezes one candidate copy in the run root, stores one artifact directory
per target, and updates Kernel Todo V2 results only after a target has an authoritative
verdict.  Transport, server, baseline, and protocol failures remain blocked; they
are never rewritten as candidate specialization failures.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from kernelgen.tools.kernel_todo_v2_pipeline import (
    KernelTodoV2PipelineResults,
    refresh_markdown_summary,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CATALOG_NAME = "flaggems-adapter-definitions"
ROW_OPERATOR = re.compile(r"`([^`]+)`")
MARK_NOTE = "【NVIDIA 跨芯片验证："
ALL_TARGET_CHIPS = (
    "kunlunxin",
    "haiguang",
    "moer",
    "muxi",
    "huawei",
    "tianshu",
    "pingtouge",
    "suiyuan",
)
EXPECTED_BACKENDS = {
    "kunlunxin": "kunlunxin",
    "haiguang": "hygon",
    "moer": "musa",
    "muxi": "metax",
    "huawei": "npu",
    "tianshu": "iluvatar",
    "pingtouge": "thead",
    "suiyuan": "enflame",
}
TERMINAL_VERDICTS = {"REUSABLE", "NEEDS_SPECIALIZATION", "BLOCKED"}
VERDICT_LABELS = {
    "REUSABLE": "可直接复用",
    "NEEDS_SPECIALIZATION": "需要特化",
    "BLOCKED": "验证阻塞",
}
PORTABILITY_CHIPS = (
    ("muxi", "沐曦"),
    ("haiguang", "海光"),
    ("moer", "摩尔线程"),
    ("tianshu", "天数智芯"),
    ("pingtouge", "平头哥"),
    ("huawei", "昇腾"),
    ("kunlunxin", "昆仑芯"),
    ("suiyuan", "燧原"),
)
PORTABILITY_SCHEMA = "kernelgen.nvidia-portability-results/v1"


@dataclasses.dataclass(frozen=True)
class Target:
    chip: str
    server_url: str


class ValidationError(RuntimeError):
    pass


class HttpError(ValidationError):
    pass


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
            raise HttpError(f"{path} HTTP {exc.code}: {detail}") from exc
        except (OSError, ValueError) as exc:
            raise HttpError(f"{path} transport/JSON error: {exc}") from exc
        if not isinstance(value, dict):
            raise HttpError(f"{path} response root is not an object")
        return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one qualified NVIDIA candidate across KGS targets."
    )
    parser.add_argument(
        "--operator", required=True, help="source operator in results.md"
    )
    parser.add_argument("--candidate-code", type=pathlib.Path)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="CHIP=URL",
        help="target results directory name and an already reachable KGS URL",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="CHIP=REASON",
        help="explicitly account for a target excluded by current experiment policy",
    )
    parser.add_argument("--run-root", type=pathlib.Path, required=True)
    parser.add_argument("--repo-root", type=pathlib.Path, default=REPO_ROOT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="direct-reuse threshold; Kernel Todo V2 requires 0.8",
    )
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--benchmark-ms", type=int, default=100)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1500)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--force", action="store_true", help="rerun existing target summaries"
    )
    parser.add_argument("--no-update-results", action="store_true")
    parser.add_argument("--overwrite-existing-code", action="store_true")
    parser.add_argument(
        "--allow-unqualified-source",
        action="store_true",
        help="development only: bypass NVIDIA success/threshold gates",
    )
    parser.add_argument(
        "--allow-partial-target-set",
        action="store_true",
        help=(
            "development only: do not require every non-NVIDIA chip to be "
            "targeted/excluded"
        ),
    )
    args = parser.parse_args(argv)
    if not math.isclose(args.threshold, 0.8):
        parser.error("Kernel Todo V2 uses a fixed --threshold of 0.8")
    if min(
        args.warmup_ms,
        args.benchmark_ms,
        args.num_trials,
        args.timeout_seconds,
        args.max_workers,
    ) <= 0:
        parser.error("timing and worker values must be positive")
    return args


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_spec(value: str, kind: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValidationError(f"{kind} must use CHIP=VALUE: {value!r}")
    chip, data = value.split("=", 1)
    if chip not in ALL_TARGET_CHIPS or not data.strip():
        raise ValidationError(f"invalid {kind}: {value!r}")
    return chip, data.strip()


def _targets(args: argparse.Namespace) -> tuple[list[Target], dict[str, str]]:
    parsed = [_split_spec(value, "--target") for value in args.target]
    excludes = dict(_split_spec(value, "--exclude") for value in args.exclude)
    chips = [chip for chip, _ in parsed]
    if len(chips) != len(set(chips)) or len(excludes) != len(args.exclude):
        raise ValidationError("target and exclude chips must be unique")
    overlap = set(chips) & set(excludes)
    if overlap:
        raise ValidationError(
            f"chips cannot be targeted and excluded: {sorted(overlap)}"
        )
    accounted = set(chips) | set(excludes)
    missing = set(ALL_TARGET_CHIPS) - accounted
    if missing and not args.allow_partial_target_set:
        raise ValidationError(
            "every non-NVIDIA chip must be targeted or explicitly excluded; "
            f"missing={sorted(missing)}"
        )
    if not parsed:
        raise ValidationError("at least one --target is required")
    for chip, url in parsed:
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValidationError(f"invalid KGS URL for {chip}: {url!r}")
    return [Target(chip, url) for chip, url in parsed], excludes


def _results_rows(lines: list[str]) -> dict[str, tuple[int, list[str]]]:
    rows: dict[str, tuple[int, list[str]]] = {}
    for index, line in enumerate(lines):
        if not line.startswith("| `"):
            continue
        columns = [item.strip() for item in line.strip().strip("|").split("|")]
        match = ROW_OPERATOR.search(columns[0])
        if match and len(columns) >= 9:
            rows[match.group(1)] = (index, columns)
    return rows


def _load_results(
    path: pathlib.Path,
) -> tuple[list[str], dict[str, tuple[int, list[str]]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines, _results_rows(lines)


def _speed(columns: list[str]) -> float | None:
    try:
        value = float(columns[4].removesuffix("x"))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _refresh_summary(
    lines: list[str],
    rows: dict[str, tuple[int, list[str]]],
    pipeline: KernelTodoV2PipelineResults | None = None,
) -> None:
    del rows
    refresh_markdown_summary(lines, pipeline)


def _inventory_entry(repo_root: pathlib.Path, source_operator: str) -> dict[str, Any]:
    inventory = _read_json(
        repo_root / "kernel_todo_v2/pytest_conversion_inventory.json"
    ).get("operators")
    matches = [
        entry
        for entry in inventory or []
        if entry.get("source_operator") == source_operator
    ]
    if len(matches) != 1:
        raise ValidationError(
            "inventory needs exactly one source row for "
            f"{source_operator!r}: {len(matches)}"
        )
    return matches[0]


def _link_target(results_path: pathlib.Path, value: str) -> pathlib.Path | None:
    matches = re.findall(r"\]\(([^)]+)\)", value)
    if not matches:
        return None
    return (results_path.parent / matches[0]).resolve()


def _source_candidate(
    args: argparse.Namespace,
    repo_root: pathlib.Path,
    source_operator: str,
    definition: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    results_path = repo_root / "kernel_todo_v2/nvidia/results.md"
    _, rows = _load_results(results_path)
    if source_operator not in rows:
        raise ValidationError(f"NVIDIA results row not found: {source_operator}")
    columns = rows[source_operator][1]
    speed = _speed(columns)
    qualified = (
        columns[3] == "成功"
        and speed is not None
        and speed >= args.threshold
    )
    if not qualified and not args.allow_unqualified_source:
        raise ValidationError(
            "NVIDIA source must be successful and meet the threshold: "
            f"status={columns[3]!r}, speed={speed!r}"
        )
    candidate = (
        args.candidate_code.resolve()
        if args.candidate_code
        else _link_target(results_path, columns[8])
    )
    if candidate is None or not candidate.is_file():
        raise ValidationError(
            "candidate file is missing; pass --candidate-code or fix NVIDIA code_path"
        )
    source = candidate.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(candidate))
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run"
        for node in module.body
    ):
        raise ValidationError("candidate must define a top-level run function")
    return candidate, {
        "results_path": str(results_path),
        "status": columns[3],
        "speedup": speed,
        "hack": columns[5],
        "definition": definition,
    }


def _server_identity(status: dict[str, Any]) -> dict[str, Any]:
    return {
        key: status.get(key)
        for key in (
            "api_version",
            "server_version",
            "backend",
            "devices",
            "workers",
            "timing",
            "target",
            "software",
        )
    }


def _implementation(definition: str, source: str, candidate_sha: str) -> dict[str, Any]:
    return {
        "name": f"nvidia-portability-{definition}-{candidate_sha[:12]}",
        "definition": definition,
        "language": "triton",
        "entrypoint": "main.py::run",
        "sources": [{"path": "main.py", "content": source}],
    }


def _bound_request(
    definition: str,
    implementation: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "api_version": "v6.2",
        "binding": {"catalog_name": CATALOG_NAME, "definition": definition},
        "implementation": implementation,
        "settings": settings,
    }


def _blocked(
    target: Target,
    candidate_sha: str,
    definition: str,
    stage: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": "kernelgen.nvidia-portability-target/v1",
        "chip": target.chip,
        "server_url": target.server_url,
        "candidate_sha256": candidate_sha,
        "definition": definition,
        "verdict": "BLOCKED",
        "stage": stage,
        "reason": reason,
        "geo_mean": None,
        "completed_at": _now(),
    }


def _target_baseline_gate(
    repo_root: pathlib.Path, target: Target, source_operator: str
) -> tuple[bool, str, bool]:
    results_path = repo_root / "kernel_todo_v2" / target.chip / "results.md"
    if not results_path.is_file():
        return True, "target results.md is missing; validate artifact only", False
    _, rows = _load_results(results_path)
    if source_operator not in rows:
        return True, "operator row is missing; validate artifact only", False
    columns = rows[source_operator][1]
    if columns[1] != "通过" or columns[2] != "通过":
        return (
            False,
            "target baseline gate failed: "
            f"Reference={columns[1]}, Gems可计时={columns[2]}",
            True,
        )
    return True, "", True


def _validate_target(
    *,
    target: Target,
    repo_root: pathlib.Path,
    run_root: pathlib.Path,
    source_operator: str,
    definition: str,
    source: str,
    candidate_sha: str,
    threshold: float,
    settings: dict[str, Any],
    transport_timeout: float,
    force: bool,
    client_factory: Callable[[str], Any],
) -> dict[str, Any]:
    target_root = run_root / "targets" / target.chip
    summary_path = target_root / "summary.json"
    if summary_path.is_file() and not force:
        summary = _read_json(summary_path)
        identity = (
            summary.get("candidate_sha256") == candidate_sha
            and summary.get("definition") == definition
            and summary.get("server_url") == target.server_url
        )
        if identity and summary.get("verdict") in TERMINAL_VERDICTS:
            return summary
        raise ValidationError(
            f"existing summary identity differs for {target.chip}; use a new run root"
        )
    target_root.mkdir(parents=True, exist_ok=True)
    allowed, reason, has_results_row = _target_baseline_gate(
        repo_root, target, source_operator
    )
    if not allowed:
        summary = _blocked(target, candidate_sha, definition, "baseline", reason)
        summary["has_results_row"] = has_results_row
        _atomic_json(summary_path, summary)
        return summary
    client = client_factory(target.server_url)
    summary: dict[str, Any]
    server_identity: dict[str, Any] | None = None
    try:
        status_before = client.request("GET", "/status", timeout=60)
        server_identity = _server_identity(status_before)
        _atomic_json(target_root / "status_before.json", status_before)
        if status_before.get("api_version") != "v6.2":
            raise ValidationError(
                "server api_version is "
                f"{status_before.get('api_version')!r}, expected 'v6.2'"
            )
        if "flaggems" not in {
            str(item) for item in status_before.get("evaluation_adapters", [])
        }:
            raise ValidationError("server does not advertise the FlagGems adapter")
        expected_backend = EXPECTED_BACKENDS.get(target.chip)
        if expected_backend and status_before.get("backend") != expected_backend:
            raise ValidationError(
                "target backend mismatch: "
                f"{status_before.get('backend')!r} != {expected_backend!r}"
            )
        inspect = client.request(
            "POST",
            "/inspect",
            {
                "api_version": "v6.2",
                "binding": {
                    "catalog_name": CATALOG_NAME,
                    "definition": definition,
                },
            },
            timeout=transport_timeout,
        )
        _atomic_json(target_root / "inspect.json", inspect)
        cases = (inspect.get("case_list") or {}).get("cases")
        contract = inspect.get("candidate_contract")
        if not isinstance(cases, list) or not cases:
            raise ValidationError("/inspect returned no timing cases")
        if not isinstance(contract, dict) or not contract.get("signature"):
            raise ValidationError("/inspect returned no candidate contract")
        fingerprint = inspect.get("benchmark_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValidationError("/inspect returned no benchmark fingerprint")
        case_fingerprint = (inspect.get("case_list") or {}).get(
            "benchmark_fingerprint"
        )
        if case_fingerprint != fingerprint:
            raise ValidationError(
                "/inspect contains inconsistent benchmark fingerprints"
            )
        case_ids = [item.get("case_id") for item in cases if isinstance(item, dict)]
        if len(case_ids) != len(cases) or len(set(case_ids)) != len(case_ids):
            raise ValidationError(
                "/inspect returned invalid or duplicate case_id values"
            )
        implementation = _implementation(definition, source, candidate_sha)
        request = _bound_request(definition, implementation, settings)
        preflight = client.request(
            "POST", "/preflight", request, timeout=transport_timeout
        )
        _atomic_json(target_root / "preflight.json", preflight)
        if preflight.get("api_version") != "v6.2":
            raise ValidationError("/preflight returned an unexpected api_version")
        if preflight.get("benchmark_fingerprint") != fingerprint:
            raise ValidationError("benchmark fingerprint changed after /inspect")
        preflight_status = preflight.get("status")
        if preflight_status != "PASSED":
            if preflight_status in {"TIMEOUT", "SUSPECTED_DEVICE_ERROR"}:
                verdict = "BLOCKED"
            else:
                verdict = "NEEDS_SPECIALIZATION"
            summary = {
                "schema_version": "kernelgen.nvidia-portability-target/v1",
                "chip": target.chip,
                "server_url": target.server_url,
                "candidate_sha256": candidate_sha,
                "definition": definition,
                "benchmark_fingerprint": fingerprint,
                "verdict": verdict,
                "stage": "preflight",
                "reason": str(preflight.get("log") or preflight_status),
                "geo_mean": None,
                "is_hack": bool(preflight.get("is_hack")),
                "hack_reason": str(preflight.get("hack_reason") or ""),
                "has_results_row": has_results_row,
                "completed_at": _now(),
            }
        else:
            if preflight.get("num_cases") != len(case_ids):
                raise ValidationError(
                    "preflight case count changed: "
                    f"{preflight.get('num_cases')} != {len(case_ids)}"
                )
            evaluation = client.request(
                "POST", "/evaluate", request, timeout=transport_timeout
            )
            _atomic_json(target_root / "evaluate.json", evaluation)
            if evaluation.get("api_version") != "v6.2":
                raise ValidationError("/evaluate returned an unexpected api_version")
            if evaluation.get("server_backend") != status_before.get("backend"):
                raise ValidationError("/evaluate server_backend differs from /status")
            status = evaluation.get("status")
            if status in {"TIMEOUT", "SUSPECTED_DEVICE_ERROR"}:
                verdict = "BLOCKED"
            elif status != "PASSED":
                verdict = "NEEDS_SPECIALIZATION"
            else:
                rows = evaluation.get("per_workload")
                if not isinstance(rows, list) or not rows:
                    raise ValidationError("PASSED evaluation returned no workloads")
                if evaluation.get("num_workloads") != len(rows):
                    raise ValidationError(
                        "PASSED evaluation num_workloads differs from per_workload"
                    )
                if evaluation.get("num_passed") != len(rows):
                    raise ValidationError(
                        "PASSED evaluation num_passed differs from per_workload"
                    )
                if any(item.get("status") != "PASSED" for item in rows):
                    raise ValidationError(
                        "PASSED evaluation contains a failed workload"
                    )
                timing_rows = [
                    item for item in rows if item.get("phase") == "timing"
                ]
                if not any(item.get("phase") == "correctness" for item in rows):
                    raise ValidationError(
                        "PASSED evaluation returned no correctness workloads"
                    )
                if {item.get("uuid") for item in timing_rows} != set(case_ids):
                    raise ValidationError(
                        "evaluate timing case_ids differ from /inspect"
                    )
                for item in timing_rows:
                    for key in ("speedup", "latency_ms", "reference_latency_ms"):
                        value = item.get(key)
                        if (
                            not isinstance(value, (int, float))
                            or not math.isfinite(value)
                            or value <= 0
                        ):
                            raise ValidationError(
                                f"invalid timing {key} for {item.get('uuid')}"
                            )
                geo_mean = evaluation.get("geo_mean")
                if not isinstance(geo_mean, (int, float)) or not math.isfinite(
                    geo_mean
                ):
                    raise ValidationError("PASSED evaluation has no finite geo_mean")
                if evaluation.get("is_hack"):
                    verdict = "NEEDS_SPECIALIZATION"
                else:
                    verdict = (
                        "REUSABLE"
                        if geo_mean >= threshold
                        else "NEEDS_SPECIALIZATION"
                    )
            reason = str(evaluation.get("log") or status)
            if status == "PASSED" and evaluation.get("is_hack"):
                reason = str(
                    evaluation.get("hack_reason")
                    or "Server marked candidate as hack"
                )
            elif status == "PASSED" and evaluation.get("geo_mean", 0) < threshold:
                reason = (
                    f"geo_mean={float(evaluation['geo_mean']):.6f} "
                    f"< threshold={threshold:.6f}"
                )
            elif status == "PASSED":
                reason = "all correctness and timing workloads passed"
            summary = {
                "schema_version": "kernelgen.nvidia-portability-target/v1",
                "chip": target.chip,
                "server_url": target.server_url,
                "candidate_sha256": candidate_sha,
                "definition": definition,
                "benchmark_fingerprint": fingerprint,
                "verdict": verdict,
                "stage": "evaluate",
                "reason": reason,
                "geo_mean": evaluation.get("geo_mean"),
                "min_speedup": evaluation.get("min_speedup"),
                "num_workloads": evaluation.get("num_workloads"),
                "num_passed": evaluation.get("num_passed"),
                "is_hack": bool(evaluation.get("is_hack")),
                "hack_reason": str(evaluation.get("hack_reason") or ""),
                "has_results_row": has_results_row,
                "completed_at": _now(),
            }
        inspect_after = client.request(
            "POST",
            "/inspect",
            {
                "api_version": "v6.2",
                "binding": {
                    "catalog_name": CATALOG_NAME,
                    "definition": definition,
                },
            },
            timeout=transport_timeout,
        )
        _atomic_json(target_root / "inspect_after.json", inspect_after)
        if inspect_after.get("benchmark_fingerprint") != fingerprint:
            raise ValidationError("benchmark fingerprint changed during validation")
    except HttpError as exc:
        summary = _blocked(
            target, candidate_sha, definition, "transport/server", str(exc)
        )
        summary["has_results_row"] = has_results_row
    except ValidationError as exc:
        summary = _blocked(target, candidate_sha, definition, "gate/protocol", str(exc))
        summary["has_results_row"] = has_results_row
    if server_identity is not None:
        summary["server"] = server_identity
    try:
        status_after = client.request("GET", "/status", timeout=60)
        _atomic_json(target_root / "status_after.json", status_after)
        summary["scheduler_after"] = status_after.get("scheduler")
    except (HttpError, ValidationError) as exc:
        summary["post_status_error"] = str(exc)
    _atomic_json(summary_path, summary)
    return summary


def _sanitize(value: object, limit: int = 1200) -> str:
    text = " ".join(str(value or "").split()).replace("|", "\\|")
    return text[:limit]


def _relative(from_dir: pathlib.Path, target: pathlib.Path) -> str:
    relative = os.path.relpath(target.resolve(), from_dir.resolve())
    return pathlib.Path(relative).as_posix()


def _replace_note(existing: str, note: str) -> str:
    marker_at = existing.find(MARK_NOTE)
    base = (existing[:marker_at] if marker_at >= 0 else existing).rstrip()
    if base and base != "—":
        return f"{base} {MARK_NOTE}{note}】"
    return f"{MARK_NOTE}{note}】"


def _copy_candidate(
    candidate: pathlib.Path,
    destination: pathlib.Path,
    *,
    overwrite: bool,
) -> tuple[bool, str]:
    if destination.is_file():
        if _sha256(destination) == _sha256(candidate):
            return True, "already identical"
        if not overwrite:
            return False, "target codes file exists with different content"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(candidate, temporary)
    if _sha256(temporary) != _sha256(candidate):
        temporary.unlink(missing_ok=True)
        raise ValidationError(f"candidate copy verification failed: {destination}")
    temporary.replace(destination)
    return True, "copied"


def _normalize_portability_cell(cell: str) -> str:
    normalized = cell.replace("[summary](", "[证据](")
    normalized = re.sub(r"<br>SHA `[0-9a-fA-F]+`", "", normalized)
    return (
        normalized.replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("&lt;br&gt;", "<br>")
    )


def _portability_rows(lines: list[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    chip_by_label = {label: chip for chip, label in PORTABILITY_CHIPS}
    matrix_header: list[str] | None = None
    for line in lines:
        if not line.startswith("| NVIDIA 算子 |"):
            continue
        columns = [item.strip() for item in line.strip().strip("|").split("|")]
        if columns[1:] == [label for _, label in PORTABILITY_CHIPS]:
            matrix_header = columns
            break
    if matrix_header is not None:
        for line in lines:
            if not line.startswith("| `"):
                continue
            columns = [item.strip() for item in line.strip().strip("|").split("|")]
            if len(columns) != len(matrix_header):
                continue
            names = ROW_OPERATOR.findall(columns[0])
            if not names:
                continue
            source_operator = names[0]
            rows[source_operator] = {
                "definition": names[1] if len(names) > 1 else source_operator,
                "cells": {
                    chip_by_label[label]: _normalize_portability_cell(columns[index])
                    for index, label in enumerate(matrix_header[1:], start=1)
                    if columns[index] != "—"
                },
            }
        return rows

    # Migrate the former one-row-per-target layout on the next write.
    for line in lines:
        if not line.startswith("| `"):
            continue
        columns = [item.strip() for item in line.strip().strip("|").split("|")]
        names = ROW_OPERATOR.findall(columns[0]) if len(columns) >= 10 else []
        if not names or columns[2] not in ALL_TARGET_CHIPS:
            continue
        source_operator = names[0]
        candidate_names = ROW_OPERATOR.findall(columns[1])
        parts = [f"{columns[3]} · {columns[4]}" if columns[4] != "—" else columns[3]]
        if columns[5].startswith("是"):
            parts.append(f"Hack：{columns[5][1:].lstrip('：')}")
        if columns[3] != VERDICT_LABELS["REUSABLE"] and columns[6] != "—":
            parts.append(columns[6])
        parts.append(columns[8].replace("[summary](", "[证据]("))
        record = rows.setdefault(
            source_operator,
            {
                "definition": (
                    candidate_names[0] if candidate_names else source_operator
                ),
                "cells": {},
            },
        )
        record["cells"][columns[2]] = "<br>".join(parts)
    return rows


def _portability_detail(summary: dict[str, Any], evidence: str) -> dict[str, Any]:
    geo_mean = summary.get("geo_mean")
    if not isinstance(geo_mean, (int, float)) or not math.isfinite(geo_mean):
        geo_mean = None
    return {
        "verdict": str(summary["verdict"]),
        "geo_mean": geo_mean,
        "stage": str(summary.get("stage") or ""),
        "reason": str(summary.get("reason") or ""),
        "is_hack": bool(summary.get("is_hack")),
        "hack_reason": str(summary.get("hack_reason") or ""),
        "evidence": evidence,
        "completed_at": str(summary.get("completed_at") or ""),
    }


def _portability_ledger_from_markdown(
    nvidia_root: pathlib.Path, lines: list[str]
) -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "schema_version": PORTABILITY_SCHEMA,
        "operators": {},
    }
    verdict_by_label = {label: verdict for verdict, label in VERDICT_LABELS.items()}
    for source_operator, row in _portability_rows(lines).items():
        targets: dict[str, Any] = {}
        for chip, cell in row["cells"].items():
            evidence_match = re.search(r"\[证据\]\(([^)]+)\)", cell)
            evidence = evidence_match.group(1) if evidence_match else ""
            evidence_path = (nvidia_root / evidence).resolve() if evidence else None
            if evidence_path is not None and evidence_path.is_file():
                detail = _portability_detail(_read_json(evidence_path), evidence)
            else:
                label = next(
                    (item for item in verdict_by_label if cell.startswith(item)),
                    VERDICT_LABELS["BLOCKED"],
                )
                speed_match = re.search(r"([0-9]+(?:\.[0-9]+)?)x", cell)
                parts = cell.split("<br>")
                reasons = [
                    part
                    for part in parts[1:]
                    if not part.startswith(("Hack：", "[证据](", "SHA `"))
                ]
                detail = {
                    "verdict": verdict_by_label[label],
                    "geo_mean": float(speed_match.group(1)) if speed_match else None,
                    "stage": "",
                    "reason": " ".join(reasons),
                    "is_hack": any(part.startswith("Hack：") for part in parts),
                    "hack_reason": next(
                        (
                            part.removeprefix("Hack：")
                            for part in parts
                            if part.startswith("Hack：")
                        ),
                        "",
                    ),
                    "evidence": evidence,
                    "completed_at": "",
                }
            targets[chip] = detail
        ledger["operators"][source_operator] = {
            "definition": row["definition"],
            "targets": targets,
        }
    return ledger


def _load_portability_ledger(
    nvidia_root: pathlib.Path, json_path: pathlib.Path, markdown_path: pathlib.Path
) -> dict[str, Any]:
    if json_path.is_file():
        ledger = _read_json(json_path)
        if ledger.get("schema_version") != PORTABILITY_SCHEMA or not isinstance(
            ledger.get("operators"), dict
        ):
            raise ValidationError(f"invalid portability ledger: {json_path}")
        return ledger
    lines = (
        markdown_path.read_text(encoding="utf-8").splitlines()
        if markdown_path.is_file()
        else []
    )
    return _portability_ledger_from_markdown(nvidia_root, lines)


def _portability_markdown(ledger: dict[str, Any]) -> str:
    operators = ledger["operators"]
    status_labels = ("已验证", *VERDICT_LABELS.values())
    counts: dict[str, dict[str, int]] = {
        status: {chip: 0 for chip, _ in PORTABILITY_CHIPS}
        for status in status_labels
    }
    for record in operators.values():
        for chip, detail in record["targets"].items():
            if chip not in counts["已验证"]:
                continue
            counts["已验证"][chip] += 1
            label = VERDICT_LABELS.get(str(detail.get("verdict")))
            if label:
                counts[label][chip] += 1
    chip_labels = [label for _, label in PORTABILITY_CHIPS]
    lines = [
        "# NVIDIA Triton 跨芯片验证结果",
        "",
        "本页由跨芯片评测脚本自动维护，以 NVIDIA 算子为行、目标芯片为列；"
        "单元格只展示结论和加速比，详细原因、阶段与证据见 "
        "`portability_results.json`。NVIDIA 自身的优化状态和加速比仍以 "
        "`results.md` 为准。",
        "",
        "## 汇总",
        "",
        "| 状态 | " + " | ".join(chip_labels) + " |",
        "| --- | " + " | ".join("---:" for _ in PORTABILITY_CHIPS) + " |",
    ]
    lines.extend(
        "| "
        + status
        + " | "
        + " | ".join(str(counts[status][chip]) for chip, _ in PORTABILITY_CHIPS)
        + " |"
        for status in status_labels
    )
    lines.extend(
        (
            "",
            "## 算子矩阵",
            "",
            "| NVIDIA 算子 | " + " | ".join(chip_labels) + " |",
            "| --- | " + " | ".join("---" for _ in PORTABILITY_CHIPS) + " |",
        )
    )
    for source_operator in sorted(operators):
        record = operators[source_operator]
        definition = str(record["definition"])
        operator_cell = (
            f"`{source_operator}`"
            if definition == source_operator
            else f"`{source_operator}`（`{definition}`）"
        )
        target_cells = []
        for chip, _ in PORTABILITY_CHIPS:
            detail = record["targets"].get(chip)
            if detail is None:
                target_cells.append("—")
                continue
            label = VERDICT_LABELS[str(detail["verdict"])]
            geo_mean = detail.get("geo_mean")
            if isinstance(geo_mean, (int, float)) and math.isfinite(geo_mean):
                target_cells.append(f"{label} · {float(geo_mean):.3f}x")
            else:
                target_cells.append(label)
        lines.append("| " + operator_cell + " | " + " | ".join(target_cells) + " |")
    return "\n".join(lines) + "\n"


def _sync_nvidia_portability_result(
    *,
    repo_root: pathlib.Path,
    source_operator: str,
    definition: str,
    run_root: pathlib.Path,
    summary: dict[str, Any],
) -> dict[str, str]:
    nvidia_root = repo_root / "kernel_todo_v2/nvidia"
    path = nvidia_root / "portability_results.md"
    json_path = nvidia_root / "portability_results.json"
    nvidia_root.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(nvidia_root, os.O_RDONLY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        ledger = _load_portability_ledger(nvidia_root, json_path, path)
        chip = str(summary["chip"])
        evidence_path = run_root / "targets" / chip / "summary.json"
        detail = _portability_detail(
            summary,
            _relative(nvidia_root, evidence_path),
        )
        record = ledger["operators"].get(source_operator)
        old_detail = None if record is None else record["targets"].get(chip)
        old_definition = None if record is None else record["definition"]
        if old_detail is None:
            action = "created"
        elif old_detail == detail and old_definition == definition:
            action = "unchanged"
        else:
            action = "updated"
        if action != "unchanged":
            record = ledger["operators"].setdefault(
                source_operator, {"definition": definition, "targets": {}}
            )
            record["definition"] = definition
            record["targets"][chip] = detail
        _atomic_json(json_path, ledger)
        _atomic_text(path, _portability_markdown(ledger))
        return {
            "chip": chip,
            "action": action,
            "path": str(path),
            "details_path": str(json_path),
        }
    finally:
        fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)


def _update_one_results(
    *,
    repo_root: pathlib.Path,
    source_operator: str,
    definition: str,
    candidate: pathlib.Path,
    candidate_sha: str,
    run_root: pathlib.Path,
    summary: dict[str, Any],
    overwrite_existing_code: bool,
) -> dict[str, Any]:
    chip = str(summary["chip"])
    results_path = repo_root / "kernel_todo_v2" / chip / "results.md"
    if not results_path.is_file():
        return {
            "chip": chip,
            "action": "artifact_only",
            "reason": "results.md missing",
        }
    lines, rows = _load_results(results_path)
    if source_operator not in rows:
        return {
            "chip": chip,
            "action": "artifact_only",
            "reason": "operator row missing",
        }
    index, columns = rows[source_operator]
    evidence_path = run_root / "targets" / chip / "summary.json"
    evidence = f"[跨芯片验证]({_relative(results_path.parent, evidence_path)})"
    old_status = columns[3]
    verdict = summary["verdict"]
    if verdict == "REUSABLE" and old_status != "成功":
        code_path = (
            repo_root / "kernel_todo_v2" / chip / "codes" / f"{definition}.py"
        )
        copied, copy_reason = _copy_candidate(
            candidate, code_path, overwrite=overwrite_existing_code
        )
        if not copied:
            return {"chip": chip, "action": "update_blocked", "reason": copy_reason}
        geo_mean = float(summary["geo_mean"])
        columns[3] = "成功"
        columns[4] = f"{geo_mean:.3f}x"
        columns[5] = "未发现（NVIDIA 通用候选跨芯片复验）"
        columns[6] = f"—（{evidence}）"
        columns[7] = "NVIDIA 通用候选已在本芯片全量验证并达到 0.8x，可直接复用。"
        columns[8] = f"[codes/{definition}.py](codes/{definition}.py)"
        action = "reused"
    elif verdict == "REUSABLE":
        columns[6] = _replace_note(
            columns[6], f"NVIDIA 待测代码验证达标；{evidence}"
        )
        action = "preserved_existing_success"
    elif verdict == "NEEDS_SPECIALIZATION" and old_status != "成功":
        columns[3] = "需要特化"
        columns[4] = (
            f"{float(summary['geo_mean']):.3f}x"
            if isinstance(summary.get("geo_mean"), (int, float))
            else "—"
        )
        columns[5] = (
            f"是：{_sanitize(summary.get('hack_reason'))}"
            if summary.get("is_hack")
            else "未发现（NVIDIA 通用候选跨芯片复验）"
        )
        columns[6] = (
            f"NVIDIA 通用候选不能在本芯片直接复用：{_sanitize(summary.get('reason'))}"
            f"（{evidence}）"
        )
        columns[7] = "进入本芯片专用特化 Batch；不得修改原 NVIDIA 候选或覆盖验证证据。"
        action = "needs_specialization"
    elif verdict == "NEEDS_SPECIALIZATION":
        columns[6] = _replace_note(
            columns[6], f"通用候选未达标但已有本地成功结果保持不变；{evidence}"
        )
        action = "preserved_existing_success"
    else:
        columns[6] = _replace_note(
            columns[6], f"跨芯片验证阻塞：{_sanitize(summary.get('reason'))}；{evidence}"
        )
        if old_status != "成功":
            columns[7] = "修复环境、baseline 或协议阻塞后，使用本次 run root 中的待测代码续跑。"
        action = "blocked_recorded"
    lines[index] = "| " + " | ".join(columns) + " |"
    rows[source_operator] = (index, columns)
    pipeline_path = results_path.parent / "pipeline_results.json"
    pipeline = (
        KernelTodoV2PipelineResults.model_validate_json(
            pipeline_path.read_text(encoding="utf-8")
        )
        if pipeline_path.is_file()
        else None
    )
    _refresh_summary(lines, rows, pipeline)
    _atomic_text(results_path, "\n".join(lines) + "\n")
    return {"chip": chip, "action": action, "status": columns[3]}


def run(
    args: argparse.Namespace,
    *,
    client_factory: Callable[[str], Any] = JsonClient,
) -> tuple[int, dict[str, Any]]:
    repo_root = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    campaign_root = (repo_root / "runs/kernel_todo_v2").resolve()
    if run_root == campaign_root or not run_root.is_relative_to(campaign_root):
        raise ValidationError(
            f"--run-root must be a new campaign below {campaign_root}"
        )
    targets, excludes = _targets(args)
    entry = _inventory_entry(repo_root, args.operator)
    definition = str(entry["operator"])
    source_candidate, source_result = _source_candidate(
        args, repo_root, args.operator, definition
    )
    candidate_sha = _sha256(source_candidate)
    candidate = run_root / "candidate.py"
    copied, copy_reason = _copy_candidate(source_candidate, candidate, overwrite=False)
    if not copied:
        raise ValidationError(f"run root candidate conflict: {copy_reason}")
    source = candidate.read_text(encoding="utf-8")
    manifest = {
        "schema_version": "kernelgen.nvidia-portability/v1",
        "created_at": _now(),
        "source_operator": args.operator,
        "definition": definition,
        "source_candidate_path": str(source_candidate),
        "candidate_path": str(candidate),
        "candidate_sha256": candidate_sha,
        "source_result": source_result,
        "threshold": args.threshold,
        "targets": [dataclasses.asdict(target) for target in targets],
        "excluded": excludes,
    }
    manifest_path = run_root / "manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        identity_fields = (
            "source_operator",
            "definition",
            "candidate_sha256",
            "threshold",
            "targets",
            "excluded",
        )
        if any(existing.get(key) != manifest.get(key) for key in identity_fields):
            raise ValidationError(
                "run root manifest belongs to a different candidate/run"
            )
        manifest["created_at"] = existing.get("created_at", manifest["created_at"])
    _atomic_json(manifest_path, manifest)
    settings = {
        "warmup_ms": args.warmup_ms,
        "benchmark_ms": args.benchmark_ms,
        "num_trials": args.num_trials,
        "timeout_seconds": args.timeout_seconds,
    }
    transport_timeout = float(args.timeout_seconds + 180)
    summaries: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.max_workers, len(targets))
    ) as executor:
        futures = {
            executor.submit(
                _validate_target,
                target=target,
                repo_root=repo_root,
                run_root=run_root,
                source_operator=args.operator,
                definition=definition,
                source=source,
                candidate_sha=candidate_sha,
                threshold=args.threshold,
                settings=settings,
                transport_timeout=transport_timeout,
                force=args.force,
                client_factory=client_factory,
            ): target
            for target in targets
        }
        for future in concurrent.futures.as_completed(futures):
            target = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                summary = _blocked(
                    target,
                    candidate_sha,
                    definition,
                    "orchestrator",
                    f"{type(exc).__name__}: {exc}",
                )
                summary_path = run_root / "targets" / target.chip / "summary.json"
                _atomic_json(summary_path, summary)
            summaries[target.chip] = summary
            print(
                json.dumps(
                    {
                        "chip": target.chip,
                        "verdict": summary["verdict"],
                        "geo_mean": summary.get("geo_mean"),
                        "reason": summary.get("reason"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    updates: list[dict[str, Any]] = []
    nvidia_updates: list[dict[str, str]] = []
    if not args.no_update_results:
        for target in targets:
            updates.append(
                _update_one_results(
                    repo_root=repo_root,
                    source_operator=args.operator,
                    definition=definition,
                    candidate=candidate,
                    candidate_sha=candidate_sha,
                    run_root=run_root,
                    summary=summaries[target.chip],
                    overwrite_existing_code=args.overwrite_existing_code,
                )
            )
            nvidia_updates.append(
                _sync_nvidia_portability_result(
                    repo_root=repo_root,
                    source_operator=args.operator,
                    definition=definition,
                    run_root=run_root,
                    summary=summaries[target.chip],
                )
            )
    verdict_counts = {
        verdict: sum(item["verdict"] == verdict for item in summaries.values())
        for verdict in sorted(TERMINAL_VERDICTS)
    }
    report = {
        "schema_version": "kernelgen.nvidia-portability-report/v1",
        "completed_at": _now(),
        "source_operator": args.operator,
        "definition": definition,
        "candidate_sha256": candidate_sha,
        "threshold": args.threshold,
        "verdict_counts": verdict_counts,
        "excluded": excludes,
        "targets": summaries,
        "updates": updates,
        "nvidia_updates": nvidia_updates,
    }
    _atomic_json(run_root / "report.json", report)
    update_blocked = any(item.get("action") == "update_blocked" for item in updates)
    if verdict_counts["BLOCKED"] or update_blocked:
        return 3, report
    if verdict_counts["NEEDS_SPECIALIZATION"]:
        return 2, report
    return 0, report


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        exit_code, report = run(args)
    except (ValidationError, OSError, SyntaxError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    print(json.dumps(report["verdict_counts"], ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate Kernel Todo V2 FlagGems bindings through the KGS /inspect API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.request
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ROW_OPERATOR = re.compile(r"`([^`]+)`")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--results-md", type=pathlib.Path)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument(
        "--group-size",
        type=int,
        help="optional diagnostic split; default runs all remaining operators together",
    )
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--no-update-results", action="store_true")
    args = parser.parse_args()
    if (
        args.max_parallel < 1
        or (args.group_size is not None and args.group_size < 1)
        or args.timeout_seconds < 1
    ):
        parser.error("parallelism, an explicit group size and timeout must be positive")
    return args


def _request_json(
    server_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        server_url.rstrip("/") + path,
        data=data,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {detail}") from exc


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _results_rows(path: pathlib.Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        columns = [item.strip() for item in line.strip().strip("|").split("|")]
        match = _ROW_OPERATOR.search(columns[0])
        if match and len(columns) >= 9:
            rows[match.group(1)] = columns
    return rows


def _load_targets(
    chip: str,
    results_path: pathlib.Path,
    selected: set[str],
) -> list[dict[str, Any]]:
    source_ops = [
        line.strip()
        for line in (REPO_ROOT / "kernel_todo_v2" / chip / "failed_ops.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    inventory = json.loads(
        (REPO_ROOT / "kernel_todo_v2/pytest_conversion_inventory.json").read_text(
            encoding="utf-8"
        )
    )["operators"]
    by_source = {
        entry["source_operator"]: entry
        for entry in inventory
        if chip in entry["chips"] or chip == "nvidia"
    }
    rows = _results_rows(results_path)
    missing_inventory = [name for name in source_ops if name not in by_source]
    missing_rows = [name for name in source_ops if name not in rows]
    if missing_inventory or missing_rows:
        raise RuntimeError(
            f"missing inventory={missing_inventory}, "
            f"missing results rows={missing_rows}"
        )
    targets = []
    for name in source_ops:
        row = rows[name]
        if row[3] != "未跑" or row[1] != "通过" or row[2] != "通过":
            continue
        if selected and name not in selected:
            continue
        targets.append(by_source[name])
    unknown = selected - set(source_ops)
    if unknown:
        raise RuntimeError(
            "selected operators are not in failed_ops.txt: " f"{sorted(unknown)}"
        )
    return targets


def _scheduler_snapshot(server_url: str, label: str) -> dict[str, Any]:
    status = _request_json(server_url, "/status", timeout=30)
    scheduler = status["scheduler"]
    if scheduler["broken"] != 0 or scheduler["checking"] != 0:
        raise RuntimeError(f"unhealthy scheduler at {label}: {scheduler}")
    if scheduler["active"] != 0 or scheduler["waiting"] != 0:
        raise RuntimeError(f"scheduler is not idle at {label}: {scheduler}")
    return {"label": label, "timestamp": time.time(), "scheduler": scheduler}


def _verdict(payload: dict[str, Any]) -> tuple[bool, str, int]:
    case_list = payload.get("case_list")
    if not isinstance(case_list, dict):
        return False, "响应缺少 case_list。", 0
    cases = case_list.get("cases")
    if not isinstance(cases, list) or not cases:
        return False, "`/inspect` 未枚举出 Workload。", 0
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or not all(
        isinstance(case_id, str) and case_id for case_id in case_ids
    ):
        return False, "至少一个 Workload 缺少稳定 case_id。", len(cases)
    if len(set(case_ids)) != len(case_ids):
        return False, "`/inspect` 返回了重复 case_id。", len(cases)
    if not all(case.get("phase") == "timing" for case in cases):
        return False, "`/inspect` 混入了非 timing Workload。", len(cases)
    contract = payload.get("candidate_contract")
    if (
        not isinstance(contract, dict)
        or not contract.get("entrypoint")
        or not contract.get("signature")
    ):
        return False, "响应缺少完整 candidate contract。", len(cases)
    if not payload.get("benchmark_fingerprint"):
        return False, "响应缺少 benchmark fingerprint。", len(cases)
    return True, "", len(cases)


def _inspect_one(
    entry: dict[str, Any],
    *,
    server_url: str,
    output_root: pathlib.Path,
    timeout_seconds: int,
) -> tuple[str, dict[str, Any]]:
    source_operator = entry["source_operator"]
    local_dir = output_root / "operators" / source_operator.replace("/", "_")
    local_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "api_version": "v6.2",
        "binding": {
            "catalog_name": "flaggems-adapter-definitions",
            "definition": entry["operator"],
        },
    }
    started = time.time()
    try:
        payload = _request_json(
            server_url,
            "/inspect",
            payload=request,
            timeout=timeout_seconds,
        )
        _atomic_json(local_dir / "inspect.json", payload)
        passed, reason, case_count = _verdict(payload)
        result = {
            "schema_version": "kernelgen.kernel-todo-v2-inspect/v1",
            "source_operator": source_operator,
            "definition": entry["operator"],
            "request": request,
            "passed": passed,
            "case_count": case_count,
            "reason": reason,
            "benchmark_fingerprint": payload.get("benchmark_fingerprint"),
            "started_at": started,
            "completed_at": time.time(),
        }
    except BaseException as exc:
        result = {
            "schema_version": "kernelgen.kernel-todo-v2-inspect/v1",
            "source_operator": source_operator,
            "definition": entry["operator"],
            "request": request,
            "passed": False,
            "case_count": 0,
            "reason": f"{type(exc).__name__}: {exc}",
            "started_at": started,
            "completed_at": time.time(),
        }
    _atomic_json(local_dir / "summary.json", result)
    return source_operator, result


def _evidence_link(results_path: pathlib.Path, summary_path: pathlib.Path) -> str:
    return pathlib.Path(
        os.path.relpath(summary_path, start=results_path.parent)
    ).as_posix()


def _update_results(
    results_path: pathlib.Path,
    source_operator: str,
    result: dict[str, Any],
    summary_path: pathlib.Path,
) -> None:
    lines = results_path.read_text(encoding="utf-8").splitlines()
    evidence = _evidence_link(results_path, summary_path)
    updated = False
    for index, line in enumerate(lines):
        if not line.startswith("| `"):
            continue
        parts = line.split("|")
        if len(parts) < 11:
            continue
        match = _ROW_OPERATOR.search(parts[1])
        if not match or match.group(1) != source_operator:
            continue
        current_reason = parts[7].strip()
        baseline = current_reason if "baseline证据" in current_reason else "—"
        if result["passed"]:
            parts[7] = f" {baseline.rstrip('）')}；[inspect证据]({evidence})） "
            parts[8] = " V2 baseline 与 `/inspect` 已通过；待进入 sample/BatchSimpleOpt。 "
        else:
            parts[7] = (
                f" {result['reason']}（[inspect证据]({evidence})；"
                f"{baseline.lstrip('—（').rstrip('）')}） "
            )
            parts[8] = " 保持未跑；修复 Definition/pytest 协议后重新执行 `/inspect`。 "
        lines[index] = "|".join(parts)
        updated = True
        break
    if not updated:
        raise RuntimeError(f"results row not found: {source_operator}")
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = (
        args.results_md.resolve()
        if args.results_md
        else REPO_ROOT / "kernel_todo_v2" / args.chip / "results.md"
    )
    progress_path = output_root / "inspect_progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {
            "schema_version": "kernelgen.kernel-todo-v2-inspect-progress/v1",
            "results": {},
        }
    )
    selected = set(args.operator)
    targets = _load_targets(args.chip, results_path, selected)
    targets = [
        entry
        for entry in targets
        if entry["source_operator"] not in progress.get("results", {})
    ]
    if args.limit is not None:
        targets = targets[: args.limit]
    print(
        f"targets={len(targets)} completed={len(progress.get('results', {}))}",
        flush=True,
    )
    snapshots = progress.setdefault("scheduler_snapshots", [])
    group_size = args.group_size or max(1, len(targets))
    for offset in range(0, len(targets), group_size):
        group = targets[offset : offset + group_size]
        group_number = offset // group_size + 1
        snapshots.append(
            _scheduler_snapshot(args.server_url, f"group-{group_number}-before")
        )
        _atomic_json(progress_path, progress)
        print(f"group={group_number} count={len(group)} start", flush=True)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.max_parallel, len(group))
        ) as executor:
            futures = {
                executor.submit(
                    _inspect_one,
                    entry,
                    server_url=args.server_url,
                    output_root=output_root,
                    timeout_seconds=args.timeout_seconds,
                ): entry
                for entry in group
            }
            for future in concurrent.futures.as_completed(futures):
                source_operator, result = future.result()
                progress.setdefault("results", {})[source_operator] = result
                if not args.no_update_results:
                    summary_path = (
                        output_root
                        / "operators"
                        / source_operator.replace("/", "_")
                        / "summary.json"
                    )
                    _update_results(results_path, source_operator, result, summary_path)
                _atomic_json(progress_path, progress)
                print(
                    json.dumps(
                        {
                            "operator": source_operator,
                            "passed": result["passed"],
                            "case_count": result["case_count"],
                            "reason": result["reason"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        snapshots.append(
            _scheduler_snapshot(args.server_url, f"group-{group_number}-after")
        )
        _atomic_json(progress_path, progress)
    passed = sum(item["passed"] for item in progress.get("results", {}).values())
    print(
        f"done total={len(progress.get('results', {}))} passed={passed} "
        f"failed={len(progress.get('results', {})) - passed}",
        flush=True,
    )


if __name__ == "__main__":
    main()

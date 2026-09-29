#!/usr/bin/env python3
"""Run Kernel Todo V2 FlagGems core baselines through KGS Debug Jobs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import re
import time
import urllib.error
import urllib.request
from typing import Any

from kernelgen.tools.kernel_todo_v2_pipeline import (
    KernelTodoV2PipelineResults,
    refresh_markdown_summary,
)


REPO_ROOT = pathlib.Path(
    os.environ.get("KERNELGEN_REPO_ROOT", pathlib.Path(__file__).resolve().parents[2])
).resolve()
_ROW_OPERATOR = re.compile(r"`([^`]+)`")
BASELINE_PHASES = ("combined", "reference", "gems")
_PHASE_NOTE_PATTERNS = {
    "reference": re.compile(r"【Reference 复检：[^】]*】"),
    "gems": re.compile(r"【Gems 可计时复检：[^】]*】"),
}
_PHASE_EVIDENCE_PATTERNS = {
    "reference": re.compile(r"\[Reference 复检证据\]\(([^)]+)\)"),
    "gems": re.compile(r"\[Gems 可计时复检证据\]\(([^)]+)\)"),
}

REMOTE_RUNNER = r'''
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
import traceback


def main():
    config = json.loads(os.environ["FLAGGEMS_BASELINE_CONFIG"])
    root = pathlib.Path(os.environ["FLAGGEMS_ROOT"])
    artifacts = pathlib.Path(os.environ["KGS_DEBUG_ARTIFACTS"])
    os.chdir(root)
    output = artifacts / "benchmark.json"
    env = os.environ.copy()
    inherited_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(root / "src"), str(root), inherited_pythonpath)
        if part
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *config["benchmark_files"],
        "-m",
        config["pytest_mark"],
        "--level",
        "core",
        *(
            ["--mode", "operator", "--metrics", "latency_base"]
            if config.get("baseline_phase") == "reference"
            else ["--mode", "kernel", "--metrics", "latency"]
            if config.get("baseline_phase") == "gems"
            else []
        ),
        "--record",
        "json",
        "--output",
        str(output),
    ]
    started = time.time()
    summary = {
        "schema_version": "kernelgen.kernel-todo-v2-baseline/v1",
        "operator": config,
        "assigned_device": os.environ.get("KGS_ASSIGNED_DEVICE"),
        "device": os.environ.get("KGS_DEVICE"),
        "flaggems_root": str(root),
        "python": sys.executable,
        "started_at": started,
        "command": command,
    }
    try:
        process = subprocess.run(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(os.environ["FLAGGEMS_BASELINE_TIMEOUT"]),
            check=False,
        )
        summary.update(
            {
                "status": "completed",
                "exit_code": process.returncode,
                "stdout_tail": process.stdout[-12000:],
                "stderr_tail": process.stderr[-12000:],
            }
        )
        (artifacts / "benchmark.stdout.log").write_text(
            process.stdout, encoding="utf-8"
        )
        (artifacts / "benchmark.stderr.log").write_text(
            process.stderr, encoding="utf-8"
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        summary.update(
            {
                "status": "timeout",
                "exit_code": None,
                "stdout_tail": stdout[-12000:],
                "stderr_tail": stderr[-12000:],
            }
        )
        (artifacts / "benchmark.stdout.log").write_text(stdout, encoding="utf-8")
        (artifacts / "benchmark.stderr.log").write_text(stderr, encoding="utf-8")
    except BaseException:
        summary["runner_error"] = traceback.format_exc()
    summary["completed_at"] = time.time()
    summary["duration_seconds"] = summary["completed_at"] - started
    (artifacts / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "operator": config["source_operator"],
                "status": summary.get("status"),
                "exit_code": summary.get("exit_code"),
                "duration_seconds": round(summary["duration_seconds"], 3),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
'''


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--flaggems-root", required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--results-md", type=pathlib.Path)
    parser.add_argument("--max-parallel", type=int, required=True)
    parser.add_argument(
        "--group-size",
        type=int,
        help="optional diagnostic split; default runs all remaining operators together",
    )
    parser.add_argument("--phase", choices=BASELINE_PHASES, default="combined")
    parser.add_argument("--phase-timeout-seconds", type=int, default=1500)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument(
        "--debug-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra environment passed to each trusted Server Debug Job",
    )
    parser.add_argument("--no-update-results", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.phase_timeout_seconds <= 1740:
        parser.error("--phase-timeout-seconds must be between 1 and 1740")
    if args.max_parallel < 1 or (args.group_size is not None and args.group_size < 1):
        parser.error("parallelism and an explicit group size must be positive")
    args.debug_env = dict(item.split("=", 1) for item in args.debug_env)
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
        server_url + path,
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


def _download(server_url: str, path: str, *, timeout: float = 120) -> bytes:
    request = urllib.request.Request(server_url + path, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return response.read()


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


def _refresh_results_summary(
    lines: list[str],
    pipeline: KernelTodoV2PipelineResults | None = None,
) -> None:
    refresh_markdown_summary(lines, pipeline)


def _load_targets(
    chip: str,
    results_path: pathlib.Path,
    selected: set[str],
    phase: str = "combined",
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
        if row[3] != "未跑":
            continue
        if phase == "combined" and row[1] == "通过" and row[2] == "通过":
            continue
        if phase == "reference" and row[1] == "通过":
            continue
        if phase == "gems" and row[2] == "通过":
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


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _scheduler_snapshot(server_url: str, label: str) -> dict[str, Any]:
    status = _request_json(server_url, "/status", timeout=30)
    scheduler = status["scheduler"]
    if scheduler["broken"] != 0 or scheduler["checking"] != 0:
        raise RuntimeError(f"unhealthy scheduler at {label}: {scheduler}")
    if scheduler["active"] != 0 or scheduler["waiting"] != 0:
        raise RuntimeError(f"scheduler is not idle at {label}: {scheduler}")
    return {"label": label, "timestamp": time.time(), "scheduler": scheduler}


def _case_records(payload: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return records
    for operator in payload.values():
        if not isinstance(operator, dict):
            continue
        for detail in operator.get("details", []):
            if not isinstance(detail, dict):
                continue
            result = detail.get("result", [])
            if isinstance(result, list):
                records.extend(item for item in result if isinstance(item, dict))
    return records


def _positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _verdict(result: dict[str, Any], phase: str = "combined") -> dict[str, Any]:
    summary = result.get("summary", {})
    job = result.get("job", {})
    benchmark_path = result.get("local_dir")
    benchmark_file = (
        pathlib.Path(benchmark_path, "benchmark.json") if benchmark_path else None
    )
    payload = None
    parse_error = ""
    if benchmark_file and benchmark_file.is_file():
        try:
            payload = json.loads(benchmark_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    records = _case_records(payload)
    command_ok = (
        job.get("status") == "SUCCEEDED"
        and summary.get("status") == "completed"
        and summary.get("exit_code") == 0
    )
    checks_reference = phase in {"combined", "reference"}
    checks_timing = phase in {"combined", "gems"}
    reference_passed = checks_reference and command_ok and bool(records) and all(
        _positive_finite(item.get("latency_base")) for item in records
    )
    timing_passed = checks_timing and command_ok and bool(records) and all(
        _positive_finite(item.get("latency")) and not item.get("error_msg")
        for item in records
    )
    if result.get("client_error"):
        reason = result["client_error"]
    elif summary.get("status") == "timeout" or job.get("status") == "TIMEOUT":
        reason = "core benchmark 达到阶段超时上限，未形成完整 baseline。"
    elif parse_error:
        reason = f"benchmark JSON 无法解析：{parse_error}"
    elif not records:
        reason = "core benchmark 没有产生可判定的 Workload 记录。"
    elif not command_ok:
        reason = (
            f"core benchmark 未正常退出：job={job.get('status')}，"
            f"exit_code={summary.get('exit_code')}。"
        )
    elif checks_reference and not reference_passed:
        reason = "至少一个 core Workload 缺少有限正数 latency_base。"
    elif checks_timing and not timing_passed:
        reason = "至少一个 core Workload 缺少有效 Gems latency 或包含 error_msg。"
    else:
        reason = ""
    return {
        "phase": phase,
        "reference": (
            "通过" if reference_passed else "未通过" if checks_reference else None
        ),
        "timing": "通过" if timing_passed else "无法计时" if checks_timing else None,
        "case_count": len(records),
        "reason": reason,
    }


def _evidence_link(results_path: pathlib.Path, summary_path: pathlib.Path) -> str:
    return pathlib.Path(
        os.path.relpath(summary_path, start=results_path.parent)
    ).as_posix()


def _update_results(
    results_path: pathlib.Path,
    source_operator: str,
    verdict: dict[str, Any],
    summary_path: pathlib.Path,
    phase: str = "combined",
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
        if verdict["reference"] is not None:
            parts[2] = f" {verdict['reference']} "
        if verdict["timing"] is not None:
            parts[3] = f" {verdict['timing']} "
        if (
            phase == "combined"
            and verdict["reference"] == "通过"
            and verdict["timing"] == "通过"
        ):
            parts[7] = f" —（[baseline证据]({evidence})） "
            parts[8] = " V2 baseline 已通过；完成 `/inspect` 后进入 sample/BatchSimpleOpt。 "
        elif phase == "combined":
            parts[7] = f" {verdict['reason']}（[baseline证据]({evidence})） "
        else:
            current_reason = parts[7].strip()
            current_reason = _PHASE_NOTE_PATTERNS[phase].sub("", current_reason).strip()
            note_label = "Reference 复检" if phase == "reference" else "Gems 可计时复检"
            status = verdict["reference"] if phase == "reference" else verdict["timing"]
            evidence_label = (
                "Reference 复检证据"
                if phase == "reference"
                else "Gems 可计时复检证据"
            )
            detail = verdict["reason"] or "全部 core Workload 完整通过"
            note = f"【{note_label}：{status}；{detail}；[{evidence_label}]({evidence})】"
            parts[7] = f" {current_reason} {note} ".replace(" — ", " ")
            reference_ok = parts[2].strip() == "通过"
            timing_ok = parts[3].strip() == "通过"
            if reference_ok and timing_ok:
                combined_reason = parts[7]
                evidence_links = []
                for checked_phase in ("reference", "gems"):
                    match = _PHASE_EVIDENCE_PATTERNS[checked_phase].search(
                        combined_reason
                    )
                    if match:
                        checked_label = (
                            "Reference 复检证据"
                            if checked_phase == "reference"
                            else "Gems 可计时复检证据"
                        )
                        evidence_links.append(f"[{checked_label}]({match.group(1)})")
                parts[7] = f" —（{'；'.join(evidence_links)}） "
                parts[8] = " V2 baseline 已通过；完成 `/inspect` 后进入 sample/BatchSimpleOpt。 "
        lines[index] = "|".join(parts)
        updated = True
        break
    if not updated:
        raise RuntimeError(f"results row not found: {source_operator}")
    pipeline_path = results_path.parent / "pipeline_results.json"
    pipeline = (
        KernelTodoV2PipelineResults.model_validate_json(
            pipeline_path.read_text(encoding="utf-8")
        )
        if pipeline_path.is_file()
        else None
    )
    _refresh_results_summary(lines, pipeline)
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _submit_operator(
    entry: dict[str, Any],
    *,
    server_url: str,
    flaggems_root: str,
    output_root: pathlib.Path,
    phase_timeout_seconds: int,
    debug_env: dict[str, str],
    phase: str,
) -> tuple[str, dict[str, Any]]:
    baseline_config = {**entry, "baseline_phase": phase}
    request = {
        "command": ["{python}", "run_baseline.py"],
        "files": [{"path": "run_baseline.py", "content": REMOTE_RUNNER}],
        "env": {
            "FLAGGEMS_ROOT": flaggems_root,
            "FLAGGEMS_BASELINE_CONFIG": json.dumps(
                baseline_config, ensure_ascii=False
            ),
            "FLAGGEMS_BASELINE_TIMEOUT": str(phase_timeout_seconds),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            **debug_env,
        },
        "timeout_seconds": min(1800, phase_timeout_seconds + 60),
    }
    started = time.time()
    try:
        job = _request_json(
            server_url,
            "/debug/jobs",
            payload=request,
            timeout=min(1820, phase_timeout_seconds + 80),
        )
        local_dir = output_root / "operators" / entry["source_operator"].replace(
            "/", "_"
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = {}
        for artifact in job.get("artifacts", []):
            content = _download(server_url, artifact["download_url"])
            digest = hashlib.sha256(content).hexdigest()
            if digest != artifact["sha256"]:
                raise RuntimeError(
                    "artifact digest mismatch: "
                    f"{entry['source_operator']}:{artifact['path']}"
                )
            target = local_dir / artifact["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            downloaded[artifact["path"]] = {
                "path": str(target),
                "size_bytes": len(content),
                "sha256": digest,
            }
        result = {
            "source_operator": entry["source_operator"],
            "entry": entry,
            "job": job,
            "downloaded_artifacts": downloaded,
            "local_dir": str(local_dir),
            "started_at": started,
            "completed_at": time.time(),
        }
        summary_path = local_dir / "summary.json"
        if summary_path.exists():
            result["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        result["verdict"] = _verdict(result, phase)
        return entry["source_operator"], result
    except BaseException as exc:
        local_dir = output_root / "operators" / entry["source_operator"].replace(
            "/", "_"
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "source_operator": entry["source_operator"],
            "entry": entry,
            "client_error": f"{type(exc).__name__}: {exc}",
            "local_dir": str(local_dir),
            "started_at": started,
            "completed_at": time.time(),
        }
        result["verdict"] = _verdict(result, phase)
        _atomic_json(local_dir / "summary.json", result)
        return entry["source_operator"], result


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = (
        args.results_md.resolve()
        if args.results_md
        else REPO_ROOT / "kernel_todo_v2" / args.chip / "results.md"
    )
    progress_path = output_root / "baseline_progress.json"
    targets = _load_targets(args.chip, results_path, set(args.operator), args.phase)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["server_url"] != args.server_url:
            raise RuntimeError("existing progress uses a different Server URL")
        if progress["flaggems_root"] != args.flaggems_root:
            raise RuntimeError("existing progress uses a different FlagGems root")
        if progress.get("phase", "combined") != args.phase:
            raise RuntimeError("existing progress uses a different baseline phase")
    else:
        progress = {
            "schema_version": "kernelgen.kernel-todo-v2-baseline-progress/v1",
            "chip": args.chip,
            "server_url": args.server_url,
            "flaggems_root": args.flaggems_root,
            "phase": args.phase,
            "phase_timeout_seconds": args.phase_timeout_seconds,
            "results": {},
            "scheduler_snapshots": [],
        }
    remaining = [
        entry
        for entry in targets
        if entry["source_operator"] not in progress["results"]
    ]
    if args.limit is not None:
        remaining = remaining[: args.limit]
    print(
        f"targets={len(targets)} completed={len(progress['results'])} "
        f"remaining={len(remaining)}",
        flush=True,
    )
    group_size = args.group_size or max(1, len(remaining))
    for group_number, start in enumerate(range(0, len(remaining), group_size), start=1):
        group = remaining[start : start + group_size]
        before = _scheduler_snapshot(
            args.server_url, f"group-{group_number}-before"
        )
        progress["scheduler_snapshots"].append(before)
        _atomic_json(progress_path, progress)
        print(f"group={group_number} count={len(group)} start", flush=True)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_parallel
        ) as executor:
            futures = [
                executor.submit(
                    _submit_operator,
                    entry,
                    server_url=args.server_url,
                    flaggems_root=args.flaggems_root,
                    output_root=output_root,
                    phase_timeout_seconds=args.phase_timeout_seconds,
                    debug_env=args.debug_env,
                    phase=args.phase,
                )
                for entry in group
            ]
            for future in concurrent.futures.as_completed(futures):
                source_operator, result = future.result()
                progress["results"][source_operator] = result
                _atomic_json(progress_path, progress)
                summary_path = (
                    output_root
                    / "operators"
                    / source_operator.replace("/", "_")
                    / "summary.json"
                )
                if not args.no_update_results:
                    _update_results(
                        results_path,
                        source_operator,
                        result["verdict"],
                        summary_path,
                        args.phase,
                    )
                print(
                    json.dumps(
                        {
                            "operator": source_operator,
                            "job_status": result.get("job", {}).get("status"),
                            **result["verdict"],
                            "client_error": result.get("client_error"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        after = _scheduler_snapshot(
            args.server_url, f"group-{group_number}-after"
        )
        progress["scheduler_snapshots"].append(after)
        _atomic_json(progress_path, progress)
        print(
            f"group={group_number} done scheduler={after['scheduler']}", flush=True
        )


if __name__ == "__main__":
    main()

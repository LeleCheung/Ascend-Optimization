#!/usr/bin/env python3
"""Run reviewed pytest conversions whenever local NVIDIA cards are truly free."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_state.json"
DEFAULT_BATCH_ROOT = KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_batches"
DEFAULT_MANIFEST = KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_inventory.json"
DEFAULT_APPROVED_FILES = (
    KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_approved_files"
)
DEFAULT_DEVICE_RUNS = KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_device_runs"
DEFAULT_RUN_ROOT = (
    KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_backlog_runs"
)
NVIDIA_VALIDATOR = Path(__file__).with_name("validate_batch_nvidia.py")
REMOTE_VALIDATOR = Path(__file__).with_name("validate_batch_metax.py")
OPEN_REVIEW = Path(__file__).with_name("open_batch_review.py")
RECORD_REVIEW = Path(__file__).with_name("record_review.py")
RESOURCE_PATTERNS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "outofmemoryerror",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _latest_decisions(state: dict[str, Any]) -> dict[str, str]:
    decisions = {}
    for review in state.get("history", []):
        decisions[review["source_operator"]] = review["decision"]
    return decisions


@dataclass(frozen=True)
class Candidate:
    batch: Path
    source_operator: str
    operator: str
    target_paths: tuple[str, ...]


@dataclass
class Running:
    candidate: Candidate
    device: str
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: Any
    started_at: str


def _candidates(
    batches: list[Path],
    *,
    state_path: Path,
    flaggems_repo: Path,
    blocked: set[str],
    cooldown_until: dict[str, float],
    running: dict[str, Running],
    priority: list[str],
) -> tuple[list[Candidate], dict[str, str]]:
    state = _read_json(state_path)
    approved = {
        operator
        for operator, decision in _latest_decisions(state).items()
        if decision == "approved"
    }
    excluded = set(state.get("protocol_gaps", {})) | set(
        state.get("validation_blockers", {})
    )
    now = time.monotonic()
    selected: dict[str, Candidate] = {}
    stale: dict[str, str] = {}
    for batch_path in sorted(batches):
        batch = _read_json(batch_path)
        for item in batch["items"]:
            source_operator = item["source_operator"]
            if (
                item["status"] not in ("reviewed", "reviewed_with_fixes")
                or item.get("review_decision") != "approved"
                or source_operator in approved
                or source_operator in excluded
                or source_operator in blocked
                or source_operator in running
                or cooldown_until.get(source_operator, 0) > now
            ):
                continue
            current_hashes = {
                path: _file_hash(flaggems_repo / path)
                for path in item["target_paths"]
            }
            if current_hashes != item.get("reviewed_hashes"):
                stale[source_operator] = (
                    f"reviewed hashes are stale in {batch_path}"
                )
                continue
            selected[source_operator] = Candidate(
                batch=batch_path,
                source_operator=source_operator,
                operator=item["operator"],
                target_paths=tuple(item["target_paths"]),
            )
    priority_rank = {operator: index for index, operator in enumerate(priority)}
    values = list(selected.values())
    values.sort(
        key=lambda candidate: (
            priority_rank.get(candidate.source_operator, len(priority_rank)),
            str(candidate.batch),
            candidate.source_operator,
        )
    )
    return values, stale


def _free_nvidia_devices(
    container: str, *, max_used_mib: int, max_utilization: int
) -> list[str]:
    completed = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = []
    for line in completed.stdout.splitlines():
        index, used, utilization = [part.strip() for part in line.split(",")]
        if int(used) <= max_used_mib and int(utilization) <= max_utilization:
            result.append(index)
    return result


def _ssh(args: argparse.Namespace) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
        "-i",
        str(args.identity_file),
        "-p",
        str(args.port),
        "-l",
        args.login,
        args.host,
    ]


def _free_ppu_devices(args: argparse.Namespace) -> list[str]:
    remote = [
        "sudo",
        "-n",
        "docker",
        "exec",
        args.container,
        "ppu-smi",
        "--query-ppu=index,memory.used,utilization.ppu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        [*_ssh(args), shlex.join(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = []
    for line in completed.stdout.splitlines():
        index, used, utilization = [part.strip() for part in line.split(",")]
        if int(used) <= args.max_used_mib and int(utilization) <= args.max_utilization:
            result.append(index)
    return result


def _free_devices(args: argparse.Namespace) -> list[str]:
    if args.backend == "remote-ppu":
        devices = _free_ppu_devices(args)
    else:
        devices = _free_nvidia_devices(
            args.container,
            max_used_mib=args.max_used_mib,
            max_utilization=args.max_utilization,
        )
    allowed = {value.strip() for value in args.devices.split(",") if value.strip()}
    return [device for device in devices if not allowed or device in allowed]


def _validation_result(candidate: Candidate) -> dict[str, Any] | None:
    batch = _read_json(candidate.batch)
    for item in batch["items"]:
        if item["source_operator"] == candidate.source_operator:
            return item.get("device_validation")
    return None


def _resource_failure(running: Running) -> bool:
    text = running.log_path.read_text(encoding="utf-8", errors="replace").lower()
    result = _validation_result(running.candidate) or {}
    for key in ("log", "record"):
        path_value = result.get(key)
        if path_value and Path(path_value).is_file():
            text += Path(path_value).read_text(
                encoding="utf-8", errors="replace"
            ).lower()
    return any(pattern in text for pattern in RESOURCE_PATTERNS)


def _approve(
    candidate: Candidate,
    result: dict[str, Any],
    *,
    validation_label: str,
    state_path: Path,
    flaggems_repo: Path,
    approved_files: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(OPEN_REVIEW),
            "--batch",
            str(candidate.batch),
            "--operator",
            candidate.source_operator,
            "--state",
            str(state_path),
            "--flaggems-repo",
            str(flaggems_repo),
        ],
        cwd=KERNELGEN_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(RECORD_REVIEW),
            "--operator",
            candidate.source_operator,
            "--decision",
            "approved",
            "--notes",
            f"Primary review and {validation_label} full validation passed; "
            f"record={result.get('record')}",
            "--state",
            str(state_path),
            "--flaggems-repo",
            str(flaggems_repo),
            "--approved-files",
            str(approved_files),
        ],
        cwd=KERNELGEN_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _start(
    candidate: Candidate,
    *,
    device: str,
    run_root: Path,
    args: argparse.Namespace,
) -> Running:
    log_path = run_root / (
        f"{_safe_name(candidate.source_operator)}-device-{device}.log"
    )
    log_handle = log_path.open("a", encoding="utf-8")
    validator = REMOTE_VALIDATOR if args.backend == "remote-ppu" else NVIDIA_VALIDATOR
    command = [
        sys.executable,
        str(validator),
        "--batch",
        str(candidate.batch),
        "--state",
        str(args.state.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--flaggems-repo",
        str(args.flaggems_repo.resolve()),
        "--approved-files",
        str(args.approved_files.resolve()),
        "--runs-dir",
        str(args.runs_dir.resolve()),
        "--operators",
        candidate.source_operator,
        "--devices",
        device,
    ]
    if args.backend == "remote-ppu":
        command.extend(
            [
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--login",
                args.login,
                "--identity-file",
                str(args.identity_file),
                "--container",
                args.container,
                "--remote-template-root",
                args.remote_template_root,
                "--remote-python",
                args.remote_python,
                "--with-site",
                "--session-python",
                args.session_python,
                "--device-env",
                "CUDA_VISIBLE_DEVICES",
                "--remote-site-packages",
                args.remote_site_packages,
                "--connect-timeout",
                str(args.connect_timeout),
                "--control-persist",
                str(args.control_persist),
                "--command-timeout",
                str(args.command_timeout),
            ]
        )
    process = subprocess.Popen(
        command,
        cwd=KERNELGEN_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return Running(
        candidate=candidate,
        device=device,
        process=process,
        log_path=log_path,
        log_handle=log_handle,
        started_at=_now(),
    )


def _stop_running_jobs(running: dict[str, Running]) -> None:
    active = [job for job in running.values() if job.process.poll() is None]
    for job in active:
        try:
            os.killpg(job.process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    for job in active:
        try:
            job.process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(job.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for job in active:
        if job.process.poll() is None:
            try:
                job.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(job.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                job.process.wait()
    for job in running.values():
        if not job.log_handle.closed:
            job.log_handle.close()


def _parse_args(*, default_backend: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("local-nvidia", "remote-ppu"),
        default=default_backend,
    )
    parser.add_argument("--batches", nargs="*", type=Path)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--approved-files", type=Path, default=DEFAULT_APPROVED_FILES
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_DEVICE_RUNS)
    parser.add_argument(
        "--flaggems-repo", type=Path, default=KERNELGEN_ROOT.parent / "FlagGems-master"
    )
    parser.add_argument("--container")
    parser.add_argument("--devices", default="")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--resource-cooldown-seconds", type=int, default=120)
    parser.add_argument("--max-resource-retries", type=int, default=20)
    parser.add_argument("--max-used-mib", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=5)
    parser.add_argument("--priority", nargs="*", default=[])
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--login")
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--remote-template-root")
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--session-python", default="python3")
    parser.add_argument(
        "--remote-site-packages",
        default="",
    )
    parser.add_argument("--connect-timeout", type=int, default=20)
    parser.add_argument("--control-persist", type=int, default=600)
    parser.add_argument("--command-timeout", type=int, default=1800)
    parser.add_argument("--validation-label")
    args = parser.parse_args()
    if args.container is None and args.backend == "local-nvidia":
        args.container = "kernelgen-nvidia-cu128"
    if args.backend == "remote-ppu" and not args.remote_template_root:
        parser.error("--remote-template-root is required for remote PPU validation")
    if args.backend == "remote-ppu":
        missing = [
            option
            for option, value in (
                ("--host", args.host),
                ("--port", args.port),
                ("--login", args.login),
                ("--identity-file", args.identity_file),
                ("--container", args.container),
            )
            if value is None
        ]
        if missing:
            parser.error(
                "remote PPU validation requires " + ", ".join(missing)
            )
    if args.validation_label is None:
        args.validation_label = (
            "remote T-Head PPU"
            if args.backend == "remote-ppu"
            else "local NVIDIA"
        )
    return args


def main(*, default_backend: str = "local-nvidia") -> None:
    args = _parse_args(default_backend=default_backend)
    batches = [path.resolve() for path in (args.batches or [])]
    if not batches:
        batches = sorted(DEFAULT_BATCH_ROOT.glob("*/manifest.json"))
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_root = (args.run_root or (DEFAULT_RUN_ROOT / timestamp)).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "summary.json"
    summary: dict[str, Any] = {
        "schema_version": "kernelgen.pytest-conversion-device-backlog/v1",
        "backend": args.backend,
        "started_at": _now(),
        "batches": [str(path) for path in batches],
        "events": [],
        "blocked": {},
        "resource_retries": {},
    }
    running: dict[str, Running] = {}
    blocked: set[str] = set()
    cooldown_until: dict[str, float] = {}
    resource_retries: dict[str, int] = {}
    stale_reported: set[str] = set()
    approval_halted: str | None = None

    def save() -> None:
        summary["approval_halted"] = approval_halted
        summary["running"] = {
            operator: {
                "device": job.device,
                "batch": str(job.candidate.batch),
                "log": str(job.log_path),
                "started_at": job.started_at,
            }
            for operator, job in running.items()
        }
        summary["resource_retries"] = resource_retries
        _write_json(summary_path, summary)

    save()
    try:
        while True:
            for operator, job in list(running.items()):
                returncode = job.process.poll()
                if returncode is None:
                    continue
                job.log_handle.close()
                result = _validation_result(job.candidate) or {}
                event = {
                    "operator": operator,
                    "device": job.device,
                    "batch": str(job.candidate.batch),
                    "log": str(job.log_path),
                    "finished_at": _now(),
                    "returncode": returncode,
                }
                if (
                    returncode == 0
                    and result.get("status") == "passed"
                    and approval_halted is None
                ):
                    try:
                        _approve(
                            job.candidate,
                            result,
                            validation_label=args.validation_label,
                            state_path=args.state.resolve(),
                            flaggems_repo=args.flaggems_repo.resolve(),
                            approved_files=args.approved_files.resolve(),
                        )
                        event["status"] = "approved"
                        print(f"[approved] {operator} device={job.device}", flush=True)
                    except Exception as exc:
                        event["status"] = "approval_failed"
                        event["error"] = f"{type(exc).__name__}: {exc}"
                        approval_halted = event["error"]
                        blocked.add(operator)
                        summary["blocked"][operator] = event["error"]
                        print(f"[approval_failed] {operator}: {exc}", flush=True)
                elif returncode == 0 and result.get("status") == "passed":
                    event["status"] = "approval_pending"
                    blocked.add(operator)
                    summary["blocked"][operator] = (
                        "validation passed but automatic approval is halted: "
                        f"{approval_halted}"
                    )
                    print(f"[approval_pending] {operator}", flush=True)
                elif _resource_failure(job):
                    retries = resource_retries.get(operator, 0) + 1
                    resource_retries[operator] = retries
                    event["status"] = "resource_retry"
                    cooldown_until[operator] = time.monotonic() + (
                        args.resource_cooldown_seconds
                    )
                    if retries >= args.max_resource_retries:
                        blocked.add(operator)
                        summary["blocked"][operator] = (
                            f"resource retry limit reached ({retries})"
                        )
                    print(
                        f"[resource_retry] {operator} device={job.device} "
                        f"attempt={retries}",
                        flush=True,
                    )
                else:
                    event["status"] = "validation_failed"
                    event["validation"] = result
                    blocked.add(operator)
                    summary["blocked"][operator] = (
                        f"validation failed; log={job.log_path}"
                    )
                    print(
                        f"[validation_failed] {operator} device={job.device}",
                        flush=True,
                    )
                summary["events"].append(event)
                del running[operator]
                save()

            candidates, stale = _candidates(
                batches,
                state_path=args.state.resolve(),
                flaggems_repo=args.flaggems_repo.resolve(),
                blocked=blocked,
                cooldown_until=cooldown_until,
                running=running,
                priority=args.priority,
            )
            for operator, reason in stale.items():
                if operator not in stale_reported:
                    stale_reported.add(operator)
                    blocked.add(operator)
                    summary["blocked"][operator] = reason
                    print(f"[stale_review] {operator}: {reason}", flush=True)
            occupied_devices = {job.device for job in running.values()}
            occupied_paths = {
                path for job in running.values() for path in job.candidate.target_paths
            }
            free_devices = []
            if approval_halted is None:
                free_devices = [
                    device
                    for device in _free_devices(args)
                    if device not in occupied_devices
                ]
            for device in free_devices:
                candidate = next(
                    (
                        item
                        for item in candidates
                        if item.source_operator not in running
                        and occupied_paths.isdisjoint(item.target_paths)
                    ),
                    None,
                )
                if candidate is None:
                    break
                job = _start(
                    candidate,
                    device=device,
                    run_root=run_root,
                    args=args,
                )
                running[candidate.source_operator] = job
                occupied_paths.update(candidate.target_paths)
                print(
                    f"[started] {candidate.source_operator} device={device} "
                    f"batch={candidate.batch.parent.name}",
                    flush=True,
                )
                save()

            remaining, _ = _candidates(
                batches,
                state_path=args.state.resolve(),
                flaggems_repo=args.flaggems_repo.resolve(),
                blocked=blocked,
                cooldown_until={},
                running=running,
                priority=args.priority,
            )
            if not running and (not remaining or approval_halted is not None):
                break
            time.sleep(max(1, args.poll_seconds))
    except KeyboardInterrupt:
        summary["interrupted_at"] = _now()
        raise
    except Exception as exc:
        summary["fatal_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _stop_running_jobs(running)
        summary["finished_at"] = _now()
        save()
        print(f"Backlog summary: {summary_path}", flush=True)

    if summary["blocked"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

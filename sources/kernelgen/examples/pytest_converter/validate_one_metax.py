#!/usr/bin/env python3
"""Validate one active pytest conversion in an isolated device checkout."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any

KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLAGGEMS_REPO = KERNELGEN_ROOT.parent / "FlagGems-master"
DEFAULT_STATE = KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_state.json"
DEFAULT_MANIFEST = (
    KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_inventory.json"
)
DEFAULT_APPROVED_FILES = (
    KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_approved_files"
)
DEFAULT_RUNS = KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_device_runs"
PROBE = Path(__file__).with_name("remote_override_probe.py")
REMOTE_EXECUTOR = r"""
import base64
import json
import subprocess
import sys

for line in sys.stdin:
    request = json.loads(line)
    if request.get("action") == "exit":
        break
    try:
        completed = subprocess.run(
            request["command"],
            input=base64.b64decode(request.get("input", "")),
            capture_output=True,
            timeout=request["timeout"],
        )
        response = {
            "returncode": completed.returncode,
            "stdout": base64.b64encode(completed.stdout).decode(),
            "stderr": base64.b64encode(completed.stderr).decode(),
        }
    except subprocess.TimeoutExpired as exc:
        response = {
            "returncode": 124,
            "stdout": base64.b64encode(exc.stdout or b"").decode(),
            "stderr": base64.b64encode(
                (exc.stderr or b"") + b"remote command timed out\n"
            ).decode(),
        }
    sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    sys.stdout.flush()
"""


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _find_item(manifest: dict[str, Any], source_operator: str) -> dict[str, Any]:
    return next(
        item
        for item in manifest["operators"]
        if item["source_operator"] == source_operator
    )


def _git_file(repo: Path, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _tar_payload(files: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for relative_path, content in files.items():
            info = tarfile.TarInfo(relative_path)
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    return payload.getvalue()


def _pytest_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in ("passed", "failed", "skipped", "xfailed", "xpassed"):
        matches = re.findall(rf"(\d+) {name}", output)
        if matches:
            counts[name] = int(matches[-1])
    return counts


def _benchmark_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in value.values():
        for detail in result.get("details", []):
            rows.extend(detail.get("result", []))
    return rows


def _benchmark_signature(value: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row.get("case_id"),
            "dtype": row.get("dtype"),
            "legacy_shape": row.get("legacy_shape"),
            "shape_detail": row.get("shape_detail"),
        }
        for row in _benchmark_rows(value)
    ]


def _case_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        case
        for benchmark in value.get("benchmarks", [])
        for case in benchmark.get("cases", [])
    ]


class MetaXRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.commands: list[dict[str, Any]] = []
        self.session: subprocess.Popen[bytes] | None = None

    @property
    def ssh(self) -> list[str]:
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.args.connect_timeout}",
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPersist={self.args.control_persist}",
            "-o",
            f"ControlPath={self.args.control_path}",
            "-i",
            str(self.args.identity_file),
            "-p",
            str(self.args.port),
            "-l",
            self.args.login,
            self.args.host,
        ]

    def _docker(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        interactive: bool = False,
    ) -> list[str]:
        result = ["sudo", "-n", "docker", "exec"]
        if interactive:
            result.append("-i")
        if workdir:
            result.extend(["-w", workdir])
        pythonpath = [self.args.remote_root + "/src"]
        if self.args.remote_site_packages:
            pythonpath.append(self.args.remote_site_packages)
        pythonpath.append("/tmp")
        result.extend(
            [
                "-e",
                f"{self.args.device_env}={self.args.device}",
                "-e",
                "PYTHONPATH=" + ":".join(pythonpath),
                self.args.container,
                *command,
            ]
        )
        return result

    def prepare(self, source_root: str) -> None:
        self.run(
            ["mkdir", "-p", self.args.remote_root],
            label="prepare-remote-root",
        )
        self.run(
            [
                "cp",
                "-a",
                "--reflink=auto",
                f"{source_root}/.",
                self.args.remote_root,
            ],
            label="copy-remote-checkout",
        )

    def cleanup(self) -> None:
        root = Path(self.args.remote_root)
        if not root.name.startswith("kernelgen-pytest-"):
            raise RuntimeError(f"refusing to remove unsafe remote root: {root}")
        self.run(
            ["rm", "-rf", "--", str(root)],
            check=False,
            label="cleanup-remote-root",
        )

    def _start_session(self) -> None:
        if self.session is not None:
            return
        remote = [self.args.session_python, "-u", "-c", REMOTE_EXECUTOR]
        exact = self.ssh + [shlex.join(remote)]
        print(f"[ssh-session] {shlex.join(exact)}", flush=True)
        self.session = subprocess.Popen(
            exact,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
        )

    def close(self) -> None:
        if self.session is None:
            return
        if self.session.poll() is None and self.session.stdin is not None:
            try:
                self.session.stdin.write(b'{"action":"exit"}\n')
                self.session.stdin.flush()
                self.session.stdin.close()
                self.session.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.session.kill()
                self.session.wait()
        self.session = None

    def run(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        input_bytes: bytes | None = None,
        check: bool = True,
        label: str,
    ) -> subprocess.CompletedProcess[bytes]:
        remote = self._docker(
            command, workdir=workdir, interactive=input_bytes is not None
        )
        print(f"[{label}] {shlex.join(remote)}", flush=True)
        self._start_session()
        assert self.session is not None
        assert self.session.stdin is not None
        assert self.session.stdout is not None
        request = {
            "command": remote,
            "input": base64.b64encode(input_bytes or b"").decode(),
            "timeout": self.args.command_timeout,
        }
        self.session.stdin.write(
            (json.dumps(request, separators=(",", ":")) + "\n").encode()
        )
        self.session.stdin.flush()
        response_line = self.session.stdout.readline()
        if not response_line:
            raise RuntimeError(
                "persistent SSH command session exited without a response"
            )
        response = json.loads(response_line)
        completed = subprocess.CompletedProcess(
            remote,
            response["returncode"],
            base64.b64decode(response["stdout"]),
            base64.b64decode(response["stderr"]),
        )
        output = (completed.stdout + completed.stderr).decode(
            "utf-8", errors="replace"
        )
        print(output[-4000:], flush=True)
        self.commands.append(
            {
                "label": label,
                "command": remote,
                "exit_code": completed.returncode,
                "output_tail": output[-12000:],
            }
        )
        if check and completed.returncode != 0:
            raise RuntimeError(
                f"{label} failed with exit code {completed.returncode}"
            )
        return completed

    def install(self, files: dict[str, bytes], *, label: str) -> None:
        self.run(
            ["tar", "-xf", "-"],
            workdir=self.args.remote_root,
            input_bytes=_tar_payload(files),
            label=label,
        )

    def pytest(
        self,
        path: str,
        marker: str,
        extra: list[str],
        *,
        label: str,
    ) -> str:
        command = [self.args.remote_python]
        if not self.args.with_site:
            command.append("-S")
        command.extend(["-m", "pytest", "-q", path, "-m", marker, *extra])
        completed = self.run(
            command, workdir=self.args.remote_root, label=label
        )
        return (completed.stdout + completed.stderr).decode(
            "utf-8", errors="replace"
        )

    def read_json(self, path: str, *, label: str) -> dict[str, Any]:
        completed = self.run(["cat", path], label=label)
        return json.loads(completed.stdout.decode("utf-8"))

    def hashes(self, paths: list[str], *, label: str) -> dict[str, str]:
        completed = self.run(
            ["sha256sum", *paths], workdir=self.args.remote_root, label=label
        )
        result = {}
        for line in completed.stdout.decode().splitlines():
            digest, relative_path = line.split(maxsplit=1)
            result[relative_path] = digest
        return result


class LocalNvidiaRunner(MetaXRunner):
    def _docker(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        interactive: bool = False,
    ) -> list[str]:
        result = ["docker", "exec"]
        if interactive:
            result.append("-i")
        if workdir:
            result.extend(["-w", workdir])
        result.extend(
            [
                "-e",
                f"CUDA_VISIBLE_DEVICES={self.args.device}",
                "-e",
                f"PYTHONPATH={self.args.remote_root}/src:/tmp",
                self.args.container,
                *command,
            ]
        )
        return result

    def run(
        self,
        command: list[str],
        *,
        workdir: str | None = None,
        input_bytes: bytes | None = None,
        check: bool = True,
        label: str,
    ) -> subprocess.CompletedProcess[bytes]:
        exact = self._docker(
            command, workdir=workdir, interactive=input_bytes is not None
        )
        print(f"[{label}] {shlex.join(exact)}", flush=True)
        try:
            completed = subprocess.run(
                exact,
                input=input_bytes,
                capture_output=True,
                timeout=self.args.command_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            completed = subprocess.CompletedProcess(
                exact,
                124,
                exc.stdout or b"",
                (exc.stderr or b"") + b"local command timed out\n",
            )
        output = (completed.stdout + completed.stderr).decode(
            "utf-8", errors="replace"
        )
        print(output[-4000:], flush=True)
        self.commands.append(
            {
                "label": label,
                "command": exact,
                "exit_code": completed.returncode,
                "output_tail": output[-12000:],
            }
        )
        if check and completed.returncode != 0:
            raise RuntimeError(
                f"{label} failed with exit code {completed.returncode}"
            )
        return completed

    def pytest(
        self,
        path: str,
        marker: str,
        extra: list[str],
        *,
        label: str,
    ) -> str:
        command = [self.args.remote_python, "-m", "pytest", "-q"]
        command.extend([path, "-m", marker, *extra])
        completed = self.run(
            command, workdir=self.args.remote_root, label=label
        )
        return (completed.stdout + completed.stderr).decode(
            "utf-8", errors="replace"
        )

    def prepare(self, source_root: Path) -> None:
        self.run(
            ["mkdir", "-p", self.args.remote_root],
            label="prepare-local-root",
        )
        self.run(
            ["cp", "-a", f"{source_root}/.", self.args.remote_root],
            label="copy-local-checkout",
        )

    def cleanup(self) -> None:
        root = Path(self.args.remote_root)
        if root.parent != Path("/tmp") or not root.name.startswith(
            "kernelgen-pytest-nvidia-"
        ):
            raise RuntimeError(f"refusing to remove unsafe local root: {root}")
        self.run(
            ["rm", "-rf", "--", str(root)],
            check=False,
            label="cleanup-local-root",
        )


def _bootstrap_snapshots(
    state: dict[str, Any], repo: Path, approved_files: Path
) -> int:
    if state.get("active_review"):
        raise RuntimeError("bootstrap requires an empty review gate")
    copied = 0
    for relative_path, expected in state.get("approved_file_hashes", {}).items():
        source = repo / relative_path
        if _file_hash(source) != expected:
            raise RuntimeError(
                f"current file no longer matches approved hash: {relative_path}"
            )
        destination = approved_files / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        copied += 1
    return copied


def _baseline_files(
    paths: list[str],
    state: dict[str, Any],
    repo: Path,
    approved_files: Path,
) -> dict[str, bytes]:
    result = {}
    approved = state.get("approved_file_hashes", {})
    for relative_path in paths:
        if relative_path in approved:
            source = approved_files / relative_path
            if not source.is_file() or _file_hash(source) != approved[relative_path]:
                raise RuntimeError(
                    f"missing approved snapshot for {relative_path}; bootstrap first"
                )
            result[relative_path] = source.read_bytes()
        else:
            result[relative_path] = _git_file(repo, relative_path)
    return result


def _parse_args(*, default_local_nvidia: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-approved", action="store_true")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--flaggems-repo", type=Path, default=DEFAULT_FLAGGEMS_REPO)
    parser.add_argument("--approved-files", type=Path, default=DEFAULT_APPROVED_FILES)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--host", default="bastion.aiops.baai.ac.cn")
    parser.add_argument("--port", type=int, default=2224)
    parser.add_argument("--login", default="xuyao@secure@192.168.2.115")
    parser.add_argument("--identity-file", type=Path, default="/root/.ssh/id_ed25519")
    parser.add_argument(
        "--local-nvidia", action="store_true", default=default_local_nvidia
    )
    parser.add_argument("--container")
    parser.add_argument(
        "--remote-root", default="/tmp/kernelgen-pytest-gt-scalar.jbsEdW"
    )
    parser.add_argument("--remote-python")
    parser.add_argument("--device-env", default="MACA_VISIBLE_DEVICES")
    parser.add_argument(
        "--remote-site-packages",
        default="/opt/conda/lib/python3.12/site-packages",
    )
    parser.add_argument("--remote-template-root")
    parser.add_argument("--with-site", action="store_true")
    parser.add_argument("--session-python", default="python3")
    parser.add_argument("--device", default="0")
    parser.add_argument("--connect-timeout", type=int, default=20)
    parser.add_argument("--control-persist", type=int, default=600)
    parser.add_argument(
        "--control-path",
        default="/tmp/kernelgen-pytest-converter-ssh-%C",
    )
    parser.add_argument("--command-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.container is None:
        args.container = (
            "kernelgen-nvidia-cu128"
            if args.local_nvidia
            else "kernelgen-ks-metax"
        )
    if args.remote_python is None:
        args.remote_python = (
            "python3" if args.local_nvidia else "/opt/conda/bin/python3"
        )
    return args


def main(*, default_local_nvidia: bool = False) -> None:
    args = _parse_args(default_local_nvidia=default_local_nvidia)
    repo = args.flaggems_repo.resolve()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    approved_files = args.approved_files.resolve()
    if args.bootstrap_approved:
        copied = _bootstrap_snapshots(state, repo, approved_files)
        print(f"Bootstrapped {copied} approved file snapshots.")
        return

    active = state.get("active_review")
    if not active or active.get("review_status") != "review_pending":
        raise RuntimeError("exactly one review_pending conversion is required")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    item = _find_item(manifest, active["source_operator"])
    paths = active["target_paths"]
    local_hashes = {path: _file_hash(repo / path) for path in paths}
    if local_hashes != active["target_hashes"]:
        raise RuntimeError("local target files changed after the Agent run")

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    safe = _safe_name(active["source_operator"])
    if args.local_nvidia:
        args.remote_root = (
            f"/tmp/kernelgen-pytest-nvidia-{safe}-{timestamp}-{os.getpid()}"
        )
    elif args.remote_template_root:
        template_parent = str(Path(args.remote_template_root).parent)
        args.remote_root = (
            f"{template_parent}/kernelgen-pytest-remote-{safe}-"
            f"{timestamp}-{os.getpid()}"
        )
    remote_prefix = f"/tmp/kernelgen-pytest-converter-{safe}-{timestamp}"
    record_path = args.runs_dir.resolve() / f"{timestamp}-{safe}.json"
    runner = LocalNvidiaRunner(args) if args.local_nvidia else MetaXRunner(args)
    record: dict[str, Any] = {
        "schema_version": "kernelgen.pytest-conversion-device-run/v1",
        "source_operator": active["source_operator"],
        "operator": active["operator"],
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "running",
        "commands": runner.commands,
    }
    try:
        if args.local_nvidia:
            runner.prepare(repo)
        elif args.remote_template_root:
            runner.prepare(args.remote_template_root)
        baseline = _baseline_files(paths, state, repo, approved_files)
        baseline_hashes = {
            path: hashlib.sha256(content).hexdigest()
            for path, content in baseline.items()
        }
        runner.install(baseline, label="install-baseline")
        if runner.hashes(paths, label="verify-baseline") != baseline_hashes:
            raise RuntimeError("remote baseline hashes do not match")

        accuracy = item["accuracy_files"][0]
        benchmark = item["benchmark_files"][0]
        marker = item["pytest_mark"]
        baseline_quick = runner.pytest(
            accuracy, marker, ["--quick"], label="baseline-accuracy-quick"
        )
        baseline_full = runner.pytest(
            accuracy, marker, [], label="baseline-accuracy-full"
        )
        baseline_result_path = remote_prefix + "-baseline-core.json"
        runner.pytest(
            benchmark,
            marker,
            [
                "--level",
                "core",
                "--record",
                "json",
                "--output",
                baseline_result_path,
            ],
            label="baseline-benchmark-core",
        )
        baseline_result = runner.read_json(
            baseline_result_path, label="read-baseline-result"
        )

        current = {path: (repo / path).read_bytes() for path in paths}
        runner.install(current, label="install-converted")
        if runner.hashes(paths, label="verify-converted") != local_hashes:
            raise RuntimeError("remote converted hashes do not match")
        converted_quick = runner.pytest(
            accuracy, marker, ["--quick"], label="converted-accuracy-quick"
        )
        converted_full = runner.pytest(
            accuracy, marker, [], label="converted-accuracy-full"
        )
        if _pytest_counts(baseline_quick) != _pytest_counts(converted_quick):
            raise RuntimeError("quick accuracy outcome counts changed")
        if _pytest_counts(baseline_full) != _pytest_counts(converted_full):
            raise RuntimeError("full accuracy outcome counts changed")
        converted_result_path = remote_prefix + "-converted-core.json"
        runner.pytest(
            benchmark,
            marker,
            [
                "--level",
                "core",
                "--record",
                "json",
                "--output",
                converted_result_path,
            ],
            label="converted-benchmark-core",
        )
        converted_result = runner.read_json(
            converted_result_path, label="read-converted-result"
        )
        baseline_rows = _benchmark_rows(baseline_result)
        converted_rows = _benchmark_rows(converted_result)
        if len(baseline_rows) != len(converted_rows):
            raise RuntimeError("core benchmark Workload count changed")
        if any(row.get("error_msg") for row in converted_rows):
            raise RuntimeError("converted core benchmark contains an error")
        if {row.get("candidate_source") for row in converted_rows} != {"default"}:
            raise RuntimeError("converted benchmark did not use the default gems_op")

        cases_path = remote_prefix + "-cases.json"
        runner.pytest(
            benchmark,
            marker,
            ["--level", "core", "--list-cases", "--output", cases_path],
            label="converted-list-cases",
        )
        cases = runner.read_json(cases_path, label="read-case-list")
        case_rows = _case_rows(cases)
        case_ids = [case["case_id"] for case in case_rows]
        if not case_ids or len(case_ids) != len(set(case_ids)):
            raise RuntimeError("case list is empty or contains duplicate IDs")
        exact_case = case_ids[0]
        runner.pytest(
            benchmark,
            marker,
            ["--level", "core", "--preflight-only"],
            label="converted-preflight",
        )
        exact_path = remote_prefix + "-exact.json"
        runner.pytest(
            benchmark,
            marker,
            [
                "--level",
                "core",
                "--case-id",
                exact_case,
                "--record",
                "json",
                "--output",
                exact_path,
            ],
            label="converted-exact-case",
        )
        runner.pytest(
            benchmark,
            marker,
            [
                "--level",
                "core",
                "--case-id",
                exact_case,
                "--profile-only",
                "--profile-warmup",
                "1",
                "--profile-iterations",
                "1",
            ],
            label="converted-profile-only",
        )

        runner.install(
            {"kernelgen_override_probe.py": PROBE.read_bytes()},
            label="install-override-probe",
        )
        # Environment values are added as docker-exec options for these probes.
        original_docker = runner._docker

        def docker_with_probe(command, *, workdir=None, interactive=False):
            result = original_docker(
                command, workdir=workdir, interactive=interactive
            )
            insertion = result.index(args.container)
            result[insertion:insertion] = [
                "-e",
                f"KERNELGEN_OVERRIDE_OPERATOR={active['operator']}",
                "-e",
                f"KERNELGEN_OVERRIDE_OUTPUT={remote_prefix}-override.json",
            ]
            return result

        runner._docker = docker_with_probe
        runner.pytest(
            accuracy,
            marker,
            ["--quick", "-p", "kernelgen_override_probe"],
            label="override-accuracy-quick",
        )
        override_accuracy = runner.read_json(
            remote_prefix + "-override.json", label="read-accuracy-override"
        )
        if not override_accuracy["call_count"] or not override_accuracy["restored"]:
            # Quick mode may select only a dtype that the test intentionally skips
            # (for example low-precision inplace erfc).  Retry the marker's full
            # correctness matrix before concluding that the direct override path
            # is unreachable.
            runner.pytest(
                accuracy,
                marker,
                ["-p", "kernelgen_override_probe"],
                label="override-accuracy-full-fallback",
            )
            override_accuracy = runner.read_json(
                remote_prefix + "-override.json",
                label="read-accuracy-override-fallback",
            )
        if not override_accuracy["call_count"] or not override_accuracy["restored"]:
            raise RuntimeError("accuracy override was not called or restored")
        runner.pytest(
            benchmark,
            marker,
            [
                "--level",
                "core",
                "--case-id",
                exact_case,
                "--profile-only",
                "--profile-warmup",
                "1",
                "--profile-iterations",
                "1",
                "-p",
                "kernelgen_override_probe",
            ],
            label="override-benchmark-exact",
        )
        override_benchmark = runner.read_json(
            remote_prefix + "-override.json", label="read-benchmark-override"
        )
        if not override_benchmark["call_count"] or not override_benchmark["restored"]:
            raise RuntimeError("benchmark override was not called or restored")

        record.update(
            {
                "status": "passed",
                "baseline_hashes": baseline_hashes,
                "converted_hashes": local_hashes,
                "accuracy": {
                    "baseline_quick": _pytest_counts(baseline_quick),
                    "converted_quick": _pytest_counts(converted_quick),
                    "baseline_full": _pytest_counts(baseline_full),
                    "converted_full": _pytest_counts(converted_full),
                },
                "benchmark": {
                    "baseline_count": len(baseline_rows),
                    "converted_count": len(converted_rows),
                    "baseline_signature": _benchmark_signature(baseline_result),
                    "converted_signature": _benchmark_signature(converted_result),
                    "case_count": len(case_ids),
                    "case_ids": case_ids,
                    "exact_case": exact_case,
                },
                "override": {
                    "accuracy": override_accuracy,
                    "benchmark": override_benchmark,
                },
            }
        )
    except KeyboardInterrupt as exc:
        record["status"] = "interrupted"
        record["error"] = f"{type(exc).__name__}: validation interrupted"
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if args.local_nvidia or args.remote_template_root:
            try:
                runner.cleanup()
            except Exception as cleanup_error:
                record["cleanup_error"] = (
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        runner.close()
        record["finished_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        _write_json(record_path, record)
        print(f"Device run record: {record_path}", flush=True)


if __name__ == "__main__":
    main()

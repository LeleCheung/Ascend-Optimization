"""Run pinned FlagGems pytest and normalize its native reports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import site
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelgen_server.evaluation.adapters.base import EvaluatorAdapter
from kernelgen_server.evaluation.candidate_admission import (
    CandidateAdmissionError, require_candidate_admission,
)
from kernelgen_server.evaluation.loader import BuildError, load_implementation
from kernelgen_server.evaluation.result import aggregate_workload_results
from kernelgen_server.profiling.models import (
    ProfileCommand,
    ProfileOptions,
    ProfileRequest,
)
from kernelgen_server.protocol.schema import (
    AdapterCapabilities,
    AdapterCase,
    AdapterManifest,
    BoundEvaluateRequest,
    CandidateContract,
    CaseList,
    EvaluateResponse,
    EvaluationStatus,
    PreflightResult,
    ReferenceRequest,
    ReferenceResult,
    WorkloadResult,
    WorkloadStatus,
)
from kernelgen_server.protocol.workload_call import definition_signature
from kernelgen_server.runtime.environment import bind_backend_device
from kernelgen_server.runtime.flaggems import flaggems_vendor

from .discovery import FlagGemsAssets, benchmark_fingerprint, discover_assets


_ABI_VALIDATION_LOCK = threading.Lock()
_NATIVE_CASE_REPORT_TIMEOUT_SECONDS = 600.0
_PYTEST_BOOTSTRAP = (
    "import runpy,site,sys;"
    "sys.path.extend(path for path in site.getsitepackages() if path not in sys.path);"
    "runpy.run_module('pytest',run_name='__main__',alter_sys=True)"
)


class _CandidateFailure(RuntimeError):
    """A failure attributable to the submitted candidate, not the Server."""


def _validate_reference_report(report, expected):
    if report.get("schema_version") != "flaggems.reference/v1" or report.get("phase") != "timing":
        raise ValueError("reference-only requires a timing flaggems.reference/v1 report")
    status = report.get("status")
    if status not in {"PASSED", "FAILED", "UNSUPPORTED", "ALL_SKIP", "NO_CASES"}:
        raise ValueError("invalid reference-only status")
    records = report.get("records")
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("invalid reference-only records")
    if status != "PASSED":
        return status
    skipped_nodes = {r.get("nodeid") for r in records if r.get("pytest_phase") and r.get("status") == "SKIP"}
    if any(r.get("status") not in {"PASSED", "SKIP"} and not (
        r.get("status") == "NOT_RUN" and r.get("nodeid") in skipped_nodes
    ) for r in records):
        raise ValueError("reference-only PASSED contradicts case results")
    seen, passed = set(), set()
    for record in records:
        case_id = record.get("case_id")
        if case_id is None:
            continue
        if not isinstance(case_id, str) or case_id not in expected or case_id in seen:
            raise ValueError("reference-only case identity differs from inspected core cases")
        seen.add(case_id)
        if record["status"] == "PASSED" and record.get("nodeid") not in skipped_nodes:
            if type(record.get("count")) is not int or record["count"] != 1:
                raise ValueError("reference-only case must execute exactly once")
            passed.add(case_id)
    covered = seen | {case for case in expected if any(
        isinstance(node, str) and case.startswith(node + "::") for node in skipped_nodes)}
    if covered != expected or not passed:
        raise ValueError("reference-only did not cover the inspected core cases")
    return status


def _pytest_marker(assets: FlagGemsAssets) -> str:
    return getattr(assets, "pytest_marker", None) or assets.native_operator


def _run_pytest(
    args: list[str],
    *,
    env: dict[str, str],
    root: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, "-s", "-c", _PYTEST_BOOTSTRAP, *args],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"FlagGems pytest timed out: {' '.join(args)}") from exc


def _load_report(
    path: Path,
    label: str,
    process: subprocess.CompletedProcess[str] | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        details = ""
        if process is not None:
            details = ((process.stdout or "") + (process.stderr or ""))[-4000:]
        raise RuntimeError(
            f"FlagGems {label} did not produce {path.name}; pytest output:\n{details}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"FlagGems {label} report must be a mapping")
    return value


def _accuracy_results(report: dict[str, Any]) -> list[WorkloadResult]:
    results: list[WorkloadResult] = []
    for nodeid, item in report.items():
        if not isinstance(item, dict):
            continue
        outcome = item.get("result")
        skipped = outcome == "skipped"
        if outcome in {"passed", "skipped"}:
            status = WorkloadStatus.PASSED
        elif outcome == "failed":
            # pytest uses "failed" for both assertion failures and exceptions
            # raised while executing the test (including compilation/import).
            # FlagGems preserves reprcrash.message in reason. Keep its exception
            # type instead of calling every failed test a numerical mismatch.
            exception = re.match(
                r"\s*(?:E\s+)?(?:\w+\.)*((?:[A-Z]\w*)?(?:Error|Exception)):",
                str(item.get("reason") or ""),
            )
            status = (
                WorkloadStatus.RUNTIME_ERROR
                if exception and exception.group(1) != "AssertionError"
                else WorkloadStatus.INCORRECT_NUMERICAL
            )
        else:
            status = WorkloadStatus.RUNTIME_ERROR
        metrics: dict[str, Any] = {"native_outcome": outcome or "unknown"}
        if skipped:
            metrics["skipped"] = True
        params = item.get("params")
        results.append(
            WorkloadResult(
                uuid=f"flaggems::{nodeid}",
                axes=params if isinstance(params, dict) else {},
                phase="correctness",
                status=status,
                metrics=metrics,
                log=str(item.get("reason") or ""),
            )
        )
    return results


def _timing_results(
    report: dict[str, Any],
    expected_case_ids: set[str],
    native_operator: str,
) -> list[WorkloadResult]:
    entry = report.get(native_operator, {})
    details = entry.get("details", []) if isinstance(entry, dict) else []
    results: list[WorkloadResult] = []
    seen_case_ids: set[str] = set()
    for detail in details:
        if not isinstance(detail, dict):
            continue
        dtype = str(detail.get("dtype", "unknown")).removeprefix("torch.")
        nodeid = detail.get("nodeid") or entry.get("test_case")
        metrics = detail.get("result", [])
        if not isinstance(metrics, list):
            continue
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            case_id = metric.get("case_id")
            if metric.get("candidate_source") != "override":
                raise RuntimeError(
                    "FlagGems timing case did not resolve the KernelGen "
                    f"gems_op override: nodeid={nodeid!r}, case_id={case_id!r}"
                )
            if not isinstance(case_id, str) or case_id not in expected_case_ids:
                raise RuntimeError(
                    f"timing report returned an unknown case_id: nodeid={nodeid!r}, "
                    f"case_id={case_id!r}"
                )
            if case_id in seen_case_ids:
                raise RuntimeError(
                    f"timing report returned duplicate case_id: {case_id}"
                )
            seen_case_ids.add(case_id)
            latency = metric.get("latency")
            reference = metric.get("latency_base")
            speedup = metric.get("speedup")
            error = metric.get("error_msg")
            valid = all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) > 0
                for value in (latency, reference, speedup)
            )
            results.append(
                WorkloadResult(
                    uuid=case_id,
                    axes={
                        "dtype": dtype,
                        "shape_detail": metric.get("shape_detail"),
                        "benchmark_level": detail.get("level"),
                    },
                    phase="timing",
                    status=(
                        WorkloadStatus.PASSED
                        if error is None and valid
                        else WorkloadStatus.RUNTIME_ERROR
                    ),
                    speedup=float(speedup) if valid else None,
                    latency_ms=float(latency) if valid else None,
                    reference_latency_ms=float(reference) if valid else None,
                    log=str(error or ""),
                )
            )
    missing = sorted(expected_case_ids - seen_case_ids)
    if missing:
        raise RuntimeError(
            "timing report omitted case_id values: " + ", ".join(missing[:20])
        )
    return results


def _parse_case_list(
    report: dict[str, Any],
    native_operator: str,
    level: str,
) -> list[AdapterCase]:
    """Extract and validate AdapterCase list from a --list-cases report."""
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise RuntimeError("FlagGems case list contains no benchmark entries")
    schema_version = report["schema_version"]
    cases: list[AdapterCase] = []
    seen_case_ids: set[str] = set()
    global_ordinal = 0
    for benchmark in benchmarks:
        if not isinstance(benchmark, dict):
            raise RuntimeError("FlagGems benchmark case entry must be a mapping")
        if benchmark.get("schema_version") != schema_version:
            raise RuntimeError("FlagGems benchmark case schema version mismatch")
        if benchmark.get("op_name") != native_operator:
            raise RuntimeError("FlagGems case list operator mismatch")
        if benchmark.get("level") != level:
            raise RuntimeError("FlagGems case list benchmark level mismatch")
        native_cases = benchmark.get("cases")
        if not isinstance(native_cases, list):
            raise RuntimeError("FlagGems case list is missing cases")
        for native in native_cases:
            if not isinstance(native, dict) or not isinstance(
                native.get("case_id"), str
            ):
                raise RuntimeError("FlagGems native case is missing case_id")
            case_id = native["case_id"]
            if case_id in seen_case_ids:
                raise RuntimeError(
                    f"FlagGems case list contains duplicate case_id: {case_id}"
                )
            seen_case_ids.add(case_id)
            # FlagGems v2 owns this opaque identity and accepts it back via
            # --case-id.  Do not synthesize or parse a second Server ID.
            cases.append(
                AdapterCase(
                    case_id=case_id,
                    ordinal=global_ordinal,
                    dtype=str(native.get("dtype")) if native.get("dtype") else None,
                    shape=native.get("shape"),
                    params=native.get("params") or {},
                )
            )
            global_ordinal += 1
    return cases


@dataclass
class _WorkContext:
    """Shared setup computed once at the start of preflight/evaluate."""
    manifest: AdapterManifest
    assets: FlagGemsAssets
    hack: Any
    work: Path
    candidate_root: Path
    coverage_env: dict[str, str]

    @property
    def performance_paths(self) -> list[str]:
        return [
            path.relative_to(self.assets.root).as_posix()
            for path in self.assets.performance
        ]

    @property
    def correctness_paths(self) -> list[str]:
        return [
            path.relative_to(self.assets.root).as_posix()
            for path in self.assets.correctness
        ]


class FlagGemsEvaluationAdapter(EvaluatorAdapter):
    kind = "flaggems"

    @property
    def _operator_name(self) -> str:
        return self.operator.definition.name

    @property
    def _level(self) -> str:
        return str(self.catalog.manifest["benchmark_level"])

    def _assets(self) -> FlagGemsAssets:
        source = self.catalog.manifest.get("definition_source")
        if source is None:
            return discover_assets(self._operator_name)
        from kernelgen_server.operator_bundles import GemsDefinitionSource
        contract = GemsDefinitionSource.model_validate(source)
        assets = discover_assets(self._operator_name, contract.source_revision)
        required = {path.relative_to(assets.root).as_posix() for path in (*assets.correctness, *assets.performance)}
        if required - contract.source_files.keys():
            raise ValueError("uploaded Definition does not cover the target's original pytest suites")
        for relative, digest in contract.source_files.items():
            path = (assets.root / relative).resolve()
            if not path.is_relative_to(assets.root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"uploaded Definition source differs from target checkout: {relative}")
        return assets

    def _base_env(self, root: Path) -> dict[str, str]:
        env = dict(os.environ)
        if self.backend:
            env["GEMS_VENDOR"] = flaggems_vendor(self.backend)
        if self.backend and self.device_string:
            bind_backend_device(env, self.backend, self.device_string)
        site_paths = {
            str(Path(path).resolve()) for path in site.getsitepackages()
        }
        inherited = [
            path
            for path in env.get("PYTHONPATH", "").split(os.pathsep)
            if path and str(Path(path).resolve()) not in site_paths
        ]
        python_paths = [
            str(root / "src"),
            str(Path(__file__).resolve().parents[4]),
            *inherited,
        ]
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        return env

    def _native_case_report(
        self,
        assets: FlagGemsAssets,
        *,
        timeout: float = _NATIVE_CASE_REPORT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        performance = [
            path.relative_to(assets.root).as_posix()
            for path in assets.performance
        ]
        with tempfile.TemporaryDirectory(prefix="kernelgen-flaggems-inspect-") as tmp:
            output = Path(tmp) / "cases.json"
            process = _run_pytest(
                [
                    "-q",
                    *performance,
                    "-m",
                    _pytest_marker(assets),
                    "--level",
                    self._level,
                    "--list-cases",
                    "--output",
                    str(output),
                ],
                env=self._base_env(assets.root),
                root=assets.root,
                timeout=timeout,
            )
            report = _load_report(output, "case list", process)
            if process.returncode != 0:
                raise RuntimeError(
                    "FlagGems case listing failed:\n"
                    + ((process.stdout or "") + (process.stderr or ""))[-4000:]
                )
            return report

    def _manifest(self) -> AdapterManifest:
        assets = self._assets()
        report = self._native_case_report(assets)
        if report.get("schema_version") != "flaggems.benchmark-case-list/v2":
            raise RuntimeError("FlagGems case list must use schema v2")
        fingerprint = benchmark_fingerprint(
            assets,
            operator=self._operator_name,
            benchmark_level=self._level,
            native_case_report=report,
        )
        cases = _parse_case_list(report, assets.native_operator, self._level)
        case_list = CaseList(
            adapter_kind="flaggems",
            operator=self._operator_name,
            benchmark_fingerprint=fingerprint,
            cases=cases,
        )
        return AdapterManifest(
            kind="flaggems",
            benchmark_fingerprint=fingerprint,
            candidate_contract=CandidateContract(
                signature=str(definition_signature(self.operator.definition))
            ),
            capabilities=AdapterCapabilities(preflight=True, profile=True),
            case_list=case_list,
        )

    def inspect(self) -> AdapterManifest:
        return self._manifest()

    @staticmethod
    def _materialize_candidate(request: BoundEvaluateRequest, root: Path) -> None:
        for source in request.implementation.sources:
            path = root / source.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source.content, encoding="utf-8")

    def _candidate_env(
        self,
        assets: FlagGemsAssets,
        candidate_root: Path,
    ) -> dict[str, str]:
        env = self._base_env(assets.root)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(candidate_root), env["PYTHONPATH"]]
        )
        return env

    def _override_args(
        self,
        request: BoundEvaluateRequest,
        candidate_root: Path,
    ) -> list[str]:
        """Use one FlagGems dynamic override for every execution mode."""
        entry_path, separator, function = request.implementation.entrypoint.partition("::")
        if not separator or not function:
            raise ValueError(
                "FlagGems candidates require an entrypoint of the form path::function"
            )
        candidate_path = (candidate_root / entry_path).resolve()
        candidate_path.relative_to(candidate_root.resolve())
        return [
            "--override",
            f"{self._operator_name}:{candidate_path}:{function}",
        ]

    @staticmethod
    def _read_preflight_coverage(
        path: Path, process: subprocess.CompletedProcess[str], expected_operator: str,
    ) -> list[dict[str, Any]]:
        try:
            report = _load_report(path, "preflight", process)
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"invalid FlagGems preflight report: {exc}") from exc
        if report.get("schema_version") != "flaggems.preflight/v1":
            raise RuntimeError("FlagGems preflight report must use schema flaggems.preflight/v1")
        records = report.get("records")
        if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            raise RuntimeError("FlagGems preflight records are malformed")
        for record in records:
            if record.get("operator") != expected_operator:
                raise _CandidateFailure("FlagGems candidate resolved the wrong public operator")
            if record.get("status") != "passed" or record.get("override") is not True:
                raise RuntimeError("FlagGems preflight did not complete the injected candidate: " + str(record))
        return records

    def _validate_candidate_abi(
        self,
        request: BoundEvaluateRequest,
        assets: FlagGemsAssets,
    ) -> None:
        source = str(assets.root / "src")
        # Profile commands are built in Server threads, so guard process-global
        # sys.path changes made both here and by load_implementation().
        with _ABI_VALIDATION_LOCK:
            sys.path.insert(0, source)
            try:
                load_implementation(request.implementation, self.operator.definition)
            finally:
                sys.path.remove(source)

    @staticmethod
    def _assert_accuracy_call_coverage(
        report: dict[str, Any], operator: str
    ) -> None:
        expected = {
            nodeid
            for nodeid, item in report.items()
            if isinstance(item, dict) and item.get("result") != "skipped"
        }
        if not report:
            raise RuntimeError("FlagGems correctness report contains no cases")
        observed = set()
        for nodeid, item in report.items():
            calls = item.get("candidate_calls", {}) if isinstance(item, dict) else {}
            count = calls.get(f"flag_gems.{operator}") if isinstance(calls, dict) else None
            if type(count) is int and count > 0:
                observed.add(nodeid)
        missing = sorted(expected - observed)
        if missing:
            raise _CandidateFailure(
                "FlagGems correctness cases bypassed the candidate gems_op: "
                + ", ".join(missing[:20])
            )

    @staticmethod
    def _assert_timing_call_coverage(
        expected_case_ids: set[str], calls: list[dict[str, Any]]
    ) -> None:
        observed = {
            item.get("case_id"): item.get("count")
            for item in calls
            if isinstance(item.get("case_id"), str)
        }
        observed_case_ids = set(observed)
        missing = sorted(expected_case_ids - observed_case_ids)
        unexpected = sorted(observed_case_ids - expected_case_ids)
        repeated = sorted(
            case_id
            for case_id in expected_case_ids & observed_case_ids
            if type(observed[case_id]) is not int or observed[case_id] != 1
        )
        duplicate_records = len(calls) != len(observed)
        if missing or unexpected or repeated or duplicate_records:
            raise RuntimeError(
                "FlagGems timing candidate coverage mismatch: "
                f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
                f"not_called_once={repeated[:20]}, duplicate_or_invalid_records={duplicate_records}"
            )

    def _prepare_workdir(
        self,
        request: BoundEvaluateRequest,
        work: Path,
    ) -> tuple[AdapterManifest, FlagGemsAssets, Path, dict[str, str]]:
        """Validate candidate, compute manifest, write candidate files, build env.

        Returns (manifest, assets, candidate_root, subprocess_env).
        Preflight owns admission before calling this shared preparation helper.
        """
        # FlagGems owns execution through fresh pytest subprocesses.  Do not
        # initialize the accelerator in this isolated parent first: MetaX's
        # runtime can crash a later pytest child during ``import torch`` when
        # the parent already established a CUDA/MACA context.  _base_env()
        # narrows every subprocess to the scheduler-assigned device.
        assets = self._assets()
        self._validate_candidate_abi(request, assets)
        manifest = self._manifest()
        candidate_root = work / "candidate"
        candidate_root.mkdir(parents=True, exist_ok=True)
        self._materialize_candidate(request, candidate_root)
        env = self._candidate_env(assets, candidate_root)
        return manifest, assets, candidate_root, env

    def _candidate_evaluation_failure(
        self,
        request: BoundEvaluateRequest,
        *,
        stage: str,
        error: Exception,
    ) -> EvaluateResponse:
        return EvaluateResponse(
            status="RUNTIME_ERROR",
            device=self.device_string,
            server_backend=self.backend,
            num_workloads=0,
            num_passed=0,
            log=f"candidate {stage} failed: {error}",
        )

    def _candidate_preflight_failure(
        self,
        request: BoundEvaluateRequest,
        *,
        stage: str,
        error: Exception,
        manifest: AdapterManifest | None = None,
    ) -> PreflightResult:
        resolved_manifest = manifest or self._manifest()
        return PreflightResult(
            status="FAILED",
            stage=stage,
            log=f"candidate {stage} failed: {error}",
            benchmark_fingerprint=resolved_manifest.benchmark_fingerprint,
            num_cases=len(resolved_manifest.case_list.cases),
        )

    def reference(self, request: ReferenceRequest) -> ReferenceResult:
        """Core benchmark baseline readiness; no candidate or accuracy pytest."""
        if self.device is None:
            raise RuntimeError("FlagGems reference execution requires a device slot")
        if self._level != "core":
            return ReferenceResult(status="UNSUPPORTED", log="reference readiness requires a core benchmark binding")
        with tempfile.TemporaryDirectory(prefix="kernelgen-flaggems-reference-") as tmp:
            path = Path(tmp) / "reference.json"
            report = {}
            try:
                manifest = self.inspect()
                if manifest.benchmark_fingerprint != request.benchmark_fingerprint:
                    return ReferenceResult(status="FAILED", log="benchmark fingerprint changed before reference validation")
                assets = self._assets()
                process = _run_pytest(
                    ["-q", *[p.relative_to(assets.root).as_posix() for p in assets.performance],
                     "-m", _pytest_marker(assets), "--reference-only", "--output", str(path)],
                    env=self._base_env(assets.root), root=assets.root, timeout=request.settings.timeout_seconds,
                )
                log = ((process.stdout or "") + (process.stderr or ""))[-8000:]
                if process.returncode == 4 and not path.exists():
                    return ReferenceResult(status="UNSUPPORTED", log=log)
                report = _load_report(path, "reference", process)
                status = _validate_reference_report(report, {case.case_id for case in manifest.case_list.cases})
                if process.returncode != 0 and status == "PASSED":
                    raise ValueError("reference-only exited unsuccessfully despite a PASSED report")
                if self._assets().revision != assets.revision:
                    raise ValueError("Gems source changed during reference validation")
                return ReferenceResult(status=status, benchmark_fingerprint=manifest.benchmark_fingerprint,
                                       report=report, log=log)
            except (ValueError, RuntimeError) as exc:
                return ReferenceResult(status="RUNTIME_ERROR", report=report, log=str(exc))

    def preflight(self, request: BoundEvaluateRequest) -> PreflightResult:
        if self.device is None:
            raise RuntimeError("FlagGems preflight requires a device")
        with tempfile.TemporaryDirectory(prefix="kernelgen-flaggems-preflight-") as tmp:
            work = Path(tmp)
            report_path = work / "preflight.json"
            try:
                hack = require_candidate_admission(
                    request.implementation, operator_name=self._operator_name, evaluator_kind=self.kind
                )
                manifest, assets, candidate_root, env = self._prepare_workdir(
                    request, work
                )
            except CandidateAdmissionError as exc:
                return PreflightResult(
                    status="FAILED", stage="candidate_admission", is_hack=True,
                    hack_reason=str(exc), log="candidate_admission: " + str(exc),
                )
            except BuildError as exc:
                return self._candidate_preflight_failure(
                    request, stage="candidate_import", error=exc
                )
            process = _run_pytest(
                [
                    "-q",
                    *[path.relative_to(assets.root).as_posix() for path in assets.performance],
                    "-m",
                    _pytest_marker(assets),
                    "--level",
                    self._level,
                    "--preflight-only",
                    "--record", "json",
                    "--output", str(report_path),
                    *self._override_args(request, candidate_root),
                ],
                env=env,
                root=assets.root,
                timeout=request.settings.timeout_seconds,
            )
            expected = len(manifest.case_list.cases)
            try:
                self._assert_timing_call_coverage(
                    {case.case_id for case in manifest.case_list.cases},
                    self._read_preflight_coverage(
                        report_path, process, assets.native_operator
                    ),
                )
                coverage_error = ""
            except _CandidateFailure as exc:
                return self._candidate_preflight_failure(
                    request,
                    stage="candidate_contract",
                    error=exc,
                    manifest=manifest,
                )
            except RuntimeError as exc:
                coverage_error = str(exc)
            if process.returncode != 0 or coverage_error:
                details = ((process.stdout or "") + (process.stderr or ""))[-8000:]
                return PreflightResult(
                    status="RUNTIME_ERROR",
                    stage="timing_candidate_smoke",
                    log=f"{coverage_error}\n{details}",
                    is_hack=hack.is_hack,
                    hack_reason=hack.hack_reason,
                    benchmark_fingerprint=manifest.benchmark_fingerprint,
                    num_cases=expected,
                )
            return PreflightResult(
                status="PASSED",
                stage="complete",
                log=hack.log,
                is_hack=hack.is_hack,
                hack_reason=hack.hack_reason,
                benchmark_fingerprint=manifest.benchmark_fingerprint,
                num_cases=expected,
            )

    def evaluate(self, request: BoundEvaluateRequest) -> EvaluateResponse:
        if self.device is None:
            raise RuntimeError("FlagGems evaluation requires a device")
        deadline = time.monotonic() + request.settings.timeout_seconds

        def remaining_timeout() -> float:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("FlagGems evaluation exhausted its timeout")
            return remaining

        with tempfile.TemporaryDirectory(prefix="kernelgen-flaggems-evaluate-") as tmp:
            work = Path(tmp)
            try:
                manifest, assets, candidate_root, env = self._prepare_workdir(
                    request, work
                )
            except BuildError as exc:
                return self._candidate_evaluation_failure(
                    request, stage="import", error=exc
                )
            accuracy_report = work / "accuracy.json"
            benchmark_report = work / "benchmark.json"

            accuracy = _run_pytest(
                [
                    "-q",
                    *[path.relative_to(assets.root).as_posix() for path in assets.correctness],
                    "-m",
                    _pytest_marker(assets),
                    "--record",
                    "json",
                    "--output",
                    str(accuracy_report),
                    *self._override_args(request, candidate_root),
                ],
                env=env,
                root=assets.root,
                timeout=remaining_timeout(),
            )
            accuracy_data = _load_report(accuracy_report, "accuracy", accuracy)
            all_skipped = bool(accuracy_data) and all(
                isinstance(item, dict) and item.get("result") == "skipped"
                for item in accuracy_data.values()
            )
            if not all_skipped:
                try:
                    self._assert_accuracy_call_coverage(accuracy_data, assets.native_operator)
                except _CandidateFailure as exc:
                    return self._candidate_evaluation_failure(
                        request, stage="contract", error=exc
                    )

            performance_env = self._candidate_env(assets, candidate_root)
            performance = _run_pytest(
                [
                    "-q",
                    *[path.relative_to(assets.root).as_posix() for path in assets.performance],
                    "-m",
                    _pytest_marker(assets),
                    "--level",
                    self._level,
                    "--warmup",
                    str(request.settings.warmup_ms),
                    "--iter",
                    str(request.settings.benchmark_ms),
                    "--record",
                    "json",
                    "--output",
                    str(benchmark_report),
                    *self._override_args(request, candidate_root),
                ],
                env=performance_env,
                root=assets.root,
                timeout=remaining_timeout(),
            )
            benchmark_data = _load_report(benchmark_report, "benchmark", performance)
            results = _accuracy_results(accuracy_data)
            try:
                timing = _timing_results(
                    benchmark_data,
                    {case.case_id for case in manifest.case_list.cases},
                    assets.native_operator,
                )
            except RuntimeError as exc:
                if performance.returncode == 0:
                    raise
                # A failed candidate can abort pytest before it emits every
                # timing case. Do not turn its execution error into HTTP 500.
                detail = ((performance.stdout or "") + (performance.stderr or ""))[-8000:]
                return self._candidate_evaluation_failure(
                    request, stage="benchmark", error=RuntimeError(f"{exc}\n{detail}")
                )
            if len(timing) != len(manifest.case_list.cases):
                raise RuntimeError(
                    "FlagGems timing case count changed between inspect and evaluate: "
                    f"{len(manifest.case_list.cases)} != {len(timing)}"
                )
            results.extend(timing)
            response = aggregate_workload_results(
                results,
                device=self.device_string,
                backend=self.backend,
            )
            if (accuracy.returncode != 0 or performance.returncode != 0) and (
                response.status == "PASSED"
            ):
                raise RuntimeError(
                    "FlagGems pytest exited non-zero without a failing report: "
                    + ((accuracy.stderr or "") + (performance.stderr or ""))[-4000:]
                )
            if response.status == "PASSED" and all_skipped:
                response.status = EvaluationStatus.ALL_SKIP
            return response

    def build_profile_command(
        self,
        request: ProfileRequest,
        options: ProfileOptions,
        artifact_dir: Path,
        device: str,
    ) -> ProfileCommand:
        del device  # required by the base class interface; this adapter uses self.backend
        if request.expected_backend != self.backend:
            raise ValueError(
                f"profile expects {request.expected_backend!r}, adapter uses {self.backend!r}"
            )
        manifest = self._manifest()
        if request.benchmark_fingerprint != manifest.benchmark_fingerprint:
            raise ValueError("benchmark fingerprint changed after inspect")
        if request.case_id not in {
            case.case_id for case in manifest.case_list.cases
        }:
            raise ValueError(
                f"unknown FlagGems timing case_id: {request.case_id}"
            )

        target_dir = artifact_dir / "target"
        candidate_root = target_dir / "candidate"
        candidate_root.mkdir(parents=True, exist_ok=False)
        bound_request = BoundEvaluateRequest(
            binding=request.binding,
            implementation=request.implementation,
        )
        assets = self._assets()
        self._validate_candidate_abi(bound_request, assets)
        self._materialize_candidate(bound_request, candidate_root)
        env = self._candidate_env(assets, candidate_root)
        completion_marker = target_dir / "runner-completed"
        return ProfileCommand(
            argv=(
                sys.executable,
                "-m",
                "kernelgen_server.profiling.gems_runner",
                "--backend", self.backend,
                "--case-id", request.case_id,
                "--completion-marker", str(completion_marker),
                "--",
                "-q",
                *[path.relative_to(assets.root).as_posix() for path in assets.performance],
                "-m",
                _pytest_marker(assets),
                "--level",
                self._level,
                "--profile-only",
                "--case-id",
                request.case_id,
                "--profile-warmup",
                str(options.warmup),
                "--profile-iterations",
                str(options.iterations),
                *self._override_args(
                    bound_request,
                    candidate_root,
                ),
            ),
            cwd=str(assets.root),
            env=env,
            source_roots=(str(candidate_root),),
            completion_marker_path=str(completion_marker),
        )


__all__ = [
    "FlagGemsEvaluationAdapter",
    "_accuracy_results",
]

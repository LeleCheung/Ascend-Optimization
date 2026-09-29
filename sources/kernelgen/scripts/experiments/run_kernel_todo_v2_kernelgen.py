#!/usr/bin/env python3
"""Select and run the Native KernelGen stage for one Kernel Todo V2 chip."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from kernelgen.data.ledger import Ledger
from kernelgen.tools.kernel_todo_v2_pipeline import (
    PipelineStatus,
    StageName,
    StageResult,
    StageVerdict,
    load_pipeline_results,
    refresh_markdown_summary,
    write_pipeline_results,
    write_second_stage_markdown,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CAMPAIGN_SCRIPT = REPO_ROOT / "examples/kernel_gen/run_campaign.py"
MANIFEST_SCHEMA = "kernelgen.kernel-todo-v2-kernelgen-campaign/v1"
ROW_OPERATOR = re.compile(r"`([^`]+)`")
SMOKE_PREFERENCES = ("gelu", "relu", "add", "mul")


@dataclass(frozen=True)
class SelectedOperator:
    source_operator: str
    definition_name: str
    profile_evidence: pathlib.Path
    seed_code_path: pathlib.Path
    native_operator_root: pathlib.Path


@dataclass(frozen=True)
class CollectedResult:
    definition_name: str
    complete: bool
    geo_mean: float | None
    best_code: str
    evidence: list[pathlib.Path]
    reason: str = ""


class LaunchError(RuntimeError):
    pass


class JsonClient:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(self, path: str, *, timeout: float) -> dict[str, Any]:
        request = urllib.request.Request(self.server_url + path, method="GET")
        try:
            with self.opener.open(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LaunchError(f"{path} HTTP {exc.code}: {detail}") from exc
        except (OSError, ValueError) as exc:
            raise LaunchError(f"{path} transport/JSON error: {exc}") from exc
        if not isinstance(value, dict):
            raise LaunchError(f"{path} response root is not an object")
        return value


def _atomic_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_mapping(values: list[str], *, label: str) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for value in values:
        operator, separator, raw_path = value.partition("=")
        if not separator or not operator.strip() or not raw_path.strip():
            raise LaunchError(f"{label} must use OPERATOR=PATH: {value!r}")
        operator = operator.strip()
        if operator in result:
            raise LaunchError(f"duplicate {label} for {operator!r}")
        path = pathlib.Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise LaunchError(f"{label} is missing for {operator}: {path}")
        result[operator] = path
    return result


def _scheduler_gate(status: dict[str, Any], *, expected_backend: str) -> None:
    if status.get("api_version") != "v6.2":
        raise LaunchError(
            f"server api_version={status.get('api_version')!r}, expected 'v6.2'"
        )
    if expected_backend and status.get("backend") != expected_backend:
        raise LaunchError(
            f"server backend={status.get('backend')!r}, "
            f"expected {expected_backend!r}"
        )
    scheduler = status.get("scheduler")
    if not isinstance(scheduler, dict):
        raise LaunchError("/status has no scheduler")
    slots = scheduler.get("device_slots")
    if (
        not isinstance(slots, int)
        or slots < 1
        or scheduler.get("healthy") != slots
        or scheduler.get("available") != slots
        or scheduler.get("active") != 0
        or scheduler.get("waiting") != 0
        or scheduler.get("checking") != 0
        or scheduler.get("broken") != 0
    ):
        raise LaunchError(f"scheduler is not idle and healthy: {scheduler}")


def _select_operators(
    *,
    pipeline_path: pathlib.Path,
    chip: str,
    requested: set[str],
    profile_evidence: dict[str, pathlib.Path],
    seed_code_paths: dict[str, pathlib.Path],
    native_catalog_root: pathlib.Path,
) -> tuple[Any, list[SelectedOperator]]:
    if not pipeline_path.is_file():
        raise LaunchError(f"pipeline results are missing: {pipeline_path}")
    manifest_path = native_catalog_root / "manifest.json"
    try:
        native_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LaunchError(f"cannot read Native catalog manifest: {exc}") from exc
    if not isinstance(native_manifest, dict) or (
        native_manifest.get("api_version"), native_manifest.get("evaluator")
    ) != ("v6.2", "native"):
        raise LaunchError(f"not a Native v6.2 catalog: {native_catalog_root}")
    pipeline = load_pipeline_results(pipeline_path, chip=chip)
    selected: list[SelectedOperator] = []
    ineligible: dict[str, str] = {}
    for source_operator, record in sorted(pipeline.operators.items()):
        if requested and source_operator not in requested:
            continue
        simple = record.simple_opt.current
        native = record.native_baseline.current if record.native_baseline else None
        reason = ""
        if simple.verdict != StageVerdict.PASSED or simple.geo_mean is None:
            reason = "SimpleOpt did not produce a passed timing result"
        elif simple.geo_mean >= pipeline.threshold:
            reason = "SimpleOpt already meets the first-stage threshold"
        elif native is None or native.verdict != StageVerdict.PASSED:
            reason = "Native baseline has not passed"
        elif record.kernelgen_native is not None:
            reason = "KernelGen Native already has a recorded attempt"
        elif record.final_gems is not None:
            reason = "final Gems already has a recorded attempt"
        if reason:
            if source_operator in requested:
                ineligible[source_operator] = reason
            continue
        profile = profile_evidence.get(source_operator)
        if profile is None:
            ineligible[source_operator] = "explicit profiling evidence is missing"
            continue
        seed_code = seed_code_paths.get(source_operator)
        if seed_code is None:
            ineligible[source_operator] = "explicit seed code is missing"
            continue
        operator_root = native_catalog_root / "ops" / record.definition_name
        required = {
            "definition.json",
            "oracle.py",
            "correctness.jsonl",
            "timing.jsonl",
        }
        actual = (
            {
                path.name
                for path in operator_root.iterdir()
                if path.is_file()
            }
            if operator_root.is_dir()
            else set()
        )
        missing = sorted(required - actual)
        if missing:
            ineligible[source_operator] = (
                f"Native v6.2 package is incomplete: missing={missing}"
            )
            continue
        selected.append(
            SelectedOperator(
                source_operator=source_operator,
                definition_name=record.definition_name,
                profile_evidence=profile,
                seed_code_path=seed_code,
                native_operator_root=operator_root,
            )
        )
    unknown = requested - set(pipeline.operators)
    if unknown:
        raise LaunchError(
            f"requested operators are absent from pipeline: {sorted(unknown)}"
        )
    if ineligible:
        raise LaunchError(
            "operator gates failed: "
            + "; ".join(
                f"{operator}: {reason}"
                for operator, reason in sorted(ineligible.items())
            )
        )
    if not selected:
        raise LaunchError("no operator is eligible for Native KernelGen")
    unused_evidence = set(profile_evidence) - {
        item.source_operator for item in selected
    }
    if unused_evidence:
        raise LaunchError(
            f"profiling evidence was supplied for unselected operators: "
            f"{sorted(unused_evidence)}"
        )
    unused_seeds = set(seed_code_paths) - {
        item.source_operator for item in selected
    }
    if unused_seeds:
        raise LaunchError(
            "seed code was supplied for unselected operators: "
            f"{sorted(unused_seeds)}"
        )
    return pipeline, selected


def _smoke_operator(
    selected: list[SelectedOperator], requested: str | None
) -> SelectedOperator:
    if requested:
        match = next(
            (item for item in selected if item.source_operator == requested),
            None,
        )
        if match is None:
            raise LaunchError(
                f"smoke operator is not in the selected set: {requested}"
            )
        return match
    for preferred in SMOKE_PREFERENCES:
        match = next(
            (
                item
                for item in selected
                if preferred in {item.source_operator, item.definition_name}
            ),
            None,
        )
        if match is not None:
            return match
    return selected[0]


def _campaign_command(
    args: argparse.Namespace,
    *,
    definitions: list[str],
    seed_code_paths: dict[str, pathlib.Path],
    workspace: pathlib.Path,
    smoke: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(CAMPAIGN_SCRIPT),
        "--definitions",
        *definitions,
        "--workspace-root",
        str(workspace),
        "--knowledge-catalog-path",
        str(args.knowledge_catalog_path.resolve()),
        "--knowledge-mode",
        "read_only_v1" if smoke else args.knowledge_mode,
        "--catalog-name",
        args.catalog_name,
        "--eval-server",
        args.server_url,
        "--target-hardware",
        args.target_hardware,
        "--runtime",
        args.runtime,
        "--max-operators",
        "1" if smoke else str(args.max_operators),
        "--agents-per-operator",
        "1" if smoke else str(args.agents_per_operator),
        "--start-epoch",
        "1",
        "--n-epoch",
        "1" if smoke else str(args.epochs),
        "--start-mode",
        "fresh",
        "--early-stop-rounds",
        "0" if smoke else str(args.early_stop_rounds),
        "--min-rounds",
        "1" if smoke else str(args.min_rounds),
        "--max-round",
        "1" if smoke else str(args.max_round),
        "--timeout",
        str(args.coder_timeout),
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.no_reduced_precision:
        command.append("--no-reduced-precision")
    for definition in definitions:
        command.extend(
            [
                "--seed-code-path",
                f"{definition}={seed_code_paths[definition]}",
            ]
        )
    return command


def _run_command(command: list[str], log_path: pathlib.Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def _collect_result(
    workspace: pathlib.Path,
    definition_name: str,
    final_epoch: int,
) -> CollectedResult:
    operator_root = workspace / definition_name
    manifest = operator_root / f"{final_epoch}R" / "epoch-completion.json"
    if not manifest.is_file():
        return CollectedResult(
            definition_name=definition_name,
            complete=False,
            geo_mean=None,
            best_code="",
            evidence=[],
            reason=f"final epoch completion manifest is missing: {manifest}",
        )
    ledgers = sorted(operator_root.glob("*R/agent*/.ledger.json"))
    candidates: list[tuple[float, str, pathlib.Path]] = []
    for ledger_path in ledgers:
        ledger = Ledger(ledger_path.parent)
        geo_mean = ledger.history.best_geo_mean
        code = ledger.best.get("code", "") or ledger.history.best_code
        if geo_mean > 0 and code:
            candidates.append((geo_mean, code, ledger_path))
    if not candidates:
        return CollectedResult(
            definition_name=definition_name,
            complete=True,
            geo_mean=None,
            best_code="",
            evidence=[manifest],
            reason="completed KernelGen campaign has no correct timed candidate",
        )
    geo_mean, code, ledger_path = max(candidates, key=lambda item: item[0])
    return CollectedResult(
        definition_name=definition_name,
        complete=True,
        geo_mean=geo_mean,
        best_code=code,
        evidence=[manifest, ledger_path],
    )


def _relative(from_dir: pathlib.Path, target: pathlib.Path) -> str:
    return pathlib.Path(
        os.path.relpath(target.resolve(), from_dir.resolve())
    ).as_posix()


def _validate_results_rows(path: pathlib.Path, sources: set[str]) -> None:
    if not path.is_file():
        raise LaunchError(f"results.md is missing: {path}")
    present = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        match = ROW_OPERATOR.search(line)
        if match:
            present.add(match.group(1))
    missing = sources - present
    if missing:
        raise LaunchError(f"results rows are missing: {sorted(missing)}")


def _sync_results_rows(
    *,
    results_path: pathlib.Path,
    pipeline: Any,
    source_results: dict[str, tuple[StageResult, pathlib.Path | None]],
) -> None:
    lines = results_path.read_text(encoding="utf-8").splitlines()
    updated: set[str] = set()
    for index, line in enumerate(lines):
        if not line.startswith("| `"):
            continue
        columns = [item.strip() for item in line.strip().strip("|").split("|")]
        match = ROW_OPERATOR.search(columns[0])
        if not match or match.group(1) not in source_results or len(columns) < 9:
            continue
        source = match.group(1)
        result, code_path = source_results[source]
        status = pipeline.operators[source].status(pipeline.threshold)
        columns[3] = {
            PipelineStatus.SECOND_STAGE_RUNNING: "二阶段处理中",
            PipelineStatus.FAILED: "失败",
            PipelineStatus.BLOCKED: "阻塞",
        }[status]
        columns[6] = result.reason or "—"
        columns[7] = (
            "KernelGen Native 已完成；迁回 FlagGems 做最终正确性和性能验收。"
            if status == PipelineStatus.SECOND_STAGE_RUNNING
            else (
                "排除环境或设备阻塞后，使用新的 attempt_id 复用原 workspace 续跑。"
                if status == PipelineStatus.BLOCKED
                else "根据 KernelGen ledger 分析未获得正确候选的原因。"
            )
        )
        if code_path is not None:
            columns[8] = (
                f"[Native best code]({_relative(results_path.parent, code_path)})"
            )
        lines[index] = "| " + " | ".join(columns) + " |"
        updated.add(source)
    missing = set(source_results) - updated
    if missing:
        raise LaunchError(f"results rows are missing: {sorted(missing)}")
    refresh_markdown_summary(lines, pipeline)
    _atomic_text(results_path, "\n".join(lines) + "\n")


def run(
    args: argparse.Namespace,
    *,
    command_runner: Callable[[list[str], pathlib.Path], int] = _run_command,
    client_factory: Callable[[str], Any] = JsonClient,
    result_collector: Callable[[pathlib.Path, str, int], CollectedResult] = (
        _collect_result
    ),
) -> tuple[int, dict[str, Any]]:
    results_path = args.results_md.resolve()
    pipeline_path = args.pipeline_json.resolve()
    kernelgen_results_path = args.kernelgen_results_md.resolve()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = run_root / "campaign_manifest.json"
    if existing_manifest_path.is_file():
        try:
            existing_manifest = json.loads(
                existing_manifest_path.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            raise LaunchError(
                f"existing campaign manifest is invalid: {existing_manifest_path}"
            ) from exc
        if not isinstance(existing_manifest, dict) or not existing_manifest.get(
            "dry_run"
        ):
            raise LaunchError(
                "run root already contains a started campaign; continuation must "
                "use an explicit resume workflow"
            )
    evidence = _parse_mapping(args.profile_evidence, label="profile evidence")
    seed_code = _parse_mapping(
        args.seed_code_path,
        label="seed code",
    )
    pipeline, selected = _select_operators(
        pipeline_path=pipeline_path,
        chip=args.chip,
        requested=set(args.operator),
        profile_evidence=evidence,
        seed_code_paths=seed_code,
        native_catalog_root=args.native_catalog_root.resolve(),
    )
    seed_code_paths = {
        item.definition_name: item.seed_code_path
        for item in selected
    }
    smoke = _smoke_operator(selected, args.smoke_operator)
    _validate_results_rows(
        results_path,
        {item.source_operator for item in selected},
    )
    smoke_workspace = run_root / "smoke"
    campaign_workspace = run_root / "campaign"
    smoke_command = _campaign_command(
        args,
        definitions=[smoke.definition_name],
        seed_code_paths=seed_code_paths,
        workspace=smoke_workspace,
        smoke=True,
    )
    campaign_command = _campaign_command(
        args,
        definitions=[item.definition_name for item in selected],
        seed_code_paths=seed_code_paths,
        workspace=campaign_workspace,
        smoke=False,
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "attempt_id": args.attempt_id,
        "chip": args.chip,
        "server_url": args.server_url.rstrip("/"),
        "expected_backend": args.expected_backend,
        "catalog_name": args.catalog_name,
        "native_catalog_root": str(args.native_catalog_root.resolve()),
        "pipeline_json": str(pipeline_path),
        "results_md": str(results_path),
        "smoke_operator": smoke.source_operator,
        "operators": [
            {
                "source_operator": item.source_operator,
                "definition_name": item.definition_name,
                "profile_evidence": str(item.profile_evidence),
                "profile_evidence_sha256": _sha256(item.profile_evidence),
                "seed_code_path": str(item.seed_code_path),
                "seed_code_sha256": _sha256(item.seed_code_path),
                "native_operator_root": str(item.native_operator_root),
            }
            for item in selected
        ],
        "smoke_command": smoke_command,
        "campaign_command": campaign_command,
        "dry_run": args.dry_run,
    }
    _atomic_json(run_root / "campaign_manifest.json", manifest)
    if args.dry_run:
        return 0, manifest

    client = client_factory(args.server_url)
    status_before = client.get("/status", timeout=60)
    _atomic_json(run_root / "status_before_smoke.json", status_before)
    _scheduler_gate(status_before, expected_backend=args.expected_backend)
    smoke_rc = command_runner(smoke_command, run_root / "smoke.log")
    status_after_smoke = client.get("/status", timeout=60)
    _atomic_json(run_root / "status_after_smoke.json", status_after_smoke)
    _scheduler_gate(status_after_smoke, expected_backend=args.expected_backend)
    if smoke_rc != 0:
        manifest.update({"smoke_returncode": smoke_rc, "status": "SMOKE_FAILED"})
        _atomic_json(run_root / "campaign_manifest.json", manifest)
        return 1, manifest

    campaign_rc = command_runner(campaign_command, run_root / "campaign.log")
    status_after_campaign = client.get("/status", timeout=60)
    _atomic_json(run_root / "status_after_campaign.json", status_after_campaign)
    _scheduler_gate(status_after_campaign, expected_backend=args.expected_backend)

    collected: dict[str, CollectedResult] = {}
    for item in selected:
        try:
            collected[item.source_operator] = result_collector(
                campaign_workspace,
                item.definition_name,
                args.epochs,
            )
        except Exception as exc:  # noqa: BLE001 - isolate artifact corruption
            collected[item.source_operator] = CollectedResult(
                definition_name=item.definition_name,
                complete=False,
                geo_mean=None,
                best_code="",
                evidence=[],
                reason=f"cannot collect KernelGen result: {type(exc).__name__}: {exc}",
            )
    source_results: dict[str, tuple[StageResult, pathlib.Path | None]] = {}
    result_rows = []
    for item in selected:
        result = collected[item.source_operator]
        best_code_path = None
        evidence_paths = [
            item.seed_code_path,
            item.profile_evidence,
            *result.evidence,
        ]
        if result.complete and result.geo_mean is not None and result.best_code:
            best_code_path = run_root / "best" / f"{item.definition_name}.py"
            _atomic_text(best_code_path, result.best_code.rstrip() + "\n")
            evidence_paths.append(best_code_path)
            stage_result = StageResult(
                attempt_id=args.attempt_id,
                verdict=StageVerdict.PASSED,
                geo_mean=result.geo_mean,
                evidence=[str(path.resolve()) for path in evidence_paths],
            )
        elif result.complete:
            stage_result = StageResult(
                attempt_id=args.attempt_id,
                verdict=StageVerdict.FAILED,
                evidence=[str(path.resolve()) for path in evidence_paths],
                reason=result.reason,
            )
        else:
            evidence_paths.append(run_root / "campaign.log")
            stage_result = StageResult(
                attempt_id=args.attempt_id,
                verdict=StageVerdict.BLOCKED,
                evidence=[str(path.resolve()) for path in evidence_paths],
                reason=result.reason,
            )
        pipeline.update_stage(
            source_operator=item.source_operator,
            name=StageName.KERNELGEN_NATIVE,
            result=stage_result,
        )
        source_results[item.source_operator] = (stage_result, best_code_path)
        result_rows.append(
            {
                "source_operator": item.source_operator,
                "definition_name": item.definition_name,
                "verdict": stage_result.verdict.value,
                "geo_mean": stage_result.geo_mean,
                "best_code": str(best_code_path) if best_code_path else None,
                "reason": stage_result.reason,
                "evidence": stage_result.evidence,
            }
        )

    write_pipeline_results(pipeline_path, pipeline)
    write_second_stage_markdown(kernelgen_results_path, pipeline)
    _sync_results_rows(
        results_path=results_path,
        pipeline=pipeline,
        source_results=source_results,
    )
    result_report = {
        "schema_version": MANIFEST_SCHEMA,
        "attempt_id": args.attempt_id,
        "campaign_returncode": campaign_rc,
        "results": result_rows,
    }
    _atomic_json(run_root / "campaign_results.json", result_report)
    manifest.update(
        {
            "smoke_returncode": smoke_rc,
            "campaign_returncode": campaign_rc,
            "status": (
                "COMPLETED"
                if campaign_rc == 0
                and all(row["verdict"] == "passed" for row in result_rows)
                else "COMPLETED_WITH_FAILURES"
            ),
            "campaign_results": str(run_root / "campaign_results.json"),
        }
    )
    _atomic_json(run_root / "campaign_manifest.json", manifest)
    passed = campaign_rc == 0 and all(
        row["verdict"] == "passed" for row in result_rows
    )
    return (0 if passed else 1), manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--smoke-operator")
    parser.add_argument("--profile-evidence", action="append", required=True)
    parser.add_argument(
        "--seed-code-path",
        action="append",
        required=True,
        help="Validated Native baseline seed using SOURCE_OPERATOR=PATH",
    )
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--expected-backend", default="")
    parser.add_argument("--target-hardware", required=True)
    parser.add_argument("--catalog-name", default="flaggems-native")
    parser.add_argument("--native-catalog-root", type=pathlib.Path, required=True)
    parser.add_argument("--knowledge-catalog-path", type=pathlib.Path, required=True)
    parser.add_argument(
        "--knowledge-mode",
        choices=["read_write_v1", "read_only_v1"],
        default="read_write_v1",
    )
    parser.add_argument("--run-root", type=pathlib.Path, required=True)
    parser.add_argument("--results-md", type=pathlib.Path, required=True)
    parser.add_argument("--pipeline-json", type=pathlib.Path, required=True)
    parser.add_argument("--kernelgen-results-md", type=pathlib.Path, required=True)
    parser.add_argument("--runtime", choices=["claude", "codex"], default="claude")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-operators", type=int, default=2)
    parser.add_argument("--agents-per-operator", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--early-stop-rounds", type=int, default=3)
    parser.add_argument("--min-rounds", type=int, default=2)
    parser.add_argument("--max-round", type=int, default=15)
    parser.add_argument("--coder-timeout", type=int, default=3600)
    parser.add_argument("--no-reduced-precision", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    positive = (
        args.max_operators,
        args.agents_per_operator,
        args.epochs,
        args.min_rounds,
        args.max_round,
        args.coder_timeout,
    )
    if min(positive) < 1 or args.early_stop_rounds < 0:
        parser.error("workers, epochs, rounds and timeouts must be positive")
    if not args.attempt_id.strip():
        parser.error("--attempt-id must be non-empty")
    if args.min_rounds > args.max_round:
        parser.error("--min-rounds cannot exceed --max-round")
    if args.epochs > 1 and args.agents_per_operator < 2:
        parser.error("multi-epoch runs require at least two agents per operator")
    if not args.knowledge_catalog_path.is_dir():
        parser.error(
            f"knowledge catalog is missing: {args.knowledge_catalog_path}"
        )
    return args


def main() -> int:
    try:
        exit_code, report = run(_parse_args())
    except LaunchError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

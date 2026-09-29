#!/usr/bin/env python3
"""Run multiple Knowledge-enabled KernelGen operators by epoch.

Operators in one epoch run concurrently.  The next epoch starts only after
every still-active operator has completed publication and synthesis.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kernelgen.data.catalog import DEFAULT_CATALOG_NAME
from kernelgen.framework.runtime import CLI_RUNTIME_NAMES
from kernelgen.knowledge.config import KnowledgeMode, KnowledgeReviewerMode


def _optional_env_float(name: str):
    value = os.environ.get(name)
    return None if value in {None, ""} else float(value)


@dataclass(frozen=True)
class CommandResult:
    definition: str
    epoch: int
    returncode: int
    log_path: Path


def build_command(
    args: argparse.Namespace,
    definition: str,
    epoch: int,
) -> list[str]:
    """Build one run_example invocation without shell interpolation."""
    command = [
        sys.executable,
        "-u",
        str(args.run_example),
        "--definition",
        definition,
        "--implementation-language",
        args.implementation_language,
        "--target-hardware",
        args.target_hardware,
        "--eval-server",
        args.eval_server,
        "--workspace",
        str(args.workspace_root / definition),
        "--knowledge-catalog-path",
        str(args.knowledge_catalog_path),
        "--knowledge-mode",
        args.knowledge_mode,
        "--knowledge-reviewer-mode",
        args.knowledge_reviewer_mode,
        "--knowledge-run-id",
        f"{args.workspace_root.name}--{definition}",
        "--catalog-name",
        args.catalog_name,
        "--runtime",
        args.runtime,
        "--start-mode",
        args.start_mode if epoch == args.start_epoch else "resume",
        "--n-parallel",
        str(args.agents_per_operator),
        "--start-epoch",
        str(epoch),
        "--n-epoch",
        str(epoch),
        "--early-stop-rounds",
        str(args.early_stop_rounds),
        "--min-rounds",
        str(args.min_rounds),
        "--max-round",
        str(args.max_round),
        "--timeout",
        str(args.timeout),
    ]
    if args.knowledge_run_archive_path is not None:
        command.extend(
            [
                "--knowledge-run-archive-path",
                str(args.knowledge_run_archive_path),
            ]
        )
    if args.knowledge_derived_path is not None:
        command.extend(
            [
                "--knowledge-derived-path",
                str(args.knowledge_derived_path),
            ]
        )
    if args.model:
        command.extend(["--model", args.model])
    if args.eval_atol is not None:
        command.extend(
            [
                "--eval-atol",
                str(args.eval_atol),
                "--eval-rtol",
                str(args.eval_rtol),
            ]
        )
    if args.eval_tolerance_mode:
        command.extend(["--eval-tolerance-mode", args.eval_tolerance_mode])
    command.extend(
        [
            "--eval-required-matched-ratio",
            str(args.eval_required_matched_ratio),
        ]
    )
    if args.no_reduced_precision:
        command.append("--no-reduced-precision")
    if args.no_profile:
        command.append("--no-profile")
    if epoch == 1 and definition in args.seed_code_paths:
        command.extend(
            ["--seed-code-path", str(args.seed_code_paths[definition])]
        )
    if args.clean and epoch == 1:
        command.append("--clean")
    return command


def _run_operator(
    *,
    definition: str,
    epoch: int,
    command: Sequence[str],
    log_path: Path,
    cwd: Path,
) -> CommandResult:
    print(
        f"[{epoch}R] START {definition} (log: {log_path})",
        flush=True,
    )
    returncode = 127
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"$ {shlex.join(command)}\n\n")
            log.flush()
            completed = subprocess.run(
                list(command),
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        returncode = completed.returncode
    except OSError as exc:
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nCAMPAIGN_LAUNCH_ERROR: {exc}\n")
        except OSError:
            pass
    print(
        f"[{epoch}R] END   {definition} rc={returncode}",
        flush=True,
    )
    return CommandResult(
        definition=definition,
        epoch=epoch,
        returncode=returncode,
        log_path=log_path,
    )


def _published_epoch(
    workspace: Path,
    epoch: int,
    expected_agents: int,
) -> bool:
    """Return true when publication and required synthesis are durable."""
    epoch_root = workspace / f"{epoch}R"
    manifest_path = epoch_root / "epoch-completion.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(manifest, dict):
                return False
            attempted_agents = manifest.get("attempted_agents")
            successful_agents = manifest.get("successful_agents")
            failed_agents = manifest.get("failed_agents")
            failed_names = [item["name"] for item in failed_agents]
        except (KeyError, OSError, TypeError, ValueError):
            return False
        if (
            manifest.get("status") != "complete"
            or not isinstance(attempted_agents, list)
            or not isinstance(successful_agents, list)
            or not successful_agents
            or not isinstance(failed_agents, list)
            or not all(isinstance(name, str) for name in attempted_agents)
            or not all(isinstance(name, str) for name in successful_agents)
            or not all(
                isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and isinstance(item.get("error_type"), str)
                and isinstance(item.get("error"), str)
                for item in failed_agents
            )
            or len(set(attempted_agents)) != len(attempted_agents)
            or len(set(successful_agents)) != len(successful_agents)
            or len(set(failed_names)) != len(failed_names)
            or set(successful_agents) & set(failed_names)
            or (
                set(successful_agents) | set(failed_names)
                != set(attempted_agents)
            )
            or set(attempted_agents)
            != {f"agent{index}" for index in range(expected_agents)}
        ):
            return False
    else:
        # Backward compatibility: old epochs have no manifest and retain the
        # original strict all-agents completion contract.
        successful_agents = [
            f"agent{index}" for index in range(expected_agents)
        ]

    for name in successful_agents:
        agent = epoch_root / name
        ledger = agent / ".ledger.json"
        publish_result = (
            agent / ".kernelgen" / "knowledge" / "publish-result.json"
        )
        if not ledger.is_file() or not publish_result.is_file():
            return False
        try:
            publish_payload = json.loads(
                publish_result.read_text(encoding="utf-8")
            )
            if not isinstance(publish_payload, dict):
                return False
            status = publish_payload.get("status")
        except (OSError, ValueError):
            return False
        if status not in {"published", "noop"}:
            return False
    if manifest_path.is_file() or expected_agents > 1:
        return (epoch_root / "synthesis" / "synthesis.json").is_file()
    return True


def _partition_epoch(
    definitions: Sequence[str],
    workspace_root: Path,
    epoch: int,
    expected_agents: int,
    *,
    skip_completed: bool,
) -> tuple[list[str], list[str]]:
    """Split durable completions from operators that still need this epoch."""
    if not skip_completed:
        return [], list(definitions)

    completed: list[str] = []
    pending: list[str] = []
    for definition in definitions:
        if _published_epoch(
            workspace_root / definition,
            epoch,
            expected_agents,
        ):
            completed.append(definition)
        else:
            pending.append(definition)
    return completed, pending


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple operators in parallel with one shared V1 "
            "Knowledge Catalog."
        )
    )
    parser.add_argument(
        "--definitions",
        nargs="+",
        required=True,
        help="Definition names to optimize.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        required=True,
        help="One child workspace is created per definition.",
    )
    parser.add_argument(
        "--knowledge-catalog-path",
        type=Path,
        required=True,
        help="Shared Knowledge Catalog.",
    )
    parser.add_argument(
        "--knowledge-mode",
        choices=[item.value for item in KnowledgeMode],
        default=KnowledgeMode.READ_WRITE_V1.value,
        help="Knowledge access mode (default: read_write_v1).",
    )
    parser.add_argument(
        "--knowledge-derived-path",
        type=Path,
        default=None,
        help=(
            "Optional external directory for rebuildable Knowledge indexes."
        ),
    )
    parser.add_argument(
        "--knowledge-run-archive-path",
        type=Path,
        default=None,
        help="Optional shared immutable run archive root.",
    )
    parser.add_argument(
        "--knowledge-reviewer-mode",
        choices=[item.value for item in KnowledgeReviewerMode],
        default=os.environ.get(
            "KERNELGEN_KNOWLEDGE_REVIEWER_MODE",
            KnowledgeReviewerMode.OFF.value,
        ),
        help=(
            "Runtime Candidate review mode passed to every operator: "
            "off, shadow, or enforce (default: off)."
        ),
    )
    parser.add_argument(
        "--max-operators",
        type=int,
        default=2,
        help="Maximum operator workflows running at once (default: 2).",
    )
    parser.add_argument(
        "--agents-per-operator",
        type=int,
        default=2,
        help="Parallel optimization agents inside each operator (default: 2).",
    )
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--n-epoch", type=int, default=1)
    parser.add_argument(
        "--start-mode",
        choices=["fresh", "fork", "resume"],
        default=None,
    )
    parser.add_argument(
        "--seed-code-path",
        action="append",
        default=[],
        metavar="DEFINITION=PATH",
        help=(
            "Validated Native baseline for one definition. Repeat for multiple "
            "definitions; forwarded only to epoch 1."
        ),
    )
    parser.add_argument(
        "--catalog-name",
        default=os.environ.get("KERNELGEN_CATALOG_NAME", DEFAULT_CATALOG_NAME),
    )
    parser.add_argument(
        "--eval-server",
        default=os.environ.get("FIB_EVAL_SERVER", "http://localhost:8000"),
    )
    parser.add_argument("--target-hardware", default="Ascend910B")
    parser.add_argument(
        "--eval-atol",
        type=float,
        default=_optional_env_float("KERNELGEN_EVAL_ATOL"),
        help="Declared evaluator absolute tolerance (or KERNELGEN_EVAL_ATOL)",
    )
    parser.add_argument(
        "--eval-rtol",
        type=float,
        default=_optional_env_float("KERNELGEN_EVAL_RTOL"),
        help="Declared evaluator relative tolerance (or KERNELGEN_EVAL_RTOL)",
    )
    parser.add_argument(
        "--eval-tolerance-mode",
        choices=["fixed", "strict"],
        default=os.environ.get("KERNELGEN_EVAL_TOLERANCE_MODE", ""),
        help="Declared evaluator tolerance mode",
    )
    parser.add_argument(
        "--eval-required-matched-ratio",
        type=float,
        default=float(os.environ.get("KERNELGEN_EVAL_REQUIRED_MATCHED_RATIO", "1.0")),
        help="Required matched ratio (or KERNELGEN_EVAL_REQUIRED_MATCHED_RATIO)",
    )
    parser.add_argument(
        "--no-reduced-precision",
        action="store_true",
        help="Tell agents not to explore reduced-precision internal computation",
    )
    parser.add_argument("--implementation-language", default="triton")
    parser.add_argument(
        "--runtime",
        choices=CLI_RUNTIME_NAMES,
        default=os.environ.get("KERNELGEN_RUNTIME", "claude"),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--early-stop-rounds", type=int, default=3)
    parser.add_argument("--min-rounds", type=int, default=2)
    parser.add_argument("--max-round", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean each operator workspace before 1R only.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Keep operators with durable epoch publication/synthesis and run "
            "only incomplete operators."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without starting operators.",
    )
    parser.add_argument(
        "--run-example",
        type=Path,
        default=Path(__file__).with_name("run_example.py"),
        help=argparse.SUPPRESS,
    )
    return parser


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.model is None:
        args.model = (
            os.environ.get("CODEX_MODEL", "")
            if args.runtime == "codex"
            else os.environ.get("MODEL", "")
        )
    if args.knowledge_reviewer_mode not in {
        item.value for item in KnowledgeReviewerMode
    }:
        parser.error(
            "KERNELGEN_KNOWLEDGE_REVIEWER_MODE must be off, shadow, or enforce"
        )
    if (
        args.knowledge_mode == KnowledgeMode.READ_ONLY_V1.value
        and args.knowledge_reviewer_mode != KnowledgeReviewerMode.OFF.value
    ):
        parser.error("read_only_v1 requires --knowledge-reviewer-mode off")
    if len(set(args.definitions)) != len(args.definitions):
        parser.error("--definitions contains duplicates")
    if any(
        not item
        or item in {".", ".."}
        or "/" in item
        or "\\" in item
        for item in args.definitions
    ):
        parser.error("definition names cannot contain path separators")
    if args.max_operators < 1:
        parser.error("--max-operators must be >= 1")
    if args.agents_per_operator < 1:
        parser.error("--agents-per-operator must be >= 1")
    if args.start_epoch < 1 or args.start_epoch > args.n_epoch:
        parser.error("--start-epoch must be between 1 and --n-epoch")
    args.start_mode = args.start_mode or (
        "resume" if args.start_epoch > 1 else "fresh"
    )
    if args.start_mode != "resume" and args.start_epoch > 1:
        parser.error("--start-epoch > 1 requires --start-mode resume")
    if args.n_epoch > args.start_epoch and args.agents_per_operator < 2:
        parser.error("multi-epoch runs require --agents-per-operator >= 2")
    if args.clean and args.start_epoch > 1:
        parser.error("--clean cannot be used with --start-epoch > 1")
    seed_code_paths = {}
    for value in args.seed_code_path:
        definition, separator, raw_path = value.partition("=")
        if not separator or not definition or not raw_path:
            parser.error("--seed-code-path must use DEFINITION=PATH")
        if definition in seed_code_paths:
            parser.error(
                f"duplicate --seed-code-path for definition {definition!r}"
            )
        if definition not in args.definitions:
            parser.error(
                f"--seed-code-path definition is not selected: {definition!r}"
            )
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"seed code is missing or empty: {path}")
        seed_code_paths[definition] = path
    if seed_code_paths and (args.start_mode != "fresh" or args.start_epoch != 1):
        parser.error(
            "--seed-code-path requires --start-mode fresh and --start-epoch 1"
        )
    args.seed_code_paths = seed_code_paths

    if (args.eval_atol is None) != (args.eval_rtol is None):
        parser.error("--eval-atol and --eval-rtol must be provided together")
    if args.eval_atol is not None and (args.eval_atol < 0 or args.eval_rtol < 0):
        parser.error("--eval-atol and --eval-rtol must be non-negative")
    if not 0 < args.eval_required_matched_ratio <= 1:
        parser.error("--eval-required-matched-ratio must be in (0, 1]")

def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    args.run_example = args.run_example.resolve()
    args.workspace_root = args.workspace_root.resolve()
    args.knowledge_catalog_path = args.knowledge_catalog_path.resolve()
    if args.knowledge_derived_path is not None:
        args.knowledge_derived_path = args.knowledge_derived_path.resolve()
    if args.knowledge_run_archive_path is not None:
        args.knowledge_run_archive_path = (
            args.knowledge_run_archive_path.resolve()
        )

    print(
        "Campaign: "
        f"{len(args.definitions)} operators, "
        f"{args.start_epoch}R-{args.n_epoch}R, "
        f"max {args.max_operators} operators × "
        f"{args.agents_per_operator} agents",
        flush=True,
    )
    print(f"Shared KB: {args.knowledge_catalog_path}", flush=True)
    print(f"Knowledge mode: {args.knowledge_mode}", flush=True)

    print(
        "Eval contract: "
        f"mode={args.eval_tolerance_mode or 'unspecified'} "
        f"atol={args.eval_atol} rtol={args.eval_rtol} "
        f"matched_ratio={args.eval_required_matched_ratio} "
        f"reduced_precision={not args.no_reduced_precision}",
        flush=True,
    )
    if args.dry_run:
        active = list(args.definitions)
        for epoch in range(args.start_epoch, args.n_epoch + 1):
            print(f"\n# {epoch}R")
            completed, pending = _partition_epoch(
                active,
                args.workspace_root,
                epoch,
                args.agents_per_operator,
                skip_completed=args.skip_completed,
            )
            for definition in completed:
                print(f"# SKIP {definition} (durably complete)")
            for definition in pending:
                print(shlex.join(build_command(args, definition, epoch)))
        return 0
    if not args.run_example.is_file():
        parser.error(f"run_example.py not found: {args.run_example}")
    if not args.knowledge_catalog_path.is_dir():
        parser.error(
            "knowledge catalog not found: "
            f"{args.knowledge_catalog_path}"
        )

    args.workspace_root.mkdir(parents=True, exist_ok=True)
    logs_root = args.workspace_root / "_logs"
    active = list(args.definitions)
    failures: list[str] = []

    for epoch in range(args.start_epoch, args.n_epoch + 1):
        print(
            f"\n=== {epoch}R: {len(active)} operators in parallel ===",
            flush=True,
        )
        completed, pending = _partition_epoch(
            active,
            args.workspace_root,
            epoch,
            args.agents_per_operator,
            skip_completed=args.skip_completed,
        )
        for definition in completed:
            print(
                f"[{epoch}R] SKIP  {definition} (durably complete)",
                flush=True,
            )

        results: dict[str, CommandResult] = {}
        with ThreadPoolExecutor(max_workers=args.max_operators) as pool:
            futures = {}
            for definition in pending:
                log_path = (
                    logs_root / definition / f"epoch-{epoch:04d}.log"
                )
                future = pool.submit(
                    _run_operator,
                    definition=definition,
                    epoch=epoch,
                    command=build_command(args, definition, epoch),
                    log_path=log_path,
                    cwd=args.run_example.parents[2],
                )
                futures[future] = definition
            for future in as_completed(futures):
                definition = futures[future]
                try:
                    results[definition] = future.result()
                except Exception as exc:
                    log_path = (
                        logs_root
                        / definition
                        / f"epoch-{epoch:04d}.log"
                    )
                    results[definition] = CommandResult(
                        definition=definition,
                        epoch=epoch,
                        returncode=127,
                        log_path=log_path,
                    )
                    failures.append(
                        f"{definition} {epoch}R launcher failed: {exc}"
                    )

        next_active = list(completed)
        for definition in pending:
            result = results[definition]
            workspace = args.workspace_root / definition
            if _published_epoch(
                workspace,
                epoch,
                args.agents_per_operator,
            ):
                next_active.append(definition)
                continue
            failures.append(
                f"{definition} {epoch}R incomplete "
                f"(rc={result.returncode}, log={result.log_path})"
            )
        active = next_active
        if not active:
            break

    print("\n=== Campaign summary ===", flush=True)
    print(
        "Completed through requested epoch: "
        + (", ".join(active) if active else "none"),
        flush=True,
    )
    if failures:
        print("Incomplete:", flush=True)
        for item in failures:
            print(f"  - {item}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare one Native KernelGen result for review in a FlagGems worktree."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from kernelgen.framework.runtime import (
    CLI_RUNTIME_NAMES,
    create_cli_runtime,
    resolve_cli_runtime_options,
)
from kernelgen.workflows.native_to_flaggems import NativeToFlagGemsWorkflow


DEFAULT_CLAUDE_MODEL = os.environ.get("MODEL", "inherit")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--source-operator", default="")
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--native-kernel-path", type=Path, required=True)
    parser.add_argument("--native-definition-path", type=Path, required=True)
    parser.add_argument("--flaggems-worktree", type=Path, required=True)
    parser.add_argument("--accuracy-file", action="append", required=True)
    parser.add_argument("--benchmark-file", action="append", required=True)
    parser.add_argument(
        "--standard-path",
        default="docs/content/zh-cn/testing/kernelgen-integration.md",
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument(
        "--runtime",
        choices=CLI_RUNTIME_NAMES,
        default=os.environ.get("KERNELGEN_RUNTIME", "claude"),
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--auth-token")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = _parse_args()
    runtime_options = resolve_cli_runtime_options(
        args.runtime,
        model=args.model,
        base_url=args.base_url,
        auth_token=args.auth_token,
        claude_default_model=DEFAULT_CLAUDE_MODEL,
    )
    args.workspace = args.workspace.expanduser().resolve()
    args.workspace.mkdir(parents=True, exist_ok=True)

    def runtime_factory(path: str):
        return create_cli_runtime(
            args.runtime,
            workspace=path,
            model=runtime_options["model"],
            base_url=runtime_options["base_url"],
            auth_token=runtime_options["auth_token"],
            timeout=args.timeout,
            idle_timeout=max(120, args.timeout // 2),
        )

    output = NativeToFlagGemsWorkflow(
        cwd=str(args.workspace),
        runtime_factory=runtime_factory,
    ).run(
        {
            "operator": args.operator,
            "source_operator": args.source_operator,
            "vendor": args.vendor,
            "native_kernel_path": str(args.native_kernel_path),
            "native_definition_path": str(args.native_definition_path),
            "flaggems_worktree": str(args.flaggems_worktree),
            "accuracy_files": args.accuracy_file,
            "benchmark_files": args.benchmark_file,
            "standard_path": args.standard_path,
            "run_tests": args.run_tests,
        }
    )
    print(output.model_dump_json(indent=2))
    return 0 if output.ready_for_target_validation else 1


if __name__ == "__main__":
    raise SystemExit(main())

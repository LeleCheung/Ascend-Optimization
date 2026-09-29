#!/usr/bin/env python
"""Resume a historical Python Batch; new batches use kg run --batch-file."""

import argparse
import os
import shutil
from pathlib import Path

from kernelgen.framework import copy_claude_directory, copy_mcp_configuration
from kernelgen.framework.runtime import (
    CLI_RUNTIME_NAMES,
    create_cli_runtime,
    materialize_claude_runtime_config,
    resolve_cli_runtime_options,
)
from kernelgen.data.catalog import DEFAULT_CATALOG_NAME
from kernelgen.data.timeout_policy import (
    DEFAULT_EVAL_TIMEOUT_SECONDS,
    TimeoutPolicy,
)
from kernelgen.workflows.legacy.batch_simple_opt_definition import (
    BATCH_SIMPLE_OPT_OUTPUT_FILENAME,
    BatchSimpleOptDefinitionWorkflow,
)


KERNELGEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLAUDE_MODEL = os.environ.get("MODEL", "deepseek-v4-pro[1m]")


def bind_reference_code_paths(
    reference_paths: list[Path],
    definition_names: list[str],
) -> dict[str, Path]:
    """Bind explicit ``<definition_name>.<extension>`` source paths."""
    definitions = set(definition_names)
    bound: dict[str, Path] = {}
    for path in reference_paths:
        definition_name = path.stem
        if definition_name not in definitions:
            raise ValueError(
                "reference code filename does not match a requested definition: "
                f"{path}"
            )
        if definition_name in bound:
            raise ValueError(
                f"duplicate reference code path for {definition_name!r}"
            )
        if not path.is_file():
            raise ValueError(f"reference code path is not a file: {path}")
        bound[definition_name] = path
    return bound


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple SimpleOpt definitions in parallel",
    )
    parser.add_argument(
        "--definition-name",
        "-n",
        dest="definition_names",
        action="append",
        required=True,
        help="Definition name; repeat this option for each task",
    )
    parser.add_argument(
        "--catalog-name",
        default=os.environ.get(
            "KERNELGEN_CATALOG_NAME",
            DEFAULT_CATALOG_NAME,
        ),
    )
    parser.add_argument("--workspace", "-w", type=Path, default=None)
    parser.add_argument(
        "--eval-server",
        default=os.environ.get("FIB_EVAL_SERVER", "http://localhost:8000"),
    )
    parser.add_argument("--target-hardware", default="Ascend910B")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument(
        "--launch-interval-seconds",
        type=float,
        default=1.0,
        help="Delay between parallel workflow submissions (default: 1.0s)",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=15,
        help="Maximum total measured rounds per definition (default: 15)",
    )
    parser.add_argument(
        "--early-stop-rounds",
        type=int,
        default=3,
        help=(
            "Stop after this many measured rounds without improvement per "
            "definition; 0 disables plateau stopping (default: 3)"
        ),
    )
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=2,
        help="Minimum measured rounds per definition (default: 2)",
    )
    parser.add_argument("--max-coder-sessions", type=int, default=3)
    parser.add_argument(
        "--reference-code-path",
        "--reference-code",
        "--reference-triton-path",
        "--reference-triton",
        dest="reference_code_paths",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional per-definition source named <definition_name>.<extension>; "
            "repeat for multiple definitions. Directory discovery belongs to the "
            "calling shell; legacy --reference-triton flags remain accepted"
        ),
    )
    parser.add_argument(
        "--reference-code-prompt-path",
        "--reference-triton-prompt-path",
        type=Path,
        default=None,
        help=(
            "Optional shared guidance for matched reference sources, including "
            "hardware provenance, useful ideas, and transfer limitations"
        ),
    )
    parser.add_argument(
        "--runtime",
        choices=CLI_RUNTIME_NAMES,
        default=os.environ.get("KERNELGEN_RUNTIME", "claude"),
        help="LLM CLI runtime (default: claude)",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument(
        "--eval-timeout-seconds",
        type=int,
        default=DEFAULT_EVAL_TIMEOUT_SECONDS,
        help=(
            "Server-side isolated preflight/eval execution timeout; transport, "
            "Coder idle, and Coder hard limits are derived automatically "
            f"(default: {DEFAULT_EVAL_TIMEOUT_SECONDS}s)"
        ),
    )
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    runtime_options = resolve_cli_runtime_options(
        args.runtime,
        model=args.model,
        base_url=args.base_url,
        auth_token=args.auth_token,
        claude_default_model=DEFAULT_CLAUDE_MODEL,
    )
    args.model = runtime_options["model"]
    args.base_url = runtime_options["base_url"]
    args.auth_token = runtime_options["auth_token"]
    timeout_policy = TimeoutPolicy(args.eval_timeout_seconds)
    if (
        args.reference_code_prompt_path is not None
        and not args.reference_code_paths
    ):
        parser.error(
            "--reference-code-prompt-path requires "
            "--reference-code-path"
        )
    try:
        reference_paths = bind_reference_code_paths(
            args.reference_code_paths,
            args.definition_names,
        )
    except ValueError as exc:
        parser.error(str(exc))

    workspace = args.workspace or KERNELGEN_ROOT / "runs" / "batch_simple_opt"
    if args.clean and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    copy_claude_directory(
        KERNELGEN_ROOT / ".claude",
        workspace / ".claude",
        include_skills=False,
    )
    mcp_source = KERNELGEN_ROOT / ".kernelgen" / "mcp.json"
    if mcp_source.is_file():
        copy_mcp_configuration(
            mcp_source,
            workspace / ".mcp.json",
            tool_timeout_seconds=timeout_policy.coder_idle_timeout_seconds,
        )
    claude_config_dir = (
        materialize_claude_runtime_config(workspace, args.model)
        if args.runtime == "claude"
        else None
    )

    os.environ["FIB_EVAL_SERVER"] = args.eval_server

    def make_runtime(path):
        return create_cli_runtime(
            args.runtime,
            workspace=path,
            model=args.model,
            base_url=args.base_url,
            auth_token=args.auth_token,
            claude_config_dir=claude_config_dir,
            timeout=timeout_policy.coder_hard_timeout_seconds,
            idle_timeout=timeout_policy.coder_idle_timeout_seconds,
        )

    if reference_paths:
        print(
            f"[BatchSimpleOpt] matched reference code sources for "
            f"{len(reference_paths)}/{len(args.definition_names)} definitions",
            flush=True,
        )

    definitions = [
        {
            "definition_name": definition_name,
            "catalog_name": args.catalog_name,
            "target_hardware": args.target_hardware,
            "eval_server_url": args.eval_server,
            "early_stop_rounds": args.early_stop_rounds,
            "min_rounds": args.min_rounds,
            "max_round": args.max_round,
            "max_coder_sessions": args.max_coder_sessions,
            "reference_code_path": reference_paths.get(definition_name),
            "reference_code_prompt_path": (
                args.reference_code_prompt_path
                if definition_name in reference_paths
                else None
            ),
            "eval_timeout_seconds": args.eval_timeout_seconds,
        }
        for definition_name in args.definition_names
    ]
    result = BatchSimpleOptDefinitionWorkflow(
        cwd=str(workspace),
        runtime_factory=make_runtime,
    ).run({
        "definitions": definitions,
        "max_workers": args.max_workers,
        "launch_interval_seconds": args.launch_interval_seconds,
    })

    print(result.summary)
    print(f"results: {workspace / BATCH_SIMPLE_OPT_OUTPUT_FILENAME}")
    raise SystemExit(
        0 if all(item.status == "PASSED" for item in result.results) else 1
    )


if __name__ == "__main__":
    main()

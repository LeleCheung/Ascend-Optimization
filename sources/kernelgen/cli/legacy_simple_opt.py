#!/usr/bin/env python
"""Historical v1 CLI execution with its original flat workspace layout."""

import os
import shutil
from pathlib import Path

from kernelgen.framework.run_options import launcher_parser
from kernelgen.framework import copy_claude_directory, copy_mcp_configuration
from kernelgen.framework.runtime import (
    create_cli_runtime,
    materialize_claude_runtime_config,
    resolve_cli_runtime_options,
)
from kernelgen.data.timeout_policy import (
    TimeoutPolicy,
)
from kernelgen.workflows.legacy.simple_opt import SimpleOptWorkflow


KERNELGEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLAUDE_MODEL = os.environ.get("MODEL", "deepseek-v4-pro[1m]")
DEFAULT_EVAL_SERVER = os.environ.get("FIB_EVAL_SERVER", "http://localhost:8000")


def legacy_main(argv=None):
    """Only for persisted v1 CLI campaigns with the original workspace layout."""
    parser = launcher_parser("simple_opt")
    args = parser.parse_args(argv)
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

    if args.workspace is None:
        args.workspace = (
            KERNELGEN_ROOT / "runs" / "simple_opt" / args.definition
        )
    if args.clean and args.workspace.exists():
        shutil.rmtree(args.workspace)
    args.workspace.mkdir(parents=True, exist_ok=True)

    claude_source = KERNELGEN_ROOT / ".claude"
    copy_claude_directory(
        claude_source,
        args.workspace / ".claude",
        include_skills=args.knowledge_catalog_path is not None,
    )
    mcp_source = KERNELGEN_ROOT / ".kernelgen" / "mcp.json"
    if mcp_source.is_file():
        copy_mcp_configuration(
            mcp_source,
            args.workspace / ".mcp.json",
            tool_timeout_seconds=timeout_policy.coder_idle_timeout_seconds,
        )
    claude_config_dir = (
        materialize_claude_runtime_config(args.workspace, args.model)
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

    workflow = SimpleOptWorkflow(
        cwd=str(args.workspace),
        runtime_factory=make_runtime,
    )
    result = workflow.run({
        "definition_name": args.definition,
        "catalog_name": args.catalog_name,
        "target_hardware": args.target_hardware,
        "implementation_language": args.language,
        "profile_enabled": args.profile,
        "eval_server_url": args.eval_server,
        "early_stop_rounds": args.early_stop_rounds,
        "min_rounds": args.min_rounds,
        "max_round": args.max_round,
        "warmup_ms": args.warmup_ms,
        "benchmark_ms": args.benchmark_ms,
        "num_trials": args.num_trials,
        "max_coder_sessions": args.max_coder_sessions,
        "destination_passing_style": args.dps,
        "reference_code_path": args.reference_code_path,
        "reference_code_prompt_path": args.reference_code_prompt_path,
        "knowledge_catalog_path": args.knowledge_catalog_path,
        "eval_timeout_seconds": args.eval_timeout_seconds,
    })

    print(f"definition: {result.definition_name} ({result.op_type})")
    print(f"status: {result.status}")
    print(f"rounds: {result.rounds}")
    print(f"best_geo_mean: {result.best_geo_mean}")
    print(f"workspace: {result.workspace}")
    if result.summary:
        print(f"summary: {result.summary}")

    raise SystemExit(0 if result.status == "PASSED" else 1)

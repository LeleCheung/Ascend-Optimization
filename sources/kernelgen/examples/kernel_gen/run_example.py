#!/usr/bin/env python
"""Run KernelGenWorkflow — multi-epoch, multi-agent kernel optimization.

This is the most complex pipeline: AnalyzerAgent (cold-start) → N parallel
CoderAgents (each with eval/record-finalize loop) → DistillerAgent → epoch KB reducer
→ EpochSummaryAgent → next epoch with directions.

Usage:
    cd /path/to/kernelgen
    python3 examples/kernel_gen/run_example.py \\
        --definition flaggems_rsqrt --n-parallel 3 --n-epoch 2

    # Or with all options:
    python3 examples/kernel_gen/run_example.py \\
        --definition flaggems_rsqrt \\
        --n-parallel 3 --n-epoch 2 \\
        --workspace /tmp/kernelgen_run \\
        --model deepseek-v4-pro[1m] \\
        --base-url https://zyapi.xmsxb.com \\
        --auth-token sk-xxx \\
        --eval-server http://106.63.13.115:8000 \\
        --early-stop-rounds 3 --min-rounds 2

Environment variables (override defaults):
    FIB_EVAL_SERVER         Eval server URL
    KERNELGEN_CATALOG_NAME  Server-owned built-in Catalog name
    KERNELGEN_KNOWLEDGE_REVIEWER_MODE  off, shadow, or enforce
    MODEL                   LLM model name
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from kernelgen.framework.run_options import launcher_parser
from kernelgen.framework import copy_claude_directory, materialize_mcp_configuration
from kernelgen.framework.runtime import (
    create_cli_runtime,
    materialize_claude_runtime_config,
    resolve_cli_runtime_options,
)
from kernelgen.workflows.optimization.kernelgen import KernelGenWorkflow, KernelGenInput
from kernelgen.workflows.optimization.single_coder.reference import load_reference_code
from kernelgen.data.catalog import (
    resolve_builtin_catalog_path,
)
from kernelgen.knowledge.config import (
    KnowledgeConfig,
    KnowledgeMode,
    KnowledgeReviewerMode,
)
from kernelgen.data.trace import load_catalog_optimization_context


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EVAL_SERVER = os.environ.get("FIB_EVAL_SERVER", "http://localhost:8000")
DEFAULT_MODEL = os.environ.get("MODEL", "deepseek-v4-flash[1m]")
DEFAULT_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://zyapi.xmsxb.com")
DEFAULT_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_definition(defn_name: str, catalog_name: str) -> dict:
    """Load a prompt-facing definition from a Server-owned Catalog."""
    definition, _ = load_catalog_optimization_context(
        resolve_builtin_catalog_path(catalog_name),
        defn_name,
    )
    return definition.model_dump(exclude_none=True)


def setup_workspace(
    workspace: Path,
    kernelgen_root: Path,
    *,
    knowledge_catalog_path: Path | None = None,
    cross_epoch_knowledge: bool = True,
):
    """Prepare runtime files and an optional legacy workspace-local KB."""
    workspace.mkdir(parents=True, exist_ok=True)

    # Copy provider settings and materialize neutral roles and skills.
    claude_src = kernelgen_root / ".claude"
    copy_claude_directory(claude_src, workspace / ".claude")
    print("  ✅ agent roles, skills, and provider configuration materialized")

    mcp_src = kernelgen_root / ".kernelgen" / "mcp.json"
    if mcp_src.is_file():
        materialize_mcp_configuration(mcp_src, workspace)
        print("  ✅ neutral MCP configuration materialized")

    if knowledge_catalog_path is not None or not cross_epoch_knowledge:
        return

    # Seed a legacy workspace-local KB when no shared V1 Catalog was provided.
    # Existing run KBs are left untouched when resuming.
    kb_root = workspace / "kb"
    kb_src = kernelgen_root / "kb"
    if not kb_root.exists():
        if kb_src.is_dir():
            shutil.copytree(
                kb_src,
                kb_root,
                ignore=shutil.ignore_patterns(".git"),
            )
        else:
            kb_root.mkdir(parents=True)

    # Git init base KB (epoch-level snapshots)
    if not (kb_root / ".git").exists():
        env_git = {
            **os.environ,
            "GIT_AUTHOR_NAME": "init", "GIT_AUTHOR_EMAIL": "i@i",
            "GIT_COMMITTER_NAME": "init", "GIT_COMMITTER_EMAIL": "i@i",
        }
        subprocess.run(["git", "init"], cwd=str(kb_root), capture_output=True, check=True)
        subprocess.run(["git", "add", "."], cwd=str(kb_root), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "base", "--allow-empty"],
                       cwd=str(kb_root), capture_output=True, check=True, env=env_git)
        print(f"  ✅ kb/ git initialized")


def print_results(out, workspace: Path):
    """Print workflow results and verification info."""
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n  Status:       {out.status}")
    print(f"  Best geo_mean: {out.best_geo_mean}")
    print(f"  Num agents:   {out.num_agents}")
    print(f"  Best code:    {len(out.best_code)} chars")

    if out.per_agent:
        print(f"\n  Per-agent:")
        for a in out.per_agent:
            print(f"    {a.workspace}: status={a.status}, geo_mean={a.geo_mean}")

    # Check epoch directories
    print(f"\n  Epoch directories:")
    for d in sorted(workspace.iterdir()):
        if d.is_dir() and d.name.endswith("R"):
            agents = [x.name for x in d.iterdir() if x.is_dir()]
            print(f"    {d.name}/: {agents}")

    # KB git log
    kb_root = workspace / "kb"
    if (kb_root / ".git").exists():
        r = subprocess.run(["git", "log", "--oneline"], cwd=str(kb_root),
                           capture_output=True, text=True)
        print(f"\n  KB git log:")
        for line in r.stdout.strip().splitlines():
            print(f"    {line}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def legacy_main(argv=None):
    """Only for persisted v1 CLI campaigns with the original workspace layout."""
    parser = launcher_parser("kernelgen")
    args = parser.parse_args(argv)
    try:
        reference = load_reference_code(args.reference_code_path, args.reference_code_prompt_path)
    except ValueError as exc:
        parser.error(str(exc))
    runtime_options = resolve_cli_runtime_options(
        args.runtime,
        model=args.model,
        base_url=args.base_url,
        auth_token=args.auth_token,
        claude_default_model=DEFAULT_MODEL,
    )
    args.model = runtime_options["model"]
    args.base_url = runtime_options["base_url"] or (
        DEFAULT_BASE_URL if args.runtime == "claude" else None
    )
    args.auth_token = runtime_options["auth_token"] or (
        DEFAULT_AUTH_TOKEN if args.runtime == "claude" else None
    )
    start_mode = args.start_mode or (
        "resume" if args.start_epoch > 1 else "fresh"
    )
    if start_mode != "resume" and args.start_epoch > 1:
        parser.error("--start-epoch > 1 requires --start-mode resume")
    if args.seed_code_path is not None:
        args.seed_code_path = args.seed_code_path.expanduser().resolve()
        if start_mode != "fresh" or args.start_epoch != 1:
            parser.error(
                "--seed-code-path requires --start-mode fresh and "
                "--start-epoch 1"
            )
        if (
            not args.seed_code_path.is_file()
            or args.seed_code_path.stat().st_size == 0
        ):
            parser.error(
                f"--seed-code-path is missing or empty: {args.seed_code_path}"
            )
    if args.knowledge_mode is not None and args.knowledge_catalog_path is None:
        parser.error(
            "--knowledge-catalog-path is required with --knowledge-mode"
        )
    if (
        args.knowledge_derived_path is not None
        and args.knowledge_catalog_path is None
    ):
        parser.error(
            "--knowledge-catalog-path is required with "
            "--knowledge-derived-path"
        )
    if args.knowledge_reviewer_mode not in {
        item.value for item in KnowledgeReviewerMode
    }:
        parser.error(
            "KERNELGEN_KNOWLEDGE_REVIEWER_MODE must be off, shadow, or enforce"
        )
    if (
        args.knowledge_reviewer_mode != KnowledgeReviewerMode.OFF.value
        and args.knowledge_catalog_path is None
    ):
        parser.error(
            "--knowledge-catalog-path is required when Knowledge Reviewer "
            "mode is shadow or enforce"
        )
    effective_knowledge_mode = (
        args.knowledge_mode
        or (
            KnowledgeMode.READ_WRITE_V1.value
            if args.knowledge_catalog_path is not None
            else None
        )
    )
    if (
        effective_knowledge_mode == KnowledgeMode.READ_ONLY_V1.value
        and args.knowledge_reviewer_mode != KnowledgeReviewerMode.OFF.value
    ):
        parser.error("read_only_v1 requires --knowledge-reviewer-mode off")
    if start_mode == "fork" and args.knowledge_catalog_path is None:
        parser.error(
            "--start-mode fork requires --knowledge-catalog-path"
        )

    if (args.eval_atol is None) != (args.eval_rtol is None):
        parser.error("--eval-atol and --eval-rtol must be provided together")
    if args.eval_atol is not None and (args.eval_atol < 0 or args.eval_rtol < 0):
        parser.error("--eval-atol and --eval-rtol must be non-negative")
    if not 0 < args.eval_required_matched_ratio <= 1:
        parser.error("--eval-required-matched-ratio must be in (0, 1]")
    # Resolve workspace
    if args.workspace is None:
        args.workspace = Path(f"/tmp/kernelgen_{args.definition}")

    print("=" * 70)
    print(f"  KernelGen: {args.definition}")
    print(f"  Epochs: {args.start_epoch}-{args.n_epoch}, Agents/epoch: {args.n_parallel}")
    print(f"  Start mode: {start_mode}")
    print(
        "  Initial seed: "
        + (str(args.seed_code_path) if args.seed_code_path else "none")
    )
    print(f"  Catalog: {args.catalog_name}")
    print(f"  Runtime: {args.runtime}")
    print(f"  Model: {args.model}")
    print(f"  Eval server: {args.eval_server}")
    print(f"  Target hardware: {args.target_hardware}")
    print(f"  Implementation: {args.language}")
    print(
        "  Eval contract: "
        f"mode={args.eval_tolerance_mode or 'unspecified'} "
        f"atol={args.eval_atol} rtol={args.eval_rtol} "
        f"matched_ratio={args.eval_required_matched_ratio} "
        f"reduced_precision={not args.no_reduced_precision}"
    )
    knowledge_mode = effective_knowledge_mode or "disabled"
    print(f"  Knowledge mode: {knowledge_mode}")
    reviewer_mode = (
        args.knowledge_reviewer_mode
        if args.knowledge_catalog_path
        else "disabled"
    )
    print(f"  Knowledge Reviewer: {reviewer_mode}")
    print(f"  Profile analysis: {'enabled' if args.profile else 'disabled'}")
    print(f"  Workspace: {args.workspace}")
    print("=" * 70)

    # Clean if requested
    if args.clean and args.start_epoch > 1:
        parser.error("--clean cannot be used with --start-epoch > 1")
    if args.clean and args.finalize_epoch is not None:
        parser.error("--clean cannot be used with --finalize-epoch")
    if args.clean and args.workspace.exists():
        shutil.rmtree(args.workspace)
        print("  🗑️  Workspace cleaned")

    # Setup workspace
    kernelgen_root = Path(__file__).resolve().parents[2]
    setup_workspace(
        args.workspace,
        kernelgen_root,
        knowledge_catalog_path=args.knowledge_catalog_path,
        cross_epoch_knowledge=args.cross_epoch_knowledge,
    )
    claude_config_dir = (
        materialize_claude_runtime_config(args.workspace, args.model)
        if args.runtime == "claude"
        else None
    )

    # Load the Definition from the Server-owned built-in Catalog.
    print(f"\n  Loading definition: {args.definition} ...")
    definition = load_definition(args.definition, args.catalog_name)
    print(f"  ✅ Definition loaded: {definition['name']} ({definition['op_type']})")

    # Set env for the authoritative eval/finalization tools.
    os.environ["FIB_EVAL_SERVER"] = args.eval_server

    # Runtime factory
    def make_rt(path):
        return create_cli_runtime(
            args.runtime,
            workspace=path,
            model=args.model,
            base_url=args.base_url,
            auth_token=args.auth_token,
            claude_config_dir=claude_config_dir,
            timeout=args.timeout,
            idle_timeout=args.timeout // 2,
            mirror_to_console=False,
        )

    # Build workflow input
    inp = {
        **reference,
        "definition": definition,
        "target_hardware": args.target_hardware,
        "implementation_language": args.language,
        "n_parallel": args.n_parallel,
        "n_epoch": args.n_epoch,
        "cross_epoch_knowledge": args.cross_epoch_knowledge,
        "start_epoch": args.start_epoch,
        "start_mode": start_mode,
        "initial_seed_code": (
            args.seed_code_path.read_text(encoding="utf-8")
            if args.seed_code_path is not None
            else ""
        ),
        "initial_seed_is_validated_baseline": (
            args.seed_code_path is not None
        ),
        "knowledge_run_id": args.knowledge_run_id,
        "eval_server_url": args.eval_server,
        "catalog_name": args.catalog_name,
        "profile_enabled": args.profile,
        "timeout": args.timeout,
        "evaluation_contract": {
            "tolerance_mode": args.eval_tolerance_mode,
            "atol": args.eval_atol,
            "rtol": args.eval_rtol,
            "required_matched_ratio": args.eval_required_matched_ratio,
            "consider_reduced_precision": not args.no_reduced_precision,
        },
        "early_stop_rounds": args.early_stop_rounds,
        "min_rounds": args.min_rounds,
        "max_round": args.max_round,
    }

    # Run
    print(f"\n  Running KernelGenWorkflow ...")
    print()
    knowledge_config = None
    if args.knowledge_catalog_path is not None:
        knowledge_config = KnowledgeConfig(
            mode=effective_knowledge_mode,
            catalog_root=args.knowledge_catalog_path,
            run_archive_root=args.knowledge_run_archive_path,
            derived_root=args.knowledge_derived_path,
            reviewer_mode=args.knowledge_reviewer_mode,
        )
    wf = KernelGenWorkflow(
        cwd=str(args.workspace),
        runtime_factory=make_rt,
        knowledge_config=knowledge_config,
    )
    if args.finalize_epoch is not None:
        out = wf.finalize_completed_epoch(inp, args.finalize_epoch)
    else:
        out = wf.run(inp)

    # Results
    print_results(out, args.workspace)

    # Exit code
    sys.exit(0 if out.status == "PASSED" else 1)


def main(argv=None):
    from kernelgen.cli.optimization import run_optimization_example
    return run_optimization_example("kernelgen", argv)


if __name__ == "__main__":
    raise SystemExit(main())

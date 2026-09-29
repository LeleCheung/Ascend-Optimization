#!/usr/bin/env python
"""Run extract_opt — extract a PyTorch operator and optimize every definition.

The workflow has no analyzer, multi-epoch loop, or cross-definition KB writeback.
Each extracted definition runs its own Coder → Distiller → Merge lifecycle.

Usage:
    cd /data/akg_kernel_bench_lite
    # Default mode (uses dummy data from kernelgen/tmp/{operator}/):
    PYTHONPATH=. python3 kernelgen/examples/extract_opt/run_example.py \
        --operator add --eval-server http://localhost:8000

    # With all options:
    PYTHONPATH=. python3 kernelgen/examples/extract_opt/run_example.py \
        --operator softmax \
        --eval-server http://localhost:8000 \
        --target-hardware A100 \
        --model deepseek-v4-pro[1m] \
        --workspace /tmp/extract_opt_softmax \
        --early-stop-rounds 3 --min-rounds 2
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kernelgen.framework import copy_claude_directory
from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.workflows.extract_opt import ExtractOptWorkflow


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EVAL_SERVER = "http://localhost:8000"
DEFAULT_MODEL = os.environ.get("MODEL", "deepseek-v4-pro[1m]")
DEFAULT_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL")
DEFAULT_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN")
KERNELGEN_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="extract_opt: extract PyTorch definitions + parallel kernel optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using dummy data (pre-extracted add operator)
  PYTHONPATH=. python3 kernelgen/examples/extract_opt/run_example.py \\
    --operator add --eval-server http://localhost:8000

  # With custom settings
  PYTHONPATH=. python3 kernelgen/examples/extract_opt/run_example.py \\
    --operator gelu --eval-server http://localhost:8000 --target-hardware A100
        """,
    )
    parser.add_argument("--operator", "-o", type=str, required=True,
                        help="PyTorch operator name (e.g. add, gelu, softmax, matmul)")
    parser.add_argument("--workspace", "-w", type=Path, default=None,
                        help="Workspace directory (default: kernelgen/runs/{operator})")
    parser.add_argument("--eval-server", type=str, default=DEFAULT_EVAL_SERVER,
                        help=f"Eval server URL (default: {DEFAULT_EVAL_SERVER})")
    parser.add_argument("--target-hardware", type=str, default="",
                        help="Target hardware (e.g. A100, Ascend910B)")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help="API base URL")
    parser.add_argument("--auth-token", type=str, default=DEFAULT_AUTH_TOKEN,
                        help="API auth token")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Per-agent timeout in seconds (default: 1200)")
    parser.add_argument("--early-stop-rounds", type=int, default=3,
                        help="Rounds without improvement before stopping (default: 3)")
    parser.add_argument("--min-rounds", type=int, default=2,
                        help="Minimum rounds before early stop (default: 2)")
    parser.add_argument(
        "--max-round",
        type=int,
        default=15,
        help="Maximum total measured rounds before hard stop (default: 15)",
    )
    parser.add_argument("--clean", action="store_true",
                        help="Clean workspace before starting")
    args = parser.parse_args()

    # Resolve workspace — default to kernelgen/runs/{operator}
    if args.workspace is None:
        args.workspace = KERNELGEN_ROOT / "runs" / args.operator

    print("=" * 70)
    print(f"  ExtractOpt: {args.operator}")
    print(f"  Eval server: {args.eval_server}")
    print(f"  Model: {args.model}")
    print(f"  Target hardware: {args.target_hardware or '(auto)'}")
    print(f"  Workspace: {args.workspace}")
    print("=" * 70)

    # Clean workspace
    if args.clean and args.workspace.exists():
        shutil.rmtree(args.workspace)
        print("  🗑️  Workspace cleaned")

    # Setup workspace
    args.workspace.mkdir(parents=True, exist_ok=True)

    # Materialize provider settings, roles, and skills.
    claude_src = KERNELGEN_ROOT / ".claude"
    copy_claude_directory(claude_src, args.workspace / ".claude")

    # Copy dummy extractor data to workspace
    # (remove this block when switching to real PyTorchV5ExtractorAgent)
    dummy_src = KERNELGEN_ROOT / "tmp" / args.operator
    if dummy_src.exists():
        dummy_dst = args.workspace / "tmp" / args.operator
        shutil.copytree(dummy_src, dummy_dst, dirs_exist_ok=True)
        print(f"  ✅ Dummy data copied from {dummy_src}")

    # Set env for tools
    os.environ["FIB_EVAL_SERVER"] = args.eval_server
    os.environ["FIB_TRACE_ROOT"] = str(KERNELGEN_ROOT / "trace_data")
    os.environ["FIB_TRACE_SET_KEY"] = ""

    # Runtime factory
    def make_rt(path):
        return ClaudeRuntime(
            workspace=path,
            model=args.model,
            base_url=args.base_url,
            auth_token=args.auth_token,
            timeout=args.timeout,
            idle_timeout=args.timeout // 2,
        )

    # Build workflow input
    wf_input = {
        "operator": args.operator,
        "target_hardware": args.target_hardware,
        "eval_server_url": args.eval_server,
        "trace_root": str(KERNELGEN_ROOT / "trace_data"),
        "trace_set_key": "",
        "early_stop_rounds": args.early_stop_rounds,
        "min_rounds": args.min_rounds,
        "max_round": args.max_round,
    }

    # Run workflow
    print(f"\n  Running ExtractOptWorkflow ...")
    print()
    wf = ExtractOptWorkflow(cwd=str(args.workspace), runtime_factory=make_rt)
    out = wf.run(wf_input)

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Operator: {out.operator}")
    print(f"  Definitions extracted: {out.num_definitions}")
    print()
    for r in out.results:
        status_icon = "✅" if r.status == "PASSED" else "❌"
        geo_str = f"{r.best_geo_mean:.3f}x" if r.best_geo_mean else "N/A"
        print(f"  {status_icon} {r.definition_name}: {r.status} (geo_mean: {geo_str})")
        if r.summary:
            print(f"     {r.summary[:100]}")
        print(f"     workspace: {r.workspace}")
    print()
    print("=" * 70)

    # Exit code
    any_passed = any(r.status == "PASSED" for r in out.results)
    sys.exit(0 if any_passed else 1)


if __name__ == "__main__":
    main()

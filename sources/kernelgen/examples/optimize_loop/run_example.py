#!/usr/bin/env python
"""Run optimize_loop example — iterative kernel optimization.

Each operator gets its own OptimizeWorkflow: Python controls the iteration loop,
calling OptimizeAgent once per round. Between rounds, Python saves versions and
updates PERFORMANCE.md so the next round's agent sees its history.

Multiple operators run in parallel via run_parallel(OptimizeWorkflow, ...).

Usage:
    cd /data/akg_kernel_bench_lite
    PYTHONPATH=. python3 kernelgen/examples/optimize_loop/run_example.py --input ops.jsonl
    PYTHONPATH=. python3 kernelgen/examples/optimize_loop/run_example.py \\
        --input ops.jsonl --max-iters 10 --target-speedup 1.5

Input JSONL format (one JSON object per line):
    {"operator": "softmax"}
    {"operator": "layernorm"}
"""

import argparse
import json
import sys
from pathlib import Path

KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KERNELGEN_ROOT.parent))

from kernelgen.framework.parallel import IsolatedDirectory, run_parallel
from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.workflows.optimize import OptimizeWorkflow


def load_jsonl(path: Path) -> list[dict]:
    """Load operator inputs from a JSONL file."""
    inputs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                op = data.get("operator", data.get("OPERATOR", ""))
                if op:
                    inputs.append({"operator": op})
    return inputs


def main():
    parser = argparse.ArgumentParser(
        description="optimize_loop: iterative kernel optimization via kernelgen framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  PYTHONPATH=. python3 kernelgen/examples/optimize_loop/run_example.py --input ops.jsonl
  PYTHONPATH=. python3 kernelgen/examples/optimize_loop/run_example.py \\
    --input ops.jsonl --max-iters 10 --target-speedup 1.5
        """,
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="JSONL file with operator names")
    parser.add_argument("--workspace", "-w", type=Path, default=Path("/tmp/optimize_workspace"),
                        help="Workspace base directory (default: /tmp/optimize_workspace)")
    parser.add_argument("--model", "-m", type=str, default="inherit",
                        help="LLM model name")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API base URL")
    parser.add_argument("--auth-token", type=str, default=None,
                        help="API auth token")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Max parallel operators (default: 4)")
    parser.add_argument("--max-iters", type=int, default=20,
                        help="Max optimization iterations per operator (default: 20)")
    parser.add_argument("--target-speedup", type=float, default=0.8,
                        help="Stop when speedup >= target (default: 0.8)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-iteration timeout in seconds (default: 3600)")
    args = parser.parse_args()

    raw_inputs = load_jsonl(args.input)
    if not raw_inputs:
        print("Error: no operators found in input file", file=sys.stderr)
        sys.exit(1)

    # Build workflow inputs: each operator gets max_iters + target_speedup
    workflow_inputs = [
        {
            "operator": inp["operator"],
            "max_iters": args.max_iters,
            "target_speedup": args.target_speedup,
        }
        for inp in raw_inputs
    ]

    ops = [inp["operator"] for inp in raw_inputs]
    print(f"Operators: {ops}")
    print(f"Workspace: {args.workspace}")
    print(f"Model: {args.model}")
    print(f"Max workers: {args.max_workers}")
    print(f"Max iters: {args.max_iters}, Target speedup: {args.target_speedup}")
    print()

    args.workspace.mkdir(parents=True, exist_ok=True)
    ws = IsolatedDirectory(
        base=args.workspace / "operators",
        claude_source=KERNELGEN_ROOT / ".claude",
    )

    def make_rt(path):
        return ClaudeRuntime(
            workspace=path,
            model=args.model,
            base_url=args.base_url,
            auth_token=args.auth_token,
            timeout=args.timeout,
            idle_timeout=args.timeout // 2,
        )

    # run_parallel fans out OptimizeWorkflow — one per operator, each running
    # its own iterative loop internally.
    results = run_parallel(
        OptimizeWorkflow,
        workflow_inputs,
        workspace=ws,
        runtime_factory=make_rt,
        max_workers=args.max_workers,
        task_name="operator",
    )

    print()
    print("=" * 60)
    reached = sum(1 for r, _ in results if r.target_reached)
    print(f"RESULTS: {reached}/{len(results)} reached target speedup")
    print("=" * 60)
    for result, ws_name in results:
        out = result.model_dump()
        out["workspace"] = ws_name
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

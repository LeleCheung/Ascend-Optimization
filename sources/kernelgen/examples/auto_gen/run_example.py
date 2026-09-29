#!/usr/bin/env python
"""Run auto_gen example — parallel FlagGems operator generation.

Each operator gets its own Claude Code agent in an isolated workspace.
No iteration — one agent does the complete job (implement → register → test → benchmark).

Usage:
    cd /data/akg_kernel_bench_lite
    PYTHONPATH=. python3 kernelgen/examples/auto_gen/run_example.py --input ops.jsonl
    PYTHONPATH=. python3 kernelgen/examples/auto_gen/run_example.py --input ops.jsonl --max-workers 8

Input JSONL format (one JSON object per line):
    {"operator": "relu"}
    {"operator": "gelu"}
    {"operator": "silu"}
"""

import argparse
import json
import sys
from pathlib import Path

KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KERNELGEN_ROOT.parent))

from kernelgen.agents.auto_gen import AutoGenAgent
from kernelgen.framework.parallel import IsolatedDirectory, run_parallel
from kernelgen.framework.runtime.claude import ClaudeRuntime


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
        description="auto_gen: generate FlagGems operators via kernelgen framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  PYTHONPATH=. python3 kernelgen/examples/auto_gen/run_example.py --input ops.jsonl
  PYTHONPATH=. python3 kernelgen/examples/auto_gen/run_example.py \\
    --input ops.jsonl --workspace /tmp/auto_gen --model deepseek-v4-pro[1m]
        """,
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="JSONL file with operator names")
    parser.add_argument("--workspace", "-w", type=Path, default=Path("/tmp/auto_gen_workspace"),
                        help="Workspace base directory (default: /tmp/auto_gen_workspace)")
    parser.add_argument("--model", "-m", type=str, default="inherit",
                        help="LLM model name (default: inherit from env)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API base URL")
    parser.add_argument("--auth-token", type=str, default=None,
                        help="API auth token")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Max parallel agents (default: 4)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-agent timeout in seconds (default: 1800)")
    args = parser.parse_args()

    inputs = load_jsonl(args.input)
    if not inputs:
        print("Error: no operators found in input file", file=sys.stderr)
        sys.exit(1)

    ops = [inp["operator"] for inp in inputs]
    print(f"Operators: {ops}")
    print(f"Workspace: {args.workspace}")
    print(f"Model: {args.model}")
    print(f"Max workers: {args.max_workers}")
    print()

    args.workspace.mkdir(parents=True, exist_ok=True)
    ws = IsolatedDirectory(
        base=args.workspace / "agents",
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

    results = run_parallel(
        AutoGenAgent,
        inputs,
        workspace=ws,
        runtime_factory=make_rt,
        max_workers=args.max_workers,
        task_name="operator",
    )

    print()
    print("=" * 60)
    passed = sum(1 for r, _ in results if r.status == "success")
    failed = len(results) - passed
    print(f"RESULTS: {passed} passed / {failed} failed / {len(results)} total")
    print("=" * 60)
    for result, ws_name in results:
        print(json.dumps(result.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    main()

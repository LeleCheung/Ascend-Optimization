#!/usr/bin/env python3
"""Batch TestWriter: generate FlagGems correctness + benchmark tests for a list
of torch.ops.aten operators.

Usage:
    cd /share-evpfs/tj/workspace/code-integration/kernelgen
    PYTHONPATH=.:/share-evpfs/tj/workspace/code-integration/FlagGems/src \
        python3 -u examples/test_writer/run_example.py \
        --operators-file operators.txt \
        --flaggems-dir /share-evpfs/tj/workspace/code-integration/FlagGems \
        --max-workers 4
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kernelgen.framework.agent_roles import materialize_agent_role
from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.framework.parallel import Directory, run_parallel
from kernelgen.agents.test_writer import TestWriterAgent


def main():
    parser = argparse.ArgumentParser(description="Generate FlagGems tests for aten operators")
    parser.add_argument("--operators", "-n", nargs="+", default=[],
                        help="Operator names to generate tests for")
    parser.add_argument("--operators-file", "-f", type=Path, default=None,
                        help="File with operator names (one per line, # comments)")
    parser.add_argument("--flaggems-dir", type=Path,
                        default=os.environ.get("FLAGGEMS_DIR",
                            "/share-evpfs/tj/workspace/code-integration/FlagGems"),
                        help="FlagGems checkout root")
    parser.add_argument("--max-verify-retries", type=int, default=5)
    parser.add_argument("--workspace", "-w", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--model", default=os.environ.get("MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("ANTHROPIC_BASE_URL"))
    parser.add_argument("--auth-token", default=os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    args = parser.parse_args()

    operators = list(args.operators)
    if args.operators_file:
        operators += [
            line.strip() for line in args.operators_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if not operators:
        parser.error("No operators specified. Use --operators or --operators-file")

    flaggems_dir = args.flaggems_dir.resolve()
    assert flaggems_dir.exists(), f"flaggems_dir not found: {flaggems_dir}"

    KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
    workspace_root = args.workspace or KERNELGEN_ROOT / "runs" / "test_writer"
    workspace_root.mkdir(parents=True, exist_ok=True)

    agents_src = KERNELGEN_ROOT / ".kernelgen" / "agents" / "kernel-test-writer.md"

    def make_rt(path):
        if agents_src.exists():
            materialize_agent_role(agents_src, Path(path))
        return ClaudeRuntime(
            workspace=path,
            model=args.model,
            base_url=args.base_url,
            auth_token=args.auth_token,
            timeout=args.timeout,
            idle_timeout=args.timeout // 2,
        )

    inputs = [{
        "operator": op,
        "flaggems_dir": str(flaggems_dir),
        "max_verify_retries": args.max_verify_retries,
    } for op in operators]

    print(f"\nTestWriter batch: {len(operators)} operators")
    print(f"  flaggems_dir: {flaggems_dir}")
    print(f"  max_verify_retries: {args.max_verify_retries}")
    print(f"  max_workers: {args.max_workers}\n")

    results = run_parallel(
        TestWriterAgent,
        inputs,
        workspace=Directory(base=workspace_root),
        runtime_factory=make_rt,
        max_workers=args.max_workers,
        task_name="operator",
    )

    print(f"\n{'='*60}")
    for output, ws_name in results:
        for r in output.results if hasattr(output, "results") else [output]:
            icon = {"WRITTEN": "✅", "SKIPPED_EXISTS": "⏭️", "FAILED": "❌"}.get(r.status, "❓")
            print(f"  {icon} {r.operator:20} {r.status:15} {r.summary[:80]}")
            for f in r.files_written:
                print(f"        {f}")
    print(f"{'='*60}")

    written = [r for out, _ in results for r in (out.results if hasattr(out, "results") else [out]) if r.status == "WRITTEN"]
    sys.exit(0 if written else 1)


if __name__ == "__main__":
    main()

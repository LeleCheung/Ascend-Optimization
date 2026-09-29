#!/usr/bin/env python3
"""Batch PyTorch operator extraction with parallel execution and server verification.

Extracts operators from PyTorch native_functions.yaml, generates v5.1 definitions
and workloads, and verifies them against kernelgen_server.

Usage:
    cd /share-evpfs/tj/workspace/code-integration/kernelgen
    python3 -u examples/pytorch_extract/run_example.py \
        --operators gelu relu softmax matmul \
        --server-url http://localhost:8001 \
        --max-workers 10

    # Or from a file (one operator per line):
    python3 -u examples/pytorch_extract/run_example.py \
        --operators-file operators.txt \
        --server-url http://localhost:8001
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kernelgen.framework.agent_roles import materialize_agent_role
from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.framework.parallel import Directory, run_parallel
from kernelgen.agents.extractor.pytorch import (
    PyTorchV5ExtractorAgent,
    OperatorNotFoundError,
    write_catalog,
)


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract PyTorch operators to v5.1 catalog",
    )
    parser.add_argument(
        "--operators", "-n", nargs="+", default=[],
        help="Operator names to extract (repeat or space-separated)",
    )
    parser.add_argument(
        "--operators-file", "-f", type=Path, default=None,
        help="File with operator names (one per line)",
    )
    parser.add_argument("--server-url", default="http://localhost:8001",
                        help="kernelgen_server URL for verification (empty to skip)")
    parser.add_argument("--workspace", "-w", type=Path, default=None,
                        help="Output workspace directory")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--model", default=os.environ.get("MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("ANTHROPIC_BASE_URL"))
    parser.add_argument("--auth-token", default=os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    parser.add_argument("--check-only", action="store_true",
                        help="Only check which operators exist in aten, don't extract")
    args = parser.parse_args()

    # Collect operators
    operators = list(args.operators)
    if args.operators_file:
        operators += [
            line.strip() for line in args.operators_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if not operators:
        parser.error("No operators specified. Use --operators or --operators-file")

    # Import check functions
    from kernelgen.agents.extractor.pytorch import _extract_native_entries, _check_aten_op

    # Pre-check all operators
    valid = []
    invalid = []
    for op in operators:
        native = bool(_extract_native_entries(op))
        aten = _check_aten_op(op)
        if native or aten:
            valid.append(op)
        else:
            invalid.append(op)

    print(f"\nOperator check: {len(valid)} valid, {len(invalid)} not in aten")
    if invalid:
        print(f"  Skipping: {invalid}")
    if args.check_only:
        print(f"\n  Valid operators:")
        for op in valid:
            print(f"    {op}")
        sys.exit(0 if not invalid else 1)

    if not valid:
        print("No valid operators to extract.")
        sys.exit(1)

    # Setup workspace
    KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
    workspace_root = args.workspace or KERNELGEN_ROOT / "runs" / "pytorch_extract"
    workspace_root.mkdir(parents=True, exist_ok=True)
    agents_src = KERNELGEN_ROOT / ".kernelgen" / "agents" / "kernel-pytorch-extractor.md"

    def make_rt(path):
        materialize_agent_role(agents_src, Path(path))
        return ClaudeRuntime(
            workspace=path,
            model=args.model,
            base_url=args.base_url,
            auth_token=args.auth_token,
            timeout=args.timeout,
            idle_timeout=args.timeout // 2,
        )

    inputs = [{"operator": op, "server_url": args.server_url} for op in valid]
    workspace = Directory(base=workspace_root)

    print(f"\nExtracting {len(valid)} operators (max_workers={args.max_workers})...")
    print(f"  Workspace: {workspace_root}")
    print(f"  Server: {args.server_url or 'disabled'}")
    print(f"  Model: {args.model}\n")

    try:
        results = run_parallel(
            PyTorchV5ExtractorAgent,
            inputs,
            workspace=workspace,
            runtime_factory=make_rt,
            max_workers=args.max_workers,
            task_name="operator",
        )
    except Exception as e:
        print(f"\n❌ run_parallel failed: {type(e).__name__}: {e}")
        sys.exit(1)

    # Collect results
    all_results = []
    passed = 0
    partial = 0
    failed = 0
    for output, ws_name in results:
        for r in output.results:
            all_results.append(r)
            v = r.verification_status or "not_verified"
            if v == "PASSED":
                passed += 1
            elif "PASS" in v:
                partial += 1
            else:
                failed += 1
            print(f"  [{v:15}] {r.definition.name} ({r.group}): "
                  f"{len(r.correctness_workloads)}c + {len(r.timing_workloads)}t",
                  flush=True)

    # Write catalog
    if all_results:
        catalog_root = workspace_root / "catalog"
        write_catalog(all_results, catalog_root)
        print(f"\nCatalog: {catalog_root}")

    print(f"\n{'='*60}")
    print(f"  Extracted: {len(all_results)}/{len(valid)}")
    print(f"  PASSED={passed}  PARTIAL_PASS={partial}  FAILED/ERROR={failed}")
    if invalid:
        print(f"  Skipped (not in aten): {len(invalid)}")
    print(f"{'='*60}")
    sys.exit(0)


if __name__ == "__main__":
    main()

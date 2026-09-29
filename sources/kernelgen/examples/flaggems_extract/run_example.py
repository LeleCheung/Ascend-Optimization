#!/usr/bin/env python
"""Extract a FlagGems operator into a native-suite adapter registry record.

The V6 adapter path is deterministic and does not invoke a model.

Usage:
    cd /path/to/kernelgen
    python3 examples/flaggems_extract/run_example.py \\
        --operator addmm_

    # Custom FlagGems repo + output location:
    python3 examples/flaggems_extract/run_example.py \\
        --operator addmm_ --flaggems-repo /workspace/FlagGems \\
        --adapter-root /tmp/flaggems-adapter-definitions
"""

import argparse
import json
from pathlib import Path

from kernelgen.framework.agent_roles import materialize_agent_role
from kernelgen.workflows.flaggems_adapter_extract import (
    DEFAULT_ADAPTER_ROOT,
    FlagGemsAdapterExtractWorkflow,
)


KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLAGGEMS = str(KERNELGEN_ROOT / "third_party" / "FlagGems")


def prepare_native_agent(workspace: Path) -> Path:
    """Legacy translated-catalog helper; the V6 adapter path does not call it."""
    source = (
        KERNELGEN_ROOT
        / ".kernelgen"
        / "agents"
        / "kernel-flaggems-extractor.md"
    )
    if not source.is_file():
        raise FileNotFoundError(f"missing native extractor agent: {source}")
    materialize_agent_role(source, workspace)
    return workspace / ".claude" / "agents" / source.name


def main():
    parser = argparse.ArgumentParser(
        description="Extract a FlagGems native-suite adapter registry record",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--operator", "-o", type=str, required=True,
                        help="FlagGems operator name (V6 POC: addmm_)")
    parser.add_argument("--flaggems-repo", type=str, default=DEFAULT_FLAGGEMS,
                        help=f"FlagGems repo path (default: {DEFAULT_FLAGGEMS})")
    parser.add_argument(
        "--adapter-root",
        type=str,
        default=DEFAULT_ADAPTER_ROOT,
        help=f"Output adapter registry directory (default: {DEFAULT_ADAPTER_ROOT})",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"  FlagGems Extract: {args.operator}")
    print(f"  FlagGems repo: {args.flaggems_repo}")
    print(f"  adapter root: {args.adapter_root}")
    print("=" * 70, flush=True)

    wf = FlagGemsAdapterExtractWorkflow()

    result = wf.run({
        "operator": args.operator,
        "flaggems_repo": args.flaggems_repo,
        "adapter_root": args.adapter_root,
    })

    print()
    print("=" * 70)
    print("RESULT:")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print("=" * 70)

    # Show what was written
    print(f"\nWrote: {result.adapter_path}")


if __name__ == "__main__":
    main()

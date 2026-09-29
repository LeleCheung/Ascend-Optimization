"""Foreground Catalog extraction command; optimization stays under kg run."""

import json
import os
from pathlib import Path
import sys
import uuid

from kernelgen.cli import api
from kernelgen.framework.local_state import state_home
from kernelgen.framework.runtime import CLI_RUNTIME_NAMES
from kernelgen.framework.run_control import RunCancelled
from kernelgen.workflows.catalog_extract_review import CatalogExtractionBlocked, CatalogReviewRequired


def add_extract_parser(subparsers):
    parser = subparsers.add_parser("extract", help="extract and review a Native Catalog (foreground)",
                                   allow_abbrev=False)
    parser.add_argument("--operator", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--flaggems-repo", type=Path, help="existing FlagGems/FlagGems-vllm checkout")
    source.add_argument("--pr-url", help="supported FlagGems/FlagGems-vllm GitHub PR URL")
    parser.add_argument("--workspace", "-w", type=Path)
    parser.add_argument("--runtime", choices=CLI_RUNTIME_NAMES,
                        default=os.environ.get("KERNELGEN_RUNTIME", "claude"))
    parser.add_argument("--model", "-m")
    parser.add_argument("--timeout", type=int, default=900, help="timeout for each model invocation, seconds")
    parser.add_argument("--max-review-rounds", type=int, default=3)
    parser.set_defaults(handler=run_extract_command)


def run_extract_command(args):
    workspace = args.workspace or state_home() / "extracts" / uuid.uuid4().hex[:12]
    inp = {"operator": args.operator, "pr_url": args.pr_url,
           "flaggems_repo": str(args.flaggems_repo) if args.flaggems_repo else None,
           "max_review_rounds": args.max_review_rounds}
    try:
        result = api.extract_catalog(inp, workspace=workspace, runtime=args.runtime,
                                     model=args.model, timeout=args.timeout)
    except CatalogExtractionBlocked as exc:
        print(f"kg: {exc}", file=sys.stderr)
        return 1
    except CatalogReviewRequired as exc:
        print(f"kg: extraction requires review changes: {exc}", file=sys.stderr)
        return 1
    except RunCancelled:
        print(f"kg: extraction cancelled; evidence: {workspace}", file=sys.stderr)
        return 130
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

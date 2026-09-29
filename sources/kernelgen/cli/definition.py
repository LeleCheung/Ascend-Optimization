"""Export a Gems adapter Definition; no Native extraction or model execution."""

import json
from pathlib import Path
import uuid

from kernelgen.cli import api
from kernelgen.framework.local_state import state_home


def add_definition_parser(subparsers):
    parser = subparsers.add_parser(
        "definition", help="export a Gems adapter Definition from existing pytest (foreground)",
        allow_abbrev=False,
    )
    parser.add_argument("--flaggems-repo", type=Path, required=True, help="clean, committed Gems checkout")
    parser.add_argument("--pytest-path", type=Path, required=True, help="correctness pytest path within the checkout")
    parser.add_argument("--operator", help="operator name for a shared test file; otherwise inferred from the filename")
    parser.add_argument("--workspace", "-w", type=Path, help="output directory outside the Gems checkout")
    parser.set_defaults(handler=run_definition_command)


def run_definition_command(args):
    workspace = args.workspace or state_home() / "definitions" / uuid.uuid4().hex[:12]
    result = api.export_gems_definition(
        {"flaggems_repo": args.flaggems_repo, "pytest_path": args.pytest_path, "operator": args.operator},
        workspace=workspace,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

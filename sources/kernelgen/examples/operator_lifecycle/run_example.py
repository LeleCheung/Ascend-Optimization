"""Python companion to kg run --mode lifecycle, with status/cancel utilities."""

import argparse
import json
from pathlib import Path

from kernelgen.cli.lifecycle import add_lifecycle_run_arguments, run_lifecycle_command
from kernelgen.workflows.operator_development import (
    campaign_status, cancel_lifecycle,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--operators", nargs="+", required=True)
    add_lifecycle_run_arguments(run)
    status = commands.add_parser("status")
    status.add_argument("workspace", type=Path)
    cancel = commands.add_parser("cancel")
    cancel.add_argument("workspace", type=Path, help="One operator's workspace, not the campaign root")
    cancel.add_argument("--reason", default="operator lifecycle cancellation")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run_lifecycle_command(args, operators=args.operators, workspace=args.workspace)
        elif args.command == "status":
            result = campaign_status(args.workspace)
        else:
            result = cancel_lifecycle(args.workspace, args.reason)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"operator_lifecycle: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

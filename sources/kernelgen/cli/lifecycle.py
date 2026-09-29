"""Shared foreground dummy lifecycle adapter for kg and the Python example."""

import json

from kernelgen.cli.models import RunMode
from kernelgen.workflows.operator_development import STAGES, DummyWorkflowCall, run_campaign


def add_lifecycle_run_arguments(parser) -> tuple[str, ...]:
    """Declare options once; return their destinations for cross-mode validation."""
    start = len(parser._actions)
    group = parser.add_argument_group("lifecycle (foreground dummy only)")
    group.add_argument("--dummy", action="store_true", default=None,
                       help="required for lifecycle; never call real agents or devices")
    group.add_argument("--resume", action="store_true", default=None,
                       help="resume the same lifecycle campaign and execution plan")
    group.add_argument("--stages", nargs="+", choices=STAGES)
    group.add_argument("--optimize-mode", choices=[mode.value for mode in RunMode])
    for option in ("fail", "wait", "cancel"):
        group.add_argument(f"--{option}-stage", choices=STAGES,
                           help=f"inject a dummy {option} at the selected stage")
    return tuple(action.dest for action in parser._actions[start:])


def run_lifecycle_command(args, *, operators, workspace) -> int:
    if not args.dummy:
        raise ValueError("lifecycle currently requires --dummy; real agents are not connected")
    result = run_campaign(
        workspace, operators, dummy=True, resume=bool(args.resume), stages=args.stages,
        optimize={"mode": args.optimize_mode} if args.optimize_mode is not None else None,
        workflow_call=DummyWorkflowCall(fail_at=args.fail_stage, wait_at=args.wait_stage,
                                     cancel_at=args.cancel_stage),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    states = {item["state"] for item in result["operators"]}
    if "CANCELLED" in states:
        return 130
    return 0 if states == {"SUCCEEDED"} else 1

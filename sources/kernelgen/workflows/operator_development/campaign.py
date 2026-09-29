"""An immutable operator index, not another lifecycle state store or scheduler."""

import json
from pathlib import Path

from kernelgen.framework.local_state import file_lock
from kernelgen.data._atomic import atomic_write_json

from .contracts import STAGES, OperatorDevelopmentInput
from .workflow import OperatorDevelopmentWorkflow, lifecycle_status


INDEX = ".kernelgen/operator-lifecycle-campaign.json"


def run_campaign(workspace, operators, *, stages=None, optimize=None,
                 dummy=False, resume=False, workflow_call=None) -> dict:
    if not dummy:
        raise NotImplementedError("campaign only supports explicit dummy=True")
    if not operators or len(set(operators)) != len(operators):
        raise ValueError("operators must be nonempty and unique")
    requests = [OperatorDevelopmentInput(operator=operator, dummy=True,
                              stages=STAGES if stages is None else stages, optimize=optimize)
                for operator in operators]
    root = Path(workspace).expanduser().resolve()
    index = root / INDEX
    if resume and not index.exists():
        raise FileNotFoundError("no lifecycle campaign exists to resume")
    if not index.exists() and root.exists() and any(root.iterdir()):
        raise ValueError("new campaign requires an empty workspace")
    children = {operator: f"operators/{operator}" for operator in operators}
    plan = {"schema_version": "1.1", "simulated": True, "operators": children,
            "execution": requests[0].stage_plan()}
    with file_lock(root / ".kernelgen/submission.lock"):
        if index.exists():
            if not resume:
                raise ValueError("campaign already exists; use explicit resume")
            existing = json.loads(index.read_text())
            if existing.get("operators") != children or list(existing["operators"]) != list(children):
                raise ValueError("campaign operator list changed; choose a new campaign")
            if existing != plan:
                raise ValueError("campaign execution plan changed; choose a new campaign")
        else:
            atomic_write_json(index, plan)
    for request in requests:
        child = root / children[request.operator]
        if child.resolve() != child:
            raise ValueError(f"operator workspace must not redirect through a symlink: {child}")
        request.resume = resume and (child / ".kernelgen/operator-lifecycle.json").is_file()
        result = OperatorDevelopmentWorkflow(cwd=child, workflow_call=workflow_call).run(request)
        if result.state == "CANCELLED":
            break  # Do not start more operators after a foreground cancellation.
    return campaign_status(root)


def campaign_status(workspace) -> dict:
    root = Path(workspace).expanduser().resolve()
    plan = json.loads((root / INDEX).read_text())
    operators = []
    for operator, relative in plan["operators"].items():
        child = root / relative
        if (child / ".kernelgen/operator-lifecycle.json").is_file():
            operators.append(lifecycle_status(child))
        else:
            operators.append({"operator": operator, "workspace": str(child),
                              "simulated": True, "state": "PENDING", "progress": None})
    return {"schema_version": "2.0", "kind": "operator_lifecycle_campaign",
            "workspace": str(root), "simulated": True, "operators": operators}

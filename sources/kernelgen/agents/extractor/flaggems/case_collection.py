"""Agent-written, workspace-local pytest collection adapters for trusted source."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess

from pydantic import BaseModel, Field

from kernelgen.data._atomic import atomic_write_json
from kernelgen.framework.agent_roles import materialize_agent_role
from kernelgen.framework.base import BaseAgent
from kernelgen.framework.local_state import file_lock
from kernelgen.framework.run_control import RunCancelled, WorkspaceRunControl
from .collection_config import load_extraction_config
from .collection_executor import CollectionTransportError, execute_pytest, snapshot
from .source_inventory import collect_source_inventory
from .v62_agent import load_flaggems_timing_cases


class CollectionInput(BaseModel):
    source_root: str
    operator: str
    source_files: list[str]
    previous_error: str = ""


class CollectionProposal(BaseModel):
    source: str = Field(min_length=1, description="Python module defining collect_cases(source_root, operator), returning the timing case-list report")


class CaseListAgent(BaseAgent):
    name = "gems-case-collector"
    InputModel = CollectionInput
    OutputModel = CollectionProposal

    def postprocess(self, raw, runtime):
        proposal = super().postprocess(raw, runtime)
        try:
            tree = ast.parse(proposal.source)
        except SyntaxError as exc:
            raise ValueError(f"invalid collector Python: {exc}") from exc
        if not any(isinstance(n, ast.FunctionDef) and n.name == "collect_cases" for n in tree.body):
            raise ValueError("collector must define collect_cases(source_root, operator)")
        return proposal


_PYTEST = '''import json
import sys
from pathlib import Path
from collector import collect_cases

def test_collect_cases():
    config = json.loads(Path("request.json").read_text())
    root = Path(config["source_root"]).resolve()
    prefix = str(root) + "/"
    executed = set()
    def trace(frame, event, arg):
        if event == "call" and frame.f_code.co_name != "<module>" and frame.f_code.co_filename.startswith(prefix):
            path = Path(frame.f_code.co_filename).resolve()
            if path.is_relative_to(root):
                executed.add(path.relative_to(root).as_posix())
    sys.setprofile(trace)
    try:
        report = collect_cases(str(root), config["operator"])
    finally:
        sys.setprofile(None)
    assert any(p.startswith("benchmark/") for p in executed), "collector did not execute original benchmark code"
    Path("cases.json").write_text(json.dumps(report))
    Path("executed-source.json").write_text(json.dumps(sorted(executed)))
'''


def prepare_case_list(source_root, operator, workspace, runtime_factory, *, timeout=300) -> Path:
    """Generate and run a collector, then reuse only a verified immutable receipt."""
    source_root, workspace = Path(source_root).resolve(), Path(workspace).resolve()
    if workspace.is_relative_to(source_root):
        raise ValueError("collection workspace must be outside the source checkout")
    workspace.mkdir(parents=True, exist_ok=True)
    with file_lock(workspace / ".collection.lock"):
        _, digest, revision = snapshot(source_root)
        identity = {"source_root": str(source_root), "operator": operator,
                    "source_sha256": digest, "source_revision": revision}
        request = workspace / "request.json"
        if request.exists() and json.loads(request.read_text()) != identity:
            raise ValueError("collection source/operator changed; use a new workspace")
        atomic_write_json(request, identity)
        receipt = workspace / "collection.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            path = workspace / saved["attempt"] / "cases.json"
            if not path.resolve().is_relative_to(workspace) or hashlib.sha256(path.read_bytes()).hexdigest() != saved["sha256"]:
                raise ValueError("collected case list changed")
            if not load_flaggems_timing_cases(path, operator):
                raise ValueError("collected case list is empty")
            return path
        config = load_extraction_config()
        control = WorkspaceRunControl(workspace, source="case_collection")
        control.checkpoint("BEFORE_CASE_COLLECTION")
        inventory = collect_source_inventory(str(source_root), operator)
        error = ""
        for _ in range(2):
            control.checkpoint("BEFORE_CASE_COLLECTION_MODEL")
            number = 1
            while (workspace / f"{number:02d}").exists():
                number += 1
            attempt = workspace / f"{number:02d}"
            attempt.mkdir()
            agent_workspace = attempt / "agent"
            agent = CaseListAgent()
            runtime = runtime_factory(str(agent_workspace))
            materialize_agent_role(agent._native_definition_path(), getattr(runtime, "workspace", None) or agent_workspace)
            proposal = agent.run({**{k: identity[k] for k in ("source_root", "operator")},
                                  "source_files": [str(p) for p in (inventory.implementation_file, *inventory.test_files, *inventory.benchmark_files, source_root / "benchmark/base.py", source_root / "benchmark/conftest.py")],
                                  "previous_error": error}, runtime)
            (attempt / "collector.py").write_text(proposal.source)
            (attempt / "test_collect_cases.py").write_text(_PYTEST)
            atomic_write_json(attempt / "request.json", identity)
            try:
                execution = execute_pytest(attempt, timeout, config,
                                           checkpoint=lambda: control.checkpoint("CASE_COLLECTION"))
                cases = attempt / "cases.json"
                if not load_flaggems_timing_cases(cases, operator):
                    raise ValueError("no timing cases collected; skip is not a successful collection")
            except (RunCancelled, CollectionTransportError):
                raise
            except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
                error = str(exc)
                continue
            finally:
                if snapshot(source_root)[1:] != (digest, revision):
                    raise ValueError("source changed during collection; preserve evidence and use a new workspace")
            atomic_write_json(receipt, {"attempt": attempt.name, "sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
                                       **execution, "target_validation": "NOT_RUN"})
            return cases
        raise RuntimeError(f"case collection failed after two attempts: {error}")

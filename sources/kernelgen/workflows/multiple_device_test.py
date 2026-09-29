"""Retest one frozen Gems candidate against independent target KGS instances."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kernelgen.data._atomic import atomic_write_json
from kernelgen.framework.cancellation import cooperative_sigint
from kernelgen.framework.run_control import RunCancelled, RunState, WorkspaceRunControl
from kernelgen.framework.workflow import Workflow
from kernelgen.framework.workflow_result import WorkflowResult
from kernelgen.tools.kernelgen_server_adapter import get_service_status, tracked_server_operation
from kernelgen.workflows.optimization.artifacts import operator_bundle_source
from kernelgen_client import (
    Catalog,
    BoundEvaluateRequest,
    EvaluationSettings,
    EvaluatorBinding,
    Implementation,
    InspectRequest,
    ReferenceRequest,
)
from kernelgen_client.operator_bundles import GemsDefinitionSource, pack_operator_bundle
from kernelgen_client.protocol.schema import PreflightResult
from kernelgen_client.protocol.version import KERNELGEN_API_VERSION
from kernelgen_client import http as client


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeviceTarget(_Model):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    server_url: str

    @model_validator(mode="after")
    def local_proxy_only(self):
        parsed = urlsplit(self.server_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.port is None or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment):
            raise ValueError("target KGS must be reached through a local loopback URL")
        return self


class MultipleDeviceTestInput(_Model):
    operator: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
    catalog_path: Path
    candidate_path: Path
    targets: list[DeviceTarget] = Field(min_length=1)
    settings: EvaluationSettings = Field(default_factory=EvaluationSettings)
    reference_timeout_seconds: int = Field(default=1500, gt=0)

    @model_validator(mode="after")
    def distinct_targets(self):
        if len({target.name for target in self.targets}) != len(self.targets):
            raise ValueError("target names must be unique")
        if len({target.server_url.rstrip("/") for target in self.targets}) != len(self.targets):
            raise ValueError("target KGS endpoints must be unique")
        return self


class DeviceTestResult(_Model):
    name: str
    server_url: str
    state: Literal["PASSED", "REFERENCE_FAILED", "PREFLIGHT_FAILED", "EVALUATION_FAILED", "ERROR"]
    phase: str
    backend: str = ""
    device: str = ""
    bundle_id: str = ""
    reference_status: str = ""
    preflight_status: str = ""
    evaluation_status: str = ""
    geo_mean: float | None = None
    num_passed: int = 0
    num_workloads: int = 0
    error: str = ""
    result_path: str = ""


class MultipleDeviceTestOutput(_Model):
    operator: str
    workspace: str
    source_revision: str
    candidate_sha256: str
    bundle_id: str
    completed_targets: int
    passed_targets: int
    results: list[DeviceTestResult]


class MultipleDeviceTestWorkflow(Workflow):
    """Run reference, then the unchanged candidate, once per target."""

    name = "multiple_device_test"
    InputModel = MultipleDeviceTestInput
    OutputModel = WorkflowResult[MultipleDeviceTestOutput]

    def __init__(self, *, cwd="."):
        self.root = Path(cwd).expanduser().resolve()

    def _freeze(self, inp: MultipleDeviceTestInput):
        catalog = Catalog(inp.catalog_path)
        if catalog.evaluator != "flaggems" or catalog.api_version != "v6.0":
            raise ValueError("MultipleDeviceTest requires a v6.0 Gems Definition Catalog")
        catalog.load(inp.operator)
        source = GemsDefinitionSource.model_validate(catalog.manifest.get("definition_source"))
        candidate = inp.candidate_path.expanduser().resolve()
        code = candidate.read_text(encoding="utf-8")
        if not code.strip():
            raise ValueError("candidate code is empty")
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "candidate.py").write_text(code, encoding="utf-8")
        with operator_bundle_source(inp.catalog_path, inp.operator) as directory:
            shutil.copytree(directory, self.root / "bundle")
        with tempfile.TemporaryFile(mode="w+b") as archive:
            digest, _ = pack_operator_bundle(self.root / "bundle", archive)
        plan = {
            "operator": inp.operator,
            "catalog_path": str(inp.catalog_path.expanduser().resolve()),
            "source_revision": source.source_revision,
            "candidate_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "bundle_id": "sha256:" + digest,
            "targets": [target.model_dump(mode="json") for target in inp.targets],
            "settings": inp.settings.model_dump(mode="json"),
            "reference_timeout_seconds": inp.reference_timeout_seconds,
        }
        atomic_write_json(self.root / "plan.json", plan)
        return plan, code

    @staticmethod
    def _ready(status, phase):
        if status.get("api_version") != KERNELGEN_API_VERSION:
            raise ValueError(f"{phase}: incompatible KGS protocol {status.get('api_version')!r}")
        scheduler = status.get("scheduler") or {}
        if scheduler.get("healthy", 0) < 1 or scheduler.get("broken", 0):
            raise ValueError(f"{phase}: no fully healthy device slot")
        capabilities = status.get("capabilities") or {}
        bundle = capabilities.get("operator_bundle_upload") or {}
        reference = capabilities.get("benchmark_reference") or {}
        if (bundle.get("enabled") is not True or bundle.get("evaluation_binding") is not True
                or "flaggems" not in bundle.get("evaluators", [])):
            raise ValueError(f"{phase}: Gems Bundle evaluation binding is unavailable")
        if (reference.get("enabled") is not True or reference.get("level") != "core"
                or "flaggems" not in reference.get("evaluators", [])):
            raise ValueError(f"{phase}: Gems core reference-only is unavailable")

    def _test_target(self, inp, target, plan, code, control):
        root = self.root / "targets" / target.name
        root.mkdir(parents=True, exist_ok=True)
        current = {"phase": "status"}
        values = dict(name=target.name, server_url=target.server_url, state="ERROR", phase="status",
                      bundle_id=plan["bundle_id"], result_path=str(root / "result.json"))

        def save(name, model):
            atomic_write_json(root / name, model.model_dump(mode="json") if hasattr(model, "model_dump") else model)

        try:
            control.checkpoint("BEFORE_TARGET_STATUS")
            status = get_service_status(target.server_url)
            save("status-before.json", status)
            values.update(backend=status.get("backend", ""),
                          device=(status.get("target") or {}).get("device", ""))
            self._ready(status, "status")
            current["phase"] = "upload"
            info = client.upload_operator_bundle(self.root / "bundle", target.server_url)
            save("bundle.json", info)
            if info.bundle_id != plan["bundle_id"]:
                raise ValueError("target installed a different Definition Bundle")
            binding = EvaluatorBinding(bundle_id=info.bundle_id, definition=inp.operator)
            current["phase"] = "inspect"
            manifest = client.inspect(InspectRequest(binding=binding), target.server_url)
            save("inspect.json", manifest)
            if manifest.kind != "flaggems" or not manifest.case_list.cases:
                raise ValueError("target did not inspect a nonempty Gems core benchmark")
            current["phase"] = "reference"
            control.update_progress(state=RunState.RUNNING, stage="REFERENCE")
            with tracked_server_operation(control, status, target.server_url, "reference") as operation_id:
                reference = client.reference(
                    ReferenceRequest(binding=binding, benchmark_fingerprint=manifest.benchmark_fingerprint,
                                     settings={"timeout_seconds": inp.reference_timeout_seconds}),
                    target.server_url, operation_id=operation_id)
            save("reference.json", reference)
            values["reference_status"] = reference.status
            skipped_cases = [record for record in reference.report.get("records", [])
                             if record.get("status") == "SKIP"]
            if reference.status != "PASSED" or skipped_cases:
                failed_cases = [record for record in reference.report.get("records", [])
                                if record.get("status") == "FAILED"]
                reason = ((failed_cases[0].get("failure") or {}).get("message", "")
                          if failed_cases else (f"reference-only skipped {len(skipped_cases)} core record(s)"
                                                if skipped_cases else reference.log))
                values.update(state="REFERENCE_FAILED", phase="reference", error=str(reason)[:500])
            else:
                implementation = Implementation(name="multiple-device-retest", definition=inp.operator,
                                                language="triton", entrypoint="main.py::run",
                                                sources=[{"path": "main.py", "content": code}])
                request = BoundEvaluateRequest(binding=binding, implementation=implementation, settings=inp.settings)
                current["phase"] = "preflight"
                control.update_progress(state=RunState.RUNNING, stage="PREFLIGHT")
                with tracked_server_operation(control, status, target.server_url, "preflight") as operation_id:
                    preflight = PreflightResult.model_validate(client.preflight(
                        request, target.server_url, operation_id=operation_id))
                save("preflight.json", preflight)
                values["preflight_status"] = preflight.status
                if preflight.status != "PASSED":
                    values.update(state="PREFLIGHT_FAILED", phase="preflight",
                                  error=preflight.log.splitlines()[0][:500] if preflight.log else "")
                else:
                    current["phase"] = "evaluate"
                    control.update_progress(state=RunState.RUNNING, stage="EVALUATING")
                    with tracked_server_operation(control, status, target.server_url, "evaluate") as operation_id:
                        evaluation = client.evaluate(request, target.server_url, operation_id=operation_id)
                    save("evaluate.json", evaluation)
                    values.update(evaluation_status=evaluation.status.value, geo_mean=evaluation.geo_mean,
                                  num_passed=evaluation.num_passed, num_workloads=evaluation.num_workloads)
                    passed = (evaluation.status.value == "PASSED" and not evaluation.is_hack
                              and not evaluation.timing_skipped and evaluation.geo_mean is not None
                              and evaluation.num_workloads > 0 and evaluation.num_passed == evaluation.num_workloads)
                    values.update(state="PASSED" if passed else "EVALUATION_FAILED", phase="evaluate",
                                  error=evaluation.log.splitlines()[0][:500] if not passed and evaluation.log else "")
            current["phase"] = "status_after"
            after = get_service_status(target.server_url)
            save("status-after.json", after)
            for field in ("api_version", "backend", "timing", "target", "software"):
                if after.get(field) != status.get(field):
                    raise ValueError(f"target {field} changed during retest")
            self._ready(after, "status_after")
        except RunCancelled:
            raise
        except Exception as exc:
            control.checkpoint("AFTER_TARGET_ERROR")
            values.update(state="ERROR", phase=current["phase"], error=f"{type(exc).__name__}: {exc}")
        result = DeviceTestResult.model_validate(values)
        save("result.json", result)
        return result

    def _execute(self, inp: MultipleDeviceTestInput):
        plan, code = self._freeze(inp)
        control = WorkspaceRunControl(self.root, source="workflow:multiple_device_test", root_workspace=self.root)
        control.update_progress(state=RunState.RUNNING, stage="TESTING", mode=self.name,
                                progress_kind="tasks", total_tasks=len(inp.targets), completed_tasks=0)
        results = []
        try:
            with cooperative_sigint(control):
                for target in inp.targets:
                    control.checkpoint("BEFORE_TARGET")
                    child = control.link_workspace(self.root / "targets" / target.name,
                                                   scope=f"targets/{target.name}")
                    child.update_progress(state=RunState.RUNNING, stage="STATUS", mode=self.name)
                    control.record_event("DEVICE_TEST_STARTED", stage="TESTING", data={"target": target.name})
                    result = self._test_target(inp, target, plan, code, child)
                    child.update_progress(state=RunState.SUCCEEDED if result.state == "PASSED" else RunState.FAILED,
                                          stage=result.phase.upper(), message=result.state)
                    results.append(result)
                    control.update_progress(completed_tasks=len(results))
                    control.record_event("DEVICE_TEST_COMPLETED", stage="TESTING",
                                         data={"target": target.name, "state": result.state})
        except RunCancelled:
            control.acknowledge_cancellation(stage="TESTING")
            return self._finish(inp, plan, results, state="CANCELLED")
        state = "SUCCEEDED" if len(results) == len(inp.targets) and all(
            result.state == "PASSED" for result in results) else "FAILED"
        result = self._finish(inp, plan, results, state=state)
        control.update_progress(state=RunState.SUCCEEDED if state == "SUCCEEDED" else RunState.FAILED,
                                stage="COMPLETED", message=result.message)
        return result

    def _finish(self, inp, plan, results, *, state):
        output = MultipleDeviceTestOutput(
            operator=inp.operator, workspace=str(self.root), source_revision=plan["source_revision"],
            candidate_sha256=plan["candidate_sha256"], bundle_id=plan["bundle_id"],
            completed_targets=len(results), passed_targets=sum(r.state == "PASSED" for r in results),
            results=results)
        result = WorkflowResult[MultipleDeviceTestOutput](
            state=state, output=output,
            message=f"{output.passed_targets}/{output.completed_targets} targets passed")
        atomic_write_json(self.root / "workflow_result.json", result.model_dump(mode="json"))
        return result

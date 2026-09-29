"""Unified operator optimization build and shared execution policy."""

from pathlib import Path
from functools import partial
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kernelgen.framework.workflow import Workflow, WorkflowResult, WorkflowSummary
from kernelgen.data.constants import DEFAULT_CATALOG_NAME, DEFAULT_OPTIMIZATION_MODE
from .artifacts import file_digest, catalog_identity
from . import operations
from .inputs import OptimizationOptions


def _canonical_reference_paths(fields):
    """Decode old persisted path/digest keys without rewriting their evidence."""
    aliases = {"reference_triton_path": "reference_code_path",
               "reference_triton_prompt_path": "reference_code_prompt_path"}
    result = {}
    for key, value in fields.items():
        key = aliases.get(key, key)
        if key in result and result[key] != value:
            raise ValueError(f"conflicting reference input: {key}")
        result[key] = value
    return result

class OperatorOptimizeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operator: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
    dummy: bool = False
    resume: bool = False
    skip_review: bool = False
    optimization: OptimizationOptions
    target_snapshot: dict[str, Any] = Field(default_factory=dict)
    code_inputs_sha256: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def default_optimizer(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            options = value.get("optimization", {})
            if isinstance(options, dict):
                value["optimization"] = {
                    "mode": DEFAULT_OPTIMIZATION_MODE,
                    "definition_name": value.get("operator"),
                    **_canonical_reference_paths(options),
                }
            if value.get("catalog_name") is None and value.get("catalog_path") is None:
                value["catalog_name"] = DEFAULT_CATALOG_NAME
        return value

    @model_validator(mode="after")
    def validate_scope(self):
        if self.optimization.definition_name != self.operator:
            raise ValueError("optimization definition must match the selected operator")
        if self.optimization.catalog_name != DEFAULT_CATALOG_NAME:
            raise ValueError("optimization.catalog_name is derived from the selected source, not a second input")
        return self

    catalog_path: Path | None = None
    catalog_name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
    catalog_sha256: str = ""

    @model_validator(mode="after")
    def validate_source(self):
        if (self.catalog_path is None) == (self.catalog_name is None):
            raise ValueError("select exactly one of catalog_name or catalog_path")
        if self.catalog_name is not None and self.catalog_sha256:
            raise ValueError("catalog_sha256 is only for local Catalog evidence")
        return self

    def execution_plan(self):
        plan = self.model_dump(mode="json", exclude={"operator", "dummy", "resume"})
        plan["test_validation"] = "native_reference_or_gems_benchmark_core_v1"
        if self.catalog_name is None:
            plan.pop("catalog_name", None)  # Preserve existing local-Catalog plans.
        return plan


class OperatorOptimizeWorkflow(Workflow):
    """Prepare, validate, optimize and review one operator on one target."""

    # Persisted plan/scope identifiers are independent of Python import names.
    name = "catalog_optimize"
    InputModel = OperatorOptimizeInput
    OutputModel = WorkflowResult[WorkflowSummary]
    call_progress_kinds = {"optimize": "rounds"}

    def __init__(self, *, cwd=".", runtime_factory=None, workflow_call=None):
        self.root = Path(cwd).expanduser().resolve()
        self.runtime_factory = runtime_factory
        self.injected_runner = workflow_call

    def _execute(self, inp):
        from kernelgen.tools.kernelgen_server_adapter import get_service_status, require_target_context
        inp = self.prepare_input(inp)
        inp = self.prepare_code_inputs(inp)
        self.call_progress_kinds = {'optimize':'epochs' if inp.optimization.mode=='kernelgen' else 'rounds'}
        if not inp.dummy:
            status = get_service_status(inp.optimization.eval_server_url)
            target_input = inp.optimization
            if target_input.target_hardware is None:
                target_input = target_input.model_copy(update={"target_hardware": ""})
            target = require_target_context(target_input, status)
            if inp.catalog_name is not None:
                if status.get("capabilities", {}).get("operator_contract", {}).get("enabled") is not True:
                    raise RuntimeError("KGS operator contract export is required for installed Catalog input")
            else:
                capability = status.get("capabilities", {}).get("operator_bundle_upload", {})
                if capability.get("enabled") is not True or capability.get("evaluation_binding") is not True:
                    raise RuntimeError("KGS operator bundle execution is required before starting the pipeline")
                from kernelgen_client import Catalog
                if (Catalog(inp.catalog_path).evaluator == "flaggems" and
                        "flaggems" not in capability.get("evaluators", [])):
                    raise RuntimeError("KGS does not support uploaded Gems Definitions")
            observed = {key: status[key] for key in ("api_version", "backend", "timing", "target", "software")}
            if inp.target_snapshot and inp.target_snapshot != observed:
                raise ValueError("target environment differs from the supplied snapshot")
            inp = inp.model_copy(update={"target_snapshot": observed,
                                         "optimization": inp.optimization.model_copy(update={"target_hardware": target.device})})
        plan = {"schema_version": "1.1", "operator": inp.operator, **inp.execution_plan(), "simulated": inp.dummy}
        path = self.root / ".kernelgen/operator-lifecycle.json"
        if inp.resume and path.is_file():
            saved = json.loads(path.read_text())
            canonical = {**saved, "optimization": _canonical_reference_paths(saved["optimization"]),
                         "code_inputs_sha256": _canonical_reference_paths(saved.get("code_inputs_sha256", {}))}
            expected = {"stages": [name for name, _ in self.build(inp)], **plan}
            if canonical == expected:
                # The executor still compares the original immutable plan. Only
                # known spelling changes are accepted; all other changes fail.
                plan = saved
        return self.run_build(inp, plan=plan, mode="dummy" if inp.dummy else self.name,
                              validate_artifacts=self.validate_artifacts)

    def prepare_code_inputs(self, inp):
        from kernelgen.workflows.optimization.single_coder.reference import _load_reference_text
        import hashlib
        paths = {}
        digests = {}
        for name in ('reference_code_path','reference_code_prompt_path','seed_code_path'):
            path = getattr(inp.optimization,name)
            if path is not None:
                path = path.expanduser().resolve()
                text = _load_reference_text(path,name)
                paths[name] = path
                digests[name] = hashlib.sha256(text.encode()).hexdigest()
        if inp.code_inputs_sha256 and _canonical_reference_paths(inp.code_inputs_sha256) != digests:
            raise ValueError('reference or seed source differs from supplied digest')
        return inp.model_copy(update={'optimization':inp.optimization.model_copy(update=paths),
                                      'code_inputs_sha256':digests})

    def build(self, inp):
        calls = tuple((name, partial(invoke, inp, self.root, self.runtime_factory))
                      for name, invoke in (
                          ("prepare_catalog", operations.prepare_catalog),
                          ("review_tests", operations.review_tests),
                          ("optimize", operations.optimize),
                          ("code_review", operations.code_review),
                      ))
        if self.injected_runner is not None:
            return tuple((name, self.injected_runner) for name, _ in calls)
        if inp.dummy:
            def dummy(context):
                return WorkflowResult(simulated=True, output={"called": False, "adapter": context.stage})
            return tuple((name, dummy) for name, _ in calls)
        if self.runtime_factory is None:
            raise ValueError("real operator optimization requires a runtime_factory")
        return calls

    def validate_artifacts(self, result):
        for name, digest in result.output.get("artifacts", {}).items():
            path = Path(name).resolve()
            if not path.is_relative_to(self.root) or not path.is_file() or file_digest(path) != digest:
                raise ValueError(f"lifecycle artifact missing, foreign or changed: {path}")


    def prepare_input(self, inp):
        if inp.catalog_name is not None:
            return inp
        inp = inp.model_copy(update={"catalog_path": inp.catalog_path.expanduser().resolve()})
        if not inp.dummy:
            digest = catalog_identity(inp.catalog_path, inp.operator)
            if inp.catalog_sha256 and inp.catalog_sha256 != digest:
                raise ValueError("catalog differs from the supplied digest")
            inp = inp.model_copy(update={"catalog_sha256": digest})
        return inp

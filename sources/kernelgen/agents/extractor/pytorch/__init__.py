"""PyTorchV5ExtractorAgent: extract operator definition + workloads from PyTorch.

Output schema is v5-aligned (identical to FlagGems extractor output) so results
can be sent directly to kernelgen_server for evaluation without conversion.
"""

from __future__ import annotations

import ast
import json as json_mod
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from kernelgen.framework.base import BaseAgent
from kernelgen.framework.contract import render_contract


V5_GROUPS = {
    "attention",
    "backward",
    "conv",
    "convolution",
    "gemm",
    "indexing",
    "interpolate",
    "linalg",
    "norm",
    "pointwise",
    "pooling",
    "reduction",
    "rnn",
    "scatter",
    "shape",
}


# ---------------------------------------------------------------------------
# I/O models (v5 schema, aligned with FlagGems extractor)
# ---------------------------------------------------------------------------

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkloadInput(StrictModel):
    type: Literal["random", "custom", "scalar", "literal"]
    shape: JsonValue | None = None
    dtype: str | None = None
    value: JsonValue | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_kind_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        kind = data.get("type")
        if kind in {"random", "custom"}:
            if data.get("shape") is None or not data.get("dtype"):
                raise ValueError(f"{kind} inputs require shape and dtype")
        elif kind in {"scalar", "literal"} and "value" not in data:
            raise ValueError(f"{kind} inputs require value")
        return data


class Tolerance(StrictModel):
    rtol: float | None = Field(default=None, ge=0)
    atol: float | None = Field(default=None, ge=0)
    atol_scale: float = Field(default=1.0, gt=0)
    required_matched_ratio: float = Field(default=1.0, gt=0, le=1)


class Workload(StrictModel):
    name: str = Field(min_length=1)
    inputs: Dict[str, WorkloadInput]
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    tolerance: Tolerance | None = None


class Definition(StrictModel):
    api_version: Literal["v5.1"] = "v5.1"
    name: str = Field(min_length=1)
    description: str = ""
    inputs: List[str] = Field(min_length=1)
    outputs: List[str] = Field(min_length=1)
    reference: str = Field(min_length=1)
    reference_device: Literal["target", "cpu"] = "target"
    correctness_reference_entrypoint: str | None = None
    custom_inputs_entrypoint: str | None = None
    custom_valid_entrypoint: str | None = None


class ExtractorResult(StrictModel):
    group: str
    definition: Definition
    correctness_workloads: List[Workload] = Field(default_factory=list)
    timing_workloads: List[Workload] = Field(default_factory=list)
    verification_status: Optional[str] = Field(default=None, description="PASSED / PARTIAL_PASS / FAILED / ERROR / None=not verified")
    verification_detail: str = ""

    @property
    def workloads(self) -> List[Workload]:
        return self.correctness_workloads + self.timing_workloads


class PyTorchExtractorInput(StrictModel):
    operator: str
    torch_version: Optional[str] = None
    default_params: Optional[Dict[str, Any]] = None
    server_url: str = Field(default="", description="kernelgen_server URL for post-extraction verification")


class PyTorchExtractorOutput(StrictModel):
    results: List[ExtractorResult]

    @model_validator(mode="after")
    def validate_v5_protocol(self) -> "PyTorchExtractorOutput":
        definition_names: set[str] = set()
        workload_names: set[str] = set()
        for result in self.results:
            definition = result.definition
            if result.group not in V5_GROUPS:
                raise ValueError(
                    f"{definition.name}: unsupported v5 group {result.group!r}; "
                    f"expected one of {sorted(V5_GROUPS)}"
                )
            if not definition.name.startswith("flaggems_"):
                raise ValueError(
                    f"{definition.name}: v5 names must start with 'flaggems_'"
                )
            if definition.name in definition_names:
                raise ValueError(f"duplicate definition name {definition.name!r}")
            definition_names.add(definition.name)
            if len(definition.inputs) != len(set(definition.inputs)):
                raise ValueError(f"{definition.name}: duplicate input names")
            if len(definition.outputs) != len(set(definition.outputs)):
                raise ValueError(f"{definition.name}: duplicate output names")

            try:
                module = ast.parse(definition.reference)
            except SyntaxError as error:
                raise ValueError(
                    f"{definition.name}: invalid reference source: {error}"
                ) from error
            functions = {
                node.name
                for node in module.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if "run" not in functions:
                raise ValueError(f"{definition.name}: reference must define run()")
            for entrypoint in (
                definition.correctness_reference_entrypoint,
                definition.custom_inputs_entrypoint,
                definition.custom_valid_entrypoint,
            ):
                if entrypoint and entrypoint not in functions:
                    raise ValueError(
                        f"{definition.name}: missing top-level {entrypoint}()"
                    )
            if not result.workloads:
                raise ValueError(f"{definition.name}: no workloads")

            expected_inputs = set(definition.inputs)
            for workload in result.workloads:
                if workload.name in workload_names:
                    raise ValueError(
                        f"{definition.name}: duplicate workload name {workload.name!r}"
                    )
                workload_names.add(workload.name)
                provided = set(workload.inputs.keys())
                custom_ep = definition.custom_inputs_entrypoint
                if custom_ep:
                    missing = expected_inputs - provided
                    extra = provided - expected_inputs
                    if missing and not any(
                        w.type == "custom" for w in workload.inputs.values()
                    ):
                        raise ValueError(
                            f"{definition.name}/{workload.name}: missing inputs {missing}"
                        )
                else:
                    if provided != expected_inputs:
                        raise ValueError(
                            f"{definition.name}/{workload.name}: "
                            f"expected inputs {sorted(expected_inputs)}, "
                            f"got {sorted(provided)}"
                        )
        return self


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_NATIVE_FUNCTIONS_PATH = Path(__file__).resolve().parents[3] / ".claude" / "references" / "native_functions.yaml"
_CORE_SHAPES_PATH = Path(__file__).resolve().parents[3] / ".claude" / "references" / "core_shapes.yaml"


class OperatorNotFoundError(ValueError):
    """Raised when the operator is not a standard PyTorch aten op."""


def _check_aten_op(operator: str) -> bool:
    """Check if operator exists in torch.ops.aten at runtime."""
    try:
        import torch
        return hasattr(torch.ops.aten, operator)
    except Exception:
        return False


def _extract_native_entries(operator: str) -> str:
    """Extract all native_functions.yaml entries matching the operator name."""
    if not _NATIVE_FUNCTIONS_PATH.exists():
        return ""
    lines = _NATIVE_FUNCTIONS_PATH.read_text(encoding="utf-8").splitlines()
    entries: list[str] = []
    current: list[str] = []
    capturing = False
    for line in lines:
        if line.startswith("- func:"):
            if capturing and current:
                entries.append("\n".join(current))
            func_name = line.split("(")[0].replace("- func: ", "").split(".")[0]
            capturing = func_name == operator or func_name == f"_{operator}"
            current = [line] if capturing else []
        elif capturing:
            if line and not line.startswith("- func:"):
                current.append(line)
    if capturing and current:
        entries.append("\n".join(current))
    return "\n\n".join(entries)


class PyTorchV5ExtractorAgent(BaseAgent):
    """Extract PyTorch operator definitions and workloads (v5 schema).

    Neutral role: ``.kernelgen/agents/kernel-pytorch-extractor.md``.
    """

    name = "pytorch_extractor"
    InputModel = PyTorchExtractorInput
    OutputModel = PyTorchExtractorOutput

    def preprocess(self, inp: PyTorchExtractorInput, runtime) -> str:
        role = self._role_for(runtime)
        task_block = (
            "<task>\n"
            f"Operator: {inp.operator}\n"
            f"PyTorch version: {inp.torch_version or 'auto-detect'}\n"
            "</task>"
        )

        native_entries = _extract_native_entries(inp.operator)
        native_block = ""
        if native_entries:
            native_block = (
                "<native_functions>\n"
                "Authoritative operator schema from PyTorch native_functions.yaml:\n\n"
                f"{native_entries}\n"
                "</native_functions>"
            )

        shapes_block = self._build_shapes_reference(inp.operator)

        params_block = ""
        if inp.default_params:
            params_block = (
                f"\n\nDefault parameters provided by user:\n"
                f"```json\n{json_mod.dumps(inp.default_params, indent=2)}\n```\n"
                f"Use these values for shapes in the workloads."
            )

        contract = render_contract(self.OutputModel)

        return "\n\n".join(filter(None, [
            role,
            task_block,
            native_block,
            shapes_block,
            params_block,
            f"--- OUTPUT CONTRACT ---\n{contract}",
        ]))

    def _build_shapes_reference(self, operator: str) -> str:
        """Build a shapes reference block from core_shapes.yaml (timing) and accuracy_utils.py (correctness)."""
        sections: list[str] = []

        # Timing shapes from core_shapes.yaml
        if _CORE_SHAPES_PATH.exists():
            import yaml
            try:
                data = yaml.safe_load(_CORE_SHAPES_PATH.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            if isinstance(data, dict):
                if operator in data:
                    entry = data[operator]
                    sections.append(f"[timing] Exact match '{operator}':\n  {entry}")

                base = operator.rstrip("_").removesuffix("_backward")
                similar = [
                    (k, v) for k, v in data.items()
                    if k != operator and (
                        k.startswith(base) or base in k
                    ) and isinstance(v, dict) and "shapes" in v
                ]
                for k, v in similar[:5]:
                    sections.append(f"[timing] Similar '{k}':\n  shapes: {v['shapes']}\n  desc: {v.get('shape_desc', '')}")

        # Correctness shapes from accuracy_utils.py
        _accuracy_utils_path = Path(__file__).resolve().parents[3] / ".claude" / "references" / "accuracy_utils.py"
        if _accuracy_utils_path.exists():
            import re
            content = _accuracy_utils_path.read_text(encoding="utf-8")
            shape_constants = re.findall(
                r'^([A-Z][A-Z_]*SHAPES?[A-Z_]*)\s*=\s*(.+?)(?=\n[A-Z]|\n\n|\Z)',
                content, re.MULTILINE | re.DOTALL,
            )
            for name, value in shape_constants[:8]:
                clean = value.strip().rstrip(",")
                if len(clean) < 300:
                    sections.append(f"[correctness] {name} = {clean}")

        if not sections:
            return ""
        return (
            "<core_shapes_reference>\n"
            "Reference shapes for workload generation.\n"
            "[timing] = from benchmark/core_shapes.yaml (use for timing_workloads)\n"
            "[correctness] = from tests/accuracy_utils.py (use for correctness_workloads)\n\n"
            + "\n\n".join(sections) + "\n"
            "</core_shapes_reference>"
        )

    def postprocess(self, raw: str, runtime) -> PyTorchExtractorOutput:
        from kernelgen.framework.contract import extract_json

        output = self.OutputModel.model_validate(extract_json(raw))
        # Clear verification fields — only server-side verification can set these
        for result in output.results:
            result.verification_status = None
            result.verification_detail = ""
        return output

    def _execute(self, inp: PyTorchExtractorInput) -> PyTorchExtractorOutput:
        """Override to add pre-check and post-extraction server verification with retry."""
        from kernelgen.framework.base import AgentContractError
        from kernelgen.framework.contract import append_repair, extract_json
        from pydantic import ValidationError

        _VERIFY_MAX_RETRIES = 10
        from pydantic import ValidationError

        # Pre-check: operator must exist in native_functions.yaml OR torch.ops.aten
        entries = _extract_native_entries(inp.operator)
        aten_exists = _check_aten_op(inp.operator)
        if not entries and not aten_exists:
            raise OperatorNotFoundError(
                f"Operator '{inp.operator}' not found in PyTorch native_functions.yaml "
                f"or torch.ops.aten. Only standard aten operators are supported."
            )

        if not inp.server_url:
            return super()._execute(inp)

        # Custom retry loop: schema validation + server verification
        runtime = self._runtime
        prompt = self.preprocess(inp, runtime)
        last_err: Optional[Exception] = None
        last_output: Optional[PyTorchExtractorOutput] = None

        for attempt in range(max(1, _VERIFY_MAX_RETRIES)):
            raw = self._invoke_runtime(runtime, prompt)
            try:
                output = self.postprocess(raw, runtime)
            except (ValidationError, ValueError) as e:
                last_err = e
                prompt = append_repair(prompt, e)
                continue

            # Schema valid — now verify with server
            import tempfile
            all_passed = True
            with tempfile.TemporaryDirectory(prefix="extract_verify_") as tmpdir:
                catalog_root = write_catalog(output.results, tmpdir)
                for result in output.results:
                    eval_result = verify_extraction(result, catalog_root, server_url=inp.server_url)
                    status = eval_result.get("status", "UNKNOWN")
                    num_passed = eval_result.get("num_passed", 0)
                    num_wl = eval_result.get("num_workloads", 0)
                    result.verification_status = status
                    if status == "PASSED":
                        print(
                            f"  [verify] {result.definition.name}: PASSED "
                            f"({num_passed}/{num_wl} workloads, "
                            f"geo_mean={eval_result.get('geo_mean'):.4f})",
                            flush=True,
                        )
                    else:
                        all_passed = False
                        log = eval_result.get("log", "")[:200]
                        result.verification_detail = f"{status} ({num_passed}/{num_wl}). {log}"
                        print(
                            f"  [verify] {result.definition.name}: {status} "
                            f"({num_passed}/{num_wl} workloads). {log}",
                            flush=True,
                        )

            last_output = output
            if all_passed:
                return output

            # Verification failed — retry with error feedback
            failed_details = [r.verification_detail for r in output.results if r.verification_status != "PASSED"]
            last_err = ValueError(f"Server verification failed: {'; '.join(failed_details)}")
            prompt = append_repair(prompt, last_err)

        # All retries exhausted — return last output with verification_status marked
        if last_output is not None:
            return last_output
        raise AgentContractError(
            f"{self.name}: no valid output after {_VERIFY_MAX_RETRIES} attempts: {last_err}"
        )


# Backward-compatible public name for callers that predate the explicit schema
# version in the class name. New code should use PyTorchV5ExtractorAgent.
PyTorchExtractorAgent = PyTorchV5ExtractorAgent


# ---------------------------------------------------------------------------
# Catalog I/O + verification
# ---------------------------------------------------------------------------

def write_catalog(
    results: List[ExtractorResult],
    catalog_root: str | Path,
) -> Path:
    """Write extraction results to a v5.1 catalog directory.

    Creates the standard catalog structure:
        manifest.json
        definitions/{group}/{name}.json
        workloads/{group}/{name}.correctness.jsonl
        workloads/{group}/{name}.timing.jsonl
    """
    import json as _json

    root = Path(catalog_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    operators = []
    for result in results:
        group = result.group
        name = result.definition.name
        def_rel = f"definitions/{group}/{name}.json"
        corr_rel = f"workloads/{group}/{name}.correctness.jsonl"
        timing_rel = f"workloads/{group}/{name}.timing.jsonl"

        def_path = root / def_rel
        def_path.parent.mkdir(parents=True, exist_ok=True)
        def_path.write_text(
            _json.dumps(result.definition.model_dump(mode="json", exclude_none=True), indent=2) + "\n",
            encoding="utf-8",
        )

        for rel, workloads in [(corr_rel, result.correctness_workloads), (timing_rel, result.timing_workloads)]:
            wl_path = root / rel
            wl_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for w in workloads:
                dumped = w.model_dump(mode="json", exclude_none=True)
                # Preserve "value": null for literal/scalar inputs where None is intentional
                for input_name, spec in w.inputs.items():
                    if spec.type in ("literal", "scalar") and spec.value is None:
                        dumped["inputs"][input_name]["value"] = None
                lines.append(_json.dumps(dumped, separators=(",", ":")))
            wl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        operators.append({
            "name": name,
            "group": group,
            "definition": def_rel,
            "correctness_workloads": corr_rel,
            "timing_workloads": timing_rel,
            "num_correctness_workloads": len(result.correctness_workloads),
            "num_timing_workloads": len(result.timing_workloads),
        })

    manifest = {
        "api_version": "v5.1",
        "name": "pytorch-extractor-v5.1",
        "operators": sorted(operators, key=lambda e: e["name"]),
        "counts": {
            "operators": len(operators),
            "correctness_workloads": sum(e["num_correctness_workloads"] for e in operators),
            "timing_workloads": sum(e["num_timing_workloads"] for e in operators),
        },
    }
    (root / "manifest.json").write_text(
        _json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return root


def verify_extraction(
    result: ExtractorResult,
    catalog_root: str | Path,
    server_url: str = "",
) -> dict:
    """Verify an extracted definition by running its reference as the kernel.

    Writes the reference source to a temp file and submits it to kernelgen_server.
    A PASS proves the definition + workloads + gen_inputs are internally consistent
    and executable by the eval server.

    Returns the eval result dict (status, geo_mean, per_workload, etc.)
    """
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix=f"verify_{result.definition.name}_", delete=False,
    ) as f:
        f.write(result.definition.reference)
        kernel_path = f.name

    resolved_url = server_url or os.environ.get("KERNELGEN_SERVER_URL", "")
    if not resolved_url:
        return {"status": "ERROR", "log": "server_url not provided", "num_passed": 0, "num_workloads": 0}

    try:
        from kernelgen_client import Catalog
        from kernelgen_client.http import evaluate as server_evaluate
        from kernelgen_client.protocol.schema import (
            EvaluateRequest,
            Implementation,
            SourceFile,
            Language,
        )

        catalog = Catalog(str(Path(catalog_root).resolve()))
        op_data = catalog.load(result.definition.name)

        source_content = Path(kernel_path).read_text(encoding="utf-8")
        impl = Implementation(
            name=f"verify_{result.definition.name}",
            definition=result.definition.name,
            language=Language.PYTHON,
            entrypoint="main.py::run",
            sources=[SourceFile(path="main.py", content=source_content)],
        )

        request = EvaluateRequest(
            definition=op_data.definition,
            implementation=impl,
            correctness_workloads=list(op_data.correctness_workloads),
            timing_workloads=list(op_data.timing_workloads),
        )

        response = server_evaluate(request, resolved_url)
        resp_dict = response.model_dump(mode="json")
        resp_dict.setdefault("num_passed", sum(
            1 for w in resp_dict.get("per_workload", []) if w.get("status") == "PASSED"
        ))
        resp_dict.setdefault("num_workloads", len(resp_dict.get("per_workload", [])))
        return resp_dict
    except Exception as e:
        return {"status": "ERROR", "log": str(e)[:300], "num_passed": 0, "num_workloads": 0}

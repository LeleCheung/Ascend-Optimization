"""Direct FlagGems source -> native V6.2 extraction agent.

This path deliberately has no translated-catalog input.  The model reads the
pinned FlagGems implementation, pytest and benchmark files, while Python fixes
the public Definition from the exported FlagGems signature and fixes timing
workload identity from the target ``--list-cases`` report.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, JsonValue, model_validator
from kernelgen_client.protocol.schema import SourceCondition

from kernelgen.agents.extractor.flaggems.abi_validation import (
    finite_json,
    top_level_functions,
    validate_gen_inputs_randomness,
    validate_hook_signature,
    validate_run_abi,
)
from kernelgen.agents.extractor.flaggems.adapter import (
    FlagGemsDefinitionSpec,
    extract_flaggems_definition,
)
from kernelgen.agents.extractor.flaggems.models import StrictModel, Tolerance
from kernelgen.agents.extractor.flaggems.source_inventory import (
    FlagGemsSourceInventory,
    collect_source_inventory,
)
from kernelgen.agents.extractor.flaggems.v62_accuracy import (
    accuracy_coverage_plan as _accuracy_coverage_plan,
    accuracy_coverage_report as _accuracy_coverage_report,
    accuracy_test_identities as _accuracy_test_identities,
    accuracy_test_scope as _accuracy_test_scope,
    duplicate_accuracy_test_identities as _duplicate_accuracy_test_identities,
    validate_accuracy_workload_scope as _validate_accuracy_workload_scope,
)
from .source_profile import PROFILES, checkout_profile

from kernelgen.framework.base import BaseAgent
from kernelgen.framework.contract import extract_json, render_contract
from kernelgen.agents.catalog_blocker import CatalogBlocker


_CASE_LIST_SCHEMA = "flaggems.benchmark-case-list/v2"


class FlagGemsV62ExtractorInput(StrictModel):
    operator: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    flaggems_repo: str
    case_list_path: str
    timing_reference: Literal["source", "flaggems"] = "source"


class FlagGemsV62Workload(StrictModel):
    """A KGS V6.2 workload containing JSON context, not a call expression."""

    name: str = Field(min_length=1)
    inputs: dict[str, JsonValue]
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    tolerance: Tolerance | None = None
    source_condition: SourceCondition | None = None

    @model_validator(mode="after")
    def validate_context(self) -> "FlagGemsV62Workload":
        invalid = sorted(name for name in self.inputs if not name.isidentifier())
        if invalid:
            raise ValueError(f"invalid Workload input names {invalid}")
        finite_json(self.inputs, f"workload {self.name}.inputs")
        return self


class FlagGemsV62DtypeExpansion(StrictModel):
    """Compact request for a mechanically identical target dtype counterpart.

    An omitted tolerance inherits the source workload tolerance.  When present,
    only explicitly supplied fields override the source, so case-dependent
    values such as ``atol_scale`` survive dtype expansion.
    """

    source_dtype: str = Field(min_length=1)
    target_dtype: str = Field(min_length=1)
    tolerance: Tolerance | None = None


class FlagGemsV62ExtractorOutput(StrictModel):
    """Only the source-derived assets that require semantic reconstruction."""

    blocker: CatalogBlocker | None = None
    source_policy_id: str | None = Field(default=None, min_length=1)
    oracle: str = ""
    correctness_workloads: list[FlagGemsV62Workload] = Field(default_factory=list)
    correctness_dtype_expansions: list[FlagGemsV62DtypeExpansion] = Field(
        default_factory=list
    )
    timing_workloads: list[FlagGemsV62Workload] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_or_blocked(self):
        if self.blocker is not None:
            if self.source_policy_id or self.oracle or self.correctness_workloads or self.timing_workloads or self.correctness_dtype_expansions:
                raise ValueError("return either a blocker or complete Catalog assets, not both")
        elif not self.oracle.strip() or not self.correctness_workloads or not self.timing_workloads:
            raise ValueError("successful extraction requires oracle, correctness and timing workloads")
        if any(w.source_condition is not None for w in [*self.correctness_workloads, *self.timing_workloads]) and not self.source_policy_id:
            raise ValueError("conditional workloads require source_policy_id")
        if self.oracle and not self.source_policy_id:
            try:
                module = ast.parse(self.oracle)
            except SyntaxError:
                module = None  # Detailed syntax errors belong to oracle validation.
            if module is not None and any(
                isinstance(node, ast.ImportFrom)
                and node.module == "kernelgen_server.runtime.source_policy"
                or isinstance(node, ast.Import)
                and any(alias.name == "kernelgen_server.runtime.source_policy" for alias in node.names)
                for node in ast.walk(module)
            ):
                raise ValueError("source-policy oracle imports require source_policy_id")
        return self


@dataclass(frozen=True)
class FlagGemsV62DirectExtraction:
    catalog_root: Path
    operator_root: Path
    definition_path: Path
    oracle_path: Path
    correctness_path: Path
    timing_path: Path
    adapter_definition_path: Path | None
    num_correctness_workloads: int
    num_timing_workloads: int


@dataclass(frozen=True)
class _AbiParameter:
    name: str
    kind: str
    required: bool
    default: Any
    type: str | None


def _normalized_type_hint(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace("torch.", "")


def _abi_parameters(definition: FlagGemsDefinitionSpec) -> list[_AbiParameter]:
    return [
        _AbiParameter(
            name=parameter.name,
            kind=parameter.kind,
            required=parameter.required,
            default=parameter.default,
            type=_normalized_type_hint(parameter.type_hint),
        )
        for parameter in definition.parameters
    ]


def load_flaggems_timing_cases(path: str | Path, operator: str) -> list[dict[str, Any]]:
    """Load the exact target core timing sequence for one public operator."""

    report_path = Path(path).expanduser().resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{operator}: cannot read --list-cases report: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != _CASE_LIST_SCHEMA:
        raise ValueError(f"{operator}: invalid --list-cases report schema")
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError(f"{operator}: --list-cases benchmarks must be a list")
    matches = [
        benchmark
        for benchmark in benchmarks
        if isinstance(benchmark, dict) and benchmark.get("op_name") == operator
    ]
    if not matches:
        raise ValueError(f"{operator}: --list-cases report has no matching benchmark")

    cases: list[dict[str, Any]] = []
    for benchmark in matches:
        if (
            benchmark.get("schema_version") != _CASE_LIST_SCHEMA
            or benchmark.get("phase") != "timing"
            or benchmark.get("level") != "core"
        ):
            raise ValueError(f"{operator}: --list-cases benchmark is not core timing")
        values = benchmark.get("cases")
        if not isinstance(values, list):
            raise ValueError(f"{operator}: --list-cases cases must be a list")
        cases.extend(values)

    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or any(
        not isinstance(case_id, str) or not case_id for case_id in case_ids
    ):
        raise ValueError(f"{operator}: every timing case needs a case_id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{operator}: timing case IDs must be unique")
    return cases


def _source_paths(inventory: FlagGemsSourceInventory) -> str:
    rows = [f"- implementation: `{inventory.implementation_file}`"]
    rows.extend(f"- accuracy pytest: `{path}`" for path in inventory.test_files)
    rows.extend(f"- benchmark pytest: `{path}`" for path in inventory.benchmark_files)
    return "\n".join(rows)


def _attribute_path(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        if parent is not None:
            return (*parent, node.attr)
    return None


def _writes_torch_backend_state(node: ast.AST) -> bool:
    for child in ast.walk(node):
        targets: list[ast.expr] = []
        if isinstance(child, ast.Assign):
            targets.extend(child.targets)
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
            targets.append(child.target)
        if any(
            (path := _attribute_path(target)) is not None
            and path[:2] == ("torch", "backends")
            for target in targets
        ):
            return True
    return False


def _calls_flaggems_export(
    function: ast.FunctionDef,
    operator: str,
    source_package: str = "flag_gems",
) -> bool:
    return any(
        isinstance(node, ast.Call)
        and _attribute_path(node.func) == (source_package, operator)
        for node in ast.walk(function)
    )


def _uses_raw_device_token_as_backend_type(function: ast.FunctionDef) -> bool:
    """Reject guards that compare ``cuda:0``-style tokens to ``cuda``."""

    raw_aliases = {"device"}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(function):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            copies_raw_token = (
                isinstance(value, ast.Name) and value.id in raw_aliases
            ) or (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "str"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Name)
                and value.args[0].id in raw_aliases
            )
            if not copies_raw_token:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in raw_aliases:
                    raw_aliases.add(target.id)
                    changed = True

    def raw_token(value: ast.expr) -> bool:
        return (
            isinstance(value, ast.Name) and value.id in raw_aliases
        ) or (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "str"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id in raw_aliases
        )

    return any(
        isinstance(node, ast.Compare)
        and raw_token(node.left)
        and any(
            isinstance(value, ast.Constant) and isinstance(value.value, str)
            for value in node.comparators
        )
        for node in ast.walk(function)
    )


def _uses_hard_coded_backend_in_generic_device_else(
    function: ast.FunctionDef,
) -> bool:
    """Reject a catch-all device branch that silently assumes one backend.

    A normalized device type may be ``cuda``, ``npu``, ``musa`` or another
    registered backend.  Therefore ``if type == 'musa' ... else:
    torch.backends.cuda...`` is not a faithful translation of FlagGems'
    generic ``torch_backend_device`` setup.  An explicitly guarded backend is
    fine; a final catch-all must resolve the backend from the normalized type.
    """

    normalized_aliases: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(function):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            is_normalized_type = bool(
                isinstance(value, ast.Attribute)
                and value.attr == "type"
                and isinstance(value.value, ast.Call)
                and _attribute_path(value.value.func) == ("torch", "device")
                and len(value.value.args) == 1
                and isinstance(value.value.args[0], ast.Name)
                and value.value.args[0].id == "device"
            )
            copies_normalized_type = bool(
                isinstance(value, ast.Name)
                and value.id in normalized_aliases
            )
            if not (is_normalized_type or copies_normalized_type):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in normalized_aliases:
                    normalized_aliases.add(target.id)
                    changed = True

    def guarded_type(node: ast.If) -> tuple[str, str] | None:
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == len(test.comparators) == 1
            and isinstance(test.ops[0], (ast.Eq, ast.NotEq))
        ):
            return None
        left, right = test.left, test.comparators[0]
        if (
            isinstance(left, ast.Name)
            and left.id in normalized_aliases
            and isinstance(right, ast.Constant)
            and isinstance(right.value, str)
        ):
            return left.id, right.value
        if (
            isinstance(right, ast.Name)
            and right.id in normalized_aliases
            and isinstance(left, ast.Constant)
            and isinstance(left.value, str)
        ):
            return right.id, left.value
        return None

    def written_backend_names(nodes: list[ast.stmt]) -> set[str]:
        values: set[str] = set()
        for root in nodes:
            for node in ast.walk(root):
                targets: list[ast.expr] = []
                if isinstance(node, ast.Assign):
                    targets.extend(node.targets)
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    targets.append(node.target)
                for target in targets:
                    path = _attribute_path(target)
                    if path is not None and path[:2] == ("torch", "backends"):
                        if len(path) >= 3:
                            values.add(path[2])
        return values

    for root in ast.walk(function):
        if not isinstance(root, ast.If):
            continue
        first = guarded_type(root)
        if first is None:
            continue
        selector, first_type = first
        guarded = {first_type}
        current = root
        while len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            next_if = current.orelse[0]
            item = guarded_type(next_if)
            if item is None or item[0] != selector:
                break
            guarded.add(item[1])
            current = next_if
        final_else = current.orelse
        if final_else and (written_backend_names(final_else) - guarded):
            return True
    return False


def _validate_oracle(
    operator: str,
    definition: FlagGemsDefinitionSpec,
    source: str,
    *,
    has_correctness: bool,
    has_timing: bool,
    needs_gen_inputs: bool,
    mixes_generated_and_direct_inputs: bool,
    needs_torch_fallback: bool,
    requires_float32_timing_promotion: bool = False,
    allows_flaggems_timing_reference: bool = False,
    source_package: str = "flag_gems",
    requires_valid_owned_return_contract: bool = False,
) -> None:
    try:
        module = ast.parse(source, filename=f"<{operator}/oracle.py>")
    except SyntaxError as exc:
        raise ValueError(f"{operator}: invalid oracle.py: {exc}") from exc

    errors: list[str] = []

    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    for package in {p.package for p in PROFILES} & imported_roots:
        if not allows_flaggems_timing_reference or package != source_package:
            errors.append(f"oracle.py cannot import {package}")
    dynamic_calls = sorted(
        {
            node.func.id
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"compile", "eval", "exec"}
        }
    )
    if dynamic_calls:
        errors.append(
            f"oracle.py must be static source; forbidden calls {dynamic_calls}"
        )

    reference_values: list[ast.expr | None] = []
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REFERENCE_DEVICE"
            for target in node.targets
        ):
            reference_values.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "REFERENCE_DEVICE"
        ):
            reference_values.append(node.value)
    reference_device: Any = None
    if len(reference_values) != 1:
        errors.append("oracle.py must declare REFERENCE_DEVICE once")
    else:
        try:
            reference_device = ast.literal_eval(reference_values[0])
        except (TypeError, ValueError):
            errors.append("REFERENCE_DEVICE must be a literal")
        if reference_device not in {"target", "cpu"}:
            errors.append("REFERENCE_DEVICE must be target or cpu")
        if has_timing and reference_device != "target":
            errors.append("timing workloads require REFERENCE_DEVICE='target'")

    functions = top_level_functions(module)
    parameters = _abi_parameters(definition)
    for name in ("run", "correctness_run", "timing_run"):
        function = functions.get(name)
        if function is not None:
            try:
                validate_run_abi(operator, parameters, function, name)
            except ValueError as exc:
                errors.append(str(exc).removeprefix(f"{operator}: "))
    if has_correctness and not ({"run", "correctness_run"} & functions.keys()):
        errors.append("correctness workloads require run() or correctness_run()")
    if has_timing and not ({"run", "timing_run"} & functions.keys()):
        errors.append("timing workloads require run() or timing_run()")
    timing_function = functions.get("timing_run") or functions.get("run")
    correctness_function = functions.get("correctness_run") or functions.get("run")
    if allows_flaggems_timing_reference:
        if source_package not in imported_roots:
            errors.append(
                f"self-baselined timing reference must import {source_package}"
            )
        if timing_function is None or not _calls_flaggems_export(
            timing_function, operator, source_package
        ):
            errors.append(
                "self-baselined timing_run() must call the benchmark's exact "
                f"{source_package}.{operator} export"
            )
        if (
            correctness_function is not None
            and _calls_flaggems_export(correctness_function, operator, source_package)
        ):
            errors.append(
                "correctness reference must remain the pytest Torch oracle; "
                f"the {source_package} import exception is timing-only"
            )
    if (
        requires_float32_timing_promotion
        and timing_function is not None
        and not (
            _preserves_float32_timing_promotion(timing_function)
            or _calls_flaggems_export(timing_function, operator, source_package)
        )
    ):
        errors.append(
            "timing reference must preserve the self-baselined FlagGems "
            "implementation's float16-to-float32 computation and cast its "
            "public outputs back to the input dtype"
        )
    selected_functions = {
        function
        for function in (
            functions.get("run"),
            functions.get("correctness_run"),
            functions.get("timing_run"),
        )
        if function is not None
    }
    stateful_runs = sorted(
        function.name
        for function in selected_functions
        if _writes_torch_backend_state(function)
    )
    if stateful_runs:
        errors.append(
            "benchmark setup must stay outside timed run functions; move "
            "torch.backends setup to gen_inputs(ctx, device) and return None "
            f"for direct workloads (found in {stateful_runs})"
        )
    module_state_writes = [
        node
        for node in module.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and _writes_torch_backend_state(node)
    ]
    if module_state_writes:
        errors.append(
            "oracle.py cannot change torch.backends state at import time; "
            "perform source test setup in gen_inputs(ctx, device)"
        )
    if operator.endswith("_backward") and timing_function is not None:
        calls_autograd_grad = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "grad"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "autograd"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "torch"
            for node in ast.walk(timing_function)
        )
        if calls_autograd_grad:
            errors.append(
                "backward timing reference must call the framework backward "
                "primitive directly; benchmark forward/autograd setup is "
                "outside the timed region"
            )
    if "torch_run" in functions:
        try:
            validate_run_abi(
                operator,
                parameters,
                functions["torch_run"],
                "torch_run",
            )
        except ValueError as exc:
            errors.append(str(exc).removeprefix(f"{operator}: "))
        if not needs_torch_fallback:
            errors.append(
                "oracle.py defines torch_run() without a source-backed "
                "whole-round fallback; reference dtype upcasting or an "
                "fp64/fp32 branch is not fallback evidence"
            )
    elif needs_torch_fallback:
        errors.append(
            "source accuracy reference has a vendor/capability-specific pure "
            "Torch alternative; oracle.py must expose it as ABI-identical "
            "torch_run() for whole-round readiness fallback"
        )
    if operator == "cudnn_convolution" and needs_torch_fallback:
        torch_run = functions.get("torch_run")
        if torch_run is not None:
            torch_attributes = {
                node.attr
                for node in ast.walk(torch_run)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "torch"
            }
            if "cudnn_convolution" in torch_attributes or not (
                torch_attributes & {"conv1d", "conv2d", "conv3d"}
            ):
                errors.append(
                    "cudnn_convolution torch_run() must use the source pure "
                    "Torch conv1d/conv2d/conv3d equivalent and cannot call the "
                    "unsupported primary cudnn_convolution primitive"
                )
    try:
        _validate_broadcast_tensors_run(operator, functions.get("run"))
    except ValueError as exc:
        errors.append(str(exc).removeprefix(f"{operator}: "))

    gen_inputs = functions.get("gen_inputs")
    if needs_gen_inputs and gen_inputs is None:
        errors.append(
            "workloads require gen_inputs(ctx, device) to materialize custom or "
            "dependent inputs"
        )
    elif gen_inputs is not None:
        try:
            validate_hook_signature(operator, gen_inputs, ("ctx", "device"))
        except ValueError as exc:
            errors.append(str(exc).removeprefix(f"{operator}: "))
        if _writes_torch_backend_state(
            gen_inputs
        ) and _uses_raw_device_token_as_backend_type(gen_inputs):
            errors.append(
                "gen_inputs() backend setup must compare a normalized device "
                "type such as torch.device(device).type; the raw KGS device "
                "token can include an index (for example 'cuda:0')"
            )
        if _writes_torch_backend_state(
            gen_inputs
        ) and _uses_hard_coded_backend_in_generic_device_else(gen_inputs):
            errors.append(
                "gen_inputs() generic device fallback cannot hard-code a "
                "torch.backends namespace; resolve the backend from the "
                "normalized device type or guard that backend explicitly"
            )
        for function in functions.values():
            try:
                validate_gen_inputs_randomness(
                    operator, function, allow_target_device=True
                )
            except ValueError as exc:
                errors.append(str(exc).removeprefix(f"{operator}: "))
        if mixes_generated_and_direct_inputs and not any(
            isinstance(node, ast.Return)
            and (
                node.value is None
                or (isinstance(node.value, ast.Constant) and node.value.value is None)
            )
            for node in ast.walk(gen_inputs)
        ):
            errors.append(
                "mixed generated/direct workloads require gen_inputs() to "
                "recognize ordinary recipe workloads and explicitly return None"
            )
    if "valid" in functions:
        try:
            validate_hook_signature(
                operator,
                functions["valid"],
                ("ref_outputs", "sol_outputs", "inputs", "ctx"),
            )
        except ValueError as exc:
            errors.append(str(exc).removeprefix(f"{operator}: "))
        output_collections = {"ref_outputs", "sol_outputs"}
        invalid_collection_uses = []
        for node in ast.walk(functions["valid"]):
            direct_values: list[ast.expr] = []
            if isinstance(node, ast.Attribute):
                direct_values.append(node.value)
            elif isinstance(node, ast.BinOp):
                direct_values.extend((node.left, node.right))
            elif isinstance(node, ast.UnaryOp):
                direct_values.append(node.operand)
            elif isinstance(node, ast.Compare):
                direct_values.append(node.left)
                direct_values.extend(node.comparators)
            if any(
                isinstance(value, ast.Name) and value.id in output_collections
                for value in direct_values
            ):
                invalid_collection_uses.append(node.lineno)
        if invalid_collection_uses:
            errors.append(
                "valid() receives ref_outputs/sol_outputs as top-level output "
                "lists; index a single output with [0] or explicitly iterate "
                "multiple outputs before Tensor operations"
            )
    if "torch_valid" in functions:
        try:
            validate_hook_signature(
                operator,
                functions["torch_valid"],
                ("ref_outputs", "sol_outputs", "inputs", "ctx"),
            )
        except ValueError as exc:
            errors.append(str(exc).removeprefix(f"{operator}: "))

    ownership: dict[str, Any] = {}
    for symbol in (
        "VALID_OWNS_RETURN_CONTRACT",
        "TORCH_VALID_OWNS_RETURN_CONTRACT",
    ):
        values: list[ast.expr | None] = []
        for node in module.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == symbol
                for target in node.targets
            ):
                values.append(node.value)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == symbol
            ):
                values.append(node.value)
        if len(values) > 1:
            errors.append(f"oracle.py must declare {symbol} at most once")
            continue
        if not values:
            ownership[symbol] = False
            continue
        try:
            value = ast.literal_eval(values[0])
        except (TypeError, ValueError):
            value = None
        if not isinstance(value, bool):
            errors.append(f"{symbol} must be a literal boolean")
            continue
        ownership[symbol] = value
    if ownership.get("VALID_OWNS_RETURN_CONTRACT") and "valid" not in functions:
        errors.append("VALID_OWNS_RETURN_CONTRACT=True requires valid()")
    if (
        ownership.get("TORCH_VALID_OWNS_RETURN_CONTRACT")
        and "torch_valid" not in functions
    ):
        errors.append(
            "TORCH_VALID_OWNS_RETURN_CONTRACT=True requires torch_valid()"
        )
    if (
        requires_valid_owned_return_contract
        and not ownership.get("VALID_OWNS_RETURN_CONTRACT")
    ):
        errors.append(
            "source correctness compares sequence outputs by len+zip; oracle.py "
            "must declare VALID_OWNS_RETURN_CONTRACT=True and valid() must "
            "fully reproduce the element contract"
        )
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"{operator}: oracle validation failed:\n{details}")


def _contains_nested_recipe(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("type") in {"random", "custom", "scalar", "literal"}:
            return True
        return any(_contains_nested_recipe(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nested_recipe(item) for item in value)
    return False


def _workload_requires_gen_inputs(
    definition: FlagGemsDefinitionSpec,
    workload: FlagGemsV62Workload,
) -> bool:
    required = {
        parameter.name for parameter in definition.parameters if parameter.required
    }
    if required - workload.inputs.keys():
        return True
    for value in workload.inputs.values():
        if isinstance(value, dict) and value.get("type") == "custom":
            return True
        if isinstance(value, (list, dict)) and _contains_nested_recipe(value):
            if not (
                isinstance(value, dict)
                and value.get("type")
                in {"random", "custom", "scalar", "literal"}
            ):
                return True
    return False


def _gen_inputs_requirements(
    definition: FlagGemsDefinitionSpec,
    workloads: list[FlagGemsV62Workload],
) -> tuple[bool, bool]:
    requirements = [
        _workload_requires_gen_inputs(definition, workload)
        for workload in workloads
    ]
    return any(requirements), any(requirements) and not all(requirements)


def _validate_timing_workloads(
    operator: str,
    workloads: list[FlagGemsV62Workload],
    cases: list[dict[str, Any]],
) -> None:
    if len(workloads) != len(cases):
        raise ValueError(
            f"{operator}: extracted {len(workloads)} timing workloads but "
            f"--list-cases returned {len(cases)}"
        )
    for ordinal, (workload, case) in enumerate(zip(workloads, cases, strict=True)):
        if workload.name != case["case_id"]:
            raise ValueError(
                f"{operator}: timing ordinal {ordinal} must use case_id "
                f"{case['case_id']!r}, got {workload.name!r}"
            )
        if workload.tolerance is not None:
            raise ValueError(
                f"{operator}/{workload.name}: timing workloads are "
                "performance-only and cannot define correctness tolerance"
            )
        expected_dtype = case.get("dtype")
        if not isinstance(expected_dtype, str):
            continue
        expected_dtype = expected_dtype.removeprefix("torch.")
        extracted_dtypes = {
            value.get("dtype")
            for value in workload.inputs.values()
            if isinstance(value, dict)
            and value.get("type") in {"random", "custom"}
            and isinstance(value.get("dtype"), str)
        }
        if extracted_dtypes and expected_dtype not in extracted_dtypes:
            raise ValueError(
                f"{operator}/{workload.name}: listed dtype {expected_dtype} "
                f"not in extracted inputs {sorted(extracted_dtypes)}"
            )
        expected_shapes = case.get("shape")
        if isinstance(expected_shapes, dict):
            for input_name, expected_shape in expected_shapes.items():
                if not isinstance(expected_shape, list):
                    continue
                recipe = workload.inputs.get(input_name)
                if not (
                    isinstance(recipe, dict)
                    and recipe.get("type") in {"random", "custom"}
                    and isinstance(recipe.get("shape"), list)
                ):
                    continue
                if recipe["shape"] != expected_shape:
                    raise ValueError(
                        f"{operator}/{workload.name}: listed shape for "
                        f"{input_name} is {expected_shape}, extracted recipe uses "
                        f"{recipe['shape']}"
                    )


def _validate_random_recipe_devices(
    operator: str,
    workloads: list[FlagGemsV62Workload],
) -> None:
    missing = []
    invalid = []
    for workload in workloads:
        for name, value in workload.inputs.items():
            if not (isinstance(value, dict) and value.get("type") == "random"):
                continue
            device = value.get("device")
            if device is None:
                missing.append(f"{workload.name}:{name}")
            elif device != "target":
                invalid.append(f"{workload.name}:{name}={device!r}")
    if missing or invalid:
        raise ValueError(
            f"{operator}: direct KGS random recipes must use "
            "device='target'; when the FlagGems builder explicitly uses CPU "
            "generation or a distribution outside KGS recipe semantics, use "
            "exact case context plus gen_inputs. This preserves the source "
            f"allocation path; missing={missing[:5]}, invalid={invalid[:5]}"
        )


_V62_RECIPE_TYPES = {"random", "custom", "scalar", "literal"}


def _validate_recipe_types(
    operator: str,
    workloads: list[FlagGemsV62Workload],
) -> None:
    invalid: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if "type" in value and value.get("type") not in _V62_RECIPE_TYPES:
                invalid.append(f"{path}={value.get('type')!r}")
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    for workload in workloads:
        for name, value in workload.inputs.items():
            visit(value, f"{workload.name}:{name}")
    if invalid:
        raise ValueError(
            f"{operator}: unsupported Workload recipe type; v6.2 accepts only "
            f"{sorted(_V62_RECIPE_TYPES)}. Floating torch.randn inputs use "
            f"type='random', not type='randn'. invalid={invalid[:5]}"
        )


_PRIMARY_FLOAT_DTYPES = {"float16", "float32"}
_ALL_FLOAT_DTYPES = _PRIMARY_FLOAT_DTYPES | {"bfloat16"}
_STRICT_FLOAT_TOLERANCES = {
    "float16": (1e-3, 1e-4),
    "bfloat16": (0.016, 1e-4),
    "float32": (1.3e-6, 1e-4),
    "float64": (1e-7, 1e-4),
    "complex64": (1.3e-6, 1e-4),
    "complex128": (1e-7, 1e-4),
}


def _nested_float_dtypes(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = {
            dtype
            for key, dtype in value.items()
            if key == "dtype" and isinstance(dtype, str) and dtype in _ALL_FLOAT_DTYPES
        }
        for item in value.values():
            found.update(_nested_float_dtypes(item))
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found.update(_nested_float_dtypes(item))
        return found
    return set()


def _drop_redundant_strict_tolerances(
    workloads: list[FlagGemsV62Workload],
) -> list[FlagGemsV62Workload]:
    """Omit explicit tolerance that is identical to KGS strict comparison."""

    normalized = []
    for workload in workloads:
        tolerance = workload.tolerance
        dtypes = _nested_float_dtypes(workload.inputs)
        if tolerance is None or len(dtypes) != 1:
            normalized.append(workload)
            continue
        dtype = next(iter(dtypes))
        strict = _STRICT_FLOAT_TOLERANCES.get(dtype)
        if strict is None:
            normalized.append(workload)
            continue
        strict_rtol, strict_atol = strict
        effective_rtol = (
            strict_rtol if tolerance.rtol is None else tolerance.rtol
        )
        effective_atol = (
            strict_atol if tolerance.atol is None else tolerance.atol
        )
        is_redundant = (
            effective_rtol == strict_rtol
            and effective_atol == strict_atol
            and tolerance.atol_scale == 1.0
            and tolerance.required_matched_ratio == 1.0
            and tolerance.equal_nan is False
        )
        normalized.append(
            workload.model_copy(update={"tolerance": None})
            if is_redundant
            else workload
        )
    return normalized


def _replace_float_dtypes(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "<float-dtype>"
                if key == "dtype"
                and isinstance(item, str)
                and item in _ALL_FLOAT_DTYPES
                else _replace_float_dtypes(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_float_dtypes(item) for item in value]
    return value


def _is_fixed_dtype_workload(workload: FlagGemsV62Workload, sources: set[str] | None) -> bool:
    # Full source identities prevent a same-named test in another file matching.
    return any(workload.name == source or workload.name.startswith((source + "::", source + "["))
               for source in sources or ())


def _validate_dynamic_float_dtypes(
    operator: str,
    workloads: list[FlagGemsV62Workload],
    *,
    require_bfloat16: bool,
    require_primary_pair: bool = False,
    fixed_sources: set[str] | None = None,
) -> None:
    required_dtypes: set[str] = set()
    if require_primary_pair:
        required_dtypes.update(_PRIMARY_FLOAT_DTYPES)
    if require_bfloat16:
        required_dtypes.add("bfloat16")
    if not required_dtypes:
        return
    groups: dict[str, set[str]] = {}
    for workload in workloads:
        if _is_fixed_dtype_workload(workload, fixed_sources):
            continue
        dtypes = _nested_float_dtypes(workload.inputs)
        if len(dtypes) != 1:
            continue
        dtype = next(iter(dtypes))
        key = json.dumps(
            _replace_float_dtypes(workload.inputs),
            sort_keys=True,
            separators=(",", ":"),
        )
        groups.setdefault(key, set()).add(dtype)
    relevant_dtypes = set(required_dtypes)
    if require_bfloat16:
        relevant_dtypes.update(_PRIMARY_FLOAT_DTYPES)
    missing = [
        sorted(dtypes)
        for dtypes in groups.values()
        if dtypes & relevant_dtypes and not required_dtypes.issubset(dtypes)
    ]
    if missing:
        requirement = (
            "a bfloat16 counterpart"
            if require_bfloat16 and not require_primary_pair
            else f"counterparts for {sorted(required_dtypes)}"
        )
        raise ValueError(
            f"{operator}: accuracy uses target-dependent FLOAT_DTYPES and the "
            "target timing list requires every floating correctness "
            f"combination to have {requirement} "
            f"({len(missing)} combination(s) incomplete)"
        )


def _replace_dtype(value: Any, source_dtype: str, target_dtype: str) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                target_dtype
                if key == "dtype" and item == source_dtype
                else _replace_dtype(item, source_dtype, target_dtype)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_dtype(item, source_dtype, target_dtype) for item in value
        ]
    return value


def _dtype_neutral_inputs(value: Any) -> str:
    return json.dumps(
        _replace_float_dtypes(value),
        sort_keys=True,
        separators=(",", ":"),
    )


def _merge_expansion_tolerance(
    source: Tolerance | None,
    override: Tolerance | None,
) -> Tolerance | None:
    if override is None:
        return source
    if source is None:
        return override
    return source.model_copy(
        update={
            field: getattr(override, field)
            for field in override.model_fields_set
        }
    )


def _expand_dynamic_float_dtypes(
    operator: str,
    workloads: list[FlagGemsV62Workload],
    expansions: list[FlagGemsV62DtypeExpansion],
    *,
    require_bfloat16: bool,
    require_primary_pair: bool = False,
    target_dtypes: set[str] | None = None,
    fixed_sources: set[str] | None = None,
) -> list[FlagGemsV62Workload]:
    """Expand source-identical dtype rows without repeating workload matrices."""

    if not expansions:
        return workloads
    if any(workload.source_condition is not None for workload in workloads):
        raise ValueError("conditional dtype cases must be explicit; dtype expansion cannot infer their source conditions")
    if target_dtypes is not None:
        available = target_dtypes & _ALL_FLOAT_DTYPES
        allowed_pairs = {
            (source_dtype, target_dtype)
            for source_dtype in _ALL_FLOAT_DTYPES
            for target_dtype in available
            if source_dtype != target_dtype
        }
    else:
        allowed_pairs: set[tuple[str, str]] = set()
        if require_primary_pair:
            allowed_pairs.update(
                {
                    ("float16", "float32"),
                    ("float32", "float16"),
                }
            )
        if require_bfloat16:
            allowed_pairs.update(
                (source_dtype, "bfloat16")
                for source_dtype in _PRIMARY_FLOAT_DTYPES
            )
    if not allowed_pairs:
        raise ValueError(
            f"{operator}: dtype expansion requires source FLOAT_DTYPES "
            "counterparts present in the target timing list"
        )

    expanded = list(workloads)
    names = {workload.name for workload in expanded}
    neutral_dtypes: dict[str, set[str]] = {}
    for workload in expanded:
        if _is_fixed_dtype_workload(workload, fixed_sources):
            continue
        dtypes = _nested_float_dtypes(workload.inputs)
        if len(dtypes) == 1:
            neutral_dtypes.setdefault(
                _dtype_neutral_inputs(workload.inputs), set()
            ).update(dtypes)

    for expansion in expansions:
        if (expansion.source_dtype, expansion.target_dtype) not in allowed_pairs:
            raise ValueError(
                f"{operator}: unsupported correctness dtype expansion "
                f"{expansion.source_dtype!r}->{expansion.target_dtype!r}"
            )
        sources = [
            workload
            for workload in workloads
            if _nested_float_dtypes(workload.inputs) == {expansion.source_dtype}
            and not _is_fixed_dtype_workload(workload, fixed_sources)
        ]
        if not sources:
            raise ValueError(
                f"{operator}: dtype expansion source "
                f"{expansion.source_dtype!r} has no correctness workloads"
            )
        for source in sources:
            neutral = _dtype_neutral_inputs(source.inputs)
            if expansion.target_dtype in neutral_dtypes.get(neutral, set()):
                continue
            if expansion.source_dtype in source.name:
                name = source.name.replace(
                    expansion.source_dtype, expansion.target_dtype
                )
            else:
                name = f"{source.name}::dtype-{expansion.target_dtype}"
            if name in names:
                raise ValueError(
                    f"{operator}: dtype expansion creates duplicate correctness "
                    f"workload {name!r}"
                )
            clone = source.model_copy(
                update={
                    "name": name,
                    "inputs": _replace_dtype(
                        source.inputs,
                        expansion.source_dtype,
                        expansion.target_dtype,
                    ),
                    "tolerance": _merge_expansion_tolerance(
                        source.tolerance,
                        expansion.tolerance,
                    ),
                }
            )
            expanded.append(clone)
            names.add(name)
            neutral_dtypes.setdefault(neutral, set()).add(expansion.target_dtype)
    return expanded


def _requires_bfloat16_accuracy(
    inventory: FlagGemsSourceInventory,
    cases: list[dict[str, Any]],
    case_list_path: str | Path | None = None,
) -> bool:
    uses_float_dtypes = _uses_direct_float_dtype_parametrization(inventory)
    target_has_bfloat16 = any(
        str(case.get("dtype", "")).removeprefix("torch.") == "bfloat16"
        for case in cases
    )
    if not target_has_bfloat16 and case_list_path is not None:
        try:
            report = json.loads(
                Path(case_list_path).expanduser().resolve().read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError):
            report = {}
        target_has_bfloat16 = any(
            str(case.get("dtype", "")).removeprefix("torch.") == "bfloat16"
            for benchmark in report.get("benchmarks", [])
            if isinstance(benchmark, dict)
            for case in benchmark.get("cases", [])
            if isinstance(case, dict)
        )
    return uses_float_dtypes and target_has_bfloat16


def _requires_primary_float_dtype_pair(
    inventory: FlagGemsSourceInventory,
    cases: list[dict[str, Any]],
) -> bool:
    uses_float_dtypes = _uses_direct_float_dtype_parametrization(inventory)
    target_dtypes = {
        str(case.get("dtype", "")).removeprefix("torch.") for case in cases
    }
    return uses_float_dtypes and _PRIMARY_FLOAT_DTYPES.issubset(target_dtypes)


def _resolved_target_float_dtypes(
    cases: list[dict[str, Any]],
    *,
    require_bfloat16: bool,
) -> set[str]:
    """Return target-supported dtypes used to validate compact expansions.

    ``require_bfloat16`` may be inferred from another benchmark in the same
    target ``--list-cases`` report when the current operator has no bf16 timing
    row. Keep that target capability when validating this operator's
    correctness expansion instead of narrowing it back to its timing rows.
    """

    target_dtypes = {
        str(case.get("dtype", "")).removeprefix("torch.") for case in cases
    }
    if require_bfloat16:
        target_dtypes.add("bfloat16")
    return target_dtypes


def _uses_direct_float_dtype_parametrization(
    inventory: FlagGemsSourceInventory,
) -> bool:
    return bool(_dynamic_float_dtype_identities(inventory))


def _dynamic_float_dtype_identities(inventory: FlagGemsSourceInventory) -> set[str]:
    """Detect an independent ``dtype=FLOAT_DTYPES`` pytest axis.

    A combined axis such as ``shape,dtype=MVLGAMMA_CASES`` may deliberately
    filter shapes by dtype.  Treating any textual ``FLOAT_DTYPES`` reference as
    a full Cartesian matrix invents accuracy cases that pytest never runs.
    """

    identities = set()
    for path in inventory.test_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        # A source-local list may deliberately exclude bf16 or narrow QUICK_MODE.
        # Its spelling is not evidence of the shared, device-dependent helper.
        local_names = {
            node.id
            for statement in tree.body
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        for function in (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            markers = {
                marker
                for decorator in function.decorator_list
                if (
                    marker := _attribute_path(
                        decorator.func if isinstance(decorator, ast.Call) else decorator
                    )
                )
                is not None
                and len(marker) == 3
                and marker[:2] == ("pytest", "mark")
            }
            if ("pytest", "mark", inventory.operator) not in markers:
                continue
            for decorator in function.decorator_list:
                if not isinstance(decorator, ast.Call) or len(decorator.args) < 2:
                    continue
                if _attribute_path(decorator.func) != (
                    "pytest",
                    "mark",
                    "parametrize",
                ):
                    continue
                try:
                    names = ast.literal_eval(decorator.args[0])
                except (TypeError, ValueError):
                    continue
                dtype_source = _attribute_path(decorator.args[1])
                if dtype_source == ("FLOAT_DTYPES",) and "FLOAT_DTYPES" in local_names:
                    continue
                if names == "dtype" and dtype_source is not None and (
                    dtype_source[-1] == "FLOAT_DTYPES"
                ):
                    identities.add(f"{path.relative_to(inventory.repo).as_posix()}::{function.name}")
    return identities


_INTEGER_DTYPES = {"int8", "int16", "int32", "int64", "uint8"}


def _requires_full_range_integer_timing(
    inventory: FlagGemsSourceInventory,
    cases: list[dict[str, Any]],
) -> bool:
    """Detect the shared pointwise builder's CPU full-range randint path.

    A KGS ``random`` recipe intentionally uses a small fixed integer range and
    therefore cannot represent FlagGems' ``iinfo(dtype).min/max`` builder.
    Integer GCD/LCM latency is data dependent, so shape/dtype equality alone is
    insufficient for native/adapter timing parity.
    """

    case_dtypes = {
        str(case.get("dtype", "")).removeprefix("torch.") for case in cases
    }
    if not (case_dtypes & _INTEGER_DTYPES):
        return False
    benchmark_source = "\n".join(
        path.read_text(encoding="utf-8") for path in inventory.benchmark_files
    )
    uses_pointwise_builder = "BinaryPointwiseBenchmark(" in benchmark_source
    helper = inventory.helper_context
    return (
        uses_pointwise_builder
        and "torch.randint" in helper
        and "torch.iinfo(dtype).min" in helper
        and "torch.iinfo(dtype).max" in helper
        and ('device="cpu"' in helper or "device='cpu'" in helper)
    )


def _benchmark_uses_exported_operator_as_baseline(
    inventory: FlagGemsSourceInventory,
) -> bool:
    """Detect benchmarks whose timing baseline is the exported FlagGems op."""

    for path in inventory.benchmark_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for call in (
            node for node in ast.walk(tree) if isinstance(node, ast.Call)
        ):
            for keyword in call.keywords:
                if keyword.arg != "torch_op":
                    continue
                value = _attribute_path(keyword.value)
                if value == (checkout_profile(inventory.repo).package, inventory.operator):
                    return True
    return False


def _requires_implementation_float32_timing_promotion(
    inventory: FlagGemsSourceInventory,
) -> bool:
    """Detect a self-baselined FlagGems op with an fp16-to-fp32 path."""

    if not _benchmark_uses_exported_operator_as_baseline(inventory):
        return False

    try:
        implementation = ast.parse(
            inventory.implementation_file.read_text(encoding="utf-8"),
            filename=str(inventory.implementation_file),
        )
    except (OSError, SyntaxError):
        return False
    public_names = {
        signature.implementation_name for signature in inventory.public_signatures
    } or {inventory.operator}
    functions = [
        node
        for node in implementation.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in public_names
    ]
    for function in functions:
        paths = {
            path
            for node in ast.walk(function)
            if isinstance(node, ast.Attribute)
            and (path := _attribute_path(node)) is not None
        }
        if ("torch", "float16") not in paths or ("torch", "float32") not in paths:
            continue
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to"
            and any(
                _attribute_path(argument) == ("torch", "float32")
                for argument in node.args
            )
            for node in ast.walk(function)
        ):
            return True
    return False


def _preserves_float32_timing_promotion(function: ast.FunctionDef) -> bool:
    paths = {
        path
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and (path := _attribute_path(node)) is not None
    }
    has_float32_cast = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to"
        and any(
            _attribute_path(argument) == ("torch", "float32")
            for argument in node.args
        )
        for node in ast.walk(function)
    )
    casts_back_to_public_dtype = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to"
        and any(
            isinstance(argument, ast.Attribute) and argument.attr == "dtype"
            for argument in node.args
        )
        for node in ast.walk(function)
    )
    return (
        ("torch", "float16") in paths
        and ("torch", "float32") in paths
        and has_float32_cast
        and casts_back_to_public_dtype
    )


def _accuracy_sequence_comparison_owns_return_contract(
    inventory: FlagGemsSourceInventory,
    operator: str,
) -> bool:
    """Detect marked tests that compare sequence results by ``len`` and ``zip``."""

    for path in inventory.test_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ):
            markers = {
                _attribute_path(
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                for decorator in function.decorator_list
            }
            if ("pytest", "mark", operator) not in markers:
                continue
            length_pairs: set[frozenset[str]] = set()
            zip_pairs: set[frozenset[str]] = set()
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Compare)
                    and len(node.ops) == 1
                    and isinstance(node.ops[0], ast.Eq)
                    and len(node.comparators) == 1
                ):
                    names: list[str] = []
                    for expression in (node.left, node.comparators[0]):
                        if (
                            isinstance(expression, ast.Call)
                            and isinstance(expression.func, ast.Name)
                            and expression.func.id == "len"
                            and len(expression.args) == 1
                            and not expression.keywords
                            and isinstance(expression.args[0], ast.Name)
                        ):
                            names.append(expression.args[0].id)
                    if len(names) == 2:
                        length_pairs.add(frozenset(names))
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "zip"
                    and len(node.args) == 2
                    and not node.keywords
                    and all(isinstance(argument, ast.Name) for argument in node.args)
                ):
                    zip_pairs.add(frozenset(argument.id for argument in node.args))
            if length_pairs & zip_pairs:
                return True
    return False


def _accuracy_manual_seeds(
    inventory: FlagGemsSourceInventory,
    operator: str,
) -> dict[str, int]:
    """Extract literal ``torch.manual_seed`` calls from marked pytest functions."""

    seeds: dict[str, int] = {}
    for path in inventory.test_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ):
            markers = {
                _attribute_path(
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                for decorator in function.decorator_list
            }
            if ("pytest", "mark", operator) not in markers:
                continue
            values: set[int] = set()
            for call in (
                node for node in ast.walk(function) if isinstance(node, ast.Call)
            ):
                if _attribute_path(call.func) != ("torch", "manual_seed"):
                    continue
                if len(call.args) != 1 or call.keywords:
                    continue
                try:
                    value = ast.literal_eval(call.args[0])
                except (TypeError, ValueError):
                    continue
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    values.add(value)
            if len(values) > 1:
                raise ValueError(
                    f"{operator}: {function.name} uses multiple literal manual seeds"
                )
            if values:
                seeds[function.name] = next(iter(values))
    return seeds


def _normalize_accuracy_manual_seeds(
    workloads: list[FlagGemsV62Workload],
    seeds: dict[str, int],
) -> list[FlagGemsV62Workload]:
    if not seeds:
        return workloads
    normalized: list[FlagGemsV62Workload] = []
    for workload in workloads:
        matches = {
            name
            for name in seeds
            if any(
                segment.split("[", 1)[0] == name
                for segment in workload.name.split("::")
            )
        }
        if not matches and len(seeds) == 1:
            matches = set(seeds)
        if len(matches) > 1:
            raise ValueError(
                f"accuracy workload {workload.name!r} matches multiple seeded tests"
            )
        seed = seeds[next(iter(matches))] if matches else workload.seed
        normalized.append(workload.model_copy(update={"seed": seed}))
    return normalized


def _accuracy_dtype_tolerances(
    inventory: FlagGemsSourceInventory,
    operator: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Extract simple literal ``dtype == torch.X`` tolerance branches."""

    result: dict[str, dict[str, dict[str, float]]] = {}
    for path in inventory.test_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ):
            markers = {
                _attribute_path(
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                for decorator in function.decorator_list
            }
            if ("pytest", "mark", operator) not in markers:
                continue
            used_tolerances = {
                keyword.value.id
                for call in ast.walk(function)
                if isinstance(call, ast.Call)
                and _attribute_path(call.func) is not None
                and _attribute_path(call.func)[-1] in {"gems_assert_close", "assert_close"}
                for keyword in call.keywords
                if keyword.arg in {"atol", "rtol"}
                and isinstance(keyword.value, ast.Name)
            }
            for assignment in (
                node for node in ast.walk(function) if isinstance(node, ast.Assign)
            ):
                if (
                    len(assignment.targets) != 1
                    or not isinstance(assignment.targets[0], ast.Name)
                    or assignment.targets[0].id not in used_tolerances
                    or assignment.targets[0].id not in {"atol", "rtol"}
                    or not isinstance(assignment.value, ast.IfExp)
                ):
                    continue
                condition = assignment.value.test
                if (
                    not isinstance(condition, ast.Compare)
                    or len(condition.ops) != 1
                    or not isinstance(condition.ops[0], ast.Eq)
                    or len(condition.comparators) != 1
                ):
                    continue
                left, right = condition.left, condition.comparators[0]
                if isinstance(left, ast.Name) and left.id == "dtype":
                    dtype_path = _attribute_path(right)
                elif isinstance(right, ast.Name) and right.id == "dtype":
                    dtype_path = _attribute_path(left)
                else:
                    continue
                if (
                    dtype_path is None
                    or len(dtype_path) != 2
                    or dtype_path[0] != "torch"
                    or dtype_path[1] not in _ALL_FLOAT_DTYPES
                ):
                    continue
                try:
                    selected = float(ast.literal_eval(assignment.value.body))
                    fallback = float(ast.literal_eval(assignment.value.orelse))
                except (TypeError, ValueError):
                    continue
                field = assignment.targets[0].id
                for dtype in _ALL_FLOAT_DTYPES:
                    value = selected if dtype == dtype_path[1] else fallback
                    result.setdefault(function.name, {}).setdefault(dtype, {})[
                        field
                    ] = value
    return result


def _normalize_accuracy_dtype_tolerances(
    workloads: list[FlagGemsV62Workload],
    tolerances: dict[str, dict[str, dict[str, float]]],
) -> list[FlagGemsV62Workload]:
    if not tolerances:
        return workloads
    normalized: list[FlagGemsV62Workload] = []
    for workload in workloads:
        tests = {
            name
            for name in tolerances
            if any(
                segment.split("[", 1)[0] == name
                for segment in workload.name.split("::")
            )
        }
        if not tests and len(tolerances) == 1:
            tests = set(tolerances)
        dtypes = _nested_float_dtypes(workload.inputs)
        if len(tests) == 1 and len(dtypes) == 1:
            overrides = tolerances[next(iter(tests))].get(next(iter(dtypes)))
        else:
            overrides = None
        if overrides:
            tolerance = workload.tolerance or Tolerance()
            workload = workload.model_copy(
                update={"tolerance": tolerance.model_copy(update=overrides)}
            )
        normalized.append(workload)
    return normalized


def _validate_full_range_integer_timing_workloads(
    operator: str,
    workloads: list[FlagGemsV62Workload],
    cases: list[dict[str, Any]],
    *,
    required: bool,
) -> None:
    if not required:
        return
    for workload, case in zip(workloads, cases, strict=True):
        direct_random = [
            name
            for name, value in workload.inputs.items()
            if isinstance(value, dict) and value.get("type") == "random"
        ]
        context = workload.inputs.get("case")
        expected_dtype = str(case.get("dtype", "")).removeprefix("torch.")
        case_shape = case.get("shape")
        expected_shape = (
            case_shape.get("inputs") if isinstance(case_shape, dict) else None
        )
        if direct_random or not isinstance(context, dict):
            raise ValueError(
                f"{operator}/{workload.name}: FlagGems timing uses CPU "
                "full-range integer randint, which a KGS random recipe cannot "
                "represent; encode exact case context and materialize it in "
                f"gen_inputs (direct random inputs={direct_random})"
            )
        if (
            context.get("phase") != "timing"
            or context.get("dtype") != expected_dtype
            or context.get("shape") != expected_shape
            or context.get("params", {}) != case.get("params", {})
        ):
            raise ValueError(
                f"{operator}/{workload.name}: generated timing context must "
                "preserve phase, dtype, shape and params from --list-cases"
            )


def _normalize_full_range_integer_timing_contexts(
    workloads: list[FlagGemsV62Workload],
    cases: list[dict[str, Any]],
    *,
    required: bool,
) -> list[FlagGemsV62Workload]:
    """Fill authoritative generated timing metadata from ``--list-cases``."""

    if not required:
        return workloads
    normalized = []
    for workload, case in zip(workloads, cases, strict=True):
        context = workload.inputs.get("case")
        if not isinstance(context, dict):
            normalized.append(workload)
            continue
        inputs = dict(workload.inputs)
        case_shape = case.get("shape")
        input_shapes = (
            case_shape.get("inputs") if isinstance(case_shape, dict) else None
        )
        inputs["case"] = {
            "phase": "timing",
            "dtype": str(case.get("dtype", "")).removeprefix("torch."),
            "shape": input_shapes,
            "params": case.get("params", {}),
        }
        normalized.append(workload.model_copy(update={"inputs": inputs}))
    return normalized


def _normalize_implicit_softmax_timing_dims(
    operator: str,
    workloads: list[FlagGemsV62Workload],
    cases: list[dict[str, Any]],
) -> list[FlagGemsV62Workload]:
    """Resolve ``F.softmax(input)`` to the public operator's explicit dim.

    ``UnaryReductionBenchmark`` omits ``dim`` for a 1-D input and records empty
    params, while ``torch.nn.functional.softmax`` resolves that invocation to
    dimension zero before dispatch reaches FlagGems' required ``dim`` ABI.
    Preserve the benchmark call semantics in the native public workload.
    """

    if operator != "softmax":
        return workloads
    normalized = []
    for workload, case in zip(workloads, cases, strict=True):
        shape = case.get("shape")
        input_shape = shape.get("input") if isinstance(shape, dict) else None
        params = case.get("params", {})
        dim = workload.inputs.get("dim")
        if (
            isinstance(input_shape, list)
            and len(input_shape) == 1
            and isinstance(params, dict)
            and "dim" not in params
            and (
                dim is None
                or (
                    isinstance(dim, dict)
                    and dim.get("type") == "scalar"
                    and dim.get("value") is None
                )
            )
        ):
            inputs = dict(workload.inputs)
            inputs["dim"] = {"type": "scalar", "value": 0}
            workload = workload.model_copy(update={"inputs": inputs})
        normalized.append(workload)
    return normalized


def _validate_full_range_integer_timing_oracle(
    operator: str,
    source: str,
    *,
    required: bool,
    input_shape_count: int = 1,
) -> None:
    if not required:
        return
    module = ast.parse(source, filename=f"<{operator}/oracle.py>")
    function = top_level_functions(module).get("gen_inputs")
    if function is None:
        raise ValueError(
            f"{operator}: CPU full-range integer timing requires "
            "gen_inputs(ctx, device)"
        )
    context_keys = {
        node.slice.value
        for node in ast.walk(function)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    if "shapes" in context_keys:
        raise ValueError(
            f"{operator}: generated timing uses canonical "
            "ctx['inputs']['case']['shape']; do not introduce a parallel "
            "'shapes' field"
        )

    iinfo_aliases = {
        target.id
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "torch"
        and node.value.func.attr == "iinfo"
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    bound_aliases: dict[str, list[ast.expr]] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                bound_aliases.setdefault(target.id, []).append(node.value)
            elif (
                isinstance(target, (ast.Tuple, ast.List))
                and isinstance(node.value, (ast.Tuple, ast.List))
                and len(target.elts) == len(node.value.elts)
            ):
                for item, value in zip(target.elts, node.value.elts, strict=True):
                    if isinstance(item, ast.Name):
                        bound_aliases.setdefault(item.id, []).append(value)

    def subscript_index(value: ast.expr) -> str | int | None:
        if isinstance(value, ast.Subscript) and isinstance(
            value.slice, ast.Constant
        ):
            item = value.slice.value
            if isinstance(item, (str, int)) and not isinstance(item, bool):
                return item
        return None

    # ``shape.inputs`` stores one ordered shape per public Tensor argument.
    # Track simple aliases so ``tuple(shapes[0])`` is distinguishable from the
    # incorrect ``tuple(case["shape"])`` that treats the whole list as a size.
    shape_collection_aliases: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, values in bound_aliases.items():
            if name in shape_collection_aliases:
                continue
            if any(
                subscript_index(value) == "shape"
                or (
                    isinstance(value, ast.Name)
                    and value.id in shape_collection_aliases
                )
                for value in values
            ):
                shape_collection_aliases.add(name)
                changed = True

    def indexed_shape(value: ast.expr | None) -> int | None:
        if value is None:
            return None
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in {"tuple", "list"}
            and len(value.args) == 1
        ):
            return indexed_shape(value.args[0])
        if isinstance(value, ast.Subscript):
            index = subscript_index(value)
            if (
                isinstance(index, int)
                and isinstance(value.value, ast.Name)
                and value.value.id in shape_collection_aliases
            ):
                return index
        if isinstance(value, ast.Name):
            for alias_value in bound_aliases.get(value.id, []):
                index = indexed_shape(alias_value)
                if index is not None:
                    return index
        return None

    def iinfo_bound(
        value: ast.expr | None,
        bound: str,
        seen: frozenset[str] = frozenset(),
    ) -> bool:
        direct = bool(
            isinstance(value, ast.Attribute)
            and value.attr == bound
            and (
                (
                    isinstance(value.value, ast.Call)
                    and isinstance(value.value.func, ast.Attribute)
                    and isinstance(value.value.func.value, ast.Name)
                    and value.value.func.value.id == "torch"
                    and value.value.func.attr == "iinfo"
                )
                or (
                    isinstance(value.value, ast.Name)
                    and value.value.id in iinfo_aliases
                )
            )
        )
        if direct:
            return True
        if (
            isinstance(value, ast.Name)
            and value.id not in seen
            and value.id in bound_aliases
        ):
            return any(
                iinfo_bound(item, bound, seen | {value.id})
                for item in bound_aliases[value.id]
            )
        return False

    def phase_literal(node: ast.If) -> tuple[str, bool] | None:
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value in {"timing", "correctness"}
            and isinstance(test.ops[0], (ast.Eq, ast.NotEq))
        ):
            return None
        value = str(test.comparators[0].value)
        equals = isinstance(test.ops[0], ast.Eq)
        return value, equals

    explicit_timing_scopes: list[ast.AST] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        phase = phase_literal(node)
        if phase == ("timing", True) or phase == ("correctness", False):
            explicit_timing_scopes.extend(node.body)
        elif phase == ("timing", False) or phase == ("correctness", True):
            explicit_timing_scopes.extend(node.orelse)
    timing_scope_has_factory = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "randint"
        for root in explicit_timing_scopes
        for node in ast.walk(root)
    )
    search_roots: list[ast.AST] = (
        explicit_timing_scopes
        if timing_scope_has_factory
        else [function]
    )

    valid_factory = False
    materialized_shape_indices: set[int] = set()
    for root in search_roots:
        for node in ast.walk(root):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "randint"
            ):
                continue
            keywords = {item.arg: item.value for item in node.keywords if item.arg}
            low = keywords.get("low") or (
                node.args[0] if len(node.args) > 0 else None
            )
            high = keywords.get("high") or (
                node.args[1] if len(node.args) > 1 else None
            )
            device = keywords.get("device")
            if (
                iinfo_bound(low, "min")
                and iinfo_bound(high, "max")
                and isinstance(device, ast.Constant)
                and device.value == "cpu"
            ):
                valid_factory = True
                size = keywords.get("size") or (
                    node.args[2] if len(node.args) > 2 else None
                )
                index = indexed_shape(size)
                if index is not None:
                    materialized_shape_indices.add(index)
    if not valid_factory:
        raise ValueError(
            f"{operator}: gen_inputs() must reproduce timing with "
            "torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, "
            "..., device='cpu') before target transfer"
        )
    required_shape_indices = set(range(input_shape_count))
    if input_shape_count > 1 and not required_shape_indices.issubset(
        materialized_shape_indices
    ):
        raise ValueError(
            f"{operator}: full-range timing has {input_shape_count} ordered "
            "input shapes; gen_inputs() must materialize each Tensor from "
            "case['shape'][0], case['shape'][1], ... separately (possibly "
            "through simple aliases), not use the entire shape list as one "
            f"Tensor size; observed indices={sorted(materialized_shape_indices)}"
        )


def _requires_source_torch_fallback(
    inventory: FlagGemsSourceInventory,
) -> bool:
    """Detect an accuracy helper with both primary and vendor Torch paths."""

    def returned_torch_paths(statements: list[ast.stmt]) -> set[str]:
        if not any(
            isinstance(node, ast.Return)
            for statement in statements
            for node in ast.walk(statement)
        ):
            return set()
        return {
            path[-1]
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Attribute)
            and (path := _attribute_path(node)) is not None
            and path[0] == "torch"
        }

    def has_vendor_guard(node: ast.If) -> bool:
        return any(
            isinstance(child, ast.Attribute)
            and child.attr == "vendor_name"
            and isinstance(child.value, ast.Name)
            and child.value.id == checkout_profile(inventory.repo).package
            for child in ast.walk(node.test)
        )

    for path in inventory.test_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            # A vendor skip plus ordinary setup/reference Torch calls is not a
            # fallback.  Require the guarded branch itself to return a Torch
            # computation and a distinct returned Torch path to remain in its
            # ``else`` or the following statements.
            for index, statement in enumerate(function.body):
                if not isinstance(statement, ast.If) or not has_vendor_guard(
                    statement
                ):
                    continue
                guarded_paths = returned_torch_paths(statement.body)
                alternate_paths = returned_torch_paths(
                    statement.orelse or function.body[index + 1 :]
                )
                if (
                    guarded_paths
                    and alternate_paths
                    and guarded_paths != alternate_paths
                ):
                    return True
    return False


def _validate_conj_workloads(
    operator: str,
    correctness: list[FlagGemsV62Workload],
    timing: list[FlagGemsV62Workload],
) -> None:
    if operator != "conj":
        return
    for workload in correctness:
        case = workload.inputs.get("case")
        dtype = case.get("dtype") if isinstance(case, dict) else None
        shape = case.get("shape") if isinstance(case, dict) else None
        if isinstance(shape, dict):
            shape = shape.get("input")
        if "input" in workload.inputs or dtype not in {"complex32", "complex64"}:
            raise ValueError(
                f"{operator}/{workload.name}: correctness must preserve pytest's "
                "float32 real/imag construction through case + gen_inputs; direct "
                "complex random recipes use a different distribution"
            )
        if not isinstance(shape, list) or any(
            not isinstance(size, int) or size < 0 for size in shape
        ):
            raise ValueError(
                f"{operator}/{workload.name}: correctness case needs a valid shape"
            )
    for workload in timing:
        recipe = workload.inputs.get("input")
        if not (
            isinstance(recipe, dict)
            and recipe.get("type") == "random"
            and recipe.get("dtype") == "complex64"
        ):
            raise ValueError(
                f"{operator}/{workload.name}: timing must preserve the benchmark's "
                "direct complex64 random input recipe"
            )


def _validate_broadcast_tensors_shapes(
    operator: str,
    workloads: list[FlagGemsV62Workload],
) -> None:
    if operator != "broadcast_tensors":
        return
    missing_metadata: list[tuple[str, str]] = []
    for workload in workloads:
        case = workload.inputs.get("case")
        phase = case.get("phase") if isinstance(case, dict) else None
        shape = case.get("shape") if isinstance(case, dict) else None
        input_shapes = shape.get("inputs") if isinstance(shape, dict) else None
        output_shape = shape.get("output") if isinstance(shape, dict) else None
        if not isinstance(input_shapes, list) or not isinstance(output_shape, list):
            missing_metadata.append(
                (workload.name, phase if isinstance(phase, str) else "unknown")
            )
    if missing_metadata:
        phases = sorted({phase for _, phase in missing_metadata})
        examples = [name for name, _ in missing_metadata[:3]]
        raise ValueError(
            f"{operator}: all Tensor-container workloads must record "
            "shape.inputs and shape.output; "
            f"{len(missing_metadata)} workload(s) are missing metadata across "
            f"phases {phases}, including {examples}"
        )

    for workload in workloads:
        case = workload.inputs.get("case")
        shape = case.get("shape") if isinstance(case, dict) else None
        input_shapes = shape.get("inputs") if isinstance(shape, dict) else None
        output_shape = shape.get("output") if isinstance(shape, dict) else None
        assert isinstance(input_shapes, list)
        assert isinstance(output_shape, list)
        if not input_shapes or any(not isinstance(value, list) for value in input_shapes):
            raise ValueError(
                f"{operator}/{workload.name}: shape.inputs must be non-empty lists"
            )
        rank = max(len(value) for value in input_shapes)
        result = [1] * rank
        for value in input_shapes:
            padded = [1] * (rank - len(value)) + value
            for axis, size in enumerate(padded):
                if not isinstance(size, int) or size < 0:
                    raise ValueError(
                        f"{operator}/{workload.name}: invalid input shape {value}"
                    )
                if result[axis] == 1:
                    result[axis] = size
                elif size == 1 or result[axis] == size:
                    continue
                else:
                    raise ValueError(
                        f"{operator}/{workload.name}: input shapes do not broadcast"
                    )
        if result != output_shape:
            raise ValueError(
                f"{operator}/{workload.name}: input shapes broadcast to {result}, "
                f"not recorded output {output_shape}"
            )


def _validate_broadcast_tensors_run(
    operator: str,
    function: ast.FunctionDef | None,
) -> None:
    if operator != "broadcast_tensors":
        return
    if function is None:
        raise ValueError(f"{operator}: a shared run() is required")
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    valid = (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and isinstance(body[0].value, ast.Call)
        and isinstance(body[0].value.func, ast.Attribute)
        and isinstance(body[0].value.func.value, ast.Name)
        and body[0].value.func.value.id == "torch"
        and body[0].value.func.attr == "broadcast_tensors"
        and len(body[0].value.args) == 1
        and isinstance(body[0].value.args[0], ast.Starred)
        and isinstance(body[0].value.args[0].value, ast.Name)
        and body[0].value.args[0].value.id == "tensors"
        and not body[0].value.keywords
    )
    if not valid:
        raise ValueError(
            f"{operator}: run() must directly return "
            "torch.broadcast_tensors(*tensors); do not copy FlagGems input "
            "normalization or output conversion into the Torch oracle"
        )


class FlagGemsV62ExtractorAgent(BaseAgent):
    """Extract native V6.2 assets from FlagGems source only."""

    name = "flaggems_v62_extractor"
    InputModel = FlagGemsV62ExtractorInput
    OutputModel = FlagGemsV62ExtractorOutput

    def preprocess(self, inp: FlagGemsV62ExtractorInput, runtime) -> str:
        inventory = collect_source_inventory(inp.flaggems_repo, inp.operator)
        definition = extract_flaggems_definition(inp.flaggems_repo, inp.operator)
        cases = load_flaggems_timing_cases(inp.case_list_path, inp.operator)
        self._source_package = checkout_profile(inventory.repo).package
        self._definition = definition
        self._timing_cases = cases
        self._operator = inp.operator
        self._require_bfloat16_accuracy = _requires_bfloat16_accuracy(
            inventory, cases, inp.case_list_path
        )
        self._require_primary_float_dtype_pair = (
            _requires_primary_float_dtype_pair(inventory, cases)
        )
        self._require_torch_fallback = _requires_source_torch_fallback(inventory)
        self._require_full_range_integer_timing = (
            _requires_full_range_integer_timing(inventory, cases)
        )
        self._require_float32_timing_promotion = (
            _requires_implementation_float32_timing_promotion(inventory)
        )
        self._allow_flaggems_timing_reference = (
            inp.timing_reference == "flaggems" or _benchmark_uses_exported_operator_as_baseline(inventory)
        )
        self._accuracy_coverage_plan = _accuracy_coverage_plan(
            inventory, inp.operator
        )
        self._fixed_float_dtype_sources = (
            set(self._accuracy_coverage_plan.identities) - _dynamic_float_dtype_identities(inventory)
        )
        self._accuracy_tests = set(
            self._accuracy_coverage_plan.marked_functions
        )
        self._all_accuracy_tests = set(
            self._accuracy_coverage_plan.all_test_functions
        )
        self._accuracy_test_identities = set(
            self._accuracy_coverage_plan.identities
        )
        self._required_accuracy_test_identities = set(
            self._accuracy_coverage_plan.required_identities
        )
        self._excluded_accuracy_test_identities = dict(
            self._accuracy_coverage_plan.excluded_identities
        )
        self._accuracy_manual_seeds = _accuracy_manual_seeds(
            inventory, inp.operator
        )
        self._accuracy_dtype_tolerances = _accuracy_dtype_tolerances(
            inventory, inp.operator
        )
        self._require_valid_owned_return_contract = (
            _accuracy_sequence_comparison_owns_return_contract(
                inventory, inp.operator
            )
        )

        public_definition = definition.model_dump(mode="json", exclude_unset=True)
        public_definition["api_version"] = "v6.2"
        context = [
            "## Authoritative direct-extraction context",
            f"- operator: `{inp.operator}`",
            "- source policy: only the FlagGems files listed below are generation inputs; "
            "do not search or read any KernelGen/KernelGen Server catalog. Do not probe "
            "the Agent host's Torch installation, Python environments or filesystem to "
            "guess target behavior: mirror the injected source call exactly and leave "
            "target API/dtype readiness to the later Server runtime validation.",
            _source_paths(inventory),
            "- deterministic public Definition (already extracted from the FlagGems export; "
            "do not emit or reinterpret it):\n```json\n"
            + json.dumps(public_definition, indent=2, ensure_ascii=False)
            + "\n```",
            "- authoritative target core timing cases. Emit one timing Workload per "
            "case with exactly this count, order and case_id sequence. Python will "
            "validate the sequence before persistence. Materialize each case by "
            "tracing the benchmark builder:\n```json\n"
            + json.dumps(cases, indent=2, ensure_ascii=False)
            + "\n```",
        ]
        if inventory.helper_context:
            context.append(
                "- directly referenced FlagGems accuracy helpers:\n```python\n"
                + inventory.helper_context
                + "\n```"
            )
        if self._fixed_float_dtype_sources and (
            self._require_primary_float_dtype_pair or self._require_bfloat16_accuracy
        ):
            context.append(
                "- dtype expansion excludes these source tests; preserve their own dtype "
                "scope and emit their cases explicitly with full source identities: "
                + json.dumps(sorted(self._fixed_float_dtype_sources))
            )
        if self._require_primary_float_dtype_pair:
            target_dtypes = ["float16", "float32"]
            if self._require_bfloat16_accuracy:
                target_dtypes.append("bfloat16")
            context.append(
                "- OPTIONAL compact expansion for UNCONDITIONAL cases only; "
                "if any source dtype is conditional, instead emit every row explicitly "
                "with its source_condition and omit all expansion rules. "
                "For unconditional cases, the accuracy pytest uses "
                "`FLOAT_DTYPES` and the target timing list resolves "
                f"{target_dtypes}. Emit the full correctness parameter matrix "
                "for exactly one of float16 or float32, then add "
                "`correctness_dtype_expansions` entries from that chosen source "
                "dtype to every other listed dtype. Python will mechanically "
                "clone and validate all counterparts. A null expansion tolerance "
                "inherits each source workload's tolerance; an object overrides "
                "only its explicitly supplied fields. Put target-specific pytest "
                "differences on the expansion entry. Do not enumerate the cloned "
                "workloads."
            )
        elif self._require_bfloat16_accuracy:
            context.append(
                "- Only for UNCONDITIONAL cases (conditional source dtypes must "
                "all be explicit with source_condition and no expansion rules): "
                "the accuracy pytest uses "
                "`FLOAT_DTYPES` and the target timing list contains bfloat16; "
                "emit only the source primary floating correctness combinations, "
                "then add one `correctness_dtype_expansions` rule from a primary "
                "dtype to bfloat16. Python will mechanically clone the whole "
                "matrix and validate every counterpart. A null expansion tolerance "
                "inherits each source workload's tolerance; an object overrides "
                "only its explicitly supplied fields. Put the pytest's explicit "
                "bfloat16 differences on that rule. Do not enumerate the cloned "
                "bfloat16 workloads."
            )
        context.append(
            "- Torch fallback boundary: a dtype upcast/downcast chosen by "
            "`to_reference` (including fp64-versus-fp32 capability branches) "
            "is part of correctness reference precision, not evidence for a "
            "whole-round `torch_run`. Emit `torch_run` only when this context "
            "explicitly states a source fallback requirement."
        )
        if self._require_torch_fallback:
            context.append(
                "- source fallback requirement: an accuracy reference helper "
                "contains a vendor/capability-specific pure Torch alternative "
                "to its primary framework primitive. Keep the primary path in "
                "run/correctness_run/timing_run and expose the semantically "
                "equivalent pure Torch path as ABI-identical `torch_run`. The "
                "fallback is selected for the whole round when primary readiness "
                "fails; do not omit it because the current vendor is unknown."
            )
        if self._require_full_range_integer_timing:
            context.append(
                "- timing input constraint: the selected FlagGems pointwise "
                "builder generates integer tensors over the complete dtype "
                "range with `torch.randint(iinfo.min, iinfo.max, "
                "device='cpu').to(device)`. A KGS `random` recipe uses a "
                "different fixed range and allocation path, so do not emit "
                "direct random timing recipes. Store each authoritative case "
                "as `inputs.case` with phase=`timing`, exact dtype, "
                "`shape=<ordered input-shape list from shape.inputs>`, and params; "
                "then make `gen_inputs` reproduce the CPU "
                "full-range randint path. For N public Tensor inputs, read "
                "that ordered list and construct input i with "
                "`tuple(case['shape'][i])`; never wrap the whole ordered list "
                "in `tuple(...)` or reuse it as one Tensor size. Branch on "
                "phase so accuracy keeps its own source bounds."
            )
        if self._allow_flaggems_timing_reference:
            context.append(
                "- exact self-baseline policy: the selected timing reference is "
                "the exported FlagGems operator itself. Import "
                f"`{self._source_package}` and make ABI-identical `timing_run` call exactly "
                f"`{self._source_package}.{inp.operator}(...)`. This is the sole allowed "
                "FlagGems dependency and is timing-only. Keep "
                "`correctness_run` as the pure Torch semantics from pytest; "
                "do not call FlagGems there and do not translate the timing "
                "baseline into an inferred Torch formula."
            )
        if self._accuracy_tests:
            context.append(
                "- accuracy function scope: emit correctness workloads only "
                f"from pytest functions marked `pytest.mark.{inp.operator}`: "
                f"{sorted(self._required_accuracy_test_identities)}. Emit at "
                "least one workload from every required source and preserve the "
                "full `relative/path.py::function` identity in Workload.name. "
                "Other test functions in the same source file belong to sibling "
                "operator Definitions and must not be included."
            )
        if self._excluded_accuracy_test_identities:
            context.append(
                "- excluded accuracy sources: do not emit correctness workloads "
                "for these source functions because v6.2 cannot faithfully "
                "represent their assertion contract: "
                f"{self._excluded_accuracy_test_identities}. Do not replace an "
                "autograd-only test with a forward-output comparison or invent "
                "a public backward ABI."
            )
        if self._accuracy_coverage_plan.partial_identities:
            context.append(
                "- partially representable accuracy sources: emit their source "
                "forward-output cases, but do not claim to reproduce these "
                "additional contracts: "
                f"{self._accuracy_coverage_plan.partial_identities}. Python will "
                "record the uncovered contracts in the extraction coverage report."
            )
        if self._accuracy_manual_seeds:
            context.append(
                "- accuracy seed contract: marked pytest functions set literal "
                f"manual seeds {self._accuracy_manual_seeds}. Use these exact "
                "values as the corresponding correctness Workload.seed; Python "
                "will normalize them before persistence."
            )
        if self._accuracy_dtype_tolerances:
            context.append(
                "- accuracy dtype tolerance contract: Python extracted literal "
                f"dtype branches {self._accuracy_dtype_tolerances} and will "
                "apply them mechanically after dtype expansion. Do not broaden "
                "a float16-only tolerance to bfloat16."
            )
        if self._require_valid_owned_return_contract:
            context.append(
                "- return contract: the marked accuracy pytest explicitly "
                "checks equal sequence length and then compares elements with "
                "zip, so list and tuple containers are intentionally "
                "equivalent. Preserve the pytest Torch return in the run "
                "function, declare `VALID_OWNS_RETURN_CONTRACT = True`, and "
                "define exact-signature `valid(ref_outputs, sol_outputs, "
                "inputs, ctx)` that reproduces every source assertion: length, "
                "Tensor leaf type, shape, dtype and element values. Return a "
                "boolean or verdict mapping; do not accept mismatched leaves."
            )
        return "\n\n".join(
            [
                self._role_for(runtime),
                "\n".join(context),
                "--- OUTPUT CONTRACT ---\n" + render_contract(self.OutputModel),
            ]
        )

    def postprocess(self, raw: str, runtime) -> FlagGemsV62ExtractorOutput:
        output = self.OutputModel.model_validate(extract_json(raw))
        if output.blocker is not None:
            return output
        definition: FlagGemsDefinitionSpec | None = getattr(self, "_definition", None)
        cases: list[dict[str, Any]] | None = getattr(self, "_timing_cases", None)
        operator: str | None = getattr(self, "_operator", None)
        if definition is None or cases is None or operator is None:
            raise ValueError("direct extractor source context is missing")

        output = output.model_copy(
            update={
                "correctness_workloads": _drop_redundant_strict_tolerances(
                    _normalize_accuracy_dtype_tolerances(
                        _expand_dynamic_float_dtypes(
                            operator,
                            _normalize_accuracy_manual_seeds(
                                output.correctness_workloads,
                                dict(getattr(self, "_accuracy_manual_seeds", {})),
                            ),
                            output.correctness_dtype_expansions,
                            fixed_sources=getattr(self, "_fixed_float_dtype_sources", None),
                            require_bfloat16=bool(
                                getattr(self, "_require_bfloat16_accuracy", False)
                            ),
                            require_primary_pair=bool(
                                getattr(
                                    self,
                                    "_require_primary_float_dtype_pair",
                                    False,
                                )
                            ),
                            target_dtypes=_resolved_target_float_dtypes(
                                cases,
                                require_bfloat16=bool(
                                    getattr(
                                        self,
                                        "_require_bfloat16_accuracy",
                                        False,
                                    )
                                ),
                            ),
                        ),
                        dict(
                            getattr(self, "_accuracy_dtype_tolerances", {})
                        ),
                    ),
                ),
                "timing_workloads": _normalize_implicit_softmax_timing_dims(
                    operator,
                    _normalize_full_range_integer_timing_contexts(
                        output.timing_workloads,
                        cases,
                        required=bool(
                            getattr(
                                self, "_require_full_range_integer_timing", False
                            )
                        ),
                    ),
                    cases,
                ),
            }
        )

        names: set[str] = set()
        for workload in output.correctness_workloads:
            if workload.name in names:
                raise ValueError(f"{operator}: duplicate correctness workload {workload.name!r}")
            names.add(workload.name)
        _validate_accuracy_workload_scope(
            operator,
            output.correctness_workloads,
            allowed=set(getattr(self, "_accuracy_tests", set())),
            all_tests=set(getattr(self, "_all_accuracy_tests", set())),
            required_identities=set(
                getattr(self, "_required_accuracy_test_identities", set())
            ),
            excluded_identities=dict(
                getattr(self, "_excluded_accuracy_test_identities", {})
            ),
        )
        _validate_timing_workloads(operator, output.timing_workloads, cases)
        full_range_integer_timing = bool(
            getattr(self, "_require_full_range_integer_timing", False)
        )
        _validate_full_range_integer_timing_workloads(
            operator,
            output.timing_workloads,
            cases,
            required=full_range_integer_timing,
        )
        _validate_recipe_types(
            operator,
            [*output.correctness_workloads, *output.timing_workloads],
        )
        _validate_random_recipe_devices(
            operator,
            [*output.correctness_workloads, *output.timing_workloads],
        )
        _validate_dynamic_float_dtypes(
            operator,
            output.correctness_workloads,
            fixed_sources=getattr(self, "_fixed_float_dtype_sources", None),
            require_bfloat16=bool(
                getattr(self, "_require_bfloat16_accuracy", False)
            ),
            require_primary_pair=bool(
                getattr(self, "_require_primary_float_dtype_pair", False)
            ),
        )
        _validate_conj_workloads(
            operator, output.correctness_workloads, output.timing_workloads
        )
        _validate_broadcast_tensors_shapes(
            operator,
            [*output.correctness_workloads, *output.timing_workloads],
        )
        timing_names = {workload.name for workload in output.timing_workloads}
        overlap = names & timing_names
        if overlap:
            raise ValueError(f"{operator}: correctness/timing names overlap: {sorted(overlap)}")
        workloads = [*output.correctness_workloads, *output.timing_workloads]
        needs_gen_inputs, mixes_generated_and_direct_inputs = (
            _gen_inputs_requirements(definition, workloads)
        )
        _validate_oracle(
            operator,
            definition,
            output.oracle,
            has_correctness=bool(output.correctness_workloads),
            has_timing=bool(output.timing_workloads),
            needs_gen_inputs=needs_gen_inputs,
            mixes_generated_and_direct_inputs=mixes_generated_and_direct_inputs,
            needs_torch_fallback=bool(
                getattr(self, "_require_torch_fallback", False)
            ),
            requires_float32_timing_promotion=bool(
                getattr(self, "_require_float32_timing_promotion", False)
            ),
            source_package=self._source_package,
            allows_flaggems_timing_reference=bool(
                getattr(self, "_allow_flaggems_timing_reference", False)
            ),
            requires_valid_owned_return_contract=bool(
                getattr(self, "_require_valid_owned_return_contract", False)
            ),
        )
        _validate_full_range_integer_timing_oracle(
            operator,
            output.oracle,
            required=full_range_integer_timing,
            input_shape_count=max(
                (
                    len(shape["inputs"])
                    for case in cases
                    if isinstance((shape := case.get("shape")), dict)
                    and isinstance(shape.get("inputs"), list)
                ),
                default=1,
            ),
        )
        return output


def build_flaggems_v62_accuracy_coverage(
    inp: FlagGemsV62ExtractorInput,
    output: FlagGemsV62ExtractorOutput,
) -> dict[str, Any]:
    """Build the workspace audit record without adding a Catalog asset."""

    inventory = collect_source_inventory(inp.flaggems_repo, inp.operator)
    plan = _accuracy_coverage_plan(inventory, inp.operator)
    return _accuracy_coverage_report(
        inp.operator,
        plan,
        output.correctness_workloads,
    )


def persist_flaggems_v62_extraction(
    inp: FlagGemsV62ExtractorInput,
    output: FlagGemsV62ExtractorOutput,
    catalog_root: str | Path,
    *,
    adapter_catalog_root: str | Path | None = None,
) -> FlagGemsV62DirectExtraction:
    """Persist a validated direct extraction in the KGS per-operator layout."""

    if output.blocker is not None:
        raise ValueError("blocked extraction cannot be published as a Catalog")
    root = Path(catalog_root).expanduser().resolve()
    definition = extract_flaggems_definition(inp.flaggems_repo, inp.operator)
    adapter_definition_path = None
    if adapter_catalog_root is not None:
        adapter_definition_path = _validate_adapter_definition_target(
            inp.operator,
            definition,
            adapter_catalog_root,
        )
    cases = load_flaggems_timing_cases(inp.case_list_path, inp.operator)
    inventory = collect_source_inventory(inp.flaggems_repo, inp.operator)
    require_bfloat16 = _requires_bfloat16_accuracy(
        inventory, cases, inp.case_list_path
    )
    require_primary_float_dtype_pair = _requires_primary_float_dtype_pair(
        inventory, cases
    )
    require_torch_fallback = _requires_source_torch_fallback(inventory)
    require_full_range_integer_timing = _requires_full_range_integer_timing(
        inventory, cases
    )
    require_float32_timing_promotion = (
        _requires_implementation_float32_timing_promotion(inventory)
    )
    allow_flaggems_timing_reference = (
        inp.timing_reference == "flaggems" or _benchmark_uses_exported_operator_as_baseline(inventory)
    )
    require_valid_owned_return_contract = (
        _accuracy_sequence_comparison_owns_return_contract(
            inventory, inp.operator
        )
    )
    accuracy_coverage = _accuracy_coverage_plan(inventory, inp.operator)
    fixed_float_dtype_sources = set(accuracy_coverage.identities) - _dynamic_float_dtype_identities(inventory)
    accuracy_manual_seeds = _accuracy_manual_seeds(inventory, inp.operator)
    accuracy_dtype_tolerances = _accuracy_dtype_tolerances(
        inventory, inp.operator
    )
    output = output.model_copy(
        update={
            "correctness_workloads": _drop_redundant_strict_tolerances(
                _normalize_accuracy_dtype_tolerances(
                    _expand_dynamic_float_dtypes(
                        inp.operator,
                        _normalize_accuracy_manual_seeds(
                            output.correctness_workloads,
                            accuracy_manual_seeds,
                        ),
                        output.correctness_dtype_expansions,
                        fixed_sources=fixed_float_dtype_sources,
                        require_bfloat16=require_bfloat16,
                        require_primary_pair=require_primary_float_dtype_pair,
                        target_dtypes=_resolved_target_float_dtypes(
                            cases,
                            require_bfloat16=require_bfloat16,
                        ),
                    ),
                    accuracy_dtype_tolerances,
                ),
            ),
            "timing_workloads": _normalize_implicit_softmax_timing_dims(
                inp.operator,
                _normalize_full_range_integer_timing_contexts(
                    output.timing_workloads,
                    cases,
                    required=require_full_range_integer_timing,
                ),
                cases,
            ),
        }
    )
    _validate_timing_workloads(inp.operator, output.timing_workloads, cases)
    _validate_accuracy_workload_scope(
        inp.operator,
        output.correctness_workloads,
        allowed=set(accuracy_coverage.marked_functions),
        all_tests=set(accuracy_coverage.all_test_functions),
        required_identities=set(accuracy_coverage.required_identities),
        excluded_identities=accuracy_coverage.excluded_identities,
    )
    _validate_full_range_integer_timing_workloads(
        inp.operator,
        output.timing_workloads,
        cases,
        required=require_full_range_integer_timing,
    )
    _validate_recipe_types(
        inp.operator,
        [*output.correctness_workloads, *output.timing_workloads],
    )
    _validate_random_recipe_devices(
        inp.operator,
        [*output.correctness_workloads, *output.timing_workloads],
    )
    _validate_dynamic_float_dtypes(
        inp.operator,
        output.correctness_workloads,
        fixed_sources=fixed_float_dtype_sources,
        require_bfloat16=require_bfloat16,
        require_primary_pair=require_primary_float_dtype_pair,
    )
    _validate_conj_workloads(
        inp.operator, output.correctness_workloads, output.timing_workloads
    )
    _validate_broadcast_tensors_shapes(
        inp.operator,
        [*output.correctness_workloads, *output.timing_workloads],
    )
    workloads = [*output.correctness_workloads, *output.timing_workloads]
    needs_gen_inputs, mixes_generated_and_direct_inputs = (
        _gen_inputs_requirements(definition, workloads)
    )
    _validate_oracle(
        inp.operator,
        definition,
        output.oracle,
        has_correctness=bool(output.correctness_workloads),
        has_timing=bool(output.timing_workloads),
        needs_gen_inputs=needs_gen_inputs,
        mixes_generated_and_direct_inputs=mixes_generated_and_direct_inputs,
        needs_torch_fallback=require_torch_fallback,
        requires_float32_timing_promotion=require_float32_timing_promotion,
        source_package=checkout_profile(inventory.repo).package,
        allows_flaggems_timing_reference=allow_flaggems_timing_reference,
        requires_valid_owned_return_contract=require_valid_owned_return_contract,
    )
    _validate_full_range_integer_timing_oracle(
        inp.operator,
        output.oracle,
        required=require_full_range_integer_timing,
        input_shape_count=max(
            (
                len(shape["inputs"])
                for case in cases
                if isinstance((shape := case.get("shape")), dict)
                and isinstance(shape.get("inputs"), list)
            ),
            default=1,
        ),
    )

    manifest = {
        "api_version": "v6.2",
        "evaluator": "native",
        "layout": "per-operator",
    }
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or any(
            existing.get(key) != value for key, value in manifest.items()
        ):
            raise ValueError(f"{inp.operator}: target is not a v6.2 native catalog")
    operator_root = root / "ops" / inp.operator
    if operator_root.exists():
        raise FileExistsError(
            f"{inp.operator}: refusing to overwrite existing operator directory "
            f"{operator_root}"
        )

    operator_root.mkdir(parents=True)
    if not manifest_path.is_file():
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    public_definition = definition.model_dump(mode="json", exclude_unset=True)
    public_definition["api_version"] = "v6.2"
    if output.source_policy_id:
        public_definition["source_policy_id"] = output.source_policy_id
    definition_path = operator_root / "definition.json"
    oracle_path = operator_root / "oracle.py"
    correctness_path = operator_root / "correctness.jsonl"
    timing_path = operator_root / "timing.jsonl"
    definition_path.write_text(
        json.dumps(public_definition, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    oracle_path.write_text(output.oracle.rstrip() + "\n", encoding="utf-8")
    correctness_path.write_text(
        "\n".join(
            workload.model_dump_json(exclude_none=True)
            for workload in output.correctness_workloads
        )
        + "\n",
        encoding="utf-8",
    )
    timing_path.write_text(
        "\n".join(
            workload.model_dump_json(exclude_none=True)
            for workload in output.timing_workloads
        )
        + "\n",
        encoding="utf-8",
    )
    if adapter_definition_path is not None:
        _write_adapter_definition(definition, adapter_definition_path)
    return FlagGemsV62DirectExtraction(
        catalog_root=root,
        operator_root=operator_root,
        definition_path=definition_path,
        oracle_path=oracle_path,
        correctness_path=correctness_path,
        timing_path=timing_path,
        adapter_definition_path=adapter_definition_path,
        num_correctness_workloads=len(output.correctness_workloads),
        num_timing_workloads=len(output.timing_workloads),
    )


def _adapter_definition_payload(
    definition: FlagGemsDefinitionSpec,
) -> dict[str, Any]:
    payload = definition.model_dump(mode="json", exclude_unset=True)
    payload["api_version"] = "v6.0"
    return payload


def _validate_adapter_definition_target(
    operator: str,
    definition: FlagGemsDefinitionSpec,
    adapter_catalog_root: str | Path,
) -> Path:
    root = Path(adapter_catalog_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{operator}: cannot read FlagGems adapter manifest: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or (
        manifest.get("api_version"), manifest.get("evaluator")
    ) != ("v6.0", "flaggems"):
        raise ValueError(
            f"{operator}: target is not a v6.0 FlagGems adapter catalog"
        )

    path = root / "definitions" / f"{operator}.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{operator}: cannot read existing adapter Definition: {exc}"
            ) from exc
        if existing != _adapter_definition_payload(definition):
            raise ValueError(
                f"{operator}: existing adapter Definition differs from the "
                "current FlagGems export"
            )
    return path


def _write_adapter_definition(
    definition: FlagGemsDefinitionSpec,
    path: Path,
) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _adapter_definition_payload(definition),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def ensure_flaggems_v6_adapter_definition(
    flaggems_repo: str | Path,
    operator: str,
    adapter_catalog_root: str | Path,
) -> Path:
    """Ensure the adapter catalog can bind the same public FlagGems export."""

    definition = extract_flaggems_definition(flaggems_repo, operator)
    path = _validate_adapter_definition_target(
        operator,
        definition,
        adapter_catalog_root,
    )
    _write_adapter_definition(definition, path)
    return path


__all__ = [
    "FlagGemsV62DirectExtraction",
    "FlagGemsV62ExtractorAgent",
    "FlagGemsV62ExtractorInput",
    "FlagGemsV62ExtractorOutput",
    "FlagGemsV62Workload",
    "build_flaggems_v62_accuracy_coverage",
    "ensure_flaggems_v6_adapter_definition",
    "load_flaggems_timing_cases",
    "persist_flaggems_v62_extraction",
]

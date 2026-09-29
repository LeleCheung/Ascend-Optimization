#!/usr/bin/env python3
"""Standalone validator for KGS v6.2 native catalogs.

The static path uses only the Python standard library.  ``--runtime`` imports
the catalog's trusted oracle in an isolated subprocess and therefore requires
whatever runtime packages and shared libraries that oracle itself requires.
This file deliberately does not import ``kernelgen_server``.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import importlib.util
import inspect
import json
import math
import multiprocessing
import os
import queue as queue_module
import random
import re
import subprocess
import sys
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


REPORT_SCHEMA = "kgs.v6.2-native-catalog-validation/v1"
HOOKS = {
    "run",
    "correctness_run",
    "timing_run",
    "torch_run",
    "gen_inputs",
    "torch_gen_inputs",
    "valid",
    "torch_valid",
}
PHASE_HOOKS = {"run", "correctness_run", "timing_run", "torch_run"}
FIXED_HOOK_SIGNATURES = {
    "gen_inputs": ("ctx", "device"),
    "torch_gen_inputs": ("ctx", "device"),
    "valid": ("ref_outputs", "sol_outputs", "inputs", "ctx"),
    "torch_valid": ("ref_outputs", "sol_outputs", "inputs", "ctx"),
}
CONTRACT_FLAGS = {
    "VALID_OWNS_RETURN_CONTRACT": "valid",
    "TORCH_VALID_OWNS_RETURN_CONTRACT": "torch_valid",
}
PARAMETER_KINDS = {
    "positional_only": inspect.Parameter.POSITIONAL_ONLY,
    "positional_or_keyword": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "var_positional": inspect.Parameter.VAR_POSITIONAL,
    "keyword_only": inspect.Parameter.KEYWORD_ONLY,
}
IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
DEFINITION_FIELDS = {
    "api_version",
    "name",
    "description",
    "parameters",
    "outputs",
    "effects",
}
PARAMETER_FIELDS = {"name", "kind", "required", "default", "type_hint"}
EFFECT_FIELDS = {"mutates", "returns_alias_of"}
WORKLOAD_FIELDS = {
    "name",
    "inputs",
    "seed",
    "tolerance",
    "input_path",
    "output_path",
}
TOLERANCE_FIELDS = {
    "rtol",
    "atol",
    "atol_scale",
    "required_matched_ratio",
    "equal_nan",
}
KNOWN_OPERATOR_FILES = {
    "definition.json",
    "oracle.py",
    "correctness.jsonl",
    "timing.jsonl",
    "correctness_full.jsonl",
    "timing_full.jsonl",
}
FORBIDDEN_REFERENCE_FILES = {
    "correctness_reference.py",
    "timing_reference.py",
    "torch_reference.py",
}
MISSING = object()


@dataclass
class Issue:
    severity: str
    code: str
    path: str
    message: str
    operator: str | None = None
    line: int | None = None


@dataclass
class FunctionSignature:
    name: str
    parameters: list[dict[str, Any]]
    line: int
    decorated: bool = False
    asynchronous: bool = False


@dataclass
class OperatorRecord:
    name: str
    relative: str
    root: Path
    definition: dict[str, Any]
    reference_device: str | None
    hooks: set[str] = field(default_factory=set)
    correctness: list[dict[str, Any]] = field(default_factory=list)
    timing: list[dict[str, Any]] = field(default_factory=list)
    asset_files: int = 0
    asset_bytes: int = 0


class Validator:
    def __init__(
        self,
        root: Path,
        *,
        mode: str,
        delivery: bool = False,
        selected_operators: set[str] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.mode = mode
        self.delivery = delivery
        self.selected_operators = selected_operators
        self.issues: list[Issue] = []
        self.operators: dict[str, OperatorRecord] = {}
        self._workload_names: dict[str, tuple[str, str, int]] = {}
        self.manifest: dict[str, Any] = {}

    def issue(
        self,
        severity: str,
        code: str,
        path: Path | str,
        message: str,
        *,
        operator: str | None = None,
        line: int | None = None,
    ) -> None:
        try:
            shown = Path(path).resolve().relative_to(self.root).as_posix()
        except (OSError, ValueError):
            shown = str(path)
        self.issues.append(Issue(severity, code, shown, message, operator, line))

    def error(self, code: str, path: Path | str, message: str, **details: Any) -> None:
        self.issue("error", code, path, message, **details)

    def warning(
        self, code: str, path: Path | str, message: str, **details: Any
    ) -> None:
        self.issue("warning", code, path, message, **details)

    def _read_json(self, path: Path, *, label: str) -> Any:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.error("file.missing", path, f"missing {label}")
            return MISSING
        except UnicodeDecodeError as exc:
            self.error("file.encoding", path, f"{label} is not UTF-8: {exc}")
            return MISSING
        except json.JSONDecodeError as exc:
            self.error(
                "json.invalid",
                path,
                f"invalid {label}: {exc.msg}",
                line=exc.lineno,
            )
            return MISSING
        bad = _find_nonfinite(value)
        if bad is not None:
            self.error("json.nonfinite", path, f"non-finite JSON number at {bad}")
        return value

    def validate(self) -> None:
        if not self.root.is_dir():
            self.error("delivery.missing", self.root, "input directory does not exist")
            return
        if self.delivery:
            own_definition = self.root / "definition.json"
            if own_definition.is_file():
                ops_root = self.root.parent
                definition_paths = [own_definition]
            else:
                ops_root = self.root
                definition_paths = sorted(self.root.rglob("definition.json"))
        else:
            self._validate_manifest()
            ops_root = self.root / "ops"
            if not ops_root.is_dir():
                self.error("catalog.ops_missing", ops_root, "missing ops directory")
                return
            definition_paths = sorted(ops_root.rglob("definition.json"))
        if not definition_paths:
            self.error("delivery.empty", ops_root, "input contains no definitions")
            return
        for path in definition_paths:
            self._validate_operator(path, ops_root)
        if self.selected_operators:
            unknown = self.selected_operators.difference(self.operators)
            for name in sorted(unknown):
                self.error(
                    "catalog.unknown_operator",
                    ops_root,
                    f"selected operator does not exist: {name}",
                    operator=name,
                )

    def _validate_manifest(self) -> None:
        path = self.root / "manifest.json"
        raw = self._read_json(path, label="manifest.json")
        if raw is MISSING:
            return
        if not isinstance(raw, dict):
            self.error("manifest.type", path, "manifest.json must be a JSON object")
            return
        self.manifest = raw
        expected = {
            "api_version": "v6.2",
            "evaluator": "native",
            "layout": "per-operator",
        }
        for key, value in expected.items():
            if raw.get(key) != value:
                self.error(
                    f"manifest.{key}",
                    path,
                    f"{key} must be {value!r}, got {raw.get(key)!r}",
                )
        name = raw.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            self.error("manifest.name", path, "name must be a non-empty string")
        if self.mode == "standard":
            if name is None:
                self.warning(
                    "manifest.name_missing",
                    path,
                    "standard delivery should declare a catalog name",
                )
            elif name != self.root.name:
                self.warning(
                    "manifest.name_mismatch",
                    path,
                    f"manifest name {name!r} differs from directory {self.root.name!r}",
                )
        framework = raw.get("framework")
        repository = raw.get("framework_repository")
        branch = raw.get("framework_branch")
        revision = raw.get("framework_revision")
        if framework not in {None, "flaggems"}:
            self.error(
                "manifest.framework",
                path,
                "v6.2 native framework must be 'flaggems' when set",
            )
        if framework == "flaggems":
            if not isinstance(repository, str) or not repository.strip():
                self.error(
                    "manifest.framework_repository",
                    path,
                    "FlagGems-backed catalogs require framework_repository",
                )
            if not isinstance(branch, str) or not branch.strip():
                self.error(
                    "manifest.framework_branch",
                    path,
                    "FlagGems-backed catalogs require framework_branch",
                )
            if (
                not isinstance(revision, str)
                or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            ):
                self.error(
                    "manifest.framework_revision",
                    path,
                    "FlagGems-backed catalogs require an exact 40-character lowercase commit hash",
                )
            if "framework_base_revision" in raw or (
                "compatible_framework_revisions" in raw
            ):
                self.error(
                    "manifest.framework_revision_policy",
                    path,
                    "FlagGems-backed catalogs require one exact framework_revision",
                )
        if framework is None and any(
            value is not None for value in (repository, branch, revision)
        ):
            self.warning(
                "manifest.unbound_revision",
                path,
                "framework binding fields are present without framework",
            )

    def _validate_operator(self, path: Path, ops_root: Path) -> None:
        parent = path.parent
        relative = parent.relative_to(ops_root)
        parts = relative.parts
        provisional_name = parent.name
        if len(parts) not in {1, 2}:
            self.error(
                "operator.path",
                path,
                "operator path must be ops/<name> or ops/<group>/<name>",
                operator=provisional_name,
            )
        raw = self._read_json(path, label="definition.json")
        if raw is MISSING or not isinstance(raw, dict):
            if raw is not MISSING:
                self.error(
                    "definition.type",
                    path,
                    "definition.json must be a JSON object",
                    operator=provisional_name,
                )
            return
        name = raw.get("name")
        operator = name if isinstance(name, str) and name else provisional_name
        if self.selected_operators and operator not in self.selected_operators:
            return
        if parent.name != name:
            self.error(
                "operator.directory_name",
                path,
                f"operator directory {parent.name!r} must equal Definition.name {name!r}",
                operator=operator,
            )
        if operator in self.operators:
            self.error(
                "definition.duplicate_name",
                path,
                f"duplicate Definition.name: {operator}",
                operator=operator,
            )
            return
        record = OperatorRecord(operator, relative.as_posix(), parent, raw, None)
        self.operators[operator] = record
        self._validate_assets(record)
        self._validate_definition(record, path)
        self._validate_oracle(record)
        record.correctness = self._validate_workloads(record, "correctness")
        record.timing = self._validate_workloads(record, "timing")
        for phase, active in (
            ("correctness", record.correctness),
            ("timing", record.timing),
        ):
            archived = self._validate_workloads(record, phase, full=True)
            if archived:
                active_names = {
                    workload.get("name")
                    for workload in active
                    if isinstance(workload.get("name"), str)
                }
                archived_names = {
                    workload.get("name")
                    for workload in archived
                    if isinstance(workload.get("name"), str)
                }
                missing = active_names.difference(archived_names)
                if missing:
                    self.error(
                        "workload.archive_missing_active",
                        parent / f"{phase}_full.jsonl",
                        f"archive does not contain {len(missing)} active workloads",
                        operator=operator,
                    )
        if not record.correctness and not record.timing:
            self.error(
                "workload.none",
                parent,
                "operator must have at least one correctness or timing workload",
                operator=operator,
            )
        if self.mode == "standard":
            if not record.correctness:
                self.error(
                    "workload.correctness_required",
                    parent / "correctness.jsonl",
                    "standard delivery requires non-empty correctness.jsonl",
                    operator=operator,
                )
            if not record.timing:
                self.error(
                    "workload.timing_required",
                    parent / "timing.jsonl",
                    "standard delivery requires non-empty timing.jsonl",
                    operator=operator,
                )
        self._validate_phase_contract(record)
        self._validate_materialization_contract(record)

    def _validate_assets(self, record: OperatorRecord) -> None:
        for child in record.root.iterdir():
            if child.name in KNOWN_OPERATOR_FILES or child.name == "assets":
                continue
            if child.name in FORBIDDEN_REFERENCE_FILES:
                self.error(
                    "oracle.split_reference",
                    child,
                    "v6.2 keeps all reference entrypoints in oracle.py",
                    operator=record.name,
                )
            else:
                self.error(
                    "assets.outside_directory",
                    child,
                    "put every oracle dependency under the operator's assets/ directory",
                    operator=record.name,
                )
        assets = record.root / "assets"
        if assets.exists() and not assets.is_dir():
            self.error(
                "assets.not_directory",
                assets,
                "assets must be a directory",
                operator=record.name,
            )
            return
        for child in record.root.rglob("*"):
            if child.is_symlink():
                self.error(
                    "assets.symlink",
                    child,
                    "operator directories and assets cannot contain symlinks",
                    operator=record.name,
                )
        if assets.is_dir():
            files = [item for item in assets.rglob("*") if item.is_file()]
            record.asset_files = len(files)
            try:
                record.asset_bytes = sum(item.stat().st_size for item in files)
            except OSError as exc:
                self.error(
                    "assets.unreadable",
                    assets,
                    f"cannot stat an asset: {exc}",
                    operator=record.name,
                )
            if not files:
                self.warning(
                    "assets.empty",
                    assets,
                    "empty assets directory is unnecessary",
                    operator=record.name,
                )

    def _validate_definition(self, record: OperatorRecord, path: Path) -> None:
        raw = record.definition
        extra = set(raw).difference(DEFINITION_FIELDS)
        if extra:
            self.error(
                "definition.extra_fields",
                path,
                f"definition contains non-ABI fields: {sorted(extra)}",
                operator=record.name,
            )
        if raw.get("api_version") != "v6.2":
            severity = self.error if self.mode == "standard" else self.warning
            severity(
                "definition.api_version",
                path,
                "checked-in v6.2 definitions must explicitly set api_version='v6.2'",
                operator=record.name,
            )
        name = raw.get("name")
        if not isinstance(name, str) or IDENTIFIER.fullmatch(name) is None:
            self.error(
                "definition.name",
                path,
                f"invalid Python Definition.name: {name!r}",
                operator=record.name,
            )
        description = raw.get("description", "")
        if not isinstance(description, str):
            self.error(
                "definition.description",
                path,
                "description must be a string",
                operator=record.name,
            )
        parameters = raw.get("parameters")
        normalized: list[dict[str, Any]] = []
        if not isinstance(parameters, list):
            self.error(
                "definition.parameters",
                path,
                "parameters must be a list",
                operator=record.name,
            )
            parameters = []
        seen: set[str] = set()
        for index, parameter in enumerate(parameters):
            where = f"parameter[{index}]"
            if not isinstance(parameter, dict):
                self.error(
                    "parameter.type",
                    path,
                    f"{where} must be an object",
                    operator=record.name,
                )
                continue
            extra_parameter = set(parameter).difference(PARAMETER_FIELDS)
            if extra_parameter:
                self.error(
                    "parameter.extra_fields",
                    path,
                    f"{where} has unknown fields: {sorted(extra_parameter)}",
                    operator=record.name,
                )
            parameter_name = parameter.get("name")
            if (
                not isinstance(parameter_name, str)
                or IDENTIFIER.fullmatch(parameter_name) is None
            ):
                self.error(
                    "parameter.name",
                    path,
                    f"{where} has invalid Python name: {parameter_name!r}",
                    operator=record.name,
                )
                continue
            if parameter_name in seen:
                self.error(
                    "parameter.duplicate",
                    path,
                    f"duplicate parameter name: {parameter_name}",
                    operator=record.name,
                )
            seen.add(parameter_name)
            kind = parameter.get("kind", "positional_or_keyword")
            if kind not in PARAMETER_KINDS:
                self.error(
                    "parameter.kind",
                    path,
                    f"{parameter_name} has unsupported kind: {kind!r}",
                    operator=record.name,
                )
                continue
            if self.mode == "standard" and "kind" not in parameter:
                self.error(
                    "parameter.kind_missing",
                    path,
                    f"{parameter_name} must explicitly declare kind",
                    operator=record.name,
                )
            required = parameter.get("required")
            if type(required) is not bool:
                self.error(
                    "parameter.required",
                    path,
                    f"{parameter_name}.required must be a boolean",
                    operator=record.name,
                )
                continue
            has_default = "default" in parameter
            if kind == "var_positional" and (not required or has_default):
                self.error(
                    "parameter.var_positional",
                    path,
                    f"{parameter_name}: var_positional requires required=true and no default",
                    operator=record.name,
                )
            elif required and has_default:
                self.error(
                    "parameter.required_default",
                    path,
                    f"{parameter_name}: required parameters cannot declare a default",
                    operator=record.name,
                )
            elif not required and not has_default:
                self.error(
                    "parameter.optional_default",
                    path,
                    f"{parameter_name}: optional parameters require a default",
                    operator=record.name,
                )
            type_hint = parameter.get("type_hint")
            if type_hint is not None and not isinstance(type_hint, str):
                self.error(
                    "parameter.type_hint",
                    path,
                    f"{parameter_name}.type_hint must be a string or null",
                    operator=record.name,
                )
            normalized.append(
                {
                    "name": parameter_name,
                    "kind": kind,
                    "required": required,
                    **({"default": parameter.get("default")} if has_default else {}),
                }
            )
        try:
            _signature_from_parameters(normalized)
        except (TypeError, ValueError) as exc:
            self.error(
                "definition.signature",
                path,
                f"parameters do not form a valid Python signature: {exc}",
                operator=record.name,
            )

        outputs = raw.get("outputs")
        if not isinstance(outputs, list) or any(
            not isinstance(item, str) or IDENTIFIER.fullmatch(item) is None
            for item in outputs or []
        ):
            self.error(
                "definition.outputs",
                path,
                "outputs must be a list of Python identifiers",
                operator=record.name,
            )
            outputs = []
        elif len(outputs) != len(set(outputs)):
            self.error(
                "definition.outputs_duplicate",
                path,
                "Definition outputs must be unique",
                operator=record.name,
            )
        effects = raw.get("effects", {})
        if self.mode == "standard" and "effects" not in raw:
            self.error(
                "definition.effects_missing",
                path,
                "standard delivery must explicitly declare effects",
                operator=record.name,
            )
        if not isinstance(effects, dict):
            self.error(
                "definition.effects",
                path,
                "effects must be an object",
                operator=record.name,
            )
            return
        extra_effects = set(effects).difference(EFFECT_FIELDS)
        if extra_effects:
            self.error(
                "effects.extra_fields",
                path,
                f"effects has unknown fields: {sorted(extra_effects)}",
                operator=record.name,
            )
        mutates = effects.get("mutates", [])
        aliases = effects.get("returns_alias_of", {})
        parameter_names = {item["name"] for item in normalized}
        output_names = set(outputs or [])
        if not isinstance(mutates, list) or any(
            not isinstance(item, str) for item in mutates or []
        ):
            self.error(
                "effects.mutates",
                path,
                "effects.mutates must be a list of parameter names",
                operator=record.name,
            )
        else:
            unknown = set(mutates).difference(parameter_names)
            if len(mutates) != len(set(mutates)) or unknown:
                self.error(
                    "effects.mutates_contract",
                    path,
                    f"invalid effects.mutates; unknown={sorted(unknown)}",
                    operator=record.name,
                )
        if not isinstance(aliases, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in (aliases.items() if isinstance(aliases, dict) else [])
        ):
            self.error(
                "effects.aliases",
                path,
                "effects.returns_alias_of must map output names to parameter names",
                operator=record.name,
            )
        else:
            unknown_outputs = set(aliases).difference(output_names)
            unknown_targets = set(aliases.values()).difference(parameter_names)
            if unknown_outputs or unknown_targets:
                self.error(
                    "effects.alias_contract",
                    path,
                    f"aliases reference unknown outputs={sorted(unknown_outputs)} "
                    f"or parameters={sorted(unknown_targets)}",
                    operator=record.name,
                )

    def _validate_oracle(self, record: OperatorRecord) -> None:
        path = record.root / "oracle.py"
        try:
            source = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.error(
                "oracle.missing", path, "missing oracle.py", operator=record.name
            )
            return
        except UnicodeDecodeError as exc:
            self.error(
                "oracle.encoding",
                path,
                f"oracle.py is not UTF-8: {exc}",
                operator=record.name,
            )
            return
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            self.error(
                "oracle.syntax",
                path,
                f"invalid Python syntax: {exc.msg}",
                operator=record.name,
                line=exc.lineno,
            )
            return

        assignments: dict[str, list[tuple[ast.expr | None, int]]] = {}
        functions: dict[str, FunctionSignature] = {}
        aliases: dict[str, tuple[str, int]] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[node.name] = _ast_function_signature(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    assignments.setdefault(target.id, []).append(
                        (node.value, node.lineno)
                    )
                    if target.id in HOOKS and isinstance(node.value, ast.Name):
                        aliases[target.id] = (node.value.id, node.lineno)

        device_values = assignments.get("REFERENCE_DEVICE", [])
        if len(device_values) != 1:
            self.error(
                "oracle.reference_device_count",
                path,
                "oracle.py must declare REFERENCE_DEVICE exactly once",
                operator=record.name,
            )
        else:
            try:
                device = ast.literal_eval(device_values[0][0])
            except (TypeError, ValueError):
                device = None
            if device not in {"target", "cpu"}:
                self.error(
                    "oracle.reference_device",
                    path,
                    "REFERENCE_DEVICE must be the literal 'target' or 'cpu'",
                    operator=record.name,
                    line=device_values[0][1],
                )
            else:
                record.reference_device = device

        resolved: dict[str, FunctionSignature] = {}
        for hook in HOOKS:
            if hook in functions:
                resolved[hook] = functions[hook]
                continue
            target = aliases.get(hook)
            visited = {hook}
            while target is not None and target[0] not in visited:
                target_name, alias_line = target
                visited.add(target_name)
                if target_name in functions:
                    base = functions[target_name]
                    resolved[hook] = FunctionSignature(
                        hook,
                        base.parameters,
                        alias_line,
                        base.decorated,
                        base.asynchronous,
                    )
                    break
                target = aliases.get(target_name)
            if hook in assignments and hook not in resolved:
                self.error(
                    "oracle.hook_not_static_callable",
                    path,
                    f"cannot statically resolve {hook} to a top-level function",
                    operator=record.name,
                    line=assignments[hook][-1][1],
                )
        record.hooks = set(resolved)

        expected = record.definition.get("parameters")
        if not isinstance(expected, list):
            expected = []
        for hook, signature in resolved.items():
            if signature.asynchronous:
                self.error(
                    "oracle.async_hook",
                    path,
                    f"{hook} must be a synchronous function",
                    operator=record.name,
                    line=signature.line,
                )
            if signature.decorated:
                self.warning(
                    "oracle.decorated_hook",
                    path,
                    f"{hook} is decorated; use --runtime to verify its effective ABI",
                    operator=record.name,
                    line=signature.line,
                )
            if hook in PHASE_HOOKS:
                mismatch = _compare_parameter_contracts(expected, signature.parameters)
                if mismatch is not None:
                    self.error(
                        "oracle.phase_abi",
                        path,
                        f"{hook} ABI differs from Definition: {mismatch}",
                        operator=record.name,
                        line=signature.line,
                    )
            elif hook in FIXED_HOOK_SIGNATURES:
                wanted = FIXED_HOOK_SIGNATURES[hook]
                if not _is_exact_fixed_signature(signature.parameters, wanted):
                    self.error(
                        "oracle.hook_abi",
                        path,
                        f"{hook} must have exact signature ({', '.join(wanted)})",
                        operator=record.name,
                        line=signature.line,
                    )

        for symbol, validator in CONTRACT_FLAGS.items():
            values = assignments.get(symbol, [])
            if len(values) > 1:
                self.error(
                    "oracle.contract_flag_count",
                    path,
                    f"{symbol} must be assigned at most once",
                    operator=record.name,
                )
            if not values:
                continue
            try:
                value = ast.literal_eval(values[-1][0])
            except (TypeError, ValueError):
                value = None
            if type(value) is not bool:
                self.error(
                    "oracle.contract_flag_type",
                    path,
                    f"{symbol} must be a literal boolean",
                    operator=record.name,
                    line=values[-1][1],
                )
            elif value and validator not in resolved:
                self.error(
                    "oracle.contract_flag_hook",
                    path,
                    f"{symbol}=True requires {validator}",
                    operator=record.name,
                    line=values[-1][1],
                )
        if "torch_run" not in resolved and any(
            hook in resolved for hook in {"torch_gen_inputs", "torch_valid"}
        ):
            self.warning(
                "oracle.dead_torch_hook",
                path,
                "torch-specific hooks are present without torch_run",
                operator=record.name,
            )

    def _validate_workloads(
        self, record: OperatorRecord, phase: str, *, full: bool = False
    ) -> list[dict[str, Any]]:
        suffix = "_full" if full else ""
        path = record.root / f"{phase}{suffix}.jsonl"
        if not path.exists():
            return []
        workloads: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            self.error(
                "workload.encoding",
                path,
                f"workload file is not UTF-8: {exc}",
                operator=record.name,
            )
            return workloads
        local_names: dict[str, int] = {}
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                self.error(
                    "workload.json",
                    path,
                    f"invalid JSONL row: {exc.msg}",
                    operator=record.name,
                    line=line_number,
                )
                continue
            bad = _find_nonfinite(raw)
            if bad is not None:
                self.error(
                    "workload.nonfinite",
                    path,
                    f"non-finite JSON number at {bad}",
                    operator=record.name,
                    line=line_number,
                )
            if not isinstance(raw, dict):
                self.error(
                    "workload.type",
                    path,
                    "each JSONL row must be an object",
                    operator=record.name,
                    line=line_number,
                )
                continue
            extra = set(raw).difference(WORKLOAD_FIELDS)
            if extra:
                self.error(
                    "workload.extra_fields",
                    path,
                    f"unknown workload fields: {sorted(extra)}",
                    operator=record.name,
                    line=line_number,
                )
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                self.error(
                    "workload.name",
                    path,
                    "workload name must be a non-empty string",
                    operator=record.name,
                    line=line_number,
                )
            else:
                previous = (
                    None
                    if full
                    else self._workload_names.get(name)
                )
                if previous is not None:
                    severity = self.error if self.mode == "standard" else self.warning
                    severity(
                        "workload.duplicate_name",
                        path,
                        f"duplicate workload name; first used by "
                        f"{previous[0]}/{previous[1]}:{previous[2]}",
                        operator=record.name,
                        line=line_number,
                    )
                elif not full:
                    self._workload_names[name] = (record.name, phase, line_number)
                elif name in local_names:
                    self.error(
                        "workload.archive_duplicate_name",
                        path,
                        f"duplicate archived workload name; first used at "
                        f"line {local_names[name]}",
                        operator=record.name,
                        line=line_number,
                    )
                else:
                    local_names[name] = line_number
            inputs = raw.get("inputs")
            if not isinstance(inputs, dict):
                self.error(
                    "workload.inputs",
                    path,
                    "workload inputs must be an object",
                    operator=record.name,
                    line=line_number,
                )
                inputs = {}
            for input_name, value in inputs.items():
                if not isinstance(input_name, str) or IDENTIFIER.fullmatch(input_name) is None:
                    self.error(
                        "workload.input_name",
                        path,
                        f"invalid input/context name: {input_name!r}",
                        operator=record.name,
                        line=line_number,
                    )
                self._validate_recipe(record, path, line_number, input_name, value)
            for field_name in ("input_path", "output_path"):
                tensor_path = raw.get(field_name)
                if tensor_path is None:
                    continue
                if not isinstance(tensor_path, str) or not tensor_path:
                    self.error(
                        f"workload.{field_name}",
                        path,
                        f"{field_name} must be a non-empty absolute path",
                        operator=record.name,
                        line=line_number,
                    )
                elif not Path(tensor_path).is_absolute():
                    self.error(
                        f"workload.{field_name}_absolute",
                        path,
                        f"{field_name} must be absolute",
                        operator=record.name,
                        line=line_number,
                    )
            safetensor_inputs = [
                input_name
                for input_name, value in inputs.items()
                if isinstance(value, dict) and value.get("type") == "safetensor"
            ]
            if safetensor_inputs and raw.get("input_path") is None:
                self.error(
                    "workload.input_path_required",
                    path,
                    "type='safetensor' inputs require input_path",
                    operator=record.name,
                    line=line_number,
                )
            if phase == "timing" and raw.get("output_path") is not None:
                self.error(
                    "workload.timing_output_path",
                    path,
                    "timing workloads cannot use output_path; KGS must time the reference",
                    operator=record.name,
                    line=line_number,
                )
            seed = raw.get("seed", 0)
            if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
                self.error(
                    "workload.seed",
                    path,
                    "seed must be an integer in 0..2^63-1",
                    operator=record.name,
                    line=line_number,
                )
            self._validate_tolerance(record, path, line_number, raw.get("tolerance"))
            raw["_source_line"] = line_number
            workloads.append(raw)
        return workloads

    def _validate_recipe(
        self,
        record: OperatorRecord,
        path: Path,
        line: int,
        name: str,
        value: Any,
    ) -> None:
        if not isinstance(value, dict):
            return
        kind = value.get("type")
        if kind not in {"random", "custom", "scalar", "literal", "safetensor"}:
            return
        if kind == "random":
            shape = value.get("shape")
            dtype = value.get("dtype")
            if not isinstance(shape, list) or any(
                type(dimension) is not int or dimension < 0 for dimension in shape or []
            ):
                self.error(
                    "recipe.random_shape",
                    path,
                    f"{name}: random recipe requires a non-negative integer shape list",
                    operator=record.name,
                    line=line,
                )
            if not isinstance(dtype, str) or not dtype:
                self.error(
                    "recipe.random_dtype",
                    path,
                    f"{name}: random recipe requires a non-empty dtype string",
                    operator=record.name,
                    line=line,
                )
            if value.get("device", "cpu") not in {"cpu", "target"}:
                self.error(
                    "recipe.random_device",
                    path,
                    f"{name}: random device must be 'cpu' or 'target'",
                    operator=record.name,
                    line=line,
                )
        elif kind in {"scalar", "literal"} and "value" not in value:
            self.error(
                "recipe.value",
                path,
                f"{name}: {kind} recipe requires value",
                operator=record.name,
                line=line,
            )

    def _validate_tolerance(
        self,
        record: OperatorRecord,
        path: Path,
        line: int,
        tolerance: Any,
    ) -> None:
        if tolerance is None:
            return
        if not isinstance(tolerance, dict):
            self.error(
                "tolerance.type",
                path,
                "tolerance must be an object or null",
                operator=record.name,
                line=line,
            )
            return
        extra = set(tolerance).difference(TOLERANCE_FIELDS)
        if extra:
            self.error(
                "tolerance.extra_fields",
                path,
                f"unknown tolerance fields: {sorted(extra)}",
                operator=record.name,
                line=line,
            )
        for key in {"rtol", "atol"}:
            value = tolerance.get(key)
            if value is not None and (not _finite_number(value) or value < 0):
                self.error(
                    f"tolerance.{key}",
                    path,
                    f"{key} must be null or a finite non-negative number",
                    operator=record.name,
                    line=line,
                )
        scale = tolerance.get("atol_scale", 1.0)
        ratio = tolerance.get("required_matched_ratio", 1.0)
        if not _finite_number(scale) or scale <= 0:
            self.error(
                "tolerance.atol_scale",
                path,
                "atol_scale must be a finite number greater than zero",
                operator=record.name,
                line=line,
            )
        if not _finite_number(ratio) or not 0 < ratio <= 1:
            self.error(
                "tolerance.matched_ratio",
                path,
                "required_matched_ratio must be in (0, 1]",
                operator=record.name,
                line=line,
            )
        if type(tolerance.get("equal_nan", False)) is not bool:
            self.error(
                "tolerance.equal_nan",
                path,
                "equal_nan must be a boolean",
                operator=record.name,
                line=line,
            )

    def _validate_phase_contract(self, record: OperatorRecord) -> None:
        path = record.root / "oracle.py"
        if record.correctness and not ({"run", "correctness_run"} & record.hooks):
            self.error(
                "oracle.correctness_entrypoint",
                path,
                "correctness workloads require correctness_run or run",
                operator=record.name,
            )
        if record.timing and not ({"run", "timing_run"} & record.hooks):
            self.error(
                "oracle.timing_entrypoint",
                path,
                "timing workloads require timing_run or run",
                operator=record.name,
            )
        if record.timing and record.reference_device == "cpu":
            self.error(
                "oracle.cpu_timing",
                path,
                "CPU references cannot own timing workloads",
                operator=record.name,
            )

    def _validate_materialization_contract(self, record: OperatorRecord) -> None:
        parameters = record.definition.get("parameters")
        if not isinstance(parameters, list):
            return
        required = {
            item.get("name"): item
            for item in parameters
            if isinstance(item, dict) and item.get("required") is True
        }
        for phase, workloads in (
            ("correctness", record.correctness),
            ("timing", record.timing),
        ):
            for workload in workloads:
                inputs = workload.get("inputs")
                if not isinstance(inputs, dict):
                    continue
                line = workload.get("_source_line")
                if phase == "correctness" and workload.get("output_path") is not None:
                    outputs = record.definition.get("outputs")
                    if not isinstance(outputs, list) or not outputs:
                        self.error(
                            "workload.output_path_void",
                            record.root / f"{phase}.jsonl",
                            "output_path cannot represent a void operator",
                            operator=record.name,
                            line=line,
                        )
                    effects = record.definition.get("effects", {})
                    mutates = (
                        set(effects.get("mutates", []))
                        if isinstance(effects, dict)
                        else set()
                    )
                    aliases = (
                        effects.get("returns_alias_of", {})
                        if isinstance(effects, dict)
                        else {}
                    )
                    represented = set(aliases.values()) if isinstance(aliases, dict) else set()
                    missing_mutations = mutates - represented
                    if missing_mutations:
                        self.error(
                            "workload.output_path_mutation",
                            record.root / f"{phase}.jsonl",
                            "output_path cannot represent mutated parameters that are "
                            f"not returned by alias: {sorted(missing_mutations)}",
                            operator=record.name,
                            line=line,
                        )
                for name, parameter in required.items():
                    if not isinstance(name, str):
                        continue
                    value = inputs.get(name, MISSING)
                    custom = isinstance(value, dict) and value.get("type") == "custom"
                    if (value is MISSING or custom) and "gen_inputs" not in record.hooks:
                        self.error(
                            "workload.required_not_materialized",
                            record.root / f"{phase}.jsonl",
                            f"required parameter {name!r} needs a direct recipe or gen_inputs",
                            operator=record.name,
                            line=line,
                        )
                    if parameter.get("kind", "positional_or_keyword") == "var_positional":
                        if value is not MISSING and not custom and not isinstance(value, list):
                            self.error(
                                "workload.var_positional",
                                record.root / f"{phase}.jsonl",
                                f"var_positional parameter {name!r} must materialize a list/tuple",
                                operator=record.name,
                                line=line,
                            )
                if "torch_run" in record.hooks and "torch_gen_inputs" not in record.hooks:
                    needs_torch_hook = any(
                        (name not in inputs)
                        or (
                            isinstance(inputs.get(name), dict)
                            and inputs[name].get("type") == "custom"
                        )
                        for name in required
                    )
                    if needs_torch_hook:
                        self.error(
                            "workload.torch_materialization",
                            record.root / f"{phase}.jsonl",
                            "torch fallback cannot materialize this workload without "
                            "torch_gen_inputs",
                            operator=record.name,
                            line=line,
                        )

    def report(self) -> dict[str, Any]:
        errors = sum(issue.severity == "error" for issue in self.issues)
        warnings = sum(issue.severity == "warning" for issue in self.issues)
        operators = list(self.operators.values())
        return {
            "schema": REPORT_SCHEMA,
            "catalog": str(self.root),
            "mode": self.mode,
            "passed": errors == 0,
            "summary": {
                "operators": len(operators),
                "correctness_workloads": sum(len(item.correctness) for item in operators),
                "timing_workloads": sum(len(item.timing) for item in operators),
                "asset_files": sum(item.asset_files for item in operators),
                "asset_bytes": sum(item.asset_bytes for item in operators),
                "errors": errors,
                "warnings": warnings,
            },
            "issues": [asdict(issue) for issue in self.issues],
        }


def _find_nonfinite(value: Any, path: str = "$") -> str | None:
    if isinstance(value, float) and not math.isfinite(value):
        return path
    if isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_nonfinite(item, f"{path}[{index}]")
            if found is not None:
                return found
    if isinstance(value, dict):
        for key, item in value.items():
            found = _find_nonfinite(item, f"{path}.{key}")
            if found is not None:
                return found
    return None


def _finite_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _signature_from_parameters(parameters: Sequence[dict[str, Any]]) -> inspect.Signature:
    result = []
    for parameter in parameters:
        default = inspect.Parameter.empty
        if parameter.get("required") is False:
            default = parameter.get("default")
        result.append(
            inspect.Parameter(
                parameter["name"],
                PARAMETER_KINDS[parameter.get("kind", "positional_or_keyword")],
                default=default,
            )
        )
    return inspect.Signature(result)


def _normalize_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_default(item) for item in value]
    module = type(value).__module__
    if module == "torch" or module.startswith("torch."):
        rendered = str(value)
        return rendered.removeprefix("torch.")
    return value


def _ast_default(node: ast.expr) -> Any:
    try:
        return _normalize_default(ast.literal_eval(node))
    except (TypeError, ValueError):
        pass
    if isinstance(node, ast.Attribute):
        return node.attr
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "device"
        and len(node.args) == 1
        and not node.keywords
    ):
        try:
            value = ast.literal_eval(node.args[0])
        except (TypeError, ValueError):
            return MISSING
        return value if isinstance(value, str) else MISSING
    return MISSING


def _ast_function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> FunctionSignature:
    parameters: list[dict[str, Any]] = []
    positional = [*node.args.posonlyargs, *node.args.args]
    default_start = len(positional) - len(node.args.defaults)
    for index, argument in enumerate(positional):
        default_node = (
            node.args.defaults[index - default_start] if index >= default_start else None
        )
        kind = (
            "positional_only"
            if index < len(node.args.posonlyargs)
            else "positional_or_keyword"
        )
        parameters.append(
            {
                "name": argument.arg,
                "kind": kind,
                "required": default_node is None,
                **({"default": _ast_default(default_node)} if default_node else {}),
            }
        )
    if node.args.vararg is not None:
        parameters.append(
            {
                "name": node.args.vararg.arg,
                "kind": "var_positional",
                "required": True,
            }
        )
    for argument, default_node in zip(
        node.args.kwonlyargs, node.args.kw_defaults, strict=True
    ):
        parameters.append(
            {
                "name": argument.arg,
                "kind": "keyword_only",
                "required": default_node is None,
                **({"default": _ast_default(default_node)} if default_node else {}),
            }
        )
    if node.args.kwarg is not None:
        parameters.append(
            {
                "name": node.args.kwarg.arg,
                "kind": "var_keyword",
                "required": True,
            }
        )
    return FunctionSignature(
        node.name,
        parameters,
        node.lineno,
        bool(node.decorator_list),
        isinstance(node, ast.AsyncFunctionDef),
    )


def _compare_parameter_contracts(
    expected: Sequence[Any], actual: Sequence[dict[str, Any]]
) -> str | None:
    expected_items = [item for item in expected if isinstance(item, dict)]
    if len(expected_items) != len(actual):
        return f"expected {len(expected_items)} parameters, got {len(actual)}"
    for index, (wanted, got) in enumerate(zip(expected_items, actual, strict=True)):
        wanted_kind = wanted.get("kind", "positional_or_keyword")
        wanted_required = wanted.get("required")
        if wanted.get("name") != got.get("name") or wanted_kind != got.get("kind"):
            return (
                f"parameter {index}: expected {wanted_kind} {wanted.get('name')!r}, "
                f"got {got.get('kind')} {got.get('name')!r}"
            )
        if wanted_required != got.get("required"):
            return f"{wanted.get('name')!r} required/default status differs"
        if wanted_required is False:
            actual_default = got.get("default", MISSING)
            expected_default = wanted.get("default", MISSING)
            if actual_default is MISSING:
                return f"{wanted.get('name')!r} default cannot be statically resolved"
            if (
                type(actual_default) is not type(expected_default)
                or actual_default != expected_default
            ):
                return (
                    f"{wanted.get('name')!r} default differs: expected "
                    f"{expected_default!r}, got {actual_default!r}"
                )
    return None


def _is_exact_fixed_signature(
    parameters: Sequence[dict[str, Any]], names: Sequence[str]
) -> bool:
    return len(parameters) == len(names) and all(
        parameter.get("name") == name
        and parameter.get("kind") == "positional_or_keyword"
        and parameter.get("required") is True
        for parameter, name in zip(parameters, names, strict=True)
    )


def _runtime_signature(function: Any) -> list[dict[str, Any]]:
    parameters = []
    for parameter in inspect.signature(function).parameters.values():
        kinds = {
            inspect.Parameter.POSITIONAL_ONLY: "positional_only",
            inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional_or_keyword",
            inspect.Parameter.VAR_POSITIONAL: "var_positional",
            inspect.Parameter.KEYWORD_ONLY: "keyword_only",
            inspect.Parameter.VAR_KEYWORD: "var_keyword",
        }
        required = parameter.default is inspect.Parameter.empty
        parameters.append(
            {
                "name": parameter.name,
                "kind": kinds[parameter.kind],
                "required": required,
                **(
                    {"default": _normalize_default(parameter.default)}
                    if not required
                    else {}
                ),
            }
        )
    return parameters


def _module_path(module: ModuleType) -> Path | None:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str):
        return None
    try:
        return Path(value).resolve()
    except OSError:
        return None


def _module_belongs_to(module: ModuleType, root: Path) -> bool:
    path = _module_path(module)
    if path is not None and path.is_relative_to(root):
        return True
    search_paths = getattr(module, "__path__", ())
    try:
        iterator = iter(search_paths)
    except TypeError:
        return False
    for value in iterator:
        try:
            if Path(value).resolve().is_relative_to(root):
                return True
        except (OSError, TypeError):
            continue
    return False


def _load_runtime_oracle(
    operator_root: Path,
    framework_root: Path | None = None,
) -> ModuleType:
    operator_root = operator_root.resolve()
    oracle_path = operator_root / "oracle.py"
    assets_root = operator_root / "assets"
    for name, module in list(sys.modules.items()):
        if isinstance(module, ModuleType) and _module_belongs_to(module, operator_root):
            sys.modules.pop(name, None)
    module_name = f"_kgs_catalog_validator_oracle_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, oracle_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {oracle_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    roots = [operator_root]
    if assets_root.is_dir():
        roots.append(assets_root)
    if framework_root is not None:
        roots.append(framework_root / "src")
    inserted: list[str] = []
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        for root in reversed(roots):
            value = str(root)
            sys.path.insert(0, value)
            inserted.append(value)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
        for value in inserted:
            try:
                sys.path.remove(value)
            except ValueError:
                pass
        for name, imported in list(sys.modules.items()):
            if name == module_name:
                continue
            if isinstance(imported, ModuleType) and _module_belongs_to(
                imported, operator_root
            ):
                sys.modules.pop(name, None)
    return module


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "runtime validation needs Torch for this workload/hook, but torch is unavailable"
        ) from exc
    return torch


def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    value = getattr(torch, dtype_name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported random dtype: {dtype_name}")
    return value


def _random_tensor(
    torch: Any,
    spec: Mapping[str, Any],
    generator: Any,
    *,
    device: str = "cpu",
) -> Any:
    dtype_name = spec["dtype"]
    dtype = _torch_dtype(torch, dtype_name)
    source_dtype = _torch_dtype(
        torch, str(spec.get("source_dtype", dtype_name))
    )
    shape = tuple(spec["shape"])
    transforms = spec.get("transforms", [])
    if not isinstance(transforms, list):
        raise TypeError("random input transforms must be a list")

    prepared_random_values: dict[int, Any] = {}
    for index, transform in enumerate(transforms):
        if not isinstance(transform, Mapping) or not isinstance(
            transform.get("op"), str
        ):
            raise TypeError("random input transform must contain a string op")
        if transform["op"] == "multiply_random" and transform.get("before", False):
            nested = transform.get("input")
            if not isinstance(nested, Mapping):
                raise TypeError("multiply_random transform requires an input recipe")
            prepared_random_values[index] = _random_tensor(
                torch, nested, generator, device=device
            )

    distribution = spec.get("distribution", "normal")
    if distribution == "normal":
        if source_dtype.is_floating_point or source_dtype.is_complex:
            value = torch.randn(
                shape,
                dtype=source_dtype,
                device=device,
                generator=generator,
            )
        elif source_dtype is torch.bool:
            value = torch.randint(
                0,
                2,
                shape,
                dtype=torch.bool,
                device=device,
                generator=generator,
            )
        else:
            ranges = {
                torch.int8: (-128, 128),
                torch.int16: (-1024, 1024),
                torch.int32: (-1024, 1024),
                torch.int64: (-1024, 1024),
                torch.uint8: (0, 256),
            }
            if source_dtype not in ranges:
                raise ValueError(f"unsupported random dtype: {dtype_name}")
            low, high = ranges[source_dtype]
            value = torch.randint(
                low,
                high,
                shape,
                dtype=source_dtype,
                device=device,
                generator=generator,
            )
    elif distribution == "uniform":
        if not source_dtype.is_floating_point:
            raise ValueError("uniform distribution requires a floating source dtype")
        low = float(spec.get("low", 0.0))
        high = float(spec.get("high", 1.0))
        value = torch.rand(
            shape,
            dtype=source_dtype,
            device=device,
            generator=generator,
        )
        if low != 0.0 or high != 1.0:
            value = value * (high - low) + low
    elif distribution == "integer":
        value = torch.randint(
            int(spec["low"]),
            int(spec["high"]),
            shape,
            dtype=source_dtype,
            device=device,
            generator=generator,
        )
    else:
        raise ValueError(f"unsupported random distribution: {distribution}")

    for index, transform in enumerate(transforms):
        operation = transform["op"]
        if operation == "transpose":
            value = value.T
        elif operation == "symmetrize_sum":
            value = value + value.T
        elif operation == "triu":
            value = torch.triu(value, diagonal=int(transform.get("diagonal", 0)))
        elif operation == "tril":
            value = torch.tril(value, diagonal=int(transform.get("diagonal", 0)))
        elif operation == "softmax":
            value = value.softmax(dim=int(transform["dim"]))
        elif operation == "cast":
            value = value.to(dtype=_torch_dtype(torch, str(transform["dtype"])))
        elif operation == "multiply":
            value = value * transform["value"]
        elif operation == "divide":
            value = value / transform["value"]
        elif operation == "rdivide":
            value = transform["value"] / value
        elif operation == "add":
            value = value + transform["value"]
        elif operation == "subtract":
            value = value - transform["value"]
        elif operation == "rsubtract":
            value = transform["value"] - value
        elif operation == "multiply_random":
            operand = prepared_random_values.get(index)
            if operand is None:
                nested = transform.get("input")
                if not isinstance(nested, Mapping):
                    raise TypeError(
                        "multiply_random transform requires an input recipe"
                    )
                operand = _random_tensor(torch, nested, generator, device=device)
            value = value * operand
        else:
            raise ValueError(f"unsupported random transform: {operation}")
    return value.to(dtype=dtype)


def _recipe(value: Any) -> tuple[str, Mapping[str, Any]] | None:
    if not isinstance(value, Mapping):
        return None
    kind = value.get("type")
    if kind not in {"random", "custom", "scalar", "literal", "safetensor"}:
        return None
    return str(kind), value


def _load_safetensors(path: str) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "runtime validation needs safetensors for this file-backed workload"
        ) from exc
    values: dict[str, Any] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            values[key] = handle.get_tensor(key)
    return values


def _load_golden_output(
    definition: dict[str, Any], workload: dict[str, Any]
) -> Any:
    output_path = workload.get("output_path")
    if not isinstance(output_path, str):
        raise ValueError("workload.output_path is required")
    outputs = definition.get("outputs", [])
    if not outputs:
        raise ValueError("output_path cannot represent a void operator")
    loaded = _load_safetensors(output_path)
    missing = set(outputs) - set(loaded)
    if missing:
        raise ValueError(
            "output safetensors file is missing Definition output keys: "
            f"{sorted(missing)}"
        )
    values = [loaded[name] for name in outputs]
    return values[0] if len(values) == 1 else tuple(values)


def _seed_globals(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _move_tensors(value: Any, device: str, memo: dict[int, Any] | None = None) -> Any:
    try:
        import torch
    except ImportError:
        return value
    memo = {} if memo is None else memo
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if isinstance(value, torch.Tensor):
        result = value.to(device=device)
        memo[value_id] = result
        return result
    if isinstance(value, list):
        result: list[Any] = []
        memo[value_id] = result
        result.extend(_move_tensors(item, device, memo) for item in value)
        return result
    if isinstance(value, tuple):
        result = tuple(_move_tensors(item, device, memo) for item in value)
        memo[value_id] = result
        return result
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        memo[value_id] = result
        result.update((key, _move_tensors(item, device, memo)) for key, item in value.items())
        return result
    return value


def _to_cpu(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


def _materialize_call(
    definition: dict[str, Any],
    module: ModuleType,
    workload: dict[str, Any],
    device: str,
    *,
    fallback: bool,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    inputs = workload["inputs"]
    seed = workload.get("seed", 0)
    parameter_names = {
        parameter["name"] for parameter in definition.get("parameters", [])
    }
    values: dict[str, Any] = {}
    torch = None
    cpu_generator = None
    for name, raw in inputs.items():
        recipe = _recipe(raw)
        if recipe is None:
            values[name] = copy.deepcopy(raw)
            continue
        kind, spec = recipe
        if kind == "random":
            if spec.get("device", "cpu") == "target":
                continue
            torch = torch or _import_torch()
            cpu_generator = cpu_generator or torch.Generator(device="cpu").manual_seed(seed)
            values[name] = _random_tensor(
                torch, spec, cpu_generator
            )
        elif kind in {"scalar", "literal"}:
            values[name] = copy.deepcopy(spec["value"])
    file_inputs = [
        name
        for name, raw in inputs.items()
        if (recipe := _recipe(raw)) is not None and recipe[0] == "safetensor"
    ]
    input_path = workload.get("input_path")
    if file_inputs and not isinstance(input_path, str):
        raise ValueError("safetensor inputs require workload.input_path")
    if isinstance(input_path, str):
        loaded = _load_safetensors(input_path)
        missing = set(file_inputs) - set(loaded)
        if missing:
            raise ValueError(
                "input safetensors file is missing parameter keys: "
                f"{sorted(missing)}"
            )
        values.update(loaded)
    values = {name: value for name, value in values.items() if name in parameter_names}
    _seed_globals(seed)
    for name, raw in inputs.items():
        recipe = _recipe(raw)
        if name not in parameter_names or recipe is None:
            continue
        kind, spec = recipe
        if kind == "random" and spec.get("device", "cpu") == "target":
            torch = torch or _import_torch()
            values[name] = _random_tensor(
                torch, spec, None, device=device
            )
    hook_name = "torch_gen_inputs" if fallback else "gen_inputs"
    hook = getattr(module, hook_name, None)
    if hook is not None:
        torch = torch or _import_torch()
        generated = hook(
            {
                "name": workload["name"],
                "inputs": copy.deepcopy(inputs),
                "seed": seed,
                **(
                    {"input_path": workload["input_path"]}
                    if workload.get("input_path") is not None
                    else {}
                ),
                **(
                    {"output_path": workload["output_path"]}
                    if workload.get("output_path") is not None
                    else {}
                ),
            },
            torch.device(device),
        )
        if generated is not None and not isinstance(generated, Mapping):
            raise TypeError(f"{hook_name} must return a mapping or None")
        if generated is not None:
            unknown = set(generated).difference(parameter_names)
            if unknown:
                raise ValueError(f"{hook_name} returned unknown parameters: {sorted(unknown)}")
            values.update(generated)
    missing = {
        parameter["name"]
        for parameter in definition.get("parameters", [])
        if parameter.get("required") is True and parameter["name"] not in values
    }
    if missing:
        raise ValueError(f"inputs did not materialize required parameters: {sorted(missing)}")
    values = _move_tensors(values, device)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for parameter in definition.get("parameters", []):
        name = parameter["name"]
        if name not in values:
            continue
        kind = parameter.get("kind", "positional_or_keyword")
        if kind == "var_positional":
            items = values[name]
            if not isinstance(items, (list, tuple)):
                raise TypeError(f"var_positional input {name!r} must be a list or tuple")
            args.extend(items)
        elif kind == "positional_only":
            args.append(values[name])
        else:
            kwargs[name] = values[name]
    return tuple(args), kwargs


def _synchronize(module: ModuleType, device: str) -> None:
    if device == "cpu":
        return
    custom = getattr(module, "synchronize", None)
    if custom is not None:
        if not callable(custom):
            raise TypeError("oracle synchronize must be callable")
        custom(device)
        return
    torch = _import_torch()
    device_object = torch.device(device)
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and callable(getattr(accelerator, "synchronize", None)):
        accelerator.synchronize(device_object)
        return
    backend = getattr(torch, device_object.type, None)
    if backend is not None and callable(getattr(backend, "synchronize", None)):
        backend.synchronize(device_object)
        return
    raise RuntimeError(
        f"cannot find a synchronization API for {device!r}; expose it through "
        "Torch or define oracle synchronize(device)"
    )


def _release_runtime_resources(device: str | None) -> None:
    """Best-effort cleanup between isolated oracle smoke workloads."""

    gc.collect()
    if not device or device == "cpu":
        return
    try:
        torch = _import_torch()
        device_object = torch.device(device)
        backend = getattr(torch, device_object.type, None)
        synchronize = getattr(backend, "synchronize", None)
        if callable(synchronize):
            synchronize(device_object)
        empty_cache = getattr(backend, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception:
        # Cleanup must not replace the workload result (notably after an
        # asynchronous device assertion).  The per-operator process still
        # exits immediately after reporting the primary failure.
        pass
    gc.collect()


def _runtime_validate_hooks(module: ModuleType, definition: dict[str, Any]) -> None:
    expected = definition.get("parameters", [])
    for name in PHASE_HOOKS:
        function = getattr(module, name, None)
        if function is None:
            continue
        if not callable(function):
            raise TypeError(f"oracle {name} is not callable")
        mismatch = _compare_parameter_contracts(expected, _runtime_signature(function))
        if mismatch is not None:
            raise TypeError(f"oracle {name} ABI differs from Definition: {mismatch}")
    for name, wanted in FIXED_HOOK_SIGNATURES.items():
        function = getattr(module, name, None)
        if function is None:
            continue
        if not callable(function) or not _is_exact_fixed_signature(
            _runtime_signature(function), wanted
        ):
            raise TypeError(f"oracle {name} must have exact signature ({', '.join(wanted)})")


def _run_reference_source(
    operator_root: Path,
    definition: dict[str, Any],
    correctness: list[dict[str, Any]],
    timing: list[dict[str, Any]],
    target_device: str | None,
    framework_root: Path | None,
    *,
    fallback: bool,
) -> dict[str, Any]:
    module = _load_runtime_oracle(operator_root, framework_root)
    _runtime_validate_hooks(module, definition)
    reference_device = getattr(module, "REFERENCE_DEVICE")
    if reference_device == "target":
        if not target_device:
            raise RuntimeError(
                "REFERENCE_DEVICE='target' requires --target-device for runtime validation"
            )
        device = target_device
    elif reference_device == "cpu":
        device = "cpu"
    else:
        raise RuntimeError("REFERENCE_DEVICE must be 'target' or 'cpu'")
    counts = {"correctness": 0, "timing": 0}
    for phase, workloads in (("correctness", correctness), ("timing", timing)):
        if fallback:
            function = getattr(module, "torch_run", None)
        else:
            function = getattr(
                module,
                f"{phase}_run",
                getattr(module, "run", None),
            )
        if workloads and not callable(function):
            raise RuntimeError(f"selected oracle has no {phase} entrypoint")
        validator_name = "torch_valid" if fallback else "valid"
        validator = getattr(module, validator_name, None)
        for workload in workloads:
            args = kwargs = bound = output = None
            output_list = cpu_outputs = cpu_inputs = verdict = None
            try:
                args, kwargs = _materialize_call(
                    definition,
                    module,
                    workload,
                    device,
                    fallback=fallback,
                )
                bound = _signature_from_parameters(
                    definition.get("parameters", [])
                ).bind(*args, **kwargs)
                if phase == "correctness" and workload.get("output_path") is not None:
                    output = _load_golden_output(definition, workload)
                else:
                    inspect.signature(function).bind(*args, **kwargs)
                    _seed_globals(workload.get("seed", 0))
                    output = function(*args, **kwargs)
                    _synchronize(module, device)
                if phase == "correctness" and validator is not None:
                    output_list = list(output) if isinstance(output, (tuple, list)) else [output]
                    cpu_outputs = _to_cpu(output_list)
                    cpu_inputs = _to_cpu(
                        dict(bound.arguments)
                    )
                    verdict = validator(
                        cpu_outputs,
                        cpu_outputs,
                        cpu_inputs,
                        {
                            "name": workload["name"],
                            "inputs": copy.deepcopy(workload["inputs"]),
                            "seed": workload.get("seed", 0),
                            **(
                                {"input_path": workload["input_path"]}
                                if workload.get("input_path") is not None
                                else {}
                            ),
                            **(
                                {"output_path": workload["output_path"]}
                                if workload.get("output_path") is not None
                                else {}
                            ),
                        },
                    )
                    if not isinstance(verdict, (bool, dict)):
                        raise TypeError(
                            f"{validator_name} must return bool or a verdict mapping"
                        )
                    if isinstance(verdict, dict) and type(verdict.get("passed")) is not bool:
                        raise TypeError(
                            f"{validator_name} verdict must contain boolean 'passed'"
                        )
                    passed = verdict if isinstance(verdict, bool) else verdict["passed"]
                    if not passed:
                        raise ValueError(
                            f"{validator_name} rejected identical reference outputs"
                        )
                counts[phase] += 1
            except Exception as exc:
                raise RuntimeError(
                    f"{phase} workload {workload.get('name')!r} failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            finally:
                del args, kwargs, bound, output
                del output_list, cpu_outputs, cpu_inputs, verdict
                _release_runtime_resources(device)
    return {"source": "torch_fallback" if fallback else "primary", **counts}


def _runtime_operator(
    operator_root: str,
    definition: dict[str, Any],
    correctness: list[dict[str, Any]],
    timing: list[dict[str, Any]],
    target_device: str | None,
    framework_root: str | None,
) -> dict[str, Any]:
    root = Path(operator_root)
    resolved_framework_root = Path(framework_root) if framework_root else None
    try:
        return {
            "passed": True,
            **_run_reference_source(
                root,
                definition,
                correctness,
                timing,
                target_device,
                resolved_framework_root,
                fallback=False,
            ),
        }
    except Exception:
        primary_error = traceback.format_exc()
    try:
        module = _load_runtime_oracle(root, resolved_framework_root)
    except Exception:
        return {
            "passed": False,
            "source": None,
            "primary_error": primary_error,
            "error": primary_error,
        }
    if not callable(getattr(module, "torch_run", None)):
        return {
            "passed": False,
            "source": None,
            "primary_error": primary_error,
            "error": primary_error,
        }
    try:
        result = _run_reference_source(
            root,
            definition,
            correctness,
            timing,
            target_device,
            resolved_framework_root,
            fallback=True,
        )
        return {
            "passed": True,
            **result,
            "primary_error": primary_error,
        }
    except Exception:
        return {
            "passed": False,
            "source": None,
            "primary_error": primary_error,
            "fallback_error": traceback.format_exc(),
        }


def _runtime_worker(queue: Any, payload: dict[str, Any]) -> None:
    try:
        result = _runtime_operator(**payload)
        _release_runtime_resources(payload.get("target_device"))
        queue.put({"ok": True, "result": result})
    except BaseException:
        try:
            queue.put({"ok": False, "error": traceback.format_exc()})
        except Exception:
            pass


def _run_runtime_validation(
    validator: Validator,
    *,
    target_device: str | None,
    framework_root: Path | None,
    timeout_seconds: float,
    progress: bool,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    results: dict[str, Any] = {}
    selected = [
        (name, record)
        for name, record in validator.operators.items()
        if not validator.selected_operators or name in validator.selected_operators
    ]
    for ordinal, (name, record) in enumerate(selected, 1):
        if progress:
            print(
                f"runtime [{ordinal}/{len(selected)}] {name}: running",
                file=sys.stderr,
                flush=True,
            )
        queue = context.Queue(maxsize=1)
        payload = {
            "operator_root": str(record.root),
            "definition": record.definition,
            "correctness": [
                {key: value for key, value in workload.items() if key != "_source_line"}
                for workload in record.correctness
            ],
            "timing": [
                {key: value for key, value in workload.items() if key != "_source_line"}
                for workload in record.timing
            ],
            "target_device": target_device,
            "framework_root": str(framework_root) if framework_root else None,
        }
        process = context.Process(
            target=_runtime_worker,
            args=(queue, payload),
            daemon=False,
        )
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(5)
            result = {
                "passed": False,
                "source": None,
                "error": f"runtime validation timed out after {timeout_seconds:g}s",
            }
        else:
            try:
                message = queue.get(timeout=1)
            except queue_module.Empty:
                result = {
                    "passed": False,
                    "source": None,
                    "error": (
                        "runtime worker exited without a result "
                        f"(exitcode={process.exitcode})"
                    ),
                }
            else:
                result = message.get("result") if message.get("ok") else {
                    "passed": False,
                    "source": None,
                    "error": message.get("error", "runtime worker failed"),
                }
        results[name] = result
        if progress:
            state = "PASS" if result.get("passed") else "FAIL"
            source = result.get("source") or "none"
            print(
                f"runtime [{ordinal}/{len(selected)}] {name}: {state} ({source})",
                file=sys.stderr,
                flush=True,
            )
        if not result.get("passed"):
            details = result.get("fallback_error") or result.get("error") or "unknown error"
            validator.error(
                "runtime.oracle_failed",
                record.root / "oracle.py",
                _last_traceback_line(details),
                operator=name,
            )
        elif result.get("source") == "torch_fallback":
            validator.warning(
                "runtime.primary_fallback",
                record.root / "oracle.py",
                "primary readiness failed; all workloads passed with torch_run",
                operator=name,
            )
        queue.close()
        queue.join_thread()
    return {
        "enabled": True,
        "target_device": target_device,
        "framework_root": str(framework_root) if framework_root else None,
        "timeout_seconds_per_operator": timeout_seconds,
        "operators": results,
        "passed": all(value.get("passed") for value in results.values()),
    }


def _last_traceback_line(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[-1] if lines else value


def _git(
    root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _validate_runtime_framework(
    validator: Validator,
    configured_root: Path | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    framework = validator.manifest.get("framework")
    if configured_root is None:
        if framework is not None:
            validator.error(
                "runtime.framework_root_required",
                validator.root / "manifest.json",
                f"framework={framework!r} requires --framework-root for runtime validation",
            )
        return None, None

    root = configured_root.resolve()
    if not root.is_dir() or not (root / "src").is_dir():
        validator.error(
            "runtime.framework_root",
            root,
            "framework root must be a checkout containing src/",
        )
        return None, None
    information: dict[str, Any] = {"kind": framework, "root": str(root)}
    if framework != "flaggems":
        return root, information

    required = validator.manifest.get("framework_revision")
    try:
        head = _git(root, "rev-parse", "HEAD")
        pinned = _git(root, "rev-parse", "--verify", f"{required}^{{commit}}")
        status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.TimeoutExpired) as exc:
        validator.error(
            "runtime.framework_git",
            root,
            f"cannot validate framework checkout with Git: {exc}",
        )
        return None, None
    if head.returncode != 0:
        validator.error(
            "runtime.framework_git",
            root,
            "framework root is not a readable Git checkout",
        )
        return None, None
    actual_head = head.stdout.strip()
    information["head"] = actual_head
    information["required_revision"] = required
    if pinned.returncode != 0:
        validator.error(
            "runtime.framework_revision_missing",
            root,
            f"framework checkout does not contain required revision {required}",
        )
    elif actual_head != pinned.stdout.strip():
        validator.error(
            "runtime.framework_revision",
            root,
            f"framework HEAD {actual_head} does not match fixed revision {required}",
        )
    if status.returncode != 0:
        validator.error(
            "runtime.framework_git",
            root,
            "framework clean-worktree validation failed",
        )
    elif status.stdout.strip():
        dirty = status.stdout.splitlines()
        preview = ", ".join(line.strip() for line in dirty[:5])
        if len(dirty) > 5:
            preview += f", ... ({len(dirty)} entries)"
        validator.error(
            "runtime.framework_dirty",
            root,
            f"framework checkout must be clean: {preview}",
        )
    return root, information


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate v6.2 native operator assets without importing KernelGen Server."
        )
    )
    parser.add_argument(
        "catalog",
        metavar="PATH",
        type=Path,
        help="Catalog root, operator directory, or delivery bundle",
    )
    parser.add_argument(
        "--delivery",
        action="store_true",
        help=(
            "validate one operator directory or a bundle of operator directories "
            "without requiring manifest.json or ops/"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("schema", "standard"),
        default="standard",
        help="schema accepts protocol-minimal assets; standard requires both phases",
    )
    parser.add_argument(
        "--operator",
        action="append",
        default=[],
        help="validate only this Definition.name; repeat for multiple operators",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="import and smoke every selected oracle workload in isolated processes",
    )
    parser.add_argument(
        "--target-device",
        help="Torch device used when REFERENCE_DEVICE='target', for example cuda:0",
    )
    parser.add_argument(
        "--framework-root",
        type=Path,
        help=(
            "framework checkout whose src/ is made available to oracle imports"
        ),
    )
    parser.add_argument(
        "--runtime-timeout",
        type=float,
        default=300.0,
        help="timeout in seconds for each operator runtime smoke",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress per-operator runtime progress written to stderr",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="also write the complete JSON report")
    args = parser.parse_args()
    if args.runtime_timeout <= 0:
        parser.error("--runtime-timeout must be positive")
    return args


def _render_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    state = "PASS" if report["passed"] else "FAIL"
    lines = [
        f"{state} {report['catalog']} ({report['mode']})",
        (
            f"operators={summary['operators']} "
            f"correctness={summary['correctness_workloads']} "
            f"timing={summary['timing_workloads']} "
            f"assets={summary['asset_files']} files/{summary['asset_bytes']} bytes "
            f"errors={summary['errors']} warnings={summary['warnings']}"
        ),
    ]
    for issue in report["issues"]:
        location = issue["path"]
        if issue.get("line") is not None:
            location += f":{issue['line']}"
        operator = f" [{issue['operator']}]" if issue.get("operator") else ""
        lines.append(
            f"{issue['severity'].upper()} {issue['code']} {location}{operator}: "
            f"{issue['message']}"
        )
    runtime = report.get("runtime")
    if runtime:
        passed = sum(value.get("passed") is True for value in runtime["operators"].values())
        lines.append(
            f"runtime={'PASS' if runtime['passed'] else 'FAIL'} "
            f"operators={passed}/{len(runtime['operators'])} "
            f"target_device={runtime.get('target_device')!r}"
        )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    validator = Validator(
        args.catalog,
        mode=args.mode,
        delivery=args.delivery,
        selected_operators=set(args.operator) or None,
    )
    validator.validate()
    report = validator.report()
    if args.runtime and report["passed"]:
        framework_root, framework = _validate_runtime_framework(
            validator, args.framework_root
        )
        report = validator.report()
        if report["passed"]:
            runtime = _run_runtime_validation(
                validator,
                target_device=args.target_device,
                framework_root=framework_root,
                timeout_seconds=args.runtime_timeout,
                progress=not args.no_progress,
            )
            runtime["framework"] = framework
            report = validator.report() | {"runtime": runtime}
            report["passed"] = report["passed"] and runtime["passed"]
        else:
            report["runtime"] = {
                "enabled": False,
                "passed": False,
                "reason": "runtime framework validation failed",
                "target_device": args.target_device,
                "framework_root": (
                    str(args.framework_root.resolve())
                    if args.framework_root is not None
                    else None
                ),
                "operators": {},
            }
    elif args.runtime:
        report["runtime"] = {
            "enabled": False,
            "passed": False,
            "reason": "runtime validation was skipped because static validation failed",
            "target_device": args.target_device,
            "framework_root": None,
            "operators": {},
        }
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded if args.format == "json" else _render_text(report) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

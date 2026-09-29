"""Package extracted FlagGems operators as a V6.2 native catalog.

``addmm_`` keeps a fully deterministic source extractor below. Other operators
are first audited by :class:`FlagGemsExtractorAgent`, then converted from its
staging format. Both paths use FlagGems' authoritative ``--list-cases`` output
for timing case identity and cardinality.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .adapter import extract_flaggems_definition
from .models import Definition as LegacyDefinition
from .models import Workload as LegacyWorkload
from .source_inventory import FlagGemsSourceInventory, collect_source_inventory
from .workload_validation import InputToken, parse_and_bind_call


_CASE_LIST_SCHEMA = "flaggems.benchmark-case-list/v2"
_ORACLE_HEADER = '''import torch


REFERENCE_DEVICE = "target"
'''
_COLUMN_MAJOR_INPUT_HOOK = '''def gen_inputs(ctx, device):
    """Materialize the column-major mat2 cases present in this catalog."""
    spec = ctx["inputs"].get("mat2")
    if not isinstance(spec, dict) or spec.get("type") != "custom":
        return None
    params = spec.get("generator_params", {})
    if params.get("kind") != "column_major":
        raise ValueError("unsupported addmm_ custom mat2 recipe")
    k, n = spec["shape"]
    dtype = getattr(torch, spec["dtype"])
    return {"mat2": torch.randn((n, k), dtype=dtype, device=device).t()}
'''
_TIMING_RUN = '''def timing_run(self, mat1, mat2, *, beta=1, alpha=1):
    # FlagGems benchmark/base.py disables TF32 for the Torch baseline. Keep
    # that policy local to this invocation so importing oracle.py has no
    # process-wide side effects and the candidate runs under its own policy.
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        return self.addmm_(mat1, mat2, beta=beta, alpha=alpha)
    finally:
        torch.set_float32_matmul_precision(precision)
'''
_CORRECTNESS_RUN = '''def correctness_run(self, mat1, mat2, *, beta=1, alpha=1):
    self.copy_(torch.addmm(self, mat1, mat2, beta=beta, alpha=alpha))
    return self
'''


@dataclass(frozen=True)
class FlagGemsV62AddmmExtraction:
    catalog_root: Path
    operator_root: Path
    definition_path: Path
    oracle_path: Path
    correctness_path: Path
    timing_path: Path
    num_correctness_workloads: int
    num_timing_workloads: int


@dataclass(frozen=True)
class FlagGemsV62Extraction:
    catalog_root: Path
    operator_root: Path
    definition_path: Path
    oracle_path: Path
    correctness_path: Path | None
    timing_path: Path | None
    num_correctness_workloads: int
    num_timing_workloads: int


def _one_named_path(paths: Iterable[Path], name: str, label: str) -> Path:
    matches = [path for path in paths if path.name == name]
    if len(matches) != 1:
        raise ValueError(f"addmm_: expected one {label} named {name}, got {matches}")
    return matches[0]


def _assignment(nodes: Sequence[ast.stmt], name: str) -> ast.expr | None:
    for node in nodes:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return node.value
    return None


def _literal_assignment(path: Path, name: str) -> Any:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    expression = _assignment(module.body, name)
    if expression is None:
        raise ValueError(f"addmm_: {path} does not define {name}")
    try:
        return ast.literal_eval(expression)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"addmm_: {path}:{name} is not a literal") from exc


def _full_mode_shapes(test_path: Path) -> tuple[tuple[int, int, int], ...]:
    module = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
    expression: ast.expr | None = None
    for node in module.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "QUICK_MODE"
        ):
            expression = _assignment(node.orelse, "MNK_SHAPES")
            if expression is not None:
                break
    if expression is None:
        expression = _assignment(module.body, "MNK_SHAPES")
    if expression is None:
        raise ValueError(f"addmm_: {test_path} does not define full MNK_SHAPES")
    try:
        value = ast.literal_eval(expression)
    except (TypeError, ValueError) as exc:
        raise ValueError("addmm_: full MNK_SHAPES must be literal") from exc
    if not (
        isinstance(value, (list, tuple))
        and value
        and all(
            isinstance(shape, (list, tuple))
            and len(shape) == 3
            and all(isinstance(item, int) and item > 0 for item in shape)
            for shape in value
        )
    ):
        raise ValueError("addmm_: full MNK_SHAPES must contain positive M/N/K triples")
    return tuple(tuple(shape) for shape in value)


def _correctness_contract(
    repo: Path,
    inventory: FlagGemsSourceInventory,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[int | float, ...]]:
    test_path = _one_named_path(inventory.test_files, "test_addmm_.py", "accuracy test")
    _validate_correctness_reference_semantics(test_path)
    shapes = _full_mode_shapes(test_path)
    scalars = _literal_assignment(repo / "tests" / "accuracy_utils.py", "SCALARS")
    if not (
        isinstance(scalars, (list, tuple))
        and scalars
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in scalars
        )
    ):
        raise ValueError("addmm_: accuracy_utils.SCALARS must be numeric literals")
    return shapes, tuple(scalars)


def _is_true_literal(expression: ast.expr) -> bool:
    try:
        return ast.literal_eval(expression) is True
    except (TypeError, ValueError):
        return False


def _validate_correctness_reference_semantics(test_path: Path) -> None:
    """Recognize the high-precision oracle used by FlagGems' addmm_ pytest."""

    module = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
    function = next(
        (
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "test_addmm_"
        ),
        None,
    )
    if function is None:
        raise ValueError(f"addmm_: {test_path} does not define test_addmm_()")

    upcast_calls = 0
    has_addmm_reference = False
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (
            node.func.attr == "to_reference"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "utils"
        ):
            positional_upcast = len(node.args) >= 2 and _is_true_literal(node.args[1])
            keyword_upcast = any(
                keyword.arg == "upcast"
                and keyword.value is not None
                and _is_true_literal(keyword.value)
                for keyword in node.keywords
            )
            if positional_upcast or keyword_upcast:
                upcast_calls += 1
        elif node.func.attr == "addmm_":
            has_addmm_reference = True

    if upcast_calls < 3 or not has_addmm_reference:
        raise ValueError(
            "addmm_: accuracy test no longer has the recognized high-precision "
            "self/mat1/mat2 addmm_ reference; refusing to emit stale "
            "correctness_run()"
        )


def _run_timing_case_listing(
    repo: Path,
    inventory: FlagGemsSourceInventory,
    python_executable: str,
) -> dict[str, Any]:
    benchmark_path = _one_named_path(
        inventory.benchmark_files,
        "test_addmm_.py",
        "benchmark test",
    )
    with tempfile.TemporaryDirectory(prefix="kernelgen-addmm-cases-") as directory:
        output_path = Path(directory) / "cases.json"
        command = [
            python_executable,
            "-m",
            "pytest",
            benchmark_path.relative_to(repo).as_posix(),
            "--level",
            "core",
            "--list-cases",
            "--output",
            str(output_path),
            "-q",
        ]
        completed = subprocess.run(
            command,
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            details = "\n".join(
                part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
            )
            raise RuntimeError(
                "addmm_: FlagGems timing case listing failed "
                f"with exit code {completed.returncode}:\n{details[-4000:]}"
            )
        if not output_path.is_file():
            raise RuntimeError("addmm_: FlagGems --list-cases did not write its JSON output")
        return json.loads(output_path.read_text(encoding="utf-8"))


def _load_case_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"addmm_: cannot read case-list JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("addmm_: case-list report must be a JSON object")
    return value


def _addmm_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("schema_version") != _CASE_LIST_SCHEMA:
        raise ValueError(
            "addmm_: timing report must use "
            f"{_CASE_LIST_SCHEMA}, got {report.get('schema_version')!r}"
        )
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError("addmm_: timing report benchmarks must be a list")
    matches = [
        item
        for item in benchmarks
        if isinstance(item, dict) and item.get("op_name") == "addmm_"
    ]
    if len(matches) != 1:
        raise ValueError(f"addmm_: expected one benchmark case list, got {len(matches)}")
    benchmark = matches[0]
    if benchmark.get("schema_version") != _CASE_LIST_SCHEMA:
        raise ValueError("addmm_: nested benchmark case-list schema is invalid")
    if benchmark.get("phase") != "timing" or benchmark.get("level") != "core":
        raise ValueError("addmm_: timing cases must have phase=timing and level=core")
    cases = benchmark.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("addmm_: timing case list is empty")
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or any(
        not isinstance(case_id, str) or not case_id for case_id in case_ids
    ):
        raise ValueError("addmm_: every timing case requires a non-empty case_id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("addmm_: timing case_id values must be unique")
    return cases


def _dtype_token(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"addmm_: invalid timing dtype {value!r}")
    token = value.removeprefix("torch.")
    if not token.isidentifier():
        raise ValueError(f"addmm_: invalid normalized timing dtype {token!r}")
    return token


def _positive_dimension(shape: dict[str, Any], name: str) -> int:
    value = shape.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"addmm_: timing shape {name} must be a positive integer")
    return value


def _timing_workload(case: dict[str, Any]) -> dict[str, Any]:
    shape = case.get("shape")
    params = case.get("params", {})
    if not isinstance(shape, dict) or not isinstance(params, dict):
        raise ValueError("addmm_: timing case shape/params must be objects")
    unknown_params = set(params) - {"b_column_major"}
    if unknown_params:
        raise ValueError(f"addmm_: unsupported timing params {sorted(unknown_params)}")
    _positive_dimension(shape, "b")
    m = _positive_dimension(shape, "m")
    n = _positive_dimension(shape, "n")
    k = _positive_dimension(shape, "k")
    dtype = _dtype_token(case.get("dtype"))
    column_major = params.get("b_column_major", False)
    if not isinstance(column_major, bool):
        raise ValueError("addmm_: b_column_major must be boolean")
    mat2: dict[str, Any] = {
        "type": "random",
        "shape": [k, n],
        "dtype": dtype,
    }
    if column_major:
        mat2 = {
            "type": "custom",
            "shape": [k, n],
            "dtype": dtype,
            "generator_params": {"kind": "column_major"},
        }
    return {
        "name": case["case_id"],
        "inputs": {
            "self": {"type": "random", "shape": [m, n], "dtype": dtype},
            "mat1": {"type": "random", "shape": [m, k], "dtype": dtype},
            "mat2": mat2,
        },
        "seed": 0,
    }


def _oracle_source(cases: Iterable[dict[str, Any]]) -> str:
    needs_column_major_hook = any(
        case.get("params", {}).get("b_column_major", False) for case in cases
    )
    sections = [_ORACLE_HEADER]
    if needs_column_major_hook:
        sections.append(_COLUMN_MAJOR_INPUT_HOOK)
    sections.append(_TIMING_RUN)
    sections.append(_CORRECTNESS_RUN)
    return "\n\n\n".join(section.rstrip() for section in sections) + "\n"


def _ordered_dtypes(cases: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    result: list[str] = []
    for case in cases:
        dtype = _dtype_token(case.get("dtype"))
        if dtype not in result:
            result.append(dtype)
    return tuple(result)


def _scalar_name(value: int | float) -> str:
    token = str(value).replace("-", "neg").replace(".", "_")
    return token.replace("+", "")


def _correctness_workloads(
    dtypes: Iterable[str],
    shapes: Iterable[tuple[int, int, int]],
    scalars: Iterable[int | float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for dtype in dtypes:
        for scalar in scalars:
            for m, n, k in shapes:
                result.append(
                    {
                        "name": (
                            f"addmm_-corr-{dtype}-{_scalar_name(scalar)}-"
                            f"m{m}n{n}k{k}"
                        ),
                        "inputs": {
                            "self": {
                                "type": "random",
                                "shape": [m, n],
                                "dtype": dtype,
                            },
                            "mat1": {
                                "type": "random",
                                "shape": [m, k],
                                "dtype": dtype,
                            },
                            "mat2": {
                                "type": "random",
                                "shape": [k, n],
                                "dtype": dtype,
                            },
                            "alpha": {"type": "scalar", "value": scalar},
                            "beta": {"type": "scalar", "value": scalar},
                        },
                        "seed": 0,
                        "tolerance": {
                            "atol_scale": float(k),
                            "required_matched_ratio": 1.0,
                        },
                    }
                )
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(value, ensure_ascii=False, separators=(",", ":")) for value in values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_flaggems_addmm_v62(
    flaggems_repo: str | Path,
    catalog_root: str | Path,
    *,
    case_list_path: str | Path | None = None,
    python_executable: str = sys.executable,
) -> FlagGemsV62AddmmExtraction:
    """Extract one self-contained V6.2 ``addmm_`` native catalog package.

    When ``case_list_path`` is omitted, the extractor invokes FlagGems pytest
    with ``--level core --list-cases``.  Supplying a previously captured report
    supports extraction on a host without the target Torch/vendor runtime.
    """

    repo = Path(flaggems_repo).expanduser().resolve()
    root = Path(catalog_root).expanduser().resolve()
    inventory = collect_source_inventory(str(repo), "addmm_")
    definition = extract_flaggems_definition(repo, "addmm_").model_dump(
        mode="json",
        exclude_unset=True,
    )
    definition["api_version"] = "v6.2"
    reference_fields = {
        "reference",
        "correctness_reference",
        "timing_reference",
        "torch_reference",
        "reference_device",
    }
    if reference_fields & definition.keys():
        raise ValueError("addmm_: v6.2 definition.json must contain only the public ABI")

    report = (
        _load_case_report(Path(case_list_path).expanduser().resolve())
        if case_list_path is not None
        else _run_timing_case_listing(repo, inventory, python_executable)
    )
    cases = _addmm_cases(report)
    timing = [_timing_workload(case) for case in cases]
    oracle_source = _oracle_source(cases)
    shapes, scalars = _correctness_contract(repo, inventory)
    correctness = _correctness_workloads(_ordered_dtypes(cases), shapes, scalars)

    manifest = {
        "api_version": "v6.2",
        "evaluator": "native",
        "layout": "per-operator",
    }
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("addmm_: catalog manifest is not the V6.2 native layout")
    operator_root = root / "ops" / "addmm_"
    if operator_root.exists():
        raise FileExistsError(
            f"addmm_: refusing to overwrite existing operator directory {operator_root}"
        )

    root.mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(exist_ok=True)
    operator_root.mkdir()
    if not manifest_path.is_file():
        _write_json(manifest_path, manifest)
    definition_path = operator_root / "definition.json"
    oracle_path = operator_root / "oracle.py"
    correctness_path = operator_root / "correctness.jsonl"
    timing_path = operator_root / "timing.jsonl"
    _write_json(definition_path, definition)
    oracle_path.write_text(oracle_source, encoding="utf-8")
    _write_jsonl(correctness_path, correctness)
    _write_jsonl(timing_path, timing)
    return FlagGemsV62AddmmExtraction(
        catalog_root=root,
        operator_root=operator_root,
        definition_path=definition_path,
        oracle_path=oracle_path,
        correctness_path=correctness_path,
        timing_path=timing_path,
        num_correctness_workloads=len(correctness),
        num_timing_workloads=len(timing),
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object: {path}:{line_number}")
        values.append(value)
    return values


def _legacy_entries(
    staging_root: Path,
    operator: str,
) -> list[tuple[dict[str, Any], Path]]:
    manifest = _read_json(staging_root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("api_version") != "v6.0":
        raise ValueError("staging catalog must be a FlagGemsExtractorAgent v6.0 catalog")
    entries = [
        entry
        for entry in manifest.get("operators", [])
        if isinstance(entry, dict) and entry.get("name") == operator
    ]
    if not entries:
        entries = [
            entry
            for entry in manifest.get("operators", [])
            if isinstance(entry, dict)
            and str(entry.get("name", "")).lstrip("_") == operator.lstrip("_")
        ]
    if not entries:
        raise ValueError(f"{operator}: no extracted staging definition")
    result: list[tuple[dict[str, Any], Path]] = []
    for entry in entries:
        definition_path = staging_root / str(entry.get("definition", ""))
        if not definition_path.is_file():
            raise ValueError(f"{operator}: missing staging definition {definition_path}")
        result.append((entry, definition_path))
    return result


def _abi_key(definition: LegacyDefinition) -> tuple[Any, ...]:
    return tuple(
        (
            parameter.name,
            parameter.kind,
            True if parameter.kind == "var_positional" else parameter.required,
            json.dumps(parameter.default, sort_keys=True),
        )
        for parameter in definition.parameters
    )


def _public_definition(
    definition: LegacyDefinition,
    adapter_definition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if adapter_definition is not None:
        value = dict(adapter_definition)
        value["api_version"] = "v6.2"
        forbidden = {
            "reference",
            "correctness_reference",
            "timing_reference",
            "torch_reference",
            "reference_device",
            "custom_valid_entrypoint",
        }
        for field in forbidden:
            value.pop(field, None)
        return value

    parameters: list[dict[str, Any]] = []
    for parameter in definition.parameters:
        if parameter.kind == "var_keyword":
            raise ValueError(
                f"{definition.name}: v6.2 does not accept unresolved **kwargs"
            )
        value: dict[str, Any] = {
            "name": parameter.name,
            "kind": parameter.kind,
            "required": True if parameter.kind == "var_positional" else parameter.required,
        }
        if parameter.type is not None:
            value["type_hint"] = parameter.type
        if parameter.required is False:
            value["default"] = parameter.default
        parameters.append(value)
    if definition.effects.cases:
        raise ValueError(
            f"{definition.name}: conditional effects require explicit v6.2 audit"
        )
    return {
        "api_version": "v6.2",
        "name": definition.name,
        "description": definition.description,
        "parameters": parameters,
        "outputs": list(definition.outputs),
        "effects": {
            "mutates": list(definition.effects.mutates),
            "returns_alias_of": dict(definition.effects.returns_alias_of),
        },
    }


def _adapter_definition(
    adapter_catalog_root: str | Path | None,
    operator: str,
) -> dict[str, Any] | None:
    if adapter_catalog_root is None:
        return None
    root = Path(adapter_catalog_root).expanduser().resolve()
    matches: list[dict[str, Any]] = []
    for path in sorted((root / "definitions").rglob("*.json")):
        value = _read_json(path)
        if isinstance(value, dict) and value.get("name") == operator:
            matches.append(value)
    if len(matches) != 1:
        raise ValueError(
            f"{operator}: expected one adapter Definition, got {len(matches)}"
        )
    return matches[0]


def _definition_abi_key(value: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (
            parameter.get("name"),
            parameter.get("kind", "positional_or_keyword"),
            parameter.get("required"),
            json.dumps(parameter.get("default"), sort_keys=True),
        )
        for parameter in value.get("parameters", [])
        if isinstance(parameter, dict)
    )


def _token_name(value: Any, *, operator: str, workload: str, parameter: str) -> str:
    if isinstance(value, InputToken):
        return value.name
    if (
        isinstance(value, tuple)
        and len(value) == 1
        and isinstance(value[0], InputToken)
    ):
        return value[0].name
    raise ValueError(
        f"{operator}/{workload}: parameter {parameter!r} cannot be represented "
        "without Workload.call; use one named input (or one custom *args input)"
    )


def _translate_workload(
    definition: LegacyDefinition,
    raw: dict[str, Any],
    *,
    has_gen_inputs: bool,
) -> dict[str, Any]:
    workload = LegacyWorkload.model_validate(raw)
    if workload.expect is not None:
        raise ValueError(
            f"{definition.name}/{workload.name}: expected-exception workloads "
            "are not representable in the v6.2 native runner"
        )
    bound = parse_and_bind_call(definition, workload)
    translated_inputs: dict[str, Any] = {}
    aliases: dict[str, str] = {}
    for parameter_name, value in bound.arguments.items():
        source_name = _token_name(
            value,
            operator=definition.name,
            workload=workload.name,
            parameter=parameter_name,
        )
        translated_inputs[parameter_name] = raw["inputs"][source_name]
        if parameter_name != source_name:
            aliases[parameter_name] = source_name
    if aliases and has_gen_inputs:
        translated_inputs["_kgs_input_aliases"] = aliases
    translated: dict[str, Any] = {
        "name": workload.name,
        "inputs": translated_inputs,
        "seed": workload.seed,
    }
    if workload.tolerance is not None:
        translated["tolerance"] = workload.tolerance.model_dump(
            mode="json", exclude_none=True
        )
    return translated


def _has_function(source: str, name: str) -> bool:
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"invalid extracted reference source: {exc}") from exc
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in module.body
    )


def _oracle_from_legacy(
    definition: LegacyDefinition,
    correctness_source: str,
) -> str:
    timing_source = definition.reference
    has_gen_inputs = _has_function(timing_source, "gen_inputs")
    has_valid = _has_function(timing_source, "valid")
    lines = [
        "import types as _types",
        "",
        f'REFERENCE_DEVICE = {definition.reference_device!r}',
        "",
        "",
        "def _load_extracted_source(name, source):",
        "    module = _types.ModuleType(name)",
        "    exec(compile(source, f\"<{name}>\", \"exec\"), module.__dict__)",
        "    return module",
        "",
        "",
        f"_TIMING_SOURCE = {timing_source!r}",
        "_timing_oracle = _load_extracted_source(\"timing_reference\", _TIMING_SOURCE)",
        "timing_run = _timing_oracle.run",
    ]
    if correctness_source == timing_source:
        lines.extend(["correctness_run = timing_run"])
    else:
        lines.extend(
            [
                "",
                f"_CORRECTNESS_SOURCE = {correctness_source!r}",
                "_correctness_oracle = _load_extracted_source(",
                "    \"correctness_reference\", _CORRECTNESS_SOURCE",
                ")",
                "correctness_run = _correctness_oracle.run",
            ]
        )
    if has_gen_inputs:
        lines.extend(
            [
                "",
                "",
                "def gen_inputs(ctx, device):",
                "    aliases = ctx['inputs'].get('_kgs_input_aliases', {})",
                "    legacy_ctx = dict(ctx)",
                "    legacy_ctx['inputs'] = {",
                "        aliases.get(name, name): spec",
                "        for name, spec in ctx['inputs'].items()",
                "        if name != '_kgs_input_aliases'",
                "    }",
                "    return _timing_oracle.gen_inputs(legacy_ctx, device)",
            ]
        )
    if has_valid:
        lines.extend(["", "valid = _timing_oracle.valid"])
    return "\n".join(lines).rstrip() + "\n"


def _operator_cases(report: dict[str, Any], operator: str) -> list[dict[str, Any]]:
    if report.get("schema_version") != _CASE_LIST_SCHEMA:
        raise ValueError(
            f"{operator}: timing report must use {_CASE_LIST_SCHEMA}"
        )
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise ValueError(f"{operator}: timing report benchmarks must be a list")
    matches = [
        benchmark
        for benchmark in benchmarks
        if isinstance(benchmark, dict) and benchmark.get("op_name") == operator
    ]
    if not matches:
        raise ValueError(
            f"{operator}: authoritative timing report has no matching benchmark"
        )
    cases: list[dict[str, Any]] = []
    for benchmark in matches:
        if (
            benchmark.get("schema_version") != _CASE_LIST_SCHEMA
            or benchmark.get("phase") != "timing"
            or benchmark.get("level") != "core"
        ):
            raise ValueError(f"{operator}: invalid nested core timing case list")
        benchmark_cases = benchmark.get("cases")
        if not isinstance(benchmark_cases, list):
            raise ValueError(f"{operator}: timing cases must be a list")
        cases.extend(benchmark_cases)
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or any(
        not isinstance(case_id, str) or not case_id for case_id in case_ids
    ):
        raise ValueError(f"{operator}: every timing case requires a case_id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{operator}: timing case IDs must be unique")
    return cases


def _validate_and_name_timing(
    operator: str,
    workloads: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(workloads) != len(cases):
        raise ValueError(
            f"{operator}: extracted {len(workloads)} timing workloads but "
            f"--list-cases returned {len(cases)}"
        )
    result: list[dict[str, Any]] = []
    for ordinal, (workload, case) in enumerate(zip(workloads, cases, strict=True)):
        expected_dtype = case.get("dtype")
        if isinstance(expected_dtype, str):
            expected_dtype = expected_dtype.removeprefix("torch.")
            extracted_dtypes = {
                spec.get("dtype")
                for spec in workload["inputs"].values()
                if isinstance(spec, dict) and isinstance(spec.get("dtype"), str)
            }
            if extracted_dtypes and expected_dtype not in extracted_dtypes:
                raise ValueError(
                    f"{operator}: timing ordinal {ordinal} dtype mismatch: "
                    f"case={expected_dtype}, extracted={sorted(extracted_dtypes)}"
                )
        value = dict(workload)
        value["name"] = case["case_id"]
        result.append(value)
    return result


def package_flaggems_v62_extraction(
    staging_catalog_root: str | Path,
    catalog_root: str | Path,
    operator: str,
    *,
    case_list_path: str | Path,
    adapter_catalog_root: str | Path | None = None,
) -> FlagGemsV62Extraction:
    """Convert one audited agent extraction to the v6.2 per-operator layout."""

    staging_root = Path(staging_catalog_root).expanduser().resolve()
    root = Path(catalog_root).expanduser().resolve()
    entries = _legacy_entries(staging_root, operator)
    loaded: list[
        tuple[dict[str, Any], LegacyDefinition, list[dict[str, Any]], list[dict[str, Any]]]
    ] = []
    for entry, definition_path in entries:
        definition = LegacyDefinition.model_validate(_read_json(definition_path))
        correctness = _read_jsonl(staging_root / entry["correctness_workloads"])
        timing = _read_jsonl(staging_root / entry["timing_workloads"])
        loaded.append((entry, definition, correctness, timing))

    primary = next((item for item in loaded if item[3]), loaded[0])
    primary_definition = primary[1]
    adapter_definition = _adapter_definition(adapter_catalog_root, operator)
    if (
        adapter_definition is not None
        and _definition_abi_key(adapter_definition) != _abi_key(primary_definition)
    ):
        raise ValueError(
            f"{operator}: extracted ABI differs from the direct adapter Definition"
        )
    if adapter_definition is not None and (
        adapter_definition.get("outputs") != primary_definition.outputs
        or adapter_definition.get("effects", {})
        != {
            "mutates": primary_definition.effects.mutates,
            "returns_alias_of": primary_definition.effects.returns_alias_of,
        }
    ):
        raise ValueError(
            f"{operator}: extracted output/effects differ from the direct adapter Definition"
        )
    for _, definition, _, _ in loaded:
        if _abi_key(definition) != _abi_key(primary_definition):
            raise ValueError(
                f"{operator}: extracted specializations have different public ABIs"
            )
        if (
            definition.outputs != primary_definition.outputs
            or definition.effects != primary_definition.effects
        ):
            raise ValueError(
                f"{operator}: extracted specializations have different output/effects contracts"
            )
        if definition.reference != primary_definition.reference:
            raise ValueError(
                f"{operator}: extracted specializations require different timing references"
            )

    correctness_sources = {
        definition.correctness_reference or definition.reference
        for _, definition, correctness, _ in loaded
        if correctness
    }
    if len(correctness_sources) > 1:
        raise ValueError(
            f"{operator}: extracted specializations require different correctness references"
        )
    correctness_source = next(
        iter(correctness_sources),
        primary_definition.correctness_reference or primary_definition.reference,
    )
    has_gen_inputs = _has_function(primary_definition.reference, "gen_inputs")

    correctness_workloads: list[dict[str, Any]] = []
    timing_workloads: list[dict[str, Any]] = []
    correctness_names: set[str] = set()
    for entry, definition, correctness, timing in loaded:
        record_id = str(entry.get("id", definition.name))
        for raw in correctness:
            translated = _translate_workload(
                definition, raw, has_gen_inputs=has_gen_inputs
            )
            if translated["name"] in correctness_names:
                translated["name"] = f"{record_id}::{translated['name']}"
            correctness_names.add(translated["name"])
            correctness_workloads.append(translated)
        timing_workloads.extend(
            _translate_workload(definition, raw, has_gen_inputs=has_gen_inputs)
            for raw in timing
        )

    report = _load_case_report(Path(case_list_path).expanduser().resolve())
    cases = _operator_cases(report, operator)
    timing_workloads = _validate_and_name_timing(
        operator, timing_workloads, cases
    )
    overlap = correctness_names & {workload["name"] for workload in timing_workloads}
    if overlap:
        raise ValueError(f"{operator}: case IDs overlap across phases: {sorted(overlap)}")

    manifest = {
        "api_version": "v6.2",
        "evaluator": "native",
        "layout": "per-operator",
    }
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and _read_json(manifest_path) != manifest:
        raise ValueError(f"{operator}: target is not a v6.2 native catalog")
    operator_root = root / "ops" / operator
    if operator_root.exists():
        raise FileExistsError(
            f"{operator}: refusing to overwrite existing operator directory {operator_root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(exist_ok=True)
    operator_root.mkdir()
    if not manifest_path.is_file():
        _write_json(manifest_path, manifest)

    definition_path = operator_root / "definition.json"
    oracle_path = operator_root / "oracle.py"
    correctness_path = (
        operator_root / "correctness.jsonl" if correctness_workloads else None
    )
    timing_path = operator_root / "timing.jsonl" if timing_workloads else None
    _write_json(
        definition_path,
        _public_definition(primary_definition, adapter_definition),
    )
    oracle_path.write_text(
        _oracle_from_legacy(primary_definition, correctness_source),
        encoding="utf-8",
    )
    if correctness_path is not None:
        _write_jsonl(correctness_path, correctness_workloads)
    if timing_path is not None:
        _write_jsonl(timing_path, timing_workloads)
    return FlagGemsV62Extraction(
        catalog_root=root,
        operator_root=operator_root,
        definition_path=definition_path,
        oracle_path=oracle_path,
        correctness_path=correctness_path,
        timing_path=timing_path,
        num_correctness_workloads=len(correctness_workloads),
        num_timing_workloads=len(timing_workloads),
    )


__all__ = [
    "FlagGemsV62AddmmExtraction",
    "FlagGemsV62Extraction",
    "extract_flaggems_addmm_v62",
    "package_flaggems_v62_extraction",
]

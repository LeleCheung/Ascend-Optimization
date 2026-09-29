#!/usr/bin/env python3
"""Convert the legacy AKG Bench Lite trace into a Server v5 catalog."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from kernelgen_server import Catalog, KERNELGEN_API_VERSION
from kernelgen_server.schema import Definition, Workload


CATALOG_NAME = "akg-bench-lite-v5"
SOURCE_FORMAT = "unified-trace-akg-bench-lite"
COMPETITION_TOLERANCE = {"rtol": 1e-2, "atol": 1e-2}
COMPETITION_OPERATOR_NAMES = frozenset(
    {
        "abl_t1_fused_silu_and_mul",
        "abl_t1_gelu",
        "abl_t1_matmul_basic",
        "abl_t1_matmul_biasadd",
        "abl_t1_sigmoid_scale_sum",
        "abl_t1_softmax",
        "abl_t2_add_rmsnorm_cast",
        "abl_t2_add_rmsnorm_quant",
        "abl_t2_moe_topk_softmax",
        "abl_t2_rope",
        "abl_t3_causal_conv1d",
        "abl_t3_decode_mla",
        "abl_t3_layernorm_gated",
    }
)


def _read_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        yield value


def _constant_axes(old: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, spec in old.get("axes", {}).items():
        if not isinstance(spec, dict) or spec.get("type") != "const":
            raise ValueError(
                f"{old['name']}: axis {name!r} must be constant for v5 conversion"
            )
        if "value" not in spec:
            raise ValueError(f"{old['name']}: axis {name!r} has no value")
        values[name] = deepcopy(spec["value"])
    return values


def _function_nodes(source: str, definition_name: str) -> dict[str, ast.AST]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"{definition_name}: invalid reference Python: {exc}") from exc
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _argument_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [
        argument.arg
        for argument in node.args.posonlyargs + node.args.args
    ]


def _rename_function_line(
    lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    old_name: str,
    new_name: str,
) -> None:
    index = node.lineno - 1
    pattern = re.compile(
        rf"^(\s*(?:async\s+)?def\s+){re.escape(old_name)}(\s*\()"
    )
    lines[index], count = pattern.subn(
        rf"\g<1>{new_name}\g<2>",
        lines[index],
        count=1,
    )
    if count != 1:
        raise ValueError(
            f"cannot rewrite reference function {old_name!r} at line {node.lineno}"
        )


def _adapt_reference(
    old: dict[str, Any],
    axis_values: dict[str, Any],
) -> tuple[str, str | None, str | None]:
    name = old["name"]
    source = old["reference"].rstrip() + "\n"
    functions = _function_nodes(source, name)
    if "run" not in functions:
        raise ValueError(f"{name}: reference has no run()")

    custom_inputs = old.get("custom_inputs_entrypoint")
    if custom_inputs is None and "gen_inputs" in functions:
        custom_inputs = "gen_inputs"
    custom_valid = old.get("custom_valid_entrypoint")
    if custom_valid is None and "valid" in functions:
        custom_valid = "valid"

    lines = source.splitlines(keepends=True)
    wrappers: list[str] = []

    if custom_inputs is not None:
        node = functions.get(custom_inputs)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"{name}: missing {custom_inputs}()")
        arguments = _argument_names(node)
        if arguments == ["axes", "device"]:
            legacy_name = f"_legacy_{custom_inputs}"
            _rename_function_line(lines, node, custom_inputs, legacy_name)
            wrappers.append(
                "\n"
                f"def {custom_inputs}(ctx, device):\n"
                "    del ctx\n"
                f"    return {legacy_name}({axis_values!r}, device)\n"
            )
        elif arguments != ["ctx", "device"]:
            raise ValueError(
                f"{name}: {custom_inputs}() must accept (axes, device) "
                "or (ctx, device)"
            )

    if custom_valid is not None:
        node = functions.get(custom_valid)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"{name}: missing {custom_valid}()")
        arguments = _argument_names(node)
        if len(arguments) == 3:
            legacy_name = f"_legacy_{custom_valid}"
            _rename_function_line(lines, node, custom_valid, legacy_name)
            wrappers.append(
                "\n"
                f"def {custom_valid}(ref_outputs, sol_outputs, inputs, ctx):\n"
                "    del ref_outputs\n"
                f"    return {legacy_name}(sol_outputs, inputs, ctx)\n"
            )
        elif len(arguments) != 4:
            raise ValueError(
                f"{name}: {custom_valid}() must accept three legacy arguments "
                "or four v5 arguments"
            )

    adapted = "".join(lines) + "".join(wrappers)
    adapted_functions = _function_nodes(adapted, name)
    for entrypoint in ("run", custom_inputs, custom_valid):
        if entrypoint is not None and entrypoint not in adapted_functions:
            raise ValueError(f"{name}: adapted reference is missing {entrypoint}()")
    return adapted, custom_inputs, custom_valid


def _resolve_shape(
    shape: Any,
    axis_values: dict[str, Any],
    *,
    location: str,
) -> list[int]:
    if not isinstance(shape, list):
        raise ValueError(f"{location}: tensor shape must be a list")
    resolved: list[int] = []
    for dimension in shape:
        if isinstance(dimension, str):
            value = (
                int(dimension)
                if dimension.isdecimal()
                else axis_values.get(dimension)
            )
        else:
            value = dimension
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{location}: unresolved or invalid shape dimension {dimension!r}"
            )
        resolved.append(value)
    return resolved


def _new_tolerance(old: dict[str, Any] | None) -> dict[str, Any] | None:
    if not old:
        return None
    result: dict[str, Any] = {}
    if old.get("max_rtol") is not None:
        result["rtol"] = old["max_rtol"]
    if old.get("max_atol") is not None:
        result["atol"] = old["max_atol"]
    if old.get("reduce_dim") is not None:
        result["atol_scale"] = old["reduce_dim"]
    if old.get("required_matched_ratio") is not None:
        result["required_matched_ratio"] = old["required_matched_ratio"]
    return result or None


def _convert_inputs(
    old_definition: dict[str, Any],
    old_workload: dict[str, Any],
    axis_values: dict[str, Any],
) -> dict[str, Any]:
    definition_name = old_definition["name"]
    definition_inputs = old_definition["inputs"]
    workload_inputs = old_workload.get("inputs")
    if not isinstance(workload_inputs, dict):
        raise ValueError(f"{definition_name}: workload inputs must be an object")
    if set(workload_inputs) != set(definition_inputs):
        missing = sorted(set(definition_inputs) - set(workload_inputs))
        extra = sorted(set(workload_inputs) - set(definition_inputs))
        raise ValueError(
            f"{definition_name}: workload inputs differ from Definition: "
            f"missing={missing}, extra={extra}"
        )

    result: dict[str, Any] = {}
    for input_name in definition_inputs:
        old_spec = deepcopy(workload_inputs[input_name])
        kind = old_spec.get("type")
        location = f"{definition_name}/{old_workload.get('uuid')}/{input_name}"
        if kind == "safetensors":
            raise ValueError(f"{location}: safetensors inputs are unsupported")
        if kind in ("random", "custom"):
            definition_spec = definition_inputs[input_name]
            shape = old_spec.get("shape", definition_spec.get("shape"))
            dtype = old_spec.get("dtype", definition_spec.get("dtype"))
            if not isinstance(dtype, str) or not dtype:
                raise ValueError(f"{location}: tensor dtype is required")
            old_spec["shape"] = _resolve_shape(
                shape,
                axis_values,
                location=location,
            )
            old_spec["dtype"] = dtype
        elif kind in ("scalar", "literal"):
            if "value" not in old_spec:
                raise ValueError(f"{location}: {kind} input requires value")
        else:
            raise ValueError(f"{location}: unsupported input type {kind!r}")
        result[input_name] = old_spec
    return result


def _convert_workload(
    trace: dict[str, Any],
    old_definition: dict[str, Any],
    axis_values: dict[str, Any],
    *,
    phase: str,
    index: int,
    tolerance_override: dict[str, Any] | None = None,
) -> Workload:
    definition_name = old_definition["name"]
    if trace.get("definition") != definition_name:
        raise ValueError(
            f"workload points to {trace.get('definition')!r}, "
            f"expected {definition_name!r}"
        )
    workload = trace.get("workload")
    if not isinstance(workload, dict):
        raise ValueError(f"{definition_name}: trace has no workload object")
    workload_axes = workload.get("axes", {})
    if workload_axes:
        raise ValueError(
            f"{definition_name}/{workload.get('uuid')}: per-workload axes are "
            "unsupported; this converter requires constant Definition axes"
        )
    source_name = str(workload.get("uuid") or f"{definition_name}-{index}")
    suffix = "corr" if phase == "correctness" else "time"
    return Workload(
        name=f"{source_name}-{suffix}",
        inputs=_convert_inputs(old_definition, workload, axis_values),
        seed=workload.get("seed", 0) or 0,
        tolerance=(
            deepcopy(tolerance_override)
            if tolerance_override is not None
            else _new_tolerance(workload.get("tolerance"))
        ),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_workloads(path: Path, workloads: list[Workload]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            workload.model_dump(mode="json", exclude_none=True),
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for workload in workloads
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_catalog(root: Path, expected_names: set[str]) -> None:
    catalog = Catalog(root)
    if set(catalog.operator_names) != expected_names:
        raise ValueError("generated Catalog operator names do not match source")
    for name in catalog.operator_names:
        catalog.load(name)


def convert(
    source_root: Path,
    output_root: Path,
    *,
    force: bool = False,
    catalog_name: str = CATALOG_NAME,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    definitions_root = source_root / "definitions"
    workloads_root = source_root / "workloads"
    if not definitions_root.is_dir() or not workloads_root.is_dir():
        raise ValueError(f"not an AKG Bench Lite trace: {source_root}")
    if not catalog_name or Path(catalog_name).name != catalog_name:
        raise ValueError(f"invalid catalog name: {catalog_name!r}")
    if output_root.exists() and not force:
        raise FileExistsError(f"output exists (pass --force to replace): {output_root}")

    parent = output_root.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=parent))
    operators: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    all_workload_names: set[str] = set()
    correctness_total = 0
    timing_total = 0
    tolerance_override = (
        COMPETITION_TOLERANCE if catalog_name == CATALOG_NAME else None
    )
    try:
        for source_definition in sorted(definitions_root.rglob("*.json")):
            old = json.loads(source_definition.read_text(encoding="utf-8"))
            name = old["name"]
            if name in seen_names:
                raise ValueError(f"duplicate definition: {name}")
            seen_names.add(name)
            group = source_definition.parent.relative_to(definitions_root)
            axis_values = _constant_axes(old)

            source_workloads = workloads_root / group / f"{name}.jsonl"
            if not source_workloads.is_file():
                raise FileNotFoundError(source_workloads)
            traces = list(_read_json_lines(source_workloads))
            if not traces:
                raise ValueError(f"{source_workloads}: no workloads")

            reference, custom_inputs, custom_valid = _adapt_reference(
                old,
                axis_values,
            )
            definition = Definition(
                name=name,
                description=old.get("description", ""),
                inputs=list(old["inputs"]),
                outputs=list(old["outputs"]),
                reference=reference,
                reference_device=old.get("reference_device", "target"),
                custom_inputs_entrypoint=custom_inputs,
                custom_valid_entrypoint=custom_valid,
            )
            definition_rel = Path("definitions") / group / f"{name}.json"
            _write_json(
                temporary / definition_rel,
                definition.model_dump(mode="json", exclude_none=True),
            )

            phase_data: dict[str, tuple[Path, list[Workload]]] = {}
            for phase in ("correctness", "timing"):
                workloads = [
                    _convert_workload(
                        trace,
                        old,
                        axis_values,
                        phase=phase,
                        index=index,
                        tolerance_override=tolerance_override,
                    )
                    for index, trace in enumerate(traces)
                ]
                names = [workload.name for workload in workloads]
                if len(names) != len(set(names)):
                    raise ValueError(
                        f"{source_workloads}: duplicate {phase} workload names"
                    )
                overlap = all_workload_names.intersection(names)
                if overlap:
                    raise ValueError(
                        f"globally duplicate workload names: {sorted(overlap)}"
                    )
                all_workload_names.update(names)
                destination = (
                    Path("workloads") / group / f"{name}.{phase}.jsonl"
                )
                _write_workloads(temporary / destination, workloads)
                phase_data[phase] = (destination, workloads)

            correctness_total += len(phase_data["correctness"][1])
            timing_total += len(phase_data["timing"][1])
            operators.append(
                {
                    "name": name,
                    "group": str(group),
                    "definition": str(definition_rel),
                    "correctness_workloads": str(phase_data["correctness"][0]),
                    "timing_workloads": str(phase_data["timing"][0]),
                    "num_correctness_workloads": len(
                        phase_data["correctness"][1]
                    ),
                    "num_timing_workloads": len(
                        phase_data["timing"][1]
                    ),
                }
            )

        workload_files = {
            path.relative_to(workloads_root)
            for path in workloads_root.rglob("*.jsonl")
        }
        expected_workload_files = {
            path.parent.relative_to(definitions_root) / f"{path.stem}.jsonl"
            for path in definitions_root.rglob("*.json")
        }
        if workload_files != expected_workload_files:
            missing = sorted(str(path) for path in expected_workload_files - workload_files)
            extra = sorted(str(path) for path in workload_files - expected_workload_files)
            raise ValueError(
                f"Definition/workload file mismatch: missing={missing}, extra={extra}"
            )
        if catalog_name == CATALOG_NAME and seen_names != COMPETITION_OPERATOR_NAMES:
            missing = sorted(COMPETITION_OPERATOR_NAMES - seen_names)
            extra = sorted(seen_names - COMPETITION_OPERATOR_NAMES)
            raise ValueError(
                f"{CATALOG_NAME} source operator mismatch: "
                f"missing={missing}, extra={extra}"
            )

        manifest = {
            "api_version": KERNELGEN_API_VERSION,
            "name": catalog_name,
            "source_format": SOURCE_FORMAT,
            "generated_by": "tools/convert_akg_bench_lite_v5.py",
            "phase_policy": (
                "Each unphased legacy workload is emitted once for correctness "
                "and once for timing."
            ),
            "tolerance_policy": (
                "All AKG Bench Lite competition workloads use "
                "rtol=0.01 and atol=0.01."
                if tolerance_override is not None
                else "Legacy workload tolerances are preserved."
            ),
            "counts": {
                "operators": len(operators),
                "correctness_workloads": correctness_total,
                "timing_workloads": timing_total,
            },
            "operators": operators,
        }
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "README.md").write_text(
            "# AKG Bench Lite v5 operator catalog\n\n"
            "This directory is generated by "
            "`tools/convert_akg_bench_lite_v5.py`. The legacy source has one "
            "unphased workload per operator, so the converter emits distinct "
            "correctness and timing workload identities with the same inputs. "
            "For competition parity, every emitted workload explicitly uses "
            "`rtol=0.01` and `atol=0.01`.\n\n"
            "Regenerate from a sibling checkout:\n\n"
            "```bash\n"
            "PYTHONPATH=. python3 tools/convert_akg_bench_lite_v5.py "
            "../unified-trace-akg-bench-lite "
            "--output data/.old/akg-bench-lite-v5 --force\n"
            "```\n",
            encoding="utf-8",
        )
        _validate_catalog(temporary, seen_names)
        if output_root.exists():
            shutil.rmtree(output_root)
        temporary.replace(output_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert unified-trace-akg-bench-lite to a Server v5 Catalog."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/.old") / CATALOG_NAME,
    )
    parser.add_argument("--catalog-name", default=CATALOG_NAME)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = convert(
        args.source,
        args.output,
        force=args.force,
        catalog_name=args.catalog_name,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert a legacy KernelGen FlagGems trace into the v5 operator catalog."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from kernelgen_server.schema import Definition, Workload


def _read_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


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


def _convert_workload(trace: dict[str, Any], definition_name: str) -> Workload:
    if trace.get("definition") != definition_name:
        raise ValueError(
            f"workload points to {trace.get('definition')!r}, expected {definition_name!r}"
        )
    workload = trace["workload"]
    for name, spec in workload["inputs"].items():
        if spec["type"] == "safetensors":
            raise ValueError(
                f"{definition_name}/{workload['uuid']}/{name}: safetensors unsupported"
            )
        if spec["type"] in ("random", "custom"):
            if spec.get("shape") is None or spec.get("dtype") is None:
                raise ValueError(
                    f"{definition_name}/{workload['uuid']}/{name}: shape and dtype are required"
                )
    return Workload(
        name=workload["uuid"],
        inputs=workload["inputs"],
        seed=workload.get("seed", 0) or 0,
        tolerance=_new_tolerance(workload.get("tolerance")),
    )


def _reference(old: dict[str, Any]) -> str:
    reference = old["reference"].replace(
        "_v4_dtype", "_dtype_from_context"
    ).rstrip() + "\n"
    tree = ast.parse(reference)
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "run" not in functions:
        raise ValueError(f"{old['name']}: reference has no run()")
    custom_entrypoint = old.get("custom_inputs_entrypoint")
    if custom_entrypoint and custom_entrypoint not in functions:
        raise ValueError(f"{old['name']}: missing {custom_entrypoint}()")
    validator_entrypoint = old.get("custom_valid_entrypoint")
    if validator_entrypoint and validator_entrypoint not in functions:
        raise ValueError(f"{old['name']}: missing {validator_entrypoint}()")
    return reference


def _convert_definition(old: dict[str, Any]) -> Definition:
    return Definition(
        name=old["name"],
        description=old.get("description", ""),
        inputs=list(old["inputs"]),
        outputs=list(old["outputs"]),
        reference=_reference(old),
        custom_inputs_entrypoint=old.get("custom_inputs_entrypoint"),
        custom_valid_entrypoint=old.get("custom_valid_entrypoint"),
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
        )
        for workload in workloads
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert(source_root: Path, output_root: Path, *, force: bool = False) -> dict[str, Any]:
    source_root = source_root.resolve()
    definitions_root = source_root / "definitions"
    workloads_root = source_root / "phased_workloads"
    if not definitions_root.is_dir() or not workloads_root.is_dir():
        raise ValueError(f"not a v4 trace set: {source_root}")
    if output_root.exists() and not force:
        raise FileExistsError(f"output exists (pass --force to replace): {output_root}")

    parent = output_root.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=parent))
    operators = []
    seen_names: set[str] = set()
    all_workload_names: set[str] = set()
    correctness_total = 0
    timing_total = 0
    try:
        for source_definition in sorted(definitions_root.rglob("*.json")):
            old = json.loads(source_definition.read_text(encoding="utf-8"))
            name = old["name"]
            if name in seen_names:
                raise ValueError(f"duplicate definition: {name}")
            seen_names.add(name)
            group = source_definition.parent.relative_to(definitions_root)

            definition = _convert_definition(old)
            definition_rel = Path("definitions") / group / f"{name}.json"
            _write_json(
                temporary / definition_rel,
                definition.model_dump(mode="json", exclude_none=True),
            )

            phase_data: dict[str, tuple[Path, list[Workload]]] = {}
            for phase in ("correctness", "timing"):
                source_workloads = workloads_root / group / f"{name}.{phase}.jsonl"
                if not source_workloads.is_file():
                    raise FileNotFoundError(source_workloads)
                workloads = [
                    _convert_workload(trace, name)
                    for trace in _read_json_lines(source_workloads)
                ]
                if not workloads:
                    raise ValueError(f"{source_workloads}: phase cannot be empty")
                names = [workload.name for workload in workloads]
                if len(names) != len(set(names)):
                    raise ValueError(f"{source_workloads}: duplicate workload names")
                overlap = all_workload_names.intersection(names)
                if overlap:
                    raise ValueError(
                        f"globally duplicate workload names: {sorted(overlap)}"
                    )
                all_workload_names.update(names)
                destination = Path("workloads") / group / f"{name}.{phase}.jsonl"
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
                    "num_timing_workloads": len(phase_data["timing"][1]),
                }
            )

        manifest = {
            "api_version": "v5.1",
            "name": "flaggems-v5",
            "source_format": "unified-trace-flaggems-v4",
            "generated_by": "tools/convert_flaggems_v5.py",
            "counts": {
                "operators": len(operators),
                "correctness_workloads": correctness_total,
                "timing_workloads": timing_total,
            },
            "operators": operators,
        }
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "README.md").write_text(
            "# FlagGems v5 operator catalog\n\n"
            "This directory is generated by `tools/convert_flaggems_v5.py`. "
            "Definitions contain only operator reference code and entrypoints; "
            "workload files contain concrete inputs for correctness and timing.\n\n"
            "The current data was converted from the legacy v4 trace in a sibling "
            "KernelGen checkout:\n\n"
            "```bash\n"
            "PYTHONPATH=. python3 tools/convert_flaggems_v5.py "
            "../kernelgen/trace_sets/unified-trace-flaggems-v4 "
            "--output data/.old/flaggems-v5 --force\n"
            "```\n",
            encoding="utf-8",
        )
        if output_root.exists():
            shutil.rmtree(output_root)
        temporary.replace(output_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/.old/flaggems-v5"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = convert(args.source, args.output, force=args.force)
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert PR #17 KernelBench/KernelSwift v5 data to native v6.2 catalogs."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kernelgen_server import Catalog
from kernelgen_server.schema import Definition, Workload
from tools.convert_kernelgenbench_v5 import (
    _functions,
    _parameters,
    _rename_function,
    _type_hints,
    _write_json,
)


SOURCE_REVISION = "6f767e16a54d8634238a6fb013877e6ffa8c9edb"
SOURCE_API_VERSION = "v5.1"
TARGET_API_VERSION = "v6.2"
CATALOGS = {
    "kernelbench": {
        "source": "kernelbench-v5",
        "expected": (250, 1_250, 250),
    },
    "kernelswift": {
        "source": "kernelswift-v5",
        "expected": (10, 10, 10),
    },
}


class GitTree:
    def __init__(self, repository: Path, revision: str) -> None:
        self.repository = repository.resolve()
        self.revision = revision
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=self.repository,
            check=True,
            capture_output=True,
        )

    def text(self, path: str) -> str:
        return subprocess.run(
            ["git", "show", f"{self.revision}:{path}"],
            cwd=self.repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout


def _json_object(text: str, *, label: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected a JSON object")
    return value


def _workloads(
    tree: GitTree,
    path: str,
    expected_inputs: list[str],
    seen_names: set[str],
) -> tuple[list[dict[str, Any]], str]:
    source = tree.text(path)
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        if not line.strip():
            continue
        raw = _json_object(line, label=f"{path}:{line_number}")
        workload = Workload.model_validate(raw)
        if set(workload.inputs) != set(expected_inputs):
            raise ValueError(
                f"{path}:{line_number}: workload inputs differ from Definition"
            )
        if workload.name in seen_names:
            raise ValueError(f"duplicate workload name: {workload.name}")
        seen_names.add(workload.name)
        result.append(raw)
    if not result:
        raise ValueError(f"empty workload file: {path}")
    return result, source.rstrip() + "\n"


def _legacy_context_wrapper() -> str:
    return (
        "\n\ndef _legacy_context(ctx):\n"
        "    result = {\"__workload_seed\": ctx[\"seed\"]}\n"
        "    for name, spec in ctx[\"inputs\"].items():\n"
        "        kind = spec.get(\"type\") if isinstance(spec, dict) else None\n"
        "        if kind in {\"random\", \"custom\"}:\n"
        "            result[f\"{name}__shape\"] = spec[\"shape\"]\n"
        "            result[f\"{name}__dtype\"] = spec[\"dtype\"]\n"
        "            for key, value in spec.items():\n"
        "                if key not in {\"type\", \"shape\", \"dtype\"}:\n"
        "                    result[f\"{name}__{key}\"] = value\n"
        "        elif kind in {\"scalar\", \"literal\"}:\n"
        "            result[name] = spec[\"value\"]\n"
        "        else:\n"
        "            raise ValueError(f\"unsupported legacy input recipe: {name}\")\n"
        "    return result\n"
    )


def _oracle(old: dict[str, Any]) -> str:
    name = str(old["name"])
    source = str(old["reference"]).rstrip() + "\n"
    functions = _functions(source, name)
    if "run" not in functions:
        raise ValueError(f"{name}: reference has no run()")
    lines = source.splitlines(keepends=True)
    wrappers: list[str] = []
    needs_legacy_context = False

    custom_inputs = old.get("custom_inputs_entrypoint")
    if custom_inputs is not None:
        if custom_inputs != "gen_inputs" or custom_inputs not in functions:
            raise ValueError(f"{name}: unsupported custom input entrypoint")
        _rename_function(
            lines,
            functions[custom_inputs],
            custom_inputs,
            "_legacy_gen_inputs",
        )
        wrappers.append(
            "\n\ndef gen_inputs(ctx, device):\n"
            "    return _legacy_gen_inputs(_legacy_context(ctx), device)\n"
        )
        needs_legacy_context = True

    custom_valid = old.get("custom_valid_entrypoint")
    if custom_valid is not None:
        if custom_valid not in functions:
            raise ValueError(f"{name}: missing custom valid entrypoint")
        _rename_function(
            lines,
            functions[custom_valid],
            custom_valid,
            "_legacy_valid",
        )
        wrappers.append(
            "\n\nVALID_OWNS_RETURN_CONTRACT = True\n\n"
            "def valid(ref_outputs, sol_outputs, inputs, ctx):\n"
            f"    ordered = [inputs[name] for name in {list(old['inputs'])!r}]\n"
            "    return _legacy_valid(\n"
            "        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)\n"
            "    )\n"
        )
        needs_legacy_context = True

    oracle = (
        f'REFERENCE_DEVICE = {old.get("reference_device", "target")!r}\n\n'
        + "".join(lines)
        + (_legacy_context_wrapper() if needs_legacy_context else "")
        + "".join(wrappers)
    )
    oracle = "\n".join(line.rstrip() for line in oracle.splitlines()) + "\n"
    ast.parse(oracle, filename=f"{name}/oracle.py")
    return oracle


def _validate_generated(root: Path, names: set[str]) -> None:
    catalog = Catalog(root)
    if set(catalog.operator_names) != names:
        raise ValueError("generated operator names differ from source")
    for name in catalog.operator_names:
        operator = catalog.load(name)
        if operator.definition.reference is None:
            raise ValueError(f"{name}: oracle was not attached")


def convert_catalog(
    tree: GitTree,
    catalog_name: str,
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    config = CATALOGS[catalog_name]
    source_name = str(config["source"])
    source_root = f"data/{source_name}"
    source_manifest = _json_object(
        tree.text(f"{source_root}/manifest.json"),
        label=f"{source_root}/manifest.json",
    )
    if source_manifest.get("api_version") != SOURCE_API_VERSION:
        raise ValueError(f"{source_name}: expected {SOURCE_API_VERSION}")
    records = {
        str(item["name"]): item
        for item in source_manifest.get("operators", [])
        if isinstance(item, dict) and item.get("name")
    }

    output_root = output_root.resolve()
    if output_root.exists() and not force:
        raise FileExistsError(f"output exists (pass --force): {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    names: set[str] = set()
    workload_names: set[str] = set()
    operators: list[dict[str, Any]] = []
    totals = {"correctness": 0, "timing": 0}
    try:
        for name, source_record in sorted(records.items()):
            group = str(source_record["group"])
            definition_path = str(source_record["definition"])
            old = _json_object(
                tree.text(f"{source_root}/{definition_path}"),
                label=definition_path,
            )
            if old.get("name") != name or old.get("api_version") != SOURCE_API_VERSION:
                raise ValueError(f"{name}: source Definition identity changed")
            inputs = list(old["inputs"])
            phase_workloads: dict[str, list[dict[str, Any]]] = {}
            phase_sources: dict[str, str] = {}
            for phase in ("correctness", "timing"):
                relative = str(source_record[f"{phase}_workloads"])
                workloads, text = _workloads(
                    tree,
                    f"{source_root}/{relative}",
                    inputs,
                    workload_names,
                )
                phase_workloads[phase] = workloads
                phase_sources[phase] = text
                totals[phase] += len(workloads)

            functions = _functions(str(old["reference"]), name)
            combined = phase_workloads["correctness"] + phase_workloads["timing"]
            definition = Definition.model_validate(
                {
                    "api_version": TARGET_API_VERSION,
                    "name": name,
                    "description": old.get("description", ""),
                    "parameters": _parameters(
                        old,
                        functions["run"],
                        _type_hints(inputs, combined),
                    ),
                    "outputs": list(old["outputs"]),
                    "effects": {"mutates": [], "returns_alias_of": {}},
                }
            )
            operator_root = temporary / "ops" / group / name
            _write_json(
                operator_root / "definition.json",
                definition.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude_unset=True,
                ),
            )
            (operator_root / "oracle.py").write_text(
                _oracle(old),
                encoding="utf-8",
            )
            for phase in ("correctness", "timing"):
                (operator_root / f"{phase}.jsonl").write_text(
                    phase_sources[phase],
                    encoding="utf-8",
                )
            names.add(name)
            operator_manifest = {
                "name": name,
                "group": group,
                "num_correctness_workloads": len(
                    phase_workloads["correctness"]
                ),
                "num_timing_workloads": len(phase_workloads["timing"]),
            }
            for field in ("problem_id", "source"):
                if field in source_record:
                    operator_manifest[field] = source_record[field]
            operators.append(operator_manifest)

        counts = {
            "operators": len(operators),
            "correctness_workloads": totals["correctness"],
            "timing_workloads": totals["timing"],
        }
        expected_tuple = tuple(config["expected"])
        if tuple(counts.values()) != expected_tuple:
            raise ValueError(
                f"{catalog_name}: source counts changed: "
                f"{tuple(counts.values())} != {expected_tuple}"
            )
        source_counts = source_manifest.get("counts", {})
        if any(source_counts.get(key) != value for key, value in counts.items()):
            raise ValueError(f"{catalog_name}: source manifest counts are inconsistent")

        manifest = {
            "api_version": TARGET_API_VERSION,
            "name": catalog_name,
            "evaluator": "native",
            "layout": "per-operator",
            "source_format": source_manifest.get("source_format"),
            "source_api_version": SOURCE_API_VERSION,
            "source_revision": tree.revision,
            "generated_by": "tools/convert_pr17_model_catalogs_v62.py",
            "counts": counts,
            "operators": operators,
        }
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "README.md").write_text(
            f"# {catalog_name} v6.2 native catalog\n\n"
            f"Deterministically converted from `{source_root}` at Git revision "
            f"`{tree.revision}` by `tools/convert_pr17_model_catalogs_v62.py`. "
            "Embedded references are stored in each operator's `oracle.py`; "
            "Definition files contain only the v6.2 public ABI.\n",
            encoding="utf-8",
        )
        if catalog_name == "kernelbench":
            shutil.copyfile(
                REPOSITORY_ROOT / "licenses" / "KernelBench-MIT.txt",
                temporary / "LICENSE",
            )
        _validate_generated(temporary, names)

        backup: Path | None = None
        if output_root.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{output_root.name}-backup-",
                    dir=output_root.parent,
                )
            )
            backup.rmdir()
            output_root.replace(backup)
        try:
            temporary.replace(output_root)
        except BaseException:
            if backup is not None and not output_root.exists():
                backup.replace(output_root)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "catalog",
        choices=tuple(CATALOGS),
        help="source catalog to convert",
    )
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output or Path("data") / args.catalog
    manifest = convert_catalog(
        GitTree(args.repository, args.revision),
        args.catalog,
        output,
        force=args.force,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()

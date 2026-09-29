#!/usr/bin/env python3
"""Convert the checked-in KernelGenBench v5.1 catalog to native v6.2."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from kernelgen_server import Catalog
from kernelgen_server.schema import Definition, Workload
from tools.sample_v62_catalog_workloads import ALGORITHM, sample_catalog


CATALOG_NAME = "kernelgenbench"
SOURCE_API_VERSION = "v5.1"
TARGET_API_VERSION = "v6.2"
WORKLOAD_LIMIT = 200
EXPECTED_FULL_COUNTS = {
    "operators": 210,
    "correctness_workloads": 57_603,
    "timing_workloads": 28_198,
}
EXPECTED_COUNTS = {
    "operators": 210,
    "correctness_workloads": 15_353,
    "timing_workloads": 11_695,
}


# These effects are part of the public operator contract but were implicit in
# the v5.1 runner. Values are parameter names; aliases map output names to the
# mutated parameter returned by the reference.
_POINTWISE_EFFECTS: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {
    "kernelgenbench__index_put_impl_": (("x",), {"output": "x"}),
    "kernelgenbench_add_": (("x",), {"output": "x"}),
    "kernelgenbench_copy_": (("x",), {"output": "x"}),
    "kernelgenbench_div_": (("x",), {"output": "x"}),
    "kernelgenbench_exponential_": (("x",), {"output": "x"}),
    "kernelgenbench_fill_": (("x",), {"output": "x"}),
    "kernelgenbench_index_put_": (("x",), {"output": "x"}),
    "kernelgenbench_masked_fill_": (("x",), {"output": "x"}),
    "kernelgenbench_rrelu_with_noise": (("noise",), {}),
    "kernelgenbench_zero_": (("x",), {"output": "x"}),
}


_VLLM_EFFECTS: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {
    "vllm_apply_repetition_penalties_cuda": (
        ("logits",),
        {"logits_out": "logits"},
    ),
    "vllm_batched_moe_align_block_size": (
        ("sorted_ids", "expert_ids", "num_tokens_post_pad"),
        {
            "sorted_ids_out": "sorted_ids",
            "expert_ids_out": "expert_ids",
            "num_tokens_post_pad_out": "num_tokens_post_pad",
        },
    ),
    "vllm_concat_and_cache_mla": (("kv_cache",), {"kv_cache_out": "kv_cache"}),
    "vllm_convert_fp8": (("output",), {"output_out": "output"}),
    "vllm_copy_blocks": (
        ("key_caches", "value_caches"),
        {"key_caches": "key_caches", "value_caches": "value_caches"},
    ),
    "vllm_copy_blocks_mla": (("kv_caches",), {"kv_caches": "kv_caches"}),
    "vllm_cp_gather_cache": (("dst",), {"dst_out": "dst"}),
    "vllm_fused_add_rms_norm": (
        ("input", "residual"),
        {"output": "input"},
    ),
    "vllm_fused_qk_norm_rope": (("qkv",), {"qkv": "qkv"}),
    "vllm_gptq_marlin_24_gemm": (("workspace",), {}),
    "vllm_gptq_marlin_gemm": (("workspace",), {}),
    "vllm_moe_align_block_size": (
        ("sorted_ids", "expert_ids", "num_tokens_post_pad"),
        {
            "sorted_ids": "sorted_ids",
            "expert_ids": "expert_ids",
            "num_tokens_post_pad": "num_tokens_post_pad",
        },
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_json_lines(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        yield line_number, value


def _functions(source: str, name: str) -> dict[str, ast.FunctionDef]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"{name}: invalid reference Python: {exc}") from exc
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _rename_function(
    lines: list[str], node: ast.FunctionDef, old_name: str, new_name: str
) -> None:
    index = node.lineno - 1
    pattern = re.compile(rf"^(\s*def\s+){re.escape(old_name)}(\s*\()")
    lines[index], count = pattern.subn(
        rf"\g<1>{new_name}\g<2>", lines[index], count=1
    )
    if count != 1:
        raise ValueError(f"cannot rename {old_name}() at line {node.lineno}")


def _literal_default(node: ast.expr, *, definition: str, parameter: str) -> Any:
    try:
        value = ast.literal_eval(node)
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{definition}: default for {parameter!r} is not JSON-literal"
        ) from exc
    return value


def _parameters(
    old: dict[str, Any], run: ast.FunctionDef, type_hints: dict[str, str | None]
) -> list[dict[str, Any]]:
    arguments = run.args
    if arguments.kwarg is not None:
        raise ValueError(f"{old['name']}: **kwargs ABI is unsupported")

    positional = arguments.posonlyargs + arguments.args
    positional_defaults = [None] * (len(positional) - len(arguments.defaults)) + list(
        arguments.defaults
    )
    parameters: list[dict[str, Any]] = []
    for index, argument in enumerate(positional):
        default = positional_defaults[index]
        parameter: dict[str, Any] = {
            "name": argument.arg,
            "kind": (
                "positional_only"
                if index < len(arguments.posonlyargs)
                else "positional_or_keyword"
            ),
            "required": default is None,
        }
        if default is not None:
            parameter["default"] = _literal_default(
                default, definition=old["name"], parameter=argument.arg
            )
        if type_hints.get(argument.arg):
            parameter["type_hint"] = type_hints[argument.arg]
        parameters.append(parameter)

    if arguments.vararg is not None:
        parameter = {
            "name": arguments.vararg.arg,
            "kind": "var_positional",
            "required": True,
        }
        if type_hints.get(arguments.vararg.arg):
            parameter["type_hint"] = type_hints[arguments.vararg.arg]
        parameters.append(parameter)

    for argument, default in zip(
        arguments.kwonlyargs, arguments.kw_defaults, strict=True
    ):
        parameter = {
            "name": argument.arg,
            "kind": "keyword_only",
            "required": default is None,
        }
        if default is not None:
            parameter["default"] = _literal_default(
                default, definition=old["name"], parameter=argument.arg
            )
        if type_hints.get(argument.arg):
            parameter["type_hint"] = type_hints[argument.arg]
        parameters.append(parameter)

    expected = list(old.get("inputs") or [])
    actual = [parameter["name"] for parameter in parameters]
    if expected != actual:
        raise ValueError(
            f"{old['name']}: reference ABI differs from inputs: "
            f"expected={expected}, actual={actual}"
        )
    return parameters


def _value_kind(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _type_hints(
    inputs: list[str], workloads: list[dict[str, Any]]
) -> dict[str, str | None]:
    observed: dict[str, set[str]] = defaultdict(set)
    for workload in workloads:
        for name, spec in workload["inputs"].items():
            if not isinstance(spec, dict):
                observed[name].add(_value_kind(spec))
            elif spec.get("type") in {"random", "custom"}:
                observed[name].add("Tensor")
            elif spec.get("type") in {"scalar", "literal"}:
                observed[name].add(_value_kind(spec.get("value")))
            else:
                raise ValueError(
                    f"{workload.get('name')}: unsupported recipe for {name!r}"
                )
    order = ["Tensor", "bool", "int", "float", "str", "list", "dict", "None"]
    return {
        name: (
            " | ".join(sorted(observed[name], key=lambda value: order.index(value)))
            if observed[name]
            else None
        )
        for name in inputs
    }


def _cublas_effects(
    old: dict[str, Any], run: ast.FunctionDef
) -> tuple[tuple[str, ...], dict[str, str]]:
    returned = {
        node.value.id
        for node in ast.walk(run)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
    }
    inputs = set(old["inputs"])
    mutated = tuple(name for name in old["inputs"] if name in returned & inputs)
    aliases = (
        {old["outputs"][0]: mutated[0]}
        if len(mutated) == 1 and len(old["outputs"]) == 1
        else {}
    )
    if old["name"] == "cublas_cublasZswap_v2":
        mutated = ("x", "y")
    return mutated, aliases


def _effects(
    old: dict[str, Any], group: str, run: ast.FunctionDef
) -> dict[str, Any]:
    if old["name"] in _POINTWISE_EFFECTS:
        mutates, aliases = _POINTWISE_EFFECTS[old["name"]]
    elif old["name"] in _VLLM_EFFECTS:
        mutates, aliases = _VLLM_EFFECTS[old["name"]]
    elif group == "cublas":
        mutates, aliases = _cublas_effects(old, run)
    else:
        mutates, aliases = (), {}
    unknown = (set(mutates) | set(aliases.values())) - set(old["inputs"])
    unknown_outputs = set(aliases) - set(old["outputs"])
    if unknown or unknown_outputs:
        raise ValueError(
            f"{old['name']}: invalid migrated effects: "
            f"parameters={sorted(unknown)}, outputs={sorted(unknown_outputs)}"
        )
    return {"mutates": list(mutates), "returns_alias_of": aliases}


def _generic_custom_inputs(
    old: dict[str, Any], workloads: list[dict[str, Any]]
) -> str:
    custom_specs = [
        (name, spec)
        for workload in workloads
        for name, spec in workload["inputs"].items()
        if isinstance(spec, dict) and spec.get("type") == "custom"
    ]
    if not custom_specs:
        return ""
    supported = {"type", "shape", "dtype", "stride", "storage_offset"}
    unsupported = [
        (name, sorted(set(spec).difference(supported)))
        for name, spec in custom_specs
        if "stride" not in spec or set(spec).difference(supported)
    ]
    if unsupported:
        raise ValueError(
            f"{old['name']}: custom recipes require an explicit gen_inputs hook: "
            f"{unsupported[:3]}"
        )
    return (
        "\n\ndef gen_inputs(ctx, device):\n"
        "    result = {}\n"
        "    for name, spec in ctx[\"inputs\"].items():\n"
        "        if not isinstance(spec, dict) or spec.get(\"type\") != \"custom\":\n"
        "            continue\n"
        "        shape = tuple(spec[\"shape\"])\n"
        "        stride = tuple(spec[\"stride\"])\n"
        "        storage_offset = spec.get(\"storage_offset\", 0)\n"
        "        storage_size = storage_offset\n"
        "        if all(dimension > 0 for dimension in shape):\n"
        "            storage_size += 1 + sum(\n"
        "                (dimension - 1) * step\n"
        "                for dimension, step in zip(shape, stride)\n"
        "            )\n"
        "        dtype = getattr(torch, spec[\"dtype\"])\n"
        "        base = torch.empty((storage_size,), dtype=dtype, device=device)\n"
        "        if dtype.is_floating_point or dtype.is_complex:\n"
        "            base.normal_()\n"
        "        elif dtype is torch.bool:\n"
        "            base.random_(0, 2)\n"
        "        elif dtype is torch.uint8:\n"
        "            base.random_(0, 256)\n"
        "        else:\n"
        "            base.random_(-1024, 1024)\n"
        "        result[name] = torch.as_strided(\n"
        "            base, shape, stride, storage_offset\n"
        "        )\n"
        "    return result\n"
    )


def _oracle(old: dict[str, Any], workloads: list[dict[str, Any]]) -> str:
    name = old["name"]
    source = str(old["reference"]).rstrip() + "\n"
    functions = _functions(source, name)
    if "run" not in functions:
        raise ValueError(f"{name}: reference has no run()")
    lines = source.splitlines(keepends=True)
    wrappers: list[str] = []

    custom_inputs = old.get("custom_inputs_entrypoint")
    if custom_inputs is not None:
        if custom_inputs != "gen_inputs" or custom_inputs not in functions:
            raise ValueError(f"{name}: unsupported custom input entrypoint")
        _rename_function(
            lines, functions[custom_inputs], custom_inputs, "_legacy_gen_inputs"
        )
        wrappers.append(
            "\n\ndef gen_inputs(ctx, device):\n"
            "    return _legacy_gen_inputs(_legacy_context(ctx), device)\n"
        )
    else:
        wrappers.append(_generic_custom_inputs(old, workloads))

    custom_valid = old.get("custom_valid_entrypoint")
    if custom_valid is not None:
        if custom_valid != "valid" or custom_valid not in functions:
            raise ValueError(f"{name}: unsupported custom valid entrypoint")
        _rename_function(lines, functions[custom_valid], custom_valid, "_legacy_valid")
        wrappers.append(
            "\n\nVALID_OWNS_RETURN_CONTRACT = True\n\n"
            "def valid(ref_outputs, sol_outputs, inputs, ctx):\n"
            f"    ordered = [inputs[name] for name in {old['inputs']!r}]\n"
            "    return _legacy_valid(\n"
            "        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)\n"
            "    )\n"
        )

    correctness = old.get("correctness_reference_entrypoint")
    if correctness is not None:
        if correctness not in functions:
            raise ValueError(f"{name}: missing correctness entrypoint {correctness!r}")
        wrappers.append(f"\n\ncorrectness_run = {correctness}\n")

    context_wrapper = ""
    if custom_inputs is not None or custom_valid is not None:
        context_wrapper = (
            "\n\ndef _legacy_context(ctx):\n"
            "    result = {}\n"
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

    oracle = (
        f'REFERENCE_DEVICE = {old.get("reference_device", "target")!r}\n\n'
        + "".join(lines)
        + context_wrapper
        + "".join(wrappers)
    )
    ast.parse(oracle, filename=f"{name}/oracle.py")
    return oracle.rstrip() + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_workload_file(
    path: Path,
    inputs: list[str],
    seen_names: set[str],
) -> list[dict[str, Any]]:
    workloads: list[dict[str, Any]] = []
    for line_number, raw in _read_json_lines(path):
        workload = Workload.model_validate(raw)
        if set(workload.inputs) != set(inputs):
            raise ValueError(
                f"{path}:{line_number}: workload inputs differ from Definition"
            )
        if workload.name in seen_names:
            raise ValueError(f"globally duplicate workload name: {workload.name}")
        seen_names.add(workload.name)
        workloads.append(raw)
    if not workloads:
        raise ValueError(f"empty workload file: {path}")
    return workloads


def _validate_catalog(root: Path, expected: set[str]) -> None:
    catalog = Catalog(root)
    if set(catalog.operator_names) != expected:
        raise ValueError("generated Catalog operator names do not match source")
    for name in catalog.operator_names:
        operator = catalog.load(name)
        if operator.definition.reference is None:
            raise ValueError(f"{name}: generated oracle was not attached")


def convert(
    source_root: Path,
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    definitions_root = source_root / "definitions"
    workloads_root = source_root / "workloads"
    source_manifest = _read_json(source_root / "manifest.json")
    if source_manifest.get("api_version") != SOURCE_API_VERSION:
        raise ValueError(f"source catalog must use {SOURCE_API_VERSION}")
    if not definitions_root.is_dir() or not workloads_root.is_dir():
        raise ValueError(f"not a legacy KernelGenBench catalog: {source_root}")
    if output_root.exists() and not force:
        raise FileExistsError(f"output exists (pass --force): {output_root}")

    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=parent))
    seen_names: set[str] = set()
    seen_workload_names: set[str] = set()
    operators: list[dict[str, Any]] = []
    totals = {"correctness": 0, "timing": 0}
    try:
        for definition_path in sorted(definitions_root.rglob("*.json")):
            old = _read_json(definition_path)
            name = str(old.get("name") or "")
            if name in seen_names:
                raise ValueError(f"duplicate definition: {name}")
            seen_names.add(name)
            if old.get("api_version") != SOURCE_API_VERSION:
                raise ValueError(f"{name}: expected {SOURCE_API_VERSION}")
            group = definition_path.parent.relative_to(definitions_root).as_posix()
            operator_root = temporary / "ops" / group / name

            phase_workloads: dict[str, list[dict[str, Any]]] = {}
            for phase in ("correctness", "timing"):
                source_workloads = (
                    workloads_root / group / f"{name}.{phase}.jsonl"
                )
                workloads = _validate_workload_file(
                    source_workloads,
                    list(old["inputs"]),
                    seen_workload_names,
                )
                phase_workloads[phase] = workloads
                totals[phase] += len(workloads)
                destination = operator_root / f"{phase}.jsonl"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_workloads, destination)

            source = str(old["reference"])
            functions = _functions(source, name)
            combined_workloads = (
                phase_workloads["correctness"] + phase_workloads["timing"]
            )
            definition = Definition.model_validate(
                {
                    "api_version": TARGET_API_VERSION,
                    "name": name,
                    "description": old.get("description", ""),
                    "parameters": _parameters(
                        old,
                        functions["run"],
                        _type_hints(list(old["inputs"]), combined_workloads),
                    ),
                    "outputs": list(old["outputs"]),
                    "effects": _effects(old, group, functions["run"]),
                }
            )
            _write_json(
                operator_root / "definition.json",
                definition.model_dump(
                    mode="json", exclude_none=True, exclude_unset=True
                ),
            )
            (operator_root / "oracle.py").write_text(
                _oracle(old, combined_workloads), encoding="utf-8"
            )
            operators.append(
                {
                    "name": name,
                    "group": group,
                    "num_correctness_workloads": len(
                        phase_workloads["correctness"]
                    ),
                    "num_timing_workloads": len(phase_workloads["timing"]),
                }
            )

        counts = {
            "operators": len(operators),
            "correctness_workloads": totals["correctness"],
            "timing_workloads": totals["timing"],
        }
        if counts != EXPECTED_FULL_COUNTS:
            raise ValueError(
                "KernelGenBench source counts changed: "
                f"{counts} != {EXPECTED_FULL_COUNTS}"
            )
        source_counts = source_manifest.get("counts")
        if source_counts != counts:
            raise ValueError(
                f"legacy manifest counts do not match files: {source_counts} != {counts}"
            )

        manifest = {
            "api_version": TARGET_API_VERSION,
            "name": CATALOG_NAME,
            "evaluator": "native",
            "layout": "per-operator",
            "source_format": source_manifest.get("source_format"),
            "source_api_version": SOURCE_API_VERSION,
            "generated_by": "tools/convert_kernelgenbench_v5.py",
            "counts": counts,
            "operators": operators,
        }
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "README.md").write_text(
            "# KernelGenBench v6.2 native catalog\n\n"
            "This catalog is deterministically converted from the checked-in "
            "KernelGenBench v5.1 210-operator catalog by "
            "`tools/convert_kernelgenbench_v5.py`. It retains the complete "
            "57,603 correctness and 28,198 timing workloads as "
            "`correctness_full.jsonl` and `timing_full.jsonl`, and selects at "
            f"most {WORKLOAD_LIMIT} active workloads per operator and phase "
            f"with `{ALGORITHM}`. It moves each embedded "
            "reference into `ops/<group>/<name>/oracle.py`, preserves the "
            "original `pointwise`, `vllm`, and `cublas` groups, and exposes "
            "a pure v6.2 public ABI in `definition.json`.\n\n"
            "The original v5.1 source remains recoverable from Git commit "
            "`4a5769a` and its descendants before this migration. Regeneration "
            "must use a clean checkout of that source catalog and run the tool "
            "with `PYTHONPATH=.` from the KernelGen Server repository root.\n",
            encoding="utf-8",
        )
        manifest = sample_catalog(temporary, limit=WORKLOAD_LIMIT)
        if manifest["counts"] != EXPECTED_COUNTS:
            raise ValueError(
                "KernelGenBench sampled counts changed: "
                f"{manifest['counts']} != {EXPECTED_COUNTS}"
            )
        if manifest["full_counts"] != EXPECTED_FULL_COUNTS:
            raise ValueError(
                "KernelGenBench full counts changed after sampling: "
                f"{manifest['full_counts']} != {EXPECTED_FULL_COUNTS}"
            )
        _validate_catalog(temporary, seen_names)

        backup: Path | None = None
        if output_root.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output_root.name}-backup-", dir=parent)
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
    parser = argparse.ArgumentParser(
        description="Convert KernelGenBench v5.1 Catalog data to native v6.2."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/kernelgenbench"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = convert(args.source, args.output, force=args.force)
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()

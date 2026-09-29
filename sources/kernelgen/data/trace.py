"""Trace-set loading, validation, and definition-style inference."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from kernelgen.framework.models import DefinitionModel


@dataclass(frozen=True)
class WorkloadPaths:
    """Resolved workload files using the same legacy/phased precedence as TraceSet."""

    mode: Literal["legacy", "phased"]
    legacy: tuple[Path, ...] = ()
    correctness: tuple[Path, ...] = ()
    timing: tuple[Path, ...] = ()

    @property
    def all(self) -> tuple[Path, ...]:
        if self.mode == "phased":
            return self.correctness + self.timing
        return self.legacy


def load_workload_context(
    paths: WorkloadPaths,
    definition_name: str,
) -> list[dict[str, Any]]:
    """Load the workload fields that are relevant to kernel implementation.

    The Coder needs concrete axis, scalar, and literal values to implement
    host-side specialization correctly. Evaluation metadata and file paths are
    intentionally omitted from the prompt-facing representation.
    """
    if paths.mode == "phased":
        phase_files = (
            ("correctness", paths.correctness),
            ("timing", paths.timing),
        )
    else:
        phase_files = (("legacy", paths.legacy),)

    context: list[dict[str, Any]] = []
    for phase, files in phase_files:
        for path in files:
            for line_num, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid workload JSON at {path}:{line_num}: {exc}"
                    ) from exc
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"workload entry must be an object at {path}:{line_num}"
                    )

                declared_definition = entry.get("definition")
                if (
                    declared_definition is not None
                    and declared_definition != definition_name
                ):
                    raise ValueError(
                        f"workload at {path}:{line_num} declares definition "
                        f"{declared_definition!r}, expected {definition_name!r}"
                    )

                workload = entry.get("workload", entry)
                if not isinstance(workload, dict):
                    raise ValueError(
                        f"workload payload must be an object at {path}:{line_num}"
                    )
                item = {
                    "phase": phase,
                    "uuid": workload.get("uuid", ""),
                    "axes": workload.get("axes", {}),
                    "inputs": workload.get("inputs", {}),
                }
                for field in ("tolerance", "seed"):
                    if field in workload:
                        item[field] = workload[field]
                context.append(item)
    return context


def trace_base(trace_root: str | Path, trace_set_key: str = "") -> Path:
    """Resolve the directory containing a trace set."""
    root = Path(trace_root).expanduser()
    return root / trace_set_key if trace_set_key else root


def is_kernelgen_server_catalog(
    trace_root: str | Path,
    trace_set_key: str = "",
) -> bool:
    """Return whether the selected root is a schema-v1 server catalog."""
    return (trace_base(trace_root, trace_set_key) / "manifest.json").is_file()


def catalog_operator_optimization_context(
    definition: Any,
    correctness_workloads: list[Any],
    timing_workloads: list[Any],
    *,
    group: str = "",
) -> tuple[DefinitionModel, list[dict[str, Any]]]:
    """Build the prompt view from one already-resolved native contract."""

    from kernelgen_client.protocol.workload_call import definition_signature

    serialized_workloads = []
    for phase, items in (
        ("correctness", correctness_workloads),
        ("timing", timing_workloads),
    ):
        for workload in items:
            serialized_workloads.append(
                (
                    phase,
                    workload.name,
                    workload.model_dump(mode="json", exclude_none=True),
                )
            )

    input_dtypes: dict[str, str] = {}
    for parameter in definition.parameters:
        input_name = parameter.name
        observed = sorted(
            {
                str(spec["dtype"])
                for _, _, payload in serialized_workloads
                for candidate_name, spec in payload["inputs"].items()
                if candidate_name == input_name
                and isinstance(spec, dict)
                and spec.get("dtype")
            }
        )
        input_dtypes[input_name] = observed[0] if len(observed) == 1 else (
            parameter.type_hint or "dynamic"
        )
    prompt_definition = DefinitionModel(
        name=definition.name,
        op_type=group,
        description=definition.description,
        inputs={
            input_name: {
                "shape": "dynamic",
                "dtype": input_dtypes[input_name],
            }
            for input_name in input_dtypes
        },
        outputs={
            output_name: {"shape": "dynamic", "dtype": "dynamic"}
            for output_name in definition.outputs
        },
        run_signature=str(definition_signature(definition)),
        reference=definition.reference or "",
        mutation=(
            {"inputs": list(definition.effects.mutates)}
            if definition.effects.mutates
            else None
        ),
    )
    workloads = []
    for phase, workload_name, payload in serialized_workloads:
        workloads.append(
            {
                "phase": phase,
                "uuid": workload_name,
                "axes": {},
                "inputs": payload["inputs"],
                "seed": payload["seed"],
                **(
                    {"tolerance": payload["tolerance"]}
                    if "tolerance" in payload
                    else {}
                ),
            }
        )
    return prompt_definition, workloads


def load_catalog_optimization_context(
    trace_root: str | Path,
    name: str,
    trace_set_key: str = "",
) -> tuple[DefinitionModel, list[dict[str, Any]]]:
    """Load prompt-facing data from a KernelGen Server catalog."""
    from kernelgen_client import Catalog
    catalog = Catalog(trace_base(trace_root, trace_set_key))
    operator = catalog.load(name)
    op_type = ""
    for entry in catalog.manifest.get("operators", []):
        if (
            isinstance(entry, dict)
            and entry.get("name") == name
            and isinstance(entry.get("group"), str)
        ):
            op_type = entry["group"]
            break
    if not op_type and "/" in operator.relative:
        op_type = operator.relative.rsplit("/", 1)[0]
    return catalog_operator_optimization_context(
        operator.definition,
        operator.correctness_workloads,
        operator.timing_workloads,
        group=op_type,
    )


def find_definition_path(
    trace_root: str | Path,
    name: str,
    trace_set_key: str = "",
) -> Path:
    """Find exactly one definition JSON by its declared name."""
    definitions_dir = trace_base(trace_root, trace_set_key) / "definitions"
    if not definitions_dir.is_dir():
        raise FileNotFoundError(f"definitions directory not found: {definitions_dir}")

    matches = sorted(
        path for path in definitions_dir.rglob("*.json")
        if path.stem == name
    )
    if not matches:
        raise FileNotFoundError(
            f"definition {name!r} not found under {definitions_dir}"
        )
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise ValueError(f"definition {name!r} is ambiguous: {rendered}")
    return matches[0]


def load_definition(
    trace_root: str | Path,
    name: str,
    trace_set_key: str = "",
) -> tuple[DefinitionModel, Path]:
    """Load and validate a definition by name."""
    path = find_definition_path(trace_root, name, trace_set_key)
    data = json.loads(path.read_text(encoding="utf-8"))
    definition = DefinitionModel.model_validate(data)
    if definition.name != name:
        raise ValueError(
            f"definition file {path} declares name {definition.name!r}, expected {name!r}"
        )
    return definition, path


def find_workloads_path(
    trace_root: str | Path,
    name: str,
    trace_set_key: str = "",
) -> Path:
    """Find a non-empty workloads JSONL matching a definition name."""
    workloads_dir = trace_base(trace_root, trace_set_key) / "workloads"
    if not workloads_dir.is_dir():
        raise FileNotFoundError(f"workloads directory not found: {workloads_dir}")

    matches = sorted(
        path for path in workloads_dir.rglob("*.jsonl")
        if path.stem == name
    )
    if not matches:
        raise FileNotFoundError(f"workloads for {name!r} not found under {workloads_dir}")
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise ValueError(f"workloads for {name!r} are ambiguous: {rendered}")

    path = matches[0]
    if not any(line.strip() for line in path.read_text(encoding="utf-8").splitlines()):
        raise ValueError(f"workloads file is empty: {path}")
    return path


def find_workload_paths(
    trace_root: str | Path,
    name: str,
    trace_set_key: str = "",
) -> WorkloadPaths:
    """Resolve non-empty legacy or phased workload files for one definition.

    Phase-specific files take precedence when either phase contains workloads.
    Both the current ``phased_workloads`` layout and the deprecated split
    directories accepted by flashinfer-bench are recognized.
    """
    base = trace_base(trace_root, trace_set_key)

    def nonempty(paths) -> tuple[Path, ...]:
        return tuple(
            path
            for path in sorted(paths)
            if any(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
            )
        )

    phased_dir = base / "phased_workloads"
    correctness = nonempty(
        list(phased_dir.rglob(f"{name}.correctness.jsonl"))
        if phased_dir.is_dir()
        else []
    )
    timing = nonempty(
        list(phased_dir.rglob(f"{name}.timing.jsonl"))
        if phased_dir.is_dir()
        else []
    )

    deprecated_correctness = base / "correctness_workloads"
    if deprecated_correctness.is_dir():
        correctness += nonempty(
            path
            for path in deprecated_correctness.rglob("*.jsonl")
            if path.stem == name
        )
    deprecated_timing = base / "timing_workloads"
    if deprecated_timing.is_dir():
        timing += nonempty(
            path
            for path in deprecated_timing.rglob("*.jsonl")
            if path.stem == name
        )

    if correctness or timing:
        return WorkloadPaths(
            mode="phased",
            correctness=correctness,
            timing=timing,
        )

    workloads_dir = base / "workloads"
    legacy = nonempty(
        (
            path
            for path in workloads_dir.rglob("*.jsonl")
            if path.stem == name
        )
        if workloads_dir.is_dir()
        else []
    )
    if legacy:
        return WorkloadPaths(mode="legacy", legacy=legacy)

    raise FileNotFoundError(
        f"no non-empty legacy or phased workloads for {name!r} under {base}"
    )


def infer_destination_passing_style(definition: DefinitionModel) -> bool:
    """Infer DPS from the reference ``run`` signature.

    Unified-trace definitions use value-returning ``run`` functions, while
    extractor-generated definitions may append preallocated outputs. Parameter
    count distinguishes these formats without depending on output variable names.
    """
    try:
        tree = ast.parse(definition.reference)
    except (SyntaxError, TypeError):
        tree = None

    if tree is not None:
        run_functions = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ]
        if run_functions:
            args = run_functions[-1].args
            if args.vararg is None:
                positional_count = len(args.posonlyargs) + len(args.args)
                input_count = len(definition.inputs)
                dps_count = input_count + len(definition.outputs)
                if positional_count == dps_count and dps_count != input_count:
                    return True
                if positional_count == input_count:
                    return False

    # Conservative fallback for older extractor output.
    reference = definition.reference or ""
    output_names = definition.outputs.keys()
    return any(f"{name}.copy_" in reference for name in output_names)

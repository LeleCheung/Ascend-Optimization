"""Preserve local Catalog preparation for historical flat-workspace campaigns."""

from pathlib import Path
from typing import Any

from kernelgen.data.catalog import resolve_builtin_catalog_path
from kernelgen.data.evaluation_snapshot import build_catalog_evaluation_snapshot, snapshot_optimization_context
from kernelgen.data.trace import is_kernelgen_server_catalog, load_catalog_optimization_context
from kernelgen.workflows.optimization.single_coder.inputs import build_optimizer_input
from kernelgen.workflows.optimization.single_coder.reference import load_reference_code
from kernelgen.workflows.optimization.options import SimpleOptInput
from kernelgen.workflows.optimization.knowledge import materialize_knowledge

def prepare_optimization(inp: SimpleOptInput, workspace: Path) -> dict[str, Any]:
    reference = load_reference_code(inp.reference_code_path, inp.reference_code_prompt_path)
    catalog_path = resolve_builtin_catalog_path(inp.catalog_name)
    evaluation_snapshot = None
    if (
        inp.catalog_name == "kernelswift"
        and is_kernelgen_server_catalog(catalog_path)
    ):
        native_snapshot = build_catalog_evaluation_snapshot(
            inp.catalog_name,
            inp.definition_name,
            catalog_path=catalog_path,
        )
        definition, workloads = snapshot_optimization_context(
            native_snapshot
        )
        evaluation_snapshot = native_snapshot.model_dump(mode="json")
    else:
        definition, workloads = load_catalog_optimization_context(
            catalog_path,
            inp.definition_name,
        )
    inferred_dps = False
    dps = (
        inp.destination_passing_style
        if inp.destination_passing_style is not None
        else inferred_dps
    )

    workspace.mkdir(parents=True, exist_ok=True)
    knowledge_enabled = inp.knowledge_catalog_path is not None
    if knowledge_enabled:
        materialize_knowledge(inp, definition, workspace)
    return build_optimizer_input(
        inp,
        definition=definition.model_dump(exclude_none=True),
        destination_passing_style=dps,
        analysis={},
        workloads=workloads,
        evaluation_snapshot=evaluation_snapshot,
        **reference,
        knowledge_enabled=knowledge_enabled,
    )

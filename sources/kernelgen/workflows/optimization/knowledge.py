"""Optional Knowledge preparation and resume identity checks for SimpleOpt."""

from pathlib import Path
from typing import Optional

from kernelgen.data.catalog import catalog_benchmark_id
from kernelgen.knowledge.config import KnowledgeConfig
from kernelgen.knowledge.contracts import WorkspaceKnowledgeState
from kernelgen.knowledge.layout import KnowledgeLayout
from kernelgen.workflows.knowledge_bridge import KernelGenKnowledgeBridge
from kernelgen.workflows.optimization.options import SimpleOptInput


def materialize_knowledge(inp: SimpleOptInput, definition, workspace: Path, *, benchmark_id: str | None = None, run_id: str | None = None) -> None:
    catalog_path = _resolve_knowledge_catalog(
        inp.knowledge_catalog_path
    )
    knowledge = KernelGenKnowledgeBridge(
        config=KnowledgeConfig(catalog_root=catalog_path),
        definition=definition,
        target_hardware=inp.target_hardware,
        implementation_language=inp.implementation_language.value,
        eval_server_url=inp.eval_server_url,
        run_id=run_id or workspace.name,
        benchmark_id=benchmark_id or catalog_benchmark_id(inp.catalog_name),
    )
    _validate_existing_knowledge_state(knowledge, catalog_path, workspace)
    knowledge.materializer().materialize(workspace)


def _resolve_knowledge_catalog(path: Optional[Path]) -> Path:
    if path is None:
        raise ValueError(
            "knowledge_catalog_path is required for "
            "Knowledge-enabled SimpleOpt"
        )
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"knowledge_catalog_path does not exist: {path}"
        ) from exc
    if not resolved.is_dir():
        raise ValueError(
            "knowledge_catalog_path must be a directory: "
            f"{path}"
        )
    return resolved


def _validate_existing_knowledge_state(
    knowledge: KernelGenKnowledgeBridge,
    catalog_path: Path,
    workspace: Path,
) -> None:
    layout = KnowledgeLayout(workspace)
    if not layout.state.is_file():
        return
    try:
        state = WorkspaceKnowledgeState.model_validate_json(
            layout.state.read_text(encoding="utf-8")
        )
        existing_catalog = Path(state.catalog_ref)
        if not existing_catalog.is_absolute():
            existing_catalog = (workspace / existing_catalog).resolve()
        else:
            existing_catalog = existing_catalog.resolve()
        existing_signature = type(
            knowledge.operator_signature
        ).model_validate_json(
            layout.operator_signature.read_text(encoding="utf-8"),
        )
        existing_target = type(
            knowledge.target_context
        ).model_validate_json(
            layout.target_context.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "workspace contains invalid existing Knowledge state"
        ) from exc
    if (
        existing_catalog != catalog_path
        or existing_signature != knowledge.operator_signature
        or existing_target != knowledge.target_context
    ):
        raise ValueError(
            "workspace Knowledge state does not match the requested "
            "Catalog, Definition, or target; use a fresh workspace"
        )

"""Knowledge publication and legacy epoch-KB helpers."""

from __future__ import annotations

import os
from pathlib import Path

from kernelgen.data.catalog import catalog_benchmark_id
from kernelgen.framework.parallel import Workspace
from kernelgen.knowledge.config import KnowledgeConfig
from kernelgen.workflows.optimization.kernelgen.contracts import EpochResult, KernelGenInput
from kernelgen.workflows.knowledge_bridge import KernelGenKnowledgeBridge


def build_knowledge_bridge(
    *,
    config: KnowledgeConfig | None,
    cwd: Path,
    inp: KernelGenInput,
    start_mode: str,
) -> KernelGenKnowledgeBridge | None:
    if not inp.cross_epoch_knowledge and config is not None and config.writes_v1:
        raise ValueError("cross_epoch_knowledge=false requires read_only_v1 or no KnowledgeConfig")
    if config is None:
        return None
    return KernelGenKnowledgeBridge(
        config=config,
        definition=inp.definition,
        target_hardware=inp.target_hardware,
        implementation_language=inp.implementation_language.value,
        eval_server_url=inp.eval_server_url,
        run_id=inp.knowledge_run_id or cwd.name,
        benchmark_id=(f"{inp.catalog_name}-{inp.evaluation_snapshot['catalog_api_version']}"
                      if inp.evaluation_snapshot is not None else catalog_benchmark_id(inp.catalog_name)),
        start_mode=start_mode,
    )


def promote_best_solution(
    knowledge: KernelGenKnowledgeBridge | None,
    best_result: EpochResult,
) -> None:
    if knowledge is None:
        return
    if best_result.best_workspace_path is None:
        return
    manifest = knowledge.promote_solution(best_result.best_workspace_path)
    if manifest is not None:
        print(
            "[Solution] Promoted "
            f"{manifest.solution_ref} ({manifest.geo_mean:.4f}x)"
        )


def merge_epoch_kb(
    *,
    cwd: Path,
    inp: KernelGenInput,
    results,
    runtime,
    epoch_num: int,
    epoch_workspace: Workspace | None,
) -> None:
    """Reduce scored agent candidates into one bounded legacy KB entry."""
    from kernelgen.workflows.knowledge_reducer import (
        collect_knowledge_candidates,
        reduce_epoch_knowledge,
    )

    base_kb = cwd / "kb"
    if not base_kb.exists() or epoch_workspace is None:
        return

    candidates = collect_knowledge_candidates(
        results,
        epoch_workspace,
        definition_name=inp.definition.name,
        target_hardware=inp.target_hardware,
    )
    written = reduce_epoch_knowledge(
        base_kb,
        definition=inp.definition.model_dump(mode="json"),
        target_hardware=inp.target_hardware,
        candidates=candidates,
        runtime=runtime,
    )
    if written:
        git_commit_kb(
            base_kb,
            f"epoch {epoch_num}: reduced {len(candidates)} agent candidates",
        )


def git_commit_kb(kb_path: Path, message: str) -> None:
    """Commit all changes in the legacy KB repository."""
    import subprocess

    if not (kb_path / ".git").is_dir():
        return
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(kb_path),
            capture_output=True,
            text=True,
        )
        if (
            root.returncode != 0
            or Path(root.stdout.strip()).resolve() != kb_path.resolve()
        ):
            return
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(kb_path),
            capture_output=True,
            check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(kb_path),
            capture_output=True,
            text=True,
            check=True,
        )
        if status.stdout.strip():
            env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "kb-merge",
                "GIT_AUTHOR_EMAIL": "kb@merge",
                "GIT_COMMITTER_NAME": "kb-merge",
                "GIT_COMMITTER_EMAIL": "kb@merge",
            }
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(kb_path),
                capture_output=True,
                env=env,
                check=True,
            )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


__all__ = [
    "build_knowledge_bridge",
    "git_commit_kb",
    "merge_epoch_kb",
    "promote_best_solution",
]

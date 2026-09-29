"""run_parallel() — parallel fan-out execution with resource pool management."""

import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .agent import AgentDef
from .codeagent import CodeAgentConfig
from .device import GPUPool
from .platform import PlatformInfo
from .progress import TaskTracker
from .run import run
from .workspace import Workspace

logger = logging.getLogger(__name__)

_shutdown_event = threading.Event()


def run_parallel(
    agent_or_callable,
    inputs: "list[dict] | str | Path",
    *,
    gpu: GPUPool | None = None,
    workspace: Workspace | None = None,
    workspace_name_key: str | None = None,
    cleanup_workspace: bool = False,
    backend: "CodeAgentBackend | None" = None,
    agent_config: CodeAgentConfig | None = None,
    platform: PlatformInfo | None = None,
    max_retries: int = 0,
    max_resumes: int = 0,
    timeout: float | None = None,
    tracker: TaskTracker | None = None,
    log_dir: Path | None = None,
    poll_interval: float = 5.0,
) -> list[dict]:
    # Parse inputs: support JSONL file path or list
    if isinstance(inputs, (str, Path)):
        import json
        parsed = []
        with open(inputs) as f:
            for line in f:
                line = line.strip()
                if line:
                    parsed.append(json.loads(line))
        inputs = parsed

    if not inputs:
        return []

    # Auto-resolve platform from pool if not explicitly provided
    if platform is None and gpu is not None:
        platform = gpu.platform

    # Auto-resolve workspace_name_key from agent resources
    if workspace_name_key is None and hasattr(agent_or_callable, "resources"):
        workspace_name_key = agent_or_callable.resources.workspace_name_key

    # Auto-resolve workspace from agent config if not provided
    if workspace is None and hasattr(agent_or_callable, "workspace_config"):
        ws_cfg = agent_or_callable.workspace_config
        if ws_cfg and ws_cfg.repo_dir:
            from .workspace import GitWorktree, TempDir
            if ws_cfg.type == "git_worktree":
                workspace = GitWorktree(
                    repo_dir=ws_cfg.repo_dir,
                    base_branch=ws_cfg.base_branch,
                    branch_prefix=ws_cfg.branch_prefix,
                )
            elif ws_cfg.type == "tempdir":
                workspace = TempDir()

    max_workers = gpu.size if gpu else len(inputs)
    results: list[dict | None] = [None] * len(inputs)
    errors: list[Exception | None] = [None] * len(inputs)

    # Signal handling for graceful shutdown
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_shutdown(signum, frame):
        logger.info("Shutdown signal received, stopping new tasks...")
        _shutdown_event.set()

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    def _execute_one(idx: int, input_data: dict) -> tuple[int, dict]:
        if _shutdown_event.is_set():
            raise RuntimeError("Shutdown requested")

        gpu_id = None
        ws_path = None
        if workspace_name_key and workspace_name_key in input_data:
            task_name = str(input_data[workspace_name_key])
        else:
            task_name = f"task_{idx}"

        try:
            if gpu:
                gpu_id = gpu.acquire(timeout=timeout)
            if workspace:
                ws_path = workspace.allocate(task_name)

            if tracker:
                tracker.mark_running(task_name, gpu_id)

            task_log_dir = Path(log_dir) / task_name if log_dir else None

            result = run(
                agent_or_callable,
                input_data,
                gpu=gpu_id,
                workspace=ws_path,
                backend=backend,
                agent_config=agent_config,
                platform=platform,
                max_retries=max_retries,
                max_resumes=max_resumes,
                timeout=timeout,
                tracker=tracker,
                log_dir=task_log_dir,
            )
            return idx, result
        finally:
            if gpu and gpu_id is not None:
                gpu.release(gpu_id)
            if workspace and cleanup_workspace:
                workspace.deallocate(task_name)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, input_data in enumerate(inputs):
                if _shutdown_event.is_set():
                    break
                future = executor.submit(_execute_one, idx, input_data)
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result_idx, result = future.result()
                    results[result_idx] = result
                except Exception as e:
                    errors[idx] = e
                    logger.error("Task %d failed: %s", idx, e)
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        _shutdown_event.clear()

    # Raise if any tasks failed
    failed = [(i, e) for i, e in enumerate(errors) if e is not None]
    if failed:
        msg = "; ".join(f"task[{i}]: {e}" for i, e in failed)
        logger.error("run_parallel: %d/%d tasks failed: %s", len(failed), len(inputs), msg)

    return [r if r is not None else {} for r in results]

"""run() — single agent/callable execution with retry/resume/timeout."""

import logging
import time
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Callable

from .agent import AgentDef
from .codeagent import CodeAgentBackend, CodeAgentConfig, ClaudeBackend, default_output_parser
from .platform import PlatformInfo
from .progress import TaskTracker
from .template import render_template

logger = logging.getLogger(__name__)


def run(
    agent_or_callable,
    input: dict,
    *,
    gpu: int | None = None,
    workspace: Path | None = None,
    backend: CodeAgentBackend | None = None,
    agent_config: CodeAgentConfig | None = None,
    platform: PlatformInfo | None = None,
    output_parser: Callable[[str], dict | None] | None = None,
    max_retries: int = 0,
    max_resumes: int = 0,
    timeout: float | None = None,
    tracker: TaskTracker | None = None,
    log_dir: Path | None = None,
) -> dict:
    if callable(agent_or_callable) and not isinstance(agent_or_callable, AgentDef):
        return agent_or_callable(input, gpu_id=gpu, workspace=workspace)

    agent: AgentDef = agent_or_callable
    agent.validate_input(input)

    # Resolve backend from agent config if not provided
    if backend is None:
        if agent_config is None:
            bc = agent.backend_config
            agent_config = CodeAgentConfig(
                bin=bc.bin,
                model=bc.model,
                base_url=bc.base_url,
                auth_token_env=bc.auth_token_env,
                max_output_tokens=bc.max_output_tokens,
                budget=bc.budget,
                extra_flags=bc.extra_flags,
            )
        backend = ClaudeBackend(agent_config, platform)

    # Resolve retry/resume/timeout from agent config if not explicitly set
    if max_retries == 0 and agent.backend_config.max_retries:
        max_retries = agent.backend_config.max_retries
    if max_resumes == 0 and agent.backend_config.max_resumes:
        max_resumes = agent.backend_config.max_resumes
    if timeout is None:
        timeout = agent.backend_config.timeout

    # Resolve output parser
    if output_parser is None:
        output_parser = default_output_parser

    if log_dir is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("./kgrunner_runs") / agent.name / timestamp
    log_dir = Path(log_dir)

    task_id = f"{agent.name}_{id(input)}"

    for retry in range(max_retries + 1):
        start_time = time.monotonic()
        log_prefix = f"{agent.name}_r{retry}"

        # Render prompt
        variables = {k: str(v) for k, v in input.items() if v is not None}
        prompt = render_template(agent.prompt_template, variables)

        # Spawn agent
        agent_proc = backend.spawn(
            prompt,
            cwd=workspace or Path.cwd(),
            device_id=gpu,
            log_dir=log_dir,
            log_prefix=log_prefix,
        )

        # Resume loop
        resume_count = 0
        while True:
            try:
                agent_proc.wait(timeout=timeout)
            except TimeoutExpired:
                backend.kill(agent_proc)
                logger.warning(
                    "%s: timeout after %.0fs (retry %d/%d)",
                    agent.name, timeout, retry, max_retries,
                )
                break

            agent_proc.close()

            # Layer 1: extract raw text from backend-specific format
            text = backend.extract_text(agent_proc)

            # Layer 2: parse structured output from text
            output = output_parser(text)

            # If parse succeeded and not failed, return
            if output and output.get("status") != "failed":
                duration = time.monotonic() - start_time
                validated = agent.validate_output(output)
                if tracker:
                    tracker.mark_success(task_id, duration, validated)
                return validated

            # Parse failed (None) — check if it's a resumable error
            if output is None and backend.supports_resume and resume_count < max_resumes:
                if backend.detect_resumable_error(agent_proc):
                    session_id = backend.extract_session_id(agent_proc)
                    if session_id:
                        resume_count += 1
                        logger.info(
                            "%s: resumable error, resuming (%d/%d)",
                            agent.name, resume_count, max_resumes,
                        )
                        if tracker:
                            tracker.mark_resuming(task_id, session_id)
                        agent_proc = backend.resume(
                            session_id,
                            cwd=workspace or Path.cwd(),
                            device_id=gpu,
                            log_dir=log_dir,
                            log_prefix=f"{log_prefix}_resume{resume_count}",
                        )
                        continue

            # Agent reported failure or unparseable output
            error_msg = (output or {}).get("error", "Agent returned failed status or unparseable output")
            logger.warning(
                "%s: failed (retry %d/%d): %s",
                agent.name, retry, max_retries, error_msg,
            )
            if tracker:
                tracker.mark_retrying(task_id, retry + 1, error_msg)
            break

    duration = time.monotonic() - start_time
    if tracker:
        tracker.mark_failed(task_id, duration, "Exhausted retries")
    raise RuntimeError(f"Agent '{agent.name}' exhausted {max_retries + 1} attempts")

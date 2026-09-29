"""Async orchestration for debug jobs sharing evaluation device slots."""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from pathlib import Path

from ..runtime.device_pool import DevicePool
from .jobs import (
    DebugJob,
    DebugJobRequest,
    DebugJobStore,
    LocalDebugJobRunner,
)


class DebugService:
    """Own debug-job tasks, cancellation, and scheduler integration."""

    def __init__(
        self,
        store: DebugJobStore,
        runner: LocalDebugJobRunner,
        *,
        device_pool: DevicePool,
        executor: Executor,
    ) -> None:
        self.store = store
        self.runner = runner
        self.device_pool = device_pool
        self.executor = executor
        self.tasks: dict[str, asyncio.Task] = {}

    async def close(self) -> None:
        self.runner.cancel_all()
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute(self, job_id: str) -> None:
        device: str | None = None
        try:
            device = await self.device_pool.acquire()
            if not self.store.begin(job_id, device):
                return
            request = self.store.request(job_id)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.runner.run(
                    job_id,
                    request,
                    device=device,
                ),
            )
            self.store.finish(job_id, result)
        except asyncio.CancelledError:
            self.store.cancel(job_id)
            if device is not None:
                self.runner.cancel(job_id)
            raise
        except Exception as exc:
            self.store.fail(job_id, f"{type(exc).__name__}: {exc}")
        finally:
            if device is not None:
                await asyncio.shield(self.device_pool.release(device))

    def _forget(self, job_id: str) -> None:
        self.tasks.pop(job_id, None)

    async def create(self, request: DebugJobRequest) -> DebugJob:
        job = self.store.create(request)
        task = asyncio.create_task(self._execute(job.job_id))
        self.tasks[job.job_id] = task
        task.add_done_callback(
            lambda _task, job_id=job.job_id: self._forget(job_id)
        )
        await task
        return self.store.get(job.job_id)

    async def get(self, job_id: str, *, wait_seconds: float = 0) -> DebugJob:
        job = self.store.get(job_id)
        deadline = asyncio.get_running_loop().time() + wait_seconds
        while (
            not job.terminal
            and asyncio.get_running_loop().time() < deadline
        ):
            remaining = deadline - asyncio.get_running_loop().time()
            await asyncio.sleep(min(0.25, remaining))
            job = self.store.get(job_id)
        return job

    def cancel(self, job_id: str) -> DebugJob:
        before = self.store.get(job_id)
        job = self.store.cancel(job_id)
        if before.status == "QUEUED":
            task = self.tasks.get(job_id)
            if task is not None:
                task.cancel()
        elif before.status == "RUNNING":
            self.runner.cancel(job_id)
        return job

    def resolve_artifact(
        self,
        job_id: str,
        artifact_id: str,
    ) -> tuple[Path, str, str]:
        job = self.store.get(job_id)
        artifact = next(
            item for item in job.artifacts if item.id == artifact_id
        )
        path = self.store.artifact_path(job_id, artifact_id)
        return path, artifact.media_type, Path(artifact.path).name

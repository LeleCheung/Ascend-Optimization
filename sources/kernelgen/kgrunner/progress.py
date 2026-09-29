"""Task progress tracking and summary persistence."""

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    RESUMING = "resuming"


class TaskTracker:
    def __init__(self, path: Path | str, resume: bool = False):
        self.path = Path(path)
        self._tasks: dict[str, dict] = {}
        self._start_time = time.time()

        if resume and self.path.exists():
            self._load()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_task(self, task_id: str, gpu_id: int | None = None, metadata: dict | None = None) -> None:
        self._tasks[task_id] = {
            "status": TaskStatus.PENDING.value,
            "gpu_id": gpu_id,
            "metadata": metadata or {},
            "started_at": None,
            "finished_at": None,
            "duration": None,
            "result": None,
            "error": None,
            "attempts": 0,
        }
        self._save()

    def mark_running(self, task_id: str, gpu_id: int | None = None) -> None:
        task = self._tasks[task_id]
        task["status"] = TaskStatus.RUNNING.value
        task["started_at"] = time.time()
        if gpu_id is not None:
            task["gpu_id"] = gpu_id
        self._save()

    def mark_success(self, task_id: str, duration: float, result: dict | None = None) -> None:
        task = self._tasks[task_id]
        task["status"] = TaskStatus.SUCCESS.value
        task["duration"] = duration
        task["finished_at"] = time.time()
        task["result"] = result
        self._save()

    def mark_failed(self, task_id: str, duration: float, error: str) -> None:
        task = self._tasks[task_id]
        task["status"] = TaskStatus.FAILED.value
        task["duration"] = duration
        task["finished_at"] = time.time()
        task["error"] = error
        self._save()

    def mark_retrying(self, task_id: str, attempt: int, error: str) -> None:
        task = self._tasks[task_id]
        task["status"] = TaskStatus.RETRYING.value
        task["attempts"] = attempt
        task["error"] = error
        self._save()

    def mark_resuming(self, task_id: str, session_id: str) -> None:
        task = self._tasks[task_id]
        task["status"] = TaskStatus.RESUMING.value
        task["metadata"]["last_session_id"] = session_id
        self._save()

    def finalize(self) -> None:
        self._save()

    @property
    def summary(self) -> dict[str, int]:
        counts = {"total": 0, "success": 0, "failed": 0, "running": 0, "pending": 0}
        for task in self._tasks.values():
            counts["total"] += 1
            status = task["status"]
            if status == TaskStatus.SUCCESS.value:
                counts["success"] += 1
            elif status == TaskStatus.FAILED.value:
                counts["failed"] += 1
            elif status in (TaskStatus.RUNNING.value, TaskStatus.RETRYING.value, TaskStatus.RESUMING.value):
                counts["running"] += 1
            else:
                counts["pending"] += 1
        return counts

    def _save(self) -> None:
        data = {
            "start_time": self._start_time,
            "tasks": self._tasks,
            "summary": self.summary,
        }
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        self._tasks = data.get("tasks", {})
        self._start_time = data.get("start_time", time.time())

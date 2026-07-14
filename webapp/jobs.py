"""In-process background job runner for long-running training tasks.

HuggingFace ``Trainer.train()`` is a blocking, CPU/GPU-bound call, so we run it
in a background thread and expose a simple polling status object. This is
intentionally minimal (no Celery/Redis) — single-process, single-worker dev
server is enough for a research tool used by one person locally.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass
class Job:
    id: str
    kind: str
    status: str = "pending"  # pending | running | done | error
    progress_message: str = "Queued…"
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None


class JobManager:
    """Thread-safe in-memory registry of background jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, fn: Callable[[Job], dict[str, Any]]) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, kind=kind)
        with self._lock:
            self._jobs[job_id] = job

        def _runner() -> None:
            job.status = "running"
            try:
                result = fn(job)
                job.result = result
                job.status = "done"
                job.progress_message = "Finished."
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = f"{exc}\n\n{traceback.format_exc()}"
                job.progress_message = "Failed."
            finally:
                job.finished_at = datetime.now(timezone.utc)

        thread = threading.Thread(target=_runner, name=f"job-{job_id}", daemon=True)
        thread.start()
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


# Single process-wide instance — fine for a local single-worker dev server.
job_manager = JobManager()

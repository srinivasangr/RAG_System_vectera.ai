"""In-memory ingest job registry with live progress streaming.

Each upload becomes a Job. The ingestion runs in a worker thread; its
progress_cb pushes events onto a thread-safe queue that the SSE endpoint
drains in real time. We also keep the full event list so a client that
connects late still sees everything.

Single-process, single-user local tool — no external queue needed. (The
production path swaps this for Celery/Redis; see architecture_v2_final.md §10.)
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Job:
    job_id: str
    filename: str
    status: str = "queued"          # queued | running | done | error | skipped
    events: list[dict] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    _q: "queue.Queue[dict]" = field(default_factory=queue.Queue)

    def push(self, event: str, payload: dict | None = None) -> None:
        ev = {"event": event, "ts": time.time(), **(payload or {})}
        self.events.append(ev)
        self._q.put(ev)

    def drain_nowait(self) -> list[dict]:
        out = []
        try:
            while True:
                out.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return out


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], filename=filename)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def run(self, job: Job, pdf_path: Path, options: dict[str, Any]) -> None:
        """Spawn a worker thread that runs ingest_one_v2 and streams progress."""
        t = threading.Thread(target=self._worker, args=(job, pdf_path, options), daemon=True)
        t.start()

    def _worker(self, job: Job, pdf_path: Path, options: dict[str, Any]) -> None:
        # Import here so the API can boot even if heavy deps are slow to load.
        from rag_system.ingest.pipeline_v2 import ingest_one_v2
        from rag_system.llm_providers import get_llm

        job.status = "running"
        job.push("job_started", {"file": job.filename})

        def progress_cb(event: str, payload: dict) -> None:
            job.push(event, payload)

        try:
            ingest_llm = None
            prov, model = options.get("llm_provider"), options.get("llm_model")
            if prov or model:
                ingest_llm = get_llm(prov, model)

            result = ingest_one_v2(
                pdf_path,
                with_vision=options.get("with_vision", True),
                with_propositions=options.get("with_propositions", True),
                vision_budget=options.get("vision_budget"),
                force=options.get("force", True),
                progress_cb=progress_cb,
                llm=ingest_llm,
            )
            job.result = result
            if result.get("skipped"):
                job.status = "skipped"
                job.push("done", {"skipped": True, **result})
            else:
                job.status = "done"
                job.push("done", result)
        except Exception as e:  # noqa: BLE001
            job.error = str(e)
            job.status = "error"
            job.push("error", {"message": str(e)})


# Module-level singleton
registry = JobRegistry()

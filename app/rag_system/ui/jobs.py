"""Serial-queue background ingestion job manager.

Design:
  - One dispatcher thread inside the JobManager singleton.
  - `submit()` only enqueues; it does NOT start a worker.
  - The dispatcher picks the next queued job when a worker slot is free.
  - Default `max_concurrent=1` — Docling parses are CPU/memory-heavy enough that
    serial execution is the safe default. Bump this only on a beefy box.
  - All job state lives on a single shared dict guarded by one lock.

Why serial instead of "thread per upload": uploading 8 PDFs at once spawned
8 simultaneous Docling parses, each loading its own ML models and competing
for the GIL + memory + Gemini rate limits. None of them made measurable
progress. A FIFO queue with 1 active job at a time finishes each one
predictably in ~2-3 min.
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_system.ingest.pipeline import ingest_one

log = logging.getLogger(__name__)


JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_DONE = "done"
JOB_STATUS_ERROR = "error"


@dataclass
class IngestJob:
    id: str
    file_name: str
    pdf_path: Path
    with_vision: bool
    vision_call_budget: int | None
    status: str = JOB_STATUS_QUEUED
    queued_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    # Live progress
    current_stage: str = "queued"
    parse_total_pages: int = 0
    parse_done_pages: int = 0
    parse_done_pct: float = 0.0
    vision_total: int = 0
    vision_done: int = 0
    embed_total: int = 0
    embed_done: int = 0
    log_lines: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    def add_log(self, line: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self.log_lines.append(f"[{ts}] {line}")
            if len(self.log_lines) > 30:
                self.log_lines = self.log_lines[-30:]

    def overall_progress(self) -> float:
        with self._lock:
            if self.status == JOB_STATUS_DONE:
                return 1.0
            if self.status == JOB_STATUS_QUEUED:
                return 0.0
            p_parse = self.parse_done_pct
            p_vision = (self.vision_done / self.vision_total) if self.vision_total else 0
            p_embed = (self.embed_done / self.embed_total) if self.embed_total else 0
            stage_done = {
                "queued":        0.00,
                "starting":      0.00,
                "parsing":       0.00,
                "parse_done":    0.60,
                "vision":        0.60,
                "vision_done":   0.75,
                "chunking":      0.75,
                "embedding":     0.75,
                "embed_done":    0.90,
                "upserting":     0.90,
                "upserted":      1.00,
            }
            base = stage_done.get(self.current_stage, 0.0)
            if self.current_stage == "parsing":
                return min(base + 0.60 * p_parse, 0.59)
            if self.current_stage == "vision":
                return min(0.60 + 0.15 * p_vision, 0.74)
            if self.current_stage == "embedding":
                return min(0.75 + 0.15 * p_embed, 0.89)
            return base

    def elapsed_s(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def wait_s(self) -> float:
        end = self.started_at or datetime.now()
        return (end - self.queued_at).total_seconds()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "file_name": self.file_name,
                "status": self.status,
                "current_stage": self.current_stage,
                "progress": self.overall_progress(),
                "parse_total_pages": self.parse_total_pages,
                "parse_done_pages": self.parse_done_pages,
                "vision_total": self.vision_total,
                "vision_done": self.vision_done,
                "embed_total": self.embed_total,
                "embed_done": self.embed_done,
                "log_lines": list(self.log_lines),
                "result": dict(self.result) if self.result else None,
                "error": self.error,
                "elapsed_s": self.elapsed_s(),
                "wait_s": self.wait_s(),
            }

    # ------------------------------------------------------------------
    def handle_event(self, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if event == "start":
                self.current_stage = "parsing"
                self.add_log(f"start: {payload.get('file', '')}")
            elif event == "parse_start":
                self.parse_total_pages = payload.get("total", 0)
                self.add_log(f"parsing {self.parse_total_pages} pages "
                             f"(batch={payload.get('batch_size')})")
            elif event == "parse_batch":
                end = payload.get("end", 0)
                total = payload.get("total", self.parse_total_pages) or 1
                self.parse_done_pages = max(self.parse_done_pages, end)
                self.parse_done_pct = self.parse_done_pages / total
                self.add_log(
                    f"parsed pages {payload.get('start')}-{end}/{total} "
                    f"({payload.get('elapsed_s', 0):.1f}s)"
                )
            elif event == "parse_done":
                self.current_stage = "vision"
                self.add_log(
                    f"parse done: {payload.get('pages')} pages, "
                    f"{payload.get('images')} images"
                )
            elif event == "vision_start":
                self.vision_total = payload.get("total", 0)
                self.add_log(f"vision: describing {self.vision_total} images…")
            elif event == "vision_progress":
                self.vision_done = payload.get("done", 0)
            elif event == "vision_done":
                self.current_stage = "embedding"
                self.add_log(
                    f"vision done: {payload.get('described', 0)}/"
                    f"{payload.get('total', 0)} kept"
                )
            elif event == "chunk_done":
                self.add_log(
                    f"chunked: {payload.get('total')} chunks "
                    f"{dict(payload.get('by_type', {}))}"
                )
            elif event == "embed_start":
                self.current_stage = "embedding"
                self.embed_total = payload.get("total", 0)
                self.add_log(
                    f"embedding {self.embed_total} chunks "
                    f"({payload.get('provider')}, dim={payload.get('dim')})"
                )
            elif event == "embed_progress":
                self.embed_done = payload.get("done", 0)
            elif event == "embed_done":
                self.current_stage = "upserting"
                self.embed_done = self.embed_total
                self.add_log(f"embedded {self.embed_total} chunks")
            elif event == "upsert_done":
                self.current_stage = "upserted"
                self.add_log(
                    f"snowflake: {payload.get('doc_status')} · "
                    f"{payload.get('chunks')} chunks stored"
                )
            elif event == "done":
                self.status = JOB_STATUS_DONE
                self.finished_at = datetime.now()
                self.result = dict(payload)
                self.add_log(f"DONE in {payload.get('elapsed_s')}s")
            elif event == "error":
                self.status = JOB_STATUS_ERROR
                self.finished_at = datetime.now()
                self.error = f"{payload.get('stage')}: {payload.get('message')}"
                self.add_log(f"ERROR ({payload.get('stage')}): {payload.get('message')}")


# ---------------------------------------------------------------------------
class JobManager:
    def __init__(self, *, max_concurrent: int = 1) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._pending: deque[str] = deque()
        self._running: set[str] = set()
        self._max_concurrent = max_concurrent
        self._cv = threading.Condition()
        self._stop = False
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name="ingest-dispatcher", daemon=True,
        )
        self._dispatcher.start()

    # ------------------------------------------------------------------
    def submit(
        self,
        *,
        file_name: str,
        pdf_path: Path,
        with_vision: bool = True,
        vision_call_budget: int | None = None,
    ) -> str:
        job_id = uuid.uuid4().hex[:8]
        job = IngestJob(
            id=job_id,
            file_name=file_name,
            pdf_path=pdf_path,
            with_vision=with_vision,
            vision_call_budget=vision_call_budget,
        )
        with self._cv:
            self._jobs[job_id] = job
            self._pending.append(job_id)
            self._cv.notify_all()
        log.info("queued ingest job %s (%s); pending=%d running=%d",
                 job_id, file_name, len(self._pending), len(self._running))
        return job_id

    def list(self) -> list[IngestJob]:
        with self._cv:
            # Sort: running first, then queued (by queue position), then done/error newest-first
            running = [j for j in self._jobs.values() if j.status == JOB_STATUS_RUNNING]
            queued = [self._jobs[jid] for jid in self._pending]
            finished = sorted(
                [j for j in self._jobs.values()
                 if j.status in (JOB_STATUS_DONE, JOB_STATUS_ERROR)],
                key=lambda j: j.finished_at or j.queued_at, reverse=True,
            )
            return running + queued + finished

    def get(self, job_id: str) -> IngestJob | None:
        with self._cv:
            return self._jobs.get(job_id)

    def remove(self, job_id: str) -> None:
        with self._cv:
            self._jobs.pop(job_id, None)
            try:
                self._pending.remove(job_id)
            except ValueError:
                pass

    def has_active(self) -> bool:
        with self._cv:
            return bool(self._running or self._pending)

    def queue_position(self, job_id: str) -> int | None:
        """1-based position in the pending queue. None if running/done/error."""
        with self._cv:
            try:
                return list(self._pending).index(job_id) + 1
            except ValueError:
                return None

    # ------------------------------------------------------------------
    def _dispatch_loop(self) -> None:
        """Continuously pull jobs from the queue and launch workers."""
        while not self._stop:
            with self._cv:
                while not self._stop and (
                    len(self._running) >= self._max_concurrent or not self._pending
                ):
                    self._cv.wait()
                if self._stop:
                    return
                job_id = self._pending.popleft()
                self._running.add(job_id)
                job = self._jobs[job_id]
                job.status = JOB_STATUS_RUNNING
                job.started_at = datetime.now()
                job.current_stage = "starting"
            # Run the worker on a fresh thread so the dispatcher can keep going
            threading.Thread(
                target=self._run_and_release, args=(job,),
                name=f"ingest-{job_id}", daemon=True,
            ).start()

    def _run_and_release(self, job: IngestJob) -> None:
        try:
            self._run(job)
        finally:
            with self._cv:
                self._running.discard(job.id)
                self._cv.notify_all()

    def _run(self, job: IngestJob) -> None:
        try:
            result = ingest_one(
                job.pdf_path,
                with_vision=job.with_vision,
                vision_call_budget=job.vision_call_budget,
                progress_cb=job.handle_event,
            )
            with job._lock:
                if job.status == JOB_STATUS_RUNNING:
                    job.status = JOB_STATUS_DONE
                    job.finished_at = datetime.now()
                    job.result = result
        except Exception as e:
            with job._lock:
                job.status = JOB_STATUS_ERROR
                job.finished_at = datetime.now()
                job.error = f"{type(e).__name__}: {e}"
                job.add_log(f"FATAL: {type(e).__name__}: {e}")
                tb = traceback.format_exc().splitlines()[-3:]
                for line in tb:
                    job.add_log(line)


# Process-global singleton
_MANAGER: JobManager | None = None


def get_job_manager() -> JobManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = JobManager(max_concurrent=1)
    return _MANAGER

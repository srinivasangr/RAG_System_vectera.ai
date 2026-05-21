"""FastAPI backend + minimal UI (one process).

Endpoints
  GET  /                          -> minimal UI (upload + live ingest monitor)
  POST /api/ingest                -> upload a PDF, start ingestion, return job_id
  GET  /api/ingest/{job}/stream   -> SSE live progress for a job
  GET  /api/ingest/{job}          -> job snapshot (polling fallback)
  GET  /api/documents             -> ingested docs + artifact counts
  DELETE /api/documents/{doc_id}  -> delete a doc and its artifacts
  GET  /api/corpus-profile        -> runtime corpus profile (domain-agnostic)
  GET  /health                    -> liveness + dependency check

Run from app/:  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from api.jobs import registry
from rag_system.config import settings

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_INDEX_FILE = _HERE / "templates" / "index.html"

app = FastAPI(title="RAG System API", version="2.0")
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    # Read fresh each request (cheap) so UI edits show without a server restart.
    return HTMLResponse(_INDEX_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    deps = {"snowflake": False}
    try:
        from rag_system.storage.db import get_connection
        with get_connection() as conn:
            conn.cursor().execute("SELECT 1")
        deps["snowflake"] = True
    except Exception as e:  # noqa: BLE001
        deps["snowflake_error"] = str(e)[:200]
    return {"status": "ok", "deps": deps}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
@app.post("/api/ingest")
async def start_ingest(
    file: UploadFile = File(...),
    with_vision: bool = Form(True),
    with_propositions: bool = Form(True),
    llm_provider: str = Form(""),
    llm_model: str = Form(""),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported.")

    # Save into the corpus directory so the file is available for re-ingest.
    settings.documents_path.mkdir(parents=True, exist_ok=True)
    dest = settings.documents_path / file.filename
    data = await file.read()
    dest.write_bytes(data)

    job = registry.create(file.filename)
    options = {
        "with_vision": with_vision,
        "with_propositions": with_propositions,
        "llm_provider": llm_provider or None,
        "llm_model": llm_model or None,
        "force": True,
    }
    registry.run(job, dest, options)
    return {"job_id": job.job_id, "filename": file.filename, "options": options}


@app.get("/api/ingest/{job_id}")
async def ingest_snapshot(job_id: str):
    job = registry.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {
        "job_id": job.job_id, "filename": job.filename, "status": job.status,
        "events": job.events, "result": job.result, "error": job.error,
    }


@app.get("/api/ingest/{job_id}/stream")
async def ingest_stream(job_id: str):
    job = registry.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    async def gen():
        # Replay any events already produced (late-join safety)
        replayed = 0
        for ev in list(job.events):
            yield {"data": json.dumps(ev)}
            replayed += 1
            if ev["event"] in ("done", "error"):
                return
        # Stream new events
        while True:
            new = job.drain_nowait()
            for ev in new:
                yield {"data": json.dumps(ev)}
                if ev["event"] in ("done", "error"):
                    return
            if job.status in ("done", "error", "skipped") and not new:
                # ensure terminal already sent; otherwise emit a final tick
                return
            await asyncio.sleep(0.25)

    return EventSourceResponse(gen())


@app.get("/api/jobs")
async def list_jobs():
    return [
        {"job_id": j.job_id, "filename": j.filename, "status": j.status,
         "result": j.result, "error": j.error}
        for j in registry.list()
    ]


# ---------------------------------------------------------------------------
# Documents / corpus
# ---------------------------------------------------------------------------
@app.get("/api/documents")
async def documents():
    from rag_system.storage import repository_v2 as repo
    try:
        return repo.list_documents()
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    from rag_system.storage import repository_v2 as repo
    counts = repo.delete_document_v2(doc_id)
    return {"deleted": doc_id, "counts": counts}


@app.get("/api/corpus-profile")
async def corpus_profile():
    from rag_system.storage import repository_v2 as repo
    try:
        return repo.corpus_profile()
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Query (ask)
# ---------------------------------------------------------------------------
def _answer_to_dict(a) -> dict:
    cited = set(a.cited_numbers)
    sources = [{
        "n": i, "company": s.company, "doc_type": s.doc_type,
        "page_number": s.page_number,
        "as_of_date": str(s.as_of_date) if s.as_of_date else None,
        "version_label": s.version_label, "slide_title": s.slide_title,
        "filename": s.filename,
        "parent_id": s.parent_id, "doc_id": s.doc_id,
        "snippet": (s.text or "")[:400],
        "cited": i in cited,
        "conflict_group": s.conflict_group,
    } for i, s in enumerate(a.sources, start=1)]
    # provider_chain isn't JSON-critical; keep timings lean for the wire
    timings = {k: v for k, v in (a.timings or {}).items() if k != "provider_chain"}
    return {
        "question": a.question, "answer": a.answer,
        "intent": getattr(a.plan, "intent", None),
        "sub_queries": getattr(a.plan, "sub_queries", []),
        "cited_numbers": a.cited_numbers, "conflicts": a.conflicts,
        "engine": f"{a.llm_provider}/{a.llm_model}",
        "timings": timings, "sources": sources,
    }


def _build_filters(doc_ids):
    from rag_system.retrieval.filters import RetrievalFilters
    return RetrievalFilters(doc_ids=list(doc_ids or []))


@app.post("/api/query")
async def query(payload: dict):
    """Run a question end-to-end and return answer + numbered sources + trace."""
    q = (payload.get("query") or "").strip()
    if not q:
        raise HTTPException(400, "empty query")
    provider = payload.get("provider") or "gemini"
    model = payload.get("model") or "gemini-2.5-flash"
    filters = _build_filters(payload.get("doc_ids"))

    from rag_system.generation.generate_v2 import answer_query
    a = await asyncio.to_thread(
        lambda: answer_query(q, provider=provider, model=model, filters=filters))
    return _answer_to_dict(a)


@app.post("/api/query/stream")
async def query_stream(payload: dict):
    """Stream live stage events (routing/retrieving/reranking/expanding/generating)
    then the final answer, over SSE."""
    import queue
    import threading

    q = (payload.get("query") or "").strip()
    if not q:
        raise HTTPException(400, "empty query")
    provider = payload.get("provider") or "gemini"
    model = payload.get("model") or "gemini-2.5-flash"
    filters = _build_filters(payload.get("doc_ids"))

    evq: "queue.Queue[dict]" = queue.Queue()
    holder: dict = {}

    def progress_cb(stage):
        evq.put({"event": "stage", "stage": stage})

    def run():
        from rag_system.generation.generate_v2 import answer_query
        try:
            a = answer_query(q, provider=provider, model=model,
                             filters=filters, progress_cb=progress_cb)
            holder["result"] = _answer_to_dict(a)
            evq.put({"event": "done"})
        except Exception as e:  # noqa: BLE001
            holder["error"] = str(e)
            evq.put({"event": "error", "message": str(e)})

    threading.Thread(target=run, daemon=True).start()

    async def gen():
        while True:
            try:
                ev = evq.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.15)
                continue
            if ev["event"] in ("done", "error"):
                ev["result"] = holder.get("result")
                yield {"data": json.dumps(ev)}
                return
            yield {"data": json.dumps(ev)}

    return EventSourceResponse(gen())


@app.get("/api/page-image/{parent_id}")
async def page_image(parent_id: str):
    """Serve the stored page thumbnail for a citation."""
    import base64
    from fastapi.responses import Response
    from rag_system.storage import repository_v3 as repo3
    img = repo3.get_page_image(parent_id)
    if not img:
        raise HTTPException(404, "no image")
    mime, b64 = img
    return Response(content=base64.b64decode(b64), media_type=mime or "image/jpeg")

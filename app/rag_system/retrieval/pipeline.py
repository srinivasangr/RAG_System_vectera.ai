"""Multi-stage retrieval orchestrator (the public retrieval entrypoint).

Pipeline:
  route ─▶ multi-source retrieve (dense+lexical+structured) ─▶ RRF
        ─▶ hydrate chunk metadata
        ─▶ cross-encoder rerank
        ─▶ diversify (per-doc quota)
        ─▶ version-pair expansion (surface sibling-version docs)
        ─▶ small-to-big (chunk ─▶ parent slide)
        ─▶ conflict detection (same entity, multiple as-of dates)
  ─▶ ranked Sources + conflict tags + stage timings (observability)

Each stage targets a specific failure mode of naive single-shot retrieval; see
docs/architecture.md.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date

from rag_system.config import settings
from rag_system.llm_providers import get_embedder
from rag_system.retrieval.filters import RetrievalFilters
from rag_system.retrieval import reranker, router
from rag_system.retrieval.retrieve import (
    RetrievedChunk, hydrate, multi_source_retrieve,
)
from rag_system.storage.db import get_connection

log = logging.getLogger(__name__)


@dataclass
class Source:
    parent_id: str
    doc_id: str
    company: str | None
    doc_type: str | None
    doc_date: date | None
    as_of_date: date | None
    version_label: str | None
    page_number: int
    slide_title: str | None
    text: str                      # parent (slide) text — small-to-big context
    rerank_score: float | None
    filename: str | None = None    # original PDF filename (provenance)
    matched_chunk_ids: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    conflict_group: str | None = None


@dataclass
class RetrievalResult:
    query: str
    plan: router.RoutePlan
    sources: list[Source]
    conflicts: list[dict]
    timings: dict


# ---------------------------------------------------------------------------
# Diversification — per-doc quota
# ---------------------------------------------------------------------------
def _diversify(chunks: list[RetrievedChunk], *, top_k: int, per_doc: int = 3) -> list[RetrievedChunk]:
    out, doc_counts = [], {}
    for c in chunks:
        if doc_counts.get(c.doc_id, 0) >= per_doc:
            continue
        out.append(c)
        doc_counts[c.doc_id] = doc_counts.get(c.doc_id, 0) + 1
        if len(out) >= top_k:
            break
    return out


# ---------------------------------------------------------------------------
# Version-pair expansion
# ---------------------------------------------------------------------------
def _family_map() -> dict[str, list[str]]:
    """doc_family_id -> [doc_id, ...] for families with >1 document."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT doc_family_id, doc_id FROM documents WHERE doc_family_id IS NOT NULL")
        fam: dict[str, list[str]] = {}
        for fid, did in cur.fetchall():
            fam.setdefault(fid, []).append(did)
        cur.close()
    return {k: v for k, v in fam.items() if len(v) > 1}


def _expand_version_pairs(query: str, chunks: list[RetrievedChunk], *,
                          plan: router.RoutePlan) -> list[RetrievedChunk]:
    """If a result's document has sibling versions not represented, pull the
    best-matching chunk from each missing sibling so the LLM can compare/contrast
    versions. Skipped when the user scoped a specific date."""
    if plan.as_of_date_filter:
        return chunks
    fam = _family_map()
    if not fam:
        return chunks
    present_docs = {c.doc_id for c in chunks}
    # which families are represented, and which sibling docs are missing?
    missing: list[str] = []
    for c in chunks:
        sibs = fam.get(c.doc_family_id or "", [])
        for d in sibs:
            if d not in present_docs and d not in missing:
                missing.append(d)
    if not missing:
        return chunks

    embedder = get_embedder()
    qv = embedder.embed_one(query)
    vec_lit = "[" + ",".join(f"{x:.8f}" for x in qv) + "]"
    add: list[RetrievedChunk] = []
    with get_connection() as conn:
        cur = conn.cursor()
        for d in missing[:6]:
            cur.execute(f"""
                SELECT c.chunk_id, c.parent_id, c.doc_id, c.page_number, c.chunk_type,
                       c.text, c.company, c.doc_type, c.doc_date, c.as_of_date,
                       c.doc_family_id, c.version_label, c.slide_title, c.confidence,
                       VECTOR_COSINE_SIMILARITY(c.embedding, {vec_lit}::VECTOR(FLOAT,768)) AS s
                FROM chunks c WHERE c.doc_id = %s AND c.embedding IS NOT NULL
                ORDER BY s DESC LIMIT 2
            """, (d,))
            for r in cur.fetchall():
                add.append(RetrievedChunk(
                    chunk_id=r[0], parent_id=r[1], doc_id=r[2], page_number=r[3],
                    chunk_type=r[4], text=r[5], company=r[6], doc_type=r[7],
                    doc_date=r[8], as_of_date=r[9], doc_family_id=r[10],
                    version_label=r[11], slide_title=r[12], confidence=r[13],
                    score=float(r[14] or 0), rerank_score=float(r[14] or 0),
                    sources=["version_pair"],
                ))
        cur.close()
    if add:
        log.info("version-pair: added %d sibling-version chunks", len(add))
    return chunks + add


# ---------------------------------------------------------------------------
# Small-to-big — chunk -> parent slide
# ---------------------------------------------------------------------------
def _filename_map(doc_ids: list[str]) -> dict[str, str]:
    if not doc_ids:
        return {}
    uniq = list(dict.fromkeys(doc_ids))
    with get_connection() as conn:
        cur = conn.cursor()
        ph = ",".join(["%s"] * len(uniq))
        cur.execute(f"SELECT doc_id, original_filename FROM documents WHERE doc_id IN ({ph})", uniq)
        m = {r[0]: r[1] for r in cur.fetchall()}
        cur.close()
    return m


def _to_sources(chunks: list[RetrievedChunk], filenames: dict[str, str] | None = None) -> list[Source]:
    """Replace each chunk with its parent slide (deduped), preserving provenance."""
    filenames = filenames or {}
    parent_ids = [c.parent_id for c in chunks if c.parent_id]
    parents: dict[str, tuple] = {}
    if parent_ids:
        uniq = list(dict.fromkeys(parent_ids))
        with get_connection() as conn:
            cur = conn.cursor()
            ph = ",".join(["%s"] * len(uniq))
            cur.execute(f"""
                SELECT parent_id, doc_id, company, doc_type, doc_date, as_of_date,
                       version_label, page_number, slide_title, text
                FROM parent_chunks WHERE parent_id IN ({ph})
            """, uniq)
            for r in cur.fetchall():
                parents[r[0]] = r
            cur.close()

    seen: dict[str, Source] = {}
    order: list[str] = []
    for c in chunks:
        pid = c.parent_id or c.chunk_id
        if pid in seen:
            seen[pid].matched_chunk_ids.append(c.chunk_id)
            continue
        p = parents.get(c.parent_id) if c.parent_id else None
        if p:
            src = Source(
                parent_id=p[0], doc_id=p[1], company=p[2], doc_type=p[3],
                doc_date=p[4], as_of_date=p[5], version_label=p[6],
                page_number=p[7], slide_title=p[8], text=p[9],
                rerank_score=c.rerank_score, matched_chunk_ids=[c.chunk_id],
                sources=c.sources,
            )
        else:  # no parent row — fall back to the chunk itself
            src = Source(
                parent_id=pid, doc_id=c.doc_id, company=c.company, doc_type=c.doc_type,
                doc_date=c.doc_date, as_of_date=c.as_of_date, version_label=c.version_label,
                page_number=c.page_number, slide_title=c.slide_title, text=c.text,
                rerank_score=c.rerank_score, matched_chunk_ids=[c.chunk_id], sources=c.sources,
            )
        src.filename = filenames.get(src.doc_id)
        seen[pid] = src
        order.append(pid)
    return [seen[pid] for pid in order]


# ---------------------------------------------------------------------------
# Conflict detection (F5/version conflicts) — lightweight
# ---------------------------------------------------------------------------
def _detect_conflicts(sources: list[Source]) -> list[dict]:
    """Flag entities that appear with MULTIPLE as-of dates in the result set —
    a likely version/temporal conflict the answer must surface with attribution."""
    by_company: dict[str, set] = {}
    for s in sources:
        if s.company and s.as_of_date:
            by_company.setdefault(s.company, set()).add(s.as_of_date)
    conflicts = []
    for company, dates in by_company.items():
        if len(dates) > 1:
            grp = f"{company}::multi_date"
            for s in sources:
                if s.company == company and s.as_of_date in dates:
                    s.conflict_group = grp
            conflicts.append({
                "company": company,
                "as_of_dates": sorted(str(d) for d in dates),
            })
    return conflicts


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    *,
    filters: RetrievalFilters | None = None,
    top_k: int | None = None,
    candidate_k: int = 40,
    rerank_pool: int = 30,
    llm=None,
    progress_cb=None,
    plan=None,
) -> RetrievalResult:
    filters = filters or RetrievalFilters()
    top_k = top_k or settings.retrieval_top_k
    timings: dict = {}

    def _emit(stage):
        if progress_cb:
            try:
                progress_cb(stage)
            except Exception:
                pass

    # Reuse a pre-computed plan (lets the caller short-circuit before retrieval)
    # to avoid a second router LLM call.
    t = time.perf_counter()
    if plan is None:
        _emit("routing")
        plan = router.route(query, llm=llm)
    timings["route_ms"] = int((time.perf_counter() - t) * 1000)

    if plan.prefer_recent:
        filters.prefer_recent = True

    _emit("retrieving")
    t = time.perf_counter()
    cands = multi_source_retrieve(
        plan.sub_queries, filters=filters, candidate_k=candidate_k,
        use_tables=plan.needs_tables or plan.needs_charts,
    )
    chunks = hydrate(cands, limit=rerank_pool)
    timings["retrieve_ms"] = int((time.perf_counter() - t) * 1000)
    timings["n_candidates"] = len(cands)

    _emit("reranking")
    t = time.perf_counter()
    reranked = reranker.rerank(query, chunks, top_k=max(top_k * 2, 12))
    timings["rerank_ms"] = int((time.perf_counter() - t) * 1000)

    _emit("expanding")
    diversified = _diversify(reranked, top_k=top_k)
    expanded = _expand_version_pairs(query, diversified, plan=plan)
    filenames = _filename_map([c.doc_id for c in expanded])
    sources = _to_sources(expanded, filenames)
    conflicts = _detect_conflicts(sources)

    log.info("retrieve(%r): %d sources, %d conflicts, timings=%s",
             query[:50], len(sources), len(conflicts), timings)
    return RetrievalResult(query=query, plan=plan, sources=sources,
                           conflicts=conflicts, timings=timings)

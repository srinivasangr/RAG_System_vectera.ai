"""Hybrid retrieval over Snowflake: dense cosine + lexical, fused with RRF.

Why hybrid? presentation decks are dense with tickers, metric acronyms (FFO, NOI,
AFFO), company names, and specific numbers. Pure dense embeddings miss exact
matches; pure keyword misses paraphrases. Reciprocal Rank Fusion combines both
without needing tuned weights.

Why RRF? It's parameter-free, robust to score-scale differences between the
two retrievers, and trivial to implement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Sequence

from rag_system.config import settings
from rag_system.llm_providers import get_embedder
from rag_system.retrieval.filters import RetrievalFilters
from rag_system.storage.db import get_connection


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_type: str
    text: str
    company: str | None
    doc_date: date | None
    version_label: str | None
    score: float       # final fused score (higher = better)
    dense_rank: int | None = None
    lexical_rank: int | None = None
    source_path: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vec_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _filter_clauses(f: RetrievalFilters, *, table: str = "c") -> tuple[str, list]:
    """Return (sql_where, params) extending an existing WHERE clause.

    `table` is the alias to prefix on each filter column (e.g. 'c' for chunks,
    's' for scored CTE in the lexical query). Required because the SELECT joins
    documents and both tables carry overlapping columns.
    """
    clauses: list[str] = []
    params: list = []
    if f.doc_ids:
        placeholders = ",".join(["%s"] * len(f.doc_ids))
        clauses.append(f"{table}.doc_id IN ({placeholders})")
        params.extend(f.doc_ids)
    if f.companies:
        placeholders = ",".join(["%s"] * len(f.companies))
        clauses.append(f"{table}.company IN ({placeholders})")
        params.extend(f.companies)
    if f.doc_types:
        placeholders = ",".join(["%s"] * len(f.doc_types))
        # doc_type lives on documents only; join via d
        clauses.append(f"d.doc_type IN ({placeholders})")
        params.extend(f.doc_types)
    if f.date_from:
        clauses.append(f"{table}.doc_date >= %s")
        params.append(f.date_from)
    if f.date_to:
        clauses.append(f"{table}.doc_date <= %s")
        params.append(f.date_to)
    return (" AND ".join(clauses), params)


def _keyword_tokens(query: str) -> list[str]:
    """Extract meaningful tokens for the lexical search (drop stop-ish words)."""
    import re
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "of", "for", "to", "and", "or", "in", "on", "at", "by", "with",
        "what", "which", "who", "how", "why", "when", "where", "do", "does",
        "this", "that", "these", "those", "it", "its", "as", "from",
    }
    toks = [t for t in re.findall(r"[A-Za-z0-9.]{2,}", query.lower()) if t not in stop]
    # Keep all-caps acronyms / tickers from the original casing
    caps = re.findall(r"\b[A-Z]{2,}\b", query)
    for c in caps:
        if c.lower() not in toks:
            toks.append(c)
    return toks[:20]


# ---------------------------------------------------------------------------
# Recency-intent auto-detection
# ---------------------------------------------------------------------------
# Matches whole-word recency keywords. Multi-word phrases are matched explicitly
# so "recently announced" hits but "recentralize" does not (\b boundaries).
_RECENCY_RE = re.compile(
    r"\b(?:"
    r"latest|current|currently|now|today|recent|recently|"
    r"most[\s\-]+recent|newest|new|updated|up[\s\-]+to[\s\-]+date|"
    r"as[\s\-]+of[\s\-]+today|as[\s\-]+of[\s\-]+now|this[\s\-]+(?:quarter|year|month)"
    r")\b",
    re.IGNORECASE,
)


def has_recency_intent(query: str) -> bool:
    """True if the query implies the user wants the most recent doc per company."""
    return bool(_RECENCY_RE.search(query or ""))


# ---------------------------------------------------------------------------
# The two retrievers
# ---------------------------------------------------------------------------
def _dense_retrieve(
    cur, *,
    query_vec: list[float],
    filters: RetrievalFilters,
    k: int,
) -> list[tuple]:
    extra_where, extra_params = _filter_clauses(filters, table="c")
    vec_lit = _vec_literal(query_vec)
    where = f"WHERE {extra_where}" if extra_where else ""
    sql = f"""
        SELECT c.chunk_id, c.doc_id, c.page_number, c.chunk_type, c.text,
               c.company, c.doc_date, c.version_label,
               VECTOR_COSINE_SIMILARITY(c.embedding, {vec_lit}::VECTOR(FLOAT, 768)) AS score,
               d.source_path
        FROM chunks c
        LEFT JOIN documents d ON d.doc_id = c.doc_id
        {where}
        ORDER BY score DESC
        LIMIT {int(k)}
    """
    cur.execute(sql, extra_params)
    return cur.fetchall()


def _lexical_retrieve(
    cur, *,
    tokens: list[str],
    filters: RetrievalFilters,
    k: int,
) -> list[tuple]:
    if not tokens:
        return []
    # In the outer SELECT below the scored CTE is aliased as `s`
    extra_where, extra_params = _filter_clauses(filters, table="s")

    # Build a relevance score = sum over tokens of CONTAINS-as-int.
    # Use a CTE so the IFF expressions (and their params) only appear once.
    score_parts: list[str] = []
    params: list = []
    for t in tokens:
        score_parts.append("IFF(LOWER(c.text) LIKE %s, 1, 0)")
        params.append(f"%{t.lower()}%")
    score_expr = " + ".join(score_parts) if score_parts else "0"

    extra = f"AND {extra_where}" if extra_where else ""
    params_full = list(params) + list(extra_params)

    sql = f"""
        WITH scored AS (
          SELECT c.chunk_id, c.doc_id, c.page_number, c.chunk_type, c.text,
                 c.company, c.doc_date, c.version_label,
                 ({score_expr}) AS score
          FROM chunks c
        )
        SELECT s.chunk_id, s.doc_id, s.page_number, s.chunk_type, s.text,
               s.company, s.doc_date, s.version_label, s.score,
               d.source_path
        FROM scored s
        LEFT JOIN documents d ON d.doc_id = s.doc_id
        WHERE s.score > 0 {extra}
        ORDER BY s.score DESC, s.doc_date DESC NULLS LAST
        LIMIT {int(k)}
    """
    cur.execute(sql, params_full)
    return cur.fetchall()


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def _rrf_fuse(
    dense_rows: list[tuple],
    lexical_rows: list[tuple],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank))."""
    by_id: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    def _row_to_chunk(row, *, dense_rank=None, lexical_rank=None) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=row[0], doc_id=row[1], page_number=row[2],
            chunk_type=row[3], text=row[4],
            company=row[5], doc_date=row[6], version_label=row[7],
            score=0.0,
            dense_rank=dense_rank, lexical_rank=lexical_rank,
            source_path=row[9] if len(row) > 9 else None,
        )

    for rank, row in enumerate(dense_rows, start=1):
        cid = row[0]
        by_id[cid] = _row_to_chunk(row, dense_rank=rank)
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    for rank, row in enumerate(lexical_rows, start=1):
        cid = row[0]
        if cid in by_id:
            by_id[cid].lexical_rank = rank
        else:
            by_id[cid] = _row_to_chunk(row, lexical_rank=rank)
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    for cid, s in scores.items():
        by_id[cid].score = s

    return sorted(by_id.values(), key=lambda c: c.score, reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Recency boost (version awareness)
# ---------------------------------------------------------------------------
def _apply_recency_boost(
    chunks: list[RetrievedChunk],
    *, boost: float = 0.05,
) -> list[RetrievedChunk]:
    """For each company, boost the chunk-set from the most recent doc_date."""
    if not chunks:
        return chunks
    latest_by_company: dict[str, date] = {}
    for c in chunks:
        if c.company and c.doc_date:
            cur = latest_by_company.get(c.company)
            if not cur or c.doc_date > cur:
                latest_by_company[c.company] = c.doc_date
    for c in chunks:
        if c.company and c.doc_date and latest_by_company.get(c.company) == c.doc_date:
            c.score += boost
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    *,
    filters: RetrievalFilters | None = None,
    top_k: int | None = None,
    candidate_k: int | None = None,
) -> list[RetrievedChunk]:
    """Hybrid retrieve: dense cosine ∪ lexical, RRF-fused, optionally recency-boosted."""
    filters = filters or RetrievalFilters()
    top_k = top_k or settings.retrieval_top_k
    candidate_k = candidate_k or settings.retrieval_candidate_k

    embedder = get_embedder()
    qv = embedder.embed_one(query)
    tokens = _keyword_tokens(query)

    with get_connection() as conn:
        cur = conn.cursor()
        dense_rows = _dense_retrieve(cur, query_vec=qv, filters=filters, k=candidate_k)
        lexical_rows = _lexical_retrieve(cur, tokens=tokens, filters=filters, k=candidate_k)
        cur.close()

    fused = _rrf_fuse(dense_rows, lexical_rows, top_k=top_k)

    # Apply recency boost if the user explicitly asked for it OR the query
    # contains a recency-intent keyword ("latest", "current", "now", etc.).
    # We only promote — never override — the explicit user choice.
    apply_recency = filters.prefer_recent or has_recency_intent(query)
    if apply_recency:
        fused = _apply_recency_boost(fused)
    return fused

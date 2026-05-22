"""Multi-source retrieval + RRF fusion (stage 2 of the pipeline).

For each sub-query we hit up to three sources, all keyed back to a CHUNK:
  - dense     : cosine over `propositions.embedding`  -> proposition's chunk_id
  - lexical   : keyword LIKE over `chunks.text`(+footnote) -> chunk_id
  - structured: keyword LIKE over `table_rows.flat_text` and
                `chart_records` (label/value/description) -> chunk_id

Each source yields a ranked list of chunk_ids; we fuse them with Reciprocal
Rank Fusion (rank-based, parameter-free, scale-robust). The fused chunk_ids are
then hydrated with full chunk metadata for the rerank/expansion stages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from rag_system.llm_providers import get_embedder
from rag_system.retrieval.filters import RetrievalFilters
from rag_system.storage.db import get_connection

log = logging.getLogger(__name__)

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "for", "to", "and",
    "or", "in", "on", "at", "by", "with", "what", "which", "who", "how", "why",
    "when", "where", "do", "does", "this", "that", "these", "those", "it", "its",
    "as", "from", "their", "between", "across", "each",
}


@dataclass
class Candidate:
    chunk_id: str
    rrf: float = 0.0
    dense_rank: int | None = None
    lexical_rank: int | None = None
    struct_rank: int | None = None
    matched_text: str = ""          # best snippet that matched (for rerank input)
    sources: set = field(default_factory=set)


def _vec_literal(vec) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _tokens(text: str) -> list[str]:
    toks = [t for t in re.findall(r"[A-Za-z0-9.]{2,}", (text or "").lower()) if t not in _STOP]
    caps = re.findall(r"\b[A-Z]{2,}\b", text or "")
    for c in caps:
        if c.lower() not in toks:
            toks.append(c)
    return toks[:20]


def _filter_sql(f: RetrievalFilters, alias: str) -> tuple[str, list]:
    clauses, params = [], []
    if f.doc_ids:
        clauses.append(f"{alias}.doc_id IN ({','.join(['%s'] * len(f.doc_ids))})")
        params += f.doc_ids
    if f.companies:
        clauses.append(f"{alias}.company IN ({','.join(['%s'] * len(f.companies))})")
        params += f.companies
    if f.date_from:
        clauses.append(f"{alias}.as_of_date >= %s"); params.append(f.date_from)
    if f.date_to:
        clauses.append(f"{alias}.as_of_date <= %s"); params.append(f.date_to)
    return (" AND ".join(clauses), params)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def _dense(cur, query_vec, filters, k) -> list[tuple[str, str]]:
    """cosine over propositions -> [(chunk_id, prop_text)] best-per-chunk."""
    where, params = _filter_sql(filters, "p")
    where_sql = f"WHERE {where}" if where else ""
    sql = f"""
        SELECT p.chunk_id, p.text,
               VECTOR_COSINE_SIMILARITY(p.embedding, {_vec_literal(query_vec)}::VECTOR(FLOAT,768)) AS s
        FROM propositions p
        {where_sql}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY p.chunk_id ORDER BY s DESC) = 1
        ORDER BY s DESC
        LIMIT {int(k)}
    """
    cur.execute(sql, params)
    return [(r[0], r[1]) for r in cur.fetchall()]


def _dense_chunks(cur, query_vec, filters, k) -> list[tuple[str, str]]:
    """cosine over chunks.embedding — covers ALL chunk types (prose, table,
    chart, figure). Propositions only exist for prose, so this is what makes
    chart/table/figure slides semantically findable (closes the index hole)."""
    where, params = _filter_sql(filters, "c")
    where_sql = f"WHERE {where}" if where else ""
    sql = f"""
        SELECT c.chunk_id, c.text,
               VECTOR_COSINE_SIMILARITY(c.embedding, {_vec_literal(query_vec)}::VECTOR(FLOAT,768)) AS s
        FROM chunks c
        {where_sql}
        ORDER BY s DESC
        LIMIT {int(k)}
    """
    cur.execute(sql, params)
    return [(r[0], r[1]) for r in cur.fetchall()]


def _lexical(cur, tokens, filters, k) -> list[tuple[str, str]]:
    if not tokens:
        return []
    where, params = _filter_sql(filters, "c")
    score = " + ".join(["IFF(LOWER(c.text) LIKE %s, 1, 0)"] * len(tokens))
    like_params = [f"%{t.lower()}%" for t in tokens]
    extra = f"AND {where}" if where else ""
    sql = f"""
        SELECT c.chunk_id, c.text, ({score}) AS sc
        FROM chunks c
        WHERE ({score.replace(' + ', ' + ')}) > 0 {extra}
        ORDER BY sc DESC
        LIMIT {int(k)}
    """
    # score appears twice (SELECT + WHERE) -> params twice, then filter params
    cur.execute(sql, like_params + like_params + params)
    return [(r[0], r[1]) for r in cur.fetchall()]


def _structured(cur, tokens, filters, k) -> list[tuple[str, str]]:
    """keyword match over table_rows.flat_text + chart_records text -> chunk_id."""
    if not tokens:
        return []
    out: list[tuple[str, str]] = []

    # table_rows
    twhere, tparams = _filter_sql(filters, "t")
    tscore = " + ".join(["IFF(LOWER(t.flat_text) LIKE %s, 1, 0)"] * len(tokens))
    like = [f"%{t.lower()}%" for t in tokens]
    textra = f"AND {twhere}" if twhere else ""
    cur.execute(f"""
        SELECT t.chunk_id, t.flat_text, ({tscore}) AS sc
        FROM table_rows t
        WHERE ({tscore}) > 0 {textra}
        ORDER BY sc DESC LIMIT {int(k)}
    """, like + like + tparams)
    out += [(r[0], r[1]) for r in cur.fetchall()]

    # chart_records (label/value/description)
    cwhere, cparams = _filter_sql(filters, "cr")
    cfield = "LOWER(COALESCE(cr.label,'')||' '||COALESCE(cr.value,'')||' '||COALESCE(cr.description,''))"
    cscore = " + ".join([f"IFF({cfield} LIKE %s, 1, 0)"] * len(tokens))
    cextra = f"AND {cwhere}" if cwhere else ""
    cur.execute(f"""
        SELECT cr.chunk_id, COALESCE(cr.description, cr.label||': '||cr.value) AS txt, ({cscore}) AS sc
        FROM chart_records cr
        WHERE ({cscore}) > 0 {cextra}
        ORDER BY sc DESC LIMIT {int(k)}
    """, like + like + cparams)
    out += [(r[0], r[1]) for r in cur.fetchall() if r[0]]
    return out[: int(k) * 2]


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------
def _rrf_add(cands: dict, ranked: list[tuple[str, str]], *, source: str, rrf_k: int = 60):
    for rank, (chunk_id, text) in enumerate(ranked, start=1):
        if not chunk_id:
            continue
        c = cands.get(chunk_id)
        if c is None:
            c = Candidate(chunk_id=chunk_id)
            cands[chunk_id] = c
        c.rrf += 1.0 / (rrf_k + rank)
        c.sources.add(source)
        if source == "dense" and c.dense_rank is None:
            c.dense_rank = rank
        elif source == "lexical" and c.lexical_rank is None:
            c.lexical_rank = rank
        elif source == "structured" and c.struct_rank is None:
            c.struct_rank = rank
        if not c.matched_text and text:
            c.matched_text = text[:500]


def multi_source_retrieve(
    sub_queries: list[str],
    *,
    filters: RetrievalFilters,
    candidate_k: int = 40,
    use_tables: bool = True,
) -> list[Candidate]:
    """Run dense+lexical+structured for each sub-query, fuse with RRF."""
    embedder = get_embedder()
    cands: dict[str, Candidate] = {}
    with get_connection() as conn:
        cur = conn.cursor()
        for sq in sub_queries:
            qv = embedder.embed_one(sq)
            toks = _tokens(sq)
            _rrf_add(cands, _dense(cur, qv, filters, candidate_k), source="dense")          # propositions
            _rrf_add(cands, _dense_chunks(cur, qv, filters, candidate_k), source="dense_chunk")  # all chunks
            _rrf_add(cands, _lexical(cur, toks, filters, candidate_k), source="lexical")
            if use_tables:
                _rrf_add(cands, _structured(cur, toks, filters, candidate_k), source="structured")
        cur.close()
    ranked = sorted(cands.values(), key=lambda c: c.rrf, reverse=True)
    log.info("multi-source: %d unique candidates from %d sub-queries",
             len(ranked), len(sub_queries))
    return ranked


# ---------------------------------------------------------------------------
# Hydrate chunk metadata for the top candidates
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    chunk_id: str
    parent_id: str | None
    doc_id: str
    page_number: int
    chunk_type: str
    text: str
    company: str | None
    doc_type: str | None
    doc_date: date | None
    as_of_date: date | None
    doc_family_id: str | None
    version_label: str | None
    slide_title: str | None
    confidence: float | None
    rrf: float = 0.0
    dense_rank: int | None = None
    lexical_rank: int | None = None
    struct_rank: int | None = None
    rerank_score: float | None = None
    score: float = 0.0
    sources: list = field(default_factory=list)


def hydrate(cands: list[Candidate], *, limit: int) -> list[RetrievedChunk]:
    """Fetch full chunk rows for the top `limit` candidates, preserving RRF order."""
    top = cands[:limit]
    if not top:
        return []
    ids = [c.chunk_id for c in top]
    by_id = {c.chunk_id: c for c in top}
    placeholders = ",".join(["%s"] * len(ids))
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT chunk_id, parent_id, doc_id, page_number, chunk_type, text,
                   company, doc_type, doc_date, as_of_date, doc_family_id,
                   version_label, slide_title, confidence
            FROM chunks WHERE chunk_id IN ({placeholders})
        """, ids)
        rows = {r[0]: r for r in cur.fetchall()}
        cur.close()
    out = []
    for cid in ids:
        r = rows.get(cid)
        if not r:
            continue
        c = by_id[cid]
        out.append(RetrievedChunk(
            chunk_id=r[0], parent_id=r[1], doc_id=r[2], page_number=r[3],
            chunk_type=r[4], text=r[5], company=r[6], doc_type=r[7], doc_date=r[8],
            as_of_date=r[9], doc_family_id=r[10], version_label=r[11], slide_title=r[12],
            confidence=r[13], rrf=c.rrf, dense_rank=c.dense_rank,
            lexical_rank=c.lexical_rank, struct_rank=c.struct_rank,
            score=c.rrf, sources=sorted(c.sources),
        ))
    return out

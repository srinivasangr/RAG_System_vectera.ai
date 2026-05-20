"""Snowflake DAO for the v2 multi-vector schema.

Kept separate from v1 `repository.py` so the working v1 path is untouched.
Writes: documents (v2 columns), parent_chunks, chunks (v2 columns),
propositions, table_rows, chart_records, and ingest_checkpoints.

Vectors are inlined as VECTOR literals (the connector can't bind VECTOR via
pyformat) — fine at our scale.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Iterator, Sequence

from rag_system.storage.db import get_connection

log = logging.getLogger(__name__)


@contextmanager
def _use_connection(conn) -> Iterator:
    if conn is not None:
        yield conn
    else:
        with get_connection() as fresh:
            yield fresh


def _vec(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


# ---------------------------------------------------------------------------
# Checkpoints (per-stage resume)
# ---------------------------------------------------------------------------
def mark_stage(doc_id: str, checksum: str, stage: str, status: str,
               detail: str = "", *, conn=None) -> None:
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM ingest_checkpoints WHERE doc_id=%s AND stage=%s",
                    (doc_id, stage))
        cur.execute(
            """INSERT INTO ingest_checkpoints (doc_id, checksum, stage, status, detail)
               VALUES (%s, %s, %s, %s, %s)""",
            (doc_id, checksum, stage, status, detail[:1000]),
        )
        conn.commit()
        cur.close()


def is_complete(doc_id: str, checksum: str, *, conn=None) -> bool:
    """True if this exact (doc_id, checksum) finished the 'complete' stage."""
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT status FROM ingest_checkpoints
               WHERE doc_id=%s AND checksum=%s AND stage='complete'""",
            (doc_id, checksum),
        )
        row = cur.fetchone()
        cur.close()
    return bool(row and row[0] == "done")


def is_complete_by_checksum(checksum: str, *, conn=None) -> bool:
    """True if any doc with this file checksum finished — lets us skip re-parsing
    a file before we even know its doc_id (which requires parsing)."""
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT status FROM ingest_checkpoints
               WHERE checksum=%s AND stage='complete'""",
            (checksum,),
        )
        row = cur.fetchone()
        cur.close()
    return bool(row and row[0] == "done")


# ---------------------------------------------------------------------------
# Documents (v2)
# ---------------------------------------------------------------------------
def upsert_document_v2(meta, page_count: int, *, conn=None) -> str:
    """Insert/update a documents row with v2 columns. Returns status."""
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.execute("SELECT checksum FROM documents WHERE doc_id=%s", (meta.doc_id,))
        row = cur.fetchone()
        params = (
            meta.source_path, meta.company, meta.ticker, meta.doc_date, meta.doc_type,
            meta.doc_type_conf, meta.version_label, meta.as_of_date, meta.as_of_source,
            meta.doc_family_id, page_count, meta.checksum,
        )
        if row is None:
            cur.execute(
                """INSERT INTO documents
                     (doc_id, source_path, company, ticker, doc_date, doc_type,
                      doc_type_conf, version_label, as_of_date, as_of_source,
                      doc_family_id, page_count, checksum)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (meta.doc_id, *params),
            )
            status = "inserted"
        else:
            cur.execute(
                """UPDATE documents SET
                     source_path=%s, company=%s, ticker=%s, doc_date=%s, doc_type=%s,
                     doc_type_conf=%s, version_label=%s, as_of_date=%s, as_of_source=%s,
                     doc_family_id=%s, page_count=%s, checksum=%s,
                     ingested_at=CURRENT_TIMESTAMP()
                   WHERE doc_id=%s""",
                (*params, meta.doc_id),
            )
            status = "updated"
        conn.commit()
        cur.close()
    return status


def delete_doc_artifacts_v2(doc_id: str, *, conn=None) -> dict:
    """Remove all v2 artifacts for a doc so a re-ingest starts clean."""
    counts = {}
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        for tbl in ("propositions", "table_rows", "chart_records",
                    "chunk_images", "chunks", "parent_images", "parent_chunks"):
            if tbl == "parent_images":
                cur.execute(
                    "DELETE FROM parent_images WHERE parent_id IN "
                    "(SELECT parent_id FROM parent_chunks WHERE doc_id=%s)", (doc_id,))
            elif tbl == "chunk_images":
                cur.execute(
                    "DELETE FROM chunk_images WHERE chunk_id IN "
                    "(SELECT chunk_id FROM chunks WHERE doc_id=%s)", (doc_id,))
            else:
                cur.execute(f"DELETE FROM {tbl} WHERE doc_id=%s", (doc_id,))
            counts[tbl] = cur.rowcount or 0
        conn.commit()
        cur.close()
    return counts


# ---------------------------------------------------------------------------
# Parent chunks
# ---------------------------------------------------------------------------
def insert_parent_chunks(parents, *, conn=None) -> int:
    parents = list(parents)
    if not parents:
        return 0
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        for p in parents:
            cur.execute(
                """INSERT INTO parent_chunks
                     (parent_id, doc_id, page_number, slide_title, text, token_count,
                      company, doc_type, doc_date, as_of_date, doc_family_id, version_label)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (p.parent_id, p.doc_id, p.page_number, p.slide_title, p.text,
                 p.token_count, p.company, p.doc_type, p.doc_date, p.as_of_date,
                 p.doc_family_id, p.version_label),
            )
        conn.commit()
        cur.close()
    return len(parents)


# ---------------------------------------------------------------------------
# Child chunks (v2)
# ---------------------------------------------------------------------------
def insert_children_v2(children, embeddings: list[list[float]], *, conn=None) -> int:
    children = list(children)
    if len(children) != len(embeddings):
        raise ValueError(f"children={len(children)} != embeddings={len(embeddings)}")
    if not children:
        return 0
    dim = len(embeddings[0])
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        for ch, vec in zip(children, embeddings):
            cur.execute(
                f"""INSERT INTO chunks
                     (chunk_id, doc_id, parent_id, page_number, chunk_index, text,
                      token_count, chunk_type, embedding, company, doc_date,
                      version_label, footnote_text, qualifier_text, doc_type,
                      as_of_date, doc_family_id, slide_title)
                   SELECT %s,%s,%s,%s,%s,%s,%s,%s,
                          {_vec(vec)}::VECTOR(FLOAT,{dim}),
                          %s,%s,%s,%s,%s,%s,%s,%s,%s""",
                (ch.chunk_id, ch.doc_id, ch.parent_id, ch.page_number, ch.chunk_index,
                 ch.text, ch.token_count, ch.chunk_type, ch.company, ch.doc_date,
                 ch.version_label, ch.footnote_text, ch.qualifier_text, ch.doc_type,
                 ch.as_of_date, ch.doc_family_id, ch.slide_title),
            )
        conn.commit()
        cur.close()
    return len(children)


# ---------------------------------------------------------------------------
# Propositions
# ---------------------------------------------------------------------------
def insert_propositions(props, embeddings: list[list[float]], *, conn=None) -> int:
    props = list(props)
    if len(props) != len(embeddings):
        raise ValueError(f"props={len(props)} != embeddings={len(embeddings)}")
    if not props:
        return 0
    dim = len(embeddings[0])
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        for p, vec in zip(props, embeddings):
            cur.execute(
                f"""INSERT INTO propositions
                     (prop_id, chunk_id, parent_id, doc_id, page_number, text,
                      embedding, company, doc_type, doc_date, as_of_date,
                      doc_family_id, version_label)
                   SELECT %s,%s,%s,%s,%s,%s,
                          {_vec(vec)}::VECTOR(FLOAT,{dim}),
                          %s,%s,%s,%s,%s,%s""",
                (p["prop_id"], p["chunk_id"], p["parent_id"], p["doc_id"],
                 p["page_number"], p["text"], p["company"], p["doc_type"],
                 p["doc_date"], p["as_of_date"], p["doc_family_id"], p["version_label"]),
            )
        conn.commit()
        cur.close()
    return len(props)


# ---------------------------------------------------------------------------
# Table rows
# ---------------------------------------------------------------------------
def insert_table_rows(rows, embeddings: list[list[float]], *, conn=None) -> int:
    rows = list(rows)
    if len(rows) != len(embeddings):
        raise ValueError(f"rows={len(rows)} != embeddings={len(embeddings)}")
    if not rows:
        return 0
    dim = len(embeddings[0])
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        for r, vec in zip(rows, embeddings):
            cur.execute(
                f"""INSERT INTO table_rows
                     (row_id, chunk_id, doc_id, page_number, table_id, row_idx,
                      columns_json, flat_text, embedding, company, doc_type,
                      doc_date, as_of_date, doc_family_id)
                   SELECT %s,%s,%s,%s,%s,%s,
                          PARSE_JSON(%s),%s,
                          {_vec(vec)}::VECTOR(FLOAT,{dim}),
                          %s,%s,%s,%s,%s""",
                (r.row_id, r.chunk_id, r.doc_id, r.page_number, r.table_id, r.row_idx,
                 json.dumps(r.columns), r.flat_text, r.company, r.doc_type,
                 r.doc_date, r.as_of_date, r.doc_family_id),
            )
        conn.commit()
        cur.close()
    return len(rows)


# ---------------------------------------------------------------------------
# Chart records (no embedding — retrieved by company + label/value match)
# ---------------------------------------------------------------------------
def list_documents(*, conn=None) -> list[dict]:
    """List ingested documents with per-doc artifact counts (for the UI)."""
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT d.doc_id, d.company, d.ticker, d.doc_type, d.as_of_date,
                      d.doc_date, d.version_label, d.doc_family_id, d.page_count,
                      d.ingested_at,
                      (SELECT COUNT(*) FROM chunks       c WHERE c.doc_id=d.doc_id),
                      (SELECT COUNT(*) FROM parent_chunks p WHERE p.doc_id=d.doc_id),
                      (SELECT COUNT(*) FROM propositions  pr WHERE pr.doc_id=d.doc_id),
                      (SELECT COUNT(*) FROM table_rows    t WHERE t.doc_id=d.doc_id),
                      (SELECT COUNT(*) FROM chart_records cr WHERE cr.doc_id=d.doc_id)
               FROM documents d
               ORDER BY d.ingested_at DESC NULLS LAST, d.company"""
        )
        cols = ["doc_id", "company", "ticker", "doc_type", "as_of_date", "doc_date",
                "version_label", "doc_family_id", "page_count", "ingested_at",
                "chunks", "parents", "propositions", "table_rows", "chart_records"]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.close()
    # JSON-friendly dates
    for r in rows:
        for k in ("as_of_date", "doc_date", "ingested_at"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


def corpus_profile(*, conn=None) -> dict:
    """Runtime corpus profile — what the corpus actually contains. Fed to the
    router (domain-agnostic) and shown in the UI."""
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT doc_type FROM documents WHERE doc_type IS NOT NULL")
        doc_types = sorted(r[0] for r in cur.fetchall())
        cur.execute("SELECT DISTINCT company FROM documents WHERE company IS NOT NULL")
        entities = sorted(r[0] for r in cur.fetchall())
        cur.execute("SELECT MIN(as_of_date), MAX(as_of_date) FROM documents")
        lo, hi = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM documents")
        n_docs = cur.fetchone()[0]
        cur.close()
    return {
        "n_documents": n_docs,
        "doc_types": doc_types,
        "entities": entities,
        "date_range": [str(lo) if lo else None, str(hi) if hi else None],
    }


def delete_document_v2(doc_id: str, *, conn=None) -> dict:
    """Delete a document and all its v2 artifacts."""
    with _use_connection(conn) as conn:
        counts = delete_doc_artifacts_v2(doc_id, conn=conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))
        counts["documents"] = cur.rowcount or 0
        cur.execute("DELETE FROM ingest_checkpoints WHERE doc_id=%s", (doc_id,))
        conn.commit()
        cur.close()
    return counts


def insert_chart_records(records, *, conn=None) -> int:
    records = list(records)
    if not records:
        return 0
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        for r in records:
            cur.execute(
                """INSERT INTO chart_records
                     (record_id, chunk_id, doc_id, page_number, chart_id, chart_kind,
                      label, value, unit, bbox, confidence, vision_model,
                      company, doc_type, doc_date, as_of_date, doc_family_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (r.record_id, r.chunk_id, r.doc_id, r.page_number, r.chart_id,
                 r.chart_kind, r.label, r.value, r.unit, r.bbox, r.confidence,
                 r.vision_model, r.company, r.doc_type, r.doc_date, r.as_of_date,
                 r.doc_family_id),
            )
        conn.commit()
        cur.close()
    return len(records)

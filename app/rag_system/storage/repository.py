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


def _batched_vector_insert(cur, *, table, col_names, col_exprs, rows, embeddings,
                           dim, batch=40):
    """Insert rows that include one VECTOR column, in batches.

    VECTOR literals can't be bound, so each row is a `SELECT <exprs>, [v]::VECTOR`
    and we UNION ALL up to `batch` of them into one INSERT — turning N network
    round-trips into N/batch.

    col_names : full column list, vector column LAST.
    col_exprs : SQL expr per scalar column (each containing exactly one %s),
                e.g. "%s" or "PARSE_JSON(%s)". len = len(col_names) - 1.
    rows      : list of scalar param-tuples (one per row, in col_exprs order).
    embeddings: parallel list of vectors.
    """
    assert len(rows) == len(embeddings)
    cols_sql = "(" + ", ".join(col_names) + ")"
    for i in range(0, len(rows), batch):
        chunk_rows = rows[i:i + batch]
        chunk_vecs = embeddings[i:i + batch]
        selects, params = [], []
        for scalars, vec in zip(chunk_rows, chunk_vecs):
            selects.append(
                "SELECT " + ", ".join(col_exprs)
                + f", {_vec(vec)}::VECTOR(FLOAT,{dim})"
            )
            params.extend(scalars)
        sql = f"INSERT INTO {table} {cols_sql} " + " UNION ALL ".join(selects)
        cur.execute(sql, params)


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
def delete_doc_artifacts(doc_id: str, *, conn=None) -> dict:
    """Remove all v2 artifacts for a doc so a re-ingest starts clean."""
    counts = {}
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        for tbl in ("propositions", "table_rows", "chart_records",
                    "chunks", "page_images", "document_files", "parent_chunks"):
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
        cur.executemany(
            """INSERT INTO parent_chunks
                 (parent_id, doc_id, page_number, slide_title, text, token_count,
                  company, doc_type, doc_date, as_of_date, doc_family_id, version_label)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(p.parent_id, p.doc_id, p.page_number, p.slide_title, p.text,
              p.token_count, p.company, p.doc_type, p.doc_date, p.as_of_date,
              p.doc_family_id, p.version_label) for p in parents],
        )
        conn.commit()
        cur.close()
    return len(parents)


# ---------------------------------------------------------------------------
# Child chunks (v2)
# ---------------------------------------------------------------------------
def insert_children(children, embeddings: list[list[float]], *, conn=None) -> int:
    children = list(children)
    if len(children) != len(embeddings):
        raise ValueError(f"children={len(children)} != embeddings={len(embeddings)}")
    if not children:
        return 0
    dim = len(embeddings[0])
    # Note column order: embedding is placed LAST (the vector column).
    col_names = [
        "chunk_id", "doc_id", "parent_id", "page_number", "chunk_index", "text",
        "token_count", "chunk_type", "company", "doc_date", "version_label",
        "footnote_text", "qualifier_text", "doc_type", "as_of_date", "doc_family_id",
        "slide_title", "confidence", "kind_detail", "embedding",
    ]
    col_exprs = ["%s"] * (len(col_names) - 1)
    rows = [
        (ch.chunk_id, ch.doc_id, ch.parent_id, ch.page_number, ch.chunk_index,
         ch.text, ch.token_count, ch.chunk_type, ch.company, ch.doc_date,
         ch.version_label, ch.footnote_text, ch.qualifier_text, ch.doc_type,
         ch.as_of_date, ch.doc_family_id, ch.slide_title,
         getattr(ch, "confidence", None), getattr(ch, "kind_detail", None))
        for ch in children
    ]
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        _batched_vector_insert(cur, table="chunks", col_names=col_names,
                               col_exprs=col_exprs, rows=rows, embeddings=embeddings, dim=dim)
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
    col_names = [
        "prop_id", "chunk_id", "parent_id", "doc_id", "page_number", "text",
        "company", "doc_type", "doc_date", "as_of_date", "doc_family_id",
        "version_label", "embedding",
    ]
    col_exprs = ["%s"] * (len(col_names) - 1)
    rows = [
        (p["prop_id"], p["chunk_id"], p["parent_id"], p["doc_id"], p["page_number"],
         p["text"], p["company"], p["doc_type"], p["doc_date"], p["as_of_date"],
         p["doc_family_id"], p["version_label"])
        for p in props
    ]
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        _batched_vector_insert(cur, table="propositions", col_names=col_names,
                               col_exprs=col_exprs, rows=rows, embeddings=embeddings, dim=dim)
        conn.commit()
        cur.close()
    return len(props)


# ---------------------------------------------------------------------------
# Table rows
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


def delete_document(doc_id: str, *, conn=None) -> dict:
    """Delete a document and all its v2 artifacts."""
    with _use_connection(conn) as conn:
        counts = delete_doc_artifacts(doc_id, conn=conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))
        counts["documents"] = cur.rowcount or 0
        cur.execute("DELETE FROM ingest_checkpoints WHERE doc_id=%s", (doc_id,))
        conn.commit()
        cur.close()
    return counts


def upsert_document(meta, file_meta: dict, page_count: int,
                       stored_pdf_path: str, *, conn=None) -> str:
    """Insert/update a documents row with identity (meta) + file metadata."""
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.execute("SELECT checksum FROM documents WHERE doc_id=%s", (meta.doc_id,))
        exists = cur.fetchone() is not None
        cols_vals = {
            "source_path": stored_pdf_path,           # relative, not absolute
            "company": meta.company, "ticker": meta.ticker,
            "doc_date": meta.doc_date, "doc_type": meta.doc_type,
            "doc_type_conf": meta.doc_type_conf, "version_label": meta.version_label,
            "as_of_date": meta.as_of_date, "as_of_source": meta.as_of_source,
            "doc_family_id": meta.doc_family_id, "page_count": page_count,
            "checksum": meta.checksum,
            "original_filename": file_meta.get("original_filename"),
            "stored_pdf_path": stored_pdf_path,
            "file_size_bytes": file_meta.get("file_size_bytes"),
            "mime_type": file_meta.get("mime_type"),
            "pdf_author": file_meta.get("pdf_author"),
            "pdf_title": file_meta.get("pdf_title"),
            "pdf_created": file_meta.get("pdf_created"),
            "pdf_page_count": file_meta.get("pdf_page_count"),
        }
        if exists:
            sets = ", ".join(f"{k}=%s" for k in cols_vals)
            cur.execute(
                f"UPDATE documents SET {sets}, ingested_at=CURRENT_TIMESTAMP() "
                f"WHERE doc_id=%s",
                (*cols_vals.values(), meta.doc_id),
            )
            status = "updated"
        else:
            cols = ", ".join(["doc_id", *cols_vals.keys()])
            ph = ", ".join(["%s"] * (1 + len(cols_vals)))
            cur.execute(
                f"INSERT INTO documents ({cols}) VALUES ({ph})",
                (meta.doc_id, *cols_vals.values()),
            )
            status = "inserted"
        conn.commit()
        cur.close()
    return status


def insert_document_file(doc_id: str, filename: str, mime: str,
                         size_bytes: int, content_b64: str, *, conn=None) -> None:
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM document_files WHERE doc_id=%s", (doc_id,))
        cur.execute(
            """INSERT INTO document_files (doc_id, filename, mime, size_bytes, content_b64)
               VALUES (%s,%s,%s,%s,%s)""",
            (doc_id, filename, mime, size_bytes, content_b64),
        )
        conn.commit()
        cur.close()


def insert_page_images(rows, *, conn=None) -> int:
    """rows: iterable of dicts {parent_id, doc_id, page_number, width, height, image_b64}."""
    rows = list(rows)
    if not rows:
        return 0
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.executemany(
            """INSERT INTO page_images
                 (parent_id, doc_id, page_number, width, height, mime_type, image_b64)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            [(r["parent_id"], r["doc_id"], r["page_number"], r.get("width"),
              r.get("height"), r.get("mime_type", "image/png"), r["image_b64"])
             for r in rows],
        )
        conn.commit()
        cur.close()
    return len(rows)


def insert_chart_records(records, *, conn=None) -> int:
    """records: iterable of dicts with keys matching chart_records (+ description)."""
    records = list(records)
    if not records:
        return 0
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.executemany(
            """INSERT INTO chart_records
                 (record_id, chunk_id, doc_id, page_number, chart_id, chart_kind,
                  label, value, unit, bbox, confidence, vision_model, description,
                  company, doc_type, doc_date, as_of_date, doc_family_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(r["record_id"], r.get("chunk_id"), r["doc_id"], r.get("page_number"),
              r.get("chart_id"), r.get("chart_kind"), r.get("label", ""),
              r.get("value", ""), r.get("unit", ""), r.get("bbox", ""),
              r.get("confidence", 0.0), r.get("vision_model", ""),
              r.get("description", ""), r.get("company"), r.get("doc_type"),
              r.get("doc_date"), r.get("as_of_date"), r.get("doc_family_id"))
             for r in records],
        )
        conn.commit()
        cur.close()
    return len(records)


def log_query(*, question, answer, intent, sub_queries, retrieved_ids,
                 retrieval_stages, conflict_pairs, provider_chain, llm_provider,
                 llm_model, total_latency_ms, doc_ids=None, conn=None) -> str:
    """Append one row to query_log with the v2 trace columns. Best-effort —
    callers wrap in try/except so logging never breaks a query."""
    import json as _json
    import uuid as _uuid
    qid = _uuid.uuid4().hex
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO query_log
                 (query_id, question, filters, retrieved_ids, answer,
                  llm_provider, llm_model, latency_ms, router_intent, sub_queries,
                  retrieval_stages, conflict_pairs, provider_chain, total_latency_ms)
               SELECT %s, %s, PARSE_JSON(%s), PARSE_JSON(%s)::ARRAY, %s,
                      %s, %s, %s, %s, PARSE_JSON(%s),
                      PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), %s""",
            (qid, question, _json.dumps({"doc_ids": list(doc_ids or [])}),
             _json.dumps(list(retrieved_ids or [])), answer,
             llm_provider, llm_model, total_latency_ms, intent,
             _json.dumps(list(sub_queries or [])),
             _json.dumps(retrieval_stages or {}),
             _json.dumps(conflict_pairs or []),
             _json.dumps(provider_chain or []),
             total_latency_ms),
        )
        conn.commit()
        cur.close()
    return qid


def recent_queries(limit: int = 50, *, conn=None) -> list[dict]:
    """Recent query_log entries for the History tab."""
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT query_id, question, answer, router_intent, llm_provider,
                      llm_model, total_latency_ms, created_at
               FROM query_log
               ORDER BY created_at DESC NULLS LAST
               LIMIT %s""", (limit,))
        cols = ["query_id", "question", "answer", "intent", "llm_provider",
                "llm_model", "total_latency_ms", "created_at"]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.close()
    for r in rows:
        if r.get("created_at") is not None:
            r["created_at"] = str(r["created_at"])
    return rows


def get_page_image(parent_id: str, *, conn=None):
    """Return (mime_type, image_b64) for a page, or None."""
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.execute("SELECT mime_type, image_b64 FROM page_images WHERE parent_id=%s",
                    (parent_id,))
        row = cur.fetchone()
        cur.close()
    return (row[0], row[1]) if row and row[1] else None


def insert_table_rows(rows, embeddings, *, conn=None) -> int:
    """rows: list of dicts; embeddings parallel list. Writes table_rows."""
    rows = list(rows)
    if len(rows) != len(embeddings):
        raise ValueError(f"rows={len(rows)} != embeddings={len(embeddings)}")
    if not rows:
        return 0
    dim = len(embeddings[0])
    # columns_json needs PARSE_JSON(%s); embedding (vector) is placed LAST.
    col_names = [
        "row_id", "chunk_id", "doc_id", "page_number", "table_id", "row_idx",
        "columns_json", "flat_text", "company", "doc_type", "doc_date",
        "as_of_date", "doc_family_id", "embedding",
    ]
    col_exprs = ["%s", "%s", "%s", "%s", "%s", "%s", "PARSE_JSON(%s)", "%s",
                 "%s", "%s", "%s", "%s", "%s"]
    scalar_rows = [
        (r["row_id"], r.get("chunk_id"), r["doc_id"], r.get("page_number"),
         r.get("table_id"), r.get("row_idx"), json.dumps(r.get("columns", {})),
         r.get("flat_text", ""), r.get("company"), r.get("doc_type"),
         r.get("doc_date"), r.get("as_of_date"), r.get("doc_family_id"))
        for r in rows
    ]
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        _batched_vector_insert(cur, table="table_rows", col_names=col_names,
                               col_exprs=col_exprs, rows=scalar_rows,
                               embeddings=embeddings, dim=dim)
        conn.commit()
        cur.close()
    return len(rows)

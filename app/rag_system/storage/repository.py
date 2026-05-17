"""Snowflake DAO — idempotent upserts for documents + chunks, query log writes.

We use 'MERGE INTO' for idempotency: re-running ingestion on the same PDF
(same checksum) is a no-op. Re-running on a *changed* file updates the row.

Each public function accepts an optional `conn` argument so a caller (e.g. the
ingest pipeline) can reuse one Snowflake connection across many calls. When
`conn` is None we open and close one ourselves.
"""

from __future__ import annotations

import base64
import json
import uuid
from contextlib import contextmanager, nullcontext
from datetime import date
from typing import Iterable, Iterator, Sequence

from rag_system.ingest.chunk import Chunk
from rag_system.ingest.metadata import DocMeta
from rag_system.storage.db import get_connection


@contextmanager
def _use_connection(conn) -> Iterator:
    """Yield `conn` if given, else open + close a fresh one."""
    if conn is not None:
        yield conn
    else:
        with get_connection() as fresh:
            yield fresh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vec_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def upsert_document(meta: DocMeta, page_count: int, *, conn=None) -> str:
    """Insert or update a document row by doc_id. Returns 'inserted' or 'updated'."""
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        # Check existence and current checksum
        cur.execute("SELECT checksum FROM documents WHERE doc_id = %s", (meta.doc_id,))
        row = cur.fetchone()

        if row is None:
            cur.execute(
                """
                INSERT INTO documents
                  (doc_id, source_path, company, doc_date, doc_type,
                   version_label, page_count, checksum)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    meta.doc_id, meta.source_path, meta.company, meta.doc_date,
                    meta.doc_type, meta.version_label, page_count, meta.checksum,
                ),
            )
            status = "inserted"
        else:
            if row[0] == meta.checksum:
                status = "unchanged"
            else:
                cur.execute(
                    """
                    UPDATE documents
                    SET source_path=%s, company=%s, doc_date=%s, doc_type=%s,
                        version_label=%s, page_count=%s, checksum=%s,
                        ingested_at=CURRENT_TIMESTAMP()
                    WHERE doc_id=%s
                    """,
                    (
                        meta.source_path, meta.company, meta.doc_date, meta.doc_type,
                        meta.version_label, page_count, meta.checksum, meta.doc_id,
                    ),
                )
                status = "updated"
        conn.commit()
        cur.close()
    return status


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------
def delete_chunks_for_doc(doc_id: str, *, conn=None) -> int:
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM chunk_images "
            "WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = %s)",
            (doc_id,),
        )
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        n = cur.rowcount
        conn.commit()
        cur.close()
    return n or 0


def insert_chunks(
    chunks: Iterable[Chunk],
    embeddings: list[list[float]],
    *,
    conn=None,
) -> int:
    """Bulk-insert chunks with their embeddings. Lists must be the same length.

    We use one INSERT per row because the VECTOR cast can't be parameterized
    via Snowflake's bind protocol — vectors are inlined as a literal. This is
    fine at our scale (~hundreds of chunks per doc).
    """
    chunks = list(chunks)
    if len(chunks) != len(embeddings):
        raise ValueError(f"chunks={len(chunks)} != embeddings={len(embeddings)}")
    if not chunks:
        return 0

    dim = len(embeddings[0])
    with _use_connection(conn) as conn:
        cur = conn.cursor()
        for ch, vec in zip(chunks, embeddings):
            if len(vec) != dim:
                raise ValueError(f"inconsistent embedding dim: {len(vec)} vs {dim}")
            vec_lit = _vec_literal(vec)
            cur.execute(
                f"""
                INSERT INTO chunks
                  (chunk_id, doc_id, page_number, chunk_index, text, token_count,
                   chunk_type, embedding, company, doc_date, version_label)
                SELECT %s, %s, %s, %s, %s, %s, %s,
                       {vec_lit}::VECTOR(FLOAT, {dim}),
                       %s, %s, %s
                """,
                (
                    ch.chunk_id, ch.doc_id, ch.page_number, ch.chunk_index,
                    ch.text, ch.token_count, ch.chunk_type,
                    ch.company, ch.doc_date, ch.version_label,
                ),
            )
            # If this chunk carries an image (chart_description), store it too
            if ch.image_png_bytes:
                b64 = base64.b64encode(ch.image_png_bytes).decode("ascii")
                cur.execute(
                    """
                    INSERT INTO chunk_images
                      (chunk_id, width, height, mime_type, image_b64)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (ch.chunk_id, ch.image_width, ch.image_height, "image/png", b64),
                )
        conn.commit()
        cur.close()
    return len(chunks)


def get_chunk_image_b64(chunk_id: str) -> str | None:
    """Fetch the base64 image for a chunk (used by the UI to render charts)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT image_b64 FROM chunk_images WHERE chunk_id = %s",
            (chunk_id,),
        )
        row = cur.fetchone()
        cur.close()
    return row[0] if row else None


def delete_document(doc_id: str) -> dict:
    """Delete a document and all its chunks + chunk images from Snowflake.

    Returns counts of rows removed. Operates inside one transaction so an
    error mid-way doesn't leave dangling rows.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("BEGIN")
        try:
            cur.execute(
                "DELETE FROM chunk_images WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE doc_id = %s)",
                (doc_id,),
            )
            n_imgs = cur.rowcount or 0
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            n_chunks = cur.rowcount or 0
            cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            n_docs = cur.rowcount or 0
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        finally:
            cur.close()
    return {"documents": n_docs, "chunks": n_chunks, "images": n_imgs}


# ---------------------------------------------------------------------------
# Query log (used by retrieval path, defined here so all DAO lives together)
# ---------------------------------------------------------------------------
def log_query(
    *,
    question: str,
    filters: dict,
    retrieved_ids: list[str],
    answer: str,
    llm_provider: str,
    llm_model: str,
    latency_ms: int,
) -> str:
    qid = str(uuid.uuid4())
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO query_log
              (query_id, question, filters, retrieved_ids, answer,
               llm_provider, llm_model, latency_ms)
            SELECT %s, %s, PARSE_JSON(%s), %s, %s, %s, %s, %s
            """,
            (
                qid, question, json.dumps(filters), retrieved_ids, answer,
                llm_provider, llm_model, latency_ms,
            ),
        )
        conn.commit()
        cur.close()
    return qid


# ---------------------------------------------------------------------------
# Stats (handy for the UI / debugging)
# ---------------------------------------------------------------------------
def corpus_stats() -> dict:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents")
        n_docs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        n_chunks = cur.fetchone()[0]
        cur.execute(
            "SELECT company, COUNT(DISTINCT doc_id), COUNT(*) "
            "FROM chunks GROUP BY company ORDER BY company"
        )
        per_company = [(c, d, k) for c, d, k in cur.fetchall()]
        cur.close()
    return {"documents": n_docs, "chunks": n_chunks, "per_company": per_company}

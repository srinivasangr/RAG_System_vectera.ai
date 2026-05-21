"""v3 DAO additions: file/PDF metadata on documents, the raw PDF
(document_files), per-page images (page_images), and chart_records with
descriptions. Builds on repository_v2 (which still handles parents/children/
propositions/table_rows/checkpoints)."""

from __future__ import annotations

import json
from rag_system.storage.repository_v2 import (  # reuse helpers
    _batched_vector_insert, _use_connection, _vec,
)


def upsert_document_v3(meta, file_meta: dict, page_count: int,
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


def insert_chart_records_v3(records, *, conn=None) -> int:
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


def insert_table_rows_v3(rows, embeddings, *, conn=None) -> int:
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

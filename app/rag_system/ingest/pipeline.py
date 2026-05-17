"""End-to-end ingestion: parse → vision → chunk → embed → upsert.

Usage:
  python -m rag_system.ingest.pipeline                    # all PDFs
  python -m rag_system.ingest.pipeline --limit 1          # first PDF only
  python -m rag_system.ingest.pipeline --doc <basename>   # one PDF by name
  python -m rag_system.ingest.pipeline --no-vision        # skip image descriptions
  python -m rag_system.ingest.pipeline --dry-run          # parse+chunk, don't write to Snowflake
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from rag_system.config import settings
from rag_system.ingest.chunk import Chunk, chunk_page
from rag_system.ingest.metadata import DocMeta, extract_metadata
from rag_system.ingest.parse import ParsedDocument, parse_pdf
from rag_system.ingest.vision_extract import describe_images
from rag_system.llm_providers import get_embedder
from rag_system.storage.db import get_connection
from rag_system.storage.repository import (
    delete_chunks_for_doc,
    insert_chunks,
    upsert_document,
)

log = logging.getLogger(__name__)


def _build_chunks(
    meta: DocMeta,
    parsed: ParsedDocument,
    *,
    with_vision: bool,
    vision_call_budget: int | None,
    progress_cb=None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    remaining_budget = vision_call_budget

    # Run vision over ALL pages' images in one parallel pass (faster + lets us
    # emit a single coherent vision progress stream). Then chunk per page.
    all_images = []
    for page in parsed.pages:
        all_images.extend(page.images)

    descs_all: dict = {}
    if with_vision and all_images:
        descs_all = describe_images(
            all_images,
            max_calls=remaining_budget,
            progress_cb=progress_cb,
        )

    for page in parsed.pages:
        # Pull this page's chart descriptions out of the global map
        page_descs = [
            descs_all[(page.page_number, im.image_index)]
            for im in page.images
            if (page.page_number, im.image_index) in descs_all
        ]
        page_chunks = chunk_page(
            doc_id=meta.doc_id,
            page_number=page.page_number,
            page_markdown=page.markdown,
            chart_descriptions=page_descs,
            company=meta.company,
            doc_date=meta.doc_date,
            version_label=meta.version_label,
        )
        chunks.extend(page_chunks)

    return chunks


def ingest_one(
    pdf_path: Path,
    *,
    with_vision: bool = True,
    vision_call_budget: int | None = None,
    dry_run: bool = False,
    progress_cb=None,
) -> dict:
    """Ingest one PDF end-to-end.

    progress_cb is called with (event_name: str, payload: dict) at key stages:
      start, parse_start, parse_batch, parse_done, vision_start,
      vision_progress, vision_done, chunk_done, embed_start, embed_progress,
      embed_done, upsert_done, done, error.
    """
    def _emit(ev, **kw):
        if progress_cb:
            try:
                progress_cb(ev, kw)
            except Exception:
                pass  # never let the UI callback break ingestion

    t0 = time.perf_counter()
    meta = extract_metadata(pdf_path)
    _emit("start",
          file=pdf_path.name, company=meta.company,
          version=meta.version_label, doc_id=meta.doc_id)
    log.info(
        "ingest start: %s [%s, %s]",
        pdf_path.name, meta.company, meta.version_label or "undated",
    )

    try:
        parsed = parse_pdf(pdf_path, progress_cb=progress_cb)
    except Exception as e:
        _emit("error", stage="parse", message=str(e))
        raise
    _emit("parse_done", pages=parsed.page_count,
          images=sum(len(p.images) for p in parsed.pages))
    log.info("  parsed: %d pages, %d images",
             parsed.page_count, sum(len(p.images) for p in parsed.pages))

    chunks = _build_chunks(
        meta, parsed,
        with_vision=with_vision,
        vision_call_budget=vision_call_budget,
        progress_cb=progress_cb,
    )
    by_type = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    _emit("chunk_done", total=len(chunks), by_type=by_type)
    log.info("  chunked: %d total %s", len(chunks), by_type)

    if dry_run:
        log.info("  [dry-run] skipping embedding + Snowflake writes")
        _emit("done", stored=False, chunks=len(chunks),
              elapsed_s=round(time.perf_counter() - t0, 2))
        return {
            "doc_id": meta.doc_id,
            "pages": parsed.page_count,
            "chunks": len(chunks),
            "by_type": by_type,
            "stored": False,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    # Embed (batched — was 1 call per chunk, now 1 call per ~100 chunks)
    embedder = get_embedder()
    _emit("embed_start", total=len(chunks), dim=embedder.dim, provider=embedder.name)
    log.info("  embedding %d chunks (dim=%d) via %s", len(chunks), embedder.dim, embedder.name)
    try:
        vectors = embedder.embed([c.text for c in chunks], progress_cb=progress_cb)
    except TypeError:
        # Embedders without progress_cb support
        vectors = embedder.embed([c.text for c in chunks])
    _emit("embed_done", total=len(vectors))

    # Upsert — single Snowflake connection reused across all DAO calls
    with get_connection() as conn:
        status = upsert_document(meta, page_count=parsed.page_count, conn=conn)
        if status != "unchanged":
            delete_chunks_for_doc(meta.doc_id, conn=conn)
        n = insert_chunks(chunks, vectors, conn=conn)
    _emit("upsert_done", doc_status=status, chunks=n)
    log.info("  stored: doc=%s chunks=%d", status, n)

    elapsed_s = round(time.perf_counter() - t0, 2)
    _emit("done", stored=True, chunks=n, doc_status=status, elapsed_s=elapsed_s)
    return {
        "doc_id": meta.doc_id,
        "pages": parsed.page_count,
        "chunks": n,
        "by_type": by_type,
        "stored": True,
        "doc_status": status,
        "elapsed_s": elapsed_s,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest PDFs into the RAG corpus")
    p.add_argument("--limit", type=int, default=None, help="process only first N PDFs")
    p.add_argument("--doc", type=str, default=None, help="process only this filename")
    p.add_argument("--no-vision", action="store_true", help="skip Gemini vision pass")
    p.add_argument("--vision-budget", type=int, default=None,
                   help="cap total vision calls across this run (free-tier safety)")
    p.add_argument("--dry-run", action="store_true",
                   help="parse + chunk only; don't embed or write to Snowflake")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    docs_dir = settings.documents_path
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if args.doc:
        pdfs = [p for p in pdfs if p.name == args.doc]
    if args.limit is not None:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print("No PDFs to process.")
        return 1

    print(f"Ingesting {len(pdfs)} PDF(s) from {docs_dir}")
    results = []
    for pdf in pdfs:
        try:
            r = ingest_one(
                pdf,
                with_vision=not args.no_vision,
                vision_call_budget=args.vision_budget,
                dry_run=args.dry_run,
            )
            results.append(r)
        except Exception as e:
            log.exception("FAILED %s: %s", pdf.name, e)
            results.append({"doc": pdf.name, "error": str(e)})

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

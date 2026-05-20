"""End-to-end v2 ingestion (checkpointed, multi-vector).

Stages per document:
  parse -> identify -> chunk -> vision -> chart-summaries -> propositions
        -> embed -> upsert -> complete

Each stage writes an ingest_checkpoints row. A document whose file checksum
already reached 'complete' is skipped, so a crash/restart resumes at the next
unfinished document instead of redoing everything (closes the expensive-restart
pain). Intra-document, a re-run recomputes that one document from scratch.

Writes to: documents, parent_chunks, chunks, propositions, table_rows,
chart_records.

Usage:
  python -m rag_system.ingest.pipeline_v2                 # all PDFs
  python -m rag_system.ingest.pipeline_v2 --limit 1       # first PDF
  python -m rag_system.ingest.pipeline_v2 --doc <name>    # one PDF
  python -m rag_system.ingest.pipeline_v2 --no-vision     # skip vision
  python -m rag_system.ingest.pipeline_v2 --force         # ignore checkpoints
  python -m rag_system.ingest.pipeline_v2 --dry-run       # no DB writes
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from rag_system.config import settings
from rag_system.ingest.chunk_v2 import ChunkV2, chunk_page_v2
from rag_system.ingest.metadata import file_checksum
from rag_system.ingest.metadata_v2 import extract_metadata_v2
from rag_system.ingest.parse import parse_pdf
from rag_system.ingest.propositions import extract_propositions
from rag_system.ingest.vision_v2 import LOW_CONFIDENCE, extract_chart_records
from rag_system.llm_providers import get_embedder, get_llm
from rag_system.storage import repository_v2 as repo
from rag_system.storage.db import get_connection

log = logging.getLogger(__name__)


def _chart_summary_chunks(records, parents_by_page, doc_id) -> list[ChunkV2]:
    """Synthesize one retrievable 'chart' child chunk per figure from its records.

    The structured chart_records remain the precise lookup target; this synthetic
    chunk lets dense/lexical retrieval surface the figure by topic too. Low-
    confidence mappings are marked so downstream generation stays honest.
    """
    by_chart: dict[str, list] = {}
    for r in records:
        by_chart.setdefault(r.chart_id, []).append(r)

    out: list[ChunkV2] = []
    # chart summaries use a high chunk_index (900+) to avoid colliding with the
    # page's own children (which use 0..N).
    for chart_id, recs in by_chart.items():
        r0 = recs[0]
        parent_id = r0.chunk_id  # set to the page parent in vision_v2
        lines = []
        any_low = False
        for r in recs:
            lbl = r.label or "(unlabeled)"
            val = (r.value + (r.unit or "")) if r.value else ""
            mark = "" if r.confidence >= LOW_CONFIDENCE else "  [uncertain mapping]"
            if r.confidence < LOW_CONFIDENCE:
                any_low = True
            lines.append(f"- {lbl}: {val}{mark}".rstrip())
        header = f"Figure ({r0.chart_kind}) on page {r0.page_number}"
        if any_low:
            header += " — some label/value mappings are uncertain"
        text = header + "\n" + "\n".join(lines)

        parent = parents_by_page.get(r0.page_number)
        out.append(ChunkV2(
            chunk_id=f"{chart_id}::summary",
            doc_id=doc_id, parent_id=parent_id,
            page_number=r0.page_number, chunk_index=900 + len(out),
            text=text, chunk_type="chart",
            token_count=len(text.split()),
            slide_title=(parent.slide_title if parent else None),
            footnote_text=None, qualifier_text=None,
            company=r0.company, doc_type=r0.doc_type, doc_date=r0.doc_date,
            as_of_date=r0.as_of_date, doc_family_id=r0.doc_family_id,
            version_label=(parent.version_label if parent else None),
        ))
    return out


def ingest_one_v2(
    pdf_path: Path,
    *,
    with_vision: bool = True,
    with_propositions: bool = True,
    vision_budget: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    progress_cb=None,
    llm=None,
) -> dict:
    def _emit(ev, **kw):
        if progress_cb:
            try:
                progress_cb(ev, kw)
            except Exception:
                pass

    t0 = time.perf_counter()
    checksum = file_checksum(pdf_path)

    # Resume: skip a file whose checksum already completed.
    if not force and not dry_run and repo.is_complete_by_checksum(checksum):
        log.info("skip (already complete): %s", pdf_path.name)
        _emit("skipped", file=pdf_path.name)
        return {"file": pdf_path.name, "skipped": True}

    llm = llm or get_llm()
    _emit("start", file=pdf_path.name)

    # --- parse ---
    parsed = parse_pdf(pdf_path, progress_cb=progress_cb)
    first_pages = "\n\n".join(p.markdown for p in parsed.pages[:2])
    _emit("parse_done", pages=parsed.page_count)

    # --- identify (domain-agnostic) ---
    meta = extract_metadata_v2(pdf_path, first_pages_text=first_pages, llm=llm)
    doc_id = meta.doc_id
    _emit("identify_done", doc_id=doc_id, company=meta.company,
          doc_type=meta.doc_type, as_of=str(meta.as_of_date))
    if not dry_run:
        repo.mark_stage(doc_id, checksum, "parse", "done", f"{parsed.page_count} pages")

    # --- chunk (parents + children + table rows) ---
    parents, children, table_rows = [], [], []
    parents_by_page = {}
    for page in parsed.pages:
        pc = chunk_page_v2(
            doc_id=doc_id, page_number=page.page_number,
            page_markdown=page.markdown,
            company=meta.company, doc_type=meta.doc_type, doc_date=meta.doc_date,
            as_of_date=meta.as_of_date, doc_family_id=meta.doc_family_id,
            version_label=meta.version_label,
        )
        parents.append(pc.parent)
        parents_by_page[page.page_number] = pc.parent
        children.extend(pc.children)
        table_rows.extend(pc.table_rows)
    _emit("chunk_done", parents=len(parents), children=len(children),
          table_rows=len(table_rows))
    if not dry_run:
        repo.mark_stage(doc_id, checksum, "chunk", "done",
                        f"{len(parents)}p/{len(children)}c/{len(table_rows)}tr")

    # --- vision -> chart_records ---
    chart_records = []
    if with_vision:
        all_images = [im for pg in parsed.pages for im in pg.images]
        try:
            chart_records = extract_chart_records(
                all_images, doc_id=doc_id, company=meta.company,
                doc_type=meta.doc_type, doc_date=meta.doc_date,
                as_of_date=meta.as_of_date, doc_family_id=meta.doc_family_id,
                max_calls=vision_budget, progress_cb=progress_cb,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("vision stage failed (continuing text-only): %s", e)
            if not dry_run:
                repo.mark_stage(doc_id, checksum, "vision", "failed", str(e))
    _emit("vision_done", records=len(chart_records))
    if not dry_run and with_vision:
        repo.mark_stage(doc_id, checksum, "vision", "done", f"{len(chart_records)} records")

    # synthesize retrievable chart-summary chunks
    if chart_records:
        children.extend(_chart_summary_chunks(chart_records, parents_by_page, doc_id))

    # --- propositions (from prose + chart children) ---
    # Optional: 1 LLM call per chunk is the most rate-limit-heavy stage. When
    # off, child-chunk embeddings remain the dense retrieval target (still good).
    prop_dicts = []
    if with_propositions:
        prose_children = [c for c in children if c.chunk_type in ("prose", "chart")]
        for i, c in enumerate(prose_children):
            for p_i, ptext in enumerate(extract_propositions(c.text, llm=llm)):
                prop_dicts.append({
                    "prop_id": f"{c.chunk_id}::prop{p_i:02d}",
                    "chunk_id": c.chunk_id, "parent_id": c.parent_id, "doc_id": doc_id,
                    "page_number": c.page_number, "text": ptext,
                    "company": c.company, "doc_type": c.doc_type, "doc_date": c.doc_date,
                    "as_of_date": c.as_of_date, "doc_family_id": c.doc_family_id,
                    "version_label": c.version_label,
                })
            if progress_cb and (i % 10 == 0):
                _emit("propositions_progress", done=i, total=len(prose_children))
        _emit("propositions_done", total=len(prop_dicts))
        if not dry_run:
            repo.mark_stage(doc_id, checksum, "propositions", "done", f"{len(prop_dicts)} props")

    if dry_run:
        elapsed = round(time.perf_counter() - t0, 2)
        _emit("done", stored=False, elapsed_s=elapsed)
        return {
            "doc_id": doc_id, "company": meta.company, "doc_type": meta.doc_type,
            "as_of_date": str(meta.as_of_date), "pages": parsed.page_count,
            "parents": len(parents), "children": len(children),
            "table_rows": len(table_rows), "chart_records": len(chart_records),
            "propositions": len(prop_dicts), "stored": False, "elapsed_s": elapsed,
        }

    # --- embed (children, propositions, table_rows) ---
    embedder = get_embedder()
    _emit("embed_start", children=len(children), props=len(prop_dicts),
          rows=len(table_rows))
    child_vecs = embedder.embed([c.text for c in children]) if children else []
    prop_vecs = embedder.embed([p["text"] for p in prop_dicts]) if prop_dicts else []
    row_vecs = embedder.embed([r.flat_text for r in table_rows]) if table_rows else []
    repo.mark_stage(doc_id, checksum, "embed", "done",
                    f"{len(child_vecs)}+{len(prop_vecs)}+{len(row_vecs)}")
    _emit("embed_done")

    # --- upsert (single connection, clean re-ingest) ---
    with get_connection() as conn:
        status = repo.upsert_document_v2(meta, page_count=parsed.page_count, conn=conn)
        repo.delete_doc_artifacts_v2(doc_id, conn=conn)
        repo.insert_parent_chunks(parents, conn=conn)
        repo.insert_children_v2(children, child_vecs, conn=conn)
        repo.insert_propositions(prop_dicts, prop_vecs, conn=conn)
        repo.insert_table_rows(table_rows, row_vecs, conn=conn)
        repo.insert_chart_records(chart_records, conn=conn)
        repo.mark_stage(doc_id, checksum, "upsert", "done", status, conn=conn)
        repo.mark_stage(doc_id, checksum, "complete", "done", "", conn=conn)

    elapsed = round(time.perf_counter() - t0, 2)
    _emit("done", stored=True, doc_status=status, elapsed_s=elapsed)
    log.info("stored %s: %s parents=%d children=%d props=%d rows=%d charts=%d (%.1fs)",
             doc_id, status, len(parents), len(children), len(prop_dicts),
             len(table_rows), len(chart_records), elapsed)
    return {
        "doc_id": doc_id, "company": meta.company, "doc_type": meta.doc_type,
        "as_of_date": str(meta.as_of_date), "pages": parsed.page_count,
        "parents": len(parents), "children": len(children),
        "table_rows": len(table_rows), "chart_records": len(chart_records),
        "propositions": len(prop_dicts), "stored": True, "doc_status": status,
        "elapsed_s": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    p = argparse.ArgumentParser(description="v2 ingestion")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--doc", type=str, default=None)
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--no-propositions", action="store_true",
                   help="skip per-chunk proposition extraction (rate-limit-heavy); "
                        "child-chunk embeddings remain the dense retrieval target")
    p.add_argument("--vision-budget", type=int, default=None)
    p.add_argument("--force", action="store_true", help="ignore checkpoints")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--llm-provider", type=str, default=None,
                   help="LLM for identification + propositions (e.g. gemini). "
                        "Defaults to configured provider. Use paid Gemini to avoid "
                        "Cerebras free-tier rate limits.")
    p.add_argument("--llm-model", type=str, default=None,
                   help="LLM model name (e.g. gemini-2.5-flash-lite)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    pdfs = sorted(settings.documents_path.glob("*.pdf"))
    if args.doc:
        pdfs = [x for x in pdfs if x.name == args.doc]
    if args.limit is not None:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print("No PDFs to process.")
        return 1

    ingest_llm = None
    if args.llm_provider or args.llm_model:
        ingest_llm = get_llm(args.llm_provider, args.llm_model)
        print(f"ingest LLM: {args.llm_provider or 'default'} / {args.llm_model or 'default'}")

    print(f"v2 ingesting {len(pdfs)} PDF(s) from {settings.documents_path}")
    results = []
    for pdf in pdfs:
        try:
            results.append(ingest_one_v2(
                pdf, with_vision=not args.no_vision,
                with_propositions=not args.no_propositions,
                vision_budget=args.vision_budget,
                dry_run=args.dry_run, force=args.force,
                llm=ingest_llm,
            ))
        except Exception as e:  # noqa: BLE001
            log.exception("FAILED %s", pdf.name)
            results.append({"doc": pdf.name, "error": str(e)})

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

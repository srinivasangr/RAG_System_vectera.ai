"""End-to-end v3 ingestion — page-level vision, stored PDF + page images,
file metadata, content dedup.

Per document:
  0 dedup (update-if-changed by sha256)
  1 file + PDF metadata; store the raw PDF
  2 parse (Docling: text + which pages are visual)
  3 identify (LLM, text)
  4 chunk prose (Docling) -> parents + prose children
  5 page-vision (LLM, image) on VISUAL pages -> tables/charts/figures
        + render & store each visual page image
  6 propositions (LLM, text) from prose + visual descriptions
  7 embed (local BGE): children + propositions + table_rows
  8 store everything (one connection)
  9 checkpoint complete

Docling = text/layout (fast, local). Vision = all visual content (tables,
charts, maps, logos) with descriptions. Domain-agnostic throughout.

Usage:
  python -m rag_system.ingest.pipeline_v3 --doc <name>
  python -m rag_system.ingest.pipeline_v3 --force --vision-model gemini-3.1-flash-lite
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rag_system.config import settings
from rag_system.ingest.chunk_v2 import ChunkV2, chunk_page_v2
from rag_system.ingest.metadata import file_checksum
from rag_system.ingest.metadata_v2 import extract_file_meta, extract_metadata_v2
from rag_system.ingest.parse import parse_pdf
from rag_system.ingest.propositions import extract_propositions
from rag_system.ingest.vision_page import (
    DEFAULT_VISION_MODEL, extract_page_elements, render_page_png,
)
from rag_system.llm_providers import get_embedder, get_llm, get_vision
from rag_system.storage import repository_v2 as repo
from rag_system.storage import repository_v3 as repo3
from rag_system.storage.db import get_connection

log = logging.getLogger(__name__)

_TABLE_MD = re.compile(r"\|[^\n]+\|\n\|[\s\-:|]+\|")


def _first_pages_text(pdf_path: Path, parsed, n: int = 3) -> str:
    txt = ""
    try:
        from pypdf import PdfReader
        for pg in PdfReader(str(pdf_path)).pages[:n]:
            txt += (pg.extract_text() or "") + "\n"
    except Exception:  # noqa: BLE001
        txt = ""
    if len(txt.strip()) < 100:
        txt = "\n\n".join(p.markdown for p in parsed.pages[:n])
    return txt


def _png_to_jpeg_thumb(png_bytes: bytes, *, max_w: int = 1000, quality: int = 72) -> tuple[bytes, int, int]:
    """Downscale a page PNG to a compact JPEG thumbnail for storage.

    Vision still gets the full-res PNG; only the STORED provenance image is
    shrunk (cuts ~5-10x size, so big decks don't bloat Snowflake / slow upsert).
    """
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if img.width > max_w:
        h = int(img.height * max_w / img.width)
        img = img.resize((max_w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), img.width, img.height


def _page_is_visual(page) -> bool:
    """Render+vision a page if it has pictures, a table, or little text."""
    if page.images:
        return True
    md = page.markdown or ""
    if _TABLE_MD.search(md):
        return True
    return len(md.strip()) < 300


def ingest_one_v3(
    pdf_path: Path,
    *,
    with_vision: bool = True,
    with_propositions: bool = True,
    vision_model: str = DEFAULT_VISION_MODEL,
    force: bool = False,
    dry_run: bool = False,
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

    # 0. dedup — update-if-changed
    if not force and not dry_run and repo.is_complete_by_checksum(checksum):
        log.info("skip (unchanged, already complete): %s", pdf_path.name)
        _emit("done", skipped=True, file=pdf_path.name)
        return {"file": pdf_path.name, "skipped": True}

    llm = llm or get_llm()
    vision = get_vision() if with_vision else None
    _emit("start", file=pdf_path.name)

    # 1. file metadata + raw PDF
    file_meta = extract_file_meta(pdf_path)
    pdf_bytes = pdf_path.read_bytes()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")

    # 2. parse
    parsed = parse_pdf(pdf_path, progress_cb=progress_cb)
    first_pages = _first_pages_text(pdf_path, parsed, n=3)
    _emit("parse_done", pages=parsed.page_count)

    # 3. identify
    meta = extract_metadata_v2(pdf_path, first_pages_text=first_pages, llm=llm)
    doc_id = meta.doc_id
    stored_pdf_path = meta.source_path  # we overwrite with relative below
    _emit("identify_done", doc_id=doc_id, company=meta.company,
          doc_type=meta.doc_type, as_of=str(meta.as_of_date))
    if not dry_run:
        repo.mark_stage(doc_id, checksum, "parse", "done", f"{parsed.page_count} pages")

    # 4. chunk prose (+ parents). Tables come from vision, so keep only prose children.
    parents, children = [], []
    parents_by_page = {}
    for page in parsed.pages:
        pc = chunk_page_v2(
            doc_id=doc_id, page_number=page.page_number, page_markdown=page.markdown,
            company=meta.company, doc_type=meta.doc_type, doc_date=meta.doc_date,
            as_of_date=meta.as_of_date, doc_family_id=meta.doc_family_id,
            version_label=meta.version_label,
        )
        parents.append(pc.parent)
        parents_by_page[page.page_number] = pc.parent
        children.extend([c for c in pc.children if c.chunk_type == "prose"])
    _emit("chunk_done", parents=len(parents), children=len(children))
    if not dry_run:
        repo.mark_stage(doc_id, checksum, "chunk", "done", f"{len(parents)}p/{len(children)}c")

    # 5. page-vision on visual pages -> tables/charts/figures + page images
    table_rows: list[dict] = []
    chart_records: list[dict] = []
    page_images: list[dict] = []
    vis_idx = 0
    if with_vision:
        visual_pages = [p for p in parsed.pages if _page_is_visual(p)]
        _emit("vision_start", total=len(visual_pages))
        workers = int(os.environ.get("VISION_CONCURRENCY", "5"))

        def _work(page):
            """Render + vision one page (runs in a worker thread → parallel API calls)."""
            png, w, h = render_page_png(str(pdf_path), page.page_number)
            _summary, elements = extract_page_elements(png, model=vision_model, vision=vision)
            try:
                jpg, jw, jh = _png_to_jpeg_thumb(png)
            except Exception:  # noqa: BLE001
                jpg, jw, jh = png, w, h
            return page, (jpg, jw, jh), elements

        results = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_work, p): p for p in visual_pages}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    log.warning("vision page failed: %s", e)
                done += 1
                _emit("vision_progress", done=done, total=len(visual_pages))

        # Process in page order so chunk ids are deterministic.
        results.sort(key=lambda r: r[0].page_number)
        for page, (jpg, jw, jh), elements in results:
            parent = parents_by_page.get(page.page_number)
            page_images.append({
                "parent_id": parent.parent_id if parent else f"{doc_id}::p{page.page_number:03d}",
                "doc_id": doc_id, "page_number": page.page_number,
                "width": jw, "height": jh, "mime_type": "image/jpeg",
                "image_b64": base64.b64encode(jpg).decode("ascii"),
            })
            for el in elements:
                routed = _route_element(
                    el, doc_id=doc_id, page=page, parents_by_page=parents_by_page,
                    meta=meta, vis_idx=vis_idx,
                )
                vis_idx += 1
                children.extend(routed["chunks"])
                table_rows.extend(routed["table_rows"])
                chart_records.extend(routed["chart_records"])
        _emit("vision_done", records=len(chart_records), figures=vis_idx,
              pages=len(visual_pages))
        if not dry_run:
            repo.mark_stage(doc_id, checksum, "vision", "done",
                            f"{vis_idx} elements, {len(chart_records)} chart recs")

    # 6. propositions from prose + visual chunk descriptions
    prop_dicts = []
    if with_propositions:
        # Only PROSE produces propositions. Tables/charts/figures already carry
        # structured rows (table_rows/chart_records) + an embedded description
        # chunk, so decomposing them too just creates noise. Skip tiny chunks
        # (headings) that yield trivial facts.
        src = [c for c in children
               if c.chunk_type == "prose" and (c.token_count or 0) >= 25]
        for i, c in enumerate(src):
            for p_i, ptext in enumerate(extract_propositions(c.text, llm=llm)):
                prop_dicts.append({
                    "prop_id": f"{c.chunk_id}::prop{p_i:02d}",
                    "chunk_id": c.chunk_id, "parent_id": c.parent_id, "doc_id": doc_id,
                    "page_number": c.page_number, "text": ptext,
                    "company": c.company, "doc_type": c.doc_type, "doc_date": c.doc_date,
                    "as_of_date": c.as_of_date, "doc_family_id": c.doc_family_id,
                    "version_label": c.version_label,
                })
            if progress_cb and i % 10 == 0:
                _emit("propositions_progress", done=i, total=len(src))
        _emit("propositions_done", total=len(prop_dicts))
        if not dry_run:
            repo.mark_stage(doc_id, checksum, "propositions", "done", f"{len(prop_dicts)}")

    if dry_run:
        return {
            "doc_id": doc_id, "company": meta.company, "doc_type": meta.doc_type,
            "as_of_date": str(meta.as_of_date), "pages": parsed.page_count,
            "parents": len(parents), "children": len(children),
            "table_rows": len(table_rows), "chart_records": len(chart_records),
            "page_images": len(page_images), "propositions": len(prop_dicts),
            "stored": False, "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    # 7. embed
    embedder = get_embedder()
    _emit("embed_start", children=len(children), props=len(prop_dicts), rows=len(table_rows))
    child_vecs = embedder.embed([c.text for c in children]) if children else []
    prop_vecs = embedder.embed([p["text"] for p in prop_dicts]) if prop_dicts else []
    row_vecs = embedder.embed([r["flat_text"] for r in table_rows]) if table_rows else []
    _emit("embed_done")

    # 8. store
    rel_path = file_meta.get("original_filename") or pdf_path.name  # relative, no abs path
    with get_connection() as conn:
        status = repo3.upsert_document_v3(meta, file_meta, parsed.page_count, rel_path, conn=conn)
        repo.delete_doc_artifacts_v2(doc_id, conn=conn)
        repo3.insert_document_file(doc_id, file_meta["original_filename"],
                                   file_meta["mime_type"], file_meta["file_size_bytes"],
                                   pdf_b64, conn=conn)
        repo.insert_parent_chunks(parents, conn=conn)
        repo3.insert_page_images(page_images, conn=conn)
        repo.insert_children_v2(children, child_vecs, conn=conn)
        repo.insert_propositions(prop_dicts, prop_vecs, conn=conn)
        repo3.insert_table_rows_v3(table_rows, row_vecs, conn=conn)
        repo3.insert_chart_records_v3(chart_records, conn=conn)
        repo.mark_stage(doc_id, checksum, "upsert", "done", status, conn=conn)
        repo.mark_stage(doc_id, checksum, "complete", "done", "", conn=conn)

    elapsed = round(time.perf_counter() - t0, 2)
    _emit("done", stored=True, doc_status=status, elapsed_s=elapsed,
          doc_id=doc_id, children=len(children), propositions=len(prop_dicts),
          table_rows=len(table_rows), chart_records=len(chart_records),
          page_images=len(page_images))
    log.info("stored %s: %s parents=%d children=%d props=%d rows=%d charts=%d imgs=%d (%.1fs)",
             doc_id, status, len(parents), len(children), len(prop_dicts),
             len(table_rows), len(chart_records), len(page_images), elapsed)
    return {
        "doc_id": doc_id, "company": meta.company, "doc_type": meta.doc_type,
        "as_of_date": str(meta.as_of_date), "pages": parsed.page_count,
        "parents": len(parents), "children": len(children),
        "table_rows": len(table_rows), "chart_records": len(chart_records),
        "page_images": len(page_images), "propositions": len(prop_dicts),
        "stored": True, "doc_status": status, "elapsed_s": elapsed,
    }


def _route_element(el, *, doc_id, page, parents_by_page, meta, vis_idx) -> dict:
    """Turn one vision PageElement into chunks + table_rows + chart_records."""
    parent = parents_by_page.get(page.page_number)
    parent_id = parent.parent_id if parent else f"{doc_id}::p{page.page_number:03d}"
    base = dict(
        doc_id=doc_id, company=meta.company, doc_type=meta.doc_type,
        doc_date=meta.doc_date, as_of_date=meta.as_of_date,
        doc_family_id=meta.doc_family_id, version_label=meta.version_label,
    )
    chunks, trows, crecs = [], [], []
    title = el.title or ""
    desc = el.description or ""

    def _chunk(text, ctype, kind_detail):
        return ChunkV2(
            chunk_id=f"{parent_id}::v{vis_idx:03d}",
            doc_id=doc_id, parent_id=parent_id, page_number=page.page_number,
            chunk_index=900 + vis_idx, text=text.strip(), chunk_type=ctype,
            token_count=len(text.split()), slide_title=parent.slide_title if parent else title,
            confidence=el.confidence, kind_detail=kind_detail,
            company=meta.company, doc_type=meta.doc_type, doc_date=meta.doc_date,
            as_of_date=meta.as_of_date, doc_family_id=meta.doc_family_id,
            version_label=meta.version_label,
        )

    if el.kind == "table" and el.columns and el.rows:
        # structured rows
        table_id = f"{parent_id}::vt{vis_idx:02d}"
        for ri, row in enumerate(el.rows):
            cols = {}
            for ci, cell in enumerate(row):
                label = el.columns[ci] if ci < len(el.columns) else f"col{ci}"
                cols[label or f"col{ci}"] = cell
            flat = "; ".join(f"{k}: {v}" for k, v in cols.items() if v)
            if not flat.strip():
                continue
            trows.append({
                "row_id": f"{table_id}::r{ri:02d}", "chunk_id": f"{parent_id}::v{vis_idx:03d}",
                "page_number": page.page_number, "table_id": table_id, "row_idx": ri,
                "columns": cols, "flat_text": flat, **base,
            })
        body = "\n".join("; ".join(f"{el.columns[ci] if ci < len(el.columns) else ''}: {c}"
                                   for ci, c in enumerate(r)) for r in el.rows)
        chunks.append(_chunk(f"Table: {title}\n{desc}\n{body}", "table", el.kind_detail or "table"))

    elif el.kind == "chart":
        chart_id = f"{parent_id}::vc{vis_idx:02d}"
        lines = []
        for si, s in enumerate(el.series):
            lbl, val, unit = s.get("label"), s.get("value"), s.get("unit") or ""
            crecs.append({
                "record_id": f"{chart_id}::r{si:02d}", "chunk_id": f"{parent_id}::v{vis_idx:03d}",
                "doc_id": doc_id, "page_number": page.page_number, "chart_id": chart_id,
                "chart_kind": el.kind_detail or "chart",
                "label": "" if lbl is None else str(lbl), "value": "" if val is None else str(val),
                "unit": str(unit), "bbox": "", "confidence": el.confidence,
                "vision_model": "", "description": desc,
                "company": meta.company, "doc_type": meta.doc_type, "doc_date": meta.doc_date,
                "as_of_date": meta.as_of_date, "doc_family_id": meta.doc_family_id,
            })
            if lbl is not None:
                lines.append(f"- {lbl}: {val}{unit}")
        chunks.append(_chunk(f"Chart: {title}\n{desc}\n" + "\n".join(lines),
                             "chart", el.kind_detail or "chart"))

    else:  # figure | map | logo | other
        ent = (" Entities: " + ", ".join(el.entities)) if el.entities else ""
        chunks.append(_chunk(f"Figure: {title}\n{desc}{ent}", "figure",
                             el.kind_detail or el.kind))

    return {"chunks": chunks, "table_rows": trows, "chart_records": crecs}


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    p = argparse.ArgumentParser(description="v3 ingestion")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--doc", type=str, default=None)
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--no-propositions", action="store_true")
    p.add_argument("--vision-model", type=str, default=DEFAULT_VISION_MODEL)
    p.add_argument("--llm-provider", type=str, default=None)
    p.add_argument("--llm-model", type=str, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    pdfs = sorted(settings.documents_path.glob("*.pdf"))
    if args.doc:
        pdfs = [x for x in pdfs if x.name == args.doc]
    if args.limit is not None:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print("No PDFs."); return 1

    ingest_llm = get_llm(args.llm_provider, args.llm_model) if (args.llm_provider or args.llm_model) else None
    print(f"v3 ingesting {len(pdfs)} PDF(s); vision_model={args.vision_model}")
    for pdf in pdfs:
        try:
            r = ingest_one_v3(pdf, with_vision=not args.no_vision,
                              with_propositions=not args.no_propositions,
                              vision_model=args.vision_model, force=args.force,
                              dry_run=args.dry_run, llm=ingest_llm)
            print(f"  {r}")
        except Exception as e:  # noqa: BLE001
            log.exception("FAILED %s", pdf.name)
            print(f"  FAILED {pdf.name}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

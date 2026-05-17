"""PDF → structured content via Docling.

Returns per-page records carrying:
  - the page's Markdown text (incl. tables serialized to Markdown)
  - the list of images extracted from the page (PNG bytes)

We assemble per-page Markdown by walking the DoclingDocument tree and
grouping each text/table item by its page provenance, since Docling's
top-level `export_to_markdown` doesn't accept a page filter.

Docling first-run downloads ~300MB of layout/OCR models — one-time cost.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

log = logging.getLogger(__name__)


@dataclass
class PageImage:
    page_number: int
    image_index: int
    png_bytes: bytes
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class PageRecord:
    page_number: int
    markdown: str
    images: list[PageImage] = field(default_factory=list)


@dataclass
class ParsedDocument:
    page_count: int
    pages: list[PageRecord]
    full_markdown: str


# ---------------------------------------------------------------------------
# Converter (lazy-init module singleton)
# ---------------------------------------------------------------------------
_CONVERTER: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _CONVERTER
    if _CONVERTER is None:
        opts = PdfPipelineOptions()
        opts.generate_picture_images = True
        # 1.0 is safer for memory than 2.0; investor decks have many large bitmaps
        opts.images_scale = 1.0
        opts.do_ocr = False
        opts.do_table_structure = True
        _CONVERTER = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
            }
        )
    return _CONVERTER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pil_to_png_bytes(pil_img) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _item_page_no(item) -> int | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    p = prov[0]
    return getattr(p, "page_no", None) or getattr(p, "page", None)


def _text_item_markdown(item) -> str:
    """Render a text-ish item as a Markdown line, respecting its label."""
    text = (getattr(item, "text", "") or "").strip()
    if not text:
        return ""
    label = (getattr(item, "label", "") or "").lower()
    # Common Docling labels: 'title', 'section_header', 'list_item', 'caption',
    # 'footnote', 'page_header', 'page_footer', 'text', 'paragraph'.
    if label in ("title",):
        return f"# {text}"
    if label in ("section_header",):
        return f"## {text}"
    if label in ("list_item",):
        return f"- {text}"
    if label in ("caption", "footnote"):
        return f"*{text}*"
    # 'page_header' / 'page_footer' are usually noise; keep as plain text
    return text


def _table_to_markdown(table) -> str:
    """Render a Docling TableItem as a GitHub-flavored Markdown table.

    Prefers the item's own export_to_markdown if available; otherwise
    builds it from `.data.table_cells`.
    """
    if hasattr(table, "export_to_markdown"):
        try:
            md = table.export_to_markdown()
            if md:
                return md
        except Exception:
            pass
    # Fallback: best-effort cell grid
    data = getattr(table, "data", None)
    if not data:
        return ""
    cells = getattr(data, "table_cells", None) or getattr(data, "grid", None)
    if not cells:
        return ""
    try:
        # cells: list of dicts with row, col, text
        max_r = max(c.start_row_offset_idx for c in cells)
        max_c = max(c.start_col_offset_idx for c in cells)
        grid = [["" for _ in range(max_c + 1)] for _ in range(max_r + 1)]
        for c in cells:
            grid[c.start_row_offset_idx][c.start_col_offset_idx] = (c.text or "").strip()
        # Render: header = row 0
        header = "| " + " | ".join(grid[0]) + " |"
        sep = "| " + " | ".join("---" for _ in grid[0]) + " |"
        body = "\n".join("| " + " | ".join(r) + " |" for r in grid[1:])
        return f"{header}\n{sep}\n{body}".strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Document-tree walker — merges into accumulators rather than building a doc
# ---------------------------------------------------------------------------
def _accumulate_from_doc(
    doc,
    *,
    per_page_lines: dict[int, list[str]],
    per_page_images: dict[int, list[PageImage]],
) -> None:
    for txt in getattr(doc, "texts", []) or []:
        page_no = _item_page_no(txt)
        if not page_no:
            continue
        md_line = _text_item_markdown(txt)
        if md_line:
            per_page_lines.setdefault(page_no, []).append(md_line)

    for tbl in getattr(doc, "tables", []) or []:
        page_no = _item_page_no(tbl)
        if not page_no:
            continue
        md = _table_to_markdown(tbl)
        if md:
            per_page_lines.setdefault(page_no, []).append(md)

    for pic in getattr(doc, "pictures", []) or []:
        page_no = _item_page_no(pic)
        if not page_no:
            continue
        pil_img = getattr(pic, "image", None)
        pil_img = getattr(pil_img, "pil_image", pil_img) if pil_img is not None else None
        if pil_img is None:
            continue
        lst = per_page_images.setdefault(page_no, [])
        lst.append(
            PageImage(
                page_number=page_no,
                image_index=len(lst),
                png_bytes=_pil_to_png_bytes(pil_img),
                width=pil_img.width,
                height=pil_img.height,
            )
        )


def _count_pages(pdf_path: Path) -> int:
    from pypdf import PdfReader
    try:
        return len(PdfReader(str(pdf_path)).pages)
    except Exception as e:
        log.warning("pypdf page-count failed for %s: %s", pdf_path.name, e)
        return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
PAGE_BATCH_SIZE = 10  # start big; fall back to 5 then 2 on memory failure
PAGE_BATCH_FALLBACKS = [10, 5, 2]


def _try_convert_batch(converter, pdf_path: Path, start: int, end: int):
    """Run Docling on (start..end) and return the document, or None on OOM."""
    try:
        result = converter.convert(str(pdf_path), page_range=(start, end))
        return result.document
    except Exception as e:
        msg = str(e)
        # Docling raises a generic Exception when its C++ raster preprocess
        # hits std::bad_alloc. Recognise that pattern; let other errors fall
        # through unchanged.
        if "bad_alloc" in msg or "alloc" in msg.lower() or "MemoryError" in msg:
            log.warning("  pages %d-%d OOM (%s) — will retry smaller", start, end, msg[:80])
            return None
        log.warning("  pages %d-%d FAILED: %s", start, end, msg[:200])
        return None


def parse_pdf(
    pdf_path: Path,
    *,
    batch_size: int = PAGE_BATCH_SIZE,
    progress_cb=None,
) -> ParsedDocument:
    """Parse a PDF in page batches to keep peak memory low.

    Docling can hit std::bad_alloc when rendering many large bitmap pages in
    one pass; splitting into small page ranges avoids that without losing
    layout/table quality. On OOM we retry the same range with a smaller batch
    size (10 -> 5 -> 2 -> single-page).

    progress_cb (optional): called as progress_cb("parse_batch", {"start": s,
    "end": e, "total": T, "elapsed_s": dt}) after each batch.
    """
    import time as _time
    log.info("Parsing %s", pdf_path.name)
    converter = _get_converter()
    total = _count_pages(pdf_path) or 0

    per_page_lines: dict[int, list[str]] = {}
    per_page_images: dict[int, list[PageImage]] = {}

    if total == 0:
        # Fall back to one-shot conversion
        result = converter.convert(str(pdf_path))
        _accumulate_from_doc(
            result.document,
            per_page_lines=per_page_lines,
            per_page_images=per_page_images,
        )
        total = len(getattr(result.document, "pages", {}) or {}) or 0
        if progress_cb:
            progress_cb("parse_batch", {"start": 1, "end": total, "total": total, "elapsed_s": 0})
    else:
        if progress_cb:
            progress_cb("parse_start", {"total": total, "batch_size": batch_size})

        start = 1
        while start <= total:
            end = min(start + batch_size - 1, total)
            t0 = _time.perf_counter()
            doc = _try_convert_batch(converter, pdf_path, start, end)

            # OOM fallback: retry the same range with progressively smaller batches
            if doc is None and end > start:
                for fb_size in PAGE_BATCH_FALLBACKS:
                    if fb_size >= (end - start + 1):
                        continue
                    log.info("  retrying pages %d-%d with batch size %d", start, end, fb_size)
                    sub_ok = True
                    for sub_start in range(start, end + 1, fb_size):
                        sub_end = min(sub_start + fb_size - 1, end)
                        sub_doc = _try_convert_batch(converter, pdf_path, sub_start, sub_end)
                        if sub_doc is None:
                            sub_ok = False
                            break
                        _accumulate_from_doc(
                            sub_doc,
                            per_page_lines=per_page_lines,
                            per_page_images=per_page_images,
                        )
                    if sub_ok:
                        doc = "handled"
                        break
                if doc is None:
                    log.warning("  pages %d-%d permanently failed", start, end)

            elif doc is not None and doc != "handled":
                _accumulate_from_doc(
                    doc,
                    per_page_lines=per_page_lines,
                    per_page_images=per_page_images,
                )

            elapsed = _time.perf_counter() - t0
            log.info("  pages %d-%d done (%.1fs)", start, end, elapsed)
            if progress_cb:
                progress_cb("parse_batch", {
                    "start": start, "end": end, "total": total, "elapsed_s": elapsed,
                })
            start = end + 1

    # Build the final per-page records
    pages: dict[int, PageRecord] = {}
    for p in range(1, max(total, 1) + 1):
        pages[p] = PageRecord(
            page_number=p,
            markdown="\n\n".join(per_page_lines.get(p, [])).strip(),
            images=per_page_images.get(p, []),
        )

    page_list = [pages[p] for p in sorted(pages.keys())]
    full_md = "\n\n".join(p.markdown for p in page_list if p.markdown)

    return ParsedDocument(
        page_count=total or len(page_list),
        pages=page_list,
        full_markdown=full_md,
    )

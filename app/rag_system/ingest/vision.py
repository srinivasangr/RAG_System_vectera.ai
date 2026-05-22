"""Page-level vision: render a PDF page to an image, then have the vision LLM
CLASSIFY and EXTRACT every content element on it.

Why page-level (not per-cropped-image):
  - The model sees legends, footnotes, axis labels, and multi-panel layouts in
    full spatial context — the hard battery cases (re-sorted subcharts, geo maps,
    logo tenant tables) need this.
  - Docling's table detection / image cropping is imperfect on decks; giving the
    model the whole page sidesteps that.

The model returns a typed list of elements. The pipeline routes them:
  table  -> table_rows (+ a 'table' chunk)
  chart  -> chart_records (+ a 'chart' chunk)
  figure/map/logo -> a 'figure' chunk (description + entities)

Domain-agnostic: nothing here assumes the subject matter.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from rag_system.llm_providers import get_vision

log = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "gemini-3.1-flash-lite"
RENDER_DPI = 150


@dataclass
class PageElement:
    kind: str                       # table|chart|figure|map|logo|prose|decorative
    title: str | None = None
    description: str = ""
    kind_detail: str = ""
    confidence: float = 0.0
    columns: list[str] = field(default_factory=list)      # tables
    rows: list[list[str]] = field(default_factory=list)   # tables
    series: list[dict] = field(default_factory=list)      # charts: {label,value,unit}
    entities: list[str] = field(default_factory=list)     # names visible


_PROMPT = """You are analyzing ONE page of a document, given as an image.
Identify every distinct CONTENT element on the page and return STRICT JSON only:

{
  "page_summary": "one sentence on what this page covers",
  "elements": [
    {
      "kind": "table | chart | figure | map | logo | prose | decorative",
      "title": "the element's title/heading if visible, else null",
      "description": "a clear natural-language explanation of what it shows, INCLUDING the key insight or takeaway and the important numbers",
      "kind_detail": "short tag, e.g. bar_chart, line_chart, pie, world_map, comparison_table, tenant_table, photo, org_diagram",
      "columns": ["col headers, TABLES ONLY"],
      "rows": [["one array per row, every cell as a string, TABLES ONLY"]],
      "series": [{"label": "...", "value": "...", "unit": "%|$|M sq ft|..."}],
      "entities": ["named entities visible, e.g. company/tenant/place names"],
      "confidence": 0.0
    }
  ]
}

RULES:
- TABLE: fill columns + rows with EVERY cell. If a cell is a logo/icon, put the
  entity name if you can identify it, else "". Keep each value with its column.
- CHART: fill series with one {label,value,unit} per data point. Pair a value
  with a label ONLY when visually adjacent; if bar order differs from the legend,
  use the bar's own label. If a mapping is uncertain, lower confidence or set
  value to null. NEVER guess.
- FIGURE/MAP/LOGO: describe it in 'description' and list visible 'entities'.
- Skip purely 'decorative' elements (backgrounds, page furniture).
- 'prose' elements may be omitted (page text is captured separately).
- Report ONLY what is visibly present. Do not infer or compute new numbers.
"""


def render_page_png(pdf_path: str, page_number: int, *, dpi: int = RENDER_DPI) -> tuple[bytes, int, int]:
    """Render a 1-based PDF page to PNG bytes via PyMuPDF."""
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png"), pix.width, pix.height
    finally:
        doc.close()


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_page_elements(
    png_bytes: bytes,
    *,
    model: str = DEFAULT_VISION_MODEL,
    vision=None,
) -> tuple[str, list[PageElement]]:
    """Send a page image to the vision LLM → (page_summary, [PageElement])."""
    vision = vision or get_vision()
    fallback_models = [model, "gemini-2.5-flash"]  # try requested model, then a known-good one
    data = None
    used = ""
    for m in fallback_models:
        try:
            reply = vision.describe_image(
                image_bytes=png_bytes, prompt=_PROMPT, mime_type="image/png", model=m,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("page-vision %s failed: %s", m, e)
            continue
        data = _extract_json(reply)
        if data is not None:
            used = m
            break
    if not data:
        return "", []

    summary = str(data.get("page_summary") or "")
    out: list[PageElement] = []
    for el in data.get("elements") or []:
        kind = (el.get("kind") or "figure").strip().lower()
        if kind in ("decorative", "prose"):
            continue
        try:
            conf = float(el.get("confidence"))
        except (TypeError, ValueError):
            conf = 0.0
        out.append(PageElement(
            kind=kind,
            title=el.get("title"),
            description=str(el.get("description") or "").strip(),
            kind_detail=str(el.get("kind_detail") or "").strip(),
            confidence=conf,
            columns=[str(c) for c in (el.get("columns") or [])],
            rows=[[str(c) for c in row] for row in (el.get("rows") or []) if isinstance(row, list)],
            series=[s for s in (el.get("series") or []) if isinstance(s, dict)],
            entities=[str(e) for e in (el.get("entities") or [])],
        ))
    log.debug("page-vision(%s): %d elements", used, len(out))
    return summary, out

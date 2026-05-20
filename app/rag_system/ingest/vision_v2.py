"""Vision pass v2 — structured chart/figure extraction.

v1 asked the model for a prose *description* of each figure. That produces
confidently-wrong answers on charts where the bar order differs from the legend
order, or where labels are rendered as logos (battery failures Q15, Q16).

v2 asks the model for STRUCTURED records — (label, value, unit, confidence)
tuples — and stores them in `chart_records`. The model is explicitly told:
  - only pair a value with a label when they are visually adjacent,
  - lower its confidence (or leave value null) when the mapping is ambiguous,
  - never invent a label it cannot read (e.g. a logo it doesn't recognize).

This lets generation either use a high-confidence mapping OR honestly report
"I can see values but can't reliably map them" — instead of fabricating.

Domain-agnostic: nothing here knows the figure is about REITs or storage. It
extracts whatever labels/values are visible.

Provider failover: gemini-2.5-flash → gemini-2.5-pro → (give up, page stays
text-only and is flagged vision_unavailable by the caller).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date

from rag_system.ingest.parse import PageImage
from rag_system.ingest.vision_extract import _should_describe
from rag_system.llm_providers import get_vision

log = logging.getLogger(__name__)

# Ordered failover chain. Flash is free + fast; Pro is stronger on hard spatial
# layouts but has a tighter free-tier RPM, so it's only a fallback.
VISION_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

# Records below this confidence are kept but flagged so generation can be honest.
LOW_CONFIDENCE = 0.6


@dataclass
class ChartRecordRec:
    record_id: str
    chunk_id: str          # the page parent this figure belongs to
    doc_id: str
    page_number: int
    chart_id: str
    chart_kind: str
    label: str
    value: str
    unit: str
    bbox: str
    confidence: float
    vision_model: str
    company: str | None = None
    doc_type: str | None = None
    doc_date: date | None = None
    as_of_date: date | None = None
    doc_family_id: str | None = None


_BBOX_PROMPT = """You extract STRUCTURED data from a single figure (chart, map,
table, or logo strip) in a document. Do not write prose.

Return STRICT JSON only:
{
  "chart_kind": "bar | line | pie | map | table | logo_table | other | not_a_chart",
  "title": "the figure's title if visible, else null",
  "records": [
    {"label": "<the category/series/entity this value belongs to>",
     "value": "<the number or short text shown, or null if none>",
     "unit":  "<% , $, M sq ft, GW, ... or null>",
     "confidence": <0..1: how sure you are this value maps to THIS label>}
  ],
  "notes": "<one short note on any mapping ambiguity, else null>"
}

CRITICAL RULES:
- If the image is a logo, photo, or decoration with no data, return
  chart_kind "not_a_chart" and an empty records list.
- Pair a value with a label ONLY when they are visually adjacent / clearly
  belong together. When bar order differs from the legend order, use the bar's
  OWN adjacent label, not the legend position.
- If you cannot reliably tell which label a value belongs to, set that record's
  "value" to null OR set "confidence" below 0.5. NEVER guess a mapping.
- If a label is rendered as a logo/icon you cannot identify, use "label": null
  rather than inventing a name.
- Only report numbers you can actually read. Do not compute or infer.
"""


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


def _describe_structured(im: PageImage, vision) -> tuple[dict | None, str]:
    """Try the failover chain. Returns (parsed_json_or_None, model_used)."""
    for model in VISION_MODELS:
        try:
            reply = vision.describe_image(
                image_bytes=im.png_bytes,
                prompt=_BBOX_PROMPT,
                mime_type="image/png",
                model=model,
            )
        except Exception as e:  # noqa: BLE001 — rate limit / 5xx → try next model
            log.warning("vision %s failed p%s img%s: %s",
                        model, im.page_number, im.image_index, e)
            continue
        data = _extract_json(reply)
        if data is not None:
            return data, model
        log.warning("vision %s returned unparseable JSON p%s img%s",
                    model, im.page_number, im.image_index)
    return None, ""


def extract_chart_records(
    images: list[PageImage],
    *,
    doc_id: str,
    company: str | None = None,
    doc_type: str | None = None,
    doc_date: date | None = None,
    as_of_date: date | None = None,
    doc_family_id: str | None = None,
    max_calls: int | None = None,
    progress_cb=None,
) -> list[ChartRecordRec]:
    """Run structured vision over candidate figures → flat ChartRecordRec list."""
    if not images:
        return []

    candidates = [im for im in images if _should_describe(im)]
    if max_calls is not None:
        candidates = candidates[:max_calls]
    if not candidates:
        return []

    if progress_cb:
        progress_cb("vision_start", {"total": len(candidates)})

    vision = get_vision()
    out: list[ChartRecordRec] = []
    done = 0
    figures_with_data = 0

    for im in candidates:
        data, model = _describe_structured(im, vision)
        done += 1
        if progress_cb:
            progress_cb("vision_progress", {
                "done": done, "total": len(candidates), "described": figures_with_data,
            })
        if not data:
            continue
        kind = (data.get("chart_kind") or "other").strip().lower()
        if kind == "not_a_chart":
            continue
        records = data.get("records") or []
        if not records:
            continue
        figures_with_data += 1

        parent_id = f"{doc_id}::p{im.page_number:03d}"
        chart_id = f"{parent_id}::img{im.image_index}"
        for r_i, rec in enumerate(records):
            label = rec.get("label")
            value = rec.get("value")
            # keep a record if it has at least a label or a value
            if label is None and value is None:
                continue
            try:
                conf = float(rec.get("confidence"))
            except (TypeError, ValueError):
                conf = 0.0
            out.append(ChartRecordRec(
                record_id=f"{chart_id}::r{r_i:02d}",
                chunk_id=parent_id,
                doc_id=doc_id,
                page_number=im.page_number,
                chart_id=chart_id,
                chart_kind=kind,
                label="" if label is None else str(label),
                value="" if value is None else str(value),
                unit=str(rec.get("unit") or ""),
                bbox="",  # current prompt returns no bbox; reserved for future
                confidence=conf,
                vision_model=model,
                company=company, doc_type=doc_type, doc_date=doc_date,
                as_of_date=as_of_date, doc_family_id=doc_family_id,
            ))

    if progress_cb:
        progress_cb("vision_done", {"total": len(candidates), "described": figures_with_data})
    log.info("  vision v2: %d figures with data, %d records from %d candidate images",
             figures_with_data, len(out), len(candidates))
    return out

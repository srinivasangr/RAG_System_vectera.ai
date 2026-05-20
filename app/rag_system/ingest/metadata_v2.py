"""Domain-agnostic document identification.

v1 (`metadata.py`) used a hardcoded REIT alias table + keyword doc_type map —
that overfits to the example corpus. v2 identifies a document by *reading it*:

  - primary_entity  : the organization/subject the document is about (LLM-read)
  - doc_type        : a GENERIC taxonomy label (no domain assumptions)
  - as_of_date      : the date the content's data is "as of" (from cover content,
                      not just the filename — closes failure mode F3)
  - doc_date        : nominal publication date
  - doc_family_id   : groups versions/snapshots of the same recurring series so
                      version-pair retrieval can surface them together (F2)

Nothing here knows what a "REIT" or an "FFO" is. Swap in medical, legal, or
financial PDFs and the same code identifies them — because all domain knowledge
comes from the document's own content, never from this module.

The ONLY heuristics retained are domain-neutral: sha256 checksums and a generic
date regex used purely as a fallback when the LLM can't find a date.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rag_system.ingest.metadata import _find_date, file_checksum  # generic helpers only
from rag_system.llm_providers import get_llm
from rag_system.llm_providers.base import Message

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic doc-type taxonomy (domain-neutral)
# ---------------------------------------------------------------------------
# These describe a document's PURPOSE/SHAPE, not its industry. They apply to a
# pitch deck, a clinical report, a legal brief, or an earnings update equally.
DOC_TYPES = [
    "presentation",        # slide deck / investor or corporate presentation
    "periodic_update",     # recurring update tied to a period (quarterly/monthly/annual)
    "annual_report",       # comprehensive yearly report
    "transaction",         # merger / acquisition / offering / deal material
    "roadshow",            # roadshow / marketing deck
    "report",              # standalone analytical or research report
    "third_party_report",  # report authored by someone other than the subject
    "filing",              # regulatory filing
    "factsheet",           # short summary / one-pager
    "other",
]


@dataclass(frozen=True)
class DocMetaV2:
    doc_id: str
    source_path: str
    company: str | None          # primary entity/issuer (generic), content-derived
    ticker: str | None           # only if present in content
    title: str | None
    doc_date: date | None        # nominal publication date
    as_of_date: date | None      # date the data is "as of"
    as_of_source: str            # 'content' | 'filename' | 'none'
    doc_type: str                # one of DOC_TYPES
    doc_type_conf: float         # 0..1
    version_label: str | None    # human label e.g. "Mar 2026"
    doc_family_id: str           # hash(entity + doc_type) — version grouping key
    checksum: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w]+", "_", s).strip("_").lower()
    return s[:80] or "doc"


def _norm_entity(s: str | None) -> str:
    """Normalize an entity name for stable family grouping (lowercase, no punctuation)."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"\b(inc|corp|corporation|company|co|ltd|llc|lp|plc|group|properties|trust)\b",
               "", s.lower())
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _parse_iso(d: str | None) -> date | None:
    if not d or not isinstance(d, str):
        return None
    d = d.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d"):
        try:
            from datetime import datetime
            return datetime.strptime(d, fmt).date()
        except ValueError:
            continue
    # year only
    m = re.fullmatch(r"(20\d{2}|19\d{2})", d)
    if m:
        return date(int(m.group(1)), 1, 1)
    return None


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply (handles ```json fences)."""
    if not text:
        return None
    # strip code fences
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    # find the first balanced {...}
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
                blob = text[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


def _version_label(d: date | None) -> str | None:
    return d.strftime("%b %Y") if d else None


# ---------------------------------------------------------------------------
# LLM identification
# ---------------------------------------------------------------------------
_IDENTIFY_SYSTEM = """You identify documents for a general document Q&A system.
You are given a filename and the text of the first one or two pages (the cover /
opening). Identify the document WITHOUT assuming any particular industry.

Return STRICT JSON only, no prose:
{
  "primary_entity": "the main organization/person/subject this document is about, or null",
  "ticker": "stock ticker or short code if explicitly present in the text, else null",
  "title": "the document's title if visible, else null",
  "doc_type": one of %s,
  "doc_subtype": "a short free-text descriptor, e.g. 'Q4 earnings deck', 'merger proposal'",
  "doc_date": "publication date as YYYY-MM-DD (or YYYY-MM / YYYY) if shown, else null",
  "as_of_date": "the date the DATA in the document is stated to be 'as of' (often in a footnote or subtitle) as YYYY-MM-DD/YYYY-MM/YYYY, else null",
  "confidence": a number 0..1 for how confident you are in doc_type
}

Rules:
- doc_type must be exactly one value from the allowed list.
- Choose the GENERIC purpose, not the industry (a real-estate investor deck and a
  biotech investor deck are both "presentation").
- as_of_date may differ from doc_date (e.g. a March report citing December-31 data).
- Extract dates only if present in the text; do not guess. Use null when unknown.
""" % DOC_TYPES


def _identify_with_llm(filename: str, first_pages_text: str, *, llm=None) -> dict:
    llm = llm or get_llm()
    user = (
        f"FILENAME: {filename}\n\n"
        f"FIRST PAGES TEXT (truncated):\n{first_pages_text[:4000]}"
    )
    try:
        reply = llm.generate(
            [Message(role="system", content=_IDENTIFY_SYSTEM),
             Message(role="user", content=user)],
            temperature=0.0,
            max_tokens=600,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("doc-identify LLM call failed for %s: %s", filename, e)
        return {}
    data = _extract_json(reply) or {}
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_metadata_v2(
    pdf_path: Path,
    *,
    first_pages_text: str = "",
    llm=None,
) -> DocMetaV2:
    """Identify a document from its content (LLM) with domain-neutral fallbacks.

    `first_pages_text` should be the parsed text of the first ~2 pages. When
    empty (e.g. before parsing), identification falls back to the filename only.
    """
    name = pdf_path.stem
    checksum = file_checksum(pdf_path)

    data = _identify_with_llm(pdf_path.name, first_pages_text, llm=llm) if first_pages_text else {}

    # --- entity / ticker / title (content-derived; no alias table) ---
    company = (data.get("primary_entity") or None)
    if isinstance(company, str) and company.strip().lower() in ("", "null", "none", "unknown"):
        company = None
    ticker = data.get("ticker") or None
    if isinstance(ticker, str) and ticker.strip().lower() in ("", "null", "none"):
        ticker = None
    title = data.get("title") or None

    # --- doc_type (generic taxonomy) ---
    doc_type = (data.get("doc_type") or "").strip().lower()
    if doc_type not in DOC_TYPES:
        doc_type = "other"
    try:
        doc_type_conf = float(data.get("confidence"))
    except (TypeError, ValueError):
        doc_type_conf = 0.0

    # --- dates: prefer content (LLM), fall back to a generic filename date regex ---
    as_of_date = _parse_iso(data.get("as_of_date"))
    doc_date = _parse_iso(data.get("doc_date"))
    as_of_source = "content" if as_of_date else "none"

    if doc_date is None:
        fn_date = _find_date(name)  # generic month/quarter/numeric regex — not domain-specific
        if fn_date:
            doc_date = fn_date
    if as_of_date is None and doc_date is not None:
        as_of_date = doc_date
        as_of_source = "filename" if as_of_source == "none" else as_of_source

    version_label = _version_label(as_of_date or doc_date)

    # --- doc_family_id: groups recurring versions of the same series ---
    # Generic key = normalized entity + generic doc_type. Two snapshots of the
    # same series from the same entity (e.g. two quarterly decks) share a family,
    # enabling version-pair surfacing (F2) without any domain knowledge.
    family_seed = f"{_norm_entity(company) or _slugify(name)}|{doc_type}"
    doc_family_id = hashlib.sha1(family_seed.encode("utf-8")).hexdigest()[:16]

    # --- doc_id: stable, content-changing → new id (checksum suffix) ---
    parts = [
        _slugify(company) if company else _slugify(name),
        (as_of_date or doc_date).strftime("%Y_%m") if (as_of_date or doc_date) else "undated",
        checksum[:8],
    ]
    doc_id = "__".join(parts)

    log.info(
        "identified %s -> entity=%r type=%s(%.2f) as_of=%s family=%s",
        pdf_path.name, company, doc_type, doc_type_conf, as_of_date, doc_family_id,
    )

    return DocMetaV2(
        doc_id=doc_id,
        source_path=str(pdf_path),
        company=company,
        ticker=ticker,
        title=title,
        doc_date=doc_date,
        as_of_date=as_of_date,
        as_of_source=as_of_source,
        doc_type=doc_type,
        doc_type_conf=doc_type_conf,
        version_label=version_label,
        doc_family_id=doc_family_id,
        checksum=checksum,
    )

"""Filename → document metadata extraction.

Investor presentations are named inconsistently. We use a small lookup table
of company aliases + regex for dates. If we can't infer cleanly, we leave
fields null rather than guess — the prompt later will simply say "undated"
instead of making up a date.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Company canonicalization
# ---------------------------------------------------------------------------
# Keys are lowercase substrings we'll match in the filename.
# Values are (canonical name, ticker) tuples.
_COMPANY_ALIASES: list[tuple[str, tuple[str, str]]] = [
    ("digital realty", ("Digital Realty", "DLR")),
    ("bxp", ("Boston Properties", "BXP")),
    ("boston properties", ("Boston Properties", "BXP")),
    ("psa", ("Public Storage", "PSA")),
    ("public storage", ("Public Storage", "PSA")),
    ("egp", ("EastGroup Properties", "EGP")),
    ("eastgroup", ("EastGroup Properties", "EGP")),
    ("realty incom", ("Realty Income", "O")),   # filename typo "Incom"
    ("realty income", ("Realty Income", "O")),
    ("simon", ("Simon Property Group", "SPG")),
    ("vici", ("VICI Properties", "VICI")),
]

# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_RE = "|".join(_MONTHS.keys())

# Note: '_' is a word char in regex, so we use explicit separators
# rather than \b around tokens that may be flanked by '_'.
_SEP = r"(?:^|[^A-Za-z0-9])"   # left edge: start or non-alphanumeric
_END = r"(?:$|[^A-Za-z0-9])"   # right edge: end or non-alphanumeric

_DATE_PATTERNS = [
    # "March 2026", "December 2025"
    re.compile(rf"{_SEP}(?P<m>{_MONTH_RE})\s+(?P<y>20\d{{2}}){_END}", re.IGNORECASE),
    # "2026 February", "2026_February" (year-before-month, e.g. EGP filename)
    re.compile(rf"{_SEP}(?P<y>20\d{{2}})[\s_\-](?P<m>{_MONTH_RE}){_END}", re.IGNORECASE),
    # "Mar-26", "Mar 26", "Mar_26"
    re.compile(rf"{_SEP}(?P<m>{_MONTH_RE})[ \-_](?P<y2>\d{{2}}){_END}", re.IGNORECASE),
    # "3.20.2026" or "3-20-2026"
    re.compile(rf"{_SEP}(?P<mm>\d{{1,2}})[.\-/](?P<dd>\d{{1,2}})[.\-/](?P<y>20\d{{2}}){_END}"),
    # "2026-02" or "2026_02"
    re.compile(rf"{_SEP}(?P<y>20\d{{2}})[\-_](?P<mm>0?[1-9]|1[0-2]){_END}"),
    # "Q4 2025", "q4-2025", "q4_2025"
    re.compile(rf"{_SEP}Q(?P<q>[1-4])[\s\-_]*(?P<y>20\d{{2}}){_END}", re.IGNORECASE),
]
_QUARTER_END_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}

# ---------------------------------------------------------------------------
# Doc type heuristics
# ---------------------------------------------------------------------------
_DOC_TYPE_KEYWORDS = [
    ("merger", "merger_presentation"),
    ("roadshow", "roadshow"),
    ("company-update", "company_update"),
    ("company update", "company_update"),
    ("morning session", "morning_session"),
    ("investor", "investor_presentation"),
    ("impact of", "third_party_report"),
]


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    source_path: str
    company: str | None
    ticker: str | None
    doc_date: date | None
    doc_type: str | None
    version_label: str | None
    checksum: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w]+", "_", s).strip("_").lower()
    return s[:80]


def _find_company(name_lower: str) -> tuple[str | None, str | None]:
    for needle, (canonical, ticker) in _COMPANY_ALIASES:
        if needle in name_lower:
            return canonical, ticker
    return None, None


def _find_date(name: str) -> date | None:
    for pat in _DATE_PATTERNS:
        m = pat.search(name)
        if not m:
            continue
        g = m.groupdict()
        try:
            if "q" in g and g.get("q"):
                return date(int(g["y"]), _QUARTER_END_MONTH[int(g["q"])], 1)
            if g.get("m"):
                month = _MONTHS[g["m"].lower()]
                year = int(g.get("y") or (2000 + int(g["y2"])))
                return date(year, month, 1)
            if g.get("mm") and g.get("y") and not g.get("dd"):
                return date(int(g["y"]), int(g["mm"]), 1)
            if g.get("mm") and g.get("dd") and g.get("y"):
                return date(int(g["y"]), int(g["mm"]), int(g["dd"]))
        except (ValueError, KeyError):
            continue
    return None


def _find_doc_type(name_lower: str) -> str | None:
    for needle, label in _DOC_TYPE_KEYWORDS:
        if needle in name_lower:
            return label
    return None


def _format_version_label(d: date | None) -> str | None:
    if not d:
        return None
    return d.strftime("%b %Y")  # e.g. "Mar 2026"


def file_checksum(path: Path, *, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_metadata(pdf_path: Path) -> DocMeta:
    name = pdf_path.stem
    name_lower = name.lower()

    company, ticker = _find_company(name_lower)
    doc_date = _find_date(name)
    doc_type = _find_doc_type(name_lower) or "investor_presentation"
    version_label = _format_version_label(doc_date)
    checksum = file_checksum(pdf_path)

    # doc_id: company-slug + date + short-hash to disambiguate same-date docs
    parts = [
        _slugify(company) if company else _slugify(name),
        doc_date.strftime("%Y_%m") if doc_date else "undated",
        checksum[:8],
    ]
    doc_id = "__".join(parts)

    return DocMeta(
        doc_id=doc_id,
        source_path=str(pdf_path),
        company=company,
        ticker=ticker,
        doc_date=doc_date,
        doc_type=doc_type,
        version_label=version_label,
        checksum=checksum,
    )

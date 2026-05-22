"""Structure-aware chunker (v2).

Produces THREE artifacts per page, instead of v1's flat child-chunk list:

  1. ParentChunk  — the whole slide/page (small-to-big context target).
  2. Chunk[]    — child chunks for lexical retrieval, each carrying:
                      - parent_id (for small-to-big expansion)
                      - footnote_text (page footnotes glued on → F6/F7)
                      - slide_title (header context)
                      - propagated doc metadata (doc_type, as_of_date, family)
  3. TableRowRec[]— one record per table row with COLUMN LABELS preserved,
                    so "PSA 92.0% / NSA 84.3%" never loses which is which (F8).

Design choices that are domain-agnostic:
  - Slide preservation: most deck slides fit in one chunk, so we keep a page's
    prose as a single chunk unless it's oversized — this preserves bullet lists
    and 2-column summary slides intact (F-structured-list) without any
    domain rules.
  - Footnotes are detected structurally (Docling renders footnotes/captions as
    *italic*), not by matching domain phrases.
"""

from __future__ import annotations

import re
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_system.config import settings

ChunkType = Literal["prose", "table", "chart", "figure"]


# ---------------------------------------------------------------------------
# Tokenizer + splitter (token-aware via tiktoken's cl100k encoding)
# ---------------------------------------------------------------------------
_ENC: tiktoken.Encoding | None = None
_SPLITTER: RecursiveCharacterTextSplitter | None = None


def count_tokens(text: str) -> int:
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text))


def _splitter() -> RecursiveCharacterTextSplitter:
    global _SPLITTER
    if _SPLITTER is None:
        _SPLITTER = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size_tokens,
            chunk_overlap=settings.chunk_overlap_tokens,
            length_function=count_tokens,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        )
    return _SPLITTER


# Pull GitHub-style Markdown tables out as their own segments.
_TABLE_RE = re.compile(
    r"(?:^|\n)(\|[^\n]+\|\n\|[\s\-:|]+\|\n(?:\|[^\n]+\|\n?)+)",
    re.MULTILINE,
)


def _split_text_and_tables(markdown: str) -> list[tuple[str, ChunkType]]:
    """Separate Markdown tables from prose; each table becomes its own segment."""
    parts: list[tuple[str, ChunkType]] = []
    pos = 0
    for m in _TABLE_RE.finditer(markdown):
        before = markdown[pos:m.start()].strip()
        if before:
            parts.append((before, "prose"))
        table = m.group(1).strip()
        if table:
            parts.append((table, "table"))
        pos = m.end()
    tail = markdown[pos:].strip()
    if tail:
        parts.append((tail, "prose"))
    return parts


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ParentChunk:
    parent_id: str
    doc_id: str
    page_number: int
    slide_title: str | None
    text: str
    token_count: int
    company: str | None = None
    doc_type: str | None = None
    doc_date: date | None = None
    as_of_date: date | None = None
    doc_family_id: str | None = None
    version_label: str | None = None
    # optional page thumbnail (PNG) for the citation modal
    image_png_bytes: bytes | None = None
    image_width: int | None = None
    image_height: int | None = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    parent_id: str
    page_number: int
    chunk_index: int
    text: str
    chunk_type: ChunkType
    token_count: int
    slide_title: str | None = None
    footnote_text: str | None = None
    qualifier_text: str | None = None
    company: str | None = None
    doc_type: str | None = None
    doc_date: date | None = None
    as_of_date: date | None = None
    doc_family_id: str | None = None
    version_label: str | None = None
    # vision-derived chunks (table/chart/figure) carry these:
    confidence: float | None = None
    kind_detail: str | None = None
    # chart_description chunks carry their source image
    image_png_bytes: bytes | None = None
    image_width: int | None = None
    image_height: int | None = None


@dataclass
class TableRowRec:
    row_id: str
    chunk_id: str
    doc_id: str
    page_number: int
    table_id: str
    row_idx: int
    columns: dict          # {column_label: cell_value}
    flat_text: str         # "col1: v1; col2: v2; ..."
    company: str | None = None
    doc_type: str | None = None
    doc_date: date | None = None
    as_of_date: date | None = None
    doc_family_id: str | None = None


@dataclass
class PageChunks:
    parent: ParentChunk
    children: list[Chunk] = field(default_factory=list)
    table_rows: list[TableRowRec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^\s{0,3}#{1,4}\s+(.*)$", re.MULTILINE)
# A footnote line as Docling renders it: a whole line wrapped in *...* (italic),
# OR a line beginning with a footnote/superscript marker. Domain-neutral.
_FOOTNOTE_LINE_RE = re.compile(
    r"^\s*(?:\*[^*].*\*|\(?\d+\)?[\.\)]\s+.+|note[:\s].+|source[:\s].+)\s*$",
    re.IGNORECASE,
)


def _slide_title(markdown: str) -> str | None:
    m = _HEADING_RE.search(markdown)
    if m:
        return m.group(1).strip()
    # fall back to the first non-empty line, truncated
    for line in markdown.splitlines():
        s = line.strip().lstrip("#*-").strip()
        if s:
            return s[:120]
    return None


def _extract_footnotes(markdown: str) -> str | None:
    """Collect footnote/caption-like lines on the page into one block.

    These are attached to every child chunk on the page so a body number and
    its qualifying footnote ("...as of Dec 31, 2025", "yield was 5.47%") travel
    together — closing the footnote-body integrity failure modes.
    """
    notes: list[str] = []
    for line in markdown.splitlines():
        s = line.strip()
        if not s:
            continue
        if _FOOTNOTE_LINE_RE.match(s):
            notes.append(s.strip("*").strip())
    if not notes:
        return None
    # de-dupe while preserving order; cap length
    seen, out = set(), []
    for n in notes:
        if n not in seen:
            seen.add(n)
            out.append(n)
    block = " | ".join(out)
    return block[:1500] or None


def _parse_md_table(md: str) -> tuple[list[str], list[list[str]]]:
    """Parse a GitHub-flavored Markdown table into (header, body_rows)."""
    lines = [ln for ln in md.strip().splitlines() if ln.strip().startswith("|")]
    grid = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    if not grid:
        return [], []
    header = grid[0]
    # row index 1 is the |---|---| separator; body starts at 2
    body = grid[2:] if len(grid) > 2 else []
    return header, body


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def chunk_page(
    *,
    doc_id: str,
    page_number: int,
    page_markdown: str,
    company: str | None = None,
    doc_type: str | None = None,
    doc_date: date | None = None,
    as_of_date: date | None = None,
    doc_family_id: str | None = None,
    version_label: str | None = None,
    page_png: bytes | None = None,
    page_w: int | None = None,
    page_h: int | None = None,
) -> PageChunks:
    """Chunk one page into a parent + children + table rows."""
    parent_id = f"{doc_id}::p{page_number:03d}"
    title = _slide_title(page_markdown)
    footnotes = _extract_footnotes(page_markdown)

    parent = ParentChunk(
        parent_id=parent_id,
        doc_id=doc_id,
        page_number=page_number,
        slide_title=title,
        text=page_markdown.strip(),
        token_count=count_tokens(page_markdown),
        company=company, doc_type=doc_type, doc_date=doc_date,
        as_of_date=as_of_date, doc_family_id=doc_family_id,
        version_label=version_label,
        image_png_bytes=page_png, image_width=page_w, image_height=page_h,
    )

    children: list[Chunk] = []
    table_rows: list[TableRowRec] = []
    idx = 0
    table_seq = 0

    def _push_child(text: str, kind: ChunkType, **img) -> None:
        nonlocal idx
        if not text.strip():
            return
        children.append(Chunk(
            chunk_id=f"{parent_id}::c{idx:03d}",
            doc_id=doc_id, parent_id=parent_id,
            page_number=page_number, chunk_index=idx,
            text=text.strip(), chunk_type=kind,
            token_count=count_tokens(text),
            slide_title=title, footnote_text=footnotes,
            company=company, doc_type=doc_type, doc_date=doc_date,
            as_of_date=as_of_date, doc_family_id=doc_family_id,
            version_label=version_label,
            image_png_bytes=img.get("png"),
            image_width=img.get("w"), image_height=img.get("h"),
        ))
        idx += 1

    if not page_markdown.strip():
        return PageChunks(parent=parent, children=children, table_rows=table_rows)

    # Slide-preservation threshold: keep a page's prose whole unless oversized.
    # This preserves bullet lists & 2-column summary slides without splitting.
    keep_whole_limit = int(settings.chunk_size_tokens * 1.5)

    for segment, kind in _split_text_and_tables(page_markdown):
        if kind == "table":
            # 1) keep the table as one chunk (for lexical / context)
            if count_tokens(segment) <= 2 * settings.chunk_size_tokens:
                _push_child(segment, "table")
            else:
                for piece in _splitter().split_text(segment):
                    _push_child(piece, "table")
            # 2) decompose into structured rows with column labels (F8)
            header, body = _parse_md_table(segment)
            if header and body:
                table_id = f"{parent_id}::t{table_seq:02d}"
                table_seq += 1
                for r_i, row in enumerate(body):
                    cols = {}
                    for c_i, cell in enumerate(row):
                        label = header[c_i] if c_i < len(header) else f"col{c_i}"
                        cols[label or f"col{c_i}"] = cell
                    flat = "; ".join(f"{k}: {v}" for k, v in cols.items() if v)
                    if not flat.strip():
                        continue
                    table_rows.append(TableRowRec(
                        row_id=f"{table_id}::r{r_i:02d}",
                        chunk_id=f"{table_id}",
                        doc_id=doc_id, page_number=page_number,
                        table_id=table_id, row_idx=r_i,
                        columns=cols, flat_text=flat,
                        company=company, doc_type=doc_type, doc_date=doc_date,
                        as_of_date=as_of_date, doc_family_id=doc_family_id,
                    ))
        else:
            # prose: keep whole if it fits (preserves slide structure), else split
            if count_tokens(segment) <= keep_whole_limit:
                _push_child(segment, "prose")
            else:
                for piece in _splitter().split_text(segment):
                    _push_child(piece, "prose")

    return PageChunks(parent=parent, children=children, table_rows=table_rows)

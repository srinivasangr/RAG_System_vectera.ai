"""Page-aware Markdown chunker.

Each chunk:
  - never crosses a page boundary (so citations are precise to a page)
  - has chunk_type: 'prose' | 'table' | 'chart_description'
  - is ~CHUNK_SIZE_TOKENS tokens with CHUNK_OVERLAP_TOKENS overlap
  - carries enough metadata to be stored without joining the documents table

Token counts are estimated with tiktoken's cl100k_base encoding (the OpenAI
tokenizer). Different models tokenize differently, but cl100k is a good
universal proxy and avoids per-provider tokenizer dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Literal

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_system.config import settings

ChunkType = Literal["prose", "table", "chart_description"]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_index: int
    text: str
    chunk_type: ChunkType
    token_count: int
    company: str | None
    doc_date: date | None
    version_label: str | None
    # Only populated for chart_description chunks: the source image bytes
    # (PNG) + dimensions. Stored separately in `chunk_images` table.
    image_png_bytes: bytes | None = None
    image_width: int | None = None
    image_height: int | None = None


# ---------------------------------------------------------------------------
# Tokenizer (lazy)
# ---------------------------------------------------------------------------
_ENC: tiktoken.Encoding | None = None


def _enc() -> tiktoken.Encoding:
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def count_tokens(text: str) -> int:
    return len(_enc().encode(text))


# ---------------------------------------------------------------------------
# Splitter (token-aware via tiktoken length function)
# ---------------------------------------------------------------------------
_SPLITTER: RecursiveCharacterTextSplitter | None = None


def _splitter() -> RecursiveCharacterTextSplitter:
    global _SPLITTER
    if _SPLITTER is None:
        _SPLITTER = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size_tokens,
            chunk_overlap=settings.chunk_overlap_tokens,
            length_function=count_tokens,
            # Markdown-friendly separators, longest first
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        )
    return _SPLITTER


# ---------------------------------------------------------------------------
# Table isolation
# ---------------------------------------------------------------------------
_TABLE_RE = re.compile(
    r"(?:^|\n)(\|[^\n]+\|\n\|[\s\-:|]+\|\n(?:\|[^\n]+\|\n?)+)",
    re.MULTILINE,
)


def _split_text_and_tables(markdown: str) -> list[tuple[str, ChunkType]]:
    """Pull GitHub-style Markdown tables out as their own chunks; everything
    else stays as prose for the recursive splitter to handle."""
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
# Public API
# ---------------------------------------------------------------------------
def chunk_page(
    *,
    doc_id: str,
    page_number: int,
    page_markdown: str,
    chart_descriptions: Iterable = (),
    company: str | None = None,
    doc_date: date | None = None,
    version_label: str | None = None,
) -> list[Chunk]:
    """Chunk one page's content. Returns 0+ chunks tagged by content type.

    chart_descriptions may be either:
      - an iterable of strings (just text), or
      - an iterable of (text, png_bytes, width, height) tuples (so the
        source chart image can be stored alongside).
    """
    out: list[Chunk] = []
    idx = 0

    def _push(
        text: str,
        chunk_type: ChunkType,
        *,
        png: bytes | None = None,
        w: int | None = None,
        h: int | None = None,
    ) -> None:
        nonlocal idx
        if not text.strip():
            return
        out.append(
            Chunk(
                chunk_id=f"{doc_id}::p{page_number:03d}::c{idx:03d}",
                doc_id=doc_id,
                page_number=page_number,
                chunk_index=idx,
                text=text.strip(),
                chunk_type=chunk_type,
                token_count=count_tokens(text),
                company=company,
                doc_date=doc_date,
                version_label=version_label,
                image_png_bytes=png,
                image_width=w,
                image_height=h,
            )
        )
        idx += 1

    # 1. Prose + tables from the page markdown
    if page_markdown.strip():
        for segment, kind in _split_text_and_tables(page_markdown):
            if kind == "table":
                if count_tokens(segment) <= 2 * settings.chunk_size_tokens:
                    _push(segment, "table")
                else:
                    for piece in _splitter().split_text(segment):
                        _push(piece, "table")
            else:
                for piece in _splitter().split_text(segment):
                    _push(piece, "prose")

    # 2. Chart descriptions from the vision pass — one chunk each, with image
    for item in chart_descriptions:
        if isinstance(item, tuple):
            desc, png, w, h = item
            _push(desc, "chart_description", png=png, w=w, h=h)
        else:
            _push(item, "chart_description")

    return out
